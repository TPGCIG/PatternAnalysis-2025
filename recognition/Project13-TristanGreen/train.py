import os, json, math, argparse, random, csv, time
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

# our modules
from dataset import make_datasets
from modules import load_base_model, attach_lora

import evaluate  # HF evaluate -> ROUGE


def csv_logger(path: str):
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(["timestamp", "epoch", "global_step", "loss"])
    return f, w


def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_labels(tokenizer, labels: torch.Tensor) -> List[str]:
    lab = labels.clone()
    lab[lab == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(lab, skip_special_tokens=True)


def run_eval(model, tokenizer, val_loader: Optional[DataLoader], device, args, rouge_metric):
    if val_loader is None:
        return None
    model.eval()

    # speed hint for generation
    use_cache_was = getattr(model.config, "use_cache", True)
    model.config.use_cache = True

    preds, refs = [], []
    with torch.inference_mode():
        val_pbar = tqdm(val_loader, desc="Eval", unit="batch", dynamic_ncols=True)
        for vb in val_pbar:
            vb = {k: v.to(device) for k, v in vb.items()}
            gen_out = model.generate(
                input_ids=vb["input_ids"],
                attention_mask=vb["attention_mask"],
                max_new_tokens=args.eval_max_new_tokens,
                num_beams=args.eval_beams,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
            )
            pred_txt = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
            tgt = vb["labels"].clone()
            tgt[tgt == -100] = tokenizer.pad_token_id
            ref_txt = tokenizer.batch_decode(tgt, skip_special_tokens=True)
            preds.extend(pred_txt)
            refs.extend(ref_txt)

    scores = rouge_metric.compute(predictions=preds, references=refs, use_stemmer=True)
    keep = {k: float(v) for k, v in scores.items() if k in {"rouge1", "rouge2", "rougeL", "rougeLsum"}}
    model.config.use_cache = use_cache_was
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/flan-t5-base")
    p.add_argument("--train_source", default="local_jsonl")
    p.add_argument("--train_path",   default="train.jsonl")
    p.add_argument("--val_source",   default="local_jsonl")
    p.add_argument("--val_path",     default="val.jsonl")
    p.add_argument("--input_col",    default="report")
    p.add_argument("--target_col",   default="summary")
    p.add_argument("--max_input_len", type=int, default=1024)
    p.add_argument("--max_target_len", type=int, default=256)
    p.add_argument("--prefix",       default="summarize: ")
    p.add_argument("--epochs",       type=int, default=3)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--accum",        type=int, default=16)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--clip",         type=float, default=1.0)
    p.add_argument("--lora_r",       type=int, default=8)
    p.add_argument("--lora_alpha",   type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--output_dir",   default="runs/flan_t5_base_lora")
    p.add_argument("--seed",         type=int, default=1337)
    p.add_argument("--fp16",         action="store_true")
    p.add_argument("--eval_max_new_tokens", type=int, default=128)
    p.add_argument("--eval_beams",   type=int, default=4)

    # dev/fast-run controls
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Limit training examples for quick dev runs")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Limit validation examples during dev")
    p.add_argument("--eval_batch_size", type=int, default=8,
                   help="Batch size used for generation during eval")
    p.add_argument("--eval_every_steps", type=int, default=None,
                   help="If set, run eval every N optimizer steps (in addition to end-of-epoch)")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # tokenizer + datasets
    tokenizer, train_ds, val_ds, collator = make_datasets(
        tokenizer_name=args.model_name,
        train_source=(args.train_source, args.train_path),
        val_source=(args.val_source, args.val_path) if args.val_path else None,
        input_col=args.input_col,
        target_col=args.target_col,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        add_prefix=True,
        prefix_text=args.prefix,
    )

    # Subset for dev speed
    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(train_ds))
        print(f"[INFO] Using only first {n} training samples (of {len(train_ds)})")
        train_ds = Subset(train_ds, range(n))
    if (val_ds is not None) and (args.max_eval_samples is not None):
        m = min(args.max_eval_samples, len(val_ds))
        print(f"[INFO] Using only first {m} validation samples (of {len(val_ds)})")
        val_ds = Subset(val_ds, range(m))

    # DataLoaders (eval uses larger batch)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, pin_memory=True, num_workers=0
    )
    val_loader: Optional[DataLoader] = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.eval_batch_size, shuffle=False,
            collate_fn=collator, pin_memory=True, num_workers=0
        )

    # model + LoRA
    dtype = torch.float16 if (args.fp16 and torch.cuda.is_available()) else torch.float32
    model = load_base_model(args.model_name, dtype=dtype, device_map=None)
    model = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                        dropout=args.lora_dropout, target_modules=["q", "k", "v", "o"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Logging
    log_file, log_writer = csv_logger(os.path.join(args.output_dir, "train_log.csv"))
    global_step = 0
    optimizer_steps = 0

    # Optim + sched
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.accum) * args.epochs
    warmup = min(args.warmup_steps, int(0.06 * total_steps))
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device == "cuda"))

    # Load ROUGE once
    rouge_metric = evaluate.load("rouge")

    best_rougeLsum = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optim.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Train e{epoch}", unit="batch", dynamic_ncols=True)
        start_time = time.time()
        step_in_epoch = 0

        for batch in pbar:
            step_in_epoch += 1
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=(args.fp16 and device == "cuda")):
                out = model(**batch)
                loss = out.loss / args.accum

            # guard against NaN/Inf
            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at global_step {global_step}: {float(loss)}. Skipping batch.")
                optim.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            running += loss.item()

            if (global_step % args.accum) == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
                optimizer_steps += 1

                # mid-epoch eval hook
                if (args.eval_every_steps is not None) and (optimizer_steps % args.eval_every_steps == 0):
                    scores = run_eval(model, tokenizer, val_loader, device, args, rouge_metric)
                    if scores is not None:
                        print(f"[step {optimizer_steps}] ROUGE: {scores}")

            # live progress
            avg_loss = running / max(1, (step_in_epoch // args.accum))
            elapsed = time.time() - start_time
            sps = (step_in_epoch * args.batch_size) / max(1e-6, elapsed)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "sps": f"{sps:.1f}"})

            # CSV log
            if (global_step % args.accum) == 0:
                log_writer.writerow([time.time(), epoch, global_step, avg_loss])
                log_file.flush()

        print(f"[epoch {epoch}] train_loss={avg_loss:.4f}")

        # end-of-epoch eval
        scores = run_eval(model, tokenizer, val_loader, device, args, rouge_metric)
        if scores is not None:
            print(f"[epoch {epoch}] ROUGE: {scores}")
            rougeLsum = float(scores.get("rougeLsum", 0.0))
            if rougeLsum > best_rougeLsum:
                best_rougeLsum = rougeLsum
                model.save_pretrained(args.output_dir)   # saves LoRA adapters
                tokenizer.save_pretrained(args.output_dir)
                with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
                    json.dump({"best_rougeLsum": best_rougeLsum, "epoch": epoch, "scores": scores}, f, indent=2)
                print(f"[epoch {epoch}] âœ“ saved best adapters to {args.output_dir}")

    log_file.close()
    print("done.")


if __name__ == "__main__":
    main()