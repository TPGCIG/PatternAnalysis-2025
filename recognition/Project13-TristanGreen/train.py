"""
------------------------------------------------------------
 Brain-T5: FLAN-T5 + LoRA Fine-Tuning Pipeline
 -----------------------------------------------------------
 Description:
    Main training script for Brain-T5. Handles dataset loading,
    LoRA adapter attachment, training loop, logging, and evaluation.

 Key Functions:
    - run_eval(): computes ROUGE scores on validation/test splits.
    - log_val_rouge_row(): logs per-epoch ROUGE metrics to CSV.
    - plot_loss_curve(), plot_val_rouge_curve(): generate plots.

 Notes:
    - Uses AdamW + cosine schedule.
    - Gradient accumulation supported via --accum.
    - Mixed precision enabled via torch.amp.
    - Best model checkpoint chosen by highest ROUGE-Lsum.
------------------------------------------------------------
"""
import os, json, math, argparse, random, time, uuid, csv
from typing import Optional
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

from dataset import make_datasets   # locked dataset helper
from modules import load_base_model, attach_lora
import evaluate

# -----------------------
# Utils: logging & eval
# -----------------------

def csv_logger(path: str):
    """Append-mode CSV logger for training steps; writes header if file is empty."""
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(["timestamp", "epoch", "global_step", "loss"])
    return f, w

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Eval loop: generate summaries for a dataloader and compute ROUGE.
# Notes:
#  - We re-enable use_cache for fast generation.
#  - Convert label pad (-100) back to tokenizer.pad_token_id before decoding refs.
#  - no_repeat_ngram_size=3 reduces trivial repetition.
def run_eval(model, tokenizer, loader: Optional[DataLoader], device, args, rouge_metric):
    if loader is None:
        return None
    model.eval()
    use_cache_was = getattr(model.config, "use_cache", True)
    model.config.use_cache = True

    preds, refs = [], []
    with torch.inference_mode():
        for vb in tqdm(loader, desc="Eval", unit="batch", dynamic_ncols=True):
            vb = {k: v.to(device) for k, v in vb.items()}
            # Beam search generation for evaluation (deterministic-ish)
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
            preds.extend(pred_txt); refs.extend(ref_txt)

    scores = rouge_metric.compute(predictions=preds, references=refs, use_stemmer=True)
    keep = {k: float(v) for k, v in scores.items() if k in {"rouge1","rouge2","rougeL","rougeLsum"}}
    model.config.use_cache = use_cache_was
    return keep

# -----------------------
# Per-epoch ROUGE logging
# -----------------------

RUN_ID = os.environ.get("RUN_ID", str(uuid.uuid4())[:8])

# Persist per-epoch validation ROUGE to CSV for plotting and auditing.
# If multiple runs append to same file, we keep last row per epoch when plotting.
def log_val_rouge_row(run_dir, epoch, scores):
    """
    Append one row per epoch:
    run_id, timestamp, epoch, rouge1, rouge2, rougeL, rougeLsum
    Writes header if file is empty.
    """
    path = os.path.join(run_dir, "history_val.csv")
    new_file = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["run_id","timestamp","epoch","rouge1","rouge2","rougeL","rougeLsum"])
        w.writerow([
            RUN_ID,
            int(time.time()),
            int(epoch),
            float(scores.get("rouge1", 0.0)),
            float(scores.get("rouge2", 0.0)),
            float(scores.get("rougeL", 0.0)),
            float(scores.get("rougeLsum", 0.0)),
        ])

# -----------------------
# Plotting helpers
# -----------------------

# Plot validation ROUGE vs epoch.
# Robust to restarts: we select the *latest* row per epoch (by timestamp) to avoid stale re-runs.
def plot_val_rouge_curve(run_dir):
    """
    Plot Validation ROUGE vs Epoch (robust):
      - reads history_val.csv
      - keeps only the latest row per epoch (by timestamp if present)
      - sorts epochs ascending
      - falls back to metrics_val.json if no rows
    """
    import json
    path = os.path.join(run_dir, "history_val.csv")
    rows = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    def _fallback_from_metrics():
        mv = os.path.join(run_dir, "metrics_val.json")
        if not os.path.exists(mv):
            print("[plot] no history_val.csv rows and no metrics_val.json; skipping val ROUGE plot")
            return None
        obj = json.load(open(mv, "r", encoding="utf-8"))
        ep = int(obj.get("epoch", 1))
        s  = obj.get("scores", obj)
        return [ep], [float(s.get("rouge1", 0.0))], [float(s.get("rouge2", 0.0))], \
               [float(s.get("rougeL", 0.0))], [float(s.get("rougeLsum", 0.0))]

    if not rows:
        vals = _fallback_from_metrics()
        if vals is None: return
        epochs, r1, r2, rL, rS = vals
    else:
        # latest row per epoch by timestamp if present; else by order
        last_by_epoch = {}
        for r in rows:
            if "epoch" not in r:  # malformed row
                continue
            try:
                ep = int(r["epoch"])
            except Exception:
                continue
            ts = int(r.get("timestamp", 0)) if r.get("timestamp") else 0
            if (ep not in last_by_epoch) or (ts >= int(last_by_epoch[ep].get("timestamp", 0) or 0)):
                last_by_epoch[ep] = r

        if not last_by_epoch:
            vals = _fallback_from_metrics()
            if vals is None: return
            epochs, r1, r2, rL, rS = vals
        else:
            epochs = sorted(last_by_epoch.keys())
            def _f(e, k):
                v = last_by_epoch[e].get(k, None)
                return float(v) if v not in (None, "",) else 0.0
            r1 = [_f(e, "rouge1") for e in epochs]
            r2 = [_f(e, "rouge2") for e in epochs]
            rL = [_f(e, "rougeL")  for e in epochs]
            rS = [_f(e, "rougeLsum") for e in epochs]

    plt.figure()
    plt.plot(epochs, r1, label="ROUGE-1")
    plt.plot(epochs, r2, label="ROUGE-2")
    plt.plot(epochs, rL, label="ROUGE-L")
    plt.plot(epochs, rS, label="ROUGE-Lsum")
    plt.xlabel("Epoch"); plt.ylabel("ROUGE")
    plt.title("Validation ROUGE vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(run_dir, "rouge_val_curve.png")
    plt.savefig(out, dpi=160); plt.close()
    print(f"[plot] wrote {out}")

def plot_loss_curve(run_dir):
    """
    Plot a single clean Training Loss vs Steps line even if train_log.csv
    contains multiple runs or step resets.
    Strategy:
      - read train_log.csv
      - split into segments whenever global_step decreases (new run appended)
      - keep ONLY the last segment (latest run)
      - sort by step, clip outlier spikes (1..99p), smooth with small moving average
    """
    import numpy as np

    path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print("[plot] no train_log.csv; skipping loss plot"); return

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[plot] empty train_log.csv; skipping loss plot"); return

    # parse numeric
    raw_steps, raw_losses = [], []
    for r in rows:
        try:
            raw_steps.append(int(float(r["global_step"])))
            raw_losses.append(float(r["loss"]))
        except Exception:
            continue
    if not raw_steps:
        print("[plot] no numeric rows in train_log.csv; skipping"); return

    # split into segments whenever step decreases (step reset = new run)
    segs = []
    seg_s, seg_l = [raw_steps[0]], [raw_losses[0]]
    for s, l in zip(raw_steps[1:], raw_losses[1:]):
        if s < seg_s[-1]:  # reset
            segs.append((seg_s, seg_l))
            seg_s, seg_l = [s], [l]
        else:
            seg_s.append(s); seg_l.append(l)
    segs.append((seg_s, seg_l))

    # pick the LAST segment (most recent run)
    steps, losses = segs[-1]

    # sort by step
    order = np.argsort(steps)
    steps  = [steps[i]  for i in order]
    losses = [losses[i] for i in order]

    # clip extreme spikes for visualization (1..99 percentile)
    lo, hi = np.percentile(losses, [1, 99])
    keep = [(lo <= v <= hi) for v in losses]
    steps  = [s for s, m in zip(steps, keep) if m]
    losses = [v for v, m in zip(losses, keep) if m]

    # moving average smoothing
    from collections import deque
    def movavg(x, k=None):
        if len(x) == 0: return x
        if k is None:
            k = max(5, min(25, len(x)//20))  # gentle default
        out, q, s = [], deque(), 0.0
        for v in x:
            q.append(v); s += v
            if len(q) > k: s -= q.popleft()
            out.append(s / len(q))
        return out

    sm = movavg(losses)

    plt.figure()
    plt.plot(steps, sm)
    plt.xlabel("Global step (optimizer)")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Steps")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(out, dpi=160); plt.close()
    print(f"[plot] wrote {out}")

def plot_test_rouge_bar(run_dir):
    path = os.path.join(run_dir, "metrics_test.json")
    if not os.path.exists(path):
        print("[plot] no metrics_test.json; skipping test bar"); return
    m = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(m, dict) and "note" in m:
        print(f"[plot] {m['note']} — skipping test bar"); return
    labels = ["ROUGE-1","ROUGE-2","ROUGE-L","ROUGE-Lsum"]
    vals = [float(m.get("rouge1",0.0)), float(m.get("rouge2",0.0)),
            float(m.get("rougeL",0.0)), float(m.get("rougeLsum",0.0))]
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Score"); plt.title("Test ROUGE (Held-out)")
    plt.tight_layout()
    out = os.path.join(run_dir, "rouge_test_bar.png")
    plt.savefig(out, dpi=160); plt.close()
    print(f"[plot] wrote {out}")

# -----------------------
# Main
# -----------------------

def main():
    p = argparse.ArgumentParser()
    # No dataset args — we're locked to the RRG opensource track via make_datasets
    p.add_argument("--model_name", default="google/flan-t5-base")
    p.add_argument("--train_split", default="train")
    p.add_argument("--val_split",   default="validation")
    p.add_argument("--test_split",  default="test")

    # training
    p.add_argument("--epochs",       type=int, default=5)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--accum",        type=int, default=16)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--clip",         type=float, default=1.0)
    p.add_argument("--lora_r",       type=int, default=8)
    p.add_argument("--lora_alpha",   type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--output_dir",   default="runs/flan_t5_base_lora_rrg")
    p.add_argument("--seed",         type=int, default=1337)
    p.add_argument("--fp16",         action="store_true")

    # eval/generation
    p.add_argument("--eval_max_new_tokens", type=int, default=128)
    p.add_argument("--eval_beams",   type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)

    # dev-speed controls
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)

    # optional self-split (because official opensource test has empty refs)
    p.add_argument("--self_split", action="store_true",
                   help="If set, create custom 80/10/10 train/val/test from training data.")
    p.add_argument("--self_split_val", type=float, default=0.1,
                   help="Proportion of data to use as validation if self_split.")
    p.add_argument("--self_split_test", type=float, default=0.1,
                   help="Proportion of data to use as test if self_split.")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Fresh logs for this run (truncate old content so headers/steps align with current run).
    open(os.path.join(args.output_dir, "train_log.csv"), "w").close()
    open(os.path.join(args.output_dir, "history_val.csv"), "w").close()

    # tokenizer + datasets (locked to RRG via make_datasets)
    tokenizer, train_ds, val_ds, test_ds, collator = make_datasets(
        tokenizer_name=args.model_name,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        max_input_len=1024,
        max_target_len=256,
        prefix_text="summarise",
        self_split=args.self_split,
        self_split_val=args.self_split_val,
        self_split_test=args.self_split_test,
    )

    # Optional subsetting
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if val_ds is not None and args.max_eval_samples is not None:
        val_ds = val_ds.select(range(min(args.max_eval_samples, len(val_ds))))
    if test_ds is not None and args.max_test_samples is not None:
        test_ds = test_ds.select(range(min(args.max_test_samples, len(test_ds))))

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, pin_memory=True, num_workers=0)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                                collate_fn=collator, pin_memory=True, num_workers=0)

    # model + LoRA
    dtype = torch.float16 if (args.fp16 and torch.cuda.is_available()) else torch.float32
    model = load_base_model(args.model_name, dtype=dtype, device_map=None)
    model = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                        dropout=args.lora_dropout, target_modules=["q","k","v","o"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Log params & hardware (artifact for report)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open(os.path.join(args.output_dir, "params.json"), "w") as f:
        json.dump({"total": total, "trainable": trainable, "ratio": trainable/total}, f, indent=2)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        with open(os.path.join(args.output_dir, "hardware.json"), "w") as f:
            json.dump({"gpu_name": torch.cuda.get_device_name(0),
                       "total_vram_gb": round(props.total_memory/(1024**3),2),
                       "compute_capability": f"{props.major}.{props.minor}"},
                      f, indent=2)

    # Logging
    log_file, log_writer = csv_logger(os.path.join(args.output_dir, "train_log.csv"))
    global_step = 0

    # Optimizer & schedule:
    #  - AdamW with weight decay.
    #  - Cosine schedule with ~6% warmup (capped by --warmup_steps).
    #  - GradScaler enabled only when --fp16 on CUDA.
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.accum) * args.epochs
    warmup = min(args.warmup_steps, int(0.06 * total_steps))
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device == "cuda"))
    rouge_metric = evaluate.load("rouge")

    t0 = time.time()
    best_rougeLsum = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optim.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Train e{epoch}", unit="batch", dynamic_ncols=True)
        start_time = time.time()
        step_in_epoch = 0

        for batch in pbar:
            step_in_epoch += 1; global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(args.fp16 and device == "cuda")):
                out = model(**batch)
                loss = out.loss / args.accum
            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at step {global_step}: {float(loss)} — skipping.")
                optim.zero_grad(set_to_none=True); continue

            scaler.scale(loss).backward(); running += loss.item()
            if (global_step % args.accum) == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optim); scaler.update(); sched.step()
                optim.zero_grad(set_to_none=True)

                # CSV log (per optimizer step)
                avg_loss = running / max(1, (step_in_epoch // args.accum))
                log_writer.writerow([time.time(), epoch, global_step, avg_loss]); log_file.flush()

            # live status
            avg_loss = running / max(1, (step_in_epoch // args.accum))
            elapsed = time.time() - start_time
            sps = (step_in_epoch * args.batch_size) / max(1e-6, elapsed)
            # Live progress: smoothed loss and samples/sec for quick sanity checks.
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "sps": f"{sps:.1f}"})

        print(f"[epoch {epoch}] train_loss={avg_loss:.4f}")

        # validation at epoch end
        scores = run_eval(model, tokenizer, val_loader, device, args, rouge_metric)
        if scores is not None:
            print(f"[epoch {epoch}] ROUGE (val): {scores}")

            # persist per-epoch history for plotting
            log_val_rouge_row(args.output_dir, epoch, scores)

            rougeLsum = float(scores.get("rougeLsum", 0.0))
            if rougeLsum > best_rougeLsum:
                best_rougeLsum = rougeLsum
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                with open(os.path.join(args.output_dir, "metrics_val.json"), "w") as f:
                    json.dump({"best_rougeLsum": best_rougeLsum, "epoch": epoch, "scores": scores}, f, indent=2)
                print(f"[epoch {epoch}] saved best adapters to {args.output_dir}")

            # live plot after each epoch
            plot_val_rouge_curve(args.output_dir)

    # timing
    minutes = (time.time() - t0) / 60.0
    with open(os.path.join(args.output_dir, "time.json"), "w") as f:
        json.dump({"minutes": minutes, "epochs": args.epochs}, f, indent=2)

    # test evaluation (held-out; for opensource we self-split or skip if refs missing)
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=collator, pin_memory=True, num_workers=0)
        test_scores = run_eval(model, tokenizer, test_loader, device, args, rouge_metric)
        with open(os.path.join(args.output_dir, "metrics_test.json"), "w") as f:
            json.dump(test_scores, f, indent=2)
        print(f"[test] ROUGE: {test_scores}")

    # Always generate plots at the end for the report
    plot_loss_curve(args.output_dir)
    plot_val_rouge_curve(args.output_dir)
    plot_test_rouge_bar(args.output_dir)

    print("done.")

if __name__ == "__main__":
    main()
