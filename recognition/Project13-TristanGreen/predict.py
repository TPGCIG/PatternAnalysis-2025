"""
------------------------------------------------------------
 Prediction and Inference for Brain-T5
 -----------------------------------------------------------
 Description:
    Generates summaries from fine-tuned LoRA adapters.
    Supports both single-text (--text) and batch (--jsonl) modes.

 Key Functions:
    - load_model(): loads base + LoRA adapter for inference.
    - generate_batch(): batched generation with beam search.

 Notes:
    - Outputs JSONL with 'prediction' field appended to each input.
    - Uses max_new_tokens and num_beams for generation control.
------------------------------------------------------------
"""
import os, argparse, json
from typing import List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# Load tokenizer from adapter_dir (ensures identical preproc as training), attach LoRA onto base.
# dtype=float16 only when CUDA+--fp16; always move model to device and set eval().
def load_model(adapter_dir: str, base_model: str, fp16: bool):
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model, dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    return tok, model, device

def chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# Generate a batch with beam search; always prefix with instruction to match training distribution.
def generate_batch(model, tok, device, texts: List[str], max_in: int, max_new: int, beams: int, prefix: str):
    batch = [prefix + t for t in texts]
    enc = tok(batch, return_tensors="pt", truncation=True, max_length=max_in, padding=True).to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            num_beams=beams,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )
    return tok.batch_decode(out, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True, help="Path to saved LoRA adapters")
    ap.add_argument("--base_model", default="google/flan-t5-base")
    ap.add_argument("--text", default=None, help="Single input string to summarize")
    ap.add_argument("--jsonl", default=None, help="Path to JSONL with an input column")
    ap.add_argument("--input_col", default="report")
    ap.add_argument("--out_path", default="predictions.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_input_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--prefix", default="summarize: ")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    # Modes:
    #  --text "..."         -> print single summary to stdout
    #  --jsonl file.jsonl   -> stream predictions and write to --out_path
    #  --input_col selects field in JSONL to summarize (default: 'report')


    tok, model, device = load_model(args.adapter_dir, args.base_model, args.fp16)

    # single text mode
    if args.text is not None:
        outs = generate_batch(model, tok, device, [args.text], args.max_input_len, args.max_new_tokens, args.beams, args.prefix)
        print(outs[0])
        return

    # file mode
    if args.jsonl is None:
        raise SystemExit("Provide --text or --jsonl")
    rows = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    inputs = [r.get(args.input_col, "") for r in rows]
    preds = []
    for block in chunk(inputs, args.batch_size):
        preds.extend(generate_batch(model, tok, device, block, args.max_input_len, args.max_new_tokens, args.beams, args.prefix))

    # write JSONL with predictions
    with open(args.out_path, "w", encoding="utf-8") as w:
        for r, p in zip(rows, preds):
            out = dict(r)
            out["prediction"] = p
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"wrote {len(preds)} predictions to {args.out_path}")

if __name__ == "__main__":
    main()