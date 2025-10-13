import argparse, os, random
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------
# Dumb data loader (TSV: src \t tgt)
# ----------------------------
def read_tsv(path: str) -> List[Dict[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expect "source\t target"
            parts = line.split("\t")
            if len(parts) < 2:
                # skip trash lines
                continue
            src, tgt = parts[0], parts[1]
            pairs.append({"src": src, "tgt": tgt})
    if not pairs:
        raise ValueError(f"No usable lines found in {path}. Expect TSV with 'src\\t tgt'.")
    return pairs

# Tiny fallback toy data if you don't pass --train_file
def toy_pairs() -> List[Dict[str, str]]:
    return [
        {"src": "summarize: The cat sat on the mat.", "tgt": "Cat on mat."},
        {"src": "summarize: Transformers are powerful models for NLP.", "tgt": "Transformers are powerful."},
        {"src": "summarize: The sky is blue and the sun is bright.", "tgt": "Blue sky, bright sun."},
    ]

# ----------------------------
# Dataset + Collate
# ----------------------------
class PairDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def make_collate_fn(tokenizer, src_max_len: int, tgt_max_len: int):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad_token_id. Set one before training.")

    def collate(batch: List[Dict[str, str]]):
        sources = [ex["src"] for ex in batch]
        targets = [ex["tgt"] for ex in batch]

        model_inputs = tokenizer(
            sources,
            max_length=src_max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            labels = tokenizer(
                text_target=targets,
                max_length=tgt_max_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )["input_ids"]
        labels[labels == pad_id] = -100  # ignore pad in loss
        model_inputs["labels"] = labels
        return model_inputs
    return collate

# ----------------------------
# Training
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        # For some tokenizers you need to define this; T5 already has it.
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    model.train()

    # Data
    if args.train_file:
        pairs = read_tsv(args.train_file)
    else:
        pairs = toy_pairs()
    ds = PairDataset(pairs)
    collate_fn = make_collate_fn(tokenizer, args.src_max_len, args.tgt_max_len)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)  # uses labels -> returns loss
            loss = out.loss

            loss.backward()
            optim.step()
            optim.zero_grad()

            global_step += 1
            if global_step % args.log_every == 0:
                print(f"[epoch {epoch}] step {global_step} loss={loss.item():.4f}")

    # Save minimal artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Bare-bones seq2seq trainer (no LoRA, no AMP, no eval).")
    p.add_argument("--model", type=str, default="google/flan-t5-small")
    p.add_argument("--train_file", type=str, default=None, help="TSV with 'src\\t tgt'. If omitted, uses tiny toy set.")
    p.add_argument("--output_dir", type=str, default="./out_basic")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--src_max_len", type=int, default=256)
    p.add_argument("--tgt_max_len", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)