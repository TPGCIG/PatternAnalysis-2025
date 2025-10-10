# dataset.py
from __future__ import annotations
import csv, json, os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
try:
    from datasets import load_dataset  # optional
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from transformers import AutoTokenizer, DataCollatorForSeq2Seq


class SummarisationDataset(Dataset):
    """
    Flexible dataset: supports HF datasets, CSV, or JSONL.
    Expects two text fields: `input_col` (expert report) and `target_col` (lay summary).
    """
    def __init__(
        self,
        records: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        max_input_len: int = 1024,
        max_target_len: int = 256,
        add_prefix: bool = True,
        prefix_text: str = "summarize: ",
        strip_empty: bool = True,
    ):
        self.records = []
        for r in records:
            src = (r.get("input") or r.get("report") or r.get("source") or r.get("text") or "").strip()
            tgt = (r.get("target") or r.get("summary") or r.get("lay_summary") or "").strip()
            if strip_empty and (not src or not tgt):
                continue
            self.records.append({"input": src, "target": tgt})

        if len(self.records) == 0:
            raise ValueError("No usable records found (empty inputs/targets).")

        self.tok = tokenizer
        self.max_in = max_input_len
        self.max_tgt = max_target_len
        self.add_prefix = add_prefix
        self.prefix_text = prefix_text

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        ex = self.records[idx]
        src = (self.prefix_text + ex["input"]) if self.add_prefix else ex["input"]
        tgt = ex["target"]

        model_inputs = self.tok(
            src,
            max_length=self.max_in,
            truncation=True,
            padding=False,
        )
        labels = self.tok(
            text_target=tgt,             # <-- modern API
            max_length=self.max_tgt,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return {k: torch.tensor(v) for k, v in model_inputs.items()}


def load_local_csv(
    path: str,
    input_col: str = "report",
    target_col: str = "summary",
    delimiter: str = ",",
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for r in reader:
            rows.append({"input": r.get(input_col, ""), "target": r.get(target_col, "")})
    return rows


def load_local_jsonl(
    path: str,
    input_col: str = "report",
    target_col: str = "summary",
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"input": obj.get(input_col, ""), "target": obj.get(target_col, "")})
    return rows


def load_biolaysumm_hf(
    dataset_name: str = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track",
    split: str = "train",
    input_col: str = "report",
    target_col: str = "summary",
) -> List[Dict[str, str]]:
    if not HF_AVAILABLE:
        raise RuntimeError("`datasets` not installed. Use local CSV/JSONL or install `datasets`.")
    ds = load_dataset(dataset_name, split=split)
    rows = []
    for r in ds:
        rows.append({"input": r.get(input_col, ""), "target": r.get(target_col, "")})
    return rows


def make_datasets(
    tokenizer_name: str = "google/flan-t5-base",
    train_source: Tuple[str, str] = ("local_jsonl", "train.jsonl"),
    val_source: Optional[Tuple[str, str]] = ("local_jsonl", "val.jsonl"),
    input_col: str = "report",
    target_col: str = "summary",
    max_input_len: int = 1024,
    max_target_len: int = 256,
    add_prefix: bool = True,
    prefix_text: str = "summarize: ",
):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def _load(kind: str, arg: str) -> List[Dict[str, str]]:
        if kind == "local_csv":
            return load_local_csv(arg, input_col, target_col)
        elif kind == "local_jsonl":
            return load_local_jsonl(arg, input_col, target_col)
        elif kind == "hf":
            # arg should be split name if using HF
            return load_biolaysumm_hf(split=arg, input_col=input_col, target_col=target_col)
        else:
            raise ValueError(f"Unknown source kind: {kind}")

    train_kind, train_arg = train_source
    train_rows = _load(train_kind, train_arg)

    val_rows = []
    if val_source:
        val_kind, val_arg = val_source
        val_rows = _load(val_kind, val_arg)

    train_ds = SummarisationDataset(
        train_rows, tok, max_input_len, max_target_len, add_prefix, prefix_text
    )
    val_ds = SummarisationDataset(
        val_rows, tok, max_input_len, max_target_len, add_prefix, prefix_text
    ) if val_rows else None

    collator = Seq2SeqCollatorFast(tok, label_pad_token_id=-100, pad_to_multiple_of=None)  # or 8/16 if you want alignment
    return tok, train_ds, val_ds, collator


from torch.nn.utils.rnn import pad_sequence

class Seq2SeqCollatorFast:
    """
    Fast collator that pads inputs/labels with pure torch ops.
    Avoids the slow path in HF's DataCollatorForSeq2Seq that triggers
    """
    def __init__(self, tokenizer, label_pad_token_id=-100, pad_to_multiple_of=None):
        self.tok = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def _maybe_pad_to_multiple(self, tensor, pad_value):
        if self.pad_to_multiple_of is None:
            return tensor
        seq_len = tensor.size(1)
        if seq_len % self.pad_to_multiple_of == 0:
            return tensor
        pad_len = self.pad_to_multiple_of - (seq_len % self.pad_to_multiple_of)
        pad = (0, pad_len)  # pad on the right
        return torch.nn.functional.pad(tensor, pad, value=pad_value)

    def __call__(self, features):
        # features: list of dicts with torch tensors (from our Dataset)
        input_ids = [f["input_ids"] if isinstance(f["input_ids"], torch.Tensor) else torch.tensor(f["input_ids"]) for f in features]
        attn_masks = [f["attention_mask"] if isinstance(f["attention_mask"], torch.Tensor) else torch.tensor(f["attention_mask"]) for f in features]
        labels     = [f["labels"] if isinstance(f["labels"], torch.Tensor) else torch.tensor(f["labels"]) for f in features]

        # Pad inputs
        pad_id = self.tok.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

        # Pad labels with pad_token_id, then convert to -100
        labels = pad_sequence(labels, batch_first=True, padding_value=pad_id)
        labels_mask = labels.eq(pad_id)
        labels = labels.masked_fill(labels_mask, self.label_pad_token_id)

        # Optional: align to 8/16/32 for kernel efficiency
        input_ids = self._maybe_pad_to_multiple(input_ids, pad_id)
        attn_masks = self._maybe_pad_to_multiple(attn_masks, 0)
        labels = self._maybe_pad_to_multiple(labels, self.label_pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels,
        }
