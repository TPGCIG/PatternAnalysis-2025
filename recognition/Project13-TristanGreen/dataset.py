# ------------------------------------------------------------
#  Dataset Loader and Preprocessing for Brain-T5
#  -----------------------------------------------------------
#  Description:
#     Handles dataset intake and preprocessing for FLAN-T5 fine-tuning.
#     Supports Hugging Face (BioLaySumm) datasets, CSV, or JSONL inputs.
#
#  Key Components:
#     - make_datasets(): loads and tokenizes splits (train/val/test).
#     - Seq2SeqCollatorFast: dynamic padding & label masking for T5.
#
#  Notes:
#     - Automatically prefixes "summarize: " to each input.
#     - Pads to model’s max token length.
#     - Masks <pad> tokens in labels with -100 for CrossEntropyLoss.
# ------------------------------------------------------------
from __future__ import annotations
from typing import Optional, List, Dict
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
INPUT_COL  = "radiology_report"
TARGET_COL = "layman_report"

class Seq2SeqCollatorFast:
    def __init__(self, tokenizer, label_pad_token_id=-100, pad_to_multiple_of=None):
        self.tok = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def _maybe_pad_to_multiple(self, tensor, pad_value):
        if self.pad_to_multiple_of is None:
            return tensor
        L = tensor.size(1)
        if L % self.pad_to_multiple_of == 0:
            return tensor
        add = self.pad_to_multiple_of - (L % self.pad_to_multiple_of)
        return torch.nn.functional.pad(tensor, (0, add), value=pad_value)

    def __call__(self, feats: List[Dict[str, torch.Tensor]]):
        ids  = [f["input_ids"]      if isinstance(f["input_ids"],      torch.Tensor) else torch.tensor(f["input_ids"])      for f in feats]
        am   = [f["attention_mask"] if isinstance(f["attention_mask"], torch.Tensor) else torch.tensor(f["attention_mask"]) for f in feats]
        labs = [f["labels"]         if isinstance(f["labels"],         torch.Tensor) else torch.tensor(f["labels"])         for f in feats]

        pad_id = self.tok.pad_token_id
        ids  = pad_sequence(ids,  batch_first=True, padding_value=pad_id)
        am   = pad_sequence(am,   batch_first=True, padding_value=0)
        labs = pad_sequence(labs, batch_first=True, padding_value=pad_id)
        labs = labs.masked_fill(labs.eq(pad_id), self.label_pad_token_id)

        ids  = self._maybe_pad_to_multiple(ids,  pad_id)
        am   = self._maybe_pad_to_multiple(am,   0)
        labs = self._maybe_pad_to_multiple(labs, self.label_pad_token_id)
        return {"input_ids": ids, "attention_mask": am, "labels": labs}


def make_datasets(
    tokenizer_name: str = "google/flan-t5-base",
    train_split: str = "train",
    val_split: Optional[str] = "validation",
    test_split: Optional[str] = "test",
    max_input_len: int = 1024,
    max_target_len: int = 256,
    prefix_text: str = "summarize: ",
    *,
    self_split: bool = False,
    self_split_seed: int = 1337,
    self_split_val: float = 0.1,
    self_split_test: float = 0.1,
):

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(DATASET_ID)

    from datasets import DatasetDict

    if self_split:
        base = ds["train"].train_test_split(test_size=self_split_test, seed=self_split_seed)
        train_part = base["train"]
        test_part = base["test"]
        vt = train_part.train_test_split(
            test_size=self_split_val / (1.0 - self_split_test), seed=self_split_seed)
        ds = DatasetDict({
            "train": vt["train"],
            "validation": vt["test"],
            "test": test_part,
        })

    # Validate required columns exist
    for split in [s for s in [train_split, val_split, test_split] if s and s in ds]:
        cols = ds[split].column_names
        if INPUT_COL not in cols or TARGET_COL not in cols:
            raise KeyError(f"Expected columns '{INPUT_COL}', '{TARGET_COL}' in split '{split}', found {cols}")

    def encode_batch(batch):
        srcs = [prefix_text + s for s in batch[INPUT_COL]]
        enc = tok(srcs, max_length=max_input_len, truncation=True)
        tgt = tok(text_target=batch[TARGET_COL], max_length=max_target_len, truncation=True)
        enc["labels"] = tgt["input_ids"]
        return enc

    remove_cols = ds[train_split].column_names
    train_proc = ds[train_split].map(encode_batch, batched=True, remove_columns=remove_cols, desc="Tokenizing train")

    val_proc = None
    if val_split and val_split in ds:
        remove_cols_val = ds[val_split].column_names
        val_proc = ds[val_split].map(encode_batch, batched=True, remove_columns=remove_cols_val, desc="Tokenizing val")

    test_proc = None
    if test_split and test_split in ds:
        remove_cols_test = ds[test_split].column_names
        test_proc = ds[test_split].map(encode_batch, batched=True, remove_columns=remove_cols_test, desc="Tokenizing test")

    collator = Seq2SeqCollatorFast(tok, label_pad_token_id=-100, pad_to_multiple_of=None)
    return tok, train_proc, val_proc, test_proc, collator
