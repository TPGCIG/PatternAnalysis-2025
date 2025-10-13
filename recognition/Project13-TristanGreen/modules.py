#modules.py
"""
General design ideas are that the datasets are defensively imported and are not
taken for granted since this is public software. All imports have guardrails.
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def get_tokenizer(name: str = "google/flan-t5-base"):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(
    name: str = "google/flan-t5-base",
    dtype: Optional[torch.dtype] = torch.float16,  # <-- use 'dtype', not 'torch_dtype'
    device_map: Optional[str] = None,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name,
        dtype=dtype,           # <-- fixes deprecation
        device_map=device_map,
    )
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    return model

