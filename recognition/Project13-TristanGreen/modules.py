# ------------------------------------------------------------
#  Model Utilities for Brain-T5
#  -----------------------------------------------------------
#  Description:
#     Provides helper functions for loading base models and attaching
#     LoRA adapters to target layers of FLAN-T5.
#
#  Key Functions:
#     - load_base_model(): loads pretrained T5/FLAN-T5 with dtype control.
#     - attach_lora(): injects trainable low-rank adapters for fine-tuning.
#
#  Notes:
#     - Uses PEFT (Parameter-Efficient Fine-Tuning) via Hugging Face.
#     - Keeps original model frozen except LoRA-injected parameters.
# ------------------------------------------------------------
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
    dtype: Optional[torch.dtype] = torch.float16,
    device_map: Optional[str] = None,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name,
        dtype=dtype,
        device_map=device_map,
    )
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    return model


def attach_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05, target_modules: Optional[List[str]] = None):
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft not installed. `pip install peft` to use LoRA.")
    if target_modules is None:
        target_modules = ["q", "k", "v", "o"]
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="SEQ_2_SEQ_LM",
    )
    return get_peft_model(model, cfg)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    inputs: List[str],
    max_input_len: int = 1024,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    add_prefix: bool = True,
    prefix_text: str = "summarize: ",
    device: Optional[str] = None,
) -> List[str]:
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    batch = [(prefix_text + x) if add_prefix else x for x in inputs]
    enc = tokenizer(
        batch,
        max_length=max_input_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        early_stopping=True,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)
