# ------------------------------------------------------------
#  Interactive CLI for Brain-T5 (Chat Mode)
#  -----------------------------------------------------------
#  Description:
#     Lightweight interface for real-time summarization queries.
#     Runs inference loop over the fine-tuned LoRA FLAN-T5 model.
#
#  Usage:
#     $ python chat.py --model_dir runs/flan_t5_base_lora_biolaysumm
#
#  Notes:
#     - Press Enter to re-prompt; type 'exit' or 'quit' to stop.
# ------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import argparse

p = argparse.ArgumentParser()

p.add_argument("--model_dir", required=True)

# --- config ---
ADAPTER_DIR = p.parse_args().model_dir
BASE_MODEL = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX = "summarize: "
MAX_INPUT_LEN = 1024
MAX_NEW_TOKENS = 256
NUM_BEAMS = 4


# --- load model ---
print("Loading model...")
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(DEVICE).eval()
print("Ready.")

# --- chat loop ---
while True:
    user = input("\n🧠 You: ").strip()
    if not user:
        continue
    if user.lower() in {"exit", "quit", "q"}:
        print("Bye.")
        break

    enc = tok(PREFIX + user, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(**enc,
                             max_new_tokens=MAX_NEW_TOKENS,
                             num_beams=NUM_BEAMS,
                             no_repeat_ngram_size=3,
                             early_stopping=True)
    print("\n🤖 Model:", tok.decode(out[0], skip_special_tokens=True))
