<h3 align="center">
    <p>Brain-T5: A lightweight model fine-tuned for simplifying medical jargon using FLAN-T5 and LoRA.</p>
</h3>

Brain-T5 is a lightweight language model designed to translate technical clinical and biomedical text into layperson summaries so non-experts can understand them. Built on top of FLAN-T5 using LoRA fine-tuning, it is deployable on consumer grade GPUs and acts to assist research into medical fields from outer disciplines and acts as an assistant for patient communication. This repository includes full training, evaluation and inference pipelines, from dataset intake to an interactive chat mode.

## Project Motivation:
Between medical professionals and the average person or researcher in an outer discipline, the scope of what "standard language" is does not cross over very well. Jargon is used excessively inside the medical world which may cause outer folk to struggle to understand basic summaries, research abstracts/results, or diagnostic reports. The only tools that exist that fit this use case effectively are large language models such as OpenAI's GPT-3+, Google's Gemini, Anthropic's Sonnet and others, however they cannot be localised easily on consumer grade hardware and use inputted conversational data to train their models. Many medical institutions may not want their data to cross borders, making a local option preferrable.

Brain-T5 aims to close this gap by:

- Using a fine-tuning approach with LoRA to an existing reliable text model
- Using the lightweight T5 model from Hugging Face trained on reliable medical summarisations. 
- Using an open source, easy-to-install model that can be used on average consumer-grade hardware.

Brain-T5 is a major step toward bridging the gap between the average person and medical knowledge and aims to enhance both clinical practices and interdisciplinary research around the world.

## Features:
- **LoRA-based fine-tuning** - train large models on consumer-grade GPUs.
- **Supports HuggingFace datasets, CSV, JSON** - flexible with data types.
- **Built-in ROUGE evaluation** - automatic scoring after each training epoch.
- **Interactive Chat CLI(`chat.py`)** - real-time inference like a medical assistant.
- **Modular codebase** - easy to extend or adapt to alternative domains (legal, finance, etc.)

## Project Structure
```
├── train.py          # Full training pipeline with metrics and logging
├── predict.py        # Batch inference on JSONL or single text
├── chat.py           # Interactive CLI for conversational testing
├── modules.py        # Tokenizer/model loaders and LoRA attachment
├── dataset.py        # Dataset wrapper and fast collator
├── runs/             # LoRA adapters & metrics saved here
└── README.md
```

## Demo Examples:

## Installation:

## Training Usage:

## Chat Usage:

## Training Resuts:





README structure heavily inspired by HF transformers pag

Train e1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150454/150454 [5:20:10<00:00,  7.83batch/s, loss=1.3393, sps=7.8] 
[epoch 1] train_loss=1.3393
Eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [1:01:16<00:00,  2.94s/batch] 
[epoch 1] ROUGE: {'rouge1': 0.639758940949706, 'rouge2': 0.4262667182806449, 'rougeL': 0.5793631565756041, 'rougeLsum': 0.5795225947980385}
[epoch 1] ✓ saved best adapters to runs/flan_t5_base_lora_biolaysumm_debug
done.











