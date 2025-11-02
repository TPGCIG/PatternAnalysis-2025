<p align="center">
  <img src="assets/images/braint5.png" alt="Logo" />
</p>


<h3 align="center">
    <p>Brain-T5: A lightweight model fine-tuned for simplifying medical jargon using FLAN-T5 and LoRA.</p>
</h3>

Brain-T5 is a lightweight language model designed to translate technical clinical and biomedical text into layperson summaries so non-experts can understand them. Built on top of FLAN-T5 using LoRA fine-tuning, it is deployable on consumer grade GPUs and acts to assist research into medical fields from outer disciplines and acts as an assistant for patient communication. This repository includes full training, evaluation and inference pipelines, from dataset intake to an interactive chat mode.

## Table of Contents
- Brain-T5
  - [Project Motivation](#project-motivation)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Training Usage](#training-usage)
  - [Chat Usage](#chat-usage)
  - [Training Results](#training-results)
- The FLAN-T5 Model
  - [What is FLAN-T5?](#what-is-flan-t5)
  - [Why not other models?](#why-not-other-models)

## Project Motivation:
Between medical professionals and the average person or researcher in an outer discipline, the scope of what "standard language" is does not cross over very well. Jargon is used excessively inside the medical world which may cause outer folk to struggle to understand basic summaries, research abstracts/results, or diagnostic reports. The only tools that exist that fit this use case effectively are large language models such as OpenAI's GPT-3+, Google's Gemini, Anthropic's Sonnet and others, however they cannot be localised easily on consumer grade hardware and use inputted conversational data to train their models. Many medical institutions may not want their data to cross borders, making a local option preferrable.

Brain-T5 aims to close this gap by:

- Using a fine-tuning approach with LoRA to an existing reliable text model
- Using the lightweight T5 model from Hugging Face trained on reliable medical summarisations. 
- Using an open source, easy-to-install model that can be used on average consumer-grade hardware.

Brain-T5 is a major step toward bridging the gap between the average person and medical knowledge and aims to enhance both clinical practices and interdisciplinary research around the world.

## Features:
* **LoRA-based fine-tuning** - train large models on consumer-grade GPUs.
* **Supports HuggingFace datasets, CSV, JSON** - flexible with data types.
* **Built-in ROUGE evaluation** - automatic scoring after each training epoch.
* **Interactive Chat CLI(`chat.py`)** - real-time inference like a medical assistant.
* **Modular codebase** - easy to extend or adapt to alternative domains (legal, finance, etc.)

## Project Structure
```
├── train.py          # Full training pipeline with metrics and logging
├── predict.py        # Batch inference on JSONL or single text
├── chat.py           # Interactive CLI for conversational testing
├── modules.py        # Tokeniser/model loaders and LoRA attachment
├── dataset.py        # Dataset wrapper and fast collator
├── runs/             # LoRA adapters & metrics saved here
└── README.md
```

## Installation:

```
# First, clone the repository and at the same time, checkout the topic-recognition branch.
git clone -b topic-recognition https://github.com/TPGCIG/PatternAnalysis-2025/

# Change directory to the Brain-T5 one.
cd recognition/Project13-TristanGreen

# Install the dependencies.
pip install -r requirements.txt
```

You're now ready to go!

## Training Usage


### 1) Quick-start commands
**Hugging Face (BioLaySumm)**
> Requires `pip install datasets`. Uses the built-in dataset loader.  
```bash
python train.py --output_dir [dir_name]
```

### 2) For fine-grain training and control over parameters
```bash
python train.py   --train_source hf --train_path train   --val_source   hf --val_path validation   --output_dir runs/flan_t5_base_lora_biolaysumm   --batch_size 1 --accum 16 --epochs 3 --lr 2e-4 --fp16
```

### 3) What the script actually does
- Builds tokenizer + datasets via `make_datasets(...)` with `hf` and columns (`--input_col`, `--target_col`).  
- Attaches **LoRA** adapters to FLAN‑T5 and trains with AdamW + cosine schedule.  
- Evaluates with **ROUGE** at epoch. 
- Saves best adapters + tokenizer to `--output_dir`, along with `metrics.json`, `train_log.csv` and graphs for `loss` and `ROUGE` scores per-epoch.

### 4) Arguments
- **Batching**: `--batch_size`, `--accum` (effective batch = batch_size × accum)
- **Optim**: `--lr`, `--weight_decay`, `--warmup_steps`, `--clip`
- **LoRA**: `--lora_r`, `--lora_alpha`, `--lora_dropout`
- **Eval**: `--eval_batch_size`, `--eval_max_new_tokens`, `--eval_beams`
- **Misc**: `--epochs`, `--seed`, `--fp16`

### 5) Outputs (verify or it didn’t happen)
Inside your `--output_dir`:
```
runs/<name>/
├── adapter_config.json          # LoRA adapter setup (rank, alpha, target modules)
├── adapter_model.safetensors    # Actual trained LoRA weight deltas
├── hardware.json                # GPU name, VRAM, and compute capability info
├── history_val.csv              # Validation ROUGE scores per epoch (for plotting)
├── metrics_test.json            # Final held-out test ROUGE scores
├── metrics_val.json             # Best validation epoch and its ROUGE metrics
├── special_tokens_map.json      # Token IDs for <pad>, <eos>, etc. (auto from tokenizer)
├── README.md                    # Auto-generated PEFT model card (safe to delete)
├── tokenizer.json               # Full tokenizer vocab + merges
├── tokenizer_config.json        # Tokenizer settings (truncation, padding, etc.)
├── time.json                    # Total training time and epochs elapsed
├── rouge_val_curve.png          # Line plot — validation ROUGE vs epoch
├── rouge_test_bar.png           # Bar chart — test ROUGE metrics
├── loss_curve.png               # Smoothed training loss vs steps curve
├── params.json                  # Total vs trainable parameter counts (LoRA ratio)
├── metrics.json                 # Duplicate or summary of best ROUGE metrics
└── train_log.csv                # Step-wise loss log for generating loss_curve
```
You should also see console logs during training like:
```
[epoch 1] train_loss=...
[epoch 1] ROUGE: {'rouge1': ..., 'rouge2': ..., 'rougeL': ..., 'rougeLsum': ...}
```

### 6) Use the trained adapters
Single text:
```bash
python predict.py --adapter_dir runs/<name> --text "Put clinical text here" --fp16
```
Batch JSONL:
```bash
python predict.py --adapter_dir runs/<name> --jsonl dev.jsonl --input_col report --out_path predictions.jsonl
```

### 7) General usage tips 
- If CUDA OOM: increase `--accum`, or drop `--fp16` if your GPU cannot handle the defaults.
- If ROUGE is flat, your data columns are probably wrong. Print a few samples.
- If `runs/<name>` is empty, you never beat your previous best—check learning rate and dataset.

## Chat Usage:

### Examples
> This sample was trained on 7 epochs over a 39 hour period, better results can come as result of longer training and adjustable parameters. The rouge scores for this training are seen in [Training Usage](#training-usage)

You: “CT angiography shows ~70% stenosis of the proximal LAD; plan for PCI if symptoms persist despite maximal medical therapy.” <br>
Model: “There’s a notable blockage in a main heart artery. The team suggests a procedure if medicines don’t help enough.”

You: “MRI brain: multiple periventricular T2/FLAIR hyperintensities suggestive of demyelination; correlate clinically for MS.”<br>
Model: “Scans show several spots in brain areas often linked to nerve coating problems. Doctors will compare this with symptoms.”

You: “HbA1c 9.2% despite metformin 1g BID; initiate basal insulin 10u qHS and titrate to fasting 5–7 mmol/L.”<br>
Model: “Blood sugar control isn’t good with current meds. Start a nightly insulin dose and adjust to get morning sugars into a better range.”

You: “Echo: LVEF 35%, global hypokinesis, mild MR; optimize GDMT—ACEi/ARB, beta-blocker, MRA.”<br>
Model: “Heart pumping is weaker than normal with some valve leak. Use standard lung medicines to improve function.”

### Error Analysis
The model consistently drops critical figures (example 1: ~70%, example 3: 9.2%) and summarises the responses with shallow descriptions that lose a lot of meaning. This is likely since, while T5 uses attention out-of-the-box, it doesn't learn to prioritise clinically critical details like these ones. This is likely since we are fine-tuning the model that is fundamentally trained on the [Common Crawl](https://commoncrawl.org/) dataset which does not prioritise numeric values or risk markers. In downstream fine-tuning, the model should be explicitly taught that figures have substantial meaning and should be valued more than other tokens that the model processes which the Common Crawl dataset is trained to do.

The model also hallucinates (rarely though) (example 4: *use standard lung medicines* in heart context) and can forget the context loosely. This is obviously problematic but in downstream fine-tuning, the model should be explicitly over-attentive to the context as mistakes of this nature can cause problems in the medical field.  
These limitations, while problematic, can be overcome via training on consumer grade hardware. Also, under proper supervision from a medical professional, these bugs can be quickly identified and flagged.


## Training Resuts:
Training was performed on the BioLaySumm 2025 - LaymanRRG opensource track, using FLAN-T5-Base with LoRA fine-tuning for 3 epochs.
The model was trained with AdamW + cosine schedule, batch size 1 × gradient accumulation 16 (effective batch = 16), and evaluated with ROUGE-1/2/L/Lsum per epoch.

1. Training Loss (full run)
<img src="assets/images/loss_curve_full.png" width="600"/>

This plot shows the training loss vs optimizer steps over the entire fine-tuning run.
The curve steadily declines and stabilises, showing smooth convergence without major oscillation — indicating that:

* The learning rate and warm-up schedule were well-tuned.

* Gradient accumulation was effective in maintaining numerical stability under mixed-precision (--fp16) training.

* No gradient explosions or plateaus occurred (loss range ≈ 1.9 → 1.2).

2. Training Loss (medium zoom)
<img src="assets/images/loss_curve_med.png" width="600"/>

This is a zoomed-in view of the mid-training regime, showing finer granularity of step-wise noise.
Loss fluctuations at small scale are expected from single-sample batches, but the general slope continues downward, confirming consistent optimization rather than overfitting spikes.

3. Validation ROUGE Progress (full run)
<img src="assets/images/rouge_val_curve_full.png" width="600"/>

This figure tracks ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum per epoch.

Interpretation:

* ROUGE-1 and ROUGE-L steadily improve and plateau by the third epoch, showing that lexical and long-span coherence both increased.

* ROUGE-2 remains noisier, which is typical for summarization tasks where exact bigram matches are less frequent.

* The consistent upward trajectory across all four metrics indicates learning stability and effective LoRA adaptation.

4. Validation ROUGE (medium zoom)
<img src="assets/images/rouge_val_curve_med.png" width="600"/>

This mid-range view highlights the epoch-to-epoch change more clearly:

* Rapid early gains in the first epoch.

* Smaller, diminishing returns after epoch 2, suggesting convergence.

*  No regression in ROUGE-Lsum,  evidence that the checkpoint selected (highest ROUGE-Lsum) indeed corresponds to the global optimum seen during training.

Overall, Brain-T5 demonstrates reliable convergence and solid generalisation across validation and test splits.
The model maintains smooth training dynamics and rising ROUGE performance without evidence of overfitting or divergence — validating the correctness of the pipeline in train.py and the dataset tokenization logic in dataset.py


# The FLAN-T5 Model
## What is T5?
T5 (Text-to-Text Transfer Transformer) is a transformer model built completely on a text-to-text framework. This framework treats every task in Natural Language Processing (NLP), whether it be machine translation, summarisation, or question-answering, as a process of taking text as input and producing text as output. This unification allows the same model architecture, objective function, and training procedure to be applied across all tasks, massively simplifying the entire NLP training pipeline.

<img src='assets/images/t5architecture.jpg'>
The super summarised explanation on how the model works (provided by OpenAI's ChatGPT) is:

> This is the quoted text from ChatGPT.

1. Tokenise → Embed → + Position. Words become vectors, add position info so the model knows order.
2. Encoder (repeated N×):
   - Self-attention: each word looks at all other words to decide what matters.
   - Feed-forward: a per-token mini-MLP to transform features.
   - Add & Norm: residual skip + layer norm to keep training stable. <br>Output = contextual vectors for every input token.

3. Decoder input: start with <pad>/<bos> and previously generated tokens shifted right.
4. Decoder block (repeated N×):
   - Masked self-attention: looks only at past output tokens (mask stops peeking at the future).
   - Cross-attention: queries the encoder outputs so the decoder can “look up” relevant parts of the input.
   - Feed-forward and Add & Norm again.

5. Linear → Softmax: turn the decoder’s last vector into a probability over the vocabulary; pick the next token; loop 4–5 until done.

> End quote.

<img src='assets/images/t5simple.png'>

This is a representation of how T5 unifies all forms of text-to-text input/output to heavily generalise its use case and simply learning.

## What is FLAN-T5?
FLAN-T5 (Fine-tuned LAnguate Net T5) is an enhanced version of the original [T5](https://medium.com/analytics-vidhya/t5-a-detailed-explanation-a0ac9bc53e51) but fine-tuned using a technique called instruction tuning.
During training, FLAN-T5 is exposed to a massive number of tasks that are all formatted as natural language instructions (e.g. "Answer the following question: ..."). This training paradigm significantly improves the model's ability to:

1. **Follow instructions** since it is built on user prompts instead of general text data.
2. **Generalise** since the training prompts may map a new, prompted task out for the model to answer which can help it understand how to answer newer tasks it previously couldnt.
3. **Transfer knowledge efficiently** because FLAN-T5 was trained on diverse, instruction-formatted datasets, it can quickly adapt to unseen downstream tasks (like layperson medical summarization) with relatively few gradient updates.
4. **Reduce hallucination and bias** as tuning encourages models to anchor their responses to explicit prompts, producing more deterministic and context-aware outputs compared to raw pretrained T5 models.

In essence, FLAN-T5 represents a major leap in making large-scale text-to-text models usable out of the box for a wide range of natural language tasks. Its combination of instructional alignment, broad coverage, and generalization ability makes it a strong backbone for fine-tuning in specialized domains, such as Brain-T5, where the goal is translating complex biomedical text into accessible language without requiring massive compute resources.


