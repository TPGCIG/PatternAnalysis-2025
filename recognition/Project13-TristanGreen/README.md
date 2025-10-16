<h3 align="center">
    <p>Brain-T5: A lightweight model fine-tuned for simplifying medical jargon using FLAN-T5 and LoRA.</p>
</h3>
 
## Project Motivation:

## Demo Examples:

## Features:

## Installation:

## Training Usage:

## Chat Usage:

## Training Resuts:





README structure heavily inspired by HF transformers page.


Train e1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150454/150454 [5:20:10<00:00,  7.83batch/s, loss=1.3393, sps=7.8] 
[epoch 1] train_loss=1.3393
Eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [1:01:16<00:00,  2.94s/batch] 
[epoch 1] ROUGE: {'rouge1': 0.639758940949706, 'rouge2': 0.4262667182806449, 'rougeL': 0.5793631565756041, 'rougeLsum': 0.5795225947980385}
[epoch 1] ✓ saved best adapters to runs/flan_t5_base_lora_biolaysumm_debug
done.











