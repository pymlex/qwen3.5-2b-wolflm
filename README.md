# Qwen3.5-2B WolfLM

## Overview

`Qwen3.5-2B` is a pre-trained instruct SLM. The target is to teach it generate wolf quotes in Russian. We apply CPT and LoRA fine-tuning on the `pymlex/wolf-quotes` dataset.

## Dataset

The dataset contains 95k quote samples with a single `quote` field. Each sample is converted into chat format:

```text
<|im_start|>user
Напиши короткую глубокую цитату.<|im_end|>
<|im_start|>assistant
<think>

</think>

Жизнь - хороший учитель, но слишком дорого берет за уроки.<|im_end|>
```

Quotes longer than 64 tokens are filtered out before training.

![image](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/GbprwgWrGzc1VM56sn0yW.png)

## CPT

Continued pretraining was performed with the following setup:

* base model: `Qwen/Qwen3.5-2B`
* dataset: `pymlex/wolf-quotes`
* max sequence length: `96`
* batch size: `64`
* epochs: `1`
* learning rate: `1e-4`
* scheduler: cosine
* optimiser: `adamw_torch`
* LoRA rank: `16`
* LoRA alpha: `32`
* LoRA dropout: `0.05`

We set `num_cycles` for the scheduler to `0.35` so the learning rate doesn't decrease to zero at last steps.

![image](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/61enT28R32y6JuJ9wVu1-.png)

## Loss curves

Training and validation losses are tracked during the run and saved together:

![image](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/L-t_fJYV7XFR_N8F8iDUK.png)

A log-scaled version:

![image](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/FK1-EmMJmPIXHtI-gh2dP.png)

## Inference

Install dependencies. This code is suitable for Google Colab.

```bash
!pip install --upgrade transformers torchao
```

Load the base model and adapter from Hub.

```python
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "Qwen/Qwen3.5-2B"
adapter_hub_id = "pymlex/qwen3.5-2b-wolflm"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_hub_id)
model.eval()
```

Generate a quote.

```python
def generate_quote(max_new_tokens=100, temperature=0.9, top_p=0.9):
    messages = [{"role": "user", "content": "Напиши короткую глубокую цитату."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_part = full_response.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip()
    assistant_part = re.sub(r"<think>.*?</think>", "", assistant_part, flags=re.DOTALL).strip()
    return assistant_part

print(generate_quote())
```

## Perplexity evaluation

| Model           |   Loss | Perplexity |
| --------------- | -----: | ---------: |
| Base Qwen3.5-2B | 2.7744 |    16.0283 |
| WolfLM          | 1.3157 |     3.7274 |

The tuned model substantially outperforms the base checkpoint on the quote domain. Loss decreases by about `52.6%`, while perplexity drops by about `76.7%`. 
