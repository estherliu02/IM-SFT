---
library_name: peft
base_model: meta-llama/Llama-3.1-8B-Instruct
---

# LoRA Adapter

This adapter was trained using LoRA (Low-Rank Adaptation) technique.

## Model Details
- Base model: meta-llama/Llama-3.1-8B-Instruct
- LoRA rank: 8
- LoRA alpha: 32.0
- Target modules: q_proj,v_proj

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "output/llama3-8b_im_sft/step_50")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```
### Framework versions

- PEFT 0.15.2