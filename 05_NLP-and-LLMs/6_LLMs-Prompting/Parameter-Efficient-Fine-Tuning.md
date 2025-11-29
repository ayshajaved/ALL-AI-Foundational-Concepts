# Parameter-Efficient Fine-Tuning (PEFT)

> **Fine-tuning giants on consumer hardware** - LoRA, QLoRA, and Adapters

---

## 🎯 The Problem

Fine-tuning Llama-3-70B requires updating 70B parameters.
- Needs 100s of GBs of VRAM.
- Creates a huge new model file for every task.

**Solution:** PEFT. Update only a tiny subset of parameters.

---

## 🦁 LoRA (Low-Rank Adaptation)

**Idea:** The change in weights $\Delta W$ has a low intrinsic rank.
Instead of updating the full matrix $W$ ($d \times d$), we decompose the update into two small matrices $A$ ($d \times r$) and $B$ ($r \times d$), where $r \ll d$.

$$ W_{new} = W_{frozen} + B \cdot A $$

- **Rank (r):** Usually 8, 16, or 64.
- **Result:** Reduces trainable parameters by 10,000x (from 70B to 10M).
- **VRAM:** Can fine-tune Llama-7B on a single 16GB GPU.

---

## 🧊 QLoRA (Quantized LoRA)

Combines **4-bit Quantization** (NF4) of the base model with **LoRA**.
- Base model is frozen in 4-bit.
- LoRA adapters are in FP16/BF16.
- **Efficiency:** Fine-tune 65B models on a single 48GB GPU.

---

## 💻 HuggingFace PEFT Library

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,            # Rank
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# "trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06"
```

---

## 🎓 Interview Focus

1.  **Does LoRA increase inference latency?**
    - **No.** During inference, we can merge the adapter weights back into the base model: $W = W + BA$. It becomes a standard model again.

2.  **What is Catastrophic Forgetting?**
    - When fine-tuning makes the model forget its original knowledge (e.g., how to code). PEFT mitigates this because most of the original weights are frozen.

3.  **LoRA vs Full Fine-Tuning?**
    - LoRA achieves performance comparable to full fine-tuning for most tasks, with 1% of the cost.

---

**PEFT: Democratizing LLM training!**
