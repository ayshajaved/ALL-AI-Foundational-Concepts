# Efficient Training Techniques

> **Training LLMs without a supercomputer** - FlashAttention, Quantization, and PEFT

---

## ⚡ FlashAttention (2022)

**Problem:** Standard Attention is $O(N^2)$ in memory and speed. It reads/writes large $N \times N$ matrices to HBM (High Bandwidth Memory - GPU RAM), which is slow.

**Solution:** **FlashAttention** computes attention *block-by-block* inside the GPU's fast SRAM (L1 Cache). It avoids writing the huge attention matrix to HBM.

**Result:**
- **Speed:** 2-4x faster training.
- **Memory:** Linear memory complexity $O(N)$ instead of $O(N^2)$. Enables context lengths of 32k+.

```python
# In PyTorch 2.0+, it's automatic!
# Just use scaled_dot_product_attention
import torch.nn.functional as F

out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

## 📉 Quantization (4-bit / 8-bit)

**Idea:** Represent weights with fewer bits.
- FP32 (32-bit): 4 bytes per param.
- INT8 (8-bit): 1 byte per param (4x smaller).
- NF4 (4-bit): 0.5 bytes per param (8x smaller).

**QLoRA:** Fine-tune a 4-bit quantized base model using Low-Rank Adapters.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained("llama-2-7b", quantization_config=bnb_config)
```

---

## 🚀 Gradient Checkpointing

**Trade-off:** Compute vs Memory.
Instead of storing all intermediate activations for backprop (memory heavy), **recompute** them during the backward pass.
Saves 50-70% memory at cost of 20% slowdown.

```python
model.gradient_checkpointing_enable()
```

---

## 🎓 Interview Focus

1.  **How does FlashAttention achieve speedup?**
    - By being **IO-aware**. It minimizes the number of reads/writes to the slow GPU HBM, keeping operations in the fast SRAM.

2.  **What is the difference between Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)?**
    - **PTQ:** Quantize a trained model (easy, slight acc loss).
    - **QAT:** Train with quantization errors simulated (harder, best acc).

3.  **Why use Gradient Checkpointing?**
    - To fit a larger batch size or a larger model into GPU memory.

---

**Efficiency: The art of fitting Llama-70B on a consumer GPU!**
