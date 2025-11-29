# Transformer Architecture

> **"Attention is All You Need"** - The architecture that replaced RNNs

---

## 🎯 Overview

Proposed by Vaswani et al. (2017).
**Key Idea:** Discard recurrence and convolutions entirely. Rely solely on attention mechanisms.
**Benefit:** Parallelizable training (unlike RNNs which are sequential).

---

## 🏗️ Architecture Components

### 1. Encoder Stack
- Stack of $N=6$ identical layers.
- Each layer has 2 sub-layers:
    1.  **Multi-Head Self-Attention**
    2.  **Position-wise Feed-Forward Network**
- Residual connection around each sub-layer + Layer Normalization.
- Output: `LayerNorm(x + Sublayer(x))`

### 2. Decoder Stack
- Stack of $N=6$ identical layers.
- Each layer has 3 sub-layers:
    1.  **Masked Self-Attention** (Prevents looking at future tokens)
    2.  **Encoder-Decoder Attention** (Q comes from decoder, K/V from encoder)
    3.  **Feed-Forward Network**

### 3. Positional Encoding
Since Transformers have no recurrence, they have no notion of order.
We inject position information by adding vectors to input embeddings.

$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}}) $$

---

## 💻 PyTorch Implementation (Simplified)

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 1. Self-Attention + Residual + Norm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

# Usage
block = TransformerBlock(d_model=512, num_heads=8)
x = torch.randn(10, 20, 512)
out = block(x)
print(f"Block Output: {out.shape}")
```

---

## 🔍 Key Differences from RNNs

| Feature | RNN / LSTM | Transformer |
| :--- | :--- | :--- |
| **Sequentiality** | Yes (Must process $t$ before $t+1$) | No (Parallel processing) |
| **Long-term Dependency** | Path length $O(n)$ | Path length $O(1)$ |
| **Complexity** | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| **Inductive Bias** | Strong (Temporal) | Weak (Must learn positions) |

---

## 🎓 Interview Focus

1.  **Why do we need Positional Encodings?**
    - Because self-attention is permutation invariant. Without PE, "The dog bit the man" and "The man bit the dog" would look identical to the model (bag of words).

2.  **What is the purpose of Layer Normalization?**
    - Stabilizes training and allows for faster convergence. Applied *per sample* (unlike Batch Norm).

3.  **Why is the FFN dimension usually 4x the model dimension?**
    - To project the data into a higher-dimensional space where it might be easier to separate or manipulate, then project back.

---

**The Transformer: The foundation of BERT, GPT, and modern AI!**
