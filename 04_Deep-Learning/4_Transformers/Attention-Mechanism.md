# Attention Mechanism

> **Focusing on what matters** - The concept that revolutionized NLP

---

## 🎯 The Bottleneck Problem

In standard Seq2Seq (Encoder-Decoder RNNs), the encoder must compress the *entire* input sequence into a single fixed-size **context vector**.

**Issue:** Information loss for long sequences.
**Solution:** Attention allows the decoder to look at *all* encoder hidden states dynamically.

---

## 📊 How Attention Works

At each decoding step $t$:
1.  **Calculate Scores:** Compare decoder hidden state $s_{t-1}$ with all encoder hidden states $h_1, ..., h_N$.
2.  **Calculate Weights:** Softmax the scores to get attention weights $\alpha_{ti}$ (sum to 1).
3.  **Context Vector:** Compute weighted sum of encoder states: $c_t = \sum \alpha_{ti} h_i$.
4.  **Output:** Use $c_t$ and $s_{t-1}$ to predict the next word.

### Score Functions
- **Dot Product:** $score(s, h) = s^T h$
- **General (Luong):** $score(s, h) = s^T W h$
- **Concat (Bahdanau):** $score(s, h) = v^T \tanh(W [s; h])$

---

## 💻 Implementation (Dot Product Attention)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, 1, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        
        # 1. Calculate Scores (Dot Product)
        # (batch, 1, hidden) * (batch, hidden, seq_len) -> (batch, 1, seq_len)
        scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
        
        # 2. Calculate Weights (Softmax)
        attn_weights = F.softmax(scores, dim=2)
        
        # 3. Calculate Context Vector
        # (batch, 1, seq_len) * (batch, seq_len, hidden) -> (batch, 1, hidden)
        context = torch.bmm(attn_weights, encoder_outputs)
        
        return context, attn_weights

# Example
batch_size = 5
seq_len = 10
hidden_size = 20

decoder_hidden = torch.randn(batch_size, 1, hidden_size)
encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)

attn = Attention()
context, weights = attn(decoder_hidden, encoder_outputs)

print(f"Context shape: {context.shape}") # (5, 1, 20)
print(f"Weights shape: {weights.shape}") # (5, 1, 10)
```

---

## 👁️ Visualizing Attention

Attention weights $\alpha_{ti}$ can be visualized as a heatmap (Alignment Matrix).
- Rows: Decoder steps (Generated words)
- Columns: Encoder steps (Source words)
- **Interpretation:** Which source word was the model "looking at" when generating a specific target word?

---

## 🎓 Interview Focus

1.  **Why is Attention better than a fixed context vector?**
    - It creates a direct path from any encoder state to the decoder, solving the long-term dependency problem and vanishing gradient problem.

2.  **Difference between Bahdanau and Luong Attention?**
    - **Bahdanau (Additive):** Uses a feed-forward network to calculate scores. Applied *before* the decoder update.
    - **Luong (Multiplicative):** Uses dot product. Applied *after* the decoder update.

3.  **Is Attention only for NLP?**
    - No. Used in Computer Vision (Image Captioning), Speech, and even Graph Neural Networks.

---

**Attention: The key to modern AI!**
