# Self-Attention and Multi-Head Attention

> **The core of the Transformer** - Attention within a single sequence

---

## 🎯 Self-Attention

Standard attention connects two *different* sequences (Encoder $\to$ Decoder).
**Self-Attention** connects a sequence to *itself*.

**Goal:** For each word in a sentence, understand its relationship with every other word in the same sentence.
*Example:* "The **animal** didn't cross the **street** because **it** was too tired."
Self-attention helps the model understand that "**it**" refers to "**animal**".

### Query, Key, Value (Q, K, V)
For each input vector $x$, we create three vectors:
1.  **Query ($q$):** What I am looking for?
2.  **Key ($k$):** What I contain?
3.  **Value ($v$):** What information I pass along?

$$q = W_Q x, \quad k = W_K x, \quad v = W_V x$$

### Scaled Dot-Product Attention
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

- **Dot Product ($QK^T$):** Similarity between queries and keys.
- **Scale ($\sqrt{d_k}$):** Prevents gradients from vanishing when dot products get too large.
- **Softmax:** Normalizes scores to probabilities.
- **Multiply by $V$:** Aggregate relevant information.

---

## 🐙 Multi-Head Attention

Instead of one attention mechanism, run multiple in parallel.

**Why?** Allows the model to focus on different positions and different *types* of relationships simultaneously (e.g., Head 1 focuses on syntax, Head 2 on semantics).

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $$
$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections & split heads
        # (batch, seq, d_model) -> (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concat heads
        # (batch, heads, seq, d_k) -> (batch, seq, heads, d_k) -> (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)

# Usage
d_model = 512
heads = 8
mha = MultiHeadAttention(d_model, heads)

x = torch.randn(10, 20, 512) # (batch, seq, d_model)
out = mha(x, x, x) # Self-attention: Q=K=V=x
print(f"Output shape: {out.shape}") # (10, 20, 512)
```

---

## 👁️ Visualizing Attention

Understanding what the model focuses on is crucial for interpretability.

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_attention(attention_weights, tokens):
    """
    Plots attention weights as a heatmap.
    
    Args:
        attention_weights: (seq_len, seq_len) numpy array
        tokens: List of strings (words/subwords)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, 
                xticklabels=tokens, 
                yticklabels=tokens, 
                cmap='viridis', 
                annot=False)
    plt.xlabel('Keys (Source)')
    plt.ylabel('Queries (Target)')
    plt.title('Self-Attention Heatmap')
    plt.show()

# Example Usage
# Assume 'attn_output_weights' is (seq_len, seq_len) from the model
tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"]
# plot_attention(attn_output_weights, tokens)
```

---

## 🎓 Interview Focus

1.  **Why divide by $\sqrt{d_k}$?**
    - To keep the magnitude of the dot product small. Large values push softmax into regions with extremely small gradients (vanishing gradients).

2.  **What is the complexity of Self-Attention?**
    - $O(n^2 \cdot d)$ where $n$ is sequence length. This quadratic complexity makes Transformers slow for very long sequences.

3.  **Why Multi-Head?**
    - Ensemble effect. Allows the model to capture different representation subspaces at different positions.

---

**Self-Attention: The engine of the Transformer!**
