# GPT Architecture Deep Dive

> **The Decoder-Only Stack** - Causal Masking, KV Cache, and Scaling

---

## 🏗️ Architecture: Decoder-Only

Unlike BERT (Encoder), GPT uses a **Masked Self-Attention** mechanism.
It cannot see the future tokens.

$$ P(w_t | w_{1:t-1}) $$

### Causal Masking
The attention matrix is masked with $-\infty$ above the diagonal.
Position $t$ can only attend to positions $1...t$.

```python
# Lower triangular mask
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

## ⚡ KV Cache (Key-Value Cache)

**Problem:** In generation, we predict one token at a time. Recomputing Attention for *all previous tokens* at every step is wasteful ($O(N^2)$).
**Solution:** Cache the Key ($K$) and Value ($V$) matrices of past tokens.
At step $t$, we only compute $Q_t, K_t, V_t$ and attend to $[K_{past}, K_t]$ and $[V_{past}, V_t]$.

**Complexity:** Reduces generation cost from $O(N^2)$ to $O(N)$ per token.

---

## 📈 GPT-2 vs GPT-3

| Feature | GPT-2 (2019) | GPT-3 (2020) |
| :--- | :--- | :--- |
| **Params** | 1.5B | 175B |
| **Context** | 1024 | 2048 |
| **Training** | WebText | CommonCrawl (filtered) |
| **Focus** | Zero-shot | **Few-shot (In-context Learning)** |
| **Layers** | 48 | 96 |
| **d_model** | 1600 | 12288 |

**Emergence:** GPT-3 showed that simply scaling up parameters and data unlocks capabilities (arithmetic, coding) not present in smaller models.

---

## 🎓 Interview Focus

1.  **Encoder vs Decoder?**
    - **Encoder (BERT):** Bi-directional. Good for understanding (Classification, NER).
    - **Decoder (GPT):** Uni-directional. Good for generation.

2.  **Why do we need KV Cache?**
    - To speed up autoregressive generation. Without it, generating a 1000-token sequence would be prohibitively slow as complexity grows quadratically.

3.  **What is In-Context Learning?**
    - The ability of LLMs to learn a task from examples provided in the prompt *without* updating weights.

---

**GPT: Predicting the next token at scale!**
