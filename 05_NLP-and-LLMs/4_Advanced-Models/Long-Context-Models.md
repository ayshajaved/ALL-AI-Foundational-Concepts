# Long Context Models

> **Beyond 512 tokens** - RoPE, ALiBi, and Sparse Attention

---

## 🚧 The Limit

Standard Transformers ($O(N^2)$) hit a wall at ~2k-4k tokens.
Memory explodes.
Positional encodings (Sinusoidal/Learned) fail to generalize beyond training length.

---

## 🪢 RoPE (Rotary Positional Embeddings)

**Used in:** Llama, PaLM, GPT-NeoX.
**Idea:** Encode position by **rotating** the query and key vectors in complex space.
$$ (x + iy) \cdot e^{i m \theta} $$

**Properties:**
- **Relative:** Attention depends only on relative distance $m-n$.
- **Extrapolation:** Generalizes better to longer sequences than absolute embeddings.

---

## 🏔️ ALiBi (Attention with Linear Biases)

**Used in:** MPT, BLOOM.
**Idea:** Don't add positional embeddings to inputs. Instead, add a static bias to the **attention scores** based on distance.
$$ Score(q_i, k_j) = q_i \cdot k_j - m \cdot |i-j| $$

**Superpower:** Can train on 2k tokens and infer on 8k+ tokens seamlessly.

---

## 🕸️ Sparse Attention (Longformer / BigBird)

**Idea:** Don't attend to *everyone*.
1.  **Local Window:** Attend to neighbors (e.g., $\pm 512$ tokens).
2.  **Global Tokens:** Special tokens (`[CLS]`) attend to everyone.

**Complexity:** $O(N)$ (Linear).
**Trade-off:** Harder to implement, sometimes lower performance on dense tasks.

---

## 🎓 Interview Focus

1.  **Absolute vs Relative Positional Encoding?**
    - **Absolute (BERT):** "I am at index 5". Fails if test seq > train seq.
    - **Relative (RoPE/ALiBi):** "I am 3 steps away from you". Extrapolates better.

2.  **How does RoPE work intuitively?**
    - Imagine the vector as a clock hand. Position is encoded by the angle. Dot product (similarity) depends on the angle difference (relative distance).

3.  **Why is Long Context hard?**
    - KV Cache grows linearly (VRAM usage).
    - Attention matrix grows quadratically (Compute).

---

**Long Context: Reading entire books in one go!**
