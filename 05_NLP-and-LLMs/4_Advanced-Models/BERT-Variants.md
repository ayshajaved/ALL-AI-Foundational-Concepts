# BERT Variants

> **Optimizing the Encoder** - RoBERTa, DistilBERT, ALBERT, and DeBERTa

---

## 🚀 RoBERTa (Robustly Optimized BERT) - 2019

**Key Insight:** BERT was significantly undertrained.
**Changes:**
1.  **More Data:** Trained on 160GB (vs BERT's 16GB).
2.  **No NSP:** Removed Next Sentence Prediction (found to be unhelpful).
3.  **Dynamic Masking:** Masking pattern changes every epoch (vs static masking in BERT).
4.  **Larger Batches:** 8k batch size.

**Result:** Consistently outperforms BERT. The default "BERT" choice today.

---

## ⚡ DistilBERT (Distilled BERT) - 2019

**Goal:** Reduce size and increase speed while retaining performance.
**Method:** Knowledge Distillation.
- **Teacher:** BERT-Base.
- **Student:** DistilBERT (40% smaller, 60% faster, 97% performance).
- **Architecture:** Removed token-type embeddings and pooler. Halved the number of layers (6 vs 12).

$$ Loss = L_{ce} + L_{cosine} $$

---

## 🧩 ALBERT (A Lite BERT) - 2019

**Goal:** Reduce parameters to fit larger models in memory.
**Techniques:**
1.  **Factorized Embedding Parameterization:** Split large vocabulary matrix ($V \times H$) into two ($V \times E$ and $E \times H$).
2.  **Cross-Layer Parameter Sharing:** All layers share the *same* weights. Drastically reduces parameter count (but not inference time).

---

## 🏆 DeBERTa (Decoding-enhanced BERT) - 2020

**Key Innovation:** Disentangled Attention.
Standard attention adds content and position vectors ($H + P$).
DeBERTa computes attention scores separately:
1.  Content-to-Content
2.  Content-to-Position
3.  Position-to-Content

**Result:** State-of-the-art on NLU benchmarks (SuperGLUE).

---

## 🎓 Interview Focus

1.  **Why remove NSP in RoBERTa?**
    - It was found that training on contiguous full sentences (longer context) was more important than the artificial NSP task.

2.  **Does ALBERT speed up inference?**
    - **No.** It reduces *parameters* (memory footprint) by sharing weights, but the number of *computations* (layers) remains the same.

3.  **DistilBERT vs BERT performance gap?**
    - Usually <3% drop in accuracy for a 2x speedup. Great for production.

---

**BERT Family: The evolution of understanding!**
