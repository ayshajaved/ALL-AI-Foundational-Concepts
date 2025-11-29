# NLP Interview Prep: Top 50 Q&A

> **Mastering the Interview** - From Basics to LLM Engineering

---

## 🟢 Beginner (Concepts)

1.  **What is Tokenization?** Splitting text into units. Subword tokenization (BPE) solves OOV.
2.  **Stemming vs Lemmatization?** Stemming chops suffixes (fast); Lemmatization finds root (accurate).
3.  **What is TF-IDF?** Term Frequency - Inverse Document Frequency. Highlights rare, important words.
4.  **Word2Vec vs GloVe?** Predictive (Neural) vs Count-based (Matrix Factorization).
5.  **What is the Vanishing Gradient problem in RNNs?** Gradients shrink through time steps. LSTMs solve this with Gating.

---

## 🟡 Intermediate (Deep Learning)

6.  **Explain Self-Attention.** $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$.
7.  **Why Multi-Head Attention?** Allows model to focus on different positions/subspaces simultaneously.
8.  **BERT vs GPT?** Encoder (Bidirectional, NLU) vs Decoder (Unidirectional, Generation).
9.  **What is Positional Encoding?** Injecting order information since Attention is permutation invariant.
10. **How does Beam Search work?** Keeps Top-K sequences at each step to find a better global output.

---

## 🔴 Advanced (LLMs & Engineering)

11. **Explain FlashAttention.** IO-aware attention. Uses tiling to keep ops in SRAM, avoiding HBM access.
12. **What is KV Cache?** Caching Key/Value matrices of past tokens to speed up autoregressive generation ($O(N)$ vs $O(N^2)$).
13. **How does LoRA work?** Decomposes weight updates into low-rank matrices ($W + BA$).
14. **What is PPO in RLHF?** Proximal Policy Optimization. Updates policy while preventing drastic deviations from the old policy.
15. **What is RAG?** Retrieving context from a Vector DB to ground LLM generation.
16. **Explain RoPE.** Rotary Positional Embeddings. Encodes relative position by rotating vectors.
17. **What is the "Reversal Curse"?** LLMs trained on "A is B" often fail to answer "What is B?".
18. **How to handle Context Length limits?** Sliding window, summarization, or RAG.
19. **What is vLLM?** High-throughput serving engine using PagedAttention.
20. **Difference between FP16 and BF16?** BF16 has the same dynamic range as FP32 (more exponent bits), preventing underflow/overflow better than FP16.

---

## 🧠 System Design Scenarios

**Q: Design a News Summarization System.**
- **Data:** Web scraper $\to$ Kafka.
- **Model:** Fine-tuned BART/T5 or GPT-3.5 API.
- **Serving:** FastAPI + Redis (Cache).
- **Eval:** ROUGE score + Human feedback.

**Q: Design a Semantic Search Engine.**
- **Ingestion:** Chunking $\to$ Embedding (MiniLM) $\to$ Vector DB (Pinecone).
- **Query:** Embed Query $\to$ Cosine Similarity.
- **Optimization:** HNSW Index for speed. Hybrid search (Keyword + Vector).

---

**You are ready. Go get that job!**
