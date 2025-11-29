# RAG Fundamentals (Retrieval Augmented Generation)

> **Giving LLMs a memory** - Vector DBs and Semantic Search

---

## 🎯 The Problem: Hallucinations & Cutoff Dates

LLMs have:
1.  **Frozen Knowledge:** Cutoff date (e.g., 2023).
2.  **Hallucinations:** Confidently wrong about facts.

**Solution:** **RAG**. Retrieve relevant facts from a trusted database and feed them to the LLM.

---

## 🏗️ The Pipeline

1.  **Ingestion:**
    - Chunk documents (e.g., 500 tokens).
    - Embed chunks using an Embedding Model (OpenAI `text-embedding-3`, HuggingFace `all-MiniLM-L6-v2`).
    - Store vectors in a **Vector Database** (Pinecone, Chroma, FAISS).

2.  **Retrieval:**
    - User Query: "How do I reset my router?"
    - Embed Query.
    - Perform **Cosine Similarity Search** in Vector DB.
    - Get Top-K chunks.

3.  **Generation:**
    - Prompt:
      ```
      Context: [Chunk 1, Chunk 2, Chunk 3]
      Question: How do I reset my router?
      Answer based ONLY on the context above:
      ```

---

## 🧪 Advanced RAG

1.  **Hybrid Search:** Combine Keyword Search (BM25) with Semantic Search (Vectors). Good for exact matches (product IDs).
2.  **Re-ranking:** Retrieve Top-50 chunks, then use a powerful Cross-Encoder (Re-ranker) to sort them and pick Top-5.
3.  **RAGAS (Evaluation):**
    - **Faithfulness:** Is the answer derived from context?
    - **Answer Relevance:** Does the answer address the query?
    - **Context Precision:** Is the retrieved context actually relevant?

---

## 🎓 Interview Focus

1.  **Why use a Vector DB?**
    - Standard SQL databases are slow for high-dimensional similarity search. Vector DBs use HNSW (Hierarchical Navigable Small World) graphs for approximate nearest neighbor search (ms latency).

2.  **What is the "Lost in the Middle" phenomenon?**
    - LLMs tend to focus on the beginning and end of the context window. If the answer is buried in the middle of 10 retrieved chunks, the LLM might miss it.

3.  **Semantic Search vs Keyword Search?**
    - **Keyword:** Matches "Apple" to "Apple".
    - **Semantic:** Matches "Apple" to "iPhone" or "Fruit".

---

**RAG: The bridge between enterprise data and AI!**
