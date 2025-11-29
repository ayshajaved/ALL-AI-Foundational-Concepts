# NLP Evaluation Metrics

> **Measuring Success** - BLEU, ROUGE, Perplexity, and LLM-as-a-Judge

---

## 📏 Traditional Metrics

### 1. Perplexity (PPL)
Standard for **Language Modeling**.
Measures how "surprised" the model is by the next token.
Lower is better.
$$ PPL(X) = \exp \left( -\frac{1}{t} \sum_{i=1}^t \log P(w_i | w_{<i}) \right) $$

### 2. BLEU (Bilingual Evaluation Understudy)
Standard for **Translation**.
Precision-based overlap of n-grams.
- **Problem:** "The cat is on the mat" vs "There is a cat on the mat". Low BLEU, but same meaning.

### 3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Standard for **Summarization**.
Recall-based overlap.

---

## 🤖 Modern Metrics (LLM Era)

Traditional metrics fail for open-ended generation (Chatbots, Creative Writing).
"Write a poem about AI" has infinite valid answers.

### 1. BERTScore
Uses **Contextual Embeddings** (BERT) to compute similarity, not just exact word matching.
Matches "automobile" with "car".

### 2. LLM-as-a-Judge (G-Eval / Prometheus)
Ask a stronger LLM (GPT-4) to grade the output of a weaker LLM.

**Prompt:**
> "You are an expert evaluator. Grade the following response on a scale of 1-5 for Helpfulness and Accuracy.
> User Query: ...
> Model Response: ...
> Explanation: ...
> Score: ..."

**Pros:** Correlates highly with human judgment.
**Cons:** Slow, expensive, bias towards the judge's own style.

---

## 🧪 RAG Evaluation (RAGAS)

Specific metrics for RAG pipelines:
1.  **Faithfulness:** Is the answer supported by the retrieved context?
2.  **Answer Relevance:** Does the answer address the user's question?
3.  **Context Precision:** Did the retriever find relevant documents?

---

## 🎓 Interview Focus

1.  **Why is Perplexity not enough?**
    - A model can have low perplexity (predicts grammar well) but still hallucinate facts or be toxic.

2.  **What is the "N-gram overlap" limitation?**
    - It penalizes synonyms and paraphrasing. "The movie was great" vs "The film was excellent" has 0 overlap but identical meaning.

3.  **How to validate LLM-as-a-Judge?**
    - Compare its scores against a small set of human-labeled examples (Golden Set) to calculate correlation.

---

**Evaluation: If you can't measure it, you can't improve it!**
