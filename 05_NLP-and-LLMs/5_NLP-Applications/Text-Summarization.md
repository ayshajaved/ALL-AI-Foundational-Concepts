# Text Summarization

> **Compressing information** - Extractive vs Abstractive

---

## 🎯 The Task

Produce a shorter version of a document that retains the key information.

### 1. Extractive Summarization
Selects key sentences from the original text.
- **Pros:** Grammatically correct, factual.
- **Cons:** Can be disjointed, limited flexibility.
- **Algorithm:** TextRank (Graph-based), BERT-Extractive.

### 2. Abstractive Summarization
Generates new sentences (like a human).
- **Pros:** Fluent, concise.
- **Cons:** Hallucinations (making things up).
- **Models:** BART, T5, Pegasus.

---

## 🦇 BART & Pegasus

- **BART:** Denoising autoencoder. Great for general summarization (CNN/DailyMail).
- **Pegasus:** Pre-trained specifically for summarization. Objective: **Gap Sentence Generation** (masking entire important sentences and generating them).

---

## 💻 HuggingFace Implementation (Abstractive)

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
The Apollo program was the third United States human spaceflight program 
carried out by the National Aeronautics and Space Administration (NASA), 
which accomplished landing the first humans on the Moon from 1969 to 1972.
... (long text) ...
"""

summary = summarizer(article, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
```

---

## 📉 Evaluation: ROUGE

**R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation.
- **ROUGE-N:** Overlap of n-grams (unigrams, bigrams).
- **ROUGE-L:** Longest Common Subsequence.

Unlike BLEU (Precision-focused), ROUGE focuses on **Recall** (did we cover all the important points?).

---

## 🎓 Interview Focus

1.  **Extractive vs Abstractive?**
    - Extractive = Highlighting. Abstractive = Rewriting.

2.  **Why is ROUGE used for summarization instead of BLEU?**
    - We care more about *recall* (how much of the reference summary did we capture?) than precision.

3.  **What is the "Lead-3" baseline?**
    - A strong baseline that simply takes the first 3 sentences of the article. News articles are written in "inverted pyramid" style, so Lead-3 is surprisingly hard to beat.

---

**Summarization: TL;DR for machines!**
