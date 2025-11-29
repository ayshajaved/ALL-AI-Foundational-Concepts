# Tokenization Fundamentals

> **From strings to integers** - BPE, WordPiece, and SentencePiece

---

## 🎯 The OOV Problem

Traditional word-level tokenization fails on **Out-Of-Vocabulary (OOV)** words.
If "uninstagrammable" isn't in your dictionary, it becomes `<UNK>`.

**Solution:** Subword Tokenization. Break unknown words into known sub-parts.
`uninstagrammable` $\to$ `un`, `instagram`, `##able`.

---

## 🧩 Byte Pair Encoding (BPE)

**Used in:** GPT-2, GPT-3, Llama, RoBERTa.

**Algorithm:**
1.  Start with a vocabulary of characters.
2.  Find the most frequent *pair* of adjacent tokens in the corpus.
3.  Merge them into a new token.
4.  Repeat until vocabulary size is reached.

**Example:**
`h u g`, `p u g`, `p u n`, `b u n`
Most frequent pair: `u` + `g` $\to$ `ug`.
New tokens: `h ug`, `p ug`, `p u n`, `b u n`.

---

## 🧩 WordPiece

**Used in:** BERT, DistilBERT.

Similar to BPE, but selects merges based on **likelihood** (maximizing the probability of the training data) rather than just frequency.
Uses `##` prefix to indicate a subword is part of a previous word.

---

## 🧩 SentencePiece (Unigram)

**Used in:** T5, ALBERT, XLNet.

Treats the input as a raw stream of unicode characters (including spaces). No pre-tokenization (splitting by space) required.
Language agnostic (works great for Japanese/Chinese where spaces don't exist).

---

## 💻 HuggingFace Tokenizers

The industry standard library (written in Rust, incredibly fast).

```python
from transformers import AutoTokenizer

# Load a pre-trained tokenizer (BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Tokenization is fascinating!"

# 1. Tokenize
tokens = tokenizer.tokenize(text)
print(tokens)
# ['token', '##ization', 'is', 'fascinating', '!']

# 2. Convert to IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# [19204, 3989, 2003, 19456, 999]

# 3. Decode back
decoded = tokenizer.decode(ids)
print(decoded)
# "tokenization is fascinating !"
```

---

## 🎓 Interview Focus

1.  **Why Subword Tokenization?**
    - Solves the OOV problem.
    - Reduces vocabulary size (efficient embedding matrix).
    - Captures morphological meaning (`play`, `play##ing`, `play##ed`).

2.  **BPE vs WordPiece?**
    - BPE merges based on frequency.
    - WordPiece merges based on likelihood improvement.

3.  **What is the vocabulary size trade-off?**
    - **Small Vocab:** Longer sequences (more tokens per sentence), slower inference.
    - **Large Vocab:** Shorter sequences, but huge embedding matrix (memory intensive).

---

**Tokenization: The bridge between human language and machine math!**
