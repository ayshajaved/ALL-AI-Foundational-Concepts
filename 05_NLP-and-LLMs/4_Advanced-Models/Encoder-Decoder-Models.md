# Encoder-Decoder Models

> **The best of both worlds** - T5 and BART for Seq2Seq

---

## 🎯 Sequence-to-Sequence (Seq2Seq)

Tasks where Input and Output are both sequences, but of different lengths/structures.
- Translation
- Summarization
- Paraphrasing

---

## 🏗️ T5 (Text-to-Text Transfer Transformer) - 2020

**Philosophy:** "Treat every NLP task as a text-to-text problem."

- **Classification:** Input: "sentiment: I love this" $\to$ Output: "positive"
- **Translation:** Input: "translate English to German: Hello" $\to$ Output: "Hallo"
- **Regression:** Input: "stsb sentence1: ... sentence2: ..." $\to$ Output: "3.8"

**Pre-training Objective:** **Span Corruption**.
Input: "The cute dog <X> the ball."
Target: "<X> fetched"

---

## 🦇 BART (Bidirectional and Auto-Regressive Transformers) - 2019

**Architecture:** BERT Encoder + GPT Decoder.
1.  **Encoder:** Corrupts input (masking, shuffling, deleting). Processes full context bi-directionally.
2.  **Decoder:** Auto-regressively reconstructs the original text.

**Use Case:** Excellent for **Summarization**. The encoder understands the document, the decoder writes the summary.

---

## 💻 HuggingFace Implementation

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_text = "translate English to French: The cat is on the table."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# "Le chat est sur la table."
```

---

## 🎓 Interview Focus

1.  **T5 vs BERT?**
    - BERT outputs embeddings/classes. T5 outputs *text*.
    - T5 is an Encoder-Decoder; BERT is just an Encoder.

2.  **Why is BART good for summarization?**
    - It combines the understanding power of bidirectional encoders with the generation capability of autoregressive decoders.

3.  **What is Span Corruption?**
    - Masking contiguous spans of tokens (e.g., "New York City" $\to$ `<Mask>`) rather than individual tokens. Forces model to predict multiple words.

---

**Encoder-Decoder: The universal translators!**
