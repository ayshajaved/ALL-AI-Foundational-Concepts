# Contextual Embeddings

> **Context matters** - Why "bank" isn't always a "bank"

---

## 🎯 The Problem with Static Embeddings

In Word2Vec/GloVe:
$$ Vector(\text{"Apple"}) $$
Is the same for:
1.  "I ate an **Apple**." (Fruit)
2.  "I bought **Apple** stock." (Company)

This conflation limits performance on complex tasks.

---

## 🧠 ELMo (Embeddings from Language Models) - 2018

**Idea:** Don't just look up a fixed vector. Pass the entire sentence through a **Bi-directional LSTM**.
The embedding for "Apple" is the hidden state of the LSTM at that position.

$$ E(x_k) = \gamma \sum_{j=0}^L s_j h_{k,j}^{LM} $$

- **Result:** Distinct vectors for "Apple" (fruit) and "Apple" (company).
- **Architecture:** 2-layer BiLSTM trained on Language Modeling (predict next word).

---

## 🚀 CoVe (Context Vectors)

Used a Machine Translation LSTM encoder to generate embeddings.
Less popular than ELMo but pioneered the idea of transfer learning from other tasks.

---

## 👑 The Transformer Revolution (BERT)

ELMo used LSTMs (sequential, slow). BERT used **Transformers** (parallel, attention).

**BERT Embeddings:**
The output of the last hidden layer of BERT *is* the contextual embedding.
Actually, usually the sum or concatenation of the last 4 layers works best.

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "I went to the bank to catch a fish."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
# Last hidden state: (Batch, Seq_Len, Hidden_Dim)
# (1, 10, 768)
embeddings = outputs.last_hidden_state

# The vector for "bank" (token index 5) contains context about "fish"!
bank_vector = embeddings[0, 5, :]
```

---

## 🎓 Interview Focus

1.  **Static vs Contextual Embeddings?**
    - **Static (Word2Vec):** 1 vector per word type. Fast, low memory.
    - **Contextual (BERT):** 1 vector per word token instance. Accurate, high compute.

2.  **How does ELMo differ from BERT?**
    - ELMo uses LSTMs (recurrence). BERT uses Transformers (attention).
    - BERT is deeply bidirectional; ELMo concatenates forward and backward LSTMs.

3.  **Why did ELMo fail to persist?**
    - LSTMs are slow to train and cannot handle long-range dependencies as well as Self-Attention.

---

**Context: The key to understanding nuance!**
