# Sequence Labeling (NER/POS)

> **Tagging every token** - Named Entity Recognition and POS Tagging

---

## 🎯 The Task

Input: `["Apple", "is", "in", "California"]`
Output: `[B-ORG, O, O, B-LOC]`

**Challenges:**
1.  **Dependencies:** Tags depend on neighbors. (A `I-ORG` must follow `B-ORG`).
2.  **Context:** "Apple" is ORG here, but FRUIT in "Apple pie".

---

## 🏗️ BiLSTM-CRF (The Gold Standard pre-BERT)

1.  **BiLSTM:** Generates context-aware features for each token.
    - Output: $P(Tag_i | Word_i, Context)$
2.  **CRF (Conditional Random Field):** Models the *transition probabilities* between tags.
    - Learns that `I-ORG` cannot follow `B-LOC`.

$$ Score(y, x) = \sum_{i} \text{Emission}(x_i, y_i) + \sum_{i} \text{Transition}(y_{i-1}, y_i) $$

---

## 💻 PyTorch Implementation (BiLSTM only)

Implementing CRF from scratch is complex (Viterbi algorithm). We'll focus on the BiLSTM part, which is 95% of the work.

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch, seq_len]
        embedded = self.dropout(self.embedding(text))
        
        # outputs: [batch, seq_len, hid_dim * 2]
        outputs, _ = self.lstm(embedded)
        
        # predictions: [batch, seq_len, output_dim]
        predictions = self.fc(outputs)
        return predictions

# Loss Function
# We must ignore the padding token in the loss!
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

---

## 🏷️ IOB Format

Standard tagging scheme:
- **B-XXX:** Beginning of entity XXX.
- **I-XXX:** Inside of entity XXX.
- **O:** Outside (not an entity).

Example:
`New (B-LOC) York (I-LOC) City (I-LOC)`

---

## 🎓 Interview Focus

1.  **Why do we need CRF on top of LSTM?**
    - LSTM makes independent decisions for each token. It might predict `B-ORG` followed by `I-LOC` (invalid).
    - CRF enforces valid transition rules (global consistency).

2.  **CrossEntropyLoss `ignore_index`?**
    - Crucial. We pad sentences to the same length, but we shouldn't calculate loss or update gradients based on predicting tags for `<PAD>`.

---

**Sequence Labeling: Understanding the structure of text!**
