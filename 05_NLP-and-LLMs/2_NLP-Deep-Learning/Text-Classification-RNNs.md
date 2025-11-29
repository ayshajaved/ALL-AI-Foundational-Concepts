# Text Classification with RNNs

> **Sentiment Analysis & Intent Detection** - Using LSTMs for classification

---

## 🎯 The Architecture

**Input:** Sequence of word indices.
**Embedding Layer:** Converts indices to dense vectors.
**RNN Layer (LSTM/GRU):** Processes sequence, updates hidden state.
**Classification Head:** Takes the *final* hidden state $h_T$ and projects to classes.

$$ x \to \text{Embedding} \to \text{LSTM} \to h_T \to \text{Linear} \to \text{Softmax} $$

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. LSTM
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=True, 
                            dropout=dropout,
                            batch_first=True)
        
        # 3. Output (Hidden dim * 2 because bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch, seq_len]
        embedded = self.dropout(self.embedding(text))
        
        # output: [batch, seq, hid_dim*2]
        # hidden: [layers*2, batch, hid_dim]
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concat the final forward and backward hidden states
        # hidden[-2] is forward, hidden[-1] is backward
        hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        return self.fc(hidden_final)
```

---

## 🧩 Handling Variable Lengths (Pack Padded Sequence)

RNNs waste compute on padding tokens (`<PAD>`).
**Solution:** `pack_padded_sequence`. Tells PyTorch to ignore padding.

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# In forward pass:
# lengths = [len(x) for x in text]
packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
packed_output, (hidden, cell) = self.lstm(packed_embedded)
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
```

---

## 🎓 Interview Focus

1.  **Why use Bidirectional LSTM?**
    - Context comes from both past and future. "The movie was **not** good" needs "not" to understand "good".

2.  **Why use the final hidden state?**
    - It theoretically contains the compressed information of the entire sequence.
    - *Advanced:* Attention mechanisms are better than just taking the final state.

3.  **What is the bottleneck?**
    - Sequential processing. Cannot parallelize. Slow on long sequences.

---

**RNNs: The classic baseline for text classification!**
