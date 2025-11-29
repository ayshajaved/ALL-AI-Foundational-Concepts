# Text Classification with CNNs

> **Fast and effective** - 1D Convolutions for text

---

## 🎯 Why CNNs for Text?

RNNs are slow (sequential).
CNNs are fast (parallel).
**Intuition:** A convolution filter of size 3 acts like a **3-gram detector**. It scans for specific phrases ("not good", "very happy").

---

## 🏗️ The Architecture (KimCNN)

1.  **Input:** Sentence matrix (Seq Len $\times$ Embed Dim).
2.  **Filters:** Multiple kernel sizes (e.g., 2, 3, 4) to capture bi-grams, tri-grams, 4-grams.
3.  **Pooling:** **Max-Over-Time Pooling**. Take the maximum activation from each filter map. (Did this phrase appear *anywhere* in the sentence?).
4.  **Concat:** Combine pooled features.
5.  **Output:** Linear layer.

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # ModuleList of Convolutions
        # Input channels = 1 (treat text like a 1-channel image)
        # Output channels = n_filters
        # Kernel size = (filter_size, embed_dim) -> spans full embedding depth
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=n_filters, 
                      kernel_size=(fs, embed_dim)) 
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch, seq_len]
        embedded = self.embedding(text) # [batch, seq_len, embed_dim]
        
        # Add channel dim: [batch, 1, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)
        
        # Apply convs + ReLU
        # conv(embedded): [batch, n_filters, seq_len-filter_size+1, 1]
        # squeeze(3): [batch, n_filters, seq_len-filter_size+1]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        # Max Pooling over time
        # max_pool1d: [batch, n_filters, 1] -> squeeze -> [batch, n_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concat
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        return self.fc(cat)

# Usage
model = TextCNN(vocab_size=10000, embed_dim=100, n_filters=100, 
                filter_sizes=[3, 4, 5], output_dim=2, dropout=0.5)
```

---

## 🎓 Interview Focus

1.  **CNN vs RNN for Text?**
    - **CNN:** Faster, good at detecting key phrases (keyword spotting).
    - **RNN:** Better at long-range dependencies and complex syntax.

2.  **What is Max-Over-Time Pooling?**
    - Taking the single highest value from the feature map. It tells us *if* a feature (n-gram) was present, but discards *where* it was.

3.  **Why kernel width = embedding dimension?**
    - We want to slide over *words*, not parts of the embedding vector. The filter covers the entire vector of a word (or multiple words).

---

**TextCNN: The speed demon of NLP baselines!**
