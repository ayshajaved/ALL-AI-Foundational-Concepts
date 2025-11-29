# Practical Workflows: Text Pipeline

> **Building a robust preprocessing pipeline** - From raw text to DataLoader

---

## 🛠️ The Goal

Build a reusable pipeline that handles:
1.  Cleaning
2.  Tokenization
3.  Vocabulary Building
4.  Padding/Truncating
5.  PyTorch Dataset/DataLoader

---

## 💻 Implementation

```python
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

class TextPipeline:
    def __init__(self, max_vocab=10000, max_len=50):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.unk_token = 1
        
    def fit(self, texts):
        """Build vocabulary from list of texts"""
        counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)
            
        # Keep top N words
        most_common = counter.most_common(self.max_vocab - 2)
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)
            
    def _tokenize(self, text):
        """Simple regex tokenizer"""
        text = text.lower()
        return re.findall(r'\w+', text)
    
    def transform(self, text):
        """Convert text to list of IDs"""
        tokens = self._tokenize(text)
        ids = [self.vocab.get(t, self.unk_token) for t in tokens]
        
        # Truncate
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            
        # Pad
        padding = [0] * (self.max_len - len(ids))
        return ids + padding

# Custom Dataset
class NLPDataset(Dataset):
    def __init__(self, texts, labels, pipeline):
        self.texts = texts
        self.labels = labels
        self.pipeline = pipeline
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        input_ids = torch.tensor(self.pipeline.transform(text), dtype=torch.long)
        return input_ids, torch.tensor(label, dtype=torch.long)

# Usage
raw_texts = ["I love AI", "Deep learning is hard but fun", "NLP is cool"]
labels = [1, 1, 1]

# 1. Build Pipeline
pipe = TextPipeline(max_vocab=20, max_len=6)
pipe.fit(raw_texts)
print(f"Vocab: {pipe.vocab}")

# 2. Create DataLoader
dataset = NLPDataset(raw_texts, labels, pipe)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. Iterate
for batch_x, batch_y in loader:
    print(f"Batch shape: {batch_x.shape}") # (2, 6)
    break
```

---

## 🧩 Handling Variable Lengths (Collate Fn)

Padding to the *maximum length in the batch* (Dynamic Padding) is more efficient than padding to a fixed global max length.

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, labels = zip(*batch)
    # Pad to max length in THIS batch
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    return padded_inputs, torch.stack(labels)

# Use in DataLoader
# loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

---

**Pipelines: The infrastructure of success!**
