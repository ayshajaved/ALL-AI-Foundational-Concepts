# Practical Workflows: Deep NLP

> **Training a Sentiment Classifier** - End-to-End PyTorch Workflow

---

## 🛠️ The Project: IMDb Movie Review Classification

**Goal:** Classify reviews as Positive (1) or Negative (0).
**Dataset:** 25k Train, 25k Test.

---

## 🚀 Step-by-Step Implementation

### 1. Data Loading (TorchText / HuggingFace)
We'll use a simple custom setup for clarity.

```python
# Assume we have a TextPipeline and NLPDataset from previous section
train_dataset = NLPDataset(train_texts, train_labels, pipeline)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
```

### 2. Model Definition
```python
model = SentimentRNN(vocab_size=len(pipeline.vocab), 
                     embed_dim=100, 
                     hidden_dim=256, 
                     output_dim=2, 
                     n_layers=2, 
                     dropout=0.5).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

### 3. Training Loop
```python
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for text, labels in iterator:
        text, labels = text.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        # Calculate accuracy...
        
    return epoch_loss / len(iterator)
```

### 4. Evaluation
```python
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for text, labels in iterator:
            text, labels = text.to(device), labels.to(device)
            predictions = model(text)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)
```

### 5. Inference
```python
def predict_sentiment(sentence):
    model.eval()
    tokenized = pipeline.transform(sentence)
    tensor = torch.LongTensor(tokenized).unsqueeze(0).to(device)
    prediction = model(tensor)
    return F.softmax(prediction, dim=1)
```

---

## 🧠 Tips for Better Performance

1.  **Pre-trained Embeddings:** Initialize the embedding layer with GloVe vectors instead of random noise.
    ```python
    model.embedding.weight.data.copy_(glove_vectors)
    ```
2.  **Gradient Clipping:** Prevent exploding gradients in RNNs.
    ```python
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    ```
3.  **Packed Sequences:** Use `pack_padded_sequence` for 2x speedup.

---

**You built a Deep NLP Model from scratch!**
