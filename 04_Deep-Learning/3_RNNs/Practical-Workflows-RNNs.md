# Practical Workflows - RNNs

> **Real-world sequence modeling** - Time series and Text generation

---

## 📈 Project 1: Time Series Forecasting (Stock Price)

Predicting the next value in a sequence.

### 1. Data Preparation (Sliding Window)
Convert time series `[1, 2, 3, 4, 5]` into supervised pairs:
- X: `[1, 2, 3]`, y: `4`
- X: `[2, 3, 4]`, y: `5`

```python
import torch
import numpy as np

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Dummy data (Sine wave)
data = np.sin(np.linspace(0, 100, 1000))
X, y = create_sequences(data, seq_length=10)

# Convert to Tensor (Batch, Seq, Feature)
X_tensor = torch.FloatTensor(X).unsqueeze(2) # (990, 10, 1)
y_tensor = torch.FloatTensor(y).unsqueeze(1) # (990, 1)
```

### 2. LSTM Regressor Model

```python
class StockPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.linear = nn.Linear(50, 1)
        
    def forward(self, x):
        # x: (batch, seq, 1)
        out, _ = self.lstm(x)
        # Take last time step output
        last_out = out[:, -1, :]
        pred = self.linear(last_out)
        return pred

model = StockPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

---

## 📝 Project 2: Character-Level Text Generation

Generating text character by character (e.g., Shakespeare).

### 1. Preprocessing
- Create vocabulary (unique chars).
- Map char to int and int to char.
- One-hot encode inputs (optional, or use Embeddings).

### 2. Model with Embeddings

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden):
        # x: (batch, seq) -> (batch, seq, hidden)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        # Reshape for FC
        output = output.contiguous().view(-1, self.rnn.hidden_size)
        output = self.fc(output)
        return output, hidden
```

### 3. Generation Loop
1. Feed seed character.
2. Get probability distribution for next char.
3. Sample next char (using temperature).
4. Feed sampled char as next input.

```python
def generate(model, start_str='A', len=100):
    input = char_to_tensor(start_str)
    hidden = None
    
    for i in range(len):
        output, hidden = model(input, hidden)
        
        # Sampling logic here...
        # top_k or multinomial sampling
```

---

## 🎯 Best Practices

1.  **Scale Data:** For regression (e.g., stock prices), always normalize data (MinMax or StandardScaler) to [0, 1] or [-1, 1]. Tanh expects this range.
2.  **Batch First:** Use `batch_first=True` in PyTorch RNNs for intuitive `(batch, seq, feature)` dimensions.
3.  **Gradient Clipping:** Always use it with RNNs.
4.  **Dropout:** Use `dropout` argument in LSTM/GRU constructors for regularization between layers.

---

**From theory to practice: Building working RNN systems!**
