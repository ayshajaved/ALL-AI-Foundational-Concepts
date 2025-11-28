# Perceptron and Multi-Layer Perceptron

> **The foundation of deep learning** - From single neurons to universal function approximators

---

## 🎯 The Perceptron

### Single Neuron
```
y = f(wᵀx + b)

w: weights
b: bias
f: activation function
```

### Implementation

```python
import numpy as np

class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01):
        self.w = np.zeros(input_dim)
        self.b = 0
        self.lr = learning_rate
    
    def predict(self, x):
        return 1 if (self.w @ x + self.b) > 0 else 0
    
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                # Update rule
                self.w += self.lr * (yi - y_pred) * xi
                self.b += self.lr * (yi - y_pred)

# Example
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate

perceptron = Perceptron(input_dim=2)
perceptron.train(X, y)

for xi in X:
    print(f"Input: {xi}, Output: {perceptron.predict(xi)}")
```

---

## 📊 Multi-Layer Perceptron (MLP)

### Architecture
```
Input Layer → Hidden Layer(s) → Output Layer

Each layer: z = Wx + b, a = f(z)
```

### Forward Propagation

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example
model = MLP(input_dim=784, hidden_dims=[128, 64], output_dim=10)
x = torch.randn(32, 784)  # Batch of 32 samples
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 10)
```

---

## 🎯 Universal Approximation Theorem

**Theorem:** A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of ℝⁿ.

**Implications:**
- MLPs are universal function approximators
- Depth vs width tradeoff
- Practical considerations (training, generalization)

---

## 📈 Training MLP

```python
# Complete training example
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = MLP(784, [128, 64], 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        # Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is a perceptron?**
   - Single neuron with linear combination + activation
   - Can learn linearly separable functions
   - Foundation of neural networks

2. **Perceptron limitations?**
   - Can't learn XOR (not linearly separable)
   - Solved by MLP with hidden layers

3. **Universal approximation theorem?**
   - Single hidden layer can approximate any function
   - Doesn't guarantee efficient learning
   - Depth helps in practice

4. **Why hidden layers?**
   - Learn hierarchical features
   - Increase model capacity
   - Enable non-linear decision boundaries

---

## 📚 References

- **Papers:** "Learning representations by back-propagating errors" - Rumelhart et al.

---

**MLPs: the building blocks of deep learning!**
