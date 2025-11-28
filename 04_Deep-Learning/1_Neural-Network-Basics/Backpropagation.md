# Backpropagation

> **The algorithm that trains neural networks** - Computing gradients efficiently

---

## 🎯 Chain Rule Review

```
∂f/∂x = (∂f/∂y)(∂y/∂x)

For composition: f(g(x))
```

---

## 📊 Computational Graphs

### Forward Pass
```
x → w₁ → z₁ → a₁ → w₂ → z₂ → a₂ → L

z: linear combination
a: activation
L: loss
```

### Backward Pass
```
∂L/∂w₂ ← ∂L/∂z₂ ← ∂L/∂a₂ ← ∂L/∂z₁ ← ∂L/∂a₁ ← ∂L/∂w₁
```

---

## 🎯 Backpropagation Algorithm

### Example: 2-Layer Network

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Forward pass
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(2, 4)  # Input to hidden
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)  # Hidden to output
b2 = np.zeros((1, 1))

learning_rate = 0.1

for epoch in range(10000):
    # Forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    
    # Loss
    loss = np.mean((y - a2)**2)
    
    # Backward
    # Output layer
    dL_da2 = -2 * (y - a2) / len(X)
    da2_dz2 = sigmoid_derivative(a2)
    dL_dz2 = dL_da2 * da2_dz2
    
    dL_dW2 = a1.T @ dL_dz2
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
    
    # Hidden layer
    dL_da1 = dL_dz2 @ W2.T
    da1_dz1 = sigmoid_derivative(a1)
    dL_dz1 = dL_da1 * da1_dz1
    
    dL_dW1 = X.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
    
    # Update
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test
predictions = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
print(f"\nPredictions:\n{predictions}")
```

---

## 📈 PyTorch Autograd

```python
import torch

# Automatic differentiation
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

# Backward
y.backward()

print(f"dy/dx = {x.grad}")  # 2*x + 3 = 7
```

---

## 🎓 Interview Focus

1. **How does backpropagation work?**
   - Forward: compute outputs
   - Backward: compute gradients via chain rule
   - Update: adjust weights

2. **Why is it efficient?**
   - Reuses computations
   - O(n) for n parameters
   - Avoids numerical differentiation

3. **Vanishing gradient problem?**
   - Gradients become very small
   - Deep networks suffer
   - Solutions: ReLU, skip connections

---

**Backpropagation: the engine of deep learning!**
