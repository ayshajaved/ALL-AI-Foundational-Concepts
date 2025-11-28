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

## 📊 Gradient Flow Visualization

### Monitoring Gradients

```python
import matplotlib.pyplot as plt

def plot_grad_flow(named_parameters):
    """Visualize gradient flow through network"""
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)],
               ['max-gradient', 'mean-gradient'])
    plt.tight_layout()
    plt.show()

# Usage during training
model = MLP(784, [128, 64], 10)
optimizer = optim.Adam(model.parameters())

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Plot gradient flow
        if epoch == 0:  # First epoch
            plot_grad_flow(model.named_parameters())
        
        optimizer.step()
```

### Vanishing Gradient Example

```python
# Demonstrate vanishing gradients with deep sigmoid network
class DeepSigmoidNet(nn.Module):
    def __init__(self, depth=10):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.Sigmoid())  # Sigmoid causes vanishing gradients
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Train and observe gradient norms
deep_model = DeepSigmoidNet(depth=10)
x = torch.randn(32, 100)
y = torch.randn(32, 100)

output = deep_model(x)
loss = nn.MSELoss()(output, y)
loss.backward()

# Check gradient norms per layer
print("Gradient norms (vanishing with depth):")
for i, (name, param) in enumerate(deep_model.named_parameters()):
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"Layer {i}: {grad_norm:.6f}")
```

### Gradient Clipping

```python
# Prevent exploding gradients
max_grad_norm = 1.0

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
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
