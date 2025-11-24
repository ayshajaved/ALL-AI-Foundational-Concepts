# Chain Rule and Backpropagation

> **The heart of deep learning** - How gradients flow through neural networks

---

## 🔗 Chain Rule

### Single Variable

For composite functions f(g(x)):

```
d/dx[f(g(x))] = f'(g(x)) · g'(x)
```

**Example:**
```
h(x) = (x² + 1)³

Let u = x² + 1, then h = u³

dh/dx = dh/du · du/dx
      = 3u² · 2x
      = 3(x² + 1)² · 2x
      = 6x(x² + 1)²
```

### Multivariable Chain Rule

For z = f(x,y) where x = g(t), y = h(t):

```
dz/dt = ∂z/∂x · dx/dt + ∂z/∂y · dy/dt
```

**General Form:**
```
∂z/∂t = Σᵢ (∂z/∂xᵢ · ∂xᵢ/∂t)
```

---

## 🧠 Computational Graphs

### Forward Pass

Represent computation as directed acyclic graph (DAG):

```
Input → Operation → Operation → ... → Output
  x   →    f₁     →    f₂     → ... →   y
```

**Example:** y = (x² + 1)³

```
x → [square] → x² → [+1] → x²+1 → [cube] → y
```

### Backward Pass

Compute gradients by traversing graph backwards:

```
∂L/∂x = ∂L/∂y · ∂y/∂(x²+1) · ∂(x²+1)/∂x²  · ∂x²/∂x
```

---

## 🎯 Backpropagation Algorithm

### Definition

**Backpropagation** = efficient algorithm to compute gradients using chain rule

**Key Idea:** 
- Forward pass: compute outputs
- Backward pass: compute gradients (reuse intermediate values)

### Algorithm

**Forward Pass:**
```
1. For each layer l = 1 to L:
   z^[l] = W^[l]a^[l-1] + b^[l]
   a^[l] = g^[l](z^[l])
```

**Backward Pass:**
```
1. Compute output gradient: dL/da^[L]
2. For each layer l = L to 1:
   dL/dz^[l] = dL/da^[l] · g'^[l](z^[l])
   dL/dW^[l] = dL/dz^[l] · (a^[l-1])^T
   dL/db^[l] = sum(dL/dz^[l])
   dL/da^[l-1] = (W^[l])^T · dL/dz^[l]
```

---

## 💻 Implementation

### Manual Backpropagation

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """Initialize network with given layer sizes"""
        self.L = len(layer_sizes) - 1  # Number of layers
        self.W = {}  # Weights
        self.b = {}  # Biases
        
        # Initialize parameters
        for l in range(1, self.L + 1):
            self.W[l] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * 0.01
            self.b[l] = np.zeros((layer_sizes[l], 1))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def forward(self, X):
        """Forward propagation"""
        self.cache = {'a0': X}
        a = X
        
        for l in range(1, self.L + 1):
            z = self.W[l] @ a + self.b[l]
            a = self.sigmoid(z)
            self.cache[f'z{l}'] = z
            self.cache[f'a{l}'] = a
        
        return a
    
    def backward(self, X, y):
        """Backpropagation"""
        m = X.shape[1]  # Number of samples
        grads = {}
        
        # Output layer gradient
        dL_da = self.cache[f'a{self.L}'] - y
        
        # Backward through layers
        for l in range(self.L, 0, -1):
            a_prev = self.cache[f'a{l-1}']
            a = self.cache[f'a{l}']
            
            # Gradient of loss w.r.t. z
            dL_dz = dL_da * self.sigmoid_derivative(a)
            
            # Gradients for parameters
            grads[f'dW{l}'] = (1/m) * (dL_dz @ a_prev.T)
            grads[f'db{l}'] = (1/m) * np.sum(dL_dz, axis=1, keepdims=True)
            
            # Gradient for previous layer
            if l > 1:
                dL_da = self.W[l].T @ dL_dz
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        """Update weights and biases"""
        for l in range(1, self.L + 1):
            self.W[l] -= learning_rate * grads[f'dW{l}']
            self.b[l] -= learning_rate * grads[f'db{l}']

# Example usage
nn = NeuralNetwork([2, 3, 1])  # 2 inputs, 3 hidden, 1 output
X = np.random.randn(2, 100)  # 100 samples
y = np.random.randint(0, 2, (1, 100))

# Training step
output = nn.forward(X)
grads = nn.backward(X, y)
nn.update_parameters(grads, learning_rate=0.01)
```

### Automatic Differentiation (PyTorch)

```python
import torch
import torch.nn as nn

# Define network
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.Sigmoid(),
    nn.Linear(3, 1),
    nn.Sigmoid()
)

# Forward pass
X = torch.randn(100, 2)
y = torch.randint(0, 2, (100, 1)).float()

output = model(X)
loss = nn.BCELoss()(output, y)

# Backward pass (automatic!)
loss.backward()

# Access gradients
for name, param in model.named_parameters():
    print(f"{name} gradient: {param.grad.shape}")

# Update parameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.step()
optimizer.zero_grad()
```

---

## 🔧 Common Activation Function Derivatives

### Sigmoid
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

### Tanh
```python
def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

### ReLU
```python
def relu_derivative(x):
    return (x > 0).astype(float)
```

### Leaky ReLU
```python
def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
```

### Softmax (with Cross-Entropy)
```python
def softmax_cross_entropy_derivative(logits, y_true):
    """Combined derivative of softmax + cross-entropy"""
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    probs[range(len(y_true)), y_true] -= 1
    return probs
```

---

## 🎯 Gradient Checking

### Numerical Gradient

```python
def numerical_gradient(f, x, h=1e-5):
    """Compute gradient numerically"""
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad

def gradient_check(f, grad_f, x, epsilon=1e-7):
    """Check if analytical gradient matches numerical gradient"""
    numerical_grad = numerical_gradient(f, x)
    analytical_grad = grad_f(x)
    
    diff = np.linalg.norm(numerical_grad - analytical_grad)
    norm = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
    relative_error = diff / norm
    
    if relative_error < epsilon:
        print(f"✓ Gradient check passed! (error: {relative_error:.2e})")
    else:
        print(f"✗ Gradient check failed! (error: {relative_error:.2e})")
        print(f"Numerical: {numerical_grad}")
        print(f"Analytical: {analytical_grad}")
    
    return relative_error < epsilon

# Example
f = lambda x: np.sum(x**2)
grad_f = lambda x: 2*x

x = np.array([1.0, 2.0, 3.0])
gradient_check(f, grad_f, x)
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is backpropagation?**
   - Algorithm to compute gradients efficiently
   - Uses chain rule
   - Forward pass + backward pass

2. **Why is chain rule important?**
   - Enables gradient computation through layers
   - Foundation of backpropagation
   - Allows training deep networks

3. **What is vanishing gradient problem?**
   - Gradients become very small in deep networks
   - Caused by repeated multiplication of small derivatives
   - Solutions: ReLU, batch norm, residual connections

4. **How to debug backpropagation?**
   - Gradient checking (numerical vs analytical)
   - Check gradient magnitudes
   - Verify shapes match

5. **Computational graph benefits?**
   - Visualize computation flow
   - Automatic differentiation
   - Efficient gradient computation

### Must-Know Formulas

```
Chain rule: dz/dx = dz/dy · dy/dx
Backprop: dL/dW^[l] = dL/dz^[l] · (a^[l-1])^T
Sigmoid derivative: σ'(x) = σ(x)(1-σ(x))
ReLU derivative: 1 if x>0, else 0
```

### Common Pitfalls

- ❌ Forgetting to cache forward pass values
- ❌ Wrong matrix dimensions in backprop
- ❌ Not checking gradients numerically
- ❌ Exploding/vanishing gradients

---

## 🔗 Connections

### Prerequisites
- [Derivatives and Gradients](Derivatives-and-Gradients.md)
- Matrix operations

### Related Topics
- [Optimization](../4_Optimization/)
- Neural network training
- Automatic differentiation

### Applications in AI
- **Training Neural Networks:** Core algorithm
- **Gradient-Based Optimization:** All variants
- **Sensitivity Analysis:** Input importance
- **Adversarial Examples:** Gradient-based attacks

---

## 📚 References

- **Papers:**
  - "Learning representations by back-propagating errors" - Rumelhart et al. (1986)
  - "Automatic differentiation in machine learning: a survey" - Baydin et al. (2018)

- **Books:**
  - "Deep Learning" - Goodfellow et al. (Chapter 6)
  - "Neural Networks and Deep Learning" - Michael Nielsen

- **Online:**
  - [CS231n: Backpropagation](http://cs231n.github.io/optimization-2/)
  - [3Blue1Brown: Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

---

**Backpropagation is the engine of deep learning - master it completely!**
