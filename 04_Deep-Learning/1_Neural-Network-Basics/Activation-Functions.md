# Activation Functions

> **Non-linearity in neural networks** - From sigmoid to modern activations

---

## 🎯 Why Activation Functions?

**Purpose:**
- Introduce non-linearity
- Enable learning complex patterns
- Control gradient flow

**Without activation:** Neural network = linear regression

---

## 📊 Classic Activations

### Sigmoid
```
σ(x) = 1/(1 + e^(-x))

Range: (0, 1)
Derivative: σ'(x) = σ(x)(1 - σ(x))
```

```python
import torch
import torch.nn as nn

# Sigmoid
sigmoid = nn.Sigmoid()
x = torch.linspace(-5, 5, 100)
y = sigmoid(x)

# Plot
import matplotlib.pyplot as plt
plt.plot(x.numpy(), y.numpy())
plt.title('Sigmoid')
plt.grid(True)
plt.show()
```

**Problems:**
- Vanishing gradients (saturates at 0 and 1)
- Not zero-centered
- Computationally expensive

### Tanh
```
tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

Range: (-1, 1)
Derivative: tanh'(x) = 1 - tanh²(x)
```

**Advantages over sigmoid:**
- Zero-centered
- Stronger gradients

**Still has:** Vanishing gradient problem

---

## 🎯 ReLU Family

### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)

Derivative: 1 if x > 0, else 0
```

```python
# ReLU
relu = nn.ReLU()
x = torch.linspace(-5, 5, 100)
y = relu(x)

plt.plot(x.numpy(), y.numpy())
plt.title('ReLU')
plt.grid(True)
plt.show()
```

**Advantages:**
- No vanishing gradient (for x > 0)
- Computationally efficient
- Sparse activation

**Problem:** Dying ReLU (neurons can die if x < 0 always)

### Leaky ReLU
```
LeakyReLU(x) = max(αx, x)

α: small constant (e.g., 0.01)
```

```python
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
```

**Fixes:** Dying ReLU problem

### ELU (Exponential Linear Unit)
```
ELU(x) = x if x > 0
         α(e^x - 1) if x ≤ 0
```

```python
elu = nn.ELU(alpha=1.0)
```

**Advantages:**
- Smooth everywhere
- Negative saturation
- Zero-centered mean

---

## 📈 Modern Activations

### GELU (Gaussian Error Linear Unit)
```
GELU(x) ≈ x·Φ(x)

Φ: cumulative distribution function of standard normal
```

```python
gelu = nn.GELU()
```

**Used in:** BERT, GPT (transformers)

### Swish / SiLU
```
Swish(x) = x·σ(x)
```

```python
silu = nn.SiLU()  # Swish
```

**Properties:**
- Smooth
- Non-monotonic
- Self-gated

---

## 🎓 Interview Focus

### Key Questions

1. **Why ReLU over sigmoid?**
   - No vanishing gradient (x > 0)
   - Faster computation
   - Sparse activation

2. **Dying ReLU problem?**
   - Neurons output 0 for all inputs
   - Caused by large negative bias
   - Fix: Leaky ReLU, ELU

3. **When to use which activation?**
   - Hidden layers: ReLU, GELU
   - Output (binary): Sigmoid
   - Output (multi-class): Softmax
   - Output (regression): None (linear)

---

**Activations: bringing non-linearity to neural networks!**
