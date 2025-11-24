# Integration and Applications

> **The reverse of differentiation** - Accumulation and area under curves

---

## 📐 Integration Basics

### Indefinite Integral (Antiderivative)

```
∫f(x)dx = F(x) + C

where F'(x) = f(x)
```

### Definite Integral

```
∫ₐᵇ f(x)dx = F(b) - F(a)
```

**Geometric Interpretation:** Area under curve from a to b

---

## 🔢 Integration Rules

### Power Rule
```
∫xⁿdx = xⁿ⁺¹/(n+1) + C  (n ≠ -1)
```

### Common Integrals
```
∫eˣdx = eˣ + C
∫(1/x)dx = ln|x| + C
∫sin(x)dx = -cos(x) + C
∫cos(x)dx = sin(x) + C
```

### Integration by Parts
```
∫u dv = uv - ∫v du
```

### Substitution
```
∫f(g(x))g'(x)dx = ∫f(u)du  where u = g(x)
```

---

## 🎯 Numerical Integration

### Riemann Sum
```
∫ₐᵇ f(x)dx ≈ Σᵢ f(xᵢ)Δx
```

### Trapezoidal Rule
```
∫ₐᵇ f(x)dx ≈ (b-a)/2n · [f(x₀) + 2f(x₁) + ... + 2f(xₙ₋₁) + f(xₙ)]
```

### Simpson's Rule
```
∫ₐᵇ f(x)dx ≈ (b-a)/3n · [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]
```

---

## 💻 Practical Implementation

```python
import numpy as np
from scipy import integrate

# Numerical integration
def riemann_sum(f, a, b, n=1000):
    """Riemann sum approximation"""
    x = np.linspace(a, b, n)
    dx = (b - a) / n
    return np.sum(f(x) * dx)

def trapezoidal(f, a, b, n=1000):
    """Trapezoidal rule"""
    x = np.linspace(a, b, n+1)
    y = f(x)
    dx = (b - a) / n
    return dx * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)

# Using SciPy
f = lambda x: x**2
result, error = integrate.quad(f, 0, 1)
print(f"Integral: {result}")  # Should be 1/3

# Monte Carlo integration
def monte_carlo_integrate(f, a, b, n=10000):
    """Monte Carlo integration"""
    x = np.random.uniform(a, b, n)
    return (b - a) * np.mean(f(x))
```

---

## 🎯 AI/ML Applications

### 1. Probability Distributions

```python
# PDF must integrate to 1
from scipy.stats import norm

# Gaussian PDF
pdf = lambda x: (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)

# Verify normalization
total_prob, _ = integrate.quad(pdf, -np.inf, np.inf)
print(f"Total probability: {total_prob}")  # Should be 1.0

# CDF (cumulative distribution)
cdf = lambda x: integrate.quad(pdf, -np.inf, x)[0]
```

### 2. Expected Value

```python
# E[X] = ∫x·p(x)dx
def expected_value(pdf, a, b):
    """Compute expected value"""
    integrand = lambda x: x * pdf(x)
    return integrate.quad(integrand, a, b)[0]

# Example: Uniform distribution on [0,1]
pdf = lambda x: 1.0  # Uniform
E_X = expected_value(pdf, 0, 1)
print(f"E[X] = {E_X}")  # Should be 0.5
```

### 3. Loss Function Integration

```python
# Expected loss over distribution
def expected_loss(loss_fn, pred_dist, true_val):
    """Compute expected loss"""
    integrand = lambda y_pred: loss_fn(y_pred, true_val) * pred_dist(y_pred)
    return integrate.quad(integrand, -np.inf, np.inf)[0]
```

---

## 📚 References

- **Books:** "Calculus" - James Stewart
- **Online:** Khan Academy, 3Blue1Brown

---

**Integration completes the calculus toolkit for AI!**
