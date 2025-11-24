# Integration and Applications

> **The reverse of differentiation** - Accumulation, area, and probabilistic reasoning

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

**Historical Note:** Developed independently by Newton and Leibniz (1670s), revolutionized mathematics by connecting differentiation and integration.

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
∫1/(1+x²)dx = arctan(x) + C
```

### Integration by Parts
```
∫u dv = uv - ∫v du

Mnemonic: LIATE (choose u in this order)
L: Logarithmic
I: Inverse trig
A: Algebraic
T: Trigonometric
E: Exponential
```

### Substitution
```
∫f(g(x))g'(x)dx = ∫f(u)du  where u = g(x)
```

**Example:**
```python
# ∫2x·e^(x²)dx
# Let u = x², du = 2x dx
# ∫e^u du = e^u + C = e^(x²) + C
```

---

## 🎯 Numerical Integration

### Riemann Sum
```
∫ₐᵇ f(x)dx ≈ Σᵢ f(xᵢ)Δx

Error: O(Δx) = O((b-a)/n)
```

### Trapezoidal Rule
```
∫ₐᵇ f(x)dx ≈ (b-a)/2n · [f(x₀) + 2f(x₁) + ... + 2f(xₙ₋₁) + f(xₙ)]

Error: O((b-a)³/n²)
```

### Simpson's Rule
```
∫ₐᵇ f(x)dx ≈ (b-a)/3n · [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]

Error: O((b-a)⁵/n⁴)
Much better than trapezoidal!
```

### Gaussian Quadrature
```
∫₋₁¹ f(x)dx ≈ Σᵢ wᵢf(xᵢ)

Optimal choice of points xᵢ and weights wᵢ
Exact for polynomials up to degree 2n-1
```

---

## 💻 Practical Implementation

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

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

def simpsons(f, a, b, n=1000):
    """Simpson's rule (n must be even)"""
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n+1)
    y = f(x)
    dx = (b - a) / n
    return dx/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])

# Using SciPy (adaptive quadrature)
f = lambda x: x**2
result, error = integrate.quad(f, 0, 1)
print(f"Integral: {result:.6f}")  # Should be 1/3
print(f"Error estimate: {error:.2e}")

# Compare methods
true_value = 1/3
for n in [10, 100, 1000]:
    riemann = riemann_sum(f, 0, 1, n)
    trap = trapezoidal(f, 0, 1, n)
    simp = simpsons(f, 0, 1, n)
    print(f"n={n}:")
    print(f"  Riemann error: {abs(riemann - true_value):.2e}")
    print(f"  Trapezoid error: {abs(trap - true_value):.2e}")
    print(f"  Simpson error: {abs(simp - true_value):.2e}")
```

---

## 🎲 Monte Carlo Integration

### Basic Monte Carlo
```python
def monte_carlo_integrate(f, a, b, n=10000):
    """Monte Carlo integration"""
    x = np.random.uniform(a, b, n)
    return (b - a) * np.mean(f(x))

# Example
f = lambda x: np.exp(-x**2)
result = monte_carlo_integrate(f, 0, 1, 100000)
print(f"MC estimate: {result:.6f}")

# True value
true_val, _ = integrate.quad(f, 0, 1)
print(f"True value: {true_val:.6f}")
```

### Importance Sampling
```python
def importance_sampling(f, proposal_dist, proposal_pdf, n=10000):
    """Importance sampling for integration"""
    # Sample from proposal distribution
    samples = proposal_dist(n)
    
    # Compute weights
    weights = f(samples) / proposal_pdf(samples)
    
    return np.mean(weights)

# Example: ∫e^(-x²)dx from 0 to ∞
# Use exponential proposal
proposal_dist = lambda n: np.random.exponential(1, n)
proposal_pdf = lambda x: np.exp(-x)
f = lambda x: np.exp(-x**2)

result = importance_sampling(f, proposal_dist, proposal_pdf, 100000)
print(f"Importance sampling: {result:.6f}")
```

### Multidimensional Integration
```python
# Monte Carlo shines in high dimensions!
def mc_integrate_nd(f, bounds, n=100000):
    """Monte Carlo integration in n dimensions"""
    dim = len(bounds)
    
    # Generate random points
    points = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n, dim)
    )
    
    # Compute volume
    volume = np.prod([b[1] - b[0] for b in bounds])
    
    # Estimate integral
    return volume * np.mean(f(points))

# Example: ∫∫e^(-(x²+y²))dxdy over [0,1]×[0,1]
f = lambda p: np.exp(-np.sum(p**2, axis=1))
bounds = [(0, 1), (0, 1)]
result = mc_integrate_nd(f, bounds)
print(f"2D integral: {result:.6f}")
```

---

## 🎯 AI/ML Applications

### 1. Probability Distributions

```python
from scipy.stats import norm

# PDF must integrate to 1
pdf = lambda x: (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)

# Verify normalization
total_prob, _ = integrate.quad(pdf, -np.inf, np.inf)
print(f"Total probability: {total_prob:.6f}")  # 1.0

# CDF via integration
def cdf_from_pdf(pdf, x):
    """Compute CDF from PDF"""
    result, _ = integrate.quad(pdf, -np.inf, x)
    return result

# Compare with scipy
x = 1.96
cdf_integrated = cdf_from_pdf(pdf, x)
cdf_scipy = norm.cdf(x)
print(f"Integrated CDF: {cdf_integrated:.6f}")
print(f"SciPy CDF: {cdf_scipy:.6f}")
```

### 2. Expected Value and Moments

```python
def expected_value(pdf, a, b):
    """E[X] = ∫x·p(x)dx"""
    integrand = lambda x: x * pdf(x)
    return integrate.quad(integrand, a, b)[0]

def variance(pdf, a, b):
    """Var(X) = E[X²] - (E[X])²"""
    mean = expected_value(pdf, a, b)
    integrand = lambda x: (x - mean)**2 * pdf(x)
    return integrate.quad(integrand, a, b)[0]

def nth_moment(pdf, n, a, b):
    """E[X^n] = ∫x^n·p(x)dx"""
    integrand = lambda x: x**n * pdf(x)
    return integrate.quad(integrand, a, b)[0]

# Example: Uniform distribution on [0,1]
pdf = lambda x: 1.0
print(f"E[X] = {expected_value(pdf, 0, 1):.3f}")  # 0.5
print(f"Var(X) = {variance(pdf, 0, 1):.3f}")  # 1/12 ≈ 0.083
```

### 3. Marginal Distributions

```python
# Marginalize joint distribution
def marginalize(joint_pdf, x_val):
    """p(x) = ∫p(x,y)dy"""
    integrand = lambda y: joint_pdf(x_val, y)
    result, _ = integrate.quad(integrand, -np.inf, np.inf)
    return result

# Example: Bivariate normal
from scipy.stats import multivariate_normal

mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
mvn = multivariate_normal(mean, cov)

joint_pdf = lambda x, y: mvn.pdf([x, y])
marginal_x = marginalize(joint_pdf, 0)
print(f"Marginal p(x=0): {marginal_x:.6f}")
```

### 4. Loss Function Integration

```python
# Expected risk
def expected_risk(loss_fn, model, data_dist, x_range):
    """E[L(f(x), y)] over data distribution"""
    def integrand(x):
        # Assuming y = true_function(x) + noise
        y_true = true_function(x)
        y_pred = model(x)
        return loss_fn(y_pred, y_true) * data_dist(x)
    
    return integrate.quad(integrand, x_range[0], x_range[1])[0]
```

### 5. Variational Inference (ELBO)

```python
# Evidence Lower Bound
def elbo(q_dist, log_p, log_q, n_samples=10000):
    """ELBO = E_q[log p(x,z)] - E_q[log q(z)]"""
    samples = q_dist(n_samples)
    return np.mean(log_p(samples) - log_q(samples))
```

---

## 🛡️ Numerical Stability

### Common Issues

**1. Infinite Limits**
```python
# BAD: May not converge
result = integrate.quad(f, 0, np.inf)

# GOOD: Use appropriate method
result = integrate.quad(f, 0, np.inf, limit=100)
# Or transform to finite interval
```

**2. Singularities**
```python
# Function with singularity at x=0
f = lambda x: 1/np.sqrt(x)

# SciPy handles this
result, error = integrate.quad(f, 0, 1)
print(f"Result: {result:.6f}")  # 2.0
```

**3. Oscillatory Functions**
```python
# Highly oscillatory integrand
f = lambda x: np.sin(100*x)

# Need more subdivisions
result = integrate.quad(f, 0, 1, limit=1000)
```

---

## 🎓 Advanced Exercises

### Exercise 1: Gaussian Integral
**Problem:** Prove that ∫₋∞^∞ e^(-x²)dx = √π

**Hint:** Use polar coordinates and double integral

### Exercise 2: Implement Adaptive Quadrature
**Problem:** Write adaptive Simpson's rule that subdivides intervals where error is large

```python
def adaptive_simpson(f, a, b, tol=1e-6):
    """Adaptive Simpson's rule"""
    # Your implementation here
    pass
```

### Exercise 3: Monte Carlo Convergence
**Problem:** Show empirically that MC error decreases as O(1/√n)

---

## 🎓 Interview Focus

### Key Questions

1.  **What is the fundamental theorem of calculus?**
    - Links differentiation and integration
    - ∫ₐᵇ f'(x)dx = f(b) - f(a)
    - Foundation of calculus

2.  **Why use numerical integration?**
    - No closed-form solution
    - Complex functions
    - Empirical data

3.  **Monte Carlo vs deterministic methods?**
    - MC: O(1/√n), dimension-independent
    - Deterministic: O(1/n^(1/d)), curse of dimensionality
    - MC better for high dimensions

4.  **Integration in ML?**
    - Computing expectations
    - Marginalizing distributions
    - ELBO in variational inference

5.  **Importance sampling advantage?**
    - Reduces variance
    - Focuses samples where integrand is large
    - Critical for rare events

---

## 📚 References

-   **Books:** 
    - "Calculus" - James Stewart
    - "Numerical Recipes" - Press et al.
    - "Monte Carlo Statistical Methods" - Robert & Casella

-   **Online:**
    - [SciPy Integration](https://docs.scipy.org/doc/scipy/tutorial/integrate.html)
    - [3Blue1Brown: Integration](https://www.youtube.com/watch?v=rfG8ce4nNh0)

---

**Integration: from areas to expectations, the foundation of probabilistic ML!**
