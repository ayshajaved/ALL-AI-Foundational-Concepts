# Derivatives and Gradients

> **Foundation of optimization in AI** - Understanding how functions change

---

## 📐 Derivatives

### Definition

The **derivative** measures the rate of change of a function.

**Formal Definition:**
```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

**Notation:**
- f'(x) - Lagrange notation
- df/dx - Leibniz notation  
- Df(x) - Operator notation
- ḟ(x) - Newton notation

### Geometric Interpretation

- **Slope** of tangent line at point x
- **Instantaneous rate of change**
- Direction of steepest ascent

---

## 🔢 Basic Derivative Rules

### Power Rule
```
d/dx(xⁿ) = nxⁿ⁻¹
```

### Constant Rule
```
d/dx(c) = 0
```

### Constant Multiple Rule
```
d/dx(cf(x)) = c·f'(x)
```

### Sum/Difference Rule
```
d/dx(f(x) ± g(x)) = f'(x) ± g'(x)
```

### Product Rule
```
d/dx(f(x)g(x)) = f'(x)g(x) + f(x)g'(x)
```

### Quotient Rule
```
d/dx(f(x)/g(x)) = (f'(x)g(x) - f(x)g'(x)) / g(x)²
```

### Chain Rule
```
d/dx(f(g(x))) = f'(g(x))·g'(x)
```

---

## 📊 Common Derivatives

### Exponential & Logarithmic
```
d/dx(eˣ) = eˣ
d/dx(aˣ) = aˣ ln(a)
d/dx(ln(x)) = 1/x
d/dx(logₐ(x)) = 1/(x ln(a))
```

### Trigonometric
```
d/dx(sin(x)) = cos(x)
d/dx(cos(x)) = -sin(x)
d/dx(tan(x)) = sec²(x)
```

### Activation Functions (AI/ML)
```
d/dx(sigmoid(x)) = sigmoid(x)(1 - sigmoid(x))
d/dx(tanh(x)) = 1 - tanh²(x)
d/dx(ReLU(x)) = 1 if x>0, else 0
d/dx(LeakyReLU(x)) = 1 if x>0, else α
```

---

## 🎯 Partial Derivatives

For multivariable functions f(x₁, x₂, ..., xₙ):

**Partial derivative** with respect to xᵢ:
```
∂f/∂xᵢ = lim[h→0] (f(x₁,...,xᵢ+h,...,xₙ) - f(x₁,...,xᵢ,...,xₙ)) / h
```

**Notation:** ∂f/∂xᵢ or fₓᵢ

**Example:**
```
f(x,y) = x²y + 3xy²

∂f/∂x = 2xy + 3y²
∂f/∂y = x² + 6xy
```

---

## 🔺 Gradient

The **gradient** is a vector of all partial derivatives.

**Definition:**
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Properties:**
- Points in direction of steepest ascent
- Perpendicular to level curves/surfaces
- Magnitude = rate of steepest increase

**Example:**
```
f(x,y) = x² + y²

∇f = [2x, 2y]ᵀ

At point (1,1): ∇f = [2, 2]ᵀ
```

---

## 🎨 Directional Derivative

Rate of change in direction **u** (unit vector):

```
Dᵤf(x) = ∇f(x) · u
```

**Maximum rate of change** occurs when u is parallel to ∇f:
```
max Dᵤf = ||∇f||
```

---

## 🔧 Higher-Order Derivatives

### Second Derivative
```
f''(x) = d²f/dx²
```

**Interpretation:**
- Measures curvature
- f'' > 0: convex (curving up)
- f'' < 0: concave (curving down)

### Hessian Matrix

For f: ℝⁿ → ℝ, the **Hessian** is the matrix of second partial derivatives:

```
H = [∂²f/∂xᵢ∂xⱼ]

     [∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ]
H =  [∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ]
     [    ⋮            ⋮       ⋱        ⋮     ]
     [∂²f/∂xₙ∂x₁  ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²  ]
```

**Properties:**
- Symmetric (if f is C²)
- Describes local curvature
- Used in optimization (Newton's method)

---

## ⚡ Computational Complexity

### Gradient Computation Complexity

| Operation | Forward Pass | Backward Pass | Total | Notes |
|-----------|-------------|---------------|-------|-------|
| Scalar function | O(1) | O(1) | O(1) | Single derivative |
| Gradient (n vars) | O(f) | O(n·f) | O(n·f) | f = function cost |
| Hessian (n vars) | O(f) | O(n²·f) | O(n²·f) | Expensive! |
| Jacobian (m×n) | O(m·f) | O(m·n·f) | O(m·n·f) | Matrix output |

### Automatic Differentiation

**Forward Mode:**
- Complexity: O(n) for n inputs
- Best for: Few inputs, many outputs
- Example: f: ℝ¹⁰ → ℝ¹⁰⁰⁰

**Reverse Mode (Backpropagation):**
- Complexity: O(m) for m outputs  
- Best for: Many inputs, few outputs
- Example: f: ℝ¹⁰⁰⁰ → ℝ¹ (typical in ML!)

---

## 🛡️ Numerical Stability

### Critical Issues

**1. Vanishing Gradients**
```python
# Problem: Sigmoid saturates
x = np.array([-10, -5, 0, 5, 10])
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_grad = lambda x: sigmoid(x) * (1 - sigmoid(x))

print(sigmoid_grad(x))  # [0, 0.007, 0.25, 0.007, 0]
# Gradients vanish for |x| > 5!

# Solution: Use ReLU or careful initialization
```

**2. Exploding Gradients**
```python
# Problem: Repeated multiplication
gradients = [2.0] * 100  # Each layer multiplies by 2
total_gradient = np.prod(gradients)  # 2^100 = overflow!

# Solutions:
# 1. Gradient clipping
max_norm = 1.0
if np.linalg.norm(grad) > max_norm:
    grad = grad * (max_norm / np.linalg.norm(grad))

# 2. Batch normalization
# 3. Residual connections
```

**3. Numerical Gradient Checking**
```python
def gradient_check(f, grad_f, x, epsilon=1e-7, threshold=1e-5):
    """
    Verify analytical gradient against numerical gradient
    
    Returns:
        relative_error: Should be < 1e-5 for correct implementation
    """
    # Numerical gradient
    grad_numerical = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad_numerical[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    
    # Analytical gradient
    grad_analytical = grad_f(x)
    
    # Relative error
    numerator = np.linalg.norm(grad_numerical - grad_analytical)
    denominator = np.linalg.norm(grad_numerical) + np.linalg.norm(grad_analytical)
    relative_error = numerator / (denominator + 1e-8)
    
    if relative_error < threshold:
        print(f"✅ Gradient check PASSED (error: {relative_error:.2e})")
    else:
        print(f"❌ Gradient check FAILED (error: {relative_error:.2e})")
        print(f"Numerical: {grad_numerical}")
        print(f"Analytical: {grad_analytical}")
    
    return relative_error

# Example
f = lambda x: np.sum(x**2)
grad_f = lambda x: 2*x
x = np.array([1.0, 2.0, 3.0])
gradient_check(f, grad_f, x)
```

**4. Softmax Numerical Stability**
```python
# BAD: Overflow for large values
def softmax_unstable(x):
    return np.exp(x) / np.sum(np.exp(x))

# GOOD: Subtract max for stability  
def softmax_stable(x):
    x_shifted = x - np.max(x)
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted))

# Test
x = np.array([1000, 1001, 1002])  # Large values
print(softmax_unstable(x))  # NaN!
print(softmax_stable(x))    # Works!
```

---

## 🎓 Advanced Topics

### Matrix Calculus

**Essential Rules for Neural Networks:**

```
1. ∂(Ax)/∂x = Aᵀ
2. ∂(xᵀA)/∂x = A
3. ∂(xᵀAx)/∂x = (A + Aᵀ)x
4. ∂tr(AB)/∂A = Bᵀ
5. ∂|A|/∂A = |A|(A⁻¹)ᵀ  (determinant)
```

**Example: Linear Layer Gradient**
```python
# Forward: y = Wx + b
# Loss: L = ||y - y_true||²

# Gradients:
# ∂L/∂W = ∂L/∂y · ∂y/∂W = (y - y_true) ⊗ xᵀ
# ∂L/∂b = ∂L/∂y = (y - y_true)
# ∂L/∂x = ∂L/∂y · ∂y/∂x = Wᵀ(y - y_true)

def linear_layer_gradients(W, x, b, y_true):
    """Compute all gradients for linear layer"""
    # Forward
    y = W @ x + b
    
    # Backward
    dL_dy = y - y_true
    dL_dW = np.outer(dL_dy, x)  # Outer product
    dL_db = dL_dy
    dL_dx = W.T @ dL_dy
    
    return dL_dW, dL_db, dL_dx
```

### Jacobian and Hessian-Vector Products

**Jacobian** (for vector-valued functions):
```python
# f: ℝⁿ → ℝᵐ
# J = [∂fᵢ/∂xⱼ]

import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x[0]**2 + x[1], x[0] * x[1]])

# Jacobian
jac_f = jax.jacobian(f)
x = jnp.array([1.0, 2.0])
print(jac_f(x))
# [[2, 1],
#  [2, 1]]
```

**Hessian-Vector Product** (efficient for large dimensions):
```python
# Instead of computing full Hessian (O(n²)),
# compute Hv directly (O(n))

def hvp(f, x, v):
    """Hessian-vector product: Hv"""
    return jax.jvp(jax.grad(f), (x,), (v,))[1]

# Example
f = lambda x: jnp.sum(x**2)
x = jnp.array([1.0, 2.0, 3.0])
v = jnp.array([1.0, 0.0, 0.0])

print(hvp(f, x, v))  # [2, 0, 0] (first column of Hessian)
```

### Automatic Differentiation Theory

**Forward Mode AD:**
```python
# Compute derivative alongside function evaluation
# Uses dual numbers: x + x'ε where ε² = 0

class Dual:
    def __init__(self, value, derivative):
        self.v = value
        self.d = derivative
    
    def __add__(self, other):
        return Dual(self.v + other.v, self.d + other.d)
    
    def __mul__(self, other):
        # (a + a'ε)(b + b'ε) = ab + (a'b + ab')ε
        return Dual(self.v * other.v, 
                   self.d * other.v + self.v * other.d)

# Example: f(x) = x²
x = Dual(3.0, 1.0)  # x=3, dx/dx=1
y = x * x
print(f"f(3) = {y.v}, f'(3) = {y.d}")  # 9, 6
```

**Reverse Mode AD (Backpropagation):**
```
1. Forward pass: Compute function value, store intermediate results
2. Backward pass: Compute gradients using chain rule

Memory: O(n) where n = number of operations
Time: O(n) for gradient w.r.t. all inputs

This is why backprop is efficient for ML!
```

---

## 💻 Practical Workflows

### NumPy Implementation

```python
import numpy as np

# Numerical derivative (finite differences)
def numerical_derivative(f, x, h=1e-5):
    """Compute derivative using finite differences"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Example
f = lambda x: x**2
x = 2.0
df_dx = numerical_derivative(f, x)
print(f"f'({x}) ≈ {df_dx}")  # Should be ≈ 4

# Gradient (multivariable)
def numerical_gradient(f, x, h=1e-5):
    """Compute gradient using finite differences"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Example
f = lambda x: x[0]**2 + x[1]**2
x = np.array([1.0, 1.0])
grad = numerical_gradient(f, x)
print(f"∇f({x}) ≈ {grad}")  # Should be ≈ [2, 2]

# Hessian
def numerical_hessian(f, x, h=1e-5):
    """Compute Hessian using finite differences"""
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            
            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h
            
            H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
    
    return H

# Example
H = numerical_hessian(f, x)
print(f"Hessian:\n{H}")  # Should be ≈ [[2, 0], [0, 2]]
```

### Automatic Differentiation

```python
# Using JAX (recommended for ML)
import jax
import jax.numpy as jnp

# Define function
def f(x):
    return jnp.sum(x**2)

# Gradient
grad_f = jax.grad(f)
x = jnp.array([1.0, 2.0, 3.0])
print(f"Gradient: {grad_f(x)}")  # [2., 4., 6.]

# Hessian
hessian_f = jax.hessian(f)
print(f"Hessian:\n{hessian_f(x)}")

# Using PyTorch
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.sum(x**2)

# Compute gradient
y.backward()
print(f"Gradient: {x.grad}")  # tensor([2., 4., 6.])
```

---

## 🎯 AI/ML Applications

### 1. Gradient Descent

```python
def gradient_descent(f, grad_f, x0, lr=0.01, max_iter=1000):
    """Minimize f using gradient descent"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        x = x - lr * grad
        history.append(x.copy())
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x, history

# Example: minimize f(x,y) = x² + y²
f = lambda x: np.sum(x**2)
grad_f = lambda x: 2*x

x0 = np.array([5.0, 5.0])
x_min, history = gradient_descent(f, grad_f, x0, lr=0.1)
print(f"Minimum at: {x_min}")  # Should be ≈ [0, 0]
```

### 2. Backpropagation

```python
# Simple neural network layer
class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)
    
    def forward(self, x):
        self.x = x  # Cache for backward pass
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        # Gradients
        self.grad_W = self.x.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.W.T
        return grad_input
    
    def update(self, lr=0.01):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b
```

### 3. Loss Function Gradients

```python
# MSE loss gradient
def mse_gradient(y_pred, y_true):
    """Gradient of MSE loss"""
    return 2 * (y_pred - y_true) / len(y_true)

# Cross-entropy loss gradient (with softmax)
def cross_entropy_gradient(logits, y_true):
    """Gradient of cross-entropy with softmax"""
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    probs[range(len(y_true)), y_true] -= 1
    return probs / len(y_true)
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is a derivative?**
   - Rate of change of function
   - Slope of tangent line
   - Limit of difference quotient

2. **What is the gradient?**
   - Vector of partial derivatives
   - Points in direction of steepest ascent
   - Used in optimization

3. **Chain rule in neural networks?**
   - Backpropagation uses chain rule
   - Compute gradients layer by layer
   - ∂L/∂w = ∂L/∂y · ∂y/∂w

4. **Why is gradient descent called "gradient" descent?**
   - Uses gradient to find direction
   - Moves opposite to gradient (descent)
   - Minimizes loss function

5. **What does Hessian tell us?**
   - Second-order curvature information
   - Positive definite → local minimum
   - Used in Newton's method

### Must-Know Formulas

```
Chain rule: d/dx(f(g(x))) = f'(g(x))·g'(x)
Gradient: ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ
Gradient descent: x_{t+1} = x_t - η∇f(x_t)
Sigmoid derivative: σ'(x) = σ(x)(1-σ(x))
```

### Common Pitfalls

- ❌ Forgetting chain rule in backprop
- ❌ Not checking gradient numerically
- ❌ Confusing gradient with directional derivative
- ❌ Wrong sign in gradient descent (should be minus)

---

## 🔗 Connections

### Prerequisites
- Basic algebra
- Functions

### Related Topics
- [Chain Rule](Chain-Rule-and-Backpropagation.md)
- [Multivariate Calculus](Multivariate-Calculus.md)
- [Optimization](../4_Optimization/)

### Applications in AI
- **Gradient Descent:** Optimization algorithm
- **Backpropagation:** Training neural networks
- **Feature Importance:** Gradient magnitude
- **Adversarial Examples:** Gradient-based attacks

---

## 📚 References

- **Books:**
  - "Calculus" - James Stewart
  - "Deep Learning" - Goodfellow et al. (Chapter 4)

- **Online:**
  - [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
  - [Khan Academy: Calculus](https://www.khanacademy.org/math/calculus-1)

---

**Master derivatives and gradients - they power all of deep learning!**
