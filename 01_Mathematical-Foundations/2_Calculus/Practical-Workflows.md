# Practical Workflows - Calculus

> **Hands-on calculus for AI/ML** - NumPy, JAX, PyTorch implementations

---

## 🛠️ Essential Libraries

```python
import numpy as np
import jax
import jax.numpy as jnp
import torch
from scipy import optimize, integrate
```

---

## 📊 Common Workflows

### 1. Numerical Derivatives

```python
def numerical_derivative(f, x, h=1e-5):
    """Central difference formula"""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x, h=1e-5):
    """Gradient for vector input"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad
```

### 2. Automatic Differentiation (JAX)

```python
# JAX - Best for research
f = lambda x: jnp.sum(x**2)

grad_f = jax.grad(f)  # Gradient
hess_f = jax.hessian(f)  # Hessian
jac_f = jax.jacobian(f)  # Jacobian

x = jnp.array([1.0, 2.0, 3.0])
print(f"Gradient: {grad_f(x)}")
print(f"Hessian:\n{hess_f(x)}")
```

### 3. PyTorch Autograd

```python
# PyTorch - Best for deep learning
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.sum(x**2)

y.backward()  # Compute gradients
print(f"Gradient: {x.grad}")

# Higher-order derivatives
x = torch.tensor([1.0], requires_grad=True)
y = x**3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"Second derivative: {d2y_dx2}")
```

### 4. Optimization

```python
from scipy.optimize import minimize

# Minimize function
f = lambda x: np.sum(x**2)
grad_f = lambda x: 2*x

x0 = np.array([5.0, 5.0])
result = minimize(f, x0, jac=grad_f, method='BFGS')
print(f"Minimum at: {result.x}")
```

---

## 🎯 ML-Specific Patterns

### Gradient Descent

```python
def gradient_descent(f, grad_f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """Standard gradient descent"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        x = x - lr * grad
        history.append(x.copy())
        
        if np.linalg.norm(grad) < tol:
            break
    
    return x, np.array(history)
```

### Momentum

```python
def momentum(f, grad_f, x0, lr=0.01, beta=0.9, max_iter=1000):
    """Gradient descent with momentum"""
    x = x0.copy()
    v = np.zeros_like(x)
    
    for i in range(max_iter):
        grad = grad_f(x)
        v = beta * v + (1 - beta) * grad
        x = x - lr * v
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x
```

### Adam Optimizer

```python
def adam(f, grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
    """Adam optimizer"""
    x = x0.copy()
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment
    
    for t in range(1, max_iter + 1):
        grad = grad_f(x)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x
```

---

## 🔧 Gradient Checking

```python
def gradient_check(f, grad_f, x, epsilon=1e-7):
    """Verify analytical gradient"""
    numerical_grad = numerical_gradient(f, x)
    analytical_grad = grad_f(x)
    
    diff = np.linalg.norm(numerical_grad - analytical_grad)
    norm = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
    relative_error = diff / (norm + 1e-8)
    
    passed = relative_error < epsilon
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{status} - Relative error: {relative_error:.2e}")
    
    return passed
```

---

## 📚 Quick Reference

```python
# JAX
jax.grad(f)          # Gradient
jax.hessian(f)       # Hessian
jax.jacobian(f)      # Jacobian
jax.value_and_grad(f)  # Value + gradient

# PyTorch
x.requires_grad = True
y.backward()
x.grad

# SciPy
scipy.optimize.minimize(f, x0, jac=grad_f)
scipy.integrate.quad(f, a, b)
```

---

**Master these workflows for efficient AI development!**
