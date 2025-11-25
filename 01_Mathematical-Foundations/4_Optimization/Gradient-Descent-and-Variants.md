# Gradient Descent and Variants

> **The workhorse of machine learning** - Iterative optimization algorithms

---

## 🎯 Gradient Descent Basics

### Idea
Move in the direction opposite to the gradient (steepest descent)

```
x_{t+1} = x_t - η∇f(x_t)

η: learning rate (step size)
∇f(x_t): gradient at current point
```

### Algorithm

```python
def gradient_descent(f, grad_f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """Basic gradient descent"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        x = x - lr * grad
        history.append(x.copy())
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {i+1} iterations")
            break
    
    return x, np.array(history)
```

---

## ⚙️ Learning Rate Selection

### Fixed Learning Rate
```python
lr = 0.01  # Constant
```

**Issues:**
- Too large: divergence
- Too small: slow convergence

### Learning Rate Schedules

**1. Step Decay**
```python
def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_drop=10):
    return initial_lr * (drop ** (epoch // epochs_drop))
```

**2. Exponential Decay**
```python
def exp_decay(epoch, initial_lr=0.1, k=0.1):
    return initial_lr * np.exp(-k * epoch)
```

**3. 1/t Decay**
```python
def inverse_time_decay(epoch, initial_lr=0.1):
    return initial_lr / (1 + epoch)
```

**4. Cosine Annealing**
```python
def cosine_annealing(epoch, initial_lr=0.1, T_max=100):
    return initial_lr * (1 + np.cos(np.pi * epoch / T_max)) / 2
```

---

## 🚀 Gradient Descent Variants

### 1. Batch Gradient Descent

```python
# Use all data
grad = compute_gradient(X, y, w)
w = w - lr * grad
```

**Pros:** Stable, guaranteed convergence
**Cons:** Slow for large datasets

### 2. Stochastic Gradient Descent (SGD)

```python
# Use one sample at a time
for i in range(n):
    grad = compute_gradient(X[i], y[i], w)
    w = w - lr * grad
```

**Pros:** Fast, can escape local minima
**Cons:** Noisy, oscillates

### 3. Mini-Batch Gradient Descent

```python
# Use batch of samples
for batch in get_batches(X, y, batch_size=32):
    grad = compute_gradient(batch_X, batch_y, w)
    w = w - lr * grad
```

**Pros:** Balance of speed and stability
**Cons:** Need to tune batch size

**Best practice:** batch_size = 32, 64, 128, 256

---

## 💨 Momentum

### Idea
Add "velocity" to accelerate convergence

```
v_t = βv_{t-1} + ∇f(x_t)
x_{t+1} = x_t - ηv_t

β: momentum coefficient (typically 0.9)
```

```python
def momentum(f, grad_f, x0, lr=0.01, beta=0.9, max_iter=1000):
    """Gradient descent with momentum"""
    x = x0.copy()
    v = np.zeros_like(x)
    
    for i in range(max_iter):
        grad = grad_f(x)
        v = beta * v + grad
        x = x - lr * v
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x
```

**Benefits:**
- Faster convergence
- Dampens oscillations
- Better for ravines

---

## 🎯 Nesterov Accelerated Gradient (NAG)

### Idea
"Look ahead" before computing gradient

```
v_t = βv_{t-1} + ∇f(x_t - ηβv_{t-1})
x_{t+1} = x_t - ηv_t
```

```python
def nesterov(f, grad_f, x0, lr=0.01, beta=0.9, max_iter=1000):
    """Nesterov accelerated gradient"""
    x = x0.copy()
    v = np.zeros_like(x)
    
    for i in range(max_iter):
        # Look ahead
        x_ahead = x - lr * beta * v
        grad = grad_f(x_ahead)
        v = beta * v + grad
        x = x - lr * v
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x
```

---

## 📊 Adaptive Learning Rates

### AdaGrad

**Idea:** Adapt learning rate per parameter based on history

```
g_t = ∇f(x_t)
G_t = G_{t-1} + g_t²
x_{t+1} = x_t - (η/√(G_t + ε)) ⊙ g_t
```

```python
def adagrad(f, grad_f, x0, lr=0.01, eps=1e-8, max_iter=1000):
    """AdaGrad optimizer"""
    x = x0.copy()
    G = np.zeros_like(x)  # Sum of squared gradients
    
    for i in range(max_iter):
        grad = grad_f(x)
        G += grad**2
        x = x - lr * grad / (np.sqrt(G) + eps)
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x
```

**Issue:** Learning rate decays too aggressively

---

### RMSProp

**Idea:** Use moving average instead of sum

```
E[g²]_t = βE[g²]_{t-1} + (1-β)g_t²
x_{t+1} = x_t - (η/√(E[g²]_t + ε)) ⊙ g_t
```

```python
def rmsprop(f, grad_f, x0, lr=0.001, beta=0.9, eps=1e-8, max_iter=1000):
    """RMSProp optimizer"""
    x = x0.copy()
    E_g2 = np.zeros_like(x)
    
    for i in range(max_iter):
        grad = grad_f(x)
        E_g2 = beta * E_g2 + (1 - beta) * grad**2
        x = x - lr * grad / (np.sqrt(E_g2) + eps)
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x
```

---

### Adam (Adaptive Moment Estimation)

**Idea:** Combine momentum + RMSProp

```
m_t = β₁m_{t-1} + (1-β₁)g_t          # First moment
v_t = β₂v_{t-1} + (1-β₂)g_t²         # Second moment
m̂_t = m_t/(1-β₁ᵗ)                    # Bias correction
v̂_t = v_t/(1-β₂ᵗ)                    # Bias correction
x_{t+1} = x_t - η·m̂_t/(√v̂_t + ε)
```

```python
def adam(f, grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
    """Adam optimizer"""
    x = x0.copy()
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment
    
    for t in range(1, max_iter + 1):
        grad = grad_f(x)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update parameters
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x
```

**Default hyperparameters:**
- lr = 0.001
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8

---

## 🎓 Interview Focus

### Key Questions

1. **SGD vs Batch GD?**
   - SGD: faster, noisy, can escape local minima
   - Batch: stable, slow, guaranteed convergence

2. **Why momentum?**
   - Accelerates convergence
   - Dampens oscillations
   - Better for ravines

3. **Adam vs SGD?**
   - Adam: adaptive learning rates, works well out-of-box
   - SGD+momentum: simpler, sometimes better generalization

4. **Learning rate too large?**
   - Divergence
   - Oscillation
   - No convergence

5. **Batch size impact?**
   - Small: noisy, fast iterations, poor generalization
   - Large: stable, slow iterations, better generalization

### Must-Know Formulas

```
GD: x_{t+1} = x_t - η∇f(x_t)
Momentum: v_t = βv_{t-1} + ∇f(x_t)
Adam: m̂_t/(√v̂_t + ε)
```

---

## 📚 References

- **Papers:**
  - "Adam: A Method for Stochastic Optimization" - Kingma & Ba (2014)
  - "On the importance of initialization and momentum in deep learning" - Sutskever et al.

---

**Gradient descent: the foundation of modern ML optimization!**
