# Proximal Methods and ADMM

> **Advanced optimization for non-smooth problems** - Proximal operators, ADMM, and sparse learning

---

## 🎯 Proximal Operators

### Definition
The **proximal operator** of function f is:
```
prox_f(v) = argmin_x {f(x) + ½||x - v||²}

Balances minimizing f and staying close to v
```

### Common Proximal Operators

**1. L1 Norm (Soft Thresholding)**
```
prox_{λ||·||₁}(v) = sign(v) ⊙ max(|v| - λ, 0)
```

```python
def prox_l1(v, lambda_):
    """Proximal operator for L1 norm"""
    return np.sign(v) * np.maximum(np.abs(v) - lambda_, 0)

# Example
v = np.array([2.0, -1.5, 0.5, -0.3])
lambda_ = 1.0
result = prox_l1(v, lambda_)
print(f"prox_L1({v}) = {result}")
# [1.0, -0.5, 0.0, 0.0]
```

**2. L2 Norm**
```
prox_{λ||·||₂}(v) = max(1 - λ/||v||, 0) · v
```

**3. Indicator Function (Projection)**
```
prox_{I_C}(v) = proj_C(v)

Project onto set C
```

---

## 📊 Proximal Gradient Method

### Problem
```
minimize f(x) + g(x)

f: smooth (differentiable)
g: non-smooth (e.g., L1 norm)
```

### Algorithm
```
x_{k+1} = prox_{t_k g}(x_k - t_k ∇f(x_k))

1. Gradient step on f
2. Proximal step on g
```

```python
def proximal_gradient(f, grad_f, prox_g, x0, t=0.01, max_iter=1000):
    """
    Proximal gradient method
    
    f: smooth function
    grad_f: gradient of f
    prox_g: proximal operator of g
    t: step size
    """
    x = x0.copy()
    
    for k in range(max_iter):
        # Gradient step
        x_half = x - t * grad_f(x)
        
        # Proximal step
        x_new = prox_g(x_half, t)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-6:
            break
        
        x = x_new
    
    return x
```

---

## 🎯 ISTA (Iterative Shrinkage-Thresholding Algorithm)

### Lasso Problem
```
minimize ½||Ax - b||² + λ||x||₁
```

```python
def ista(A, b, lambda_, max_iter=1000, tol=1e-6):
    """
    ISTA for Lasso
    """
    m, n = A.shape
    x = np.zeros(n)
    
    # Step size
    L = np.linalg.norm(A.T @ A, 2)
    t = 1 / L
    
    for k in range(max_iter):
        # Gradient of ½||Ax - b||²
        grad = A.T @ (A @ x - b)
        
        # Gradient step
        x_half = x - t * grad
        
        # Soft thresholding (prox of λ||·||₁)
        x_new = prox_l1(x_half, t * lambda_)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
    
    return x

# Example
A = np.random.randn(100, 50)
x_true = np.zeros(50)
x_true[:10] = np.random.randn(10)  # Sparse signal
b = A @ x_true + 0.1 * np.random.randn(100)

lambda_ = 0.1
x_recovered = ista(A, b, lambda_)
print(f"Sparsity: {np.sum(np.abs(x_recovered) > 1e-3)}")
```

---

## 🚀 FISTA (Fast ISTA)

### Acceleration with Nesterov Momentum

```python
def fista(A, b, lambda_, max_iter=1000):
    """
    Fast ISTA (FISTA)
    """
    m, n = A.shape
    x = np.zeros(n)
    y = x.copy()
    t_old = 1
    
    L = np.linalg.norm(A.T @ A, 2)
    step = 1 / L
    
    for k in range(max_iter):
        # Gradient step on y
        grad = A.T @ (A @ y - b)
        x_new = prox_l1(y - step * grad, step * lambda_)
        
        # Momentum update
        t_new = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
        y = x_new + ((t_old - 1) / t_new) * (x_new - x)
        
        if np.linalg.norm(x_new - x) < 1e-6:
            break
        
        x = x_new
        t_old = t_new
    
    return x
```

---

## 🎯 ADMM (Alternating Direction Method of Multipliers)

### Problem
```
minimize f(x) + g(z)
subject to Ax + Bz = c
```

### Augmented Lagrangian
```
L_ρ(x, z, u) = f(x) + g(z) + uᵀ(Ax + Bz - c) + (ρ/2)||Ax + Bz - c||²
```

### Algorithm
```
x^{k+1} = argmin_x L_ρ(x, z^k, u^k)
z^{k+1} = argmin_z L_ρ(x^{k+1}, z, u^k)
u^{k+1} = u^k + ρ(Ax^{k+1} + Bz^{k+1} - c)
```

```python
def admm_lasso(A, b, lambda_, rho=1.0, max_iter=100):
    """
    ADMM for Lasso:
    minimize ½||Ax - b||² + λ||x||₁
    
    Reformulation:
    minimize ½||Ax - b||² + λ||z||₁
    subject to x = z
    """
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    
    # Precompute
    ATA = A.T @ A
    ATb = A.T @ b
    I = np.eye(n)
    
    for k in range(max_iter):
        # x-update (quadratic, closed form)
        x = np.linalg.solve(ATA + rho * I, ATb + rho * (z - u))
        
        # z-update (soft thresholding)
        z = prox_l1(x + u, lambda_ / rho)
        
        # u-update (dual variable)
        u = u + x - z
        
        # Check convergence
        primal_res = np.linalg.norm(x - z)
        if primal_res < 1e-4:
            break
    
    return z

# Example
x_admm = admm_lasso(A, b, lambda_)
print(f"ADMM sparsity: {np.sum(np.abs(x_admm) > 1e-3)}")
```

---

## 📈 Applications

### 1. Compressed Sensing
```python
# Recover sparse signal from few measurements
# minimize ||x||₁ subject to Ax = b
```

### 2. Total Variation Denoising
```python
# minimize ½||x - b||² + λ·TV(x)
# TV(x) = Σᵢ |xᵢ₊₁ - xᵢ|
```

### 3. Matrix Completion
```python
# minimize ||X||_* subject to P_Ω(X) = P_Ω(M)
# ||X||_*: nuclear norm (sum of singular values)
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is proximal operator?**
   - Generalization of projection
   - argmin of f(x) + ½||x-v||²
   - Handles non-smooth functions

2. **ISTA vs FISTA?**
   - ISTA: O(1/k) convergence
   - FISTA: O(1/k²) with momentum
   - FISTA much faster

3. **When to use ADMM?**
   - Separable objectives
   - Constraints
   - Distributed optimization

4. **Soft thresholding?**
   - Prox of L1 norm
   - Induces sparsity
   - Used in Lasso

---

## 📚 References

- **Books:**
  - "Proximal Algorithms" - Parikh & Boyd
  - "Convex Optimization" - Boyd & Vandenberghe

- **Papers:**
  - "Distributed Optimization and Statistical Learning via ADMM" - Boyd et al.

---

**Proximal methods: optimization for the non-smooth world!**
