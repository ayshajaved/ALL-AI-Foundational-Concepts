# Second-Order Methods

> **Using curvature information** - Newton's method and quasi-Newton methods

---

## 🎯 Newton's Method

### Idea
Use second-order Taylor approximation

```
f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + ½ΔxᵀH(x)Δx

Minimize to get: Δx = -H(x)⁻¹∇f(x)
```

### Algorithm
```
x_{t+1} = x_t - H(x_t)⁻¹∇f(x_t)

H(x): Hessian matrix
```

### Implementation
```python
def newtons_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    """Newton's method for optimization"""
    x = x0.copy()
    
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Newton step
        delta_x = -np.linalg.solve(hess, grad)
        x = x + delta_x
        
        if np.linalg.norm(grad) < tol:
            break
    
    return x
```

### Properties
- **Convergence:** Quadratic near optimum
- **Cost:** O(n³) per iteration (Hessian inversion)
- **Requirement:** Hessian must be positive definite

---

## 🚀 Quasi-Newton Methods

### Idea
Approximate Hessian instead of computing it

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

**Most popular quasi-Newton method**

```python
def bfgs(f, grad_f, x0, max_iter=100):
    """BFGS optimization"""
    n = len(x0)
    x = x0.copy()
    B = np.eye(n)  # Initial Hessian approximation
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        # Search direction
        p = -B @ grad
        
        # Line search
        alpha = line_search(f, grad_f, x, p)
        
        # Update
        s = alpha * p
        x_new = x + s
        grad_new = grad_f(x_new)
        y = grad_new - grad
        
        # BFGS update
        rho = 1.0 / (y.T @ s)
        I = np.eye(n)
        B = (I - rho * s @ y.T) @ B @ (I - rho * y @ s.T) + rho * s @ s.T
        
        x = x_new
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return x
```

---

## 📊 Comparison

| Method | Convergence | Cost/iter | Memory |
|--------|-------------|-----------|--------|
| Gradient Descent | Linear | O(n) | O(n) |
| Newton | Quadratic | O(n³) | O(n²) |
| BFGS | Superlinear | O(n²) | O(n²) |
| L-BFGS | Superlinear | O(n) | O(mn) |

---

## 🎓 Interview Focus

1. **Newton vs Gradient Descent?**
   - Newton: faster convergence, expensive per iteration
   - GD: slower convergence, cheap per iteration

2. **Why quasi-Newton?**
   - Avoid Hessian computation
   - Superlinear convergence
   - Practical for medium-scale problems

---

**Second-order methods: faster convergence with curvature!**
