# Line Search and Trust Region

> **Step size selection** - Ensuring descent and convergence

---

## 🎯 Line Search

### Exact Line Search
```
α* = argmin_α f(x + αp)
```

**Too expensive in practice!**

### Backtracking Line Search

```python
def backtracking_line_search(f, grad_f, x, p, alpha=1.0, rho=0.5, c=1e-4):
    """Armijo backtracking line search"""
    f_x = f(x)
    grad_x = grad_f(x)
    
    while f(x + alpha * p) > f_x + c * alpha * grad_x.T @ p:
        alpha *= rho
    
    return alpha
```

### Wolfe Conditions

**Sufficient decrease (Armijo):**
```
f(x + αp) ≤ f(x) + c₁α∇f(x)ᵀp
```

**Curvature condition:**
```
∇f(x + αp)ᵀp ≥ c₂∇f(x)ᵀp
```

---

## 🎯 Trust Region

### Idea
Solve subproblem in trusted region

```
minimize    m(p) = f(x) + ∇f(x)ᵀp + ½pᵀHp
subject to  ||p|| ≤ Δ

Δ: trust region radius
```

### Algorithm
```python
def trust_region(f, grad_f, hess_f, x0, Delta=1.0, max_iter=100):
    """Trust region method"""
    x = x0.copy()
    
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Solve trust region subproblem
        p = solve_trust_region_subproblem(grad, hess, Delta)
        
        # Actual vs predicted reduction
        actual_red = f(x) - f(x + p)
        pred_red = -(grad.T @ p + 0.5 * p.T @ hess @ p)
        rho = actual_red / pred_red
        
        # Update trust region
        if rho < 0.25:
            Delta *= 0.25
        elif rho > 0.75 and np.linalg.norm(p) == Delta:
            Delta *= 2
        
        # Accept step if rho > threshold
        if rho > 0.1:
            x = x + p
    
    return x
```

---

## 📚 References

- **Books:** "Numerical Optimization" - Nocedal & Wright

---

**Line search and trust region: ensuring progress!**
