# Constrained Optimization

> **Optimization with constraints** - Lagrange multipliers and KKT conditions

---

## 🎯 Problem Formulation

### General Form
```
minimize    f(x)
subject to  gᵢ(x) ≤ 0,  i = 1,...,m  (inequality)
            hⱼ(x) = 0,  j = 1,...,p  (equality)
```

---

## 📊 Lagrange Multipliers

### Equality Constraints Only
```
minimize    f(x)
subject to  h(x) = 0
```

**Lagrangian:**
```
L(x, ν) = f(x) + νh(x)
```

**Optimality:** ∇ₓL = 0 and h(x) = 0

### Example
```python
# Minimize x² + y² subject to x + y = 1
# L = x² + y² + ν(x + y - 1)
# ∂L/∂x = 2x + ν = 0
# ∂L/∂y = 2y + ν = 0
# x + y = 1
# Solution: x = y = 0.5
```

---

## 🎯 KKT Conditions

### For General Constrained Problem

**Necessary conditions (if constraint qualification holds):**

1. **Stationarity:**
   ```
   ∇f(x*) + Σλᵢ∇gᵢ(x*) + Σνⱼ∇hⱼ(x*) = 0
   ```

2. **Primal feasibility:**
   ```
   gᵢ(x*) ≤ 0,  hⱼ(x*) = 0
   ```

3. **Dual feasibility:**
   ```
   λᵢ ≥ 0
   ```

4. **Complementary slackness:**
   ```
   λᵢgᵢ(x*) = 0
   ```

**For convex problems:** KKT conditions are sufficient!

---

## 💻 Practical Methods

### 1. Penalty Method
```python
def penalty_method(f, g, x0, rho=1.0, max_iter=100):
    """Quadratic penalty method"""
    x = x0.copy()
    
    for k in range(max_iter):
        # Penalized objective
        def f_penalty(x):
            penalty = sum(max(0, gi(x))**2 for gi in g)
            return f(x) + rho * penalty
        
        # Minimize unconstrained problem
        x = minimize(f_penalty, x).x
        
        # Increase penalty
        rho *= 2
    
    return x
```

### 2. Augmented Lagrangian
```python
def augmented_lagrangian(f, g, h, x0, max_iter=100):
    """Augmented Lagrangian method"""
    x = x0.copy()
    lambda_g = np.zeros(len(g))
    nu_h = np.zeros(len(h))
    rho = 1.0
    
    for k in range(max_iter):
        # Augmented Lagrangian
        def L_aug(x):
            L = f(x)
            # Inequality constraints
            for i, gi in enumerate(g):
                L += lambda_g[i] * gi(x) + rho/2 * max(0, gi(x))**2
            # Equality constraints
            for j, hj in enumerate(h):
                L += nu_h[j] * hj(x) + rho/2 * hj(x)**2
            return L
        
        # Minimize
        x = minimize(L_aug, x).x
        
        # Update multipliers
        for i, gi in enumerate(g):
            lambda_g[i] = max(0, lambda_g[i] + rho * gi(x))
        for j, hj in enumerate(h):
            nu_h[j] += rho * hj(x)
    
    return x
```

---

## 📚 References

- **Books:** "Numerical Optimization" - Nocedal & Wright

---

**Constrained optimization: real-world problems have constraints!**
