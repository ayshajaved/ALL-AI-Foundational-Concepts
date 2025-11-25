# Convex Optimization

> **The foundation of tractable optimization** - When global optimum is guaranteed

---

## 🎯 Convex Sets

### Definition
A set C is **convex** if for any x, y ∈ C and λ ∈ [0,1]:
```
λx + (1-λ)y ∈ C
```

**Intuition:** Line segment between any two points stays in set

### Examples
- **Convex:** Hyperplanes, halfspaces, balls, ellipsoids
- **Not convex:** Union of disjoint sets, non-convex polygons

---

## 📈 Convex Functions

### Definition
f is **convex** if for all x, y and λ ∈ [0,1]:
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
```

### First-Order Condition
f is convex iff:
```
f(y) ≥ f(x) + ∇f(x)ᵀ(y - x)
```

### Second-Order Condition
f is convex iff Hessian is positive semidefinite:
```
∇²f(x) ⪰ 0  for all x
```

### Examples
```python
# Convex functions
f(x) = x²                    # Quadratic
f(x) = eˣ                    # Exponential
f(x) = -log(x)              # Negative log
f(x) = ||x||                # Norm
f(x) = max(x₁, x₂, ..., xₙ) # Max

# Not convex
f(x) = x³                    # Cubic
f(x) = sin(x)               # Trigonometric
```

---

## 🎯 Convex Optimization Problem

### Standard Form
```
minimize    f(x)
subject to  gᵢ(x) ≤ 0,  i = 1,...,m
            hⱼ(x) = 0,  j = 1,...,p

where f, gᵢ are convex, hⱼ are affine
```

### Key Property
**Any local minimum is a global minimum!**

---

## 📊 Common Convex Problems

### 1. Linear Programming (LP)
```
minimize    cᵀx
subject to  Ax ≤ b
```

### 2. Quadratic Programming (QP)
```
minimize    ½xᵀQx + cᵀx
subject to  Ax ≤ b

where Q ⪰ 0
```

### 3. Least Squares
```
minimize    ||Ax - b||²
```

**Closed-form solution:**
```
x* = (AᵀA)⁻¹Aᵀb
```

### 4. Ridge Regression
```
minimize    ||Ax - b||² + λ||x||²
```

**Solution:**
```
x* = (AᵀA + λI)⁻¹Aᵀb
```

### 5. Lasso (L1 Regularization)
```
minimize    ||Ax - b||² + λ||x||₁
```

**No closed form, use iterative methods**

---

## 🎯 Optimality Conditions

### Unconstrained
**Necessary:** ∇f(x*) = 0
**Sufficient (convex):** ∇f(x*) = 0

### Constrained (KKT Conditions)
For convex problem, x* is optimal iff:
```
1. Stationarity: ∇f(x*) + Σλᵢ∇gᵢ(x*) + Σνⱼ∇hⱼ(x*) = 0
2. Primal feasibility: gᵢ(x*) ≤ 0, hⱼ(x*) = 0
3. Dual feasibility: λᵢ ≥ 0
4. Complementary slackness: λᵢgᵢ(x*) = 0
```

---

## 💻 Practical Implementation

```python
import numpy as np
from scipy.optimize import minimize

# Convex quadratic function
def f(x):
    return 0.5 * x.T @ Q @ x + c.T @ x

def grad_f(x):
    return Q @ x + c

# Example: minimize ½xᵀQx + cᵀx
Q = np.array([[2, 0], [0, 2]])  # Positive definite
c = np.array([1, 1])

# Using scipy
result = minimize(f, x0=np.zeros(2), jac=grad_f, method='BFGS')
print(f"Optimal x: {result.x}")

# Analytical solution
x_opt = -np.linalg.solve(Q, c)
print(f"Analytical: {x_opt}")
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is a convex function?**
   - f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
   - Any local min is global min

2. **Why convexity matters?**
   - Guaranteed global optimum
   - Efficient algorithms
   - Tractable analysis

3. **Is neural network training convex?**
   - No! Non-convex due to composition
   - Multiple local minima
   - No global optimum guarantee

4. **Convex relaxation?**
   - Approximate non-convex with convex
   - Get lower bound
   - Used in combinatorial optimization

---

## 📚 References

- **Books:** "Convex Optimization" - Boyd & Vandenberghe

---

**Convex optimization: when we can find the best solution!**
