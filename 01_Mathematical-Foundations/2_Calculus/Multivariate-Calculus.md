# Multivariate Calculus

> **Essential for machine learning** - Functions of many variables

---

## 📐 Functions of Multiple Variables

### Definition

A function f: ℝⁿ → ℝ maps n-dimensional input to scalar output:

```
f(x₁, x₂, ..., xₙ) = y
```

**Examples:**
- f(x,y) = x² + y² (2D paraboloid)
- f(x,y,z) = x + 2y - 3z (3D plane)
- Loss function: L(w₁, w₂, ..., wₙ)

---

## 🔢 Partial Derivatives

### Definition

Derivative with respect to one variable, holding others constant:

```
∂f/∂xᵢ = lim[h→0] (f(x₁,...,xᵢ+h,...,xₙ) - f(x₁,...,xᵢ,...,xₙ)) / h
```

**Example:**
```
f(x,y) = x²y + 3xy²

∂f/∂x = 2xy + 3y²
∂f/∂y = x² + 6xy
```

### Higher-Order Partial Derivatives

```
∂²f/∂x² = ∂/∂x(∂f/∂x)
∂²f/∂x∂y = ∂/∂x(∂f/∂y)
```

**Clairaut's Theorem:** If f is C², then:
```
∂²f/∂x∂y = ∂²f/∂y∂x
```

---

## 🔺 Gradient Vector

### Definition

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Properties:**
- Points in direction of steepest ascent
- Perpendicular to level sets
- Magnitude = rate of maximum increase

**Example:**
```
f(x,y,z) = x² + 2y² + 3z²

∇f = [2x, 4y, 6z]ᵀ
```

---

## 📊 Hessian Matrix

### Definition

Matrix of second partial derivatives:

```
H(f) = [∂²f/∂xᵢ∂xⱼ]

For f: ℝⁿ → ℝ, H is n×n
```

**Example:**
```
f(x,y) = x² + xy + y²

H = [∂²f/∂x²    ∂²f/∂x∂y]   [2  1]
    [∂²f/∂y∂x   ∂²f/∂y²  ] = [1  2]
```

### Properties

- **Symmetric:** H = Hᵀ (if f is C²)
- **Positive definite** → local minimum
- **Negative definite** → local maximum
- **Indefinite** → saddle point

---

## 🎯 Taylor Series (Multivariate)

### First-Order Approximation

```
f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx
```

### Second-Order Approximation

```
f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + ½ΔxᵀH(x)Δx
```

**Used in:**
- Newton's method
- Second-order optimization
- Analyzing critical points

---

## 🔧 Critical Points

### Definition

Point where ∇f = 0

**Classification using Hessian:**

1. **Local Minimum:** H positive definite (all eigenvalues > 0)
2. **Local Maximum:** H negative definite (all eigenvalues < 0)
3. **Saddle Point:** H indefinite (mixed eigenvalues)

**Example:**
```python
import numpy as np

def classify_critical_point(H):
    """Classify critical point using Hessian"""
    eigenvalues = np.linalg.eigvals(H)
    
    if np.all(eigenvalues > 0):
        return "Local Minimum"
    elif np.all(eigenvalues < 0):
        return "Local Maximum"
    else:
        return "Saddle Point"

# Example
H = np.array([[2, 0], [0, 2]])
print(classify_critical_point(H))  # Local Minimum
```

---

## 💻 Practical Workflows

### Computing Gradients

```python
import numpy as np

def gradient(f, x, h=1e-5):
    """Numerical gradient"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
    return grad

# Example
f = lambda x: x[0]**2 + x[1]**2
x = np.array([1.0, 2.0])
print(f"Gradient: {gradient(f, x)}")  # [2., 4.]
```

### Computing Hessian

```python
def hessian(f, x, h=1e-5):
    """Numerical Hessian"""
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
            
            H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4*h**2)
    
    return H

# Example
H = hessian(f, x)
print(f"Hessian:\n{H}")  # [[2, 0], [0, 2]]
```

### Using JAX (Automatic Differentiation)

```python
import jax
import jax.numpy as jnp

# Define function
def f(x):
    return jnp.sum(x**2)

# Gradient
grad_f = jax.grad(f)
x = jnp.array([1.0, 2.0, 3.0])
print(f"Gradient: {grad_f(x)}")

# Hessian
hess_f = jax.hessian(f)
print(f"Hessian:\n{hess_f(x)}")

# Jacobian (for vector-valued functions)
def g(x):
    return jnp.array([x[0]**2, x[1]**2, x[0]*x[1]])

jac_g = jax.jacobian(g)
print(f"Jacobian:\n{jac_g(x[:2])}")
```

---

## 🎯 AI/ML Applications

### 1. Loss Surface Analysis

```python
# Visualize loss landscape
import matplotlib.pyplot as plt

def loss_landscape(w1_range, w2_range, loss_fn):
    """Plot 2D loss landscape"""
    W1, W2 = np.meshgrid(w1_range, w2_range)
    L = np.zeros_like(W1)
    
    for i in range(len(w1_range)):
        for j in range(len(w2_range)):
            w = np.array([W1[j,i], W2[j,i]])
            L[j,i] = loss_fn(w)
    
    plt.contour(W1, W2, L, levels=20)
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title('Loss Landscape')
    plt.colorbar()
    plt.show()
```

### 2. Newton's Method

```python
def newtons_method(f, grad_f, hess_f, x0, max_iter=100):
    """Newton's method for optimization"""
    x = x0.copy()
    
    for i in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        
        # Newton step: x_new = x - H⁻¹g
        try:
            x = x - np.linalg.solve(H, g)
        except:
            print("Singular Hessian!")
            break
        
        if np.linalg.norm(g) < 1e-6:
            break
    
    return x

# Example: minimize f(x,y) = x² + y²
f = lambda x: np.sum(x**2)
grad_f = lambda x: 2*x
hess_f = lambda x: 2*np.eye(len(x))

x0 = np.array([5.0, 5.0])
x_min = newtons_method(f, grad_f, hess_f, x0)
print(f"Minimum at: {x_min}")
```

### 3. Constrained Optimization (Lagrange Multipliers)

```python
# Minimize f(x,y) subject to g(x,y) = 0
# ∇f = λ∇g at optimum

def lagrange_multiplier(f, grad_f, g, grad_g, x0):
    """Find critical points using Lagrange multipliers"""
    # Solve system: ∇f - λ∇g = 0 and g = 0
    # This is a simplified example
    
    from scipy.optimize import fsolve
    
    def equations(vars):
        x, y, lam = vars
        point = np.array([x, y])
        gf = grad_f(point)
        gg = grad_g(point)
        return [
            gf[0] - lam * gg[0],
            gf[1] - lam * gg[1],
            g(point)
        ]
    
    result = fsolve(equations, [x0[0], x0[1], 1.0])
    return result[:2]  # Return x, y
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is a partial derivative?**
   - Derivative w.r.t. one variable
   - Others held constant
   - Component of gradient

2. **What does gradient point to?**
   - Direction of steepest ascent
   - Perpendicular to level curves
   - Used in gradient descent

3. **What is Hessian used for?**
   - Second-order information
   - Classifying critical points
   - Newton's method

4. **How to find minimum of function?**
   - Set gradient to zero
   - Check Hessian is positive definite
   - Or use iterative methods

5. **What is a saddle point?**
   - Critical point (∇f = 0)
   - Not min or max
   - Hessian has mixed eigenvalues

### Must-Know Formulas

```
Gradient: ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ
Hessian: H = [∂²f/∂xᵢ∂xⱼ]
Taylor: f(x+Δx) ≈ f(x) + ∇fᵀΔx + ½ΔxᵀHΔx
Critical point: ∇f = 0
```

### Common Pitfalls

- ❌ Confusing partial and total derivatives
- ❌ Not checking Hessian for critical points
- ❌ Forgetting Clairaut's theorem
- ❌ Assuming all critical points are minima

---

## 🔗 Connections

### Prerequisites
- [Derivatives and Gradients](Derivatives-and-Gradients.md)
- Linear algebra

### Related Topics
- [Optimization](../4_Optimization/)
- Lagrange multipliers
- Constrained optimization

### Applications in AI
- **Loss Functions:** Multivariate optimization
- **Neural Networks:** High-dimensional gradients
- **Second-Order Methods:** Newton, L-BFGS
- **Regularization:** Constrained optimization

---

## 📚 References

- **Books:**
  - "Calculus" - James Stewart
  - "Vector Calculus" - Marsden & Tromba

- **Online:**
  - [Khan Academy: Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)
  - [3Blue1Brown: Multivariable Calculus](https://www.youtube.com/playlist?list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7)

---

**Multivariate calculus is essential for understanding ML optimization!**
