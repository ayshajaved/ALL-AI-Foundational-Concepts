# Tensor Calculus and Einstein Notation

> **Calculus on tensors** - Derivatives of tensor operations for deep learning

---

## 🎯 Einstein Summation Convention (Review)

### Notation
```
Repeated indices are summed:
yᵢ = Σⱼ Aᵢⱼxⱼ  →  yᵢ = Aᵢⱼxⱼ

Free indices appear on both sides
Dummy indices (summed) appear only on right
```

---

## 📊 Tensor Derivatives

### Scalar-by-Vector
```
∂f/∂x = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

Example: f(x) = xᵀAx
∂f/∂x = (A + Aᵀ)x
```

### Vector-by-Scalar
```
∂y/∂x where y ∈ ℝᵐ, x ∈ ℝ
Result: m × 1 vector
```

### Vector-by-Vector (Jacobian)
```
∂y/∂x where y ∈ ℝᵐ, x ∈ ℝⁿ

J = [∂yᵢ/∂xⱼ]  (m × n matrix)
```

```python
import torch

# Automatic differentiation
x = torch.randn(5, requires_grad=True)
y = x ** 2 + 2 * x

# Jacobian
J = torch.autograd.functional.jacobian(lambda x: x**2 + 2*x, x)
print(f"Jacobian shape: {J.shape}")  # (5, 5)
```

---

## 🎯 Matrix Calculus

### Scalar-by-Matrix
```
∂f/∂A where f: ℝᵐˣⁿ → ℝ

Result: m × n matrix

Example: f(A) = tr(A)
∂f/∂A = I
```

### Common Derivatives

```python
# 1. Trace
# f(A) = tr(A) = Σᵢ Aᵢᵢ
# ∂f/∂A = I

# 2. Frobenius norm
# f(A) = ||A||²_F = tr(AᵀA)
# ∂f/∂A = 2A

# 3. Determinant
# f(A) = det(A)
# ∂f/∂A = det(A)·A⁻ᵀ

# 4. Inverse
# f(A) = A⁻¹
# ∂vec(A⁻¹)/∂vec(A) = -(A⁻ᵀ ⊗ A⁻¹)
```

---

## 📈 Tensor-by-Tensor Derivatives

### General Form
```
∂Y/∂X where Y, X are tensors

Result: higher-order tensor
```

### Example: Matrix-by-Matrix

```python
def matrix_by_matrix_derivative():
    """
    Example: Y = AXB
    ∂Y/∂X in tensor form
    """
    # Using einsum notation
    # ∂Yᵢⱼ/∂Xₖₗ = AᵢₖBₗⱼ
    
    A = torch.randn(3, 4)
    X = torch.randn(4, 5, requires_grad=True)
    B = torch.randn(5, 6)
    
    Y = A @ X @ B
    
    # Compute full derivative tensor
    # Shape: (3, 6, 4, 5)
    dY_dX = torch.zeros(3, 6, 4, 5)
    
    for i in range(3):
        for j in range(6):
            for k in range(4):
                for l in range(5):
                    dY_dX[i, j, k, l] = A[i, k] * B[l, j]
    
    return dY_dX
```

---

## 🎯 Chain Rule for Tensors

### Scalar Chain Rule
```
f(g(x)):
∂f/∂x = (∂f/∂g)(∂g/∂x)
```

### Tensor Chain Rule
```
Y = f(g(X)):
∂Y/∂X = Σ (∂Y/∂g)(∂g/∂X)

Sum over intermediate indices
```

### Example: Backpropagation

```python
# Forward: y = σ(Wx + b)
# Backward: ∂L/∂W

def backprop_example():
    """
    L = loss(y)
    y = σ(z)
    z = Wx + b
    
    ∂L/∂W = ∂L/∂y · ∂y/∂z · ∂z/∂W
           = ∂L/∂y · σ'(z) · xᵀ
    """
    # Dimensions
    batch, in_dim, out_dim = 32, 10, 5
    
    x = torch.randn(batch, in_dim)
    W = torch.randn(out_dim, in_dim, requires_grad=True)
    b = torch.randn(out_dim, requires_grad=True)
    
    # Forward
    z = x @ W.T + b  # (batch, out_dim)
    y = torch.sigmoid(z)
    
    # Loss (example)
    L = y.sum()
    
    # Backward
    L.backward()
    
    print(f"∂L/∂W shape: {W.grad.shape}")  # (out_dim, in_dim)
    
    return W.grad
```

---

## 📊 Useful Identities

### Matrix Derivatives

```python
# 1. ∂(Ax)/∂x = Aᵀ
# 2. ∂(xᵀA)/∂x = A
# 3. ∂(xᵀAx)/∂x = (A + Aᵀ)x
# 4. ∂(xᵀAy)/∂x = Ay
# 5. ∂tr(AB)/∂A = Bᵀ
# 6. ∂tr(ABA^TC)/∂A = CAB + CᵀABᵀ
# 7. ∂log det(A)/∂A = A⁻ᵀ
```

---

## 🎯 Applications in Deep Learning

### 1. Attention Mechanism

```python
# Attention: softmax(QKᵀ/√d)V
# Derivatives needed for backprop

def attention_derivative():
    """
    A = softmax(QKᵀ/√d)
    Y = AV
    
    ∂L/∂Q, ∂L/∂K, ∂L/∂V
    """
    d = 64
    Q = torch.randn(10, d, requires_grad=True)
    K = torch.randn(10, d, requires_grad=True)
    V = torch.randn(10, d, requires_grad=True)
    
    # Forward
    scores = Q @ K.T / np.sqrt(d)
    A = torch.softmax(scores, dim=-1)
    Y = A @ V
    
    # Loss
    L = Y.sum()
    L.backward()
    
    return Q.grad, K.grad, V.grad
```

### 2. Batch Normalization

```python
# y = γ(x - μ)/σ + β
# ∂L/∂x involves ∂μ/∂x and ∂σ/∂x

def batchnorm_derivative():
    """
    Batch normalization gradient
    """
    batch, dim = 32, 10
    x = torch.randn(batch, dim, requires_grad=True)
    gamma = torch.ones(dim, requires_grad=True)
    beta = torch.zeros(dim, requires_grad=True)
    
    # Forward
    mu = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    x_norm = (x - mu) / torch.sqrt(var + 1e-5)
    y = gamma * x_norm + beta
    
    # Loss
    L = y.sum()
    L.backward()
    
    return x.grad
```

---

## 🎓 Interview Focus

### Key Questions

1. **Einstein notation benefits?**
   - Concise tensor operations
   - Automatic summation
   - Used in einsum

2. **Jacobian vs Hessian?**
   - Jacobian: vector-by-vector (1st order)
   - Hessian: scalar-by-vector twice (2nd order)

3. **Chain rule in backprop?**
   - Multiply Jacobians
   - Efficient with reverse mode
   - O(n) for n parameters

4. **Why matrix calculus?**
   - Derive gradient formulas
   - Understand backprop
   - Optimize implementations

---

## 📚 References

- **Books:**
  - "The Matrix Cookbook" - Petersen & Pedersen
  - "Matrix Differential Calculus with Applications" - Magnus & Neudecker

---

**Tensor calculus: the language of deep learning gradients!**
