# Matrix Operations

> **Essential matrix manipulations for AI** - Determinants, inverses, rank, and trace

---

## 🔢 Determinant

### Definition
The **determinant** is a scalar value that encodes properties of a square matrix.

**Notation:** det(A) or |A|

### 2×2 Matrix
```
A = [a  b]
    [c  d]

det(A) = ad - bc
```

### 3×3 Matrix (Cofactor Expansion)
```
A = [a  b  c]
    [d  e  f]
    [g  h  i]

det(A) = a(ei-fh) - b(di-fg) + c(dh-eg)
```

### Properties

1. **det(I) = 1**
2. **det(AB) = det(A)det(B)**
3. **det(Aᵀ) = det(A)**
4. **det(A⁻¹) = 1/det(A)**
5. **det(αA) = αⁿdet(A)** (n = matrix size)
6. **Row swap changes sign**
7. **det(A) = 0 ⟺ A is singular (not invertible)**

### Geometric Interpretation
- **2D:** Area of parallelogram formed by column vectors
- **3D:** Volume of parallelepiped
- **Sign:** Orientation (positive = right-handed)

---

## 🔄 Matrix Inverse

### Definition
For square matrix A, the **inverse** A⁻¹ satisfies:
```
AA⁻¹ = A⁻¹A = I
```

### Existence
A⁻¹ exists if and only if:
- A is square
- det(A) ≠ 0 (A is non-singular)

### 2×2 Inverse
```
A = [a  b]      A⁻¹ = 1/(ad-bc) [ d  -b]
    [c  d]                       [-c   a]
```

### Properties

1. **(A⁻¹)⁻¹ = A**
2. **(AB)⁻¹ = B⁻¹A⁻¹** (reverse order!)
3. **(Aᵀ)⁻¹ = (A⁻¹)ᵀ**
4. **det(A⁻¹) = 1/det(A)**

### Computing Inverse

**Method 1: Gauss-Jordan Elimination**
```
[A | I] → [I | A⁻¹]
```

**Method 2: Adjugate Matrix**
```
A⁻¹ = (1/det(A)) × adj(A)
```

**Method 3: LU Decomposition** (efficient for large matrices)

---

## 📊 Rank

### Definition
The **rank** of a matrix is the dimension of the vector space spanned by its columns (or rows).

**Notation:** rank(A)

### Properties

1. **rank(A) ≤ min(m, n)** for m×n matrix
2. **rank(A) = rank(Aᵀ)**
3. **rank(AB) ≤ min(rank(A), rank(B))**
4. **Full rank:** rank(A) = min(m, n)

### Types

**Full Column Rank:**
- rank(A) = n (number of columns)
- Columns are linearly independent
- Aᵀ A is invertible

**Full Row Rank:**
- rank(A) = m (number of rows)
- Rows are linearly independent
- AAᵀ is invertible

**Rank Deficient:**
- rank(A) < min(m, n)
- Columns/rows are linearly dependent

### Computing Rank
- Count non-zero rows in row echelon form
- Count non-zero singular values
- Use SVD

---

## 🎯 Trace

### Definition
The **trace** is the sum of diagonal elements of a square matrix.

```
tr(A) = Σᵢ aᵢᵢ = a₁₁ + a₂₂ + ... + aₙₙ
```

### Properties

1. **tr(A + B) = tr(A) + tr(B)**
2. **tr(αA) = α·tr(A)**
3. **tr(AB) = tr(BA)** (cyclic property)
4. **tr(ABC) = tr(CAB) = tr(BCA)**
5. **tr(Aᵀ) = tr(A)**
6. **tr(A) = sum of eigenvalues**

### Applications in AI

**1. Frobenius Norm**
```
||A||_F = √tr(AᵀA)
```

**2. Regularization**
```
L = Loss + λ·tr(WᵀW)
```

**3. Kernel Methods**
```
K(x,y) = tr(φ(x)φ(y)ᵀ)
```

---

## 🔧 Matrix Decompositions (Overview)

### LU Decomposition
```
A = LU
L: Lower triangular
U: Upper triangular

Used for: Solving linear systems efficiently
```

### QR Decomposition
```
A = QR
Q: Orthogonal matrix
R: Upper triangular

Used for: Least squares, eigenvalue algorithms
```

### Cholesky Decomposition
```
A = LLᵀ  (for positive definite A)
L: Lower triangular

Used for: Efficient solving, numerical stability
```

---

## 💻 Practical Workflows

### NumPy Implementation

```python
import numpy as np

# Create matrix
A = np.array([[1, 2], [3, 4]])

# Determinant
det_A = np.linalg.det(A)
print(f"Determinant: {det_A}")  # -2.0

# Inverse
A_inv = np.linalg.inv(A)
print(f"Inverse:\n{A_inv}")

# Verify: AA⁻¹ = I
identity = A @ A_inv
print(f"AA⁻¹:\n{identity}")

# Rank
rank_A = np.linalg.matrix_rank(A)
print(f"Rank: {rank_A}")  # 2 (full rank)

# Trace
trace_A = np.trace(A)
print(f"Trace: {trace_A}")  # 5

# Check if invertible
is_invertible = det_A != 0
print(f"Invertible: {is_invertible}")

# Solve linear system Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(f"Solution: {x}")  # [1, 2]

# LU decomposition
from scipy.linalg import lu
P, L, U = lu(A)
print(f"L:\n{L}")
print(f"U:\n{U}")

# QR decomposition
Q, R = np.linalg.qr(A)
print(f"Q:\n{Q}")
print(f"R:\n{R}")
```

### Common Patterns

**1. Solving Linear Systems**
```python
# Instead of computing inverse
# DON'T: x = np.linalg.inv(A) @ b  # Slow, numerically unstable
# DO: x = np.linalg.solve(A, b)    # Fast, stable
```

**2. Checking Invertibility**
```python
def is_invertible(A):
    return np.linalg.det(A) != 0  # or check rank

# Better: use condition number
cond = np.linalg.cond(A)
if cond < 1e10:  # Well-conditioned
    A_inv = np.linalg.inv(A)
```

**3. Pseudo-Inverse (for non-square matrices)**
```python
# Moore-Penrose pseudo-inverse
A_pinv = np.linalg.pinv(A)
# Useful when A is not invertible
```

---

## 🎯 AI/ML Applications

### 1. Solving Normal Equations
```python
# Linear regression: β = (XᵀX)⁻¹Xᵀy
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Method 1: Using inverse (not recommended)
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# Method 2: Using solve (better)
beta = np.linalg.solve(X.T @ X, X.T @ y)

# Method 3: Using lstsq (best)
beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
```

### 2. Covariance Matrix
```python
# Covariance is symmetric, positive semi-definite
X = np.random.randn(100, 10)
X_centered = X - X.mean(axis=0)
Cov = (X_centered.T @ X_centered) / (len(X) - 1)

# Properties
assert np.allclose(Cov, Cov.T)  # Symmetric
eigenvalues = np.linalg.eigvals(Cov)
assert np.all(eigenvalues >= -1e-10)  # Non-negative eigenvalues
```

### 3. Regularization
```python
# Ridge regression: β = (XᵀX + λI)⁻¹Xᵀy
lambda_reg = 0.1
I = np.eye(X.shape[1])
beta_ridge = np.linalg.solve(X.T @ X + lambda_reg * I, X.T @ y)
```

---

## 🎓 Interview Focus

### Key Questions

1. **When is a matrix invertible?**
   - Square matrix
   - det(A) ≠ 0
   - Full rank
   - All eigenvalues non-zero

2. **What does determinant = 0 mean?**
   - Matrix is singular (not invertible)
   - Columns are linearly dependent
   - Transforms space to lower dimension

3. **Why not compute inverse directly?**
   - Numerically unstable
   - Expensive (O(n³))
   - Use `solve()` instead

4. **What is the difference between rank and dimension?**
   - Rank: dimension of column/row space
   - Dimension: size of matrix (m×n)

5. **What does trace represent?**
   - Sum of diagonal elements
   - Sum of eigenvalues
   - Invariant under similarity transformations

### Must-Know Formulas

```
det(AB) = det(A)det(B)
(AB)⁻¹ = B⁻¹A⁻¹
rank(AB) ≤ min(rank(A), rank(B))
tr(AB) = tr(BA)
```

### Common Pitfalls

- ❌ Computing inverse when not needed
- ❌ Not checking if matrix is invertible
- ❌ Forgetting (AB)⁻¹ = B⁻¹A⁻¹ (reverse order)
- ❌ Assuming det(A+B) = det(A) + det(B) (FALSE!)

---

## 🔗 Connections

### Prerequisites
- [Vectors and Matrices](Vectors-and-Matrices.md)

### Related Topics
- [Eigenvalues and Eigenvectors](Eigenvalues-and-Eigenvectors.md)
- [SVD](SVD-and-Matrix-Decomposition.md)
- Linear systems solving

### Applications in AI
- Linear regression (normal equations)
- Covariance matrices
- Neural network weight updates
- Dimensionality reduction

---

## 📚 References

- **Books:**
  - "Linear Algebra and Its Applications" - Gilbert Strang
  - "Matrix Computations" - Golub & Van Loan

- **Online:**
  - [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
  - [SciPy Linear Algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html)

---

**Master these operations - they power all of linear algebra!**
