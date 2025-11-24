# Practical Workflows - Linear Algebra

> **Hands-on guide to linear algebra in Python** - NumPy, SciPy, and real-world patterns

---

## 🛠️ Essential Libraries

### NumPy - Core Numerical Computing

```python
import numpy as np

# NumPy is the foundation for all numerical computing in Python
# Provides efficient array operations and linear algebra functions
```

### SciPy - Scientific Computing

```python
from scipy import linalg
from scipy.sparse import linalg as sparse_linalg

# SciPy extends NumPy with additional algorithms
# Especially useful for sparse matrices and advanced decompositions
```

---

## 📊 Common Workflows

### 1. Creating Matrices and Vectors

```python
import numpy as np

# Vectors
v = np.array([1, 2, 3])  # 1D array
v_col = np.array([[1], [2], [3]])  # Column vector
v_row = np.array([[1, 2, 3]])  # Row vector

# Matrices
A = np.array([[1, 2], [3, 4]])  # 2×2
B = np.random.randn(3, 4)  # 3×4 random
C = np.zeros((5, 5))  # 5×5 zeros
D = np.ones((2, 3))  # 2×3 ones
I = np.eye(4)  # 4×4 identity

# Special matrices
diag = np.diag([1, 2, 3])  # Diagonal
upper_tri = np.triu(np.ones((3, 3)))  # Upper triangular
lower_tri = np.tril(np.ones((3, 3)))  # Lower triangular

# From list of lists
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Random matrices
uniform = np.random.rand(3, 3)  # Uniform [0, 1)
normal = np.random.randn(3, 3)  # Standard normal
integers = np.random.randint(0, 10, (3, 3))  # Random integers
```

### 2. Basic Operations

```python
# Element-wise operations
A + B  # Addition
A - B  # Subtraction
A * B  # Element-wise multiplication (Hadamard product)
A / B  # Element-wise division
A ** 2  # Element-wise power

# Matrix operations
A @ B  # Matrix multiplication (Python 3.5+)
np.dot(A, B)  # Matrix multiplication (older)
A.T  # Transpose
np.linalg.inv(A)  # Inverse
np.linalg.det(A)  # Determinant
np.trace(A)  # Trace

# Norms
np.linalg.norm(v)  # L2 norm (default)
np.linalg.norm(v, 1)  # L1 norm
np.linalg.norm(v, np.inf)  # L-infinity norm
np.linalg.norm(A, 'fro')  # Frobenius norm
```

### 3. Solving Linear Systems

```python
# Solve Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# Method 1: Direct solve (BEST)
x = np.linalg.solve(A, b)

# Method 2: Using inverse (DON'T DO THIS)
x = np.linalg.inv(A) @ b  # Slower, less accurate

# Method 3: Least squares (for overdetermined systems)
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

# For multiple right-hand sides
B = np.array([[9, 1], [8, 2]])
X = np.linalg.solve(A, B)  # Solves AX = B

# Check solution
print(f"Residual: {np.linalg.norm(A @ x - b)}")
```

### 4. Eigenvalues and Eigenvectors

```python
# General matrix
A = np.array([[1, 2], [2, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)

# Symmetric matrix (faster, more accurate)
A_sym = (A + A.T) / 2  # Ensure symmetry
eigenvalues, eigenvectors = np.linalg.eigh(A_sym)

# Only eigenvalues (faster)
eigenvalues = np.linalg.eigvals(A)

# Verify Av = λv
v = eigenvectors[:, 0]
lambda_val = eigenvalues[0]
assert np.allclose(A @ v, lambda_val * v)

# Sort by eigenvalue magnitude
idx = np.argsort(np.abs(eigenvalues))[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```

### 5. SVD

```python
# Full SVD
U, s, Vt = np.linalg.svd(A, full_matrices=True)

# Compact SVD (more efficient)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Reconstruct matrix
Sigma = np.diag(s)
A_reconstructed = U @ Sigma @ Vt

# Low-rank approximation
k = 5
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Truncated SVD for large sparse matrices
from scipy.sparse.linalg import svds
U_k, s_k, Vt_k = svds(A, k=k)
```

---

## 🎯 ML-Specific Patterns

### 1. Data Preprocessing

```python
# Centering (zero mean)
X_centered = X - X.mean(axis=0)

# Standardization (zero mean, unit variance)
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# Normalization (unit norm)
X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

# Min-Max scaling
X_minmax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
```

### 2. Covariance and Correlation

```python
# Covariance matrix
X_centered = X - X.mean(axis=0)
Cov = (X_centered.T @ X_centered) / (len(X) - 1)
# Or use NumPy
Cov = np.cov(X.T)

# Correlation matrix
Corr = np.corrcoef(X.T)

# Check properties
assert np.allclose(Cov, Cov.T)  # Symmetric
eigenvalues = np.linalg.eigvals(Cov)
assert np.all(eigenvalues >= -1e-10)  # Positive semi-definite
```

### 3. PCA Implementation

```python
def pca(X, n_components):
    """Principal Component Analysis"""
    # Center data
    X_centered = X - X.mean(axis=0)
    
    # Method 1: Via covariance matrix
    Cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(Cov)
    
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    components = eigenvectors[:, :n_components]
    
    # Transform data
    X_pca = X_centered @ components
    
    # Explained variance ratio
    explained_var = eigenvalues / eigenvalues.sum()
    
    return X_pca, components, explained_var

# Method 2: Via SVD (more stable)
def pca_svd(X, n_components):
    """PCA using SVD"""
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    components = Vt[:n_components]
    X_pca = U[:, :n_components] @ np.diag(s[:n_components])
    
    explained_var = (s ** 2) / (len(X) - 1)
    explained_var = explained_var / explained_var.sum()
    
    return X_pca, components, explained_var
```

### 4. Linear Regression

```python
def linear_regression(X, y):
    """Solve linear regression: y = Xβ + ε"""
    # Add bias term
    X_b = np.c_[np.ones(len(X)), X]
    
    # Method 1: Normal equation (for small datasets)
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    # Method 2: Using solve (better)
    beta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
    
    # Method 3: Using lstsq (best, handles rank deficiency)
    beta, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)
    
    return beta

# Ridge regression (with regularization)
def ridge_regression(X, y, alpha=1.0):
    """Ridge regression: β = (XᵀX + αI)⁻¹Xᵀy"""
    X_b = np.c_[np.ones(len(X)), X]
    I = np.eye(X_b.shape[1])
    I[0, 0] = 0  # Don't regularize bias
    
    beta = np.linalg.solve(X_b.T @ X_b + alpha * I, X_b.T @ y)
    return beta
```

### 5. Batch Processing

```python
# Process multiple samples at once
def batch_transform(X, W, b):
    """Apply linear transformation to batch"""
    # X: (batch_size, input_dim)
    # W: (input_dim, output_dim)
    # b: (output_dim,)
    return X @ W + b  # Broadcasting handles bias

# Example: Neural network layer
batch_size = 32
input_dim = 784
output_dim = 128

X = np.random.randn(batch_size, input_dim)
W = np.random.randn(input_dim, output_dim) * 0.01
b = np.zeros(output_dim)

Y = batch_transform(X, W, b)
print(f"Output shape: {Y.shape}")  # (32, 128)
```

---

## ⚡ Performance Tips

### 1. Use Vectorization

```python
# BAD: Loop over elements
result = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        result[i, j] = A[i, j] * B[i, j]

# GOOD: Vectorized
result = A * B  # Much faster!
```

### 2. Avoid Unnecessary Copies

```python
# Creates copy
B = A + 0

# In-place operation (no copy)
A += 1  # Modifies A directly

# View (no copy)
B = A.T  # Just a view, very fast
```

### 3. Use Appropriate Data Types

```python
# Use float32 instead of float64 when possible
A = np.random.randn(1000, 1000).astype(np.float32)

# Use int32 for indices
indices = np.arange(1000, dtype=np.int32)
```

### 4. Preallocate Arrays

```python
# BAD: Growing array
result = []
for i in range(n):
    result.append(compute(i))
result = np.array(result)

# GOOD: Preallocate
result = np.zeros(n)
for i in range(n):
    result[i] = compute(i)
```

### 5. Use BLAS/LAPACK

```python
# NumPy automatically uses optimized BLAS/LAPACK
# For matrix multiplication, use @ or np.dot
C = A @ B  # Uses optimized BLAS

# Check BLAS configuration
np.show_config()
```

---

## 🐛 Common Debugging Patterns

### 1. Check Shapes

```python
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

# Use assertions
assert A.shape[1] == B.shape[0], "Incompatible shapes for multiplication"
```

### 2. Check for NaN/Inf

```python
assert not np.isnan(A).any(), "Matrix contains NaN"
assert not np.isinf(A).any(), "Matrix contains Inf"

# Or use
np.testing.assert_array_equal(A, A)  # Fails if NaN
```

### 3. Numerical Stability

```python
# Check condition number
cond = np.linalg.cond(A)
if cond > 1e10:
    print(f"Warning: Matrix is ill-conditioned (κ={cond:.2e})")

# Check rank
rank = np.linalg.matrix_rank(A)
if rank < min(A.shape):
    print(f"Warning: Matrix is rank deficient (rank={rank})")
```

### 4. Verify Properties

```python
# Check symmetry
assert np.allclose(A, A.T), "Matrix should be symmetric"

# Check orthogonality
assert np.allclose(Q.T @ Q, np.eye(Q.shape[1])), "Q should be orthogonal"

# Check positive definiteness
eigenvalues = np.linalg.eigvals(A)
assert np.all(eigenvalues > 0), "Matrix should be positive definite"
```

---

## 📚 Quick Reference

### NumPy Linear Algebra Functions

```python
# Basic operations
np.dot(A, B)          # Matrix multiplication
A @ B                 # Matrix multiplication (Python 3.5+)
A.T                   # Transpose
np.linalg.inv(A)      # Inverse
np.linalg.pinv(A)     # Pseudo-inverse

# Decompositions
np.linalg.eig(A)      # Eigendecomposition
np.linalg.eigh(A)     # Symmetric eigendecomposition
np.linalg.svd(A)      # SVD
np.linalg.qr(A)       # QR decomposition
np.linalg.cholesky(A) # Cholesky decomposition

# Solving systems
np.linalg.solve(A, b)     # Solve Ax = b
np.linalg.lstsq(A, b)     # Least squares

# Properties
np.linalg.det(A)          # Determinant
np.trace(A)               # Trace
np.linalg.matrix_rank(A)  # Rank
np.linalg.cond(A)         # Condition number

# Norms
np.linalg.norm(v)         # Vector L2 norm
np.linalg.norm(A, 'fro')  # Frobenius norm
np.linalg.norm(A, 2)      # Spectral norm
```

---

## 🔗 Resources

- **Documentation:**
  - [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
  - [SciPy Linear Algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html)

- **Tutorials:**
  - [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
  - [SciPy Tutorial](https://docs.scipy.org/doc/scipy/tutorial/index.html)

---

**Practice these workflows - they're essential for all ML/AI work!**
