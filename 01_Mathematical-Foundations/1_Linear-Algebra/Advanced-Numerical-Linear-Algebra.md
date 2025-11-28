# Advanced Numerical Linear Algebra

> **Efficient large-scale computations** - Iterative methods, randomized algorithms, and sparse techniques

---

## 🎯 Iterative Methods for Linear Systems

### Problem
```
Solve: Ax = b

For large sparse A (n × n), direct methods (Gaussian elimination) are O(n³)
Iterative methods: O(kn²) or better, where k << n
```

---

## 📊 Conjugate Gradient Method

### For Symmetric Positive Definite (SPD) Systems

```
Minimize: f(x) = ½xᵀAx - bᵀx

Equivalent to solving Ax = b
```

### Algorithm

```python
def conjugate_gradient(A, b, x0=None, max_iter=None, tol=1e-6):
    """
    Conjugate Gradient for Ax = b (A must be SPD)
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    if max_iter is None:
        max_iter = n
    
    r = b - A @ x  # Residual
    p = r.copy()   # Search direction
    rsold = r @ r
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @ r
        
        if np.sqrt(rsnew) < tol:
            print(f"Converged in {i+1} iterations")
            break
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x

# Example
n = 1000
A = np.random.randn(n, n)
A = A.T @ A + n * np.eye(n)  # Make SPD
b = np.random.randn(n)

x_cg = conjugate_gradient(A, b)
print(f"Residual: {np.linalg.norm(A @ x_cg - b):.2e}")
```

**Convergence:** O(√κ) iterations, where κ = cond(A)

---

## 🎯 GMRES (Generalized Minimal Residual)

### For General (Non-Symmetric) Systems

```python
from scipy.sparse.linalg import gmres

# Solve Ax = b
x, info = gmres(A, b, tol=1e-6, maxiter=100)

if info == 0:
    print("Converged!")
elif info > 0:
    print(f"Did not converge in {info} iterations")
```

---

## 📈 Krylov Subspace Methods

### Krylov Subspace
```
Kₖ(A, b) = span{b, Ab, A²b, ..., Aᵏ⁻¹b}

CG and GMRES search for solution in Kₖ
```

### Arnoldi Iteration

```python
def arnoldi(A, b, k):
    """
    Arnoldi iteration to build orthonormal basis for Kₖ(A, b)
    """
    n = len(b)
    Q = np.zeros((n, k+1))
    H = np.zeros((k+1, k))
    
    Q[:, 0] = b / np.linalg.norm(b)
    
    for j in range(k):
        v = A @ Q[:, j]
        
        # Gram-Schmidt orthogonalization
        for i in range(j+1):
            H[i, j] = Q[:, i] @ v
            v = v - H[i, j] * Q[:, i]
        
        H[j+1, j] = np.linalg.norm(v)
        if H[j+1, j] > 1e-12:
            Q[:, j+1] = v / H[j+1, j]
    
    return Q[:, :k], H[:k, :k]
```

---

## 🚀 Randomized Linear Algebra

### 1. Randomized SVD

**Idea:** Approximate SVD using random projections

```python
def randomized_svd(A, k, p=10):
    """
    Randomized SVD
    
    A: m × n matrix
    k: target rank
    p: oversampling parameter
    """
    m, n = A.shape
    
    # Random projection
    Omega = np.random.randn(n, k + p)
    Y = A @ Omega
    
    # QR decomposition
    Q, _ = np.linalg.qr(Y)
    
    # Project A onto Q
    B = Q.T @ A
    
    # SVD of smaller matrix
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Recover U
    U = Q @ U_tilde
    
    return U[:, :k], S[:k], Vt[:k, :]

# Example
m, n = 1000, 500
A = np.random.randn(m, n)

# Randomized SVD
U_rand, S_rand, Vt_rand = randomized_svd(A, k=50)

# Exact SVD
U_exact, S_exact, Vt_exact = np.linalg.svd(A, full_matrices=False)

# Compare singular values
print(f"Error in singular values: {np.linalg.norm(S_rand - S_exact[:50]):.2e}")
```

**Complexity:** O(mn log k) vs O(mn²) for exact SVD

---

### 2. Nyström Approximation

**For kernel matrices K (n × n, SPD)**

```python
def nystrom_approximation(K, k):
    """
    Nyström approximation of kernel matrix
    
    K: n × n kernel matrix
    k: number of samples
    """
    n = K.shape[0]
    
    # Random sample
    idx = np.random.choice(n, k, replace=False)
    
    # Submatrices
    C = K[:, idx]  # n × k
    W = K[idx][:, idx]  # k × k
    
    # Eigendecomposition of W
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    
    # Pseudo-inverse
    W_inv = eigenvectors @ np.diag(1 / (eigenvalues + 1e-10)) @ eigenvectors.T
    
    # Approximation
    K_approx = C @ W_inv @ C.T
    
    return K_approx
```

---

## 📊 Sparse Matrix Techniques

### Sparse Matrix Storage

```python
from scipy.sparse import csr_matrix, csc_matrix

# Create sparse matrix
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])

A_sparse = csr_matrix((data, (row, col)), shape=(3, 3))

print(f"Density: {A_sparse.nnz / (3 * 3):.2%}")
```

### Sparse Linear Solvers

```python
from scipy.sparse.linalg import spsolve, cg

# Direct solver (for sparse A)
x = spsolve(A_sparse, b)

# Iterative solver
x, info = cg(A_sparse, b)
```

---

## 🎯 Preconditioning

### Idea
```
Instead of Ax = b, solve:
M⁻¹Ax = M⁻¹b

M: preconditioner (easy to invert, M ≈ A)
```

### Incomplete Cholesky

```python
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import LinearOperator

# ILU preconditioner
ilu = spilu(A_sparse.tocsc())

# Create linear operator
M = LinearOperator(A_sparse.shape, ilu.solve)

# Solve with preconditioning
x, info = cg(A_sparse, b, M=M)
```

---

## 🎓 Interview Focus

### Key Questions

1. **CG vs direct methods?**
   - CG: O(kn²) for sparse, k << n
   - Direct: O(n³)
   - CG for large sparse SPD systems

2. **Why randomized SVD?**
   - Much faster: O(mn log k)
   - Accurate for low-rank approximation
   - Used in large-scale ML

3. **Krylov subspace?**
   - span{b, Ab, A²b, ...}
   - CG, GMRES search here
   - Dimension grows with iterations

4. **Preconditioning purpose?**
   - Improve condition number
   - Faster convergence
   - M ≈ A but easy to invert

5. **Sparse vs dense?**
   - Sparse: store only non-zeros
   - Much less memory
   - Faster operations for sparse

---

## 📚 References

- **Books:**
  - "Numerical Linear Algebra" - Trefethen & Bau
  - "Templates for the Solution of Linear Systems" - Barrett et al.

- **Papers:**
  - "Finding Structure with Randomness" - Halko et al.

---

**Advanced numerical LA: making large-scale computation practical!**
