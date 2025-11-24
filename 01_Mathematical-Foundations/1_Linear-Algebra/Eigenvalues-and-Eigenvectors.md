# Eigenvalues and Eigenvectors

> **Core concept in linear algebra** - Understanding how matrices transform space

---

## 🎯 Definition

For a square matrix A, a non-zero vector **v** is an **eigenvector** with corresponding **eigenvalue** λ if:

```
Av = λv
```

**Interpretation:** Matrix A stretches/shrinks vector v by factor λ, without changing its direction.

---

## 📐 Mathematical Foundation

### Characteristic Equation

To find eigenvalues, solve:
```
det(A - λI) = 0
```

This gives the **characteristic polynomial**.

### Example: 2×2 Matrix

```
A = [a  b]
    [c  d]

det(A - λI) = det([a-λ   b  ]) = 0
                  [ c   d-λ]

(a-λ)(d-λ) - bc = 0
λ² - (a+d)λ + (ad-bc) = 0
λ² - tr(A)λ + det(A) = 0
```

**Solutions:**
```
λ = [tr(A) ± √(tr(A)² - 4det(A))] / 2
```

### Finding Eigenvectors

Once λ is known, solve:
```
(A - λI)v = 0
```

This is a homogeneous system. Find null space of (A - λI).

---

## 🔢 Properties

### Eigenvalue Properties

1. **Sum of eigenvalues = trace**
   ```
   Σλᵢ = tr(A)
   ```

2. **Product of eigenvalues = determinant**
   ```
   Πλᵢ = det(A)
   ```

3. **Eigenvalues of Aⁿ**
   ```
   If Av = λv, then Aⁿv = λⁿv
   ```

4. **Eigenvalues of A⁻¹**
   ```
   Eigenvalues of A⁻¹ are 1/λᵢ
   ```

5. **Eigenvalues of Aᵀ**
   ```
   Same as eigenvalues of A
   ```

### Special Cases

**Symmetric Matrix (A = Aᵀ):**
- All eigenvalues are **real**
- Eigenvectors are **orthogonal**
- Can be diagonalized: A = QΛQᵀ (Q orthogonal)

**Positive Definite Matrix:**
- All eigenvalues > 0
- Common in covariance matrices

**Orthogonal Matrix (QᵀQ = I):**
- All eigenvalues have |λ| = 1
- Preserves lengths

**Diagonal Matrix:**
- Diagonal elements are eigenvalues
- Standard basis vectors are eigenvectors

---

## 🎨 Eigendecomposition

For diagonalizable matrix A:
```
A = PΛP⁻¹

P = [v₁ v₂ ... vₙ]  (eigenvectors as columns)
Λ = diag(λ₁, λ₂, ..., λₙ)  (eigenvalues on diagonal)
```

### When is A Diagonalizable?

- A has n linearly independent eigenvectors
- Always true for symmetric matrices
- Always true for distinct eigenvalues

### Power of Eigendecomposition

```
Aⁿ = PΛⁿP⁻¹

Λⁿ = diag(λ₁ⁿ, λ₂ⁿ, ..., λₙⁿ)
```

Very efficient for computing matrix powers!

---

## 🔧 Spectral Theorem

For **symmetric matrix** A:
```
A = QΛQᵀ

Q: orthogonal matrix (eigenvectors)
Λ: diagonal matrix (eigenvalues)
QᵀQ = QQᵀ = I
```

**Implications:**
- Can always diagonalize symmetric matrices
- Eigenvectors form orthonormal basis
- Fundamental for PCA, spectral clustering

---

## 💻 Practical Workflows

### NumPy Implementation

```python
import numpy as np

# Create matrix
A = np.array([[4, 2],
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
# [5. 2.]

print(f"Eigenvectors:\n{eigenvectors}")
# Each column is an eigenvector

# Verify Av = λv
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
print(f"Av: {A @ v1}")
print(f"λv: {lambda1 * v1}")
# Should be equal!

# Eigendecomposition
Lambda = np.diag(eigenvalues)
P = eigenvectors

# Reconstruct A
A_reconstructed = P @ Lambda @ np.linalg.inv(P)
print(f"Reconstructed A:\n{A_reconstructed}")
# Should equal original A

# For symmetric matrices
A_sym = np.array([[2, 1],
                  [1, 2]])
eigenvalues_sym, eigenvectors_sym = np.linalg.eigh(A_sym)
# eigh() is optimized for symmetric/Hermitian matrices

# Check orthogonality of eigenvectors
Q = eigenvectors_sym
print(f"QᵀQ:\n{Q.T @ Q}")
# Should be identity matrix
```

### Common Patterns

**1. Computing Matrix Powers**
```python
def matrix_power_eigen(A, n):
    """Compute A^n using eigendecomposition"""
    eigenvalues, P = np.linalg.eig(A)
    Lambda = np.diag(eigenvalues)
    Lambda_n = np.diag(eigenvalues ** n)
    return P @ Lambda_n @ np.linalg.inv(P)

# Much faster than repeated multiplication for large n
A_power_100 = matrix_power_eigen(A, 100)
```

**2. Checking Positive Definiteness**
```python
def is_positive_definite(A):
    """Check if matrix is positive definite"""
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > 0)

# Important for optimization, covariance matrices
```

**3. Spectral Decomposition**
```python
def spectral_decomposition(A):
    """Decompose symmetric matrix"""
    eigenvalues, Q = np.linalg.eigh(A)
    Lambda = np.diag(eigenvalues)
    return Q, Lambda

# Verify A = QΛQᵀ
Q, Lambda = spectral_decomposition(A_sym)
A_check = Q @ Lambda @ Q.T
```

---

## 🎯 AI/ML Applications

### 1. Principal Component Analysis (PCA)

```python
# PCA finds eigenvectors of covariance matrix
X = np.random.randn(100, 10)  # 100 samples, 10 features

# Center data
X_centered = X - X.mean(axis=0)

# Covariance matrix
Cov = (X_centered.T @ X_centered) / (len(X) - 1)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(Cov)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Principal components = eigenvectors
# Explained variance = eigenvalues
print(f"Explained variance ratio: {eigenvalues / eigenvalues.sum()}")

# Project to 2D
PC = eigenvectors[:, :2]  # First 2 principal components
X_reduced = X_centered @ PC
```

### 2. PageRank Algorithm

```python
# PageRank finds dominant eigenvector
def pagerank(adjacency_matrix, damping=0.85, max_iter=100):
    """Compute PageRank scores"""
    n = len(adjacency_matrix)
    
    # Transition matrix
    M = adjacency_matrix / adjacency_matrix.sum(axis=0, keepdims=True)
    
    # PageRank matrix
    P = damping * M + (1 - damping) / n * np.ones((n, n))
    
    # Power iteration to find dominant eigenvector
    v = np.ones(n) / n
    for _ in range(max_iter):
        v_new = P @ v
        if np.allclose(v, v_new):
            break
        v = v_new
    
    return v / v.sum()
```

### 3. Spectral Clustering

```python
# Spectral clustering uses eigenvectors of graph Laplacian
from sklearn.cluster import SpectralClustering

# Affinity matrix (similarity between points)
affinity = np.exp(-np.linalg.norm(X[:, None] - X, axis=2) ** 2 / 2)

# Spectral clustering
clustering = SpectralClustering(n_clusters=3, affinity='precomputed')
labels = clustering.fit_predict(affinity)
```

### 4. Stability Analysis

```python
# System stability: all eigenvalues have |λ| < 1
def is_stable(A):
    """Check if dynamical system is stable"""
    eigenvalues = np.linalg.eigvals(A)
    return np.all(np.abs(eigenvalues) < 1)

# Used in RNNs, control theory
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is an eigenvector?**
   - Vector that doesn't change direction under transformation
   - Only scaled by eigenvalue: Av = λv

2. **How do you find eigenvalues?**
   - Solve det(A - λI) = 0
   - Roots of characteristic polynomial

3. **What does eigenvalue = 0 mean?**
   - Matrix is singular (not invertible)
   - det(A) = 0
   - Null space is non-trivial

4. **Why are eigenvalues important in ML?**
   - PCA (dimensionality reduction)
   - Stability analysis
   - Spectral methods
   - Optimization (Hessian eigenvalues)

5. **What's special about symmetric matrices?**
   - Real eigenvalues
   - Orthogonal eigenvectors
   - Always diagonalizable

### Must-Know Formulas

```
Av = λv
det(A - λI) = 0
Σλᵢ = tr(A)
Πλᵢ = det(A)
A = PΛP⁻¹ (eigendecomposition)
A = QΛQᵀ (symmetric case)
```

### Common Pitfalls

- ❌ Forgetting eigenvectors must be non-zero
- ❌ Not normalizing eigenvectors
- ❌ Assuming all matrices are diagonalizable
- ❌ Confusing eigenvalues with singular values

---

## 🔗 Connections

### Prerequisites
- [Matrix Operations](Matrix-Operations.md)
- Determinants

### Related Topics
- [SVD](SVD-and-Matrix-Decomposition.md)
- PCA (in ML section)
- Spectral methods

### Applications in AI
- **PCA:** Dimensionality reduction
- **Spectral Clustering:** Graph-based clustering
- **PageRank:** Web search ranking
- **Stability:** RNN analysis
- **Optimization:** Hessian analysis

---

## 📚 References

- **Books:**
  - "Linear Algebra and Its Applications" - Strang
  - "Matrix Analysis" - Horn & Johnson

- **Online:**
  - [3Blue1Brown: Eigenvectors and Eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g)
  - [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)

- **Papers:**
  - "A Tutorial on Principal Component Analysis" - Shlens

---

**Eigenvalues and eigenvectors are fundamental to understanding how matrices transform space!**
