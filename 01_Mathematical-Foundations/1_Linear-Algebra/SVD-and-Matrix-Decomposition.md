# SVD and Matrix Decomposition

> **The Swiss Army knife of linear algebra** - Singular Value Decomposition and its applications

---

## 🎯 Singular Value Decomposition (SVD)

### Definition

For any matrix A (m×n), SVD decomposes it as:

```
A = UΣVᵀ

U: m×m orthogonal matrix (left singular vectors)
Σ: m×n diagonal matrix (singular values)
V: n×n orthogonal matrix (right singular vectors)
```

**Key Properties:**
- Works for **any** matrix (not just square!)
- Always exists
- Unique (up to sign)

### Singular Values

The diagonal elements of Σ are **singular values** σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0

**Relationship to Eigenvalues:**
- σᵢ² = eigenvalues of AᵀA (or AAᵀ)
- σᵢ = √λᵢ where λᵢ are eigenvalues of AᵀA

### Geometric Interpretation

SVD shows that any linear transformation can be decomposed into:
1. **Rotation** (V)
2. **Scaling** (Σ)
3. **Rotation** (U)

---

## 📐 Computing SVD

### Method 1: Via Eigendecomposition

**Step 1:** Compute AᵀA
```
AᵀA is n×n, symmetric, positive semi-definite
```

**Step 2:** Find eigenvalues and eigenvectors of AᵀA
```
AᵀA = VΛVᵀ
V: eigenvectors (right singular vectors)
Λ: eigenvalues
```

**Step 3:** Singular values
```
σᵢ = √λᵢ
```

**Step 4:** Left singular vectors
```
uᵢ = (1/σᵢ)Avᵢ
```

### Method 2: Numerical Algorithms

- **Golub-Reinsch algorithm** (standard)
- **Jacobi SVD** (high accuracy)
- **Randomized SVD** (large matrices)

---

## 🔧 Properties of SVD

### 1. Rank
```
rank(A) = number of non-zero singular values
```

### 2. Frobenius Norm
```
||A||_F = √(σ₁² + σ₂² + ... + σᵣ²)
```

### 3. 2-Norm (Spectral Norm)
```
||A||₂ = σ₁ (largest singular value)
```

### 4. Condition Number
```
κ(A) = σ₁/σᵣ (ratio of largest to smallest)
```

### 5. Pseudo-Inverse
```
A⁺ = VΣ⁺Uᵀ
where Σ⁺ has 1/σᵢ on diagonal (for σᵢ ≠ 0)
```

---

## 📊 Low-Rank Approximation

### Best Rank-k Approximation

The best rank-k approximation of A (in Frobenius or 2-norm) is:

```
Aₖ = Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ

Or equivalently:
Aₖ = UₖΣₖVₖᵀ

where Uₖ, Σₖ, Vₖ contain first k components
```

**Approximation Error:**
```
||A - Aₖ||_F = √(σₖ₊₁² + σₖ₊₂² + ... + σᵣ²)
||A - Aₖ||₂ = σₖ₊₁
```

This is **optimal** - no other rank-k matrix is closer to A!

---

## 💻 Practical Workflows

### NumPy Implementation

```python
import numpy as np

# Create matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])  # 4×3 matrix

# Full SVD
U, s, Vt = np.linalg.svd(A, full_matrices=True)

print(f"U shape: {U.shape}")    # (4, 4)
print(f"s shape: {s.shape}")    # (3,) - singular values
print(f"Vt shape: {Vt.shape}")  # (3, 3)

# Reconstruct A
Sigma = np.zeros((4, 3))
Sigma[:3, :3] = np.diag(s)
A_reconstructed = U @ Sigma @ Vt
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")

# Compact SVD (more efficient)
U_compact, s_compact, Vt_compact = np.linalg.svd(A, full_matrices=False)
print(f"U_compact shape: {U_compact.shape}")  # (4, 3)

# Rank
rank = np.sum(s > 1e-10)
print(f"Rank: {rank}")

# Condition number
cond = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
print(f"Condition number: {cond}")

# Pseudo-inverse
A_pinv = Vt.T @ np.linalg.inv(np.diag(s)) @ U[:, :3].T
# Or simply:
A_pinv = np.linalg.pinv(A)
```

### Low-Rank Approximation

```python
def low_rank_approx(A, k):
    """Best rank-k approximation of A"""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only first k components
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct
    A_k = U_k @ np.diag(s_k) @ Vt_k
    return A_k

# Example: compress image
from PIL import Image
img = np.array(Image.open('image.jpg').convert('L'))  # Grayscale

# Different compression levels
for k in [10, 50, 100]:
    img_compressed = low_rank_approx(img, k)
    compression_ratio = k * (img.shape[0] + img.shape[1]) / (img.shape[0] * img.shape[1])
    print(f"Rank {k}: {compression_ratio:.1%} of original size")
```

### Truncated SVD (for large matrices)

```python
from scipy.sparse.linalg import svds

# For large sparse matrices
# Only compute top k singular values/vectors
k = 10
U_k, s_k, Vt_k = svds(A, k=k)

# Note: svds returns singular values in ascending order
# Reverse for descending order
idx = np.argsort(s_k)[::-1]
s_k = s_k[idx]
U_k = U_k[:, idx]
Vt_k = Vt_k[idx, :]
```

---

## 🎯 AI/ML Applications

### 1. Dimensionality Reduction (LSA/LSI)

```python
# Latent Semantic Analysis for text
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "machine learning is great",
    "deep learning uses neural networks",
    "machine learning and deep learning are AI"
]

# TF-IDF matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()  # m×n

# SVD for dimensionality reduction
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# Reduce to k dimensions
k = 2
X_reduced = U[:, :k] @ np.diag(s[:k])  # m×k

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
```

### 2. Collaborative Filtering (Recommender Systems)

```python
# User-item rating matrix (with missing values)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# Fill missing values with mean
ratings_filled = ratings.copy()
for i in range(ratings.shape[1]):
    col = ratings[:, i]
    col_mean = col[col > 0].mean() if np.any(col > 0) else 0
    ratings_filled[col == 0, i] = col_mean

# SVD for matrix completion
U, s, Vt = np.linalg.svd(ratings_filled, full_matrices=False)

# Low-rank approximation
k = 2
ratings_pred = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

print("Predicted ratings:")
print(ratings_pred)
```

### 3. Principal Component Analysis

```python
# PCA via SVD (more numerically stable than eigendecomposition)
def pca_svd(X, n_components):
    """PCA using SVD"""
    # Center data
    X_centered = X - X.mean(axis=0)
    
    # SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Principal components = right singular vectors
    components = Vt[:n_components]
    
    # Explained variance
    explained_variance = (s ** 2) / (len(X) - 1)
    
    # Transform data
    X_transformed = X_centered @ components.T
    
    return X_transformed, components, explained_variance

# Example
X = np.random.randn(100, 10)
X_pca, components, var = pca_svd(X, n_components=3)
print(f"Explained variance ratio: {var[:3] / var.sum()}")
```

### 4. Image Compression

```python
def compress_image(img, k):
    """Compress image using SVD"""
    if len(img.shape) == 3:  # RGB
        compressed = np.zeros_like(img)
        for i in range(3):  # Each color channel
            compressed[:, :, i] = low_rank_approx(img[:, :, i], k)
        return compressed
    else:  # Grayscale
        return low_rank_approx(img, k)

# Compression ratio
def compression_ratio(img_shape, k):
    m, n = img_shape[:2]
    original_size = m * n
    compressed_size = k * (m + n + 1)
    return compressed_size / original_size

# Example
k = 50
ratio = compression_ratio((512, 512), k)
print(f"Compression: {ratio:.1%} of original")
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is SVD?**
   - Decomposition: A = UΣVᵀ
   - Works for any matrix
   - U, V orthogonal; Σ diagonal with singular values

2. **SVD vs Eigendecomposition?**
   - SVD: any matrix
   - Eigen: square matrices only
   - SVD more numerically stable

3. **What are singular values?**
   - Square roots of eigenvalues of AᵀA
   - Measure of "importance" of each dimension
   - Always non-negative

4. **How is SVD used in ML?**
   - PCA (dimensionality reduction)
   - LSA (text analysis)
   - Recommender systems
   - Image compression

5. **What is low-rank approximation?**
   - Best rank-k approximation: keep top k singular values
   - Minimizes reconstruction error
   - Used for compression, denoising

### Must-Know Formulas

```
A = UΣVᵀ
rank(A) = # non-zero singular values
||A||₂ = σ₁
A⁺ = VΣ⁺Uᵀ (pseudo-inverse)
Aₖ = UₖΣₖVₖᵀ (rank-k approx)
```

### Common Pitfalls

- ❌ Confusing singular values with eigenvalues
- ❌ Not sorting singular values in descending order
- ❌ Forgetting SVD works for non-square matrices
- ❌ Using full SVD when truncated would suffice

---

## 🔗 Connections

### Prerequisites
- [Eigenvalues and Eigenvectors](Eigenvalues-and-Eigenvectors.md)
- [Matrix Operations](Matrix-Operations.md)

### Related Topics
- PCA (Machine Learning section)
- Matrix completion
- Compressed sensing

### Applications in AI
- **Dimensionality Reduction:** PCA, LSA
- **Recommender Systems:** Matrix factorization
- **Computer Vision:** Image compression, denoising
- **NLP:** Latent semantic analysis

---

## 📚 References

- **Books:**
  - "Matrix Computations" - Golub & Van Loan
  - "Numerical Linear Algebra" - Trefethen & Bau

- **Papers:**
  - "Finding Structure with Randomness" - Halko et al. (randomized SVD)
  - "Matrix Factorization Techniques for Recommender Systems" - Koren et al.

- **Online:**
  - [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
  - [SciPy Sparse SVD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)

---

**SVD is one of the most powerful tools in linear algebra - master it!**
