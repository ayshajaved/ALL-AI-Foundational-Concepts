# Vectors and Matrices

> **Foundation of linear algebra for AI** - Understanding vectors and matrices is essential for all machine learning

---

## 📐 Vectors

### Definition
A **vector** is an ordered array of numbers representing magnitude and direction in n-dimensional space.

**Notation:**
```
v = [v₁, v₂, ..., vₙ]ᵀ  (column vector)
v = [v₁, v₂, ..., vₙ]   (row vector)
```

### Types of Vectors

**1. Zero Vector**
```
0 = [0, 0, ..., 0]ᵀ
```

**2. Unit Vector** (length = 1)
```
||e|| = 1
Standard basis: e₁ = [1,0,0], e₂ = [0,1,0], e₃ = [0,0,1]
```

**3. Sparse Vector** (mostly zeros)
- Common in NLP (word embeddings)
- Efficient storage

---

## 🔢 Vector Operations

### Addition
```
u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]ᵀ
```

**Properties:**
- Commutative: u + v = v + u
- Associative: (u + v) + w = u + (v + w)

### Scalar Multiplication
```
αv = [αv₁, αv₂, ..., αvₙ]ᵀ
```

### Dot Product (Inner Product)
```
u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ = Σᵢ uᵢvᵢ
```

**Geometric Interpretation:**
```
u · v = ||u|| ||v|| cos(θ)
where θ is angle between vectors
```

**Properties:**
- Commutative: u · v = v · u
- Distributive: u · (v + w) = u · v + u · w
- If u · v = 0, vectors are orthogonal

### Vector Norm (Length)

**L2 Norm (Euclidean)**
```
||v||₂ = √(v₁² + v₂² + ... + vₙ²) = √(v · v)
```

**L1 Norm (Manhattan)**
```
||v||₁ = |v₁| + |v₂| + ... + |vₙ|
```

**L∞ Norm (Maximum)**
```
||v||∞ = max(|v₁|, |v₂|, ..., |vₙ|)
```

**p-Norm (General)**
```
||v||ₚ = (|v₁|ᵖ + |v₂|ᵖ + ... + |vₙ|ᵖ)^(1/p)
```

### Unit Vector (Normalization)
```
v̂ = v / ||v||
```

---

## 📊 Matrices

### Definition
A **matrix** is a rectangular array of numbers arranged in rows and columns.

**Notation:**
```
A = [aᵢⱼ]  where i = row, j = column

     [a₁₁  a₁₂  ...  a₁ₙ]
A =  [a₂₁  a₂₂  ...  a₂ₙ]
     [ ⋮    ⋮    ⋱    ⋮ ]
     [aₘ₁  aₘ₂  ...  aₘₙ]

Dimensions: m × n (m rows, n columns)
```

### Special Matrices

**1. Square Matrix** (m = n)
```
A is n × n
```

**2. Identity Matrix**
```
     [1  0  0]
I =  [0  1  0]
     [0  0  1]

AI = IA = A
```

**3. Diagonal Matrix**
```
     [d₁  0   0 ]
D =  [0   d₂  0 ]
     [0   0   d₃]

All off-diagonal elements = 0
```

**4. Zero Matrix**
```
All elements = 0
```

**5. Symmetric Matrix**
```
A = Aᵀ  (aᵢⱼ = aⱼᵢ)
```

**6. Orthogonal Matrix**
```
QᵀQ = QQᵀ = I
Columns are orthonormal vectors
```

**7. Triangular Matrices**
```
Upper Triangular: aᵢⱼ = 0 for i > j
Lower Triangular: aᵢⱼ = 0 for i < j
```

---

## 🔧 Matrix Operations

### Addition
```
C = A + B
cᵢⱼ = aᵢⱼ + bᵢⱼ

Requires: Same dimensions
```

### Scalar Multiplication
```
B = αA
bᵢⱼ = α·aᵢⱼ
```

### Transpose
```
(Aᵀ)ᵢⱼ = Aⱼᵢ

If A is m×n, then Aᵀ is n×m
```

**Properties:**
- (Aᵀ)ᵀ = A
- (A + B)ᵀ = Aᵀ + Bᵀ
- (AB)ᵀ = BᵀAᵀ
- (αA)ᵀ = αAᵀ

### Matrix-Vector Multiplication
```
y = Ax

yᵢ = Σⱼ aᵢⱼxⱼ

If A is m×n and x is n×1, then y is m×1
```

**Interpretation:**
- Linear transformation
- Combination of column vectors

### Matrix-Matrix Multiplication
```
C = AB

cᵢⱼ = Σₖ aᵢₖbₖⱼ

If A is m×n and B is n×p, then C is m×p
```

**Properties:**
- NOT commutative: AB ≠ BA (in general)
- Associative: (AB)C = A(BC)
- Distributive: A(B+C) = AB + AC

---

## 🎯 AI/ML Applications

### 1. Data Representation
```python
# Dataset as matrix
X = [x₁, x₂, ..., xₙ]ᵀ  # n samples
Each xᵢ is d-dimensional feature vector

X is n × d matrix
```

### 2. Linear Transformations
```python
# Neural network layer
y = Wx + b

W: weight matrix
x: input vector
b: bias vector
y: output vector
```

### 3. Image Representation
```python
# Grayscale image: 2D matrix
# RGB image: 3D tensor (H × W × 3)
# Batch of images: 4D tensor (N × H × W × 3)
```

### 4. Word Embeddings
```python
# Embedding matrix
E: V × d  (V = vocabulary size, d = embedding dim)
Each row = word vector
```

---

## ⚡ Computational Complexity

### Operation Complexity

| Operation | Time Complexity | Space | Notes |
|-----------|----------------|-------|-------|
| Vector addition | O(n) | O(n) | Element-wise |
| Dot product | O(n) | O(1) | Single pass |
| Vector norm | O(n) | O(1) | Sum + sqrt |
| Matrix-vector mult | O(mn) | O(m) | m×n matrix |
| Matrix-matrix mult | O(mnp) | O(mp) | m×n × n×p |
| Transpose | O(1) | O(1) | View only (NumPy) |

### Scalability Considerations

**Small Dimensions (n < 1000):**
- Use dense matrices
- Standard NumPy operations
- No special optimization needed

**Medium Dimensions (1000 < n < 100,000):**
- Consider sparse matrices if >90% zeros
- Use optimized BLAS libraries
- Batch operations when possible

**Large Dimensions (n > 100,000):**
- **Must use sparse matrices**
- Distributed computing (Dask, Ray)
- Approximate methods (randomized algorithms)

---

## 🛡️ Numerical Stability

### Critical Issues in Production

**1. Overflow/Underflow**
```python
# BAD: Can overflow
exp_values = np.exp(large_numbers)

# GOOD: Subtract max for stability
exp_values = np.exp(large_numbers - np.max(large_numbers))
```

**2. Loss of Precision**
```python
# BAD: Catastrophic cancellation
result = (a + b) - (a + c)  # If a >> b,c

# GOOD: Rearrange
result = b - c
```

**3. Ill-Conditioned Matrices**
```python
# Check condition number before inversion
cond = np.linalg.cond(A)
if cond > 1e10:
    print(f"Warning: Matrix is ill-conditioned (κ={cond:.2e})")
    # Use regularization or pseudo-inverse
    A_reg = A + 1e-6 * np.eye(A.shape[0])
```

**4. Normalization Issues**
```python
# BAD: Division by zero
normalized = v / np.linalg.norm(v)

# GOOD: Add epsilon
normalized = v / (np.linalg.norm(v) + 1e-8)
```

### Production Best Practices

✅ **Always check for:**
- NaN values: `np.isnan(A).any()`
- Inf values: `np.isinf(A).any()`
- Condition number: `np.linalg.cond(A)`
- Rank deficiency: `np.linalg.matrix_rank(A)`

✅ **Use stable algorithms:**
- `np.linalg.solve()` instead of `np.linalg.inv()`
- `np.linalg.lstsq()` for overdetermined systems
- SVD for rank-deficient matrices

---

## 🎓 Advanced Topics

### Linear Independence

**Definition:** Vectors v₁, v₂, ..., vₙ are **linearly independent** if:
```
c₁v₁ + c₂v₂ + ... + cₙvₙ = 0  ⟹  c₁ = c₂ = ... = cₙ = 0
```

**Check in NumPy:**
```python
def are_linearly_independent(vectors):
    """Check if column vectors are linearly independent"""
    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)
    return rank == len(vectors)

# Example
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([1, 1, 0])  # Linear combination of v1, v2

print(are_linearly_independent([v1, v2]))  # True
print(are_linearly_independent([v1, v2, v3]))  # True (still independent)
```

### Span and Basis

**Span:** Set of all linear combinations
```
span(v₁, v₂, ..., vₙ) = {c₁v₁ + c₂v₂ + ... + cₙvₙ : cᵢ ∈ ℝ}
```

**Basis:** Linearly independent set that spans the space
```python
# Standard basis for ℝ³
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# Any vector in ℝ³ can be written as combination
v = 2*e1 + 3*e2 + 4*e3  # [2, 3, 4]
```

### Vector Spaces

**Column Space (Range):** Span of column vectors
```python
def column_space_basis(A):
    """Find basis for column space"""
    Q, R = np.linalg.qr(A)
    rank = np.linalg.matrix_rank(A)
    return Q[:, :rank]
```

**Null Space (Kernel):** Solutions to Ax = 0
```python
def null_space(A, tol=1e-10):
    """Find basis for null space"""
    U, s, Vt = np.linalg.svd(A)
    null_mask = s < tol
    return Vt[null_mask].T
```

### Outer Product

**Definition:** u ⊗ v = uvᵀ (rank-1 matrix)
```python
u = np.array([1, 2, 3])
v = np.array([4, 5])

# Outer product
A = np.outer(u, v)
print(A.shape)  # (3, 2)
print(np.linalg.matrix_rank(A))  # 1 (rank-1 matrix)
```

**Application:** Building matrices from vectors
```python
# Covariance matrix from centered data
X_centered = X - X.mean(axis=0)
Cov = (X_centered.T @ X_centered) / (len(X) - 1)
# Each term is an outer product!
```

---

## 💻 Practical Workflows

### NumPy Implementation

```python
import numpy as np

# Create vectors
v = np.array([1, 2, 3])
u = np.array([4, 5, 6])

# Vector operations
v_plus_u = v + u
v_scaled = 2 * v
dot_product = np.dot(v, u)  # or v @ u
norm = np.linalg.norm(v)  # L2 norm
unit_vector = v / norm

# Create matrices
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2
B = np.array([[7, 8, 9], [10, 11, 12]])  # 2×3

# Matrix operations
A_transpose = A.T
C = A @ B  # Matrix multiplication (3×3)
identity = np.eye(3)  # 3×3 identity

# Special matrices
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
diagonal = np.diag([1, 2, 3])

# Matrix-vector multiplication
x = np.array([1, 2])
y = A @ x  # Result: 3×1

# Element-wise operations
A_squared = A ** 2  # Element-wise square
A_times_2 = A * 2   # Element-wise multiplication
```

### Common Patterns

**1. Batch Processing**
```python
# Process multiple samples at once
X = np.random.randn(100, 784)  # 100 samples, 784 features
W = np.random.randn(784, 10)   # Weights
Y = X @ W  # (100, 10) - all samples processed together
```

**2. Broadcasting**
```python
# Add bias to all samples
X = np.random.randn(100, 10)
b = np.random.randn(10)
Y = X + b  # b is broadcast to (100, 10)
```

**3. Normalization**
```python
# Normalize each feature
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is the difference between a row vector and column vector?**
   - Row: 1×n, Column: n×1
   - Transpose relationship
   - Different multiplication rules

2. **When can you multiply two matrices?**
   - A(m×n) × B(n×p) = C(m×p)
   - Inner dimensions must match

3. **What does the dot product represent geometrically?**
   - Projection of one vector onto another
   - Measures similarity/alignment
   - Zero if orthogonal

4. **Why is matrix multiplication not commutative?**
   - AB ≠ BA in general
   - Dimensions may not even match
   - Represents different transformations

5. **What is a symmetric matrix and why is it important?**
   - A = Aᵀ
   - Real eigenvalues
   - Common in covariance matrices

### Must-Know Formulas

```
Dot product: u · v = Σᵢ uᵢvᵢ
L2 norm: ||v|| = √(Σᵢ vᵢ²)
Matrix mult: (AB)ᵢⱼ = Σₖ aᵢₖbₖⱼ
Transpose: (AB)ᵀ = BᵀAᵀ
```

### Common Pitfalls

- ❌ Forgetting dimension compatibility
- ❌ Confusing element-wise and matrix multiplication
- ❌ Not checking for square matrices when needed
- ❌ Assuming commutativity

---

## 🔗 Connections

### Prerequisites
- Basic algebra
- Coordinate systems

### Related Topics
- [Matrix Operations](Matrix-Operations.md)
- [Eigenvalues and Eigenvectors](Eigenvalues-and-Eigenvectors.md)
- [Linear Transformations](../2_Calculus/Derivatives-and-Gradients.md)

### Applications in AI
- Neural network layers
- Data preprocessing
- Dimensionality reduction (PCA)
- Embeddings

---

## 📚 References

- **Books:**
  - "Introduction to Linear Algebra" - Gilbert Strang
  - "Linear Algebra and Its Applications" - David Lay
  
- **Online:**
  - [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
  - [MIT OCW: Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
  
- **Practice:**
  - NumPy documentation
  - Linear algebra exercises on Khan Academy

---

**Master vectors and matrices - they are the language of AI!**
