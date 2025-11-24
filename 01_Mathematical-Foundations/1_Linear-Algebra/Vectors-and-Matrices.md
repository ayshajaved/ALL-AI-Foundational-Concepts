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
