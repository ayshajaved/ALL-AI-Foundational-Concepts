# Tensor Algebra and Operations

> **Multilinear algebra for deep learning** - Tensors, Einstein notation, and decompositions

---

## 🎯 What are Tensors?

### Definition
A **tensor** is a multidimensional array generalizing scalars, vectors, and matrices.

```
Scalar: 0-order tensor (rank 0)
Vector: 1-order tensor (rank 1)
Matrix: 2-order tensor (rank 2)
Tensor: n-order tensor (rank n)
```

**Example:**
```python
import numpy as np

# Scalar (rank 0)
scalar = 5

# Vector (rank 1)
vector = np.array([1, 2, 3])

# Matrix (rank 2)
matrix = np.array([[1, 2], [3, 4]])

# Tensor (rank 3)
tensor = np.random.randn(2, 3, 4)  # 2×3×4 tensor
print(f"Shape: {tensor.shape}")
print(f"Rank: {tensor.ndim}")
```

---

## 📊 Tensor Operations

### 1. Tensor Product (Outer Product)

```
A ⊗ B: tensor product

For vectors u ∈ ℝᵐ, v ∈ ℝⁿ:
(u ⊗ v)ᵢⱼ = uᵢvⱼ
```

```python
def tensor_product(u, v):
    """Tensor product of two vectors"""
    return np.outer(u, v)

u = np.array([1, 2, 3])
v = np.array([4, 5])
T = tensor_product(u, v)
print(f"u ⊗ v shape: {T.shape}")  # (3, 2)
```

### 2. Tensor Contraction

**Idea:** Sum over paired indices (generalization of matrix multiplication)

```python
# Matrix multiplication as tensor contraction
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Contract over index 1 of A and index 0 of B
C = np.tensordot(A, B, axes=([1], [0]))
# Equivalent to: C = A @ B
```

### 3. Mode-n Product

**Definition:** Multiply tensor by matrix along mode n

```python
def mode_n_product(T, M, n):
    """
    Mode-n product of tensor T with matrix M
    T: tensor of shape (I₁, I₂, ..., Iₙ, ..., Iₖ)
    M: matrix of shape (J, Iₙ)
    Result: shape (I₁, ..., Iₙ₋₁, J, Iₙ₊₁, ..., Iₖ)
    """
    return np.tensordot(M, T, axes=([1], [n]))

# Example
T = np.random.randn(3, 4, 5)
M = np.random.randn(6, 4)
result = mode_n_product(T, M, 1)
print(f"Result shape: {result.shape}")  # (6, 3, 5)
```

---

## 🎯 Einstein Summation Convention

### Notation
```
Cᵢⱼ = Σₖ AᵢₖBₖⱼ  →  Cᵢⱼ = AᵢₖBₖⱼ

Repeated indices are summed over
```

### NumPy einsum

```python
# Matrix multiplication
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Traditional
C1 = A @ B

# Einstein notation
C2 = np.einsum('ik,kj->ij', A, B)

assert np.allclose(C1, C2)
```

### Common Operations

```python
# Trace
A = np.random.randn(5, 5)
trace = np.einsum('ii->', A)  # Σᵢ Aᵢᵢ

# Diagonal
diag = np.einsum('ii->i', A)  # [A₁₁, A₂₂, ...]

# Transpose
At = np.einsum('ij->ji', A)

# Batch matrix multiplication
# A: (batch, m, k), B: (batch, k, n)
A = np.random.randn(10, 3, 4)
B = np.random.randn(10, 4, 5)
C = np.einsum('bik,bkj->bij', A, B)

# Outer product
u = np.random.randn(3)
v = np.random.randn(4)
outer = np.einsum('i,j->ij', u, v)

# Inner product
inner = np.einsum('i,i->', u, u)  # u·u

# Hadamard (element-wise) product
A = np.random.randn(3, 4)
B = np.random.randn(3, 4)
hadamard = np.einsum('ij,ij->ij', A, B)
```

---

## 📈 Tensor Decompositions

### 1. CP Decomposition (CANDECOMP/PARAFAC)

**Idea:** Decompose tensor into sum of rank-1 tensors

```
T ≈ Σᵣ λᵣ aᵣ ⊗ bᵣ ⊗ cᵣ

For 3rd-order tensor
```

```python
# Using tensorly
import tensorly as tl
from tensorly.decomposition import parafac

# Create tensor
T = np.random.randn(5, 6, 7)

# CP decomposition
rank = 3
factors = parafac(T, rank=rank)

# Reconstruct
T_reconstructed = tl.cp_to_tensor(factors)

# Error
error = np.linalg.norm(T - T_reconstructed)
print(f"Reconstruction error: {error:.4f}")
```

### 2. Tucker Decomposition

**Idea:** Decompose into core tensor and factor matrices

```
T ≈ G ×₁ A ×₂ B ×₃ C

G: core tensor
A, B, C: factor matrices
```

```python
from tensorly.decomposition import tucker

# Tucker decomposition
core, factors = tucker(T, ranks=[3, 4, 5])

# Reconstruct
T_reconstructed = tl.tucker_to_tensor((core, factors))
```

### 3. Tensor Train (TT) Decomposition

**Idea:** Represent tensor as product of 3rd-order tensors

```python
from tensorly.decomposition import tensor_train

# TT decomposition
factors = tensor_train(T, rank=[1, 3, 4, 1])

# Reconstruct
T_reconstructed = tl.tt_to_tensor(factors)
```

---

## 🎯 Applications in Deep Learning

### 1. Tensor Layers

```python
import torch
import torch.nn as nn

class TensorLayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        # Low-rank tensor factorization
        self.U = nn.Parameter(torch.randn(in_features, rank))
        self.V = nn.Parameter(torch.randn(rank, out_features))
    
    def forward(self, x):
        # x: (batch, in_features)
        return x @ self.U @ self.V
```

### 2. Attention as Tensor Contraction

```python
# Multi-head attention
# Q, K, V: (batch, seq_len, d_model)
# Attention: (batch, heads, seq_len, seq_len)

def multi_head_attention(Q, K, V, num_heads):
    """
    Using einsum for clarity
    """
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # Reshape for multi-head
    Q = Q.view(batch, seq_len, num_heads, d_k)
    K = K.view(batch, seq_len, num_heads, d_k)
    V = V.view(batch, seq_len, num_heads, d_k)
    
    # Attention scores
    # (batch, heads, seq, d_k) × (batch, heads, d_k, seq)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / np.sqrt(d_k)
    
    # Softmax
    attn = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    # (batch, heads, seq, seq) × (batch, heads, seq, d_k)
    output = torch.einsum('bhqk,bkhd->bqhd', attn, V)
    
    # Concatenate heads
    output = output.contiguous().view(batch, seq_len, d_model)
    
    return output
```

### 3. Tensor Contraction Networks

```python
# Example: Tensor network for image classification
class TensorNetwork(nn.Module):
    def __init__(self, input_shape, num_classes, bond_dim=10):
        super().__init__()
        self.bond_dim = bond_dim
        
        # Create tensor cores
        self.cores = nn.ParameterList([
            nn.Parameter(torch.randn(2, bond_dim, bond_dim))
            for _ in range(np.prod(input_shape))
        ])
        
        self.classifier = nn.Linear(bond_dim, num_classes)
    
    def forward(self, x):
        # Contract tensor network
        # ... (simplified)
        pass
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is a tensor?**
   - Multidimensional array
   - Generalizes scalars, vectors, matrices
   - Rank = number of dimensions

2. **Einstein summation?**
   - Repeated indices are summed
   - Concise notation for tensor operations
   - Implemented in np.einsum

3. **Why tensor decompositions?**
   - Compression
   - Denoising
   - Feature extraction
   - Efficient computation

4. **Tensors in transformers?**
   - Attention as tensor contraction
   - Multi-head = tensor reshaping
   - Efficient with einsum

5. **CP vs Tucker?**
   - CP: sum of rank-1 tensors
   - Tucker: core tensor + factors
   - Tucker more flexible, CP simpler

---

## 📚 References

- **Books:** 
  - "Tensor Decompositions and Applications" - Kolda & Bader
  - "Tensor Methods in Statistics" - McCullagh
  
- **Libraries:**
  - TensorLy (Python)
  - PyTorch (torch.einsum)
  - TensorFlow (tf.einsum)

---

**Tensors: the language of modern deep learning!**
