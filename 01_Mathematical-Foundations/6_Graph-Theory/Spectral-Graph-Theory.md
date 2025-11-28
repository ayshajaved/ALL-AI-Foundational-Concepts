# Spectral Graph Theory

> **Graphs through linear algebra** - Laplacians, spectral clustering, and graph signal processing

---

## 🎯 Graph Laplacian

### Adjacency Matrix
```
A[i,j] = {1 if (i,j) ∈ E
         {0 otherwise

For weighted graphs: A[i,j] = wᵢⱼ
```

### Degree Matrix
```
D[i,i] = Σⱼ A[i,j]  (degree of node i)
D[i,j] = 0 for i ≠ j
```

### Unnormalized Laplacian
```
L = D - A
```

```python
import numpy as np
import networkx as nx

# Create graph
G = nx.karate_club_graph()

# Adjacency matrix
A = nx.adjacency_matrix(G).todense()

# Degree matrix
degrees = np.array(A.sum(axis=1)).flatten()
D = np.diag(degrees)

# Laplacian
L = D - A

print(f"Laplacian shape: {L.shape}")
```

### Properties of L

1. **Symmetric:** L = Lᵀ
2. **Positive semidefinite:** xᵀLx ≥ 0
3. **Row sums zero:** L·1 = 0
4. **Smallest eigenvalue:** λ₁ = 0 with eigenvector 1

```python
# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(L)

print(f"Smallest eigenvalue: {eigenvalues[0]:.6f}")  # ≈ 0
print(f"Largest eigenvalue: {eigenvalues[-1]:.3f}")
```

---

## 📊 Normalized Laplacian

### Symmetric Normalized Laplacian
```
L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
```

```python
# Compute normalized Laplacian
D_inv_sqrt = np.diag(1 / np.sqrt(degrees))
L_sym = D_inv_sqrt @ L @ D_inv_sqrt

# Or using NetworkX
L_sym_nx = nx.normalized_laplacian_matrix(G).todense()
```

### Random Walk Normalized Laplacian
```
L_rw = D^{-1} L = I - D^{-1} A
```

---

## 🎯 Spectral Clustering

### Algorithm

1. Compute Laplacian L
2. Find k smallest eigenvectors
3. Cluster rows of eigenvector matrix

```python
from sklearn.cluster import KMeans

def spectral_clustering(A, k, normalized=True):
    """
    Spectral clustering
    
    A: adjacency matrix
    k: number of clusters
    """
    # Compute Laplacian
    degrees = np.array(A.sum(axis=1)).flatten()
    D = np.diag(degrees)
    L = D - A
    
    if normalized:
        D_inv_sqrt = np.diag(1 / np.sqrt(degrees + 1e-10))
        L = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Take k smallest eigenvectors
    U = eigenvectors[:, :k]
    
    # K-means on rows
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(U)
    
    return labels

# Example
labels = spectral_clustering(A, k=2)
print(f"Cluster labels: {labels}")

# Visualize
import matplotlib.pyplot as plt
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=labels, cmap='viridis', with_labels=True)
plt.show()
```

---

## 📈 Cheeger Inequality

### Graph Cut
```
Cut(S, S̄) = Σ_{i∈S, j∈S̄} wᵢⱼ

Measures connectivity between sets
```

### Conductance
```
φ(S) = Cut(S, S̄) / min(vol(S), vol(S̄))

vol(S) = Σ_{i∈S} dᵢ
```

### Cheeger's Inequality
```
λ₂/2 ≤ φ(G) ≤ √(2λ₂)

λ₂: second smallest eigenvalue (algebraic connectivity)
φ(G): Cheeger constant (min conductance)
```

**Implication:** λ₂ measures how well-connected the graph is!

---

## 🎯 Random Walks on Graphs

### Transition Matrix
```
P = D^{-1} A

P[i,j] = probability of moving from i to j
```

### Stationary Distribution
```
π = D·1 / (1ᵀD·1)

πᵢ = dᵢ / (2|E|)

Proportional to degree!
```

```python
def random_walk(G, start, steps=100):
    """Simulate random walk on graph"""
    current = start
    path = [current]
    
    for _ in range(steps):
        neighbors = list(G.neighbors(current))
        if neighbors:
            current = np.random.choice(neighbors)
            path.append(current)
    
    return path

# Simulate
path = random_walk(G, start=0, steps=1000)

# Empirical distribution
unique, counts = np.unique(path, return_counts=True)
empirical_dist = counts / len(path)

# Theoretical (stationary)
degrees = np.array([G.degree(i) for i in range(len(G))])
stationary = degrees / degrees.sum()

print(f"Empirical: {empirical_dist[:5]}")
print(f"Stationary: {stationary[:5]}")
```

---

## 📊 Graph Signal Processing

### Graph Signal
```
f: V → ℝ

f[i]: signal value at node i
```

### Graph Fourier Transform
```
f̂ = Uᵀf

U: eigenvectors of Laplacian
f̂: frequency domain representation
```

```python
def graph_fourier_transform(L, signal):
    """
    Graph Fourier transform
    """
    # Eigendecomposition
    eigenvalues, U = np.linalg.eigh(L)
    
    # Transform
    signal_freq = U.T @ signal
    
    return signal_freq, eigenvalues, U

def inverse_graph_fourier(signal_freq, U):
    """Inverse graph Fourier transform"""
    return U @ signal_freq

# Example: smooth signal on graph
signal = np.random.randn(len(G))

signal_freq, eigenvalues, U = graph_fourier_transform(L, signal)

# Low-pass filter (keep low frequencies)
k = 5  # Keep first k frequencies
signal_freq_filtered = signal_freq.copy()
signal_freq_filtered[k:] = 0

# Inverse transform
signal_filtered = inverse_graph_fourier(signal_freq_filtered, U)

print(f"Original signal variance: {signal.var():.3f}")
print(f"Filtered signal variance: {signal_filtered.var():.3f}")
```

---

## 🎯 Applications in ML

### 1. Graph Convolutional Networks

**Spectral convolution:**
```
g_θ * f = U g_θ(Λ) Uᵀ f

g_θ(Λ): filter in frequency domain
```

**ChebNet approximation:**
```
g_θ * f ≈ Σₖ θₖ Tₖ(L̃) f

Tₖ: Chebyshev polynomials
L̃: normalized Laplacian
```

### 2. Node Embeddings

```python
# Spectral embedding
def spectral_embedding(L, dim=2):
    """
    Embed nodes using Laplacian eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Use eigenvectors 2 to dim+1 (skip first)
    embedding = eigenvectors[:, 1:dim+1]
    
    return embedding

# Visualize
embedding = spectral_embedding(L, dim=2)
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title('Spectral Embedding')
plt.show()
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is graph Laplacian?**
   - L = D - A
   - Encodes graph structure
   - Positive semidefinite

2. **Why spectral clustering works?**
   - Eigenvectors encode connectivity
   - Relaxation of graph cut problem
   - Cheeger inequality guarantees

3. **Second eigenvalue significance?**
   - Algebraic connectivity
   - Measures how connected graph is
   - 0 iff graph disconnected

4. **Graph Fourier transform?**
   - Eigenvectors of Laplacian as basis
   - Frequency = eigenvalue
   - Smooth signals = low frequency

5. **Spectral GCNs?**
   - Convolution in spectral domain
   - Polynomial filters for efficiency
   - Foundation of graph neural networks

---

## 📚 References

- **Books:**
  - "Spectral Graph Theory" - Chung
  - "A Tutorial on Spectral Clustering" - von Luxburg

- **Papers:**
  - "Semi-Supervised Classification with Graph Convolutional Networks" - Kipf & Welling

---

**Spectral graph theory: where graphs meet linear algebra!**
