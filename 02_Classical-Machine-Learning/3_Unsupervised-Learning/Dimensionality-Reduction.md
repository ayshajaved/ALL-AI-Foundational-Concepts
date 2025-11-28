# Dimensionality Reduction

> **Reduce features while preserving information** - PCA, t-SNE, UMAP

---

## 🎯 Principal Component Analysis (PCA)

### Objective
Find directions of maximum variance

```
PC₁ = argmax Var(Xw) subject to ||w|| = 1
PC₂ = argmax Var(Xw) subject to ||w|| = 1, w ⊥ PC₁
...
```

### Implementation

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X, y = digits.data, digits.target  # 64 dimensions

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total: {pca.explained_variance_ratio_.sum():.2%}")

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.title('PCA of Digits')
plt.show()

# Scree plot
pca_full = PCA()
pca_full.fit(X)
plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1),
         np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.show()
```

---

## 📊 t-SNE

### For visualization (non-linear)

```python
from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.5)
plt.title('t-SNE of Digits')
plt.colorbar()
plt.show()
```

---

## 📈 UMAP

### Faster alternative to t-SNE

```python
import umap

# UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.5)
plt.title('UMAP of Digits')
plt.colorbar()
plt.show()
```

---

## 🎓 Interview Focus

1. **PCA vs t-SNE?**
   - PCA: linear, fast, preserves global structure
   - t-SNE: non-linear, slow, preserves local structure

2. **When to use PCA?**
   - Preprocessing for ML
   - Noise reduction
   - Visualization (2D/3D)

---

**Dimensionality reduction: see high-dimensional data!**
