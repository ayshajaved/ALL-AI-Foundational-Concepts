# Hierarchical Clustering

> **Tree-based clustering** - Build hierarchy of clusters without specifying k

---

## 🎯 Agglomerative Clustering

### Bottom-Up Approach
```
1. Start: each point is a cluster
2. Merge closest clusters
3. Repeat until one cluster (or k clusters)
4. Result: dendrogram
```

### Implementation

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.show()

# Dendrogram
Z = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

---

## 📊 Linkage Methods

### Single Linkage
```
d(C₁, C₂) = min{d(x, y) : x ∈ C₁, y ∈ C₂}
```

### Complete Linkage
```
d(C₁, C₂) = max{d(x, y) : x ∈ C₁, y ∈ C₂}
```

### Average Linkage
```
d(C₁, C₂) = avg{d(x, y) : x ∈ C₁, y ∈ C₂}
```

### Ward Linkage
```
Minimize within-cluster variance
```

```python
# Compare linkages
linkages = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, linkage_method in zip(axes.ravel(), linkages):
    agg = AgglomerativeClustering(n_clusters=3, linkage=linkage_method)
    labels = agg.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    ax.set_title(f'{linkage_method.capitalize()} Linkage')
plt.tight_layout()
plt.show()
```

---

## 🎓 Interview Focus

1. **Agglomerative vs Divisive?**
   - Agglomerative: bottom-up (common)
   - Divisive: top-down (rare)

2. **Advantages?**
   - No need to specify k
   - Dendrogram shows hierarchy
   - Deterministic

3. **Disadvantages?**
   - O(n³) time complexity
   - Can't undo merges
   - Sensitive to noise

---

**Hierarchical clustering: visualize cluster hierarchy!**
