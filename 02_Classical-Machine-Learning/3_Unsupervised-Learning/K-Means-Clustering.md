# K-Means Clustering

> **Partitional clustering** - Simple, fast, widely used unsupervised algorithm

---

## 🎯 Algorithm

### Objective
Minimize within-cluster sum of squares (WCSS):
```
J = Σₖ Σ_{x∈Cₖ} ||x - μₖ||²

μₖ: centroid of cluster k
```

### Lloyd's Algorithm
```
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat 2-3 until convergence
```

### Implementation

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Results
print(f"Centroids:\n{kmeans.cluster_centers_}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Iterations: {kmeans.n_iter_}")

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], 
           kmeans.cluster_centers_[:, 1],
           marker='x', s=200, c='red', linewidths=3)
plt.title('K-Means Clustering')
plt.show()
```

---

## 📊 Choosing k

### Elbow Method
```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()
```

### Silhouette Score
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()
```

---

## 🎯 K-Means++

**Better initialization to avoid poor local minima**

```python
# K-Means++ (default in sklearn)
kmeans_pp = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_pp.fit(X)

# Random initialization
kmeans_random = KMeans(n_clusters=4, init='random', random_state=42)
kmeans_random.fit(X)

print(f"K-Means++: {kmeans_pp.inertia_:.2f}")
print(f"Random: {kmeans_random.inertia_:.2f}")
```

---

## 📈 Mini-Batch K-Means

**For large datasets**

```python
from sklearn.cluster import MiniBatchKMeans

# Mini-batch (faster)
mb_kmeans = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42)
mb_kmeans.fit(X)

print(f"Mini-batch inertia: {mb_kmeans.inertia_:.2f}")
```

---

## 🎓 Interview Focus

### Key Questions

1. **How does K-Means work?**
   - Initialize k centroids
   - Assign points to nearest centroid
   - Update centroids
   - Repeat until convergence

2. **Limitations?**
   - Must specify k
   - Sensitive to initialization
   - Assumes spherical clusters
   - Sensitive to outliers

3. **Time complexity?**
   - O(nkdi) where n=samples, k=clusters, d=dimensions, i=iterations
   - Usually converges quickly

4. **K-Means vs Hierarchical?**
   - K-Means: faster, need to specify k
   - Hierarchical: slower, no need for k

---

**K-Means: simple, fast, effective clustering!**
