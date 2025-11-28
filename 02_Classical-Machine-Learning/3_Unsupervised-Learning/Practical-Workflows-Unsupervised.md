# Practical Workflows - Unsupervised Learning

> **End-to-end unsupervised projects** - Clustering, dimensionality reduction, anomaly detection

---

## 🎯 Complete Clustering Pipeline

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Load and explore
df = pd.read_csv('data.csv')
X = df.drop('id', axis=1)

# 2. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Choose k
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# 4. Train final model
best_k = K_range[np.argmax(silhouettes)]
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans_final.fit_predict(X_scaled)

# 5. Analyze clusters
df['cluster'] = labels
for i in range(best_k):
    print(f"\nCluster {i}:")
    print(df[df['cluster'] == i].describe())
```

---

## 📊 Dimensionality Reduction Pipeline

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA for preprocessing
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

# t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
plt.title('Clusters Visualization')
plt.show()
```

---

## 🎯 Anomaly Detection

```python
from sklearn.ensemble import IsolationForest

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X_scaled)

# -1 for anomalies, 1 for normal
print(f"Anomalies: {(anomalies == -1).sum()}")
```

---

**Complete workflows for unsupervised learning!**
