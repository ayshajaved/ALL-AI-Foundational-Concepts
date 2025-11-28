# K-Nearest Neighbors (KNN)

> **Instance-based learning** - Classify by majority vote of k nearest neighbors

---

## 🎯 Algorithm

### Idea
```
1. Store all training examples
2. For new point x:
   - Find k nearest neighbors
   - Classify by majority vote (classification)
   - Average values (regression)
```

### Distance Metrics

**Euclidean (L2):**
```
d(x, x') = √(Σᵢ(xᵢ - x'ᵢ)²)
```

**Manhattan (L1):**
```
d(x, x') = Σᵢ|xᵢ - x'ᵢ|
```

**Minkowski (General):**
```
d(x, x') = (Σᵢ|xᵢ - x'ᵢ|ᵖ)^(1/p)
```

---

## 📊 Implementation

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Train (just stores data!)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")

# From scratch
class KNN_scratch:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        # Compute distances
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # Get k nearest
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        
        # Majority vote
        return np.bincount(k_labels).argmax()
```

---

## 🎯 Choosing k

### Small k
- More complex decision boundary
- Sensitive to noise
- Overfitting

### Large k
- Smoother decision boundary
- More robust
- Underfitting

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

# Plot
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('k')
plt.ylabel('CV Accuracy')
plt.title('Choosing k')
plt.show()

best_k = k_values[np.argmax(cv_scores)]
print(f"Best k: {best_k}")
```

---

## 📈 Weighted KNN

**Give closer neighbors more weight:**

```python
# Distance weighting
model_weighted = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'  # vs 'uniform'
)
model_weighted.fit(X_train, y_train)
```

---

## 🎯 Curse of Dimensionality

**Problem:** In high dimensions, all points are far apart!

```python
# Demonstrate curse of dimensionality
dims = [2, 10, 50, 100, 500]
avg_distances = []

for d in dims:
    X_high_d = np.random.randn(1000, d)
    # Average distance to nearest neighbor
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X_high_d)
    distances, _ = nn.kneighbors(X_high_d)
    avg_distances.append(distances[:, 1].mean())

plt.plot(dims, avg_distances, marker='o')
plt.xlabel('Dimensions')
plt.ylabel('Avg Distance to Nearest Neighbor')
plt.title('Curse of Dimensionality')
plt.show()
```

**Solution:** Dimensionality reduction (PCA, t-SNE)

---

## 📊 Efficient Implementations

### KD-Tree
```python
# For low dimensions (d < 20)
model_kdtree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree'
)
```

### Ball Tree
```python
# Better for high dimensions
model_balltree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='ball_tree'
)
```

### Brute Force
```python
# Always works, slow for large n
model_brute = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='brute'
)
```

---

## 🎓 Interview Focus

### Key Questions

1. **How does KNN work?**
   - Store training data
   - Find k nearest neighbors
   - Majority vote for classification

2. **Advantages?**
   - Simple, intuitive
   - No training phase
   - Non-parametric

3. **Disadvantages?**
   - Slow prediction (O(n))
   - Memory intensive
   - Curse of dimensionality

4. **Why scale features?**
   - Distance-based algorithm
   - Features with large ranges dominate
   - Always normalize!

5. **KNN vs other methods?**
   - Good baseline
   - Works for non-linear boundaries
   - Not suitable for large datasets

---

## 📚 References

- **Books:** "Pattern Recognition and Machine Learning" - Bishop

---

**KNN: simple, non-parametric, effective for small data!**
