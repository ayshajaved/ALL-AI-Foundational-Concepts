# Gaussian Mixture Models (GMM)

> **Probabilistic clustering** - Soft assignments with EM algorithm

---

## 🎯 Model

### Mixture of Gaussians
```
p(x) = Σₖ πₖ N(x | μₖ, Σₖ)

πₖ: mixing coefficient
N: Gaussian distribution
```

### Implementation

```python
from sklearn.mixture import GaussianMixture

# GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Predict (hard assignment)
labels = gmm.predict(X)

# Soft assignment (probabilities)
probs = gmm.predict_proba(X)

print(f"Means:\n{gmm.means_}")
print(f"Covariances shape: {gmm.covariances_.shape}")
print(f"Weights: {gmm.weights_}")
```

---

## 📊 EM Algorithm

```
E-step: Compute responsibilities
M-step: Update parameters

Repeat until convergence
```

---

## 🎯 Model Selection

```python
from sklearn.metrics import silhouette_score

# BIC
bic_scores = []
for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))

plt.plot(range(1, 11), bic_scores, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('BIC')
plt.title('Model Selection')
plt.show()
```

---

## 🎓 Interview Focus

1. **GMM vs K-Means?**
   - GMM: soft assignments, probabilistic
   - K-Means: hard assignments, faster

2. **EM algorithm?**
   - E-step: estimate cluster assignments
   - M-step: update parameters

---

**GMM: probabilistic clustering with soft assignments!**
