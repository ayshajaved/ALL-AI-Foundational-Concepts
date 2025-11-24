# Multivariate Distributions

> **Joint probability distributions** - Covariance, correlation, and multivariate Gaussian

---

## 🔗 Joint Distributions

### Joint PDF/PMF
```
Discrete: p(x,y) = P(X=x, Y=y)
Continuous: f(x,y)
```

### Marginal Distributions
```
p_X(x) = Σ_y p(x,y)
f_X(x) = ∫ f(x,y)dy
```

### Conditional Distributions
```
p(x|y) = p(x,y) / p_Y(y)
```

---

## 📊 Covariance and Correlation

### Covariance
```
Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)]
         = E[XY] - E[X]E[Y]
```

### Covariance Matrix
```
Σ = [Cov(X_i, X_j)]

For X = [X₁, X₂, ..., X_n]ᵀ:
Σ_ij = Cov(X_i, X_j)
```

```python
import numpy as np

# Generate correlated data
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov, 1000)

# Compute covariance matrix
cov_matrix = np.cov(data.T)
print(f"Covariance matrix:\n{cov_matrix}")

# Correlation matrix
corr_matrix = np.corrcoef(data.T)
print(f"Correlation matrix:\n{corr_matrix}")
```

---

## 🎯 Multivariate Gaussian

### Definition
```
X ~ N(μ, Σ)

f(x) = (1/√((2π)^n|Σ|)) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
```

**Properties:**
- Marginals are Gaussian
- Conditionals are Gaussian
- Linear transformations are Gaussian

```python
from scipy.stats import multivariate_normal

# Define distribution
mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])
mvn = multivariate_normal(mean, cov)

# PDF at point
x = np.array([0, 0])
print(f"PDF at origin: {mvn.pdf(x):.4f}")

# Generate samples
samples = mvn.rvs(size=1000)

# Plot
import matplotlib.pyplot as plt
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.axis('equal')
plt.title('Multivariate Gaussian Samples')
plt.show()
```

---

## 📈 Conditional Gaussian

For X ~ N(μ, Σ), partition:
```
X = [X₁]    μ = [μ₁]    Σ = [Σ₁₁  Σ₁₂]
    [X₂]        [μ₂]        [Σ₂₁  Σ₂₂]
```

**Conditional distribution:**
```
X₁|X₂=x₂ ~ N(μ₁|₂, Σ₁|₂)

μ₁|₂ = μ₁ + Σ₁₂Σ₂₂⁻¹(x₂ - μ₂)
Σ₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁
```

---

## 🎓 Applications in ML

### 1. Gaussian Processes
```python
# GP prior: f ~ GP(m, k)
# Posterior given data is also Gaussian
```

### 2. Kalman Filter
```python
# State estimation with Gaussian noise
# Prediction and update steps use conditional Gaussians
```

### 3. Mixture of Gaussians
```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3)
gmm.fit(data)
labels = gmm.predict(data)
```

---

**Multivariate distributions: foundation of probabilistic ML!**
