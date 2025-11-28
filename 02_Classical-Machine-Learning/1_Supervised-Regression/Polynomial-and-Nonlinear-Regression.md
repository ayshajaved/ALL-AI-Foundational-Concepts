# Polynomial and Nonlinear Regression

> **Beyond linear relationships** - Capturing curves and complex patterns

---

## 🎯 Polynomial Regression

### Idea
Transform features to capture nonlinear relationships

```
y = w₀ + w₁x + w₂x² + w₃x³ + ...

Still linear in parameters!
```

### Implementation

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Create polynomial features
X = np.random.rand(100, 1) * 10
y = 2*X.squeeze()**2 + 3*X.squeeze() + np.random.randn(100)*5

# Manual
X_poly = np.c_[X, X**2, X**3]
model = LinearRegression()
model.fit(X_poly, y)

# Using sklearn
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly_sklearn = poly.fit_transform(X)

# Pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])
pipeline.fit(X, y)

# Predict
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = pipeline.predict(X_test)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X_test, y_pred, 'r-', label='Polynomial fit')
plt.legend()
plt.show()
```

---

## 📊 Bias-Variance Tradeoff

### Underfitting (High Bias)
- Model too simple
- Doesn't capture patterns
- Poor train AND test performance

### Overfitting (High Variance)
- Model too complex
- Captures noise
- Good train, poor test performance

### Finding the Sweet Spot

```python
from sklearn.model_selection import cross_val_score

degrees = range(1, 15)
train_scores = []
val_scores = []

for degree in degrees:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Train score
    pipeline.fit(X_train, y_train)
    train_scores.append(pipeline.score(X_train, y_train))
    
    # Validation score (CV)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    val_scores.append(cv_scores.mean())

# Plot
plt.plot(degrees, train_scores, label='Train')
plt.plot(degrees, val_scores, label='Validation')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.legend()
plt.title('Bias-Variance Tradeoff')
plt.show()
```

---

## 🎯 Basis Functions

### General Form
```
y = Σⱼ wⱼφⱼ(x)

φⱼ: basis functions
```

### Common Basis Functions

**1. Polynomial**
```
φⱼ(x) = xʲ
```

**2. Gaussian (RBF)**
```
φⱼ(x) = exp(-||x - μⱼ||²/(2σ²))
```

**3. Sigmoid**
```
φⱼ(x) = 1/(1 + exp(-aⱼx + bⱼ))
```

```python
# Gaussian basis functions
def gaussian_basis(X, centers, sigma=1.0):
    """Create Gaussian basis functions"""
    n_samples = len(X)
    n_centers = len(centers)
    features = np.zeros((n_samples, n_centers))
    
    for i, center in enumerate(centers):
        features[:, i] = np.exp(-np.sum((X - center)**2, axis=1) / (2*sigma**2))
    
    return features

# Use as features
centers = np.linspace(X.min(), X.max(), 10).reshape(-1, 1)
X_rbf = gaussian_basis(X, centers)

model = LinearRegression()
model.fit(X_rbf, y)
```

---

## 📈 Regularization for Polynomial Models

**Problem:** High-degree polynomials overfit

**Solution:** Use Ridge or Lasso

```python
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=10)),
    ('ridge', Ridge(alpha=1.0))
])
pipeline.fit(X_train, y_train)
```

---

## 🎓 Interview Focus

### Key Questions

1. **Polynomial regression vs linear?**
   - Still linear in parameters
   - Nonlinear in features
   - Can capture curves

2. **Bias-variance tradeoff?**
   - Bias: error from wrong assumptions
   - Variance: sensitivity to training data
   - Total error = bias² + variance + noise

3. **How to prevent overfitting?**
   - Cross-validation
   - Regularization
   - More data
   - Simpler model

4. **When to use polynomial features?**
   - Clear nonlinear relationship
   - Small number of features
   - With regularization

---

## 📚 References

- **Books:** "Pattern Recognition and Machine Learning" - Bishop

---

**Polynomial regression: simple way to capture nonlinearity!**
