# Advanced Regression Methods

> **Specialized regression techniques** - Quantile, robust, Bayesian, and time series

---

## 🎯 Quantile Regression

### Idea
Predict conditional quantiles instead of conditional mean

```
Minimize: Σᵢ ρτ(yᵢ - ŷᵢ)

ρτ(u) = u(τ - I(u < 0))  # Check function
τ: quantile (e.g., 0.5 for median)
```

### Implementation

```python
from sklearn.linear_model import QuantileRegressor
import numpy as np

# Generate data with heteroscedastic noise
np.random.seed(42)
X = np.random.randn(1000, 1) * 10
y = 2*X.squeeze() + 5 + np.random.randn(1000) * (1 + 0.5*np.abs(X.squeeze()))

# Fit multiple quantiles
quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
predictions = {}

for q in quantiles:
    qr = QuantileRegressor(quantile=q, alpha=0)
    qr.fit(X, y)
    predictions[q] = qr.predict(X)

# Plot
X_sorted = np.sort(X, axis=0)
plt.scatter(X, y, alpha=0.3, label='Data')
for q in quantiles:
    idx = np.argsort(X.squeeze())
    plt.plot(X[idx], predictions[q][idx], label=f'τ={q}')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Quantile Regression')
plt.show()
```

### Use Cases
- Uncertainty quantification
- Heteroscedastic data
- Risk assessment
- Outlier-robust predictions

---

## 📊 Robust Regression

### RANSAC (RANdom SAmple Consensus)

**Idea:** Fit model to inliers, ignore outliers

```python
from sklearn.linear_model import RANSACRegressor

# Add outliers
X_outliers = np.random.randn(50, 1) * 10
y_outliers = np.random.randn(50) * 50
X_with_outliers = np.vstack([X, X_outliers])
y_with_outliers = np.hstack([y, y_outliers])

# RANSAC
ransac = RANSACRegressor(random_state=42)
ransac.fit(X_with_outliers, y_with_outliers)

# Inlier mask
inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask

# Plot
plt.scatter(X_with_outliers[inlier_mask], y_with_outliers[inlier_mask], 
           c='blue', label='Inliers')
plt.scatter(X_with_outliers[outlier_mask], y_with_outliers[outlier_mask], 
           c='red', label='Outliers')
plt.plot(X_with_outliers, ransac.predict(X_with_outliers), 
        'g-', label='RANSAC')
plt.legend()
plt.show()
```

### Huber Regression

**Idea:** Less sensitive to outliers than OLS

```python
from sklearn.linear_model import HuberRegressor

huber = HuberRegressor(epsilon=1.35)  # Tuning parameter
huber.fit(X_with_outliers, y_with_outliers)

# Compare with OLS
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(X_with_outliers, y_with_outliers)

print(f"Huber coef: {huber.coef_}")
print(f"OLS coef: {ols.coef_}")
```

---

## 🎯 Bayesian Linear Regression

### Probabilistic approach with uncertainty

```python
from sklearn.linear_model import BayesianRidge

# Bayesian Ridge
bayesian = BayesianRidge(compute_score=True)
bayesian.fit(X, y)

# Predictions with uncertainty
y_pred, y_std = bayesian.predict(X, return_std=True)

# Plot with confidence intervals
X_sorted_idx = np.argsort(X.squeeze())
plt.scatter(X, y, alpha=0.3, label='Data')
plt.plot(X[X_sorted_idx], y_pred[X_sorted_idx], 'r-', label='Mean prediction')
plt.fill_between(X[X_sorted_idx].squeeze(),
                 (y_pred - 2*y_std)[X_sorted_idx],
                 (y_pred + 2*y_std)[X_sorted_idx],
                 alpha=0.2, label='95% CI')
plt.legend()
plt.title('Bayesian Linear Regression')
plt.show()
```

### Automatic Relevance Determination (ARD)

```python
from sklearn.linear_model import ARDRegression

# ARD - automatic feature selection
ard = ARDRegression(compute_score=True)
ard.fit(X_multi, y)

# Features with low precision are irrelevant
print(f"Feature relevance: {1/ard.lambda_}")
```

---

## 📈 Time Series Regression

### Autoregressive Models

```python
from statsmodels.tsa.ar_model import AutoReg

# AR model
model = AutoReg(y_time_series, lags=5)
results = model.fit()

# Forecast
forecast = results.predict(start=len(y_time_series), end=len(y_time_series)+10)
```

### ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA

# ARIMA(p, d, q)
model = ARIMA(y_time_series, order=(5, 1, 0))
results = model.fit()

# Forecast
forecast = results.forecast(steps=10)
```

---

## 🎓 Interview Focus

### Key Questions

1. **Quantile vs mean regression?**
   - Quantile: predicts conditional quantiles
   - Mean: predicts conditional mean
   - Quantile more robust to outliers

2. **When to use robust regression?**
   - Data has outliers
   - Can't remove outliers
   - Want robust estimates

3. **RANSAC vs Huber?**
   - RANSAC: identifies outliers explicitly
   - Huber: downweights outliers
   - RANSAC better for severe outliers

4. **Bayesian regression advantages?**
   - Uncertainty quantification
   - Automatic feature selection (ARD)
   - Regularization through priors

---

## 📚 References

- **Books:** "Quantile Regression" - Koenker

---

**Advanced regression: specialized tools for specialized problems!**
