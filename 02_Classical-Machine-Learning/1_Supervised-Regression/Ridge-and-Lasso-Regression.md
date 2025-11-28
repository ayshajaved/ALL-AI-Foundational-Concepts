# Ridge and Lasso Regression

> **Regularized linear regression** - Preventing overfitting with L1 and L2 penalties

---

## 🎯 Why Regularization?

**Problem:** Ordinary least squares can overfit when:
- Too many features (p > n)
- Multicollinearity
- Noisy data

**Solution:** Add penalty term to constrain weights

---

## 📊 Ridge Regression (L2 Regularization)

### Objective
```
L(w) = ||y - Xw||² + λ||w||²
     = Σᵢ(yᵢ - wᵀxᵢ)² + λΣⱼwⱼ²

λ: regularization strength
```

### Closed-Form Solution
```
w* = (XᵀX + λI)⁻¹Xᵀy

Always invertible for λ > 0!
```

```python
import numpy as np

def ridge_regression(X, y, lambda_=1.0):
    """Ridge regression closed form"""
    n_features = X.shape[1]
    X_bias = np.c_[np.ones(len(X)), X]
    
    # Add regularization (don't regularize bias)
    reg_matrix = lambda_ * np.eye(n_features + 1)
    reg_matrix[0, 0] = 0  # Don't regularize intercept
    
    w = np.linalg.inv(X_bias.T @ X_bias + reg_matrix) @ X_bias.T @ y
    return w

# Scikit-learn
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha = λ
model.fit(X_train, y_train)
```

---

## 🎯 Lasso Regression (L1 Regularization)

### Objective
```
L(w) = ||y - Xw||² + λ||w||₁
     = Σᵢ(yᵢ - wᵀxᵢ)² + λΣⱼ|wⱼ|
```

**Key Property:** Induces sparsity (sets some weights to exactly 0)

### No Closed Form!

Use iterative methods:
- Coordinate descent
- Proximal gradient (ISTA/FISTA)

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Check sparsity
n_nonzero = np.sum(model.coef_ != 0)
print(f"Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")
```

---

## 📈 Elastic Net

### Combines L1 and L2
```
L(w) = ||y - Xw||² + λ₁||w||₁ + λ₂||w||²
     = ||y - Xw||² + λ(ρ||w||₁ + (1-ρ)||w||²)

ρ: mixing parameter (0 = Ridge, 1 = Lasso)
```

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio = ρ
model.fit(X_train, y_train)
```

---

## 🎯 Choosing λ (Cross-Validation)

```python
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score

# Ridge with CV
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")

# Lasso with CV
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
print(f"Best alpha: {lasso_cv.alpha_}")

# Manual CV
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': alphas}
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

---

## 📊 Regularization Path

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, ridge_path

# Lasso path
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train)

plt.figure(figsize=(12, 5))

# Lasso
plt.subplot(121)
for coef in coefs_lasso:
    plt.plot(alphas_lasso, coef)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Lasso Path')

# Ridge
alphas_ridge = np.logspace(-3, 3, 100)
coefs_ridge = []
for alpha in alphas_ridge:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs_ridge.append(ridge.coef_)

plt.subplot(122)
plt.plot(alphas_ridge, coefs_ridge)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Ridge Path')

plt.tight_layout()
plt.show()
```

---

## 🎓 Interview Focus

### Key Questions

1. **Ridge vs Lasso?**
   - Ridge: L2, shrinks coefficients, doesn't zero out
   - Lasso: L1, feature selection, sparse solutions

2. **Why Lasso for feature selection?**
   - L1 penalty creates sparsity
   - Sets irrelevant features to exactly 0
   - Automatic feature selection

3. **When to use each?**
   - Ridge: many small/medium effects
   - Lasso: few large effects
   - Elastic Net: best of both

4. **How to choose λ?**
   - Cross-validation
   - Plot regularization path
   - Domain knowledge

---

## 📚 References

- **Papers:** "Regression Shrinkage and Selection via the Lasso" - Tibshirani

---

**Regularization: the key to generalization!**
