# Feature Selection

> **Selecting relevant features** - Filter, wrapper, and embedded methods

---

## 🎯 Filter Methods

### Correlation
```python
# Remove highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_filtered = X.drop(columns=to_drop)
```

### Variance Threshold
```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_high_var = selector.fit_transform(X)
```

---

## 📊 Wrapper Methods

### Recursive Feature Elimination
```python
from sklearn.feature_selection import RFE

rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
print(f"Selected features: {rfe.support_}")
```

---

## 📈 Embedded Methods

### L1 Regularization (Lasso)
```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected = np.abs(lasso.coef_) > 0
X_selected = X[:, selected]
```

---

**Select features to improve performance and reduce overfitting!**
