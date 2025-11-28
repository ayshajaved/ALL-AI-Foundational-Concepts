# Cross-Validation

> **Robust model evaluation** - K-fold, stratified, time series CV

---

## 🎯 K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# K-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 📊 Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

# Stratified (preserves class distribution)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='roc_auc')
```

---

## 📈 Time Series CV

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series (no shuffle, forward-only)
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

---

**Cross-validation: get reliable performance estimates!**
