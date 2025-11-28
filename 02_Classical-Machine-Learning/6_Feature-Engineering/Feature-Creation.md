# Feature Creation

> **Engineering new features** - Polynomial, interactions, domain-specific

---

## 🎯 Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

---

## 📊 Interaction Terms

```python
# Manual interactions
X['feature1_x_feature2'] = X['feature1'] * X['feature2']
X['feature1_div_feature2'] = X['feature1'] / (X['feature2'] + 1e-10)
```

---

## 📈 Time-Based Features

```python
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
```

---

**Create features that capture domain knowledge!**
