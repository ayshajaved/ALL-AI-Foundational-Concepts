# Boosting Methods

> **Sequential ensemble learning** - AdaBoost, Gradient Boosting, XGBoost

---

## 🎯 AdaBoost

### Idea
Train weak learners sequentially, focusing on misclassified examples

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
print(f"Accuracy: {ada.score(X_test, y_test):.4f}")
```

---

## 📊 Gradient Boosting

### Algorithm
```
1. Initialize with constant prediction
2. For m = 1 to M:
   - Compute residuals
   - Fit tree to residuals
   - Update predictions
```

```python
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
```

---

## 🚀 XGBoost

```python
import xgboost as xgb

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

---

## 🎓 Interview Focus

1. **Boosting vs Bagging?**
   - Boosting: sequential, reduces bias
   - Bagging: parallel, reduces variance

2. **Learning rate in boosting?**
   - Controls contribution of each tree
   - Lower = more trees needed, better generalization
   - Higher = faster training, risk overfitting

---

**Boosting: powerful sequential ensembles!**
