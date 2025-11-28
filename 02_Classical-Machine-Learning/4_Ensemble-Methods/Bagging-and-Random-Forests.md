# Bagging and Random Forests

> **Bootstrap aggregating** - Reduce variance through ensemble learning

---

## 🎯 Bagging

### Idea
Train multiple models on bootstrap samples and average predictions

```
1. Create B bootstrap samples
2. Train model on each sample
3. Average predictions (regression) or vote (classification)
```

### Implementation

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)
bagging.fit(X_train, y_train)
print(f"Accuracy: {bagging.score(X_test, y_test):.4f}")
```

---

## 🌲 Random Forest

### Algorithm
```
1. For each tree:
   - Bootstrap sample
   - At each split, consider random subset of features
2. Average predictions
```

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

# Out-of-bag score
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)
print(f"OOB Score: {rf_oob.oob_score_:.4f}")
```

---

## 🎓 Interview Focus

1. **Bagging vs Boosting?**
   - Bagging: parallel, reduces variance
   - Boosting: sequential, reduces bias

2. **Random Forest advantages?**
   - Handles non-linear relationships
   - Feature importance
   - Robust to overfitting
   - Works well out-of-box

3. **Why random feature subset?**
   - Decorrelates trees
   - Reduces variance further
   - Prevents dominant features

---

**Random Forests: powerful, robust, easy to use!**
