# Stacking and Blending

> **Meta-learning** - Combine multiple models with a meta-learner

---

## 🎯 Stacking

### Idea
Use predictions of base models as features for meta-model

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Stacking
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train, y_train)
print(f"Accuracy: {stacking.score(X_test, y_test):.4f}")
```

---

## 📊 Blending

### Simpler alternative to stacking

```python
# 1. Split data
X_train1, X_train2, y_train1, y_train2 = train_test_split(
    X_train, y_train, test_size=0.5, random_state=42
)

# 2. Train base models on train1
rf = RandomForestClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)

rf.fit(X_train1, y_train1)
svm.fit(X_train1, y_train1)

# 3. Get predictions on train2
rf_pred = rf.predict_proba(X_train2)
svm_pred = svm.predict_proba(X_train2)

# 4. Train meta-model on train2
meta_features = np.column_stack([rf_pred, svm_pred])
meta_model = LogisticRegression()
meta_model.fit(meta_features, y_train2)
```

---

**Stacking: combine the best of multiple models!**
