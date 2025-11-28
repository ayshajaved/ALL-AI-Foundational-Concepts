# Advanced Classification Techniques

> **Specialized classification methods** - Multi-label, ordinal, imbalance handling

---

## 🎯 Multi-Label Classification

### Problem
Each sample can belong to multiple classes simultaneously

**Example:** Document tagging, image labeling

### Binary Relevance

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Multi-label data
# y shape: (n_samples, n_classes)
y_multilabel = np.array([
    [1, 0, 1],  # Sample belongs to class 0 and 2
    [0, 1, 0],  # Sample belongs to class 1
    [1, 1, 0],  # Sample belongs to class 0 and 1
])

# Train separate classifier for each label
multi_clf = MultiOutputClassifier(RandomForestClassifier())
multi_clf.fit(X, y_multilabel)

# Predict
y_pred = multi_clf.predict(X_test)
```

### Classifier Chains

```python
from sklearn.multioutput import ClassifierChain

# Chain classifiers (captures label dependencies)
chain = ClassifierChain(RandomForestClassifier(), order='random', random_state=42)
chain.fit(X, y_multilabel)
```

### Evaluation Metrics

```python
from sklearn.metrics import hamming_loss, jaccard_score, f1_score

# Hamming loss (fraction of wrong labels)
hamming = hamming_loss(y_test, y_pred)

# Jaccard score (intersection over union)
jaccard = jaccard_score(y_test, y_pred, average='samples')

# F1 score
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_samples = f1_score(y_test, y_pred, average='samples')

print(f"Hamming Loss: {hamming:.4f}")
print(f"Jaccard Score: {jaccard:.4f}")
print(f"F1 (micro): {f1_micro:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")
```

---

## 📊 Ordinal Classification

### Problem
Classes have natural ordering (e.g., ratings: bad < ok < good)

### Ordinal Logistic Regression

```python
from mord import LogisticAT  # pip install mord

# Ordinal labels: 0 < 1 < 2 < 3
y_ordinal = np.array([0, 1, 2, 3, 1, 2, 0, 3, ...])

# Ordinal classifier
ord_clf = LogisticAT()
ord_clf.fit(X, y_ordinal)

# Predict
y_pred = ord_clf.predict(X_test)
```

### Threshold-based Approach

```python
# Convert ordinal to multiple binary problems
# y=0: all negative
# y=1: first positive, rest negative
# y=2: first two positive, last negative
# y=3: all positive

def ordinal_to_binary(y, n_classes):
    """Convert ordinal labels to binary matrix"""
    n_samples = len(y)
    Y_binary = np.zeros((n_samples, n_classes-1))
    for i in range(n_samples):
        Y_binary[i, :y[i]] = 1
    return Y_binary

# Train
Y_binary = ordinal_to_binary(y_ordinal, n_classes=4)
classifiers = []
for i in range(3):
    clf = LogisticRegression()
    clf.fit(X, Y_binary[:, i])
    classifiers.append(clf)

# Predict
def predict_ordinal(X, classifiers):
    predictions = []
    for clf in classifiers:
        predictions.append(clf.predict_proba(X)[:, 1])
    predictions = np.array(predictions).T
    return (predictions > 0.5).sum(axis=1)
```

---

## 🎯 Advanced Imbalance Handling

### SMOTE Variants

```python
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ADASYN (Adaptive Synthetic)
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

# Borderline SMOTE
borderline = BorderlineSMOTE(random_state=42)
X_borderline, y_borderline = borderline.fit_resample(X, y)
```

### Ensemble Methods for Imbalance

```python
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

# Balanced Random Forest
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X, y)

# Easy Ensemble
ee = EasyEnsembleClassifier(n_estimators=10, random_state=42)
ee.fit(X, y)
```

### Cost-Sensitive Learning

```python
# Custom class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weight_dict = {i: w for i, w in enumerate(class_weights)}

# Use in classifier
clf = RandomForestClassifier(class_weight=weight_dict)
clf.fit(X, y)
```

---

## 📈 Probability Calibration

### Why Calibrate?
Raw probabilities from classifiers may not reflect true probabilities

### Calibration Methods

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Platt scaling (logistic regression)
calibrated_platt = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
calibrated_platt.fit(X_train, y_train)

# Isotonic regression
calibrated_isotonic = CalibratedClassifierCV(clf, method='isotonic', cv=5)
calibrated_isotonic.fit(X_train, y_train)

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.plot(prob_pred, prob_true, marker='o', label='Calibrated')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()
```

---

## 🎓 Interview Focus

### Key Questions

1. **Multi-label vs multi-class?**
   - Multi-class: one label per sample
   - Multi-label: multiple labels per sample
   - Different evaluation metrics

2. **Ordinal vs nominal classification?**
   - Ordinal: ordered classes
   - Nominal: unordered classes
   - Ordinal methods preserve order

3. **SMOTE limitations?**
   - Can create unrealistic samples
   - Doesn't work well in high dimensions
   - May amplify noise

4. **When to calibrate probabilities?**
   - When using probabilities for decisions
   - Ensemble methods often need calibration
   - Check calibration curve first

---

## 📚 References

- **Papers:**
  - "SMOTE: Synthetic Minority Over-sampling Technique" - Chawla et al.
  - "A Study of Probability Calibration" - Niculescu-Mizil & Caruana

---

**Advanced classification: handling complex real-world scenarios!**
