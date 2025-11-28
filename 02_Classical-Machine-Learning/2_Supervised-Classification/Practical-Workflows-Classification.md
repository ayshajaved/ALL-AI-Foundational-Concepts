# Practical Workflows - Classification

> **End-to-end classification projects** - From data to deployment

---

## 🎯 Complete Classification Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Explore
print(df.info())
print(df.describe())
print(df['target'].value_counts())

# 3. Prepare
X = df.drop('target', axis=1)
y = df['target']

# Encode labels if needed
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob[:, 1]):.4f}")

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

---

## 📊 Handling Class Imbalance

```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 1. Class weights
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train),
                                     y=y_train)
model_weighted = LogisticRegression(class_weight='balanced')
model_weighted.fit(X_train_scaled, y_train)

# 2. SMOTE (oversampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# 3. Undersampling
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train_scaled, y_train)

# 4. Combined
from imblearn.pipeline import Pipeline as ImbPipeline
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rus', RandomUnderSampler(random_state=42)),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train_scaled, y_train)
```

---

## 🎯 Model Comparison

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Plot
names = list(results.keys())
means = [results[name]['mean'] for name in names]
stds = [results[name]['std'] for name in names]

plt.barh(names, means, xerr=stds)
plt.xlabel('ROC-AUC')
plt.title('Model Comparison')
plt.show()
```

---

## 📈 Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Calibrate probabilities
model_calibrated = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_calibrated.fit(X_train_scaled, y_train)

# Compare calibration
y_prob_uncal = model.predict_proba(X_test_scaled)[:, 1]
y_prob_cal = model_calibrated.predict_proba(X_test_scaled)[:, 1]

# Calibration curve
prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_prob_uncal, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_prob_cal, n_bins=10)

plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='Uncalibrated')
plt.plot(prob_pred_cal, prob_true_cal, marker='s', label='Calibrated')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.legend()
plt.show()
```

---

## 🎯 Production Deployment

```python
import joblib

# Save model and preprocessor
joblib.dump(model, 'classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load and predict
loaded_model = joblib.load('classifier.pkl')
loaded_scaler = joblib.load('scaler.pkl')

def predict_new(X_new):
    X_scaled = loaded_scaler.transform(X_new)
    predictions = loaded_model.predict(X_scaled)
    probabilities = loaded_model.predict_proba(X_scaled)
    return predictions, probabilities

# API endpoint
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_new = pd.DataFrame(data)
    preds, probs = predict_new(X_new)
    return jsonify({
        'predictions': preds.tolist(),
        'probabilities': probs.tolist()
    })
```

---

## 🎓 Best Practices

1. **Always stratify splits for imbalanced data**
2. **Use appropriate metrics (not just accuracy)**
3. **Calibrate probabilities for decision-making**
4. **Monitor model performance in production**
5. **Version models and data**

---

**Complete workflows for production-ready classification!**
