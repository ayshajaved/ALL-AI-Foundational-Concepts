# Evaluation Metrics

> **Measuring model performance** - Classification and regression metrics

---

## 🎯 Classification Metrics

### Confusion Matrix
```
              Predicted
            Pos    Neg
Actual Pos  TP     FN
       Neg  FP     TN
```

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision & Recall
```
Precision = TP / (TP + FP)  # How many predicted positives are correct?
Recall = TP / (TP + FN)     # How many actual positives found?
```

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Implementation

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
```

---

## 📊 ROC-AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## 📈 Regression Metrics

### MSE, RMSE, MAE
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

---

**Choose the right metric for your problem!**
