# Bias-Variance Tradeoff

> **Understanding model error** - Decomposing prediction error

---

## 🎯 Error Decomposition

```
Total Error = Bias² + Variance + Irreducible Error

Bias: error from wrong assumptions
Variance: error from sensitivity to training data
Irreducible: noise in data
```

---

## 📊 Visualizing Tradeoff

```python
from sklearn.model_selection import learning_curve

# Learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curves')
plt.show()
```

---

**Balance bias and variance for optimal performance!**
