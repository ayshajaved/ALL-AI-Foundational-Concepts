# Weight Regularization

> **Constraining complexity** - L1, L2, and Label Smoothing

---

## ⚖️ L1 and L2 Regularization

Add a penalty term to the loss function.

$$ J(\theta) = Loss + \lambda R(\theta) $$

### L2 Regularization (Weight Decay)
$$ R(\theta) = \sum \theta_i^2 $$
- Penalizes large weights.
- **Effect:** Diffuse weights (many small weights).
- **Bayesian View:** Gaussian Prior.

```python
# In PyTorch, enabled via weight_decay
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### L1 Regularization (Lasso)
$$ R(\theta) = \sum |\theta_i| $$
- **Effect:** Sparsity (many weights become exactly zero).
- **Bayesian View:** Laplacian Prior.

---

## 🏷️ Label Smoothing

**Problem:** Using one-hot targets `[0, 1, 0]` forces model to be extremely confident (logits $\to \infty$). Causes overfitting.
**Solution:** Soften targets. `[0.05, 0.9, 0.05]`.

$$ y_{new} = (1-\epsilon)y_{onehot} + \epsilon / K $$

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 🛑 Early Stopping

(Covered in Training workflows, but fundamentally a regularization technique).
Stops optimization before it overfits the training noise.

---

## 🎓 Interview Focus

1.  **L1 vs L2 Regularization?**
    - L1 leads to sparse solutions (feature selection).
    - L2 leads to small, diffuse weights (better generalization usually).

2.  **Why Label Smoothing?**
    - Prevents the model from becoming over-confident.
    - Improves calibration and generalization.

---

**Regularization: Keeping models simple!**
