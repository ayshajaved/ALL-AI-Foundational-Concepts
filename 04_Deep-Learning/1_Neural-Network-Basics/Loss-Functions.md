# Loss Functions

> **Measuring prediction error** - Choosing the right loss for your task

---

## 🎯 Regression Losses

### Mean Squared Error (MSE)
```
MSE = (1/n)Σ(yᵢ - ŷᵢ)²
```

```python
import torch.nn as nn

mse_loss = nn.MSELoss()
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.1, 2.2, 2.9])
loss = mse_loss(y_pred, y_true)
```

**Use:** Standard regression

### Mean Absolute Error (MAE)
```
MAE = (1/n)Σ|yᵢ - ŷᵢ|
```

```python
mae_loss = nn.L1Loss()
```

**Advantage:** Robust to outliers

### Huber Loss
```
L(y, ŷ) = ½(y - ŷ)² if |y - ŷ| ≤ δ
          δ|y - ŷ| - ½δ² otherwise
```

```python
huber_loss = nn.HuberLoss(delta=1.0)
```

**Combines:** MSE (small errors) + MAE (large errors)

---

## 📊 Classification Losses

### Binary Cross-Entropy
```
BCE = -[y log(ŷ) + (1-y)log(1-ŷ)]
```

```python
bce_loss = nn.BCELoss()  # Requires sigmoid output
# Or
bce_with_logits = nn.BCEWithLogitsLoss()  # Includes sigmoid
```

**Use:** Binary classification

### Cross-Entropy (Multi-class)
```
CE = -Σᵢ yᵢ log(ŷᵢ)
```

```python
ce_loss = nn.CrossEntropyLoss()  # Includes softmax

# Example
logits = torch.randn(32, 10)  # Batch of 32, 10 classes
targets = torch.randint(0, 10, (32,))
loss = ce_loss(logits, targets)
```

**Use:** Multi-class classification

---

## 🎯 Advanced Losses

### Focal Loss
```
FL = -α(1 - pₜ)^γ log(pₜ)

γ: focusing parameter
α: balancing parameter
```

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

**Use:** Class imbalance

### Dice Loss
```
Dice = 1 - (2|X∩Y|)/(|X| + |Y|)
```

**Use:** Segmentation

---

## 🎓 Interview Focus

1. **MSE vs MAE?**
   - MSE: penalizes large errors more
   - MAE: robust to outliers

2. **Why cross-entropy for classification?**
   - Probabilistic interpretation
   - Convex optimization
   - Better gradients than MSE

3. **Focal loss purpose?**
   - Handles class imbalance
   - Focuses on hard examples

---

**Loss functions: guiding the learning process!**
