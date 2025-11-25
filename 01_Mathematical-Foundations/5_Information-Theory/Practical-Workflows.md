# Practical Workflows - Information Theory

> **Hands-on information theory in Python** - Computing entropy, KL divergence, and mutual information

---

## 🛠️ Computing Entropy

```python
import numpy as np
from scipy.stats import entropy as scipy_entropy

def entropy(p, base=2):
    """Compute Shannon entropy"""
    p = np.array(p)
    p = p[p > 0]  # Remove zeros
    if base == 2:
        return -np.sum(p * np.log2(p))
    elif base == np.e:
        return -np.sum(p * np.log(p))
    else:
        return -np.sum(p * np.log(p)) / np.log(base)

# Using scipy
from scipy.stats import entropy
p = [0.5, 0.3, 0.2]
H = entropy(p, base=2)
print(f"Entropy: {H:.3f} bits")
```

---

## 📊 KL Divergence

```python
def kl_divergence(p, q, base=2):
    """Compute KL divergence D_KL(p||q)"""
    p = np.array(p)
    q = np.array(q)
    mask = p > 0
    
    if base == 2:
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))
    else:
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# Using scipy
from scipy.special import kl_div
kl = np.sum(kl_div(p, q))
```

---

## 🔗 Mutual Information

```python
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Discrete variables
x = [0, 0, 1, 1, 1, 2, 2]
y = [0, 1, 0, 1, 1, 0, 1]
mi = mutual_info_score(x, y)

# Continuous features
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
mi_scores = mutual_info_classif(X, y)
```

---

## 🎯 PyTorch Loss Functions

```python
import torch
import torch.nn.functional as F

# Binary cross-entropy
y_true = torch.tensor([1.0, 0.0, 1.0])
y_pred = torch.tensor([0.9, 0.1, 0.8])
bce = F.binary_cross_entropy(y_pred, y_true)

# Cross-entropy (with logits)
logits = torch.randn(3, 5)  # 3 samples, 5 classes
targets = torch.tensor([1, 0, 4])
ce = F.cross_entropy(logits, targets)

# KL divergence
p = F.softmax(torch.randn(10), dim=0)
q = F.softmax(torch.randn(10), dim=0)
kl = F.kl_div(q.log(), p, reduction='batchmean')
```

---

**Master these tools for information-theoretic ML!**
