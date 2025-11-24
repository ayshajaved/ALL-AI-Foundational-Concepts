# Information Theory Basics

> **Measuring information and uncertainty** - Entropy, KL divergence, mutual information

---

## 📊 Entropy

### Shannon Entropy

**Discrete:**
```
H(X) = -Σₓ p(x) log p(x)
```

**Interpretation:** Average surprise/uncertainty

**Properties:**
- H(X) ≥ 0
- H(X) = 0 iff X is deterministic
- H(X) ≤ log|X| (uniform maximizes entropy)

```python
import numpy as np

def entropy(p):
    """Compute Shannon entropy"""
    p = np.array(p)
    p = p[p > 0]  # Remove zeros
    return -np.sum(p * np.log2(p))

# Examples
print(f"Fair coin: {entropy([0.5, 0.5]):.3f} bits")  # 1.0
print(f"Biased coin: {entropy([0.9, 0.1]):.3f} bits")  # 0.469
print(f"Deterministic: {entropy([1.0, 0.0]):.3f} bits")  # 0.0
```

---

## 🎯 Cross-Entropy

### Definition
```
H(p, q) = -Σₓ p(x) log q(x)
```

**Interpretation:** Expected surprise when using q to encode p

**ML Loss Function:**
```python
# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    """Binary cross-entropy"""
    return -np.mean(y_true * np.log(y_pred + 1e-10) + 
                    (1 - y_true) * np.log(1 - y_pred + 1e-10))

# Multi-class cross-entropy
def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy"""
    return -np.sum(y_true * np.log(y_pred + 1e-10))
```

---

## 📈 KL Divergence

### Definition
```
D_KL(p||q) = Σₓ p(x) log(p(x)/q(x))
           = H(p,q) - H(p)
```

**Properties:**
- D_KL(p||q) ≥ 0
- D_KL(p||q) = 0 iff p = q
- NOT symmetric: D_KL(p||q) ≠ D_KL(q||p)

```python
def kl_divergence(p, q):
    """KL divergence D_KL(p||q)"""
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log(p / q))

# Example
p = np.array([0.5, 0.5])
q = np.array([0.9, 0.1])
print(f"D_KL(p||q) = {kl_divergence(p, q):.3f}")
print(f"D_KL(q||p) = {kl_divergence(q, p):.3f}")
# Different!
```

---

## 🔗 Mutual Information

### Definition
```
I(X;Y) = D_KL(p(x,y) || p(x)p(y))
       = H(X) + H(Y) - H(X,Y)
       = H(X) - H(X|Y)
```

**Interpretation:** Information shared between X and Y

```python
def mutual_information(joint_prob):
    """Compute mutual information"""
    # Marginals
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    
    # Mutual information
    mi = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if joint_prob[i,j] > 0:
                mi += joint_prob[i,j] * np.log(
                    joint_prob[i,j] / (p_x[i] * p_y[j])
                )
    return mi
```

---

## 🎯 Applications in ML

### 1. Loss Functions
```python
# Cross-entropy = KL divergence + constant
# Minimizing cross-entropy = minimizing KL divergence
```

### 2. Variational Inference
```
ELBO = E_q[log p(x,z)] - E_q[log q(z)]
     = log p(x) - D_KL(q(z)||p(z|x))
```

### 3. Information Bottleneck
```
Minimize: I(X;Z) - βI(Z;Y)

Compress X → Z while preserving info about Y
```

---

## 📚 References

- **Books:** "Elements of Information Theory" - Cover & Thomas

---

**Information theory: the mathematics of communication and learning!**
