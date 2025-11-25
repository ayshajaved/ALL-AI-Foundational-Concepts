# Information Theory Basics

> **Measuring information and uncertainty** - The mathematical foundation of communication and learning

---

## 📊 Shannon Entropy

### Definition

**Discrete:**
```
H(X) = -Σₓ p(x) log p(x)
```

**Interpretation:** Average surprise/uncertainty/information content

**Historical Note:** Claude Shannon introduced entropy in 1948, founding information theory and revolutionizing communication systems.

### Properties

1. **Non-negative:** H(X) ≥ 0
2. **Deterministic:** H(X) = 0 iff X is deterministic
3. **Maximum:** H(X) ≤ log|X| (uniform maximizes entropy)
4. **Concave:** H is concave in probability distribution

```python
import numpy as np

def entropy(p):
    """Compute Shannon entropy (base 2)"""
    p = np.array(p)
    p = p[p > 0]  # Remove zeros
    return -np.sum(p * np.log2(p))

# Examples
print(f"Fair coin: {entropy([0.5, 0.5]):.3f} bits")  # 1.0
print(f"Biased coin: {entropy([0.9, 0.1]):.3f} bits")  # 0.469
print(f"Deterministic: {entropy([1.0, 0.0]):.3f} bits")  # 0.0
print(f"Uniform (4): {entropy([0.25]*4):.3f} bits")  # 2.0
```

### Entropy Visualization

```
Entropy of Bernoulli(p):

1.0 |     ***
    |    *   *
0.8 |   *     *
    |  *       *
0.6 | *         *
    |*           *
0.4 |             *
    |              *
0.2 |               *
    |________________*___
0.0  0.0  0.5  1.0  p

Maximum at p=0.5 (maximum uncertainty)
```

---

## 🎯 Cross-Entropy

### Definition
```
H(p, q) = -Σₓ p(x) log q(x)
```

**Interpretation:** Expected surprise when using q to encode distribution p

**Relationship:**
```
H(p, q) = H(p) + D_KL(p||q)
```

### ML Loss Function

```python
# Binary cross-entropy
def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss"""
    epsilon = 1e-10  # Numerical stability
    return -np.mean(
        y_true * np.log(y_pred + epsilon) + 
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )

# Multi-class cross-entropy
def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy (one-hot encoded)"""
    epsilon = 1e-10
    return -np.sum(y_true * np.log(y_pred + epsilon))

# Example
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])
loss = binary_cross_entropy(y_true, y_pred)
print(f"BCE Loss: {loss:.4f}")
```

**Why Cross-Entropy for Classification?**
- Minimizing cross-entropy = minimizing KL divergence
- Equivalent to maximum likelihood estimation
- Convex for linear models

---

## 📈 KL Divergence

### Definition
```
D_KL(p||q) = Σₓ p(x) log(p(x)/q(x))
           = H(p,q) - H(p)
           = E_p[log p(x) - log q(x)]
```

**Interpretation:** "Distance" from q to p (not symmetric!)

### Properties

1. **Non-negative:** D_KL(p||q) ≥ 0
2. **Zero iff equal:** D_KL(p||q) = 0 ⟺ p = q
3. **NOT symmetric:** D_KL(p||q) ≠ D_KL(q||p)
4. **NOT a metric:** Doesn't satisfy triangle inequality

```python
def kl_divergence(p, q):
    """KL divergence D_KL(p||q)"""
    p = np.array(p)
    q = np.array(q)
    # Only sum where p > 0
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# Example: Asymmetry
p = np.array([0.5, 0.5])
q = np.array([0.9, 0.1])
print(f"D_KL(p||q) = {kl_divergence(p, q):.3f}")  # 0.510
print(f"D_KL(q||p) = {kl_divergence(q, p):.3f}")  # 0.755
# Different!
```

### Forward vs Reverse KL

```
Forward KL: D_KL(p||q)
- Mode-seeking
- q covers all modes of p
- Used in maximum likelihood

Reverse KL: D_KL(q||p)
- Mode-covering
- q focuses on one mode of p
- Used in variational inference
```

---

## 🔗 Mutual Information

### Definition
```
I(X;Y) = D_KL(p(x,y) || p(x)p(y))
       = H(X) + H(Y) - H(X,Y)
       = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
```

**Interpretation:** Information shared between X and Y

### Properties

1. **Symmetric:** I(X;Y) = I(Y;X)
2. **Non-negative:** I(X;Y) ≥ 0
3. **Zero iff independent:** I(X;Y) = 0 ⟺ X ⊥ Y
4. **Bounded:** I(X;Y) ≤ min(H(X), H(Y))

```python
def mutual_information(joint_prob):
    """Compute mutual information from joint distribution"""
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

# Example
joint = np.array([[0.4, 0.1],
                  [0.1, 0.4]])
mi = mutual_information(joint)
print(f"I(X;Y) = {mi:.3f} bits")
```

### Conditional Mutual Information

```
I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
         = E_Z[D_KL(p(x,y|z) || p(x|z)p(y|z))]
```

---

## 🎯 Advanced Concepts

### Jensen-Shannon Divergence

**Symmetric version of KL:**
```
JSD(p||q) = ½D_KL(p||m) + ½D_KL(q||m)
where m = ½(p + q)
```

**Properties:**
- Symmetric: JSD(p||q) = JSD(q||p)
- Bounded: 0 ≤ JSD ≤ 1 (in bits)
- Square root is a metric

```python
def jensen_shannon_divergence(p, q):
    """Jensen-Shannon divergence"""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
```

### Differential Entropy

**For continuous distributions:**
```
h(X) = -∫ f(x) log f(x) dx
```

**Can be negative!**

```python
from scipy.stats import norm

# Gaussian differential entropy
mu, sigma = 0, 1
h = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
print(f"h(N(0,1)) = {h:.3f} nats")  # 1.419 nats
```

---

## 🎯 Applications in ML

### 1. Loss Functions

```python
# Cross-entropy = KL divergence + constant
# Minimizing cross-entropy ⟺ minimizing KL divergence

# Softmax + Cross-Entropy
def softmax_cross_entropy(logits, labels):
    """Numerically stable softmax + cross-entropy"""
    # Softmax
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = logits_shifted - np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
    
    # Cross-entropy
    n = len(labels)
    loss = -np.sum(log_probs[range(n), labels]) / n
    return loss
```

### 2. Variational Inference

```
ELBO = E_q[log p(x,z)] - E_q[log q(z)]
     = log p(x) - D_KL(q(z)||p(z|x))

Maximize ELBO ⟺ Minimize KL divergence
```

```python
def elbo(q_samples, log_p_joint, log_q):
    """Evidence Lower Bound"""
    return np.mean(log_p_joint(q_samples) - log_q(q_samples))
```

### 3. Information Bottleneck

```
Minimize: I(X;Z) - βI(Z;Y)

Compress X → Z while preserving info about Y
```

**Applications:**
- Deep learning theory
- Feature selection
- Representation learning

### 4. Generative Models

**GANs:**
```
Jensen-Shannon divergence between real and generated distributions
```

**VAEs:**
```
ELBO = E_q[log p(x|z)] - D_KL(q(z|x)||p(z))
```

### 5. Feature Selection

```python
def information_gain(X, y, feature_idx):
    """Information gain for feature selection"""
    # I(Y;X_feature) = H(Y) - H(Y|X_feature)
    H_y = entropy(np.bincount(y) / len(y))
    
    # Conditional entropy
    H_y_given_x = 0
    for val in np.unique(X[:, feature_idx]):
        mask = X[:, feature_idx] == val
        p_val = np.mean(mask)
        if p_val > 0:
            y_subset = y[mask]
            H_y_given_x += p_val * entropy(np.bincount(y_subset) / len(y_subset))
    
    return H_y - H_y_given_x
```

---

## 🛡️ Numerical Stability

```python
# Log-sum-exp trick
def log_sum_exp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

# Stable softmax
def stable_softmax(x):
    """Numerically stable softmax"""
    x_shifted = x - np.max(x)
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted))
```

---

## 🎓 Advanced Exercises

### Exercise 1: Prove Non-negativity of KL
**Problem:** Prove D_KL(p||q) ≥ 0 using Jensen's inequality

**Hint:** log is concave

### Exercise 2: Maximum Entropy Distribution
**Problem:** Show that uniform distribution maximizes entropy

### Exercise 3: Mutual Information Bound
**Problem:** Prove I(X;Y) ≤ min(H(X), H(Y))

---

## 🎓 Interview Focus

### Key Questions

1. **What is entropy?**
   - Average information content
   - Measure of uncertainty
   - H(X) = -Σ p(x) log p(x)

2. **Why cross-entropy for classification?**
   - Equivalent to maximum likelihood
   - Minimizes KL divergence
   - Convex for linear models

3. **KL divergence properties?**
   - Non-negative
   - Zero iff distributions equal
   - NOT symmetric

4. **Mutual information vs correlation?**
   - MI: general dependence (nonlinear)
   - Correlation: linear dependence only
   - MI = 0 ⟺ independence

5. **ELBO in VAEs?**
   - Evidence Lower Bound
   - log p(x) - KL(q||p)
   - Maximizing ELBO trains VAE

### Must-Know Formulas

```
Entropy: H(X) = -Σ p(x) log p(x)
Cross-entropy: H(p,q) = -Σ p(x) log q(x)
KL divergence: D_KL(p||q) = Σ p(x) log(p(x)/q(x))
Mutual info: I(X;Y) = H(X) + H(Y) - H(X,Y)
ELBO: log p(x) - D_KL(q(z)||p(z|x))
```

---

## 📚 References

- **Books:**
  - "Elements of Information Theory" - Cover & Thomas
  - "Information Theory, Inference, and Learning Algorithms" - MacKay

- **Papers:**
  - "A Mathematical Theory of Communication" - Shannon (1948)
  - "The Information Bottleneck Method" - Tishby et al.

- **Online:**
  - [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)

---

**Information theory: the mathematics of learning and communication!**
