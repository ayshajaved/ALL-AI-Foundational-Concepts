# Information Theory in ML

> **Applications of information theory to machine learning** - From loss functions to model compression

---

## 🎯 Loss Functions

### Cross-Entropy Loss

**Binary Classification:**
```python
def binary_cross_entropy(y_true, y_pred):
    """BCE = -Σ[y log(ŷ) + (1-y)log(1-ŷ)]"""
    eps = 1e-10
    return -np.mean(
        y_true * np.log(y_pred + eps) + 
        (1 - y_true) * np.log(1 - y_pred + eps)
    )
```

**Multi-class Classification:**
```python
def categorical_cross_entropy(y_true, y_pred):
    """CCE = -Σ y log(ŷ)"""
    eps = 1e-10
    return -np.sum(y_true * np.log(y_pred + eps))
```

**Why cross-entropy?**
- Equivalent to maximum likelihood
- Minimizes KL divergence
- Proper scoring rule

---

## 📊 Model Compression

### Knowledge Distillation

**Idea:** Transfer knowledge from large model to small model

```python
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    """
    T: temperature
    alpha: balance between hard and soft targets
    """
    # Soft targets (from teacher)
    soft_targets = softmax(teacher_logits / T)
    soft_prob = softmax(student_logits / T)
    soft_loss = -np.sum(soft_targets * np.log(soft_prob)) * (T ** 2)
    
    # Hard targets (true labels)
    hard_loss = cross_entropy(labels, softmax(student_logits))
    
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### Pruning with Information Theory

**Mutual information for feature selection:**
```python
def mutual_info_feature_selection(X, y, k=10):
    """Select top k features by mutual information"""
    from sklearn.feature_selection import mutual_info_classif
    
    mi_scores = mutual_info_classif(X, y)
    top_k_idx = np.argsort(mi_scores)[-k:]
    
    return top_k_idx
```

---

## 🎯 Variational Inference

### ELBO (Evidence Lower Bound)

```
log p(x) = ELBO + KL(q||p)
ELBO = E_q[log p(x,z)] - E_q[log q(z)]
     = E_q[log p(x|z)] - KL(q(z)||p(z))
```

**Maximize ELBO ⟺ Minimize KL divergence**

```python
def elbo_loss(x, z_mean, z_logvar, decoder):
    """VAE ELBO loss"""
    # Reconstruction loss
    x_recon = decoder(z_mean)
    recon_loss = binary_cross_entropy(x, x_recon)
    
    # KL divergence (assuming prior N(0,I))
    kl_loss = -0.5 * np.sum(1 + z_logvar - z_mean**2 - np.exp(z_logvar))
    
    return recon_loss + kl_loss
```

---

## 📈 Information Bottleneck

### Principle
```
Minimize: I(X;Z) - βI(Z;Y)

Compress X → Z while preserving info about Y
```

**Applications:**
- Deep learning theory
- Representation learning
- Feature extraction

---

## 🎯 Generative Models

### GANs and JS Divergence

**Original GAN objective:**
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1-D(G(z)))]

Optimal D ⟹ minimizes JS divergence
```

### VAEs and ELBO

**VAE objective:**
```
max ELBO = E_q[log p(x|z)] - KL(q(z|x)||p(z))
```

---

## 🎓 Interview Focus

### Key Questions

1. **Why cross-entropy for classification?**
   - Maximum likelihood estimation
   - Minimizes KL divergence
   - Convex for linear models

2. **ELBO in VAEs?**
   - Evidence lower bound
   - Maximizing ELBO = minimizing KL
   - Balances reconstruction and regularization

3. **Information bottleneck?**
   - Compress while preserving task-relevant info
   - Explains deep learning generalization
   - β controls compression-accuracy trade-off

4. **Mutual information in feature selection?**
   - Measures dependence between feature and target
   - Captures nonlinear relationships
   - Better than correlation

---

## 📚 References

- **Papers:**
  - "Deep Learning and the Information Bottleneck Principle" - Tishby & Zaslavsky
  - "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"

---

**Information theory: the mathematical foundation of modern ML!**
