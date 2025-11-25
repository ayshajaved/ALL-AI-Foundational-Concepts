# Optimization Algorithms for ML

> **Practical optimizers for deep learning** - From SGD to Adam and beyond

---

## 🎯 Algorithm Comparison

| Algorithm | Learning Rate | Momentum | Adaptive | Memory |
|-----------|---------------|----------|----------|--------|
| SGD | Fixed | ❌ | ❌ | O(n) |
| SGD+Momentum | Fixed | ✅ | ❌ | O(n) |
| AdaGrad | Adaptive | ❌ | ✅ | O(n) |
| RMSProp | Adaptive | ❌ | ✅ | O(n) |
| Adam | Adaptive | ✅ | ✅ | O(n) |
| AdamW | Adaptive | ✅ | ✅ | O(n) |

---

## 🚀 Modern Optimizers

### AdamW (Adam with Weight Decay)

**Decouples weight decay from gradient**

```python
def adamw(params, grads, lr=0.001, beta1=0.9, beta2=0.999, 
          weight_decay=0.01, eps=1e-8):
    """AdamW optimizer"""
    # Adam update
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # Update with decoupled weight decay
    params = params - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * params)
    
    return params
```

### RAdam (Rectified Adam)

**Fixes Adam's warmup issue**

### Lookahead

**Wraps around any optimizer**

---

## 🎯 Hyperparameter Tuning

### Learning Rate
- **Too high:** Divergence
- **Too low:** Slow convergence
- **Typical:** 1e-4 to 1e-2

### Batch Size
- **Small (32):** Noisy, regularization effect
- **Large (256+):** Stable, faster training

### Weight Decay
- **Typical:** 1e-4 to 1e-2

---

## 📊 Practical Tips

1. **Start with Adam** (lr=1e-3)
2. **Try SGD+momentum** for better generalization
3. **Use learning rate schedules**
4. **Monitor gradient norms**

---

**Modern optimizers: standing on the shoulders of giants!**
