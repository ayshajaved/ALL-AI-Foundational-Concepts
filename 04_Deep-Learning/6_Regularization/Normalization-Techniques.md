# Normalization Techniques

> **Stabilizing distributions** - Batch, Layer, Instance, and Group Norm

---

## 🎯 Batch Normalization (BN)

**Idea:** Normalize layer inputs to mean 0, var 1 *across the batch*.
$$ \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta $$

**Pros:**
- Allows higher learning rates.
- Reduces sensitivity to initialization.
- Acts as regularization.

**Cons:**
- Depends on batch size (fails if batch < 8).
- Hard to use in RNNs.

```python
bn = nn.BatchNorm2d(num_features=64)
```

---

## 🍰 Layer Normalization (LN)

**Idea:** Normalize *across features* for a single sample. Independent of batch size.

**Use Case:** NLP (Transformers), RNNs.

```python
ln = nn.LayerNorm(normalized_shape=512)
```

---

## 🖼️ Instance Normalization (IN)

**Idea:** Normalize each channel independently for each sample.
**Use Case:** Style Transfer, GANs. (Removes style information like contrast).

```python
in_norm = nn.InstanceNorm2d(num_features=64)
```

---

## 👥 Group Normalization (GN)

**Idea:** Divide channels into groups and normalize within each group.
**Pros:** Independent of batch size (like LN/IN) but keeps spatial info (like BN).
**Use Case:** Object detection, Segmentation (where batch size is small).

```python
gn = nn.GroupNorm(num_groups=32, num_channels=64)
```

---

## 🎓 Interview Focus

1.  **Why does Batch Norm fail with small batches?**
    - Batch statistics ($\mu_B, \sigma_B$) become noisy estimates of population statistics.

2.  **BN vs LN?**
    - BN: Normalizes across batch dimension. Good for CNNs.
    - LN: Normalizes across feature dimension. Good for RNNs/Transformers.

3.  **Learnable parameters in Normalization?**
    - $\gamma$ (scale) and $\beta$ (shift). They allow the network to undo the normalization if needed.

---

**Normalization: The secret to deep training!**
