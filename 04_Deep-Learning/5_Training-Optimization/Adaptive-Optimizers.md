# Adaptive Optimizers

> **Smart learning rates** - Adagrad, RMSprop, Adam, and AdamW

---

## 🎯 The Need for Adaptation

SGD uses a fixed learning rate $\eta$ for all parameters.
**Problem:** Sparse features need larger updates; frequent features need smaller updates.
**Solution:** Adapt learning rate *per parameter*.

---

## 📊 Adagrad (Adaptive Gradient)

**Idea:** Divide learning rate by sum of squared past gradients.

$$ G_{t, ii} = \sum_{\tau=1}^t g_{\tau, i}^2 $$
$$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t, ii} + \epsilon}} g_{t, i} $$

**Pros:** Good for sparse data (NLP).
**Cons:** $G_t$ grows monotonically $\to$ LR shrinks to 0 too fast.

---

## 📉 RMSprop (Root Mean Square Propagation)

**Idea:** Fix Adagrad by using an exponentially weighted moving average of squared gradients.

$$ E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2 $$
$$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t $$

**Pros:** Works well for RNNs and non-stationary settings.

---

## 🌟 Adam (Adaptive Moment Estimation)

**Idea:** Combine Momentum (1st moment) and RMSprop (2nd moment).

1.  **Momentum:** $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
2.  **RMSprop:** $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
3.  **Bias Correction:** $\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{v}_t = v_t/(1-\beta_2^t)$
4.  **Update:** $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

**Default:** $\beta_1=0.9, \beta_2=0.999, \epsilon=1e-8$.

---

## 🛠️ AdamW (Adam with Weight Decay)

**Problem:** L2 regularization in Adam is not equivalent to Weight Decay (unlike in SGD).
**Solution:** Decouple weight decay from the gradient update.

$$ \theta_{t+1} = \theta_t - \eta (\dots) - \eta \lambda \theta_t $$

**Result:** Better generalization, standard for Transformers.

---

## 💻 PyTorch Implementation

```python
# RMSprop
optimizer_rms = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adam (Standard)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

# AdamW (Best for Transformers/Modern CNNs)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

## 🎓 Interview Focus

1.  **Why use Adam over SGD?**
    - Adam converges faster and requires less tuning of the learning rate.

2.  **When to use SGD over Adam?**
    - SGD (with momentum) often generalizes better than Adam for Computer Vision tasks (ResNets), though it takes longer to train.

3.  **What is the difference between Adam and AdamW?**
    - AdamW implements weight decay correctly (decoupled from gradient), leading to better regularization and generalization.

---

**Adam: The default optimizer for Deep Learning!**
