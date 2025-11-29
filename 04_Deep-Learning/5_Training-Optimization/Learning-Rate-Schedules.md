# Learning Rate Schedules

> **Tuning the most important hyperparameter** - Strategies for optimal convergence

---

## 🎯 Why Schedule Learning Rate?

- **High LR:** Fast initial learning, but oscillates around minimum.
- **Low LR:** Precise convergence, but gets stuck in local minima and takes forever.
- **Solution:** Start high, decay over time.

---

## 📉 Common Schedules

### 1. Step Decay
Reduce LR by a factor $\gamma$ every $N$ epochs.
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### 2. MultiStep Decay
Reduce LR at specific milestones (e.g., epoch 30, 80).
```python
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
```

### 3. Exponential Decay
$$ LR = LR_0 \times \gamma^{epoch} $$
```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

### 4. Reduce on Plateau
Reduce LR when a metric (e.g., validation loss) stops improving. **Highly recommended.**
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
# Must call step(val_loss)
```

---

## 🌊 Cosine Annealing

Decreases LR following a cosine curve. No hyperparameters to tune (except min_lr).
Very popular in modern research (ConvNeXt, ViT).

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
```

---

## 🔥 Warmup

Start with very low LR and linearly increase it for $N$ epochs, then decay.
**Why?** Stabilizes training early on when gradients are huge.

```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=1000
)
```

---

## 🔄 One Cycle Policy

1.  Increase LR from `max_lr/div_factor` to `max_lr`.
2.  Decrease LR from `max_lr` to `max_lr/div_factor`.
3.  Decrease further to near zero.

**Benefit:** Super convergence (trains 10x faster).

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=len(dataloader), epochs=10
)
```

---

## 🎓 Interview Focus

1.  **Why use Warmup?**
    - To prevent early instability. Weights are random initially, so gradients are large. High LR can cause divergence.

2.  **ReduceLROnPlateau vs Cosine Annealing?**
    - **Plateau:** Reactive. Good when you don't know how long to train.
    - **Cosine:** Proactive. Smooth decay. Often reaches better final accuracy.

3.  **What is the One Cycle Policy?**
    - A schedule that pushes LR very high (super-convergence) to traverse flat areas of loss landscape quickly, then anneals for precision.

---

**Scheduling: The art of landing the plane!**
