# Gradient Descent Variants

> **Beyond vanilla SGD** - Accelerating convergence with Momentum and Nesterov

---

## 🎯 Stochastic Gradient Descent (SGD)

Standard SGD updates weights using the gradient of the loss with respect to a *single* sample (or a mini-batch).

$$ \theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t; x^{(i)}, y^{(i)}) $$

**Problems:**
1.  **Noisy updates:** Loss fluctuates wildly.
2.  **Slow convergence:** Especially in ravines (steep in one dimension, flat in another).
3.  **Local Minima/Saddle Points:** Can get stuck.

---

## 🚀 Momentum

**Idea:** Accumulate a moving average of past gradients (velocity). Like a heavy ball rolling down a hill—it gains speed and powers through small bumps.

$$ v_{t+1} = \gamma v_t + \eta \nabla_\theta J(\theta_t) $$
$$ \theta_{t+1} = \theta_t - v_{t+1} $$

- $\gamma$: Momentum term (usually 0.9).
- $v_t$: Velocity.

**Benefit:** Dampens oscillations and speeds up convergence.

---

## 🏎️ Nesterov Accelerated Gradient (NAG)

**Idea:** "Look ahead" before making a correction. Calculate the gradient *at the position where the momentum step would take us*.

$$ v_{t+1} = \gamma v_t + \eta \nabla_\theta J(\theta_t - \gamma v_t) $$
$$ \theta_{t+1} = \theta_t - v_{t+1} $$

**Benefit:** More responsive correction, less overshooting.

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)

# 1. Vanilla SGD
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)

# 2. SGD with Momentum
optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 3. SGD with Nesterov
optimizer_nesterov = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

---

## 🎓 Interview Focus

1.  **Why use Momentum?**
    - To accelerate training in relevant directions and dampen oscillations in irrelevant directions (like ravines).

2.  **Difference between Momentum and Nesterov?**
    - Momentum computes gradient at current step.
    - Nesterov computes gradient at the *projected* future step ("lookahead").

3.  **Does SGD guarantee global minimum?**
    - For convex problems: Yes.
    - For non-convex (Neural Nets): No, but it usually finds a good local minimum.

---

**Momentum: The physics of optimization!**
