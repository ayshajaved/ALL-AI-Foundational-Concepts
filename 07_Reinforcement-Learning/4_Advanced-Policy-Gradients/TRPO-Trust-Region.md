# Trust Region Policy Optimization (TRPO)

> **Don't Step Too Far** - Monotonic Improvement Guarantee

---

## 📉 The Problem with Policy Gradients

Standard Policy Gradient (REINFORCE/A2C) is sensitive to the learning rate $\alpha$.
- $\alpha$ too small: Slow convergence.
- $\alpha$ too large: The policy changes drastically, performance collapses ("Cliff"), and it never recovers.

**Goal:** Ensure that every update *improves* (or at least doesn't degrade) the policy.

---

## 🛡️ The Trust Region

We want to find a new policy $\pi_{new}$ that maximizes the objective, subject to a constraint:
**The new policy must not be too different from the old policy.**

$$ \max_\theta \mathbb{E} \left[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A^{\pi_{old}}(s,a) \right] $$
$$ \text{subject to } \mathbb{E} [ D_{KL}(\pi_{old} || \pi_\theta) ] \le \delta $$

- **Objective:** Importance Sampling estimate of the new return.
- **Constraint:** KL Divergence (distance between distributions) must be small.

---

## 🧩 Natural Gradient

Solving this constrained optimization requires calculating the **Fisher Information Matrix** (second-order derivative) and its inverse.
$$ \theta_{new} = \theta_{old} + \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g $$
- $H$: Fisher Information Matrix (Hessian of KL).
- $g$: Gradient of objective.

**Pros:** Monotonic improvement. Very stable.
**Cons:** Calculating $H^{-1}$ is computationally expensive ($O(N^3)$) for large networks.

---

## 🎓 Interview Focus

1.  **Why KL Divergence?**
    - It measures how different two probability distributions are. In parameter space, a small change in $\theta$ might cause a huge change in $\pi$ (probabilities). KL measures change in *probability space*, which is what matters.

2.  **Why is TRPO rarely used now?**
    - It's complicated (Conjugate Gradient method) and slow. PPO achieves similar stability with first-order methods (much faster).

---

**TRPO: The mathematically rigorous ancestor of PPO!**
