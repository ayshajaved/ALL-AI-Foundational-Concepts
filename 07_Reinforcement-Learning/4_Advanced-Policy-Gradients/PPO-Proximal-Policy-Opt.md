# Proximal Policy Optimization (PPO)

> **The Industry Standard** - OpenAI's Default Algorithm

---

## 🚀 The Idea

TRPO is great but complex. PPO simplifies it.
Instead of a hard constraint ($D_{KL} \le \delta$), PPO uses a **Clipped Objective** to punish large updates.

---

## ✂️ The Clipped Surrogate Objective

Define probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$.

$$ L^{CLIP}(\theta) = \mathbb{E} [ \min( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t ) ] $$

- $A_t$: Advantage.
- $\epsilon$: Hyperparameter (usually 0.2).

**Logic:**
1.  If $A_t > 0$ (Good action): Increase prob. But stop if $r_t > 1.2$ (Don't get too greedy).
2.  If $A_t < 0$ (Bad action): Decrease prob. But stop if $r_t < 0.8$ (Don't destroy the policy).

This "clipping" prevents the policy from moving too far away from $\pi_{old}$, acting as a "Trust Region".

---

## 💻 PyTorch Implementation (PPO Step)

```python
def ppo_update(policy, optimizer, states, actions, log_probs_old, advantages, epsilon=0.2):
    for _ in range(K_epochs):
        # 1. Calculate new probs
        probs = policy(states)
        dist = Categorical(probs)
        log_probs_new = dist.log_prob(actions)
        
        # 2. Ratio
        ratio = torch.exp(log_probs_new - log_probs_old)
        
        # 3. Surrogate Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
        loss = -torch.min(surr1, surr2).mean()
        
        # 4. Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🎓 Interview Focus

1.  **Why is PPO "On-Policy"?**
    - Even though it does multiple epochs of updates on a batch, the data must be collected by the *current* (or very recent) policy. If $\pi$ changes too much, the importance sampling ratio $r_t$ becomes unstable.

2.  **PPO-Clip vs PPO-Penalty?**
    - **Clip:** Uses `clamp` (shown above). Simpler. Used everywhere.
    - **Penalty:** Adds KL-divergence as a penalty term in the loss ($L - \beta D_{KL}$). Adapts $\beta$ dynamically.

3.  **Why is PPO so popular?**
    - Balance: Easy to implement (like A2C), Stable (like TRPO), Fast (First-order optimizer).

---

**PPO: If you only learn one RL algorithm, learn this one!**
