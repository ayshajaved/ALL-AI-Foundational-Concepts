# Twin Delayed DDPG (TD3)

> **Fixing DDPG** - Reducing Overestimation in Continuous Control

---

## 📉 The Problem with DDPG

Like DQN, DDPG suffers from **Overestimation Bias**.
The Critic learns noisy Q-values, and the Actor exploits these errors (peaks).
Result: Policy breaks.

---

## 🛠️ The 3 Improvements (TD3)

1.  **Clipped Double Q-Learning:**
    - Train **two** Critics ($Q_1, Q_2$).
    - Use the **minimum** for the target.
    - $y = R + \gamma \min(Q_1'(s', a'), Q_2'(s', a'))$.
    - Underestimates value (Pessimism), which is safer.

2.  **Delayed Policy Updates:**
    - Update the Critics frequently (every step).
    - Update the Actor less frequently (every 2 steps).
    - Let the Critic settle before moving the Actor.

3.  **Target Policy Smoothing:**
    - Add noise to the *target action* when computing the target value.
    - $a' = \mu'(s') + \epsilon$.
    - Prevents the Critic from overfitting to sharp peaks in the Q-function.

---

## 💻 Implementation Snippet

```python
# 1. Select action with noise
noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
next_action = (actor_target(next_state) + noise).clamp(-1, 1)

# 2. Compute Target Q
target_Q1 = critic_target1(next_state, next_action)
target_Q2 = critic_target2(next_state, next_action)
target_Q = torch.min(target_Q1, target_Q2)
target_val = reward + (1 - done) * gamma * target_Q

# 3. Update Critics
current_Q1 = critic1(state, action)
current_Q2 = critic2(state, action)
loss = MSE(current_Q1, target_val) + MSE(current_Q2, target_val)
```

---

## 🎓 Interview Focus

1.  **Why take the minimum of two critics?**
    - If $Q_1$ overestimates and $Q_2$ underestimates, taking the min removes the overestimation bias. It's better to be pleasantly surprised than disappointed.

2.  **TD3 vs DDPG?**
    - TD3 is strictly better. It is the default baseline for deterministic continuous control.

---

**TD3: The robust successor to DDPG!**
