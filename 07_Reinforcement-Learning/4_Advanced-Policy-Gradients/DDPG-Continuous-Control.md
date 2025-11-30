# Deep Deterministic Policy Gradient (DDPG)

> **DQN for Continuous Control** - Robotics and Physics

---

## 🤖 The Challenge

DQN works for discrete actions (Left, Right).
Robots have continuous joints (Torque: $-1.0$ to $+1.0$).
Discretizing ($-1.0, -0.9, \dots$) explodes the action space.

---

## 🏗️ Architecture

DDPG combines **DQN** (Replay Buffer, Target Net) with **Actor-Critic**.

1.  **Actor ($\mu(s)$):** Deterministic. Outputs exact action value.
    $$ a = \mu(s; \theta^\mu) $$
2.  **Critic ($Q(s, a)$):** Estimates value of (state, action) pair.
    $$ L = (R + \gamma Q'(s', \mu'(s')) - Q(s, a))^2 $$

**Gradients:**
We want to maximize $Q(s, a)$.
We chain rule through the Critic to update the Actor:
$$ \nabla_{\theta^\mu} J \approx \mathbb{E} [ \nabla_a Q(s, a) \cdot \nabla_{\theta^\mu} \mu(s) ] $$
"Change Actor parameters to produce an action that increases Q."

---

## 📉 Exploration (Ornstein-Uhlenbeck Noise)

Since the policy is deterministic, it won't explore.
We add noise to the action during training:
$$ a_t = \mu(s_t) + \mathcal{N}_t $$
**OU Noise:** Correlated noise (like Brownian motion). Good for physics (momentum).

---

## 🎓 Interview Focus

1.  **Why "Deterministic" Policy Gradient?**
    - Standard PG samples from a distribution $\pi(a|s)$. DDPG outputs a single value $\mu(s)$.
    - The gradient $\nabla_a Q$ tells us exactly which way to move the action.

2.  **Soft Updates (Polyak Averaging)?**
    - Instead of copying Target Net every 1000 steps (Hard Update), DDPG updates slowly every step:
    - $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$. (e.g., $\tau=0.001$).
    - Smoother training.

---

**DDPG: Teaching robots to walk!**
