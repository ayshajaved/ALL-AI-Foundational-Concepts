# Soft Actor-Critic (SAC)

> **Maximum Entropy RL** - Exploration by Design

---

## 🌡️ The Philosophy

Standard RL: Maximize expected return.
$$ \max \sum R_t $$

**Maximum Entropy RL:** Maximize return + **Entropy** of the policy.
$$ \max \sum [ R_t + \alpha H(\pi(\cdot|s_t)) ] $$

- $H(\pi)$: Entropy (Randomness).
- $\alpha$: Temperature parameter.

**Goal:** Succeed at the task while acting **as randomly as possible**.

---

## 🌟 Benefits

1.  **Exploration:** The agent is incentivized to explore. No need for $\epsilon$-greedy or OU noise.
2.  **Robustness:** If multiple actions are equally good, the policy learns to do all of them (multimodal), preventing collapse to a brittle local optimum.

---

## 🏗️ Architecture

- **Off-Policy:** Uses Replay Buffer (Sample efficient).
- **Actor:** Outputs Mean and Std Dev of Gaussian ($\mu, \sigma$).
- **Critics:** Two Q-functions (like TD3) to prevent overestimation.
- **Value Function:** Soft Value function (includes entropy).

**Soft Bellman Update:**
$$ Q(s, a) \leftarrow R + \gamma \mathbb{E}_{s'} [ V(s') ] $$
$$ V(s') = \mathbb{E}_{a'} [ Q(s', a') - \alpha \log \pi(a'|s') ] $$

---

## 💻 The Reparameterization Trick

To backpropagate through the stochastic sampling $a \sim \mathcal{N}(\mu, \sigma)$:
$$ a = \tanh( \mu + \sigma \cdot \epsilon ) $$
where $\epsilon \sim \mathcal{N}(0, 1)$.
The $\tanh$ squashes the action to $[-1, 1]$.

---

## 🎓 Interview Focus

1.  **Why maximize entropy?**
    - It keeps options open. If the environment changes slightly, a high-entropy policy adapts faster because it hasn't "forgotten" other actions.

2.  **SAC vs PPO?**
    - **SAC:** Off-policy. Very sample efficient (needs fewer steps). Good for robotics/real-world.
    - **PPO:** On-policy. More stable, easier to tune. Good for simulation/games.

3.  **Auto-tuning Alpha?**
    - Modern SAC learns the temperature $\alpha$ automatically during training, adjusting exploration as needed.

---

**SAC: The state-of-the-art for continuous control!**
