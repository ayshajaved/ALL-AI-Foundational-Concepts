# MADDPG (Multi-Agent DDPG)

> **Centralized Training, Decentralized Execution** - The Standard for MARL

---

## 🧠 The Architecture

Extends DDPG to multi-agent environments.

1.  **Actors (Decentralized):**
    - $\mu_i(o_i)$: Takes only **local observation** $o_i$.
    - Outputs action $a_i$.
    - Used during execution (deployment).

2.  **Critics (Centralized):**
    - $Q_i(\mathbf{x}, a_1, \dots, a_N)$: Takes **global state** $\mathbf{x}$ and **all actions**.
    - Used only during training.

---

## 🎓 The Learning Process

**Critic Update:**
Minimizes loss using the joint action:
$$ y = r_i + \gamma Q_i'(\mathbf{x}', \mu_1'(o_1), \dots, \mu_N'(o_N)) $$
Since the Critic sees *everyone's* actions, the environment appears **stationary** (Markovian).

**Actor Update:**
$$ \nabla_{\theta_i} J \approx \mathbb{E} [ \nabla_{a_i} Q_i(\mathbf{x}, a_1, \dots, a_N) \nabla_{\theta_i} \mu_i(o_i) ] $$
The Actor is guided by the omniscient Critic.

---

## 💻 PyTorch Concept

```python
# Critic Input: [Obs_1, Obs_2, Act_1, Act_2]
critic_input = torch.cat([obs_1, obs_2, act_1, act_2], dim=1)
q_val = critic_network(critic_input)

# Actor Input: [Obs_1]
action_1 = actor_network(obs_1)
```

---

## 🎓 Interview Focus

1.  **Why CTDE (Centralized Training, Decentralized Execution)?**
    - During training (simulation), we have access to god-view. We should use it to stabilize learning.
    - During execution (real robots), communication is expensive/slow, so agents must act locally.

2.  **Does MADDPG scale?**
    - The Critic input grows linearly with $N$ agents. For huge swarms ($N=1000$), we need Mean Field RL (averaging neighbors).

---

**MADDPG: The omniscient teacher!**
