# Offline RL

> **Learning from History** - No Interaction Allowed

---

## 🚫 The Constraint

In many real-world settings (Healthcare, Robotics, Industrial Control), we **cannot** let an untrained agent explore randomly. It's dangerous or expensive.
We only have a **static dataset** of past interactions (collected by humans or old policies).
$$ D = \{ (s, a, r, s') \} $$

---

## 📉 The Problem: Distribution Shift

Standard Q-Learning fails in Offline RL.
$$ Q(s, a) \leftarrow R + \gamma \max_{a'} Q(s', a') $$
The `max` operator queries the Q-network for actions $a'$ that were **never seen** in the dataset.
The network hallucinates high values for these "Out-of-Distribution" (OOD) actions.
The agent tries to execute these "magic" actions and fails.

---

## 🛡️ Conservative Q-Learning (CQL)

**Idea:** Penalize Q-values for OOD actions.
Learn a lower bound on the true Q-function.

$$ L(\theta) = \underbrace{\text{DQN Loss}}_{\text{Standard}} + \alpha \underbrace{(\mathbb{E}_{a \sim \mu}[Q(s, a)] - \mathbb{E}_{a \sim \pi_\beta}[Q(s, a)])}_{\text{Conservative Penalty}} $$

- Push down Q-values for actions the model *thinks* are good ($\mu$).
- Push up Q-values for actions actually in the dataset ($\pi_\beta$).

---

## 🎓 Interview Focus

1.  **Why not just use Behavior Cloning (Supervised Learning)?**
    - BC only mimics the dataset. If the dataset is suboptimal (mediocre human), BC will be mediocre.
    - Offline RL (CQL) can **outperform** the demonstrator by stitching together the best parts of different trajectories.

2.  **What is "Stitching"?**
    - Trajectory A goes $S_{start} \to S_{mid}$ (Good).
    - Trajectory B goes $S_{mid} \to S_{goal}$ (Good).
    - Offline RL combines them to go $S_{start} \to S_{goal}$, even if no single trajectory in the dataset did that.

---

**Offline RL: Making use of big data!**
