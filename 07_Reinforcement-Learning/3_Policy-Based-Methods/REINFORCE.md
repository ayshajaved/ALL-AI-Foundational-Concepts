# REINFORCE (Monte Carlo Policy Gradient)

> **The Simplest Policy Gradient** - Trial, Error, and Update

---

## ⚙️ The Algorithm

REINFORCE uses the actual return $G_t$ of the episode as an estimate for $Q(s, a)$.

1.  **Run Episode:** $S_0, A_0, R_1, \dots, S_T, R_T$.
2.  **Calculate Returns:** For each step $t$, compute $G_t = \sum \gamma^k R_{t+k+1}$.
3.  **Update Policy:**
    $$ \theta \leftarrow \theta + \alpha \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t | S_t) $$

---

## 📉 The Baseline Trick

Raw returns $G_t$ have high variance.
- If rewards are always positive (e.g., +10, +20, +30), we increase probability of *all* actions, just by different amounts.
- We want to know: Is this action better than **average**?

**Subtract a Baseline $b(s)$:**
$$ \nabla J(\theta) = \mathbb{E} [ \nabla \log \pi(a|s) (G_t - b(s)) ] $$
- Common Baseline: The Value Function $V(s)$.
- If $G_t > V(s)$: Action was better than expected $\to$ Increase prob.
- If $G_t < V(s)$: Action was worse than expected $\to$ Decrease prob.

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

def train(env, policy, optimizer):
    # 1. Collect Trajectory
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        probs = policy(torch.tensor(state).float())
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        next_state, reward, done, _ = env.step(action.item())
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # 2. Calculate Discounted Returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9) # Normalize

    # 3. Update
    policy_loss = []
    for s, a, G in zip(states, actions, returns):
        probs = policy(torch.tensor(s).float())
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(a)
        policy_loss.append(-log_prob * G)
        
    optimizer.zero_grad()
    sum(policy_loss).backward()
    optimizer.step()
```

---

## 🎓 Interview Focus

1.  **Why is REINFORCE high variance?**
    - It uses the full Monte Carlo return $G_t$. The randomness of the entire future trajectory accumulates.

2.  **Is REINFORCE On-Policy or Off-Policy?**
    - **On-Policy.** We must discard the data after one update because the policy has changed. We cannot reuse old trajectories (Experience Replay is impossible).

---

**REINFORCE: The vanilla flavor of Policy Gradients!**
