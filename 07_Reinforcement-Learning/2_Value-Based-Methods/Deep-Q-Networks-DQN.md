# Deep Q-Networks (DQN)

> **From Tables to Tensors** - Solving Atari with Deep Learning

---

## 📉 The Problem with Q-Tables

In Q-Learning, we store a table of size $|S| \times |A|$.
- GridWorld: $16 \times 4$ (Easy).
- Atari: $210 \times 160$ pixels $\approx 10^{100}$ states. **Impossible.**

**Solution:** Use a Neural Network to *approximate* the Q-function.
$$ Q(s, a; \theta) \approx Q^*(s, a) $$
Input: State (Image). Output: Q-values for all actions.

---

## 🏗️ The DQN Algorithm (Mnih et al., 2015)

Training a Q-Network is unstable. DQN introduced two key tricks to stabilize it:

### 1. Experience Replay
Instead of training on consecutive samples (highly correlated), store transitions $(s, a, r, s')$ in a **Replay Buffer**.
Sample a random **batch** to train.
- Breaks correlation.
- Reuses data (data efficiency).

### 2. Target Network
The Q-Learning target depends on the network itself:
$$ \text{Target} = R + \gamma \max_{a'} Q(s', a'; \theta) $$
If we update $\theta$, the target moves! It's like chasing your own tail.
**Fix:** Use a separate **Target Network** with frozen parameters $\theta^-$. Update $\theta^-$ to match $\theta$ every 1000 steps.

$$ L(\theta) = \mathbb{E} [ (R + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 ] $$

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# Training Loop Snippet
memory = deque(maxlen=10000)
batch_size = 32

def train_step():
    if len(memory) < batch_size: return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert to tensors...
    
    # Current Q
    q_values = policy_net(states).gather(1, actions.unsqueeze(1))
    
    # Target Q (using Target Net)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * gamma * next_q_values
        
    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🎓 Interview Focus

1.  **Why is Experience Replay crucial?**
    - Neural Networks assume data is i.i.d (Independent and Identically Distributed). RL data is a sequence. Replay restores the i.i.d property.

2.  **What happens if we remove the Target Network?**
    - Divergence. The Q-values can explode or oscillate because the target is non-stationary.

3.  **Can DQN handle continuous action spaces?**
    - No. The `max_a Q(s, a)` operation requires iterating over all discrete actions. For continuous actions, we need Policy Gradients (DDPG/PPO).

---

**DQN: The algorithm that started the Deep RL boom!**
