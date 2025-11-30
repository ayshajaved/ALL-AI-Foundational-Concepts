# DQN Improvements

> **Fixing the Flaws** - Double DQN and Dueling DQN

---

## 1. Double DQN (DDQN)

**Problem: Maximization Bias**
Standard DQN uses:
$$ Y^{DQN} = R + \gamma \max_{a'} Q(s', a'; \theta^-) $$
The `max` operator tends to **overestimate** values. If noise makes a bad action look slightly good, `max` picks it.

**Solution:** Decouple selection and evaluation.
1.  Use **Policy Net** to *select* the best action.
2.  Use **Target Net** to *evaluate* its value.

$$ Y^{DDQN} = R + \gamma Q(s', \arg\max_{a} Q(s', a; \theta); \theta^-) $$

---

## 2. Dueling DQN

**Insight:**
In many states, it doesn't matter what action you take (e.g., driving on a straight empty road). The state value $V(s)$ is high, regardless of action.
Standard DQN learns $Q(s, a)$ for every action independently. This is inefficient.

**Architecture:**
Split the network into two streams:
1.  **Value Stream $V(s)$:** How good is the state?
2.  **Advantage Stream $A(s, a)$:** How much better is action $a$ than the average?

$$ Q(s, a) = V(s) + (A(s, a) - \text{mean}_{a'} A(s, a')) $$

**Benefit:** Learns state values faster.

---

## 💻 PyTorch Implementation (Dueling)

```python
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine
        qvals = values + (advantages - advantages.mean())
        return qvals
```

---

## 🎓 Interview Focus

1.  **Why subtract the mean in Dueling DQN?**
    - Identifiability. If $Q = V + A$, we could add $+10$ to $V$ and $-10$ to $A$ and get the same $Q$.
    - Forcing $A$ to have zero mean ensures $V(s) \approx Q(s, a^*)$.

2.  **Does Double DQN reduce variance or bias?**
    - It reduces **Maximization Bias** (Positive Bias). It doesn't necessarily reduce variance.

---

**Improvements: Small tweaks, huge gains!**
