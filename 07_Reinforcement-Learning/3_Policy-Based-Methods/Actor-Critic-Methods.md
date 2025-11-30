# Actor-Critic Methods

> **Best of Both Worlds** - Policy Gradient + Value Function

---

## 🎭 The Architecture

Combines Policy-Based (Actor) and Value-Based (Critic).

1.  **Actor ($\pi_\theta(a|s)$):** Decides which action to take.
2.  **Critic ($V_w(s)$):** Tells the Actor how good that action was.

Instead of waiting for the end of the episode (REINFORCE) to get $G_t$, we use the Critic to estimate the return **immediately**.

---

## ⚡ The Update

**TD Error (Advantage):**
$$ \delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t) $$
This $\delta_t$ is an estimate of the **Advantage** $A(s, a) = Q(s, a) - V(s)$.

1.  **Critic Update (MSE):** Minimize $\delta_t^2$.
    $$ w \leftarrow w + \beta \delta_t \nabla_w V_w(S_t) $$

2.  **Actor Update (Policy Gradient):** Use $\delta_t$ as the score.
    $$ \theta \leftarrow \theta + \alpha \delta_t \nabla_\theta \log \pi_\theta(A_t | S_t) $$

---

## 🆚 Bias-Variance Tradeoff

- **REINFORCE:** Uses $G_t$. Unbiased, High Variance.
- **Actor-Critic:** Uses $R + \gamma V(S')$. Biased (if Critic is wrong), Low Variance.

---

## 💻 Shared Network Architecture

Usually, the Actor and Critic share the initial layers (Feature Extractor) and split only at the end.

```python
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.common = nn.Sequential(nn.Linear(4, 128), nn.ReLU())
        
        # Actor Head
        self.actor = nn.Linear(128, 2) # Action logits
        
        # Critic Head
        self.critic = nn.Linear(128, 1) # State Value
        
    def forward(self, x):
        x = self.common(x)
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value
```

---

## 🎓 Interview Focus

1.  **Why is Actor-Critic lower variance than REINFORCE?**
    - Because $R + \gamma V(S')$ depends on only *one* step of randomness (the immediate reward and transition), whereas $G_t$ depends on the entire future.

2.  **What is the Advantage Function?**
    - $A(s, a) = Q(s, a) - V(s)$.
    - It measures how much better an action is compared to the *average* action in that state.

---

**Actor-Critic: The standard template for modern RL!**
