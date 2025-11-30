# Dyna-Q

> **Integrated Architectures** - Learning from Real and Simulated Experience

---

## 🏗️ The Dyna Architecture (Sutton, 1990)

Dyna-Q integrates **Learning**, **Planning**, and **Acting**.

1.  **Direct RL (Real Experience):**
    - Interact with environment: $S \to A \to R, S'$.
    - Update Q-values: $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max Q(S', a') - Q(S, A)]$.
    - **Update Model:** Learn $Model(S, A) \to (R, S')$.

2.  **Planning (Simulated Experience):**
    - Loop $N$ times:
        - Pick a random state $S$ and action $A$ (previously visited).
        - Query Model: $R, S' \leftarrow Model(S, A)$.
        - Update Q-values using simulated transition.

---

## 🚀 The Benefit

For every 1 real step, we do $N$ planning steps (e.g., $N=10$).
The Q-values converge 10x faster in terms of real interactions.
The agent "dreams" about past experiences and updates its policy.

---

## 💻 Python Implementation

```python
class DynaQAgent:
    def __init__(self):
        self.Q = {}
        self.model = {} # Dictionary: (s, a) -> (r, s_next)
        
    def update(self, s, a, r, s_next):
        # 1. Direct RL
        max_next_q = max([self.Q.get((s_next, a_prime), 0) for a_prime in actions])
        self.Q[(s, a)] += alpha * (r + gamma * max_next_q - self.Q[(s, a)])
        
        # 2. Update Model
        self.model[(s, a)] = (r, s_next)
        
        # 3. Planning (Dyna)
        for _ in range(N_planning_steps):
            # Sample random previously visited state-action
            s_sim, a_sim = random.choice(list(self.model.keys()))
            r_sim, s_next_sim = self.model[(s_sim, a_sim)]
            
            # Update Q using simulation
            max_next_q_sim = max([self.Q.get((s_next_sim, a_prime), 0) for a_prime in actions])
            self.Q[(s_sim, a_sim)] += alpha * (r_sim + gamma * max_next_q_sim - self.Q[(s_sim, a_sim)])
```

---

## 🎓 Interview Focus

1.  **What if the environment changes?**
    - Dyna-Q might plan using an outdated model ("Blocking Maze" problem).
    - **Dyna-Q+** adds an "exploration bonus" to the planning step for state-actions that haven't been tried in a long time, encouraging the agent to re-verify the model.

2.  **Is Dyna-Q Model-Based or Model-Free?**
    - It is **both**. It uses a model to generate data, but uses a model-free algorithm (Q-Learning) to update the policy.

---

**Dyna-Q: Dreaming of success!**
