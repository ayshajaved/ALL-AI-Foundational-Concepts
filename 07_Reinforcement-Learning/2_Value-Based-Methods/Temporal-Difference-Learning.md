# Temporal Difference (TD) Learning

> **Learning from Partial Experience** - Bootstrapping the Future

---

## ⏳ The TD Idea

TD Learning is a combination of **Monte Carlo (MC)** and **Dynamic Programming (DP)**.
- Like MC: It learns from raw experience (Model-Free).
- Like DP: It updates estimates based on other estimates (Bootstrapping).

**Update Rule (TD(0)):**
$$ V(S_t) \leftarrow V(S_t) + \alpha [ \underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD Target}} - V(S_t) ] $$

- **TD Error ($\delta_t$):** Difference between the estimated value ($V(S_t)$) and the better estimate ($R + \gamma V(S_{t+1})$).

---

## 🐢 SARSA (On-Policy)

**S**tate-**A**ction-**R**eward-**S**tate-**A**ction.
We learn the Q-value of the policy we are *currently following*.

$$ Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma Q(S', A') - Q(S, A) ] $$

- $A'$ is the action actually taken in the next step (using $\epsilon$-greedy).
- **Safe:** Learns to avoid cliffs if the exploration policy is random.

---

## 🐇 Q-Learning (Off-Policy)

We learn the Q-value of the *optimal* policy, regardless of what action we actually took.

$$ Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma \max_{a'} Q(S', a') - Q(S, A) ] $$

- We use the **max** over next actions.
- **Aggressive:** Assumes we will act optimally in the future, even if we are exploring randomly right now.

---

## 💻 Python Implementation (GridWorld)

```python
import numpy as np

# Q-Table: [States, Actions]
Q = np.zeros((16, 4)) 
alpha = 0.1
gamma = 0.99
epsilon = 0.1

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon-Greedy Action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
            
        next_state, reward, done, _ = env.step(action)
        
        # Q-Learning Update
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next_action]
        td_error = td_target - Q[state, action]
        
        Q[state, action] += alpha * td_error
        
        state = next_state
```

---

## 🎓 Interview Focus

1.  **Why is Q-Learning "Off-Policy"?**
    - The update uses `max Q(S', a')` (Optimal Policy), but the behavior is $\epsilon$-greedy (Exploratory Policy). The target policy $\ne$ behavior policy.

2.  **TD vs Monte Carlo?**
    - **TD:** Updates after every step. Lower variance. Can learn in continuing tasks.
    - **MC:** Updates after episode ends. Zero bias. High variance.

3.  **What is n-step TD?**
    - A middle ground. Look $n$ steps ahead before bootstrapping.
    - $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^n V(S_{t+n})$.

---

**TD Learning: Predicting the future from the present!**
