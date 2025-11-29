# Reinforcement Learning Basics

> **Learning by Trial and Error** - The Agent-Environment Loop

---

## 🔁 The RL Loop

Reinforcement Learning is distinct from Supervised Learning.
- **Supervised:** "Here is an image, here is the label 'Cat'. Learn mapping."
- **RL:** "Here is a chessboard. Make a move. I won't tell you if it's good until the end of the game."

**Components:**
1.  **Agent:** The learner (The Brain).
2.  **Environment:** The world (The Game).
3.  **State ($S_t$):** Current situation.
4.  **Action ($A_t$):** What the agent does.
5.  **Reward ($R_{t+1}$):** Feedback from environment (Scalar).

$$ S_t \xrightarrow{A_t} \text{Env} \xrightarrow{R_{t+1}, S_{t+1}} \text{Agent} $$

---

## 🎯 The Goal: Maximize Return

The agent doesn't care about the *immediate* reward. It cares about the **Cumulative Return** ($G_t$).

$$ G_t = R_{t+1} + R_{t+2} + R_{t+3} + \dots + R_T $$

**Discount Factor ($\gamma$):**
We value immediate rewards more than future rewards ($0 \le \gamma \le 1$).
- $\gamma = 0$: Myopic (Only cares about now).
- $\gamma = 1$: Far-sighted (Infinite horizon).

$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$

---

## 🧠 Exploration vs Exploitation

The fundamental dilemma.
- **Exploitation:** Choose the action you *think* is best (Greedy).
- **Exploration:** Try a random action to see if it's better.

**$\epsilon$-Greedy Strategy:**
- With probability $1-\epsilon$: Choose best action.
- With probability $\epsilon$: Choose random action.
- Decay $\epsilon$ over time ($1.0 \to 0.01$).

---

## 💻 Gymnasium (OpenAI Gym)

The standard interface for RL environments.

```python
import gymnasium as gym

# 1. Create Environment
env = gym.make("CartPole-v1", render_mode="human")

# 2. Reset
state, info = env.reset()

for _ in range(100):
    # 3. Sample Random Action
    action = env.action_space.sample() 
    
    # 4. Step
    next_state, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        state, info = env.reset()
        
env.close()
```

---

## 🎓 Interview Focus

1.  **Difference between RL and Supervised Learning?**
    - RL data is sequential (not i.i.d).
    - RL feedback is delayed (Reward comes later).
    - RL agent influences the data it sees (Actions determine future states).

2.  **What is the Credit Assignment Problem?**
    - You won a chess game after 50 moves. Which move was responsible? The last one? The 10th one? RL algorithms must figure this out.

3.  **Why do we need a Discount Factor?**
    - Mathematical convergence (keeps the sum finite in infinite horizons).
    - Models uncertainty about the future.

---

**Basics: The cycle of life (and AI)!**
