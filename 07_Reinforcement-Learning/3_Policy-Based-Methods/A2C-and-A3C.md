# A2C and A3C

> **Parallelizing RL** - Asynchronous Advantage Actor-Critic

---

## 🕷️ A3C (Asynchronous Advantage Actor-Critic)

**Idea:** RL is unstable because samples are correlated. DQN used Replay Buffer. A3C uses **Parallel Workers**.

- **Global Network:** The master brain.
- **Workers:** Multiple CPU threads, each with its own copy of the environment and local network.
- **Process:**
    1.  Workers interact with their environments.
    2.  Calculate gradients.
    3.  Push gradients to Global Network (Asynchronously).
    4.  Pull updated weights from Global Network.

**Result:** Diverse experience (different workers are in different states), breaking correlation without a Replay Buffer.

---

## 🤖 A2C (Advantage Actor-Critic)

**Idea:** Asynchrony (A3C) is messy. Workers overwrite each other's updates.
**Synchronous** version is simpler and often better.

- **Workers:** Step in parallel.
- **Wait:** Wait for *all* workers to finish a batch of steps.
- **Batch:** Aggregate transitions from all workers into one big batch.
- **Update:** Perform one gradient descent step on the Global Network.

**Benefits:**
- Uses GPU efficiently (large batch size).
- Deterministic (easier to debug).

---

## 💻 Implementation Concept (Vectorized Env)

```python
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create 4 parallel environments
def make_env():
    return gym.make("CartPole-v1")

if __name__ == "__main__":
    # Vectorized Environment
    envs = SubprocVecEnv([make_env for _ in range(4)])
    
    # Reset all
    states = envs.reset()
    
    # Step all at once
    actions = [env.action_space.sample() for _ in range(4)]
    next_states, rewards, dones, infos = envs.step(actions)
    
    # 'rewards' is now a vector of shape (4,)
```

---

## 🎓 Interview Focus

1.  **Why A2C over DQN?**
    - A2C can handle continuous action spaces (DQN cannot).
    - A2C learns stochastic policies.
    - A2C is often faster (wall-clock time) due to parallel environments.

2.  **Why A2C over A3C?**
    - A3C was designed for CPUs. A2C is designed for GPUs (batching).
    - A2C removes the noise of asynchronous updates.

---

**A2C: The parallel processing powerhouse!**
