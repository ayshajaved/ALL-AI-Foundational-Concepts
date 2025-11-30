# Distributed RL with Ray RLLib

> **Scaling Up** - Training on Clusters

---

## 🕸️ Why Ray RLLib?

Writing a PPO implementation that runs on 1 GPU is easy.
Writing one that runs on **100 CPUs and 8 GPUs** is hard.
**Ray RLLib** handles the distributed computing, fault tolerance, and resource management.

---

## 🏗️ Architecture

1.  **Driver:** The main script.
2.  **Trainer (Algorithm):** PPO, DQN, IMPALA. Manages the policy update.
3.  **Workers (Rollout Workers):**
    - Distributed across CPU cores/machines.
    - Each has a copy of the Environment and Policy.
    - Collects samples and sends them to the Trainer.

---

## 💻 Implementation

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# 1. Initialize Ray
ray.init()

# 2. Configure PPO
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .rollouts(num_rollout_workers=4) # Parallel workers
    .resources(num_gpus=1)           # Use 1 GPU for training
    .training(lr=0.0003, train_batch_size=4000)
)

# 3. Build Algorithm
algo = config.build()

# 4. Train Loop
for i in range(10):
    result = algo.train()
    print(f"Iter: {i}, Reward: {result['episode_reward_mean']}")
    
# 5. Save
algo.save("./checkpoint")
```

---

## 🎓 Interview Focus

1.  **Synchronous vs Asynchronous Sampling?**
    - **Sync:** Trainer waits for all workers to finish. (Stable).
    - **Async:** Trainer updates as soon as *some* data arrives. (Faster, but potentially unstable).

2.  **Hyperparameter Tuning (Ray Tune)?**
    - Ray includes **Tune**, a library for running experiments.
    - Supports Grid Search, Bayesian Optimization, and Population Based Training (PBT).

---

**Ray RLLib: Industrial-strength RL!**
