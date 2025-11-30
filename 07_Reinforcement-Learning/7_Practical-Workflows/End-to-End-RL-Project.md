# End-to-End RL Project: Lunar Lander

> **Landing on the Moon** - Solving a Classic Control Task

---

## 🎯 The Goal

Land the spaceship safely between the flags.
- **Input:** 8-dim vector (Coordinates, Velocity, Angle, Legs touching ground).
- **Action:** 4 discrete actions (Do nothing, Fire Left, Fire Main, Fire Right).
- **Reward:** +100 for landing, -0.3 per frame for firing engine (fuel cost).

---

## 🛠️ The Stack

- **Environment:** `Gymnasium` (LunarLander-v2).
- **Algorithm:** `Stable-Baselines3` (PPO).
- **Visualization:** `imageio` (Record video).

---

## 💻 Implementation

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 1. Vectorized Environment (4 parallel envs)
env = make_vec_env("LunarLander-v2", n_envs=4)

# 2. Model (PPO)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01 # Entropy to encourage exploration
)

# 3. Train
model.learn(total_timesteps=100000)

# 4. Save
model.save("ppo_lunar_lander")

# 5. Evaluate
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

---

## 🎓 Interview Focus

1.  **Why PPO for Lunar Lander?**
    - It's a continuous control task (physics), but the action space is discrete. PPO handles this naturally.
    - It's stable. DQN often struggles with the precise hovering needed.

2.  **Entropy Coefficient?**
    - `ent_coef=0.01`. Adds a bonus for randomness. Prevents the agent from committing too early to "Always fire main engine" (which might keep it alive but never land).

---

**Lunar Lander: One small step for code, one giant leap for AI!**
