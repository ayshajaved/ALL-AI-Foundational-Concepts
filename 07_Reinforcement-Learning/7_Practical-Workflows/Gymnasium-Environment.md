# Gymnasium (Custom Environments)

> **Building the World** - The Standard Interface

---

## 🌍 The Interface

To train an agent, you need an environment that follows the **Gymnasium** (formerly OpenAI Gym) API.

1.  `__init__`: Define Action Space and Observation Space.
2.  `reset`: Return initial state.
3.  `step`: Apply action, return `(next_state, reward, terminated, truncated, info)`.
4.  `render`: Visualize.

---

## 💻 Custom Environment: Stock Trading

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df
        
        # Action: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation: [Price, Balance, Shares]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 10000
        self.shares = 0
        self.net_worth = 10000
        
        return self._get_obs(), {}
        
    def step(self, action):
        price = self.df.iloc[self.current_step]['Close']
        
        # Execute Trade
        if action == 1: # Buy
            self.shares += self.balance // price
            self.balance %= price
        elif action == 2: # Sell
            self.balance += self.shares * price
            self.shares = 0
            
        # Next Step
        self.current_step += 1
        
        # Calculate Reward (Change in Net Worth)
        new_net_worth = self.balance + self.shares * price
        reward = new_net_worth - self.net_worth
        self.net_worth = new_net_worth
        
        # Check Done
        terminated = self.current_step >= len(self.df) - 1
        
        return self._get_obs(), reward, terminated, False, {}
        
    def _get_obs(self):
        price = self.df.iloc[self.current_step]['Close']
        return np.array([price, self.balance, self.shares], dtype=np.float32)
```

---

## 🎓 Interview Focus

1.  **Discrete vs Box Spaces?**
    - `Discrete(N)`: Actions are integers $0 \dots N-1$. (DQN).
    - `Box(low, high, shape)`: Actions are continuous floats. (PPO/DDPG).

2.  **Wrappers?**
    - Gymnasium allows wrapping environments to modify observations/rewards without changing the core code.
    - `NormalizeObservation`, `FrameStack` (for Atari), `TimeLimit`.

---

**Gymnasium: The playground for your agents!**
