# Stochastic Processes and Markov Chains

> **Modeling randomness over time** - Essential for RL, time series, and sequential decision making

---

## 🎯 Stochastic Processes Basics

### Definition
A **stochastic process** is a collection of random variables indexed by time:
```
{X(t) : t ∈ T}

T: time index set (discrete or continuous)
```

**Types:**
- **Discrete time:** t ∈ {0, 1, 2, ...}
- **Continuous time:** t ∈ [0, ∞)

---

## 📊 Markov Chains (Discrete Time)

### Markov Property
```
P(Xₙ₊₁ = j | X₀, X₁, ..., Xₙ) = P(Xₙ₊₁ = j | Xₙ)

Future depends only on present, not past
```

### Transition Matrix
```
P = [pᵢⱼ]

pᵢⱼ = P(Xₙ₊₁ = j | Xₙ = i)

Properties:
- pᵢⱼ ≥ 0
- Σⱼ pᵢⱼ = 1 (rows sum to 1)
```

```python
import numpy as np

# Example: Weather model
# States: {Sunny, Rainy}
P = np.array([
    [0.8, 0.2],  # Sunny → [Sunny, Rainy]
    [0.4, 0.6]   # Rainy → [Sunny, Rainy]
])

# Initial distribution
pi_0 = np.array([1.0, 0.0])  # Start sunny

# After n steps
n = 10
pi_n = pi_0 @ np.linalg.matrix_power(P, n)
print(f"Distribution after {n} days: {pi_n}")
```

---

## 🎯 Stationary Distribution

### Definition
```
π is stationary if: πP = π

π: steady-state distribution
```

```python
def find_stationary_distribution(P, tol=1e-10):
    """Find stationary distribution of Markov chain"""
    # Method 1: Power iteration
    pi = np.ones(len(P)) / len(P)
    
    for _ in range(10000):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi) < tol:
            break
        pi = pi_new
    
    return pi

# Method 2: Eigenvalue approach
def stationary_eigen(P):
    """Find stationary distribution using eigenvalues"""
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    
    # Find eigenvector for eigenvalue 1
    idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()  # Normalize
    
    return pi

pi = find_stationary_distribution(P)
print(f"Stationary distribution: {pi}")
```

---

## 📈 Markov Chain Properties

### 1. Irreducibility
All states communicate (can reach any state from any other)

### 2. Aperiodicity
```
gcd{n : P(Xₙ = i | X₀ = i) > 0} = 1
```

### 3. Ergodicity
Irreducible + aperiodic → unique stationary distribution

```python
def is_irreducible(P):
    """Check if Markov chain is irreducible"""
    n = len(P)
    # Compute P^n for large n
    P_n = np.linalg.matrix_power(P, n * n)
    # All entries should be > 0
    return np.all(P_n > 0)

print(f"Is irreducible: {is_irreducible(P)}")
```

---

## 🎯 Hidden Markov Models (HMM)

### Model
```
Hidden states: X₁, X₂, ..., Xₙ  (Markov chain)
Observations: Y₁, Y₂, ..., Yₙ

P(Xₜ₊₁ | Xₜ): transition probabilities
P(Yₜ | Xₜ): emission probabilities
```

### Forward Algorithm

```python
def forward_algorithm(observations, A, B, pi):
    """
    Compute P(observations | model)
    
    A: transition matrix (n_states × n_states)
    B: emission matrix (n_states × n_observations)
    pi: initial distribution
    """
    n_states = len(A)
    T = len(observations)
    
    # Initialize
    alpha = np.zeros((T, n_states))
    alpha[0] = pi * B[:, observations[0]]
    
    # Forward pass
    for t in range(1, T):
        for j in range(n_states):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, observations[t]]
    
    # Total probability
    return np.sum(alpha[-1])

# Example
A = np.array([[0.7, 0.3], [0.4, 0.6]])  # Transition
B = np.array([[0.9, 0.1], [0.2, 0.8]])  # Emission
pi = np.array([0.6, 0.4])  # Initial

observations = [0, 1, 0, 1]
prob = forward_algorithm(observations, A, B, pi)
print(f"P(observations): {prob}")
```

### Viterbi Algorithm

```python
def viterbi(observations, A, B, pi):
    """
    Find most likely state sequence
    """
    n_states = len(A)
    T = len(observations)
    
    # Initialize
    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)
    
    delta[0] = pi * B[:, observations[0]]
    
    # Forward pass
    for t in range(1, T):
        for j in range(n_states):
            probs = delta[t-1] * A[:, j] * B[j, observations[t]]
            delta[t, j] = np.max(probs)
            psi[t, j] = np.argmax(probs)
    
    # Backtrack
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]
    
    return states

states = viterbi(observations, A, B, pi)
print(f"Most likely states: {states}")
```

---

## 🎯 Continuous-Time Markov Chains

### Rate Matrix Q
```
qᵢⱼ: rate of transition from i to j (i ≠ j)
qᵢᵢ = -Σⱼ≠ᵢ qᵢⱼ

P(Xₜ = j | X₀ = i) = [e^(Qt)]ᵢⱼ
```

---

## 📊 Applications in ML

### 1. Reinforcement Learning

**MDP (Markov Decision Process):**
```
States: S
Actions: A
Transition: P(s'|s,a)
Reward: R(s,a)

Policy: π(a|s)
Value function: V^π(s) = E[Σₜ γᵗRₜ | s₀=s]
```

```python
# Simple MDP example
class MDP:
    def __init__(self, states, actions, transitions, rewards, gamma=0.9):
        self.states = states
        self.actions = actions
        self.P = transitions  # P[s][a][s']
        self.R = rewards      # R[s][a]
        self.gamma = gamma
    
    def value_iteration(self, tol=1e-6):
        """Compute optimal value function"""
        V = {s: 0 for s in self.states}
        
        while True:
            V_new = {}
            for s in self.states:
                # Bellman optimality
                V_new[s] = max(
                    self.R[s][a] + self.gamma * sum(
                        self.P[s][a][s_next] * V[s_next]
                        for s_next in self.states
                    )
                    for a in self.actions
                )
            
            if max(abs(V_new[s] - V[s]) for s in self.states) < tol:
                break
            V = V_new
        
        return V
```

### 2. Time Series Modeling

```python
# Markov chain for sequence generation
def generate_sequence(P, initial_state, length):
    """Generate sequence from Markov chain"""
    states = [initial_state]
    
    for _ in range(length - 1):
        current = states[-1]
        next_state = np.random.choice(len(P), p=P[current])
        states.append(next_state)
    
    return states

sequence = generate_sequence(P, 0, 100)
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is Markov property?**
   - Future independent of past given present
   - Memoryless property
   - Foundation of RL

2. **Stationary distribution?**
   - Steady-state probabilities
   - πP = π
   - Exists for ergodic chains

3. **HMM vs Markov chain?**
   - HMM: hidden states + observations
   - Markov chain: states directly observed
   - HMM more general

4. **MDP vs Markov chain?**
   - MDP: adds actions and rewards
   - Used in reinforcement learning
   - Policy optimization

5. **Forward vs Viterbi?**
   - Forward: P(observations)
   - Viterbi: most likely state sequence
   - Both use dynamic programming

---

## 📚 References

- **Books:**
  - "Introduction to Probability Models" - Ross
  - "Markov Chains" - Norris
  - "Reinforcement Learning" - Sutton & Barto

---

**Stochastic processes: modeling the randomness of the world!**
