# Policy Gradient Theorem

> **Optimizing the Brain Directly** - From Values to Probabilities

---

## 🧠 Value-Based vs Policy-Based

- **Value-Based (DQN):** Learn $Q(s, a)$. Pick action with max Q.
    - *Problem:* Hard to handle continuous actions (e.g., Robot torque).
    - *Problem:* Cannot learn stochastic policies (Rock-Paper-Scissors).
- **Policy-Based:** Learn $\pi_\theta(a|s)$ directly.
    - Output: Probability distribution over actions.
    - Optimization: Adjust $\theta$ to increase probability of good actions.

---

## 📜 The Theorem

We want to maximize Expected Return $J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$.
The gradient of the objective is:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s, a)] $$

**Interpretation:**
1.  **$\nabla_\theta \log \pi_\theta(a|s)$:** Direction that increases prob of action $a$.
2.  **$Q^\pi(s, a)$:** How good action $a$ is.
3.  **Result:** If $Q$ is high, move $\theta$ strongly in that direction. If $Q$ is low (negative), move away.

---

## 📝 The Log-Derivative Trick

How do we differentiate an expectation?
$$ \nabla P(x) = P(x) \frac{\nabla P(x)}{P(x)} = P(x) \nabla \log P(x) $$

This allows us to rewrite the gradient in a form that can be estimated by sampling:
$$ \nabla J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log \pi_\theta(a_i|s_i) \cdot G_t $$

---

## 🎓 Interview Focus

1.  **Why use Log Probability?**
    - Numerical stability (probabilities are small, logs are manageable).
    - It naturally arises from the derivative of the expectation.

2.  **What is the "Score Function"?**
    - $\nabla_\theta \log \pi_\theta(a|s)$. It tells us how to change the parameters to make an action more likely.

3.  **Continuous Actions?**
    - For continuous actions (e.g., steering angle), the policy outputs the **Mean** $\mu$ and **Std Dev** $\sigma$ of a Gaussian distribution.
    - $\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$.

---

**Policy Gradients: Learning to act, not just to value!**
