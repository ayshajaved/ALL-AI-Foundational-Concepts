# Bayesian Inference

> **Updating beliefs with data** - The Bayesian approach to machine learning

---

## 🎯 Bayes' Theorem

### Basic Form
```
P(θ|D) = P(D|θ)P(θ) / P(D)

Posterior = (Likelihood × Prior) / Evidence
```

**Components:**
- **Prior P(θ):** Belief before seeing data
- **Likelihood P(D|θ):** Probability of data given parameters
- **Evidence P(D):** Marginal probability of data
- **Posterior P(θ|D):** Updated belief after seeing data

---

## 📊 Bayesian vs Frequentist

| Aspect | Bayesian | Frequentist |
|--------|----------|-------------|
| Parameters | Random variables | Fixed unknowns |
| Probability | Degree of belief | Long-run frequency |
| Inference | Posterior distribution | Point estimates |
| Uncertainty | Credible intervals | Confidence intervals |

---

## 🔧 Bayesian Inference Process

### 1. Choose Prior
```python
# Example: Coin flip bias
# Prior: θ ~ Beta(α, β)
from scipy.stats import beta

alpha, beta_param = 2, 2  # Uniform-ish prior
prior = beta(alpha, beta_param)
```

### 2. Collect Data
```python
# Observe coin flips
data = [1, 1, 0, 1, 0, 1, 1, 1]  # 1=heads, 0=tails
n_heads = sum(data)
n_tails = len(data) - n_heads
```

### 3. Compute Posterior
```python
# Posterior: θ|D ~ Beta(α + n_heads, β + n_tails)
posterior = beta(alpha + n_heads, beta_param + n_tails)

# Plot
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 1, 100)
plt.plot(theta, prior.pdf(theta), label='Prior')
plt.plot(theta, posterior.pdf(theta), label='Posterior')
plt.legend()
plt.xlabel('θ (bias)')
plt.ylabel('Density')
plt.show()
```

---

## 📈 Conjugate Priors

**Conjugate prior:** Prior and posterior have same family

### Common Conjugate Pairs

| Likelihood | Prior | Posterior |
|------------|-------|-----------|
| Bernoulli | Beta | Beta |
| Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (known σ²) | Normal | Normal |
| Normal (known μ) | Inverse-Gamma | Inverse-Gamma |

### Example: Beta-Binomial
```python
# Prior: θ ~ Beta(α, β)
# Likelihood: k ~ Binomial(n, θ)
# Posterior: θ|k ~ Beta(α+k, β+n-k)

def beta_binomial_update(alpha, beta, n, k):
    """Update Beta prior with Binomial data"""
    return alpha + k, beta + (n - k)

# Start with uniform prior
alpha, beta = 1, 1

# Observe 7 heads in 10 flips
alpha_post, beta_post = beta_binomial_update(alpha, beta, 10, 7)
print(f"Posterior: Beta({alpha_post}, {beta_post})")

# Posterior mean
mean = alpha_post / (alpha_post + beta_post)
print(f"Estimated bias: {mean:.3f}")
```

---

## 🎯 Maximum A Posteriori (MAP)

**MAP estimate:** Mode of posterior
```
θ_MAP = argmax_θ P(θ|D)
      = argmax_θ P(D|θ)P(θ)
```

**vs Maximum Likelihood (ML):**
```
θ_ML = argmax_θ P(D|θ)
```

MAP = ML + regularization!

```python
# Example: Linear regression
# ML: minimize ||y - Xw||²
# MAP: minimize ||y - Xw||² + λ||w||²  (Ridge regression!)
```

---

## 🔢 Bayesian Linear Regression

```python
import numpy as np
from scipy.stats import multivariate_normal

# Generate data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2*X.flatten() + 1 + 0.5*np.random.randn(100)

# Prior: w ~ N(0, σ²_prior I)
sigma_prior = 10.0
prior_cov = sigma_prior**2 * np.eye(2)

# Likelihood: y|w ~ N(Xw, σ²I)
sigma_noise = 0.5

# Add bias term
X_bias = np.c_[np.ones(len(X)), X]

# Posterior (analytical)
# Posterior covariance
sigma_post_inv = (1/sigma_prior**2)*np.eye(2) + (1/sigma_noise**2)*(X_bias.T @ X_bias)
sigma_post = np.linalg.inv(sigma_post_inv)

# Posterior mean
mu_post = sigma_post @ (X_bias.T @ y) / sigma_noise**2

print(f"Posterior mean: {mu_post}")
print(f"Posterior covariance:\n{sigma_post}")

# Sample from posterior
posterior = multivariate_normal(mu_post, sigma_post)
w_samples = posterior.rvs(size=1000)

# Plot uncertainty
x_test = np.linspace(-3, 3, 100)
X_test = np.c_[np.ones(100), x_test]
y_pred = X_test @ mu_post

# Predictive uncertainty
y_samples = X_test @ w_samples.T
y_std = y_samples.std(axis=1)

plt.scatter(X, y, alpha=0.5)
plt.plot(x_test, y_pred, 'r-', label='Mean prediction')
plt.fill_between(x_test, y_pred - 2*y_std, y_pred + 2*y_std, 
                 alpha=0.3, label='95% credible interval')
plt.legend()
plt.show()
```

---

## 🎓 Bayesian Neural Networks

```python
# Conceptual: weights are distributions
# w ~ P(w)  (prior)
# P(w|D) ∝ P(D|w)P(w)  (posterior)

# Practical approximations:
# 1. Variational inference
# 2. Monte Carlo dropout
# 3. Laplace approximation
```

---

## 📊 Model Comparison

### Bayes Factor
```
BF = P(D|M₁) / P(D|M₂)

BF > 1: Evidence for M₁
BF < 1: Evidence for M₂
```

### Bayesian Information Criterion (BIC)
```
BIC = -2 ln(L) + k ln(n)

k = number of parameters
n = number of samples

Lower BIC = better model
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is Bayesian inference?**
   - Update beliefs with data
   - Posterior = Likelihood × Prior
   - Quantifies uncertainty

2. **Prior vs Posterior?**
   - Prior: before data
   - Posterior: after data
   - Learning = updating prior

3. **MAP vs ML?**
   - MAP includes prior
   - ML ignores prior
   - MAP = regularized ML

4. **Why use Bayesian methods?**
   - Uncertainty quantification
   - Incorporate prior knowledge
   - Natural regularization

5. **Conjugate priors?**
   - Computational convenience
   - Closed-form posteriors
   - Beta-Binomial, Normal-Normal

---

## 📚 References

- **Books:** "Bayesian Data Analysis" - Gelman et al.
- **Online:** "Probabilistic Machine Learning" - Kevin Murphy

---

**Bayesian inference: principled uncertainty quantification!**
