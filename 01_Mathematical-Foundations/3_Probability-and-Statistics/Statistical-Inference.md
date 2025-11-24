# Statistical Inference

> **Learning from data** - Estimation, confidence, and hypothesis testing

---

## 📊 Point Estimation

### Estimators

**Estimator:** Function of data that estimates parameter
```
θ̂ = g(X₁, X₂, ..., Xₙ)
```

### Properties of Estimators

**1. Unbiased**
```
E[θ̂] = θ
```

**2. Consistent**
```
θ̂ → θ as n → ∞
```

**3. Efficient**
```
Minimum variance among unbiased estimators
```

---

## 🎯 Maximum Likelihood Estimation (MLE)

### Definition
```
θ̂_ML = argmax_θ L(θ|data)
      = argmax_θ P(data|θ)
```

### Log-Likelihood
```
ℓ(θ) = log L(θ) = Σᵢ log P(xᵢ|θ)
```

### Example: Normal Distribution
```python
import numpy as np

# Data
data = np.random.randn(1000) * 2 + 5

# MLE for μ and σ²
mu_mle = np.mean(data)
sigma2_mle = np.var(data, ddof=0)  # MLE uses n, not n-1

print(f"μ̂ = {mu_mle:.3f}")
print(f"σ̂² = {sigma2_mle:.3f}")
```

---

## 📈 Confidence Intervals

### Definition
```
P(θ ∈ [L, U]) = 1 - α

[L, U] is (1-α)×100% confidence interval
```

**Common:** α = 0.05 → 95% CI

### For Normal Distribution
```python
from scipy import stats

# Sample mean CI
data = np.random.randn(100)
mean = np.mean(data)
se = stats.sem(data)  # Standard error

# 95% CI
ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

---

## 🧪 Hypothesis Testing

### Framework

**Null hypothesis H₀:** Default assumption
**Alternative H₁:** What we want to show

**Test statistic:** Measure of evidence against H₀
**p-value:** P(observe data | H₀ true)

**Decision:**
- p < α: Reject H₀
- p ≥ α: Fail to reject H₀

### t-Test

```python
from scipy.stats import ttest_1samp, ttest_ind

# One-sample t-test
# H₀: μ = μ₀
data = np.random.randn(100) + 0.5
t_stat, p_value = ttest_1samp(data, 0)
print(f"t = {t_stat:.3f}, p = {p_value:.3f}")

# Two-sample t-test
# H₀: μ₁ = μ₂
group1 = np.random.randn(100)
group2 = np.random.randn(100) + 0.5
t_stat, p_value = ttest_ind(group1, group2)
print(f"t = {t_stat:.3f}, p = {p_value:.3f}")
```

---

## 📊 Bootstrap

### Idea
Resample data with replacement to estimate sampling distribution

```python
def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, alpha=0.05):
    """Compute bootstrap confidence interval"""
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))
    
    lower = np.percentile(bootstrap_stats, 100*alpha/2)
    upper = np.percentile(bootstrap_stats, 100*(1-alpha/2))
    
    return lower, upper

# Example
data = np.random.randn(100)
ci = bootstrap_ci(data)
print(f"Bootstrap 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

---

## 🎯 Central Limit Theorem

### Statement
```
X̄ₙ = (X₁ + ... + Xₙ)/n

As n → ∞:
X̄ₙ ~ N(μ, σ²/n)
```

**Implications:**
- Sample mean is approximately normal
- Regardless of original distribution!
- Foundation of many statistical methods

```python
# Demonstration
from scipy.stats import expon

# Non-normal distribution (exponential)
samples = []
for _ in range(10000):
    sample = expon.rvs(size=30)  # n=30
    samples.append(np.mean(sample))

# Sample means are approximately normal!
plt.hist(samples, bins=50, density=True)
plt.title('Distribution of Sample Means (CLT)')
plt.show()
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is MLE?**
   - Maximum likelihood estimation
   - Find parameters that maximize P(data|θ)
   - Asymptotically optimal

2. **Confidence interval interpretation?**
   - 95% CI: If we repeat experiment, 95% of CIs contain true θ
   - NOT: θ has 95% probability of being in interval

3. **p-value meaning?**
   - P(observe data | H₀ true)
   - NOT: P(H₀ true | data)
   - Small p-value = evidence against H₀

4. **Central Limit Theorem?**
   - Sample mean → Normal as n increases
   - Works for any distribution
   - Foundation of inference

5. **Bootstrap vs analytical CI?**
   - Bootstrap: resampling, no assumptions
   - Analytical: assumes distribution
   - Bootstrap more general

---

## 📚 References

- **Books:** "Statistical Inference" - Casella & Berger
- **Online:** Khan Academy Statistics

---

**Statistical inference: learning from data with rigor!**
