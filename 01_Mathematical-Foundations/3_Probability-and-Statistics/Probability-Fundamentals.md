# Probability Fundamentals

> **Foundation of uncertainty in AI** - Understanding randomness and probability

---

## 🎲 Basic Probability

### Sample Space and Events

**Sample Space (Ω):** Set of all possible outcomes
**Event (A):** Subset of sample space

**Example:**
```
Coin flip: Ω = {H, T}
Die roll: Ω = {1, 2, 3, 4, 5, 6}
Event A = {2, 4, 6} (even numbers)
```

### Probability Axioms

**Kolmogorov Axioms:**
```
1. P(A) ≥ 0 for all events A
2. P(Ω) = 1
3. P(A ∪ B) = P(A) + P(B) if A ∩ B = ∅
```

---

## 📊 Probability Rules

### Addition Rule
```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
```

### Multiplication Rule
```
P(A ∩ B) = P(A|B)P(B) = P(B|A)P(A)
```

### Conditional Probability
```
P(A|B) = P(A ∩ B) / P(B)
```

### Law of Total Probability
```
P(A) = Σᵢ P(A|Bᵢ)P(Bᵢ)
where {Bᵢ} partition the sample space
```

### Bayes' Theorem
```
P(A|B) = P(B|A)P(A) / P(B)

Posterior = (Likelihood × Prior) / Evidence
```

---

## 🎯 Random Variables

### Definition
A **random variable** X maps outcomes to real numbers:
```
X: Ω → ℝ
```

### Types

**Discrete:** Countable outcomes
```
X ∈ {x₁, x₂, ..., xₙ}
Example: Number of heads in coin flips
```

**Continuous:** Uncountable outcomes
```
X ∈ ℝ or interval
Example: Height, temperature
```

---

## 📈 Probability Distributions

### Probability Mass Function (PMF)
For discrete X:
```
pₓ(x) = P(X = x)

Properties:
- pₓ(x) ≥ 0
- Σₓ pₓ(x) = 1
```

### Probability Density Function (PDF)
For continuous X:
```
fₓ(x) such that P(a ≤ X ≤ b) = ∫ₐᵇ fₓ(x)dx

Properties:
- fₓ(x) ≥ 0
- ∫₋∞^∞ fₓ(x)dx = 1
```

### Cumulative Distribution Function (CDF)
```
Fₓ(x) = P(X ≤ x)

For discrete: Fₓ(x) = Σₜ≤ₓ pₓ(t)
For continuous: Fₓ(x) = ∫₋∞ˣ fₓ(t)dt
```

---

## 📊 Common Distributions

### Bernoulli
```
X ∈ {0, 1}
P(X = 1) = p
P(X = 0) = 1 - p

E[X] = p
Var(X) = p(1-p)
```

### Binomial
```
X = number of successes in n trials
P(X = k) = C(n,k)pᵏ(1-p)ⁿ⁻ᵏ

E[X] = np
Var(X) = np(1-p)
```

### Poisson
```
X = number of events in interval
P(X = k) = (λᵏe⁻λ) / k!

E[X] = λ
Var(X) = λ
```

### Uniform (Continuous)
```
X ~ U(a, b)
f(x) = 1/(b-a) for x ∈ [a,b]

E[X] = (a+b)/2
Var(X) = (b-a)²/12
```

### Gaussian (Normal)
```
X ~ N(μ, σ²)
f(x) = (1/√(2πσ²))exp(-(x-μ)²/(2σ²))

E[X] = μ
Var(X) = σ²
```

### Exponential
```
X ~ Exp(λ)
f(x) = λe⁻λˣ for x ≥ 0

E[X] = 1/λ
Var(X) = 1/λ²
```

---

## 💻 Practical Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Bernoulli
p = 0.7
bernoulli = stats.bernoulli(p)
print(f"P(X=1) = {bernoulli.pmf(1)}")

# Binomial
n, p = 10, 0.5
binomial = stats.binom(n, p)
print(f"P(X=5) = {binomial.pmf(5)}")
print(f"E[X] = {binomial.mean()}")

# Poisson
lambda_param = 3
poisson = stats.poisson(lambda_param)
print(f"P(X=2) = {poisson.pmf(2)}")

# Normal
mu, sigma = 0, 1
normal = stats.norm(mu, sigma)
print(f"P(X ≤ 0) = {normal.cdf(0)}")

# Generate samples
samples = normal.rvs(size=1000)
print(f"Sample mean: {samples.mean():.3f}")
print(f"Sample std: {samples.std():.3f}")

# Plot PDF
x = np.linspace(-4, 4, 100)
plt.plot(x, normal.pdf(x))
plt.title('Standard Normal Distribution')
plt.show()
```

---

## 🎯 Expected Value and Variance

### Expected Value (Mean)
```
Discrete: E[X] = Σₓ x·P(X=x)
Continuous: E[X] = ∫₋∞^∞ x·f(x)dx
```

**Properties:**
```
E[aX + b] = aE[X] + b
E[X + Y] = E[X] + E[Y]
E[XY] = E[X]E[Y] if X,Y independent
```

### Variance
```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
```

**Properties:**
```
Var(aX + b) = a²Var(X)
Var(X + Y) = Var(X) + Var(Y) if X,Y independent
```

### Standard Deviation
```
σ = √Var(X)
```

---

## 🔗 Joint Distributions

### Joint PMF/PDF
```
Discrete: p(x,y) = P(X=x, Y=y)
Continuous: f(x,y)
```

### Marginal Distributions
```
Discrete: pₓ(x) = Σᵧ p(x,y)
Continuous: fₓ(x) = ∫₋∞^∞ f(x,y)dy
```

### Independence
```
X and Y independent ⟺ p(x,y) = pₓ(x)pᵧ(y)
```

### Covariance
```
Cov(X,Y) = E[(X-E[X])(Y-E[Y])]
         = E[XY] - E[X]E[Y]
```

### Correlation
```
ρ(X,Y) = Cov(X,Y) / (σₓσᵧ)
ρ ∈ [-1, 1]
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is Bayes' theorem?**
   - P(A|B) = P(B|A)P(A)/P(B)
   - Updates beliefs with evidence
   - Foundation of Bayesian ML

2. **Difference between PMF and PDF?**
   - PMF: discrete (probabilities)
   - PDF: continuous (densities)
   - PDF can be > 1!

3. **What is independence?**
   - P(A∩B) = P(A)P(B)
   - Knowing B doesn't change P(A)
   - Critical assumption in ML

4. **Expected value vs mean?**
   - E[X]: theoretical average
   - Mean: sample average
   - Law of large numbers connects them

5. **Why is normal distribution important?**
   - Central limit theorem
   - Many natural phenomena
   - Mathematical convenience

### Must-Know Formulas

```
Bayes: P(A|B) = P(B|A)P(A)/P(B)
E[X] = Σx·P(X=x)
Var(X) = E[X²] - (E[X])²
Cov(X,Y) = E[XY] - E[X]E[Y]
```

---

## 📚 References

- **Books:** "Probability and Statistics" - DeGroot & Schervish
- **Online:** Khan Academy, 3Blue1Brown

---

**Probability is the language of uncertainty in AI!**
