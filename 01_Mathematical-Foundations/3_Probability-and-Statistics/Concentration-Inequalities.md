# Concentration Inequalities

> **Bounding tail probabilities** - Essential for learning theory

---

## 🎯 Why Concentration Inequalities?

**Problem:** How much does sample mean deviate from true mean?

**Answer:** Concentration inequalities provide probabilistic bounds

**Applications:**
- Generalization bounds in ML
- PAC learning theory
- Algorithm analysis

---

## 📊 Markov's Inequality

### Statement
For non-negative random variable X:
```
P(X ≥ a) ≤ E[X] / a
```

**Example:**
```python
# If average height is 170cm
# P(height ≥ 340cm) ≤ 170/340 = 0.5
# Weak bound, but always valid!
```

---

## 🎯 Chebyshev's Inequality

### Statement
```
P(|X - μ| ≥ kσ) ≤ 1/k²

where μ = E[X], σ² = Var(X)
```

**Equivalent form:**
```
P(|X - μ| ≥ ε) ≤ σ²/ε²
```

**Example:**
```python
import numpy as np

# Generate data
data = np.random.randn(10000)
mu = np.mean(data)
sigma = np.std(data)

# Chebyshev bound for k=2
k = 2
empirical_prob = np.mean(np.abs(data - mu) >= k*sigma)
chebyshev_bound = 1/k**2

print(f"Empirical: {empirical_prob:.4f}")
print(f"Chebyshev bound: {chebyshev_bound:.4f}")
# Empirical ≤ Bound (usually much smaller)
```

---

## 🔥 Hoeffding's Inequality

### Statement
For independent bounded r.v. X₁,...,Xₙ ∈ [a,b]:
```
P(|X̄ - μ| ≥ ε) ≤ 2exp(-2nε²/(b-a)²)

where X̄ = (X₁ + ... + Xₙ)/n
```

**Key insight:** Exponential decay in n!

**Example:**
```python
def hoeffding_bound(n, epsilon, a=0, b=1):
    """Hoeffding bound for sample mean"""
    return 2 * np.exp(-2 * n * epsilon**2 / (b - a)**2)

# How many samples for ε=0.1 with 95% confidence?
epsilon = 0.1
delta = 0.05  # 1 - confidence

n_required = int(np.ceil((b-a)**2 * np.log(2/delta) / (2*epsilon**2)))
print(f"Samples needed: {n_required}")
```

---

## 📈 Chernoff Bound

### Statement
For sum of independent Bernoulli r.v.:
```
X = X₁ + ... + Xₙ, Xᵢ ~ Bernoulli(p)

P(X ≥ (1+δ)np) ≤ exp(-δ²np/3)  for δ ∈ [0,1]
P(X ≤ (1-δ)np) ≤ exp(-δ²np/2)  for δ ∈ [0,1]
```

**Tighter than Hoeffding for Bernoulli!**

---

## 🎯 Applications in ML

### 1. Sample Complexity

**Question:** How many samples to learn with error ≤ ε?

**Answer (via Hoeffding):**
```
n ≥ (1/(2ε²)) log(2/δ)
```

```python
def sample_complexity(epsilon, delta):
    """Samples needed for ε-accurate estimate"""
    return int(np.ceil(np.log(2/delta) / (2*epsilon**2)))

# For ε=0.01, δ=0.05
n = sample_complexity(0.01, 0.05)
print(f"Need {n} samples")  # ~18,445
```

### 2. Generalization Bound

**Empirical risk:** R̂(h) = (1/n)Σᵢ L(h(xᵢ), yᵢ)
**True risk:** R(h) = E[L(h(x), y)]

**Hoeffding bound:**
```
P(|R(h) - R̂(h)| ≥ ε) ≤ 2exp(-2nε²)
```

With probability ≥ 1-δ:
```
R(h) ≤ R̂(h) + √(log(2/δ)/(2n))
```

```python
def generalization_bound(n, delta):
    """Upper bound on generalization error"""
    return np.sqrt(np.log(2/delta) / (2*n))

# For n=1000 samples
bound = generalization_bound(1000, 0.05)
print(f"Generalization error ≤ train error + {bound:.4f}")
```

### 3. PAC Learning

**Probably Approximately Correct (PAC):**
```
P(R(h) ≤ ε) ≥ 1 - δ
```

**Sample complexity:**
```
n = O((1/ε²)log(|H|/δ))

where |H| = hypothesis class size
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is Hoeffding's inequality?**
   - Bounds deviation of sample mean
   - Exponential decay in n
   - Doesn't need variance!

2. **Why concentration inequalities matter?**
   - Generalization bounds
   - Sample complexity
   - PAC learning theory

3. **Hoeffding vs Chebyshev?**
   - Hoeffding: exponential, needs bounded r.v.
   - Chebyshev: polynomial, only needs variance

4. **Sample complexity formula?**
   - n = O(1/ε² log(1/δ))
   - Quadratic in 1/ε
   - Logarithmic in 1/δ

---

## 📚 References

- **Books:** "Concentration Inequalities" - Boucheron et al.
- **Papers:** "A Few Useful Things to Know about Machine Learning" - Domingos

---

**Concentration inequalities: theoretical foundation of ML!**
