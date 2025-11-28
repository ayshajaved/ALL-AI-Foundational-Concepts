# Measure Theory Basics

> **Rigorous foundations of probability** - σ-algebras, measurable spaces, and Lebesgue integration

---

## 🎯 Why Measure Theory?

**Motivation:**
- Rigorous foundation for probability
- Handle continuous probability spaces
- Define integration properly
- Understand convergence theorems

**Not needed for:**
- Practical ML implementation
- Most applied work

**Needed for:**
- Theoretical ML research
- Understanding proofs
- Advanced probability theory

---

## 📊 Measurable Spaces

### σ-Algebra

**Definition:** A collection F of subsets of Ω is a **σ-algebra** if:

1. **Contains Ω:** Ω ∈ F
2. **Closed under complements:** A ∈ F ⟹ Aᶜ ∈ F
3. **Closed under countable unions:** A₁, A₂, ... ∈ F ⟹ ⋃ᵢ Aᵢ ∈ F

**Example:**
```python
# Discrete example
Omega = {1, 2, 3, 4}

# Trivial σ-algebra
F_trivial = {set(), Omega}

# Power set (all subsets)
import itertools
F_power = [set(s) for r in range(len(Omega)+1) 
           for s in itertools.combinations(Omega, r)]

# Generated σ-algebra
# σ({1, 2}) = {∅, {1,2}, {3,4}, Ω}
```

### Measurable Space

**Definition:** (Ω, F) is a **measurable space**

- Ω: sample space
- F: σ-algebra on Ω

---

## 🎯 Measure

### Definition

A **measure** μ on (Ω, F) is a function μ: F → [0, ∞] such that:

1. **Non-negative:** μ(A) ≥ 0
2. **Null empty set:** μ(∅) = 0
3. **Countable additivity:** For disjoint A₁, A₂, ...:
   ```
   μ(⋃ᵢ Aᵢ) = Σᵢ μ(Aᵢ)
   ```

### Measure Space

**(Ω, F, μ)** is a **measure space**

### Examples

**1. Counting Measure**
```
μ(A) = |A|  (number of elements)
```

**2. Lebesgue Measure**
```
μ([a, b]) = b - a  (length of interval)
```

**3. Probability Measure**
```
P(Ω) = 1
```

---

## 📈 Lebesgue Integration

### Motivation

**Riemann integration limitations:**
- Only works for "nice" functions
- Doesn't handle limits well

**Lebesgue integration:**
- Works for more general functions
- Better convergence theorems

### Simple Functions

**Definition:** f is **simple** if:
```
f = Σᵢ aᵢ 1_{Aᵢ}

aᵢ: constants
1_{Aᵢ}: indicator function
```

**Integral of simple function:**
```
∫ f dμ = Σᵢ aᵢ μ(Aᵢ)
```

### General Functions

**For non-negative f:**
```
∫ f dμ = sup{∫ s dμ : s simple, s ≤ f}
```

**For general f:**
```
∫ f dμ = ∫ f⁺ dμ - ∫ f⁻ dμ

f⁺ = max(f, 0)
f⁻ = max(-f, 0)
```

---

## 🎯 Key Theorems

### 1. Monotone Convergence Theorem

**If:** 0 ≤ f₁ ≤ f₂ ≤ ... and fₙ → f

**Then:** ∫ fₙ dμ → ∫ f dμ

**Importance:** Can exchange limit and integral!

### 2. Dominated Convergence Theorem

**If:** 
- fₙ → f pointwise
- |fₙ| ≤ g for some integrable g

**Then:** ∫ fₙ dμ → ∫ f dμ

**Example:**
```python
# Conceptual illustration
import numpy as np

# Sequence of functions
def f_n(x, n):
    return n * x * np.exp(-n * x**2)

# Limit function
def f(x):
    return 0  # pointwise limit

# Dominated by g(x) = 1 on [0,1]
# So ∫ f_n dx → ∫ f dx = 0
```

### 3. Fubini's Theorem

**For product measures:**
```
∫∫ f(x,y) dμ(x)dν(y) = ∫∫ f(x,y) dν(y)dμ(x)

Can exchange order of integration
```

---

## 📊 Radon-Nikodym Theorem

### Absolute Continuity

**Definition:** ν is **absolutely continuous** w.r.t. μ (ν << μ) if:
```
μ(A) = 0 ⟹ ν(A) = 0
```

### Radon-Nikodym Derivative

**Theorem:** If ν << μ, there exists f such that:
```
ν(A) = ∫_A f dμ

f = dν/dμ  (Radon-Nikodym derivative)
```

**Example in Probability:**
```python
# Change of variables
# If X ~ f(x), Y = g(X)
# Then: f_Y(y) = f_X(g⁻¹(y)) |dg⁻¹/dy|

# This is Radon-Nikodym derivative!
```

---

## 🎯 Applications in ML

### 1. Continuous Probability

**Proper definition of PDF:**
```
P(X ∈ A) = ∫_A f(x) dμ(x)

f: probability density function
μ: Lebesgue measure
```

### 2. Expectation

**Rigorous definition:**
```
E[X] = ∫ X dP

P: probability measure
```

### 3. Convergence of Random Variables

**Almost sure convergence:**
```
Xₙ → X a.s. ⟺ P({ω : Xₙ(ω) → X(ω)}) = 1
```

**Convergence in probability:**
```
Xₙ →ᴾ X ⟺ ∀ε > 0, P(|Xₙ - X| > ε) → 0
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is σ-algebra?**
   - Collection of measurable sets
   - Closed under complements and countable unions
   - Defines what we can measure

2. **Lebesgue vs Riemann?**
   - Lebesgue: more general
   - Better convergence theorems
   - Measure horizontal slices vs vertical

3. **Why measure theory in ML?**
   - Rigorous probability foundation
   - Understand theoretical results
   - Not needed for practice

4. **Dominated convergence theorem?**
   - Exchange limit and integral
   - Need dominating function
   - Critical for many proofs

5. **Radon-Nikodym derivative?**
   - Generalizes change of variables
   - dν/dμ
   - Used in probability transformations

---

## 📚 References

- **Books:**
  - "Real Analysis" - Royden & Fitzpatrick
  - "Probability and Measure" - Billingsley
  - "Measure Theory" - Halmos

---

## 💡 Practical Note

**For most ML practitioners:**
- Understand concepts intuitively
- Don't need full rigor
- Focus on applications

**For ML researchers:**
- Measure theory is essential
- Needed for theoretical work
- Foundation of probability theory

---

**Measure theory: the rigorous foundation, not always needed but good to know!**
