# Generalized Linear Models

> **Extending linear regression** - Exponential family, link functions, and GLMs

---

## 🎯 What are GLMs?

**Generalized Linear Models** extend linear regression to:
- Non-normal response distributions
- Non-linear relationships via link functions
- Count data, binary data, positive data

### Components

1. **Random Component:** Y ~ Exponential family
2. **Systematic Component:** η = Xβ (linear predictor)
3. **Link Function:** g(μ) = η

---

## 📊 Exponential Family

### Form
```
f(y; θ, φ) = exp((yθ - b(θ))/a(φ) + c(y, φ))

θ: natural parameter
φ: dispersion parameter
```

### Common Distributions

**Normal:** Y ~ N(μ, σ²)
**Bernoulli:** Y ~ Bernoulli(p)
**Poisson:** Y ~ Poisson(λ)
**Gamma:** Y ~ Gamma(α, β)

---

## 🎯 Poisson Regression

**For count data:** Y ∈ {0, 1, 2, ...}

### Model
```
Y ~ Poisson(λ)
log(λ) = Xβ  (log link)

E[Y] = λ = exp(Xβ)
```

### Implementation

```python
import numpy as np
from sklearn.linear_model import PoissonRegressor
import statsmodels.api as sm

# Generate count data
np.random.seed(42)
X = np.random.randn(1000, 3)
lambda_true = np.exp(1 + 0.5*X[:, 0] - 0.3*X[:, 1] + 0.2*X[:, 2])
y = np.random.poisson(lambda_true)

# Sklearn
model_sklearn = PoissonRegressor()
model_sklearn.fit(X, y)

# Statsmodels (more detailed output)
X_sm = sm.add_constant(X)
model_sm = sm.GLM(y, X_sm, family=sm.families.Poisson())
result = model_sm.fit()
print(result.summary())

# Predictions
y_pred = model_sklearn.predict(X)
```

### Overdispersion

**Problem:** Var(Y) > E[Y] (violates Poisson assumption)

**Solution:** Negative Binomial or Quasi-Poisson

```python
# Negative Binomial
from statsmodels.discrete.discrete_model import NegativeBinomial

nb_model = NegativeBinomial(y, X_sm)
nb_result = nb_model.fit()
```

---

## 📈 Gamma Regression

**For positive continuous data:** Y > 0

### Model
```
Y ~ Gamma(α, β)
log(μ) = Xβ  (log link)

E[Y] = μ = exp(Xβ)
```

```python
# Generate gamma data
shape = 2.0
y_gamma = np.random.gamma(shape, np.exp(1 + 0.5*X[:, 0])/shape)

# Fit
gamma_model = sm.GLM(y_gamma, X_sm, family=sm.families.Gamma())
gamma_result = gamma_model.fit()
print(gamma_result.summary())
```

---

## 🎯 Link Functions

### Common Links

| Distribution | Canonical Link | Alternative |
|--------------|----------------|-------------|
| Normal | Identity: μ = η | - |
| Bernoulli | Logit: log(p/(1-p)) = η | Probit |
| Poisson | Log: log(λ) = η | Identity |
| Gamma | Inverse: 1/μ = η | Log |

### Custom Link

```python
# Log link for Poisson
class LogLink:
    def __call__(self, mu):
        return np.log(mu)
    
    def inverse(self, eta):
        return np.exp(eta)
    
    def deriv(self, mu):
        return 1 / mu
```

---

## 📊 Model Comparison

### Deviance

```
D = 2[log L(saturated) - log L(model)]

Lower deviance = better fit
```

### AIC & BIC

```
AIC = -2 log L + 2k
BIC = -2 log L + k log(n)

k: number of parameters
```

```python
# Compare models
print(f"AIC: {result.aic:.2f}")
print(f"BIC: {result.bic:.2f}")
print(f"Deviance: {result.deviance:.2f}")
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is a GLM?**
   - Generalizes linear regression
   - Exponential family distributions
   - Link function connects mean to linear predictor

2. **When to use Poisson regression?**
   - Count data
   - Non-negative integers
   - Events per time period

3. **Link function purpose?**
   - Transform mean to linear scale
   - Ensure valid predictions
   - Canonical link simplifies estimation

4. **GLM vs linear regression?**
   - GLM: flexible distributions
   - Linear: assumes normality
   - GLM includes linear as special case

---

## 📚 References

- **Books:** "Generalized Linear Models" - McCullagh & Nelder

---

**GLMs: flexible regression for real-world data!**
