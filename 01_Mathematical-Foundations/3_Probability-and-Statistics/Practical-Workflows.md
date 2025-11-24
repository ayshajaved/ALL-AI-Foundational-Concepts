# Practical Workflows - Probability & Statistics

> **Hands-on statistical analysis in Python** - NumPy, SciPy, statsmodels

---

## 🛠️ Essential Libraries

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats import proportion, power
```

---

## 📊 Common Workflows

### 1. Descriptive Statistics

```python
# Generate data
data = np.random.randn(1000) * 10 + 50

# Central tendency
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=True)[0][0]

# Dispersion
std = np.std(data, ddof=1)  # Sample std
var = np.var(data, ddof=1)
iqr = stats.iqr(data)

# Shape
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

print(f"Mean: {mean:.2f}")
print(f"Std: {std:.2f}")
print(f"Skewness: {skewness:.2f}")
```

### 2. Probability Distributions

```python
# Normal distribution
mu, sigma = 0, 1
normal = stats.norm(mu, sigma)

# PDF, CDF, quantiles
x = 1.96
print(f"PDF(x): {normal.pdf(x):.4f}")
print(f"CDF(x): {normal.cdf(x):.4f}")
print(f"95th percentile: {normal.ppf(0.95):.4f}")

# Generate samples
samples = normal.rvs(size=1000)

# Fit distribution to data
params = stats.norm.fit(data)
print(f"Fitted μ={params[0]:.2f}, σ={params[1]:.2f}")
```

### 3. Hypothesis Testing

```python
# One-sample t-test
t_stat, p_value = stats.ttest_1samp(data, 50)
print(f"t={t_stat:.3f}, p={p_value:.4f}")

# Two-sample t-test
group1 = np.random.randn(100)
group2 = np.random.randn(100) + 0.5
t_stat, p_value = stats.ttest_ind(group1, group2)

# Paired t-test
before = np.random.randn(50)
after = before + np.random.randn(50) * 0.5 + 0.3
t_stat, p_value = stats.ttest_rel(before, after)

# Chi-square test
observed = [50, 30, 20]
expected = [40, 40, 20]
chi2, p_value = stats.chisquare(observed, expected)
```

### 4. Correlation and Regression

```python
# Pearson correlation
x = np.random.randn(100)
y = 2*x + np.random.randn(100)
corr, p_value = stats.pearsonr(x, y)
print(f"Correlation: {corr:.3f}, p={p_value:.4f}")

# Spearman correlation (rank-based)
corr, p_value = stats.spearmanr(x, y)

# Linear regression
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(x, y)
print(f"y = {slope:.2f}x + {intercept:.2f}")
print(f"R² = {r_value**2:.3f}")
```

---

## 🎯 ML-Specific Patterns

### 1. Train-Test Split with Stratification

```python
from sklearn.model_selection import train_test_split

# Stratified split (preserve class distribution)
X = np.random.randn(1000, 10)
y = np.random.randint(0, 3, 1000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Verify stratification
print(f"Train class dist: {np.bincount(y_train) / len(y_train)}")
print(f"Test class dist: {np.bincount(y_test) / len(y_test)}")
```

### 2. Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)

print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

### 3. Statistical Tests for Model Comparison

```python
from scipy.stats import wilcoxon, mannwhitneyu

# Wilcoxon signed-rank test (paired)
model1_scores = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
model2_scores = np.array([0.86, 0.88, 0.85, 0.87, 0.86])

stat, p_value = wilcoxon(model1_scores, model2_scores)
print(f"Wilcoxon p-value: {p_value:.4f}")

# Mann-Whitney U test (unpaired)
stat, p_value = mannwhitneyu(model1_scores, model2_scores)
```

---

## 📈 Visualization

```python
# Distribution plot
sns.histplot(data, kde=True)
plt.title('Distribution with KDE')
plt.show()

# Q-Q plot (check normality)
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Box plot
sns.boxplot(data=[group1, group2])
plt.title('Group Comparison')
plt.show()

# Correlation heatmap
corr_matrix = np.corrcoef(X.T)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
```

---

## 🔧 Power Analysis

```python
from statsmodels.stats.power import ttest_power

# Required sample size
effect_size = 0.5  # Cohen's d
alpha = 0.05
power = 0.8

from statsmodels.stats.power import tt_solve_power
n = tt_solve_power(effect_size=effect_size, alpha=alpha, 
                   power=power, alternative='two-sided')
print(f"Required sample size: {int(np.ceil(n))}")
```

---

## 📚 Quick Reference

```python
# SciPy stats
stats.norm(mu, sigma)      # Normal distribution
stats.binom(n, p)          # Binomial
stats.poisson(lambda)      # Poisson
stats.expon(scale)         # Exponential

# Tests
stats.ttest_1samp()        # One-sample t-test
stats.ttest_ind()          # Two-sample t-test
stats.ttest_rel()          # Paired t-test
stats.chisquare()          # Chi-square test
stats.kstest()             # Kolmogorov-Smirnov test

# Correlation
stats.pearsonr()           # Pearson correlation
stats.spearmanr()          # Spearman correlation
```

---

**Master these workflows for data-driven AI development!**
