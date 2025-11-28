# Linear Regression

> **The foundation of supervised learning** - Predicting continuous values from features

---

## 🎯 What is Linear Regression?

**Goal:** Find the best-fitting line through data points

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
  = w₀ + wᵀx

ŷ: predicted value
w₀: intercept (bias)
w: weights (coefficients)
x: features
```

**Simple Linear Regression:** One feature (x → y)
**Multiple Linear Regression:** Multiple features (x₁, x₂, ..., xₙ → y)

---

## 📊 Mathematical Foundation

### Ordinary Least Squares (OLS)

**Objective:** Minimize sum of squared residuals

```
L(w) = Σᵢ (yᵢ - ŷᵢ)²
     = Σᵢ (yᵢ - wᵀxᵢ)²
     = ||y - Xw||²
```

### Closed-Form Solution

```
w* = (XᵀX)⁻¹Xᵀy

Requires: XᵀX is invertible
```

```python
import numpy as np

def linear_regression_closed_form(X, y):
    """
    Solve linear regression using normal equation
    
    X: (n_samples, n_features)
    y: (n_samples,)
    """
    # Add bias term
    X_bias = np.c_[np.ones(len(X)), X]
    
    # Normal equation
    w = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    
    return w

# Example
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1

w = linear_regression_closed_form(X, y)
print(f"Weights: {w}")
```

---

## 🎯 Gradient Descent Solution

### Batch Gradient Descent

```python
def linear_regression_gd(X, y, lr=0.01, epochs=1000):
    """
    Linear regression using gradient descent
    """
    n_samples, n_features = X.shape
    X_bias = np.c_[np.ones(n_samples), X]
    w = np.zeros(n_features + 1)
    
    for epoch in range(epochs):
        # Predictions
        y_pred = X_bias @ w
        
        # Gradient
        grad = -2/n_samples * X_bias.T @ (y - y_pred)
        
        # Update
        w = w - lr * grad
        
        # Loss
        if epoch % 100 == 0:
            loss = np.mean((y - y_pred)**2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w

w_gd = linear_regression_gd(X, y)
```

### Stochastic Gradient Descent (SGD)

```python
def linear_regression_sgd(X, y, lr=0.01, epochs=100):
    """
    Linear regression using SGD
    """
    n_samples, n_features = X.shape
    X_bias = np.c_[np.ones(n_samples), X]
    w = np.zeros(n_features + 1)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        for i in indices:
            # Single sample
            xi = X_bias[i:i+1]
            yi = y[i:i+1]
            
            # Prediction
            y_pred = xi @ w
            
            # Gradient
            grad = -2 * xi.T @ (yi - y_pred)
            
            # Update
            w = w - lr * grad.flatten()
    
    return w
```

---

## 📈 Using Scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

---

## 🎯 Model Assumptions

**LINEAR:** Linear relationship between X and y
**INDEPENDENCE:** Observations are independent
**NORMALITY:** Residuals are normally distributed
**EQUAL VARIANCE:** Homoscedasticity (constant variance)

### Checking Assumptions

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(12, 4))

# 1. Residuals vs Fitted
plt.subplot(131)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# 2. Q-Q plot
plt.subplot(132)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q')

# 3. Scale-Location
plt.subplot(133)
plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)
plt.xlabel('Fitted values')
plt.ylabel('√|Residuals|')
plt.title('Scale-Location')

plt.tight_layout()
plt.show()
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is linear regression?**
   - Predicts continuous output
   - Assumes linear relationship
   - Minimizes squared error

2. **Normal equation vs gradient descent?**
   - Normal: O(n³), exact solution
   - GD: O(kn²), iterative, works for large n

3. **What is R²?**
   - Coefficient of determination
   - R² = 1 - SS_res/SS_tot
   - Proportion of variance explained

4. **Assumptions of linear regression?**
   - Linearity, independence, normality, homoscedasticity

5. **When does normal equation fail?**
   - XᵀX not invertible (multicollinearity)
   - Too many features (n < p)
   - Use regularization or GD

---

## 📚 References

- **Books:** "Introduction to Statistical Learning" - James et al.
- **Scikit-learn:** [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

---

**Linear regression: simple yet powerful!**
