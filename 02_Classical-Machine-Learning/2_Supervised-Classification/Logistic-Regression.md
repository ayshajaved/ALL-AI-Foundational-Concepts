# Logistic Regression

> **Binary and multi-class classification** - The foundation of classification algorithms

---

## 🎯 Binary Logistic Regression

### Model
```
P(y=1|x) = σ(wᵀx) = 1/(1 + e^(-wᵀx))

σ: sigmoid function
Decision boundary: wᵀx = 0
```

### Sigmoid Function
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Visualize
z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()
```

---

## 📊 Maximum Likelihood Estimation

### Log-Likelihood
```
L(w) = Σᵢ [yᵢ log(σ(wᵀxᵢ)) + (1-yᵢ)log(1-σ(wᵀxᵢ))]

Maximize L(w) = Minimize -L(w)
```

### Gradient
```
∇L = Σᵢ (yᵢ - σ(wᵀxᵢ))xᵢ
```

### Implementation from Scratch
```python
def logistic_regression_gd(X, y, lr=0.01, epochs=1000):
    """Logistic regression using gradient descent"""
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(epochs):
        # Linear combination
        z = X @ w + b
        
        # Predictions
        y_pred = sigmoid(z)
        
        # Gradients
        dw = (1/n_samples) * X.T @ (y_pred - y)
        db = (1/n_samples) * np.sum(y_pred - y)
        
        # Update
        w -= lr * dw
        b -= lr * db
        
        # Loss (binary cross-entropy)
        if epoch % 100 == 0:
            loss = -np.mean(y*np.log(y_pred + 1e-10) + (1-y)*np.log(1-y_pred + 1e-10))
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w, b

# Example
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
w, b = logistic_regression_gd(X, y)
```

---

## 🎯 Using Scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

---

## 📈 Multi-class Classification

### One-vs-Rest (OvR)
```
Train K binary classifiers (one per class)
Predict class with highest probability
```

### Softmax (Multinomial)
```
P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)

Generalizes sigmoid to K classes
```

```python
# Multi-class
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# OvR (default)
model_ovr = LogisticRegression(multi_class='ovr')
model_ovr.fit(X, y)

# Multinomial
model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_multi.fit(X, y)

print(f"OvR accuracy: {model_ovr.score(X, y):.4f}")
print(f"Multinomial accuracy: {model_multi.score(X, y):.4f}")
```

---

## 🎯 Regularization

### L2 (Ridge)
```
L(w) = -log-likelihood + λ||w||²

Sklearn: penalty='l2', C=1/λ
```

### L1 (Lasso)
```
L(w) = -log-likelihood + λ||w||₁

Sklearn: penalty='l1', solver='liblinear'
```

```python
# Regularization strength
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
accuracies = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    accuracies.append(model.score(X_test, y_test))

plt.plot(C_values, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('C (1/λ)')
plt.ylabel('Accuracy')
plt.title('Regularization Strength')
plt.show()
```

---

## 📊 Decision Boundary Visualization

```python
def plot_decision_boundary(model, X, y):
    """Plot 2D decision boundary"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# Example with 2 features
X_2d, y_2d = make_classification(n_samples=200, n_features=2, 
                                  n_redundant=0, n_informative=2,
                                  random_state=42)
model_2d = LogisticRegression()
model_2d.fit(X_2d, y_2d)
plot_decision_boundary(model_2d, X_2d, y_2d)
```

---

## 🎓 Interview Focus

### Key Questions

1. **Logistic regression for classification?**
   - Predicts probabilities
   - Uses sigmoid function
   - Linear decision boundary

2. **Why sigmoid?**
   - Maps to [0, 1]
   - Smooth, differentiable
   - Probabilistic interpretation

3. **Loss function?**
   - Binary cross-entropy
   - Negative log-likelihood
   - Convex for optimization

4. **Linear vs logistic regression?**
   - Linear: continuous output
   - Logistic: probability/class
   - Both use linear combination

5. **Multi-class strategies?**
   - OvR: K binary classifiers
   - Softmax: single K-class model
   - Softmax generally better

---

## 📚 References

- **Books:** "Pattern Recognition and Machine Learning" - Bishop

---

**Logistic regression: simple, interpretable, effective!**
