# Support Vector Machines (SVM)

> **Maximum margin classification** - Finding the optimal separating hyperplane

---

## 🎯 Linear SVM

### Idea
Find hyperplane that maximizes margin between classes

```
Decision boundary: wᵀx + b = 0
Margin: 2/||w||

Maximize margin = Minimize ||w||²
```

### Hard Margin SVM

**For linearly separable data:**

```
minimize ½||w||²
subject to yᵢ(wᵀxᵢ + b) ≥ 1, ∀i
```

```python
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Linearly separable data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = 2*y - 1  # Convert to {-1, 1}

# Train
model = SVC(kernel='linear', C=1e10)  # Large C ≈ hard margin
model.fit(X, y)

# Support vectors
print(f"Support vectors: {model.support_vectors_}")
print(f"Number: {len(model.support_vectors_)}")
```

---

## 📊 Soft Margin SVM

**For non-separable data:**

```
minimize ½||w||² + C Σᵢ ξᵢ
subject to yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
           ξᵢ ≥ 0

ξᵢ: slack variables (allow misclassification)
C: regularization parameter
```

```python
# Soft margin
model_soft = SVC(kernel='linear', C=1.0)
model_soft.fit(X, y)

# C controls trade-off
C_values = [0.01, 0.1, 1, 10, 100]
for C in C_values:
    model = SVC(kernel='linear', C=C)
    model.fit(X_train, y_train)
    print(f"C={C}: {len(model.support_vectors_)} support vectors")
```

---

## 🎯 Kernel Trick

### Idea
Map to higher dimension where data is separable

```
φ: X → H (feature map)
K(x, x') = φ(x)ᵀφ(x') (kernel function)

Never compute φ explicitly!
```

### Common Kernels

**1. Linear**
```
K(x, x') = xᵀx'
```

**2. Polynomial**
```
K(x, x') = (γxᵀx' + r)ᵈ

d: degree
```

**3. RBF (Gaussian)**
```
K(x, x') = exp(-γ||x - x'||²)

γ: kernel coefficient
```

**4. Sigmoid**
```
K(x, x') = tanh(γxᵀx' + r)
```

```python
# RBF kernel
model_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)
model_rbf.fit(X, y)

# Polynomial kernel
model_poly = SVC(kernel='poly', degree=3, C=1.0)
model_poly.fit(X, y)

# Custom kernel
def custom_kernel(X, Y):
    return np.dot(X, Y.T)

model_custom = SVC(kernel=custom_kernel)
model_custom.fit(X, y)
```

---

## 📈 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")

# Use best model
best_model = grid.best_estimator_
```

---

## 🎯 Multi-class SVM

### One-vs-One (OvO)
```
Train K(K-1)/2 binary classifiers
Predict by voting
```

### One-vs-Rest (OvR)
```
Train K binary classifiers
Predict class with highest score
```

```python
# Multi-class (OvO by default)
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

model_multi = SVC(kernel='rbf', decision_function_shape='ovo')
model_multi.fit(X, y)

# OvR
model_ovr = SVC(kernel='rbf', decision_function_shape='ovr')
model_ovr.fit(X, y)
```

---

## 📊 SVM Regression (SVR)

```python
from sklearn.svm import SVR

# Regression
X_reg = np.sort(5 * np.random.rand(100, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.randn(100) * 0.1

model_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model_svr.fit(X_reg, y_reg)

# Predict
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred = model_svr.predict(X_test)
```

---

## 🎓 Interview Focus

### Key Questions

1. **What is SVM?**
   - Maximum margin classifier
   - Finds optimal hyperplane
   - Uses support vectors

2. **Kernel trick?**
   - Map to higher dimension
   - Never compute mapping explicitly
   - Use kernel function K(x,x')

3. **C parameter?**
   - Regularization strength
   - Large C: hard margin (less regularization)
   - Small C: soft margin (more regularization)

4. **γ in RBF kernel?**
   - Defines influence of single training example
   - Large γ: close influence (overfitting)
   - Small γ: far influence (underfitting)

5. **SVM vs Logistic Regression?**
   - SVM: maximum margin
   - LR: probabilistic
   - SVM better for small datasets
   - LR faster for large datasets

---

## 📚 References

- **Papers:** "A Tutorial on Support Vector Machines" - Burges

---

**SVM: powerful, kernel-based, margin-maximizing!**
