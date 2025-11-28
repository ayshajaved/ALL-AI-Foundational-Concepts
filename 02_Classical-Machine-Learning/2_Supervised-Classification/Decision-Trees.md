# Decision Trees

> **Tree-based classification and regression** - Interpretable, non-parametric, powerful

---

## 🎯 How Decision Trees Work

### Idea
Recursively split data based on features to create pure leaf nodes

```
1. Start with all data at root
2. Find best feature/threshold to split
3. Create child nodes
4. Repeat until stopping criterion
```

---

## 📊 Splitting Criteria

### Gini Impurity (Classification)
```
Gini(D) = 1 - Σₖ pₖ²

pₖ: proportion of class k
Pure node: Gini = 0
```

### Entropy (Classification)
```
Entropy(D) = -Σₖ pₖ log₂(pₖ)

Pure node: Entropy = 0
```

### Information Gain
```
IG = Entropy(parent) - Σ (|Dᵢ|/|D|) Entropy(Dᵢ)

Choose split with highest IG
```

### MSE (Regression)
```
MSE(D) = (1/|D|) Σᵢ (yᵢ - ȳ)²
```

---

## 🎯 CART Algorithm

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train
model = DecisionTreeClassifier(
    criterion='gini',  # or 'entropy'
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X, y)

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True)
plt.show()

# Feature importance
importances = model.feature_importances_
for i, imp in enumerate(importances):
    print(f"{iris.feature_names[i]}: {imp:.4f}")
```

---

## 📈 Pruning

### Pre-pruning (Early Stopping)
```python
model_pruned = DecisionTreeClassifier(
    max_depth=5,           # Maximum tree depth
    min_samples_split=20,  # Min samples to split
    min_samples_leaf=10,   # Min samples in leaf
    max_leaf_nodes=20      # Max number of leaves
)
```

### Post-pruning (Cost Complexity)
```python
# Cost complexity pruning
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Train trees with different alphas
trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

# Find best alpha via cross-validation
train_scores = [tree.score(X_train, y_train) for tree in trees]
test_scores = [tree.score(X_test, y_test) for tree in trees]

plt.plot(ccp_alphas, train_scores, label='train')
plt.plot(ccp_alphas, test_scores, label='test')
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.legend()
plt.show()
```

---

## 🎯 Regression Trees

```python
from sklearn.tree import DecisionTreeRegressor

# Generate data
X_reg = np.sort(5 * np.random.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.randn(80) * 0.1

# Train
model_reg = DecisionTreeRegressor(max_depth=5)
model_reg.fit(X_reg, y_reg)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = model_reg.predict(X_test)

# Plot
plt.scatter(X_reg, y_reg, label='data')
plt.plot(X_test, y_pred, color='red', label='prediction')
plt.legend()
plt.show()
```

---

## 📊 Advantages & Disadvantages

### Advantages ✅
- Interpretable (can visualize)
- No feature scaling needed
- Handles non-linear relationships
- Handles mixed data types
- Feature importance

### Disadvantages ❌
- Prone to overfitting
- Unstable (small data change → different tree)
- Biased toward dominant classes
- Not great for extrapolation

---

## 🎓 Interview Focus

### Key Questions

1. **How do decision trees work?**
   - Recursive binary splitting
   - Choose best feature/threshold
   - Maximize information gain

2. **Gini vs Entropy?**
   - Both measure impurity
   - Gini faster to compute
   - Usually similar results

3. **Overfitting in trees?**
   - Trees can memorize training data
   - Use pruning (max_depth, min_samples)
   - Or use ensembles (Random Forest)

4. **Feature importance?**
   - Based on reduction in impurity
   - Sum over all splits using feature
   - Normalized to sum to 1

5. **Trees vs linear models?**
   - Trees: non-linear, interpretable
   - Linear: faster, better for linear data
   - Trees need more data

---

## 📚 References

- **Books:** "The Elements of Statistical Learning" - Hastie et al.

---

**Decision trees: interpretable, powerful, foundation of ensembles!**
