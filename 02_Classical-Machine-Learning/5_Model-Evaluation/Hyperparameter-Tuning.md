# Hyperparameter Tuning

> **Optimizing model performance** - Grid search, random search, Bayesian optimization

---

## 🎯 Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'poly']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

---

## 📊 Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'C': uniform(0.1, 10),
    'gamma': uniform(0.001, 0.1),
    'kernel': ['rbf', 'poly']
}

random = RandomizedSearchCV(SVC(), param_dist, n_iter=20, cv=5, random_state=42)
random.fit(X_train, y_train)
```

---

## 🎯 Bayesian Optimization

```python
from skopt import BayesSearchCV

param_space = {
    'C': (0.1, 10.0, 'log-uniform'),
    'gamma': (0.001, 0.1, 'log-uniform')
}

bayes = BayesSearchCV(SVC(), param_space, n_iter=32, cv=5, random_state=42)
bayes.fit(X_train, y_train)
```

---

**Find optimal hyperparameters efficiently!**
