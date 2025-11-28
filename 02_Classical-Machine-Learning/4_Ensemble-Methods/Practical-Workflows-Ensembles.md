# Practical Workflows - Ensembles

> **Production ensemble systems** - XGBoost, LightGBM, hyperparameter tuning

---

## 🎯 Complete Ensemble Pipeline

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# XGBoost with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state=42)
grid = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")

# Use best model
best_model = grid.best_estimator_
```

---

## 📊 Model Comparison

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: {score:.4f}")
```

---

## 🎯 Feature Importance

```python
# XGBoost feature importance
import matplotlib.pyplot as plt

xgb.plot_importance(best_model, max_num_features=10)
plt.title('Top 10 Features')
plt.show()
```

---

**Production-ready ensemble workflows!**
