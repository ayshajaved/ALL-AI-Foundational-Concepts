# Practical Workflows - Regression

> **End-to-end regression projects** - From data to deployment

---

## 🎯 Complete Regression Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Explore
print(df.describe())
print(df.isnull().sum())

# 3. Prepare
X = df.drop('target', axis=1)
y = df['target']

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 8. Visualize
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predictions vs Actual')
plt.show()
```

---

## 📊 Model Comparison

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, 
                            cv=5, scoring='r2')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Plot comparison
names = list(results.keys())
means = [results[name]['mean'] for name in names]
stds = [results[name]['std'] for name in names]

plt.barh(names, means, xerr=stds)
plt.xlabel('R² Score')
plt.title('Model Comparison')
plt.show()
```

---

## 🎯 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'alpha': np.logspace(-3, 3, 20)
}

# Grid search
grid = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {-grid.best_score_:.4f}")

# Use best model
best_model = grid.best_estimator_
```

---

## 📈 Production Deployment

```python
import joblib

# Save model
joblib.dump(best_model, 'regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load model
loaded_model = joblib.load('regression_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Predict on new data
def predict_new(X_new):
    X_scaled = loaded_scaler.transform(X_new)
    return loaded_model.predict(X_scaled)

# API endpoint (Flask example)
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_new = pd.DataFrame(data)
    predictions = predict_new(X_new)
    return jsonify({'predictions': predictions.tolist()})
```

---

## 🎓 Best Practices

1. **Always split data first**
2. **Scale features for regularized models**
3. **Use cross-validation**
4. **Check residuals**
5. **Monitor in production**

---

**Complete workflows for production-ready regression!**
