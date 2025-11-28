# End-to-End ML Pipeline

> **Complete ML workflow** - From problem to production

---

## 🎯 Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define transformers
numeric_features = ['age', 'income']
categorical_features = ['gender', 'city']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
```

---

**Build robust, reproducible ML pipelines!**
