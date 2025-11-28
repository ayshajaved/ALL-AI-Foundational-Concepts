# Scikit-Learn Mastery

> **Advanced scikit-learn** - Pipelines, custom transformers, best practices

---

## 🎯 Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        self.param = param
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X * self.param

# Use in pipeline
pipeline = Pipeline([
    ('custom', CustomTransformer(param=2.0)),
    ('scaler', StandardScaler())
])
```

---

## 📊 Model Persistence

```python
import joblib

# Save
joblib.dump(pipeline, 'model.pkl')

# Load
loaded_pipeline = joblib.load('model.pkl')
```

---

**Master scikit-learn for production ML!**
