# Model Interpretation

> **Understanding model predictions** - SHAP, LIME, and explainable AI

---

## 🎯 Why Model Interpretation?

**Critical for:**
- Trust and transparency
- Debugging models
- Regulatory compliance (GDPR, etc.)
- Business stakeholder communication
- Bias detection

---

## 📊 SHAP (SHapley Additive exPlanations)

### Theory
Based on game theory - Shapley values distribute prediction fairly among features

```
φᵢ = Σ_{S⊆F\{i}} |S|!(|F|-|S|-1)!/|F|! [f(S∪{i}) - f(S)]

φᵢ: SHAP value for feature i
F: all features
S: subset of features
```

### Implementation

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot
shap.summary_plot(shap_values[1], X, feature_names=data.feature_names)

# Force plot (single prediction)
shap.force_plot(explainer.expected_value[1], shap_values[1][0], 
                X[0], feature_names=data.feature_names)

# Dependence plot
shap.dependence_plot("mean radius", shap_values[1], X, 
                     feature_names=data.feature_names)
```

### SHAP for Different Models

```python
# For linear models
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X, y)
explainer_lr = shap.LinearExplainer(model_lr, X)

# For neural networks
import torch
model_nn = torch.nn.Sequential(...)
explainer_nn = shap.DeepExplainer(model_nn, X_train)

# For any model (slower)
explainer_kernel = shap.KernelExplainer(model.predict_proba, X_train[:100])
```

---

## 🎯 LIME (Local Interpretable Model-agnostic Explanations)

### Idea
Explain individual predictions by fitting local linear model

```python
from lime import lime_tabular

# Create explainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=data.feature_names,
    class_names=['benign', 'malignant'],
    mode='classification'
)

# Explain single prediction
i = 0
exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=10)

# Show explanation
exp.show_in_notebook()

# Get feature importance
print(exp.as_list())

# Plot
exp.as_pyplot_figure()
```

### LIME for Text

```python
from lime.lime_text import LimeTextExplainer

# Text classifier
text_explainer = LimeTextExplainer(class_names=['negative', 'positive'])

# Explain
exp = text_explainer.explain_instance(
    text_sample, 
    classifier.predict_proba,
    num_features=10
)
```

---

## 📈 Partial Dependence Plots (PDP)

### Shows marginal effect of features

```python
from sklearn.inspection import PartialDependenceDisplay

# Partial dependence
features = [0, 1, (0, 1)]  # Single features and interaction
PartialDependenceDisplay.from_estimator(
    model, X, features, 
    feature_names=data.feature_names
)
plt.show()

# Individual Conditional Expectation (ICE)
from sklearn.inspection import plot_partial_dependence
plot_partial_dependence(
    model, X, features,
    kind='both',  # PDP + ICE
    feature_names=data.feature_names
)
```

---

## 🎯 Permutation Importance

### Measure importance by shuffling features

```python
from sklearn.inspection import permutation_importance

# Compute permutation importance
perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42
)

# Sort by importance
sorted_idx = perm_importance.importances_mean.argsort()

# Plot
plt.barh(range(len(sorted_idx)), 
         perm_importance.importances_mean[sorted_idx])
plt.yticks(range(len(sorted_idx)), 
           np.array(data.feature_names)[sorted_idx])
plt.xlabel('Permutation Importance')
plt.tight_layout()
plt.show()
```

---

## 📊 Feature Importance Comparison

```python
import pandas as pd

# Get different importance measures
importance_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Tree_Importance': model.feature_importances_,
    'Permutation_Importance': perm_importance.importances_mean,
    'SHAP_Importance': np.abs(shap_values[1]).mean(axis=0)
})

# Normalize
for col in ['Tree_Importance', 'Permutation_Importance', 'SHAP_Importance']:
    importance_df[col] = importance_df[col] / importance_df[col].sum()

# Plot comparison
importance_df.set_index('Feature').plot(kind='barh', figsize=(10, 8))
plt.xlabel('Normalized Importance')
plt.title('Feature Importance Comparison')
plt.tight_layout()
plt.show()
```

---

## 🎓 Interview Focus

### Key Questions

1. **SHAP vs LIME?**
   - SHAP: global consistency, based on game theory
   - LIME: local explanations, model-agnostic
   - SHAP more theoretically sound

2. **When to use model interpretation?**
   - High-stakes decisions (medical, finance)
   - Regulatory requirements
   - Debugging unexpected predictions
   - Building trust with stakeholders

3. **Permutation vs tree importance?**
   - Permutation: model-agnostic, more reliable
   - Tree: fast, but biased toward high-cardinality
   - Use both for validation

4. **PDP limitations?**
   - Assumes feature independence
   - Can be misleading with correlated features
   - Use ICE plots to see heterogeneity

5. **Global vs local explanations?**
   - Global: overall model behavior (SHAP summary)
   - Local: individual predictions (LIME, SHAP force)
   - Need both for complete understanding

---

## 📚 References

- **Papers:**
  - "A Unified Approach to Interpreting Model Predictions" - Lundberg & Lee (SHAP)
  - "Why Should I Trust You?" - Ribeiro et al. (LIME)
  
- **Libraries:**
  - SHAP: `pip install shap`
  - LIME: `pip install lime`

---

## 💡 Best Practices

1. **Use multiple interpretation methods** - Don't rely on one
2. **Check for consistency** - Different methods should agree
3. **Validate with domain experts** - Do explanations make sense?
4. **Document interpretations** - For reproducibility
5. **Monitor in production** - Explanations can change over time

---

**Model interpretation: making black boxes transparent!**
