# Feature Preprocessing

> **Preparing features for ML** - Scaling, encoding, handling missing data

---

## 🎯 Feature Scaling

### StandardScaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### MinMaxScaler
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

---

## 📊 Encoding Categorical Variables

### One-Hot Encoding
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_categorical)
```

### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

---

## 📈 Handling Missing Data

```python
from sklearn.impute import SimpleImputer

# Mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Median, most_frequent, constant
imputer = SimpleImputer(strategy='median')
```

---

**Proper preprocessing is crucial for model performance!**
