# Naive Bayes

> **Probabilistic classification** - Simple, fast, and surprisingly effective

---

## 🎯 Bayes' Theorem (Review)

```
P(y|x) = P(x|y)P(y) / P(x)

Posterior = (Likelihood × Prior) / Evidence
```

### Naive Assumption
**Features are conditionally independent given class:**
```
P(x₁, x₂, ..., xₙ|y) = P(x₁|y)P(x₂|y)...P(xₙ|y)
```

---

## 📊 Classification Rule

```
ŷ = argmax_y P(y|x)
  = argmax_y P(x|y)P(y)
  = argmax_y P(y) ∏ᵢ P(xᵢ|y)
```

### Log-Space (Numerical Stability)
```
ŷ = argmax_y [log P(y) + Σᵢ log P(xᵢ|y)]
```

---

## 🎯 Gaussian Naive Bayes

**For continuous features**

### Model
```
P(xᵢ|y) ~ N(μᵧ,ᵢ, σ²ᵧ,ᵢ)

Estimate μ and σ² from training data
```

### Implementation

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train
model = GaussianNB()
model.fit(X, y)

# Predict
y_pred = model.predict(X)
y_prob = model.predict_proba(X)

print(f"Accuracy: {model.score(X, y):.4f}")

# From scratch
class GaussianNB_scratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.prior[c] = len(X_c) / len(X)
    
    def _gaussian_pdf(self, x, mean, var):
        return np.exp(-0.5 * ((x - mean)**2 / var)) / np.sqrt(2 * np.pi * var)
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.prior[c])
                likelihood = np.sum(np.log(self._gaussian_pdf(x, self.mean[c], self.var[c])))
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
```

---

## 📈 Multinomial Naive Bayes

**For count/frequency data (text classification)**

### Model
```
P(xᵢ|y) = (count(xᵢ, y) + α) / (Σⱼ count(xⱼ, y) + α|V|)

α: smoothing parameter (Laplace smoothing)
|V|: vocabulary size
```

### Text Classification Example

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Example documents
docs = [
    "I love this movie",
    "This movie is great",
    "I hate this film",
    "This film is terrible"
]
labels = [1, 1, 0, 0]  # 1=positive, 0=negative

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

# Train
model = MultinomialNB(alpha=1.0)
model.fit(X, labels)

# Predict
test_docs = ["I love this film"]
X_test = vectorizer.transform(test_docs)
print(f"Prediction: {model.predict(X_test)}")
print(f"Probabilities: {model.predict_proba(X_test)}")
```

---

## 🎯 Bernoulli Naive Bayes

**For binary features**

### Model
```
P(xᵢ|y) = P(xᵢ=1|y)^xᵢ × (1-P(xᵢ=1|y))^(1-xᵢ)
```

```python
from sklearn.naive_bayes import BernoulliNB

# Binary features
X_binary = (X > X.mean()).astype(int)

model = BernoulliNB()
model.fit(X_binary, y)
```

---

## 📊 Advantages & Disadvantages

### Advantages ✅
- Fast training and prediction
- Works well with high dimensions
- Requires little training data
- Handles missing values naturally
- Probabilistic predictions

### Disadvantages ❌
- Independence assumption rarely true
- Can be outperformed by discriminative models
- Sensitive to feature scaling (Gaussian)
- Zero-frequency problem (needs smoothing)

---

## 🎓 Interview Focus

### Key Questions

1. **Why "naive"?**
   - Assumes feature independence
   - Rarely true in practice
   - Works surprisingly well anyway

2. **When to use Naive Bayes?**
   - Text classification
   - Spam filtering
   - High-dimensional data
   - Need fast baseline

3. **Gaussian vs Multinomial?**
   - Gaussian: continuous features
   - Multinomial: count/frequency data
   - Bernoulli: binary features

4. **Laplace smoothing?**
   - Prevents zero probabilities
   - Add α to all counts
   - α=1 is common (Laplace)

5. **Naive Bayes vs Logistic Regression?**
   - NB: generative model
   - LR: discriminative model
   - LR usually better with more data

---

## 📚 References

- **Papers:** "Naive Bayes at Forty" - Hand & Yu

---

**Naive Bayes: simple assumptions, powerful results!**
