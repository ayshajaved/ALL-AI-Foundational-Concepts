# AI Interview Cheatsheet

> **Quick reference for AI/ML interviews** - Essential concepts, formulas, and facts you must know

---

## 🧠 Machine Learning Essentials

### Bias-Variance Tradeoff
- **Bias:** Error from wrong assumptions (underfitting)
- **Variance:** Error from sensitivity to training data (overfitting)
- **Goal:** Minimize total error = Bias² + Variance + Irreducible Error

### Train/Validation/Test Split
- **Training:** 60-80% - Learn parameters
- **Validation:** 10-20% - Tune hyperparameters
- **Test:** 10-20% - Final evaluation (never touch during development!)

### Cross-Validation
- **K-Fold:** Split data into K folds, train on K-1, validate on 1, repeat K times
- **Stratified:** Preserve class distribution in each fold
- **Leave-One-Out:** K = N (expensive but unbiased)

### Regularization
- **L1 (Lasso):** `λ Σ|w|` → Sparse weights, feature selection
- **L2 (Ridge):** `λ Σw²` → Small weights, prevents overfitting
- **Elastic Net:** Combines L1 + L2

---

## 📊 Key Metrics

### Classification
- **Accuracy:** `(TP + TN) / Total`
- **Precision:** `TP / (TP + FP)` - Of predicted positive, how many correct?
- **Recall (Sensitivity):** `TP / (TP + FN)` - Of actual positive, how many found?
- **F1-Score:** `2 × (Precision × Recall) / (Precision + Recall)`
- **ROC-AUC:** Area under ROC curve (TPR vs FPR)

### Regression
- **MSE:** `(1/n) Σ(y - ŷ)²`
- **RMSE:** `√MSE`
- **MAE:** `(1/n) Σ|y - ŷ|`
- **R²:** `1 - (SS_res / SS_tot)` (0-1, higher better)

---

## 🔥 Deep Learning Essentials

### Activation Functions
| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| Sigmoid | `1/(1+e^-x)` | (0,1) | Binary classification output |
| Tanh | `(e^x - e^-x)/(e^x + e^-x)` | (-1,1) | Hidden layers (zero-centered) |
| ReLU | `max(0,x)` | [0,∞) | Default choice, fast |
| Leaky ReLU | `max(0.01x, x)` | (-∞,∞) | Fixes dying ReLU |
| GELU | `x·Φ(x)` | (-∞,∞) | Transformers (smooth) |

### Optimizers
- **SGD:** `w = w - η∇L`
- **Momentum:** `v = βv + ∇L; w = w - ηv`
- **Adam:** Combines momentum + RMSprop (most popular)
  - `m = β₁m + (1-β₁)∇L`
  - `v = β₂v + (1-β₂)∇L²`
  - `w = w - η·m̂/√(v̂+ε)`

### Loss Functions
- **Binary Cross-Entropy:** `-[y log(ŷ) + (1-y)log(1-ŷ)]`
- **Categorical Cross-Entropy:** `-Σ y_i log(ŷ_i)`
- **MSE:** `(1/n)Σ(y-ŷ)²`
- **Huber:** Combines MSE + MAE (robust to outliers)

### Batch Normalization
- **Formula:** `x̂ = (x - μ) / √(σ² + ε)`
- **Benefits:** Faster training, higher learning rates, regularization
- **When:** After linear layer, before activation

---

## 🏗️ Architecture Patterns

### CNN Components
- **Convolution:** Extract spatial features
- **Pooling:** Downsample (Max, Average)
- **Stride:** Step size of filter
- **Padding:** Preserve spatial dimensions
- **Receptive Field:** Input region affecting one output

### Transformer Components
- **Self-Attention:** `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Multi-Head:** Multiple parallel attention mechanisms
- **Positional Encoding:** Add position info (sin/cos)
- **Feed-Forward:** Two linear layers with activation

### ResNet Key Idea
- **Skip Connections:** `F(x) + x` instead of `F(x)`
- **Why:** Solves vanishing gradient, enables very deep networks

---

## 🎮 Reinforcement Learning

### Core Concepts
- **MDP:** (S, A, P, R, γ)
  - S: States, A: Actions, P: Transition probabilities
  - R: Rewards, γ: Discount factor
- **Value Function:** `V(s) = E[Σ γ^t R_t | s_0=s]`
- **Q-Function:** `Q(s,a) = E[Σ γ^t R_t | s_0=s, a_0=a]`
- **Bellman Equation:** `V(s) = max_a [R(s,a) + γ Σ P(s'|s,a)V(s')]`

### Key Algorithms
- **Q-Learning:** Off-policy, `Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]`
- **SARSA:** On-policy, uses actual next action
- **DQN:** Q-learning with neural networks + experience replay
- **PPO:** Policy gradient with clipped objective (most popular)
- **A3C:** Asynchronous advantage actor-critic

---

## 💬 NLP Essentials

### Tokenization
- **Word-level:** Split by words (large vocab)
- **Subword:** BPE, WordPiece, SentencePiece (balanced)
- **Character-level:** Individual characters (small vocab, long sequences)

### Embeddings
- **Word2Vec:** CBOW (context→word), Skip-gram (word→context)
- **GloVe:** Global word co-occurrence statistics
- **Contextual:** BERT, GPT (different embedding per context)

### Transformer Models
- **Encoder-only:** BERT (bidirectional, classification/NER)
- **Decoder-only:** GPT (autoregressive, generation)
- **Encoder-Decoder:** T5, BART (translation, summarization)

### Attention Formula
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
- Q: Query, K: Key, V: Value
- √d_k: Scaling factor (prevents small gradients)

---

## 📐 Must-Know Formulas

### Linear Regression
- **Normal Equation:** `β = (X^T X)^{-1} X^T y`
- **Gradient:** `∇L = (1/m) X^T (Xβ - y)`

### Logistic Regression
- **Sigmoid:** `σ(z) = 1/(1+e^{-z})`
- **Loss:** `L = -[y log(ŷ) + (1-y)log(1-ŷ)]`
- **Gradient:** `∇L = (1/m) X^T (σ(Xβ) - y)`

### Softmax
```
softmax(z_i) = e^{z_i} / Σ_j e^{z_j}
```

### Backpropagation (Chain Rule)
```
∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w
```

### Information Theory
- **Entropy:** `H(X) = -Σ p(x) log p(x)`
- **KL Divergence:** `D_KL(P||Q) = Σ p(x) log(p(x)/q(x))`
- **Cross-Entropy:** `H(P,Q) = -Σ p(x) log q(x)`

---

## 🎯 Common Interview Questions

### "Explain overfitting and how to prevent it"
**Overfitting:** Model memorizes training data, poor generalization
**Solutions:**
1. More training data
2. Regularization (L1/L2, dropout)
3. Cross-validation
4. Simpler model
5. Early stopping
6. Data augmentation

### "Batch Norm vs Layer Norm"
- **Batch Norm:** Normalize across batch dimension (CNNs)
- **Layer Norm:** Normalize across feature dimension (Transformers)
- **Why Layer Norm for Transformers?** Works with variable sequence lengths, no batch dependency

### "Why ReLU over Sigmoid?"
1. No vanishing gradient (gradient is 0 or 1)
2. Faster computation
3. Sparse activation (some neurons = 0)
4. Better empirical performance

### "Explain Attention mechanism"
Allows model to focus on relevant parts of input by computing weighted sum based on similarity (Query-Key matching).

### "Difference between Bagging and Boosting"
- **Bagging:** Parallel, independent models, reduces variance (Random Forest)
- **Boosting:** Sequential, each model corrects previous, reduces bias (XGBoost, AdaBoost)

---

## 🔍 Debugging ML Models

### Model Not Learning
1. Check learning rate (too high/low)
2. Verify loss function
3. Check data preprocessing
4. Gradient flow (vanishing/exploding)
5. Weight initialization

### High Training Error
1. Model too simple (increase capacity)
2. Learning rate too high
3. Bad features
4. Data quality issues

### High Validation Error (Low Training Error)
1. Overfitting - add regularization
2. Train/val distribution mismatch
3. Need more training data

---

## 💡 Quick Tips

### Data Preprocessing
- **Always** normalize/standardize features
- **Handle** missing values (imputation, removal)
- **Encode** categorical variables (one-hot, label encoding)
- **Check** for data leakage

### Hyperparameter Tuning Priority
1. Learning rate (most important)
2. Batch size
3. Number of layers/units
4. Regularization strength
5. Optimizer choice

### When to Use What
- **Linear Regression:** Linear relationships, interpretability
- **Random Forest:** Tabular data, feature importance
- **XGBoost:** Structured data, competitions
- **Neural Networks:** Images, text, complex patterns
- **Transformers:** NLP, sequences, attention needed

---

## 📚 System Design Patterns

### ML System Components
1. **Data Pipeline:** Collection, cleaning, feature engineering
2. **Training Pipeline:** Model training, validation, versioning
3. **Serving Pipeline:** Inference, monitoring, A/B testing
4. **Monitoring:** Data drift, model performance, latency

### Scaling Strategies
- **Data Parallelism:** Split data across GPUs
- **Model Parallelism:** Split model across GPUs
- **Batch Inference:** Process multiple requests together
- **Caching:** Cache frequent predictions

---

**Remember:** Understanding > Memorization. Know the "why" behind each concept!
