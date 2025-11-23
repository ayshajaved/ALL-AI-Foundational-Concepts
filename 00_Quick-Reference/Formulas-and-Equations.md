# Essential AI Formulas & Equations

> **Mathematical reference for AI/ML** - All critical formulas organized by topic

---

## 📊 Statistics & Probability

### Basic Statistics
```
Mean: μ = (1/n) Σ x_i
Variance: σ² = (1/n) Σ (x_i - μ)²
Standard Deviation: σ = √σ²
Covariance: Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)]
Correlation: ρ = Cov(X,Y) / (σ_X σ_Y)
```

### Probability
```
Bayes' Theorem: P(A|B) = P(B|A)P(A) / P(B)
Chain Rule: P(A,B,C) = P(A|B,C)P(B|C)P(C)
Law of Total Probability: P(A) = Σ P(A|B_i)P(B_i)
```

### Distributions
```
Gaussian/Normal: p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
Bernoulli: P(X=1) = p, P(X=0) = 1-p
Binomial: P(X=k) = C(n,k) p^k (1-p)^(n-k)
```

---

## 🧮 Linear Algebra

### Matrix Operations
```
Matrix Multiplication: (AB)_ij = Σ_k A_ik B_kj
Transpose: (A^T)_ij = A_ji
Inverse: AA^{-1} = I
Determinant: det(A) for square matrix A
Trace: tr(A) = Σ A_ii
```

### Eigenvalues & Eigenvectors
```
Av = λv  (v: eigenvector, λ: eigenvalue)
Characteristic Equation: det(A - λI) = 0
```

### SVD (Singular Value Decomposition)
```
A = UΣV^T
U: left singular vectors
Σ: singular values (diagonal)
V: right singular vectors
```

---

## 📐 Calculus & Optimization

### Derivatives
```
Power Rule: d/dx(x^n) = nx^{n-1}
Chain Rule: d/dx f(g(x)) = f'(g(x))g'(x)
Product Rule: d/dx(fg) = f'g + fg'
Quotient Rule: d/dx(f/g) = (f'g - fg')/g²
```

### Gradient
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]^T
```

### Gradient Descent
```
w_{t+1} = w_t - η∇L(w_t)
η: learning rate
∇L: gradient of loss
```

---

## 🧠 Machine Learning

### Linear Regression
```
Model: y = Xβ + ε
Normal Equation: β = (X^T X)^{-1} X^T y
Cost Function (MSE): J(β) = (1/2m) Σ(h_β(x^(i)) - y^(i))²
Gradient: ∇J = (1/m) X^T(Xβ - y)
```

### Logistic Regression
```
Sigmoid: σ(z) = 1/(1 + e^{-z})
Model: P(y=1|x) = σ(w^T x + b)
Cost Function: J = -(1/m) Σ[y log(ŷ) + (1-y)log(1-ŷ)]
Gradient: ∇J = (1/m) X^T(σ(Xw) - y)
```

### Softmax
```
softmax(z_i) = e^{z_i} / Σ_j e^{z_j}
Cross-Entropy Loss: L = -Σ y_i log(ŷ_i)
```

### Regularization
```
L1 (Lasso): J = MSE + λ Σ|w_i|
L2 (Ridge): J = MSE + λ Σw_i²
Elastic Net: J = MSE + λ₁Σ|w_i| + λ₂Σw_i²
```

### SVM
```
Objective: min (1/2)||w||² subject to y_i(w^T x_i + b) ≥ 1
Kernel Trick: K(x,x') = φ(x)^T φ(x')
RBF Kernel: K(x,x') = exp(-γ||x-x'||²)
```

### Decision Trees
```
Gini Impurity: Gini = 1 - Σ p_i²
Entropy: H = -Σ p_i log₂(p_i)
Information Gain: IG = H(parent) - Σ(n_i/n)H(child_i)
```

### K-Means
```
Objective: min Σ_k Σ_{x∈C_k} ||x - μ_k||²
Update: μ_k = (1/|C_k|) Σ_{x∈C_k} x
```

### PCA
```
Maximize: var(Xw) = w^T Σ w subject to ||w||=1
Solution: w = eigenvectors of covariance matrix Σ
```

---

## 🔥 Deep Learning

### Forward Propagation
```
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
a^{[l]} = g^{[l]}(z^{[l]})
```

### Backpropagation
```
dL/dW^{[l]} = dL/da^{[l]} · da^{[l]}/dz^{[l]} · dz^{[l]}/dW^{[l]}
            = dL/da^{[l]} · g'^{[l]}(z^{[l]}) · a^{[l-1]T}
```

### Activation Functions
```
ReLU: f(x) = max(0, x)
       f'(x) = 1 if x>0, else 0

Leaky ReLU: f(x) = max(αx, x)  (α=0.01)

Sigmoid: σ(x) = 1/(1+e^{-x})
         σ'(x) = σ(x)(1-σ(x))

Tanh: tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})
      tanh'(x) = 1 - tanh²(x)

GELU: f(x) = x·Φ(x) where Φ is CDF of N(0,1)
```

### Loss Functions
```
MSE: L = (1/n) Σ(y - ŷ)²
MAE: L = (1/n) Σ|y - ŷ|
Binary Cross-Entropy: L = -[y log(ŷ) + (1-y)log(1-ŷ)]
Categorical Cross-Entropy: L = -Σ y_i log(ŷ_i)
Hinge Loss: L = max(0, 1 - y·ŷ)
```

### Optimizers
```
SGD: w = w - η∇L

Momentum: v = βv + ∇L
          w = w - ηv

RMSprop: s = βs + (1-β)∇L²
         w = w - η∇L/√(s+ε)

Adam: m = β₁m + (1-β₁)∇L
      v = β₂v + (1-β₂)∇L²
      m̂ = m/(1-β₁^t)
      v̂ = v/(1-β₂^t)
      w = w - η·m̂/√(v̂+ε)
```

### Batch Normalization
```
μ_B = (1/m) Σ x_i
σ²_B = (1/m) Σ(x_i - μ_B)²
x̂_i = (x_i - μ_B)/√(σ²_B + ε)
y_i = γx̂_i + β  (learnable γ, β)
```

### Dropout
```
During Training: r ~ Bernoulli(p)
                 ŷ = r * y / p
During Inference: ŷ = y
```

---

## 🖼️ Computer Vision

### Convolution
```
Output Size: O = (W - F + 2P)/S + 1
W: input width
F: filter size
P: padding
S: stride
```

### Receptive Field
```
RF_{l+1} = RF_l + (kernel_size - 1) × stride_product
```

### IoU (Intersection over Union)
```
IoU = Area of Overlap / Area of Union
    = |A ∩ B| / |A ∪ B|
```

---

## 💬 NLP & Transformers

### Attention
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
Q: Query matrix
K: Key matrix
V: Value matrix
d_k: dimension of keys (scaling factor)
```

### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O
head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

### Positional Encoding
```
PE(pos,2i) = sin(pos/10000^{2i/d_model})
PE(pos,2i+1) = cos(pos/10000^{2i/d_model})
```

### Perplexity
```
PPL = exp(-(1/N)Σ log P(w_i|context))
```

### BLEU Score
```
BLEU = BP · exp(Σ w_n log p_n)
BP: brevity penalty
p_n: n-gram precision
```

---

## 🎮 Reinforcement Learning

### Value Functions
```
V^π(s) = E_π[Σ γ^t R_t | S_0=s]
Q^π(s,a) = E_π[Σ γ^t R_t | S_0=s, A_0=a]
```

### Bellman Equations
```
V(s) = max_a [R(s,a) + γ Σ P(s'|s,a)V(s')]
Q(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q(s',a')
```

### Temporal Difference
```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
TD Error: δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
```

### Q-Learning
```
Q(S_t,A_t) ← Q(S_t,A_t) + α[R_{t+1} + γ max_a Q(S_{t+1},a) - Q(S_t,A_t)]
```

### Policy Gradient
```
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) Q^π(s,a)]
REINFORCE: ∇_θ J(θ) ≈ Σ ∇_θ log π_θ(a_t|s_t) G_t
```

### Advantage Function
```
A^π(s,a) = Q^π(s,a) - V^π(s)
```

---

## 📊 Evaluation Metrics

### Classification
```
Accuracy = (TP + TN)/(TP + TN + FP + FN)
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1 = 2·(Precision·Recall)/(Precision + Recall)
F_β = (1+β²)·(Precision·Recall)/(β²·Precision + Recall)
```

### Regression
```
MSE = (1/n)Σ(y_i - ŷ_i)²
RMSE = √MSE
MAE = (1/n)Σ|y_i - ŷ_i|
R² = 1 - (SS_res/SS_tot)
   = 1 - [Σ(y_i-ŷ_i)²/Σ(y_i-ȳ)²]
```

---

## 🔢 Information Theory
```
Entropy: H(X) = -Σ p(x) log p(x)
Joint Entropy: H(X,Y) = -Σ p(x,y) log p(x,y)
Conditional Entropy: H(Y|X) = H(X,Y) - H(X)
Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
KL Divergence: D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
Cross-Entropy: H(P,Q) = -Σ p(x) log q(x)
```

---

## 🎲 Probability Bounds

### Concentration Inequalities
```
Markov: P(X ≥ a) ≤ E[X]/a
Chebyshev: P(|X-μ| ≥ kσ) ≤ 1/k²
Hoeffding: P(|X̄-μ| ≥ ε) ≤ 2exp(-2nε²)
```

---

**Use this as your formula reference during interviews and problem-solving!**
