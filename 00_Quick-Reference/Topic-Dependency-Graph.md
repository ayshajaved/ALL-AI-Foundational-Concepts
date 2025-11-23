# Topic Dependency Graph

> **Visual guide to AI concept relationships** - Understand prerequisites and learning order

---

## 🗺️ Complete Dependency Map

```
MATHEMATICAL FOUNDATIONS (Start Here)
│
├─ Linear Algebra ────────────────────────────┐
│  ├─ Vectors & Matrices                      │
│  ├─ Matrix Operations                       │
│  ├─ Eigenvalues & Eigenvectors             │
│  └─ SVD                                     │
│                                             │
├─ Calculus ──────────────────────────────────┤
│  ├─ Derivatives & Gradients                │
│  ├─ Chain Rule                             │
│  └─ Optimization Fundamentals              │
│                                             │
├─ Probability & Statistics ──────────────────┤
│  ├─ Probability Theory                     │
│  ├─ Distributions                          │
│  ├─ Bayesian Statistics                    │
│  └─ Information Theory                     │
│                                             │
└─ Optimization ──────────────────────────────┘
   ├─ Convex Optimization
   ├─ Gradient Descent
   └─ Constrained Optimization
                │
                ▼
┌───────────────────────────────────────────────┐
│         CLASSICAL AI & ML FUNDAMENTALS         │
└───────────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
   CLASSICAL AI    ML FUNDAMENTALS
   ├─ Search       ├─ Learning Paradigms
   ├─ Logic        ├─ Bias-Variance
   ├─ Planning     ├─ PAC Learning
   └─ Reasoning    └─ VC Dimension
        │               │
        │               ▼
        │      ┌────────────────┐
        │      │  MACHINE       │
        │      │  LEARNING      │
        │      └────────────────┘
        │               │
        │       ┌───────┼───────┐
        │       ▼       ▼       ▼
        │   SUPERVISED UNSUPERVISED ENSEMBLE
        │   ├─Regression ├─Clustering ├─Bagging
        │   └─Classification └─Dim.Red. └─Boosting
        │               │
        │               ▼
        │      ┌────────────────┐
        │      │ ADVANCED ML    │
        │      └────────────────┘
        │               │
        │       ┌───────┼───────┬───────┐
        │       ▼       ▼       ▼       ▼
        │   LEARNING BAYESIAN ACTIVE TRANSFER
        │   THEORY  METHODS LEARNING LEARNING
        │               │
        └───────────────┼───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │      DEEP LEARNING            │
        └───────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    NN BASICS      ARCHITECTURES   TRAINING
    ├─Perceptron   ├─CNNs         ├─Backprop
    ├─MLP          ├─RNNs         ├─Optimizers
    ├─Activation   ├─Transformers ├─Regularization
    └─Loss Fns     └─GNNs         └─Fine-tuning
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    DL THEORY    SPECIALIZED DL   CONTINUAL
    ├─NTK        ├─Bayesian DL   LEARNING
    ├─Lottery    ├─Compression
    └─Double     └─Interpretability
     Descent
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌─────────────────┐           ┌─────────────────┐
│  DOMAIN         │           │  ADVANCED       │
│  APPLICATIONS   │           │  TOPICS         │
└─────────────────┘           └─────────────────┘
        │                               │
    ┌───┼───┬───┬───┬───┐          ┌───┼───┬───┐
    ▼   ▼   ▼   ▼   ▼   ▼          ▼   ▼   ▼   ▼
   NLP  CV  RL GEN MULTI SPEECH  CAUSAL AI  MLOps
                AI  MODAL         HARDWARE
```

---

## 📚 Detailed Dependency Chains

### Chain 1: NLP & LLMs
```
Math Foundations
    ↓
Probability & Statistics
    ↓
ML Fundamentals
    ↓
Deep Learning Basics
    ↓
RNNs & Sequence Models
    ↓
Attention Mechanisms
    ↓
Transformers
    ↓
BERT/GPT Architectures
    ↓
Large Language Models
    ↓
Fine-Tuning & PEFT
    ↓
Prompt Engineering & RAG
```

### Chain 2: Computer Vision
```
Math Foundations
    ↓
Linear Algebra (heavy)
    ↓
ML Fundamentals
    ↓
Deep Learning Basics
    ↓
CNNs
    ↓
CNN Architectures (ResNet, etc.)
    ↓
Object Detection
    ↓
Segmentation
    ↓
Vision Transformers
    ↓
Generative Vision (GANs, Diffusion)
```

### Chain 3: Reinforcement Learning
```
Math Foundations
    ↓
Probability Theory
    ↓
Optimization
    ↓
MDP Framework
    ↓
Dynamic Programming
    ↓
Monte Carlo Methods
    ↓
Temporal Difference Learning
    ↓
Deep RL (DQN)
    ↓
Policy Gradients
    ↓
Actor-Critic Methods
    ↓
Model-Based RL
```

### Chain 4: Generative AI
```
Math Foundations
    ↓
Probability & Statistics
    ↓
Deep Learning Basics
    ↓
Autoencoders
    ↓
VAEs
    ↓
GANs
    ↓
Diffusion Models
    ↓
Text/Image/Audio Generation
```

### Chain 5: Causality
```
Probability & Statistics
    ↓
Bayesian Statistics
    ↓
Graph Theory
    ↓
Causal Graphs & DAGs
    ↓
Do-Calculus
    ↓
Causal Inference
    ↓
Counterfactual Reasoning
    ↓
Causal ML
```

---

## 🎯 Learning Order by Goal

### Goal: ML Engineer
```
1. Math Foundations (all)
2. Classical AI (overview)
3. ML Fundamentals
4. Supervised Learning
5. Unsupervised Learning
6. Deep Learning Basics
7. Choose Domain (NLP OR CV)
8. MLOps & Production
9. Interview Prep
```

### Goal: Research Scientist
```
1. Math Foundations (deep)
2. Learning Theory
3. ML Fundamentals
4. Bayesian Methods
5. Deep Learning + Theory
6. Neural Network Theory
7. Research Frontiers
8. Specialization (Causality/Multimodal/etc.)
9. Paper Implementation
```

### Goal: NLP Specialist
```
1. Math Foundations
2. ML Fundamentals
3. Deep Learning Basics
4. RNNs & Sequence Models
5. Attention & Transformers
6. NLP Tasks
7. LLMs
8. Fine-Tuning & PEFT
9. RAG & Prompt Engineering
10. Speech & Audio (optional)
```

### Goal: Computer Vision Specialist
```
1. Math Foundations (Linear Algebra focus)
2. ML Fundamentals
3. Deep Learning Basics
4. CNNs
5. CNN Architectures
6. Object Detection
7. Segmentation
8. Vision Transformers
9. Generative Vision
10. 3D Vision (optional)
```

---

## 🔗 Cross-Domain Dependencies

### Multimodal AI Requires:
- NLP fundamentals
- Computer Vision fundamentals
- Transformers
- Attention mechanisms
- Cross-modal learning

### AI Agents Require:
- ML fundamentals
- NLP (for language agents)
- Reinforcement Learning (for decision-making)
- Planning & Reasoning
- Tool use & function calling

### Generative AI Requires:
- Deep Learning basics
- Probability theory
- Domain knowledge (NLP for text, CV for images)
- Sampling methods

### MLOps Requires:
- ML fundamentals
- Deep Learning basics
- Software engineering
- Distributed systems
- Cloud platforms

---

## ⚠️ Common Prerequisite Mistakes

### ❌ Skipping Math
**Problem:** Jumping to deep learning without math foundations
**Result:** Superficial understanding, can't debug or innovate
**Fix:** Master linear algebra, calculus, probability first

### ❌ Ignoring Classical ML
**Problem:** Starting with deep learning directly
**Result:** Missing fundamental concepts
**Fix:** Learn classical ML algorithms first

### ❌ No Theory
**Problem:** Only learning practical implementation
**Result:** Can't understand why things work
**Fix:** Balance theory and practice

### ❌ Domain Jumping
**Problem:** Trying to learn NLP, CV, RL simultaneously
**Result:** Shallow knowledge in all areas
**Fix:** Master one domain before moving to next

---

## ✅ Recommended Learning Sequences

### Sequence 1: Foundation-First (Recommended)
```
Math (2 months) → Classical AI (2 weeks) → ML (2 months) → 
DL (2 months) → Domain (2 months) → Advanced (2 months)
```

### Sequence 2: Project-Driven
```
Math Basics (1 month) → ML Basics (1 month) → 
Simple Project → DL Basics (1 month) → 
Domain Project → Advanced Topics → 
Complex Project
```

### Sequence 3: Interview-Focused
```
Quick Math Review (2 weeks) → ML All (1 month) → 
DL All (1 month) → Domain Focus (1 month) → 
Coding Practice (2 weeks) → System Design (2 weeks)
```

---

## 📊 Prerequisite Matrix

| Topic | Requires | Optional But Helpful |
|-------|----------|---------------------|
| ML Fundamentals | Linear Algebra, Calculus, Probability | Optimization |
| Deep Learning | ML Fundamentals, Calculus | Learning Theory |
| CNNs | Deep Learning Basics | Linear Algebra (deep) |
| RNNs | Deep Learning Basics | Sequence Models |
| Transformers | Deep Learning, Attention | RNNs |
| LLMs | Transformers, NLP Basics | Scaling Laws |
| GANs | Deep Learning, Probability | Game Theory |
| Diffusion | Probability, Deep Learning | Stochastic Processes |
| Reinforcement Learning | Probability, Optimization | Game Theory |
| Causal Inference | Probability, Graph Theory | Bayesian Statistics |
| MLOps | ML/DL Fundamentals | Software Engineering |

---

## 🎓 Self-Assessment Checklist

Before moving to next level, ensure you can:

### ✅ After Math Foundations
- [ ] Multiply matrices by hand
- [ ] Compute gradients
- [ ] Explain Bayes' theorem
- [ ] Derive gradient descent update

### ✅ After ML Fundamentals
- [ ] Explain bias-variance tradeoff
- [ ] Implement linear regression from scratch
- [ ] Choose appropriate algorithm for problem
- [ ] Evaluate model performance

### ✅ After Deep Learning Basics
- [ ] Implement backpropagation
- [ ] Explain vanishing gradient
- [ ] Choose activation functions
- [ ] Debug neural network training

### ✅ After Domain Specialization
- [ ] Build end-to-end domain project
- [ ] Explain SOTA architectures
- [ ] Fine-tune pre-trained models
- [ ] Deploy domain model

---

**Use this graph to plan your learning journey and ensure you have proper prerequisites!**
