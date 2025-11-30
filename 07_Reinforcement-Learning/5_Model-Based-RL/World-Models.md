# World Models

> **Dreaming in Latent Space** - Ha & Schmidhuber (2018)

---

## 🌍 The Concept

Humans have a mental model of the world. We can close our eyes and imagine driving a car.
**World Models** train an agent entirely inside a learned "dream" environment.

---

## 🏗️ The 3 Components

1.  **Vision Model (V) - VAE:**
    - Compresses high-dimensional image observation ($64 \times 64 \times 3$) into a small latent vector $z$ ($32$-d).
    - $Observation \to z$.

2.  **Memory Model (M) - RNN (MDN-RNN):**
    - Predicts the next latent state $z_{t+1}$ based on current $z_t$ and action $a_t$.
    - Models the dynamics (physics) of the world.
    - Output: Probability distribution (Mixture Density Network) to handle stochasticity.

3.  **Controller (C) - Linear Model:**
    - Simple linear policy: $a_t = W [z_t, h_t] + b$.
    - Takes current latent $z_t$ and RNN hidden state $h_t$ (context) to output action.

---

## 😴 Training in the Dream

1.  Collect random trajectories from real environment.
2.  Train **V** (VAE) to compress images.
3.  Train **M** (RNN) to predict future $z$.
4.  **Dreaming:**
    - Disconnect the real environment.
    - Use **M** to generate a sequence of $z$ states.
    - Train **C** using Evolution Strategies (ES) inside this hallucinated environment to maximize reward.
5.  Deploy **C** to the real world.

---

## 🎓 Interview Focus

1.  **Why use a VAE?**
    - RL on pixels is hard (high dim). RL on compact latent vectors is easy. VAE handles the compression.

2.  **Why Evolution Strategies (ES) for the Controller?**
    - The Controller is small (linear). ES works well for small parameter spaces and doesn't require backpropagation through time (BPTT) across the long dream sequence.

3.  **Relation to Model-Based RL?**
    - It is the epitome of Model-Based RL. The "Model" is a neural network (VAE+RNN), and the "Planning" is training a policy inside the model.

---

**World Models: Matrix-style training!**
