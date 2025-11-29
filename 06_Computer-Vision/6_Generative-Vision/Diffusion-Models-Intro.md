# Diffusion Models Intro

> **The Physics of Noise** - DDPM and the Forward/Reverse Process

---

## 🌫️ The Intuition

**Forward Process (Diffusion):**
Slowly destroy an image by adding Gaussian noise until it becomes pure random noise.
Image $\to$ Noise.

**Reverse Process (Denoising):**
Train a Neural Network to **predict the noise** added at each step and subtract it.
Noise $\to$ Image.

---

## 📉 DDPM (Denoising Diffusion Probabilistic Models)

### 1. Forward Process ($q$)
Fixed Markov chain. Add noise $\epsilon_t$ at step $t$.
$$ x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon $$

### 2. Reverse Process ($p_\theta$)
Learn a U-Net $\epsilon_\theta(x_t, t)$ to predict the noise $\epsilon$.
$$ x_{t-1} \approx \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)) $$

### 3. Training
Pick a random timestep $t$.
Add noise to image $x_0$ to get $x_t$.
Ask model to predict the noise.
**Loss:** MSE between Added Noise and Predicted Noise.

$$ L = || \epsilon - \epsilon_\theta(x_t, t) ||^2 $$

---

## 🆚 Diffusion vs GANs

| Feature | GANs | Diffusion Models |
| :--- | :--- | :--- |
| **Quality** | High | State-of-the-Art |
| **Diversity** | Low (Mode Collapse) | High (Covers distribution) |
| **Speed** | Fast (1 pass) | Slow (1000 steps) |
| **Training** | Unstable (Minimax) | Stable (MSE Loss) |

---

## 🎓 Interview Focus

1.  **Why are Diffusion Models slower than GANs?**
    - GANs generate an image in one forward pass. Diffusion models need iterative denoising (e.g., 50 or 1000 steps) to generate one image.

2.  **What is the "Reparameterization Trick" in Diffusion?**
    - It allows us to sample $x_t$ at any timestep $t$ directly from $x_0$ without iterating through $t-1, t-2...$ during training.
    - $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

3.  **Why U-Net?**
    - The denoising task requires inputting a noisy image and outputting a noise map of the *same size*. U-Net is perfect for this pixel-to-pixel mapping.

---

**Diffusion: Creating order from chaos!**
