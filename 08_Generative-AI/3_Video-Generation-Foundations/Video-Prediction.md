# Video Prediction

> **Predicting the Future** - Unsupervised Learning

---

## 🔮 The Task

Given frames $x_1, \dots, x_t$, predict $x_{t+1}, \dots, x_{t+k}$.
This is a form of **Self-Supervised Learning**. The labels are free (just wait for the next frame).

---

## 🏗️ ConvLSTM

Standard LSTMs use fully connected layers (1D vectors). This destroys spatial info.
**ConvLSTM** replaces matrix multiplication with **Convolution**.
$$ i_t = \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + b_i) $$
- The hidden state $H_t$ is a 3D tensor (Height, Width, Channels).
- Perfect for spatiotemporal prediction (Weather forecasting, Video).

---

## 🌫️ The Blurriness Problem (L2 Loss)

If you train a model to minimize MSE ($L2$ loss) on future frames:
- **Scenario:** The person might move Left OR Right.
- **Model:** "I'm not sure, so I'll predict the *average* of Left and Right."
- **Result:** A blurry ghost image.

**Solution:**
- **Adversarial Loss (GANs):** Force sharpness.
- **Latent Prediction:** Predict the future in *latent space*, not pixel space.

---

## 🎓 Interview Focus

1.  **Difference between Video Generation and Prediction?**
    - **Generation:** Create video from scratch (noise/text).
    - **Prediction:** Continuation of an existing video.

2.  **Why is Video Prediction useful for RL?**
    - **World Models:** If an agent can predict the result of its actions (next frame), it can plan in its head without acting.

---

**Video Prediction: The key to physical understanding!**
