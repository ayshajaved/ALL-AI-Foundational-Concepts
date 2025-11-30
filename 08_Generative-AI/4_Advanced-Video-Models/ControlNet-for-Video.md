# ControlNet for Video

> **Directing the Movie** - Precise Control over Generation

---

## 🎮 The Challenge

Text prompts ("A man dancing") are vague.
We want precise control: "A man dancing *like this*" (providing a skeleton video).

---

## 🕸️ ControlNet Recap

A copy of the encoder weights (locked) + a trainable copy (unlocked) connected via "Zero Convolutions".
Allows adding conditions (Canny edge, Pose, Depth) to a pre-trained Diffusion model.

---

## 🎞️ Video ControlNet

Applying ControlNet frame-by-frame causes flickering.
**Solutions:**
1.  **Sparse Control:** Apply ControlNet only to keyframes, use Optical Flow to propagate.
2.  **3D ControlNet:** Extend the ControlNet architecture to handle 3D volumes (Temporal layers in the ControlNet copy).

**CoDeF (Content Deformation Field):**
- Decomposes video into a **Canonical Content Field** (Static texture) and a **Temporal Deformation Field** (Motion).
- Apply ControlNet to the Canonical Content (guarantees consistency).

---

## 💻 Workflow: Video-to-Video

1.  **Input:** A video of you dancing.
2.  **Process:** Extract OpenPose skeleton for every frame.
3.  **Generate:** Feed OpenPose frames + Prompt ("Iron Man dancing") to Video ControlNet.
4.  **Result:** Iron Man performing your exact moves.

---

## 🎓 Interview Focus

1.  **What is "Zero Convolution"?**
    - A $1 \times 1$ convolution initialized with zeros.
    - At the start of training, the ControlNet output is 0, so the model behaves exactly like the original pre-trained model. This prevents "catastrophic forgetting" and ensures stable fine-tuning.

2.  **Why is Pose estimation crucial for Video Gen?**
    - It separates **Motion** (Skeleton) from **Appearance** (Pixels). This disentanglement allows for high-quality Video-to-Video translation.

---

**ControlNet: You are the director!**
