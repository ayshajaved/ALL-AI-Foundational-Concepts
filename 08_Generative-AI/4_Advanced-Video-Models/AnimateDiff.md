# AnimateDiff

> **Breathing Life into Static Models** - Motion Modules

---

## 🧟 The Concept

We have thousands of amazing fine-tuned Stable Diffusion models (CivitAI) for Anime, Realistic, Oil Painting, etc.
We want to animate them **without re-training** them.

**AnimateDiff Idea:**
Train a separate **Motion Module** that can be plugged into *any* frozen SD model.

---

## 🏗️ The Motion Module

1.  **Freeze SD:** Keep the weights of the base T2I model fixed.
2.  **Insert Layers:** Inject **Temporal Attention** layers between the spatial layers of the U-Net.
3.  **Train:** Train *only* the Temporal layers on large video datasets (WebVid-10M).
    - The model learns "How things move" generally.
    - It relies on the frozen SD weights for "How things look".

---

## 🚀 The Workflow

1.  Download a specific SD checkpoint (e.g., "DreamShaper").
2.  Download the AnimateDiff Motion Module.
3.  Combine:
    - DreamShaper handles the visuals.
    - AnimateDiff handles the movement.
4.  **Result:** A high-quality video in the style of DreamShaper.

---

## 🎓 Interview Focus

1.  **Why is this "Plug-and-Play"?**
    - Because the Motion Module was trained to be compatible with the standard SD architecture. It doesn't modify the spatial weights, so the artistic style is preserved.

2.  **Context Window Limit?**
    - AnimateDiff usually generates 16 frames.
    - **Sliding Window:** To generate longer videos, we generate overlapping chunks (Frames 1-16, 8-24, 16-32) and blend them.

---

**AnimateDiff: The universal adapter for video generation!**
