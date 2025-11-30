# Temporal Consistency

> **The Flickering Problem** - Making Video Smooth

---

## 🕯️ The Problem

If you apply an Image GenAI (like Style Transfer or Stable Diffusion) to a video frame-by-frame:
- Frame 1: "A cat" (Brown).
- Frame 2: "A cat" (slightly darker Brown).
- **Result:** The video flickers uncontrollably. The texture "boils".
- **Reason:** The model is independent per frame. It doesn't know Frame 2 should look like Frame 1.

---

## 🌊 Optical Flow

To enforce consistency, we need to know how pixels move.
**Optical Flow ($u, v$):** Vector field describing motion between $I_t$ and $I_{t+1}$.
- Pixel $(x, y)$ at time $t$ moves to $(x+u, y+v)$ at time $t+1$.

**Warping:**
We can take the generated output $O_t$ and "warp" it using Optical Flow to guess what $O_{t+1}$ should look like.
$$ \hat{O}_{t+1} = \text{Warp}(O_t, \text{Flow}_{t \to t+1}) $$

---

## 🛡️ Consistency Loss

When training video models, we add a loss term:
$$ L_{temporal} = || O_{t+1} - \text{Warp}(O_t, \text{Flow}) ||^2 $$
"The output at $t+1$ should match the warped output from $t$, except in occluded regions."

---

## 🧱 Blind Video Consistency (E.g., Blind Video Temporal Consistency)

What if we don't have Optical Flow?
- Train a post-processing network to take a flickering video and smooth it.
- Uses short-term temporal loss and long-term consistency checks.

---

## 🎓 Interview Focus

1.  **What is the "Occlusion Problem" in Optical Flow?**
    - If an object moves behind a tree, its pixels disappear. We cannot warp them.
    - We need an **Occlusion Mask** to tell the loss function to ignore those regions.

2.  **How does Stable Video Diffusion handle this?**
    - It uses **Temporal Attention**. The Self-Attention mechanism looks at pixels in the same frame *and* pixels in previous/future frames.

---

**Temporal Consistency: The difference between a slideshow and a movie!**
