# 3D Computer Vision

> **Beyond 2D Pixels** - NeRFs and Gaussian Splatting

---

## 🧊 The Challenge

Standard CV works on 2D images. The world is 3D.
**Goal:** Reconstruct a 3D scene from a set of 2D photos.

---

## 🔦 NeRF (Neural Radiance Fields) - 2020

**Idea:** Represent a scene as a function $F(x, y, z, \theta, \phi) \to (RGB, \sigma)$.
- Input: 3D coordinate + Viewing Direction.
- Output: Color + Density (Opacity).

**Volumetric Rendering:**
To render a pixel, shoot a ray into the scene. Sample points along the ray. Sum up the color/density.
Train the network to minimize the difference between the rendered pixel and the real photo.

**Pros:** Photorealistic.
**Cons:** Extremely slow training and rendering (querying a NN millions of times).

---

## 💥 Gaussian Splatting - 2023

**Idea:** Represent the scene as a cloud of **3D Gaussians** (ellipsoids).
Each Gaussian has: Position, Covariance (Shape), Color, Opacity.

**Rasterization:**
Project these 3D blobs onto the 2D screen (Splatting).
Differentiable! We can move/resize the blobs to match the photos.

**Pros:** Real-time rendering (100+ FPS). Fast training. SOTA quality.

---

## ☁️ Point Clouds & Meshes

- **Point Cloud:** List of $(x, y, z)$ points. (LiDAR output). Sparse.
- **Mesh:** Vertices + Faces (Triangles). Standard for Graphics.
- **Voxel Grid:** 3D pixels (Minecraft style). Memory intensive.

---

## 🎓 Interview Focus

1.  **Structure from Motion (SfM)?**
    - Traditional algorithm (COLMAP) to estimate camera poses and a sparse point cloud from a video. Prerequisite for NeRF/Splatting.

2.  **Why is Gaussian Splatting faster than NeRF?**
    - NeRF requires a neural network forward pass for every sample point on every ray.
    - Splatting is a rasterization process (sorting and projecting), which GPUs are designed to do instantly.

3.  **Monocular Depth Estimation?**
    - Predicting depth from a *single* image using a CNN/Transformer (MiDaS, Depth Anything). "This pixel is far, this is near."

---

**3D Vision: The Matrix is real!**
