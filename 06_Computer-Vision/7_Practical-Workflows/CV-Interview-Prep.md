# CV Interview Prep: Top 50 Q&A

> **Mastering the Vision Interview** - From Kernels to Transformers

---

## 🟢 Beginner (Concepts)

1.  **What is a Convolution?** Element-wise multiplication and sum of a kernel sliding over an image. Feature extraction.
2.  **Why Max Pooling?** Reduces spatial dimensions, provides translation invariance, reduces computation.
3.  **RGB vs BGR?** OpenCV uses BGR. Matplotlib uses RGB. Always convert `cv2.cvtColor`.
4.  **What is Data Augmentation?** Flipping, rotating, cropping images to prevent overfitting.
5.  **Explain IoU.** Intersection over Union. Overlap metric for bounding boxes.

---

## 🟡 Intermediate (Architectures)

6.  **Why ResNet?** Skip connections solve vanishing gradients, allowing deep networks.
7.  **YOLO vs Faster R-CNN?** One-stage (Speed) vs Two-stage (Accuracy).
8.  **What is a $1 \times 1$ Convolution?** Channel-wise pooling. Reduces dimensionality (Bottleneck) or increases non-linearity.
9.  **Explain Semantic vs Instance Segmentation.** Classifying pixels vs separating objects.
10. **What is Transfer Learning?** Freezing early layers (feature extractors) and retraining the head.

---

## 🔴 Advanced (Modern CV)

11. **ViT vs CNN Inductive Bias?** CNNs assume locality/translation invariance. ViTs learn global relationships but need more data.
12. **How does Stable Diffusion work?** Denoising in Latent Space (VAE) conditioned on CLIP text embeddings.
13. **What is Focal Loss?** Down-weights easy examples to handle class imbalance in detection.
14. **Explain RoI Align.** Bilinear interpolation to extract features without quantization error (Mask R-CNN).
15. **What is Contrastive Learning (SimCLR)?** Pulling augmentations of the same image close, pushing different images apart.
16. **NeRF vs Gaussian Splatting?** Neural Volumetric Rendering (Slow) vs Differentiable Rasterization (Fast).
17. **What is the Receptive Field?** The region of the input image that affects a specific neuron.
18. **Explain Depthwise Separable Conv.** Splits Conv into Spatial (Depthwise) + Channel (Pointwise). Efficient.
19. **What is Mode Collapse in GANs?** Generator produces limited variety.
20. **Why use LayerNorm in ViT?** Stabilizes training for sequence data (patches), independent of batch size.

---

## 🧠 System Design Scenarios

**Q: Design a Self-Driving Car Perception System.**
- **Sensors:** Camera + LiDAR + Radar.
- **Tasks:**
    - Lane Detection (Segmentation - UNet).
    - Object Detection (YOLO/DETR - Cars/Pedestrians).
    - Traffic Light Classification (CNN).
- **Fusion:** Kalman Filter to merge Sensor data.
- **Optimization:** TensorRT for low latency.

**Q: Design an Instagram Filter App.**
- **Face Detection:** BlazeFace (MediaPipe).
- **Landmarks:** 468 points mesh.
- **Rendering:** OpenGL/WebGL overlay.
- **Style Transfer:** Fast Style Transfer (Feed-forward CNN).

---

**You see the world clearly now. Go build the future!**
