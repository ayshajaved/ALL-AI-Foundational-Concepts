# Model Optimization for CV

> **Deploying on Edge Devices** - Pruning, Quantization, and TensorRT

---

## ✂️ Pruning

Removing unnecessary connections (weights) from the neural network.
- **Unstructured Pruning:** Set individual weights to 0. Sparse matrix. Hard to accelerate on GPUs.
- **Structured Pruning:** Remove entire filters/channels. Smaller dense matrix. Easy to accelerate.

**Result:** 50% smaller model with <1% accuracy drop.

---

## 📉 Quantization

Converting weights from Float32 (32-bit) to Int8 (8-bit).
- **Size:** 4x smaller.
- **Speed:** 2-4x faster (Int8 math is cheap).

**Post-Training Quantization (PTQ):**
Calibrate on a small dataset to find the min/max range of activations.

```python
import torch.quantization

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Run calibration loop...
torch.quantization.convert(model, inplace=True)
```

---

## 🏎️ TensorRT (NVIDIA)

The ultimate optimizer for NVIDIA GPUs.
1.  **Layer Fusion:** Merges Conv + Bias + ReLU into a single kernel.
2.  **Precision Calibration:** Auto-selects FP32, FP16, or Int8 per layer.
3.  **Kernel Auto-Tuning:** Picks the fastest algorithm for the specific GPU hardware.

**Workflow:** PyTorch $\to$ ONNX $\to$ TensorRT Engine.

---

## 📱 Mobile Optimization (TFLite)

For Android/iOS.
Uses **Delegate** (GPU or NPU) to accelerate inference.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

---

## 🎓 Interview Focus

1.  **Why does Quantization work?**
    - Neural networks are robust to noise. Reducing precision is just adding a small amount of "quantization noise".

2.  **FP16 vs Int8?**
    - **FP16:** Half-precision float. Easy to use (just `.half()`). 2x speedup.
    - **Int8:** Integer. Harder (needs calibration). 4x speedup.

3.  **What is Knowledge Distillation?**
    - Training a small "Student" model to mimic the output of a large "Teacher" model.
    - $L = \alpha L_{CE}(y, \text{soft\_targets}) + \beta L_{CE}(y, \text{hard\_targets})$.

---

**Optimization: Making AI run on a toaster!**
