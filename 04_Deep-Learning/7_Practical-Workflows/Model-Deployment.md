# Model Deployment

> **From notebook to production** - ONNX, TorchScript, and Serving

---

## 🌉 ONNX (Open Neural Network Exchange)

Universal format to move models between frameworks (PyTorch $\to$ ONNX $\to$ TensorRT/Edge).

```python
# Export PyTorch to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")

# Run with ONNX Runtime (Fast CPU inference)
import onnxruntime
session = onnxruntime.InferenceSession("model.onnx")
inputs = {session.get_inputs()[0].name: numpy_data}
outs = session.run(None, inputs)
```

---

## 📜 TorchScript

Serialize PyTorch models to run in C++ environments (no Python dependency).

```python
# Tracing (Record operations on dummy input)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model.pt")

# Loading in C++
# torch::jit::load("model.pt");
```

---

## 🚀 TensorFlow Serving

Production system for serving TF models via gRPC/REST.

```bash
# Save model
model.save('saved_model/1')

# Run Docker container
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/saved_model,target=/models/my_model \
  -e MODEL_NAME=my_model -t tensorflow/serving
```

---

## 📱 Edge Deployment (TFLite)

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

**Deployment: Where value is created!**
