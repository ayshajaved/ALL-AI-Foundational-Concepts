# Production NLP

> **Speed & Scale** - vLLM, ONNX, and Quantization

---

## 🚀 Serving LLMs: vLLM

Standard HuggingFace `generate()` is too slow for production (low throughput).
**vLLM** is the state-of-the-art serving engine.

**Key Features:**
1.  **PagedAttention:** Manages KV Cache memory like an OS manages RAM (paging). Zero waste.
2.  **Continuous Batching:** Doesn't wait for all sequences in a batch to finish. Inserts new requests as soon as one finishes.

**Result:** 24x higher throughput than standard HF.

```bash
pip install vllm
python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-hf
```

---

## 🏎️ ONNX Runtime (Open Neural Network Exchange)

Standard for **Encoder Models** (BERT, ResNet).
Converts PyTorch graph to a static, optimized graph.
Runs on CPU, GPU, Edge devices.

```python
import torch.onnx

# Export
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)

# Run
import onnxruntime
session = onnxruntime.InferenceSession("model.onnx")
result = session.run(None, {"input": numpy_data})
```

---

## 📉 Quantization (GPTQ / AWQ)

Running 70B models on consumer hardware.
**GPTQ (Post-Training Quantization):**
- Quantizes weights to 4-bit.
- Calibrates using a small dataset to minimize error.
- **ExLlamaV2:** The fastest kernel for running GPTQ models.

---

## 🎓 Interview Focus

1.  **Throughput vs Latency?**
    - **Latency:** Time for 1 request (Time to First Token). Crucial for Chatbots.
    - **Throughput:** Requests per second. Crucial for Batch Processing.
    - vLLM optimizes Throughput significantly without hurting Latency.

2.  **Why PagedAttention?**
    - In standard attention, we pre-allocate memory for the maximum sequence length. If a user only types 10 words, 90% of memory is wasted. PagedAttention allocates blocks dynamically.

3.  **ONNX vs TorchScript?**
    - ONNX is cross-platform (deploy to C++, C#, JS). TorchScript is PyTorch-specific but easier to export.

---

**Production: Where research meets reality!**
