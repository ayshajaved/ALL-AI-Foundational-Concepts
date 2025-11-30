# HuggingFace Agents

> **Open Source Power** - Transformers as Tools

---

## 🤖 The Concept

HuggingFace Agents (`transformers.agents`) allows an LLM to control the entire HF Hub.
- **The Brain:** An LLM (OpenAI, StarCoder, Llama).
- **The Tools:** 100,000+ models on the Hub (Image Generation, Text Classification, VQA).

---

## 🛠️ Tool Selection

The agent doesn't just use a calculator. It uses **Models**.
- User: "Read this image and tell me what's in it."
- Agent:
    1.  Selects `ImageCaptioningTool`.
    2.  Loads a model (e.g., `Salesforce/blip-image-captioning-base`).
    3.  Runs inference.
    4.  Returns text.

---

## 💻 Implementation

```python
from transformers import HfAgent

# 1. Define Agent (uses a remote inference endpoint or local model)
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")

# 2. Run
agent.run("Generate an image of a cat and then caption it.")

# Under the hood, it generates code:
# image = image_generator("a cat")
# caption = image_captioner(image)
# print(caption)
```

---

## 🎓 Interview Focus

1.  **Code Generation vs JSON?**
    - HF Agents typically generate **Python Code** to solve the task, rather than JSON actions.
    - This is more powerful (loops, variables) but riskier (executing arbitrary code).

2.  **Local Agents?**
    - You can run the LLM and the Tools entirely locally using `transformers`.
    - Great for privacy and offline usage.

---

**HF Agents: Unlocking the power of the Hub!**
