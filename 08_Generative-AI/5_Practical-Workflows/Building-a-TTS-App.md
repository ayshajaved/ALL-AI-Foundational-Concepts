# Building a TTS App

> **From Script to Speech** - Creating a Gradio Interface

---

## 📱 The Goal

Build a web interface where users can:
1.  Type text.
2.  Select a Speaker.
3.  Adjust Speed/Pitch.
4.  Download the generated Audio.

---

## 🛠️ The Stack

- **Engine:** `Coqui TTS` (Open-source, supports VITS, YourTTS).
- **UI:** `Gradio` (Fastest way to demo ML models).

---

## 💻 Implementation

```python
import gradio as gr
from TTS.api import TTS

# 1. Load Model (YourTTS supports Zero-Shot Cloning)
# GPU is recommended
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)

def generate_speech(text, speaker_wav=None):
    if speaker_wav is None:
        # Use default speaker if no clone target provided
        output_path = "output.wav"
        tts.tts_to_file(text=text, file_path=output_path)
    else:
        # Voice Cloning
        tts.tts_with_vc_to_file(
            text=text,
            speaker_wav=speaker_wav,
            file_path="output.wav"
        )
    return "output.wav"

# 2. Build UI
with gr.Blocks() as demo:
    gr.Markdown("# 🗣️ AI Voice Cloner")
    
    with gr.Row():
        input_text = gr.Textbox(label="Enter Text")
        ref_audio = gr.Audio(label="Reference Voice (Optional)", type="filepath")
        
    btn = gr.Button("Generate")
    output_audio = gr.Audio(label="Generated Speech")
    
    btn.click(generate_speech, inputs=[input_text, ref_audio], outputs=output_audio)

# 3. Launch
demo.launch()
```

---

## 🎓 Interview Focus

1.  **Latency Optimization?**
    - **Streaming:** Don't wait for the full audio. Stream chunks as they are generated.
    - **ONNX:** Export the PyTorch model to ONNX Runtime for faster CPU inference.

2.  **Handling Long Text?**
    - TTS models have a limit (e.g., 512 chars).
    - Split text into sentences, generate audio for each, and concatenate them with a small silence gap.

---

**TTS App: Your personal narrator!**
