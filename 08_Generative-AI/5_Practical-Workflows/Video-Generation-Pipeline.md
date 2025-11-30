# Video Generation Pipeline

> **Text-to-Video with Diffusers** - Using ModelScope/Zeroscope

---

## 🎥 The Pipeline

We will use the HuggingFace `diffusers` library to run a Text-to-Video model.
**Model:** `cerspense/zeroscope_v2_576w` (A fine-tune of ModelScope, watermark-free).

---

## 💻 Implementation

```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# 1. Load Pipeline
pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w", 
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload() # Save VRAM

# 2. Optimize Scheduler (DPM++ is faster)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 3. Generate
prompt = "A cinematic drone shot of a futuristic cyberpunk city, neon lights, rain"
video_frames = pipe(
    prompt, 
    num_inference_steps=40, 
    height=320, 
    width=576, 
    num_frames=24
).frames

# 4. Save
video_path = export_to_video(video_frames, "cyberpunk_city.mp4")
print(f"Saved to {video_path}")
```

---

## 🎞️ Video-to-Video (Style Transfer)

To transform an existing video (e.g., "Turn this video of a dog into a wolf"):
1.  Load the video frames.
2.  Add noise to them (partial diffusion, e.g., start at step 20/50).
3.  Denoise using the text prompt "A wolf running".
4.  **ControlNet:** Use ControlNet-Depth to preserve the structure of the original video.

---

## 🎓 Interview Focus

1.  **VRAM Requirements?**
    - Video models are huge. Generating 24 frames of $576 \times 320$ requires ~8GB VRAM with fp16.
    - **CPU Offload:** Moves layers to CPU when not in use. Slows down inference but saves memory.

2.  **FPS vs Frame Count?**
    - The model generates *frames* (e.g., 24 images).
    - You decide the playback speed (FPS) when saving. 24 frames at 8fps = 3 seconds. 24 frames at 24fps = 1 second.

---

**Video Pipeline: Hollywood on your laptop!**
