# FFmpeg for AI Engineers

> **The Swiss Army Knife of Media** - Essential Commands

---

## 🛠️ Why FFmpeg?

AI models are picky.
- "Input must be 16kHz mono wav."
- "Video must be 256x256 mp4."
Python libraries (`librosa`, `opencv`) are slow for batch processing. **FFmpeg** is instant.

---

## 📜 Cheat Sheet

### Audio Processing

**1. Convert to 16kHz Mono WAV (Standard for ASR/TTS):**
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```
- `-ar`: Audio Rate (Sample Rate).
- `-ac`: Audio Channels (1 = Mono).

**2. Trim Audio (Start at 00:10, duration 5s):**
```bash
ffmpeg -i input.wav -ss 00:00:10 -t 5 cut.wav
```

### Video Processing

**3. Extract Frames (for Dataset Creation):**
```bash
ffmpeg -i video.mp4 -vf "fps=1" frames/out_%04d.jpg
```
- Extracts 1 frame per second.

**4. Resize and Crop (Prepare for Model):**
```bash
ffmpeg -i input.mp4 -vf "scale=256:256" output.mp4
```

**5. Merge Audio and Video:**
AI generates video (silent) and audio separately. Combine them:
```bash
ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output_final.mp4
```

---

## 💻 Python Wrapper (`ffmpeg-python`)

Instead of `subprocess.run`, use the wrapper:

```python
import ffmpeg

(
    ffmpeg
    .input('input.mp4')
    .filter('fps', fps=1, round='up')
    .output('frame_%d.jpg')
    .run()
)
```

---

## 🎓 Interview Focus

1.  **Codecs vs Containers?**
    - **Container:** `.mp4`, `.mkv`. Holds the streams.
    - **Codec:** `H.264`, `AAC`. Compresses the data.
    - AI models usually need raw decoded frames (RGB), so FFmpeg decodes the H.264 stream before passing it to the model.

---

**FFmpeg: Don't train models without it!**
