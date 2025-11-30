# GenAI (Audio/Video) Interview Prep

> **Mastering the Multimodal Interview** - Top Questions

---

## 🟢 Beginner (Concepts)

1.  **What is a Spectrogram?** A visual representation of the spectrum of frequencies of a signal as it varies with time.
2.  **What is the Mel Scale?** A perceptual scale of pitches judged by listeners to be equal in distance from one another.
3.  **TTS vs ASR?** Text-to-Speech (Generation) vs Automatic Speech Recognition (Transcription).
4.  **What is a Vocoder?** A model that converts a Mel-Spectrogram into a raw audio waveform.
5.  **What is FPS?** Frames Per Second. Standard video is 24 or 30 or 60.

---

## 🟡 Intermediate (Architectures)

6.  **Explain WaveNet.** Uses dilated causal convolutions to generate raw audio sample-by-sample.
7.  **How does Whisper handle multiple languages?** It uses a language token `<|lang|>` at the start of the sequence to condition the decoder.
8.  **What is the "Flickering" problem in Video Gen?** Lack of temporal consistency. Frame $t$ and $t+1$ are generated independently.
9.  **How does 3D Convolution work?** The kernel has 3 dimensions ($Time \times Height \times Width$) and slides across the video volume.
10. **What is a VQ-VAE (Jukebox)?** Compresses audio into discrete tokens using a codebook, allowing Transformers to predict music.

---

## 🔴 Advanced (SOTA)

11. **Explain Factorized Attention in Video Diffusion.** Splitting $T \times H \times W$ attention into Spatial ($H \times W$) and Temporal ($T$) to save compute.
12. **What is Classifier-Free Guidance (CFG)?** Training the model with and without text conditioning. During inference, extrapolating towards the text-conditioned output. Crucial for adherence.
13. **How does ControlNet work for Video?** Using a copy of the encoder to inject conditions (Pose/Depth). For video, we often use sparse control on keyframes + optical flow.
14. **What is Riffusion?** Fine-tuning Stable Diffusion on spectrogram images to generate audio.
15. **Explain Emu Video's Factorization.** Step 1: Generate Image. Step 2: Animate Image. Decouples aesthetics from motion.

---

## 🧠 System Design Scenarios

**Q: Design a Real-Time Voice Changer.**
- **Input:** Microphone Stream.
- **Pipeline:**
    1.  **VAD:** Detect voice.
    2.  **ASR (Optional):** Convert to text (slow).
    3.  **VC (Voice Conversion):** RVC (Retrieval-based Voice Conversion) is better. Audio $\to$ Content Code $\to$ Target Speaker Decoder.
- **Latency:** Must be < 50ms. Use streaming inference (process chunks).

**Q: Design a Text-to-Video Search Engine.**
- **Data:** Video clips.
- **Model:** CLIP (or VideoCLIP).
- **Process:**
    1.  Extract frames from video.
    2.  Embed frames using CLIP Image Encoder. Average them to get Video Embedding.
    3.  Embed search query using CLIP Text Encoder.
    4.  Compute Cosine Similarity.
- **Index:** Vector Database (FAISS/Pinecone).

---

**You are now a Multimodal AI Expert!**
