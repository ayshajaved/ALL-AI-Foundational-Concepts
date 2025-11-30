# Music Generation Intro

> **Symbolic vs Raw Audio** - Two Paths to Melody

---

## 🎼 Symbolic Generation (MIDI)

Treats music like text (Language Modeling).
- **Data:** MIDI files (Note On, Note Off, Pitch, Velocity).
- **Representation:** "Piano Roll" or Token sequence.
- **Models:** Transformer (Music Transformer, MuseNet).
- **Pros:** Easy to structure, editable.
- **Cons:** Sounds like a synthesizer (Game Boy music). No timbre/vocals.

---

## 🌊 Raw Audio Generation (Waveform)

Treats music like a signal.
- **Data:** MP3/WAV files.
- **Representation:** Waveforms or Spectrograms.
- **Models:** Jukebox (VQ-VAE), MusicLM (AudioLM), Stable Audio (Diffusion).
- **Pros:** Full fidelity (Vocals, Instruments, Reverb).
- **Cons:** Extremely high dimensionality (44.1kHz = 44,100 samples/sec). Hard to structure long-term (Verse-Chorus).

---

## 🎹 MIDI Tokenization

How to feed music to a GPT model?
**REMI (Revamped MIDI-derived events):**
- Tokens: `Bar`, `Position`, `Pitch`, `Duration`, `Velocity`.
- Sequence: `[Bar] [Pos_0] [Pitch_60] [Dur_4] [Pos_16] [Pitch_64] ...`

---

## 💻 Python: MIDI Processing (PrettyMIDI)

```python
import pretty_midi

# Load MIDI
midi_data = pretty_midi.PrettyMIDI("song.mid")

# Extract Notes
for instrument in midi_data.instruments:
    if not instrument.is_drum:
        for note in instrument.notes[:5]:
            print(f"Start: {note.start}, End: {note.end}, Pitch: {note.pitch}")

# Synthesize to Audio (Fluidsynth)
audio_data = midi_data.fluidsynth()
```

---

## 🎓 Interview Focus

1.  **Why is long-term structure hard in music?**
    - In text, a sentence is ~10 seconds. In music, a song is 3 minutes.
    - A 3-minute song at 44.1kHz is 8 million samples.
    - Even with MIDI, the "Verse 1" and "Verse 2" are thousands of tokens apart. Transformers struggle with this context length (though Linear Attention helps).

2.  **Symbolic vs Audio?**
    - **Symbolic:** Composition (What notes to play).
    - **Audio:** Production (How it sounds).

---

**Music Generation: From sheet music to Spotify!**
