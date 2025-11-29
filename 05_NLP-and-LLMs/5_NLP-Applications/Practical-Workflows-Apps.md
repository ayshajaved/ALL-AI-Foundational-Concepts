# Practical Workflows: NLP App

> **Deployment with Streamlit** - Building a Translation & Summarization UI

---

## 🛠️ The Project

Build a web app that:
1.  Takes text input.
2.  Translates it to a target language.
3.  Summarizes the translated text.

---

## 💻 Implementation (`app.py`)

```python
import streamlit as st
from transformers import pipeline

# 1. Load Models (Cached)
@st.cache_resource
def load_models():
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return translator, summarizer

translator, summarizer = load_models()

# 2. UI Layout
st.title("NLP Magic 🪄")
st.subheader("Translate & Summarize")

text_input = st.text_area("Enter English Text:", height=200)

if st.button("Process"):
    if text_input:
        with st.spinner("Translating..."):
            # Translate
            translation = translator(text_input)[0]['translation_text']
            st.success("Translation (French):")
            st.write(translation)
            
        with st.spinner("Summarizing..."):
            # Summarize (Original English)
            summary = summarizer(text_input, max_length=50, min_length=10)[0]['summary_text']
            st.info("Summary (English):")
            st.write(summary)
    else:
        st.warning("Please enter some text!")
```

---

## 🚀 Running the App

```bash
pip install streamlit transformers torch sentencepiece
streamlit run app.py
```

---

## 📦 Deployment (HuggingFace Spaces)

1.  Create a new Space on huggingface.co.
2.  Select SDK: **Streamlit**.
3.  Upload `app.py` and `requirements.txt`.
4.  Done! Your app is live globally.

---

**You are now a Full-Stack AI Engineer!**
