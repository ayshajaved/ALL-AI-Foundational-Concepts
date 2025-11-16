## What Are Hallucinations in LLMs?

**Hallucinations** in large language models (LLMs) are outputs that deviate from reality or logic, leading to contradictions, factual inaccuracies, or irrelevant information. They can include:[^1][^2][^3]

- **Sentence contradictions:** e.g., "The sky is blue" followed by "The sky is green."
- **Prompt contradictions:** Model returns answers that conflict with user instructions (e.g., negative review when asked for a positive one).
- **Factual errors:** Statements that are simply incorrect or invented facts (e.g., "Obama was the first US president").
- **Irrelevant or nonsensical outputs:** Content that makes no sense or doesn’t answer the user’s request.

***

## Why Do Hallucinations Occur?

LLM hallucinations arise from several intertwined causes:

### 1. **Imperfect Training Data**

- Datasets include errors, biases, or conflicting sources.[^2][^6][^7]
- Some web-scale data (e.g., Reddit, Wikipedia) are not always accurate.
- Source-reference divergence: The original source and target outputs in training differ in accuracy or meaning.[^6]


### 2. **Model Architecture and Decoding Methods**

- LLMs are typically auto-regressive: they predict the next word based on previous ones, sometimes leading them further off-track as responses grow longer.[^7][^8]
- Decoding strategies (e.g., beam search, top-k/top-p sampling) can favor fluency, diversity, or novelty at the expense of factual reliability.[^8][^6]
- High temperature or creative settings can increase hallucinations.


### 3. **Incomplete or Ambiguous Prompts**

- Unclear, underspecified, or context-free questions confuse the model, increasing the risk of contradictions or fabricated answers.[^1][^6]


### 4. **Exposure Bias \& Evaluation Issues**

- Models trained to predict likely next words may "guess" when uncertain, rather than abstain or signal uncertainty.[^5][^6]
- If evaluation focuses only on getting an answer (not its confidence/honesty), the model is *rewarded* for making plausible-sounding but wrong responses.[^5]


### 5. **Lack of Real-World Grounding**

- LLMs have no "common sense" or built-in verification—they do not consult a live knowledge base by default.[^4][^2]

***

## Types of Hallucinations (With Examples)

| Type | Description | Example |
| :-- | :-- | :-- |
| Sentence contradiction | Output contradicts itself | "The sky is blue. The sky is green." |
| Prompt contradiction | Contradicts instructions | Asked for positive review, gives negative answer |
| Factual error | Makes up or misstates facts | "Moon is 54 million km away" |
| Irrelevant/nonsense | Output is nonsensical | "Paris is a singer" (when asked capital) |


***

## How to Minimize Hallucinations: Practical Solutions

### 1. **Prompt Engineering**

- **Be specific:** Use detailed prompts, specify output format and required facts (e.g., "Summarize WW2 including countries and causes").
- **Multi-shot prompting:** Give the model several examples to follow (shows pattern/format).
- **Context enrichment:** Add relevant background/context when the question could be ambiguous.


### 2. **Model Settings and Output Controls**

- **Lower temperature:** Reduces randomness, yielding more conservative and usually more accurate outputs.
- **Tuning decoding methods:** Use deterministic decoding (e.g., greedy or beam search) for factual tasks, not high-temperature sampling.


### 3. **Data Quality and Training Improvements**

- **Better, clean data:** Use or advocate for well-curated, diverse and up-to-date datasets.
- **Anti-hallucination fine-tuning:** Retrain on datasets that penalize or correct for errors, possibly with human feedback (RLHF).[^6][^7]


### 4. **External Validation: Retrieval-Augmented Generation (RAG)**

- Combine LLMs with search or knowledge base retrieval to inject up-to-date or verified information.[^4]


### 5. **Feedback, Auditing, and Human-in-the-Loop**

- **Feedback loops:** Let users report bad outputs to improve future answers.
- **Auditing:** Regularly test and review model outputs for consistency and factuality.
- **Human review:** For high-stakes or critical cases, require human approval.

***

## Key Terms

- **Auto-regressive**: A type of model predicting each token based on all previous tokens, leading to error compounding if mistakes are made.
- **Temperature**: A generation parameter controlling randomness; lower is more focused/conservative, higher is more diverse/creative but riskier.
- **Beam search, top-k/top-p sampling**: Decoding methods balancing between accuracy and creativity.
- **Exposure bias**: Tendency for models to make more mistakes as outputs get longer, due to compounding errors.
- **Source-reference divergence**: Misalignment between the input/context and the output/target in training data.
- **RLHF**: Reinforcement Learning from Human Feedback, a method of using human corrections to reduce undesirable model behavior.

***

## Summary Checklist: Mitigating LLM Hallucinations

- Use clear, specific prompts and provide context where needed
- Prefer lower temperature for factual tasks
- Supply demonstration examples (multi-shot)
- Choose or request models trained on higher quality data
- Use retrieval-based architectures for critical factual answers
- Collect feedback and involve humans for important outputs

***

By understanding both the technical and data-driven roots of hallucination, and by applying these pragmatic strategies, users and developers can make LLMs more reliable and trustworthy for everyday and professional use.
<span style="display:none">[^10][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.youtube.com/watch?v=cfqtFvWOfg0

[^2]: https://cloud.google.com/discover/what-are-ai-hallucinations

[^3]: https://www.ibm.com/think/topics/ai-hallucinations

[^4]: https://nexla.com/ai-infrastructure/llm-hallucination/

[^5]: https://openai.com/index/why-language-models-hallucinate/

[^6]: https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)

[^7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11815294/

[^8]: https://www.lakera.ai/blog/guide-to-hallucinations-in-large-language-models

[^9]: https://www.nature.com/articles/s41586-024-07421-0

[^10]: https://www.iguazio.com/glossary/llm-hallucination/

