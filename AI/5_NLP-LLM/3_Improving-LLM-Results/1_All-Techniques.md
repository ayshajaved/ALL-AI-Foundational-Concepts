# Improving LLM Results: All Key Techniques Explained

This guide brings together **all the main techniques for customizing and improving large language model (LLM) results**, with clear explanations, step-by-step practical guidance, and side-by-side differences. It includes Gemini 2.5 Flash and models like GPT, covering:

## Prompting:
Prompting is the process of providing specific instructions to a GenAI tool to receive new information or to achieve a desired outcome on a task.

- Prompt engineering
- Prompt tuning (soft prompt tuning)
- Fine-tuning (supervised model tuning)
- Retrieval-Augmented Generation (RAG)

***

## 1. Technique Overview Table

| Technique | What Changes? | Data Needed | Expense | Use Case Example |
| :-- | :-- | :-- | :-- | :-- |
| **Prompt Engineering** | Manual text prompts | None/manual | Low | Rewriting chatbot instructions |
| **Prompt Tuning** | Soft prompt vectors (embeddings) | Labeled pairs | Medium | Domain-specific task adaptation |
| **Fine-Tuning** | Model weights | Large dataset | High | Deeply specialized legal/medical bot |
| **Retrieval-Augmented Generation** | Adds info to prompt at runtime | Up-to-date docs | Varies | Real-time document Q\&A |


***

## 2. Prompt Engineering

**What:**

- Writing and structuring better instructions, questions, or context for the model, often using examples or specific language.
- You do this by hand, designing prompt templates and rewriting natural language queries.[^1][^10]

**Steps:**

1. Clarify what answer you need—add specifics, steps, and roles ("You are an expert...").
2. Try few-shot prompting: give example Q\&A pairs.
3. Experiment with parameters (temperature, top_p) if available.
4. Iterate: run, review, and refine until results are best for your use-case.

**Example**:
Instead of "Tell me about solar panels":
_You are a solar expert. List the top three considerations for installing a home system in Pakistan, and explain each one in simple terms._

**When To Use:**

- Quickest and cheapest way to improve results for many applications, including Gemini 2.5 Flash. No code changes—just better text.[^10][^1]

***

## 3. Prompt Tuning (Soft Prompt Tuning)

**What:**

- Optimizes a small set of *soft prompt vectors* (continuous embeddings)—not visible text—to nudge model outputs for a specific task or style without changing the main model.[^2]
- Only these virtual prompt tokens are trained; all model weights are frozen.

**Steps in Practice (e.g., Gemini 2.5 Flash on Vertex AI):**

1. **Prepare Task Data:**
    - Collect input/output pairs: e.g., customer question, ideal customer support answer.
    - Format data as JSONL or CSV, upload to Google Cloud Storage.
2. **Start a Prompt Tuning Job:**
    - In Vertex AI, choose *Prompt Tuning* as your tuning type.
    - Point to your dataset and set hyperparameters like number of soft prompts (e.g., 10–50), learning rate (typically default works).
    - Start the job. Only soft prompts are optimized by the system.
3. **Model Use:**
    - Your tuned endpoint automatically prepends the optimized soft prompt vectors to every input—steering the model for best task results.
    - You call the API as usual; improvement is automatic.[^8][^2]

**Example:**

- Before: "What do I do if my solar panels are damaged?" → *generic answer*
- After prompt tuning with customer support data: Model generates answers with your company’s policies, tone, or expert advice—even if you didn't hand-write these in the prompt!

**What are Soft Prompt Vectors?**

- Invisible numerical vectors (not words), learned/optimized during tuning; model sees `[soft_p1, soft_p2, ... soft_pN, your_input_tokens]` as its input, and only the soft prompts are trained.[^2]

**When To Use:**

- When you want a task-specific, efficient upgrade (e.g., domain adaptation), but can’t—or don’t want to—touch the main model’s weights or re-train the whole network.

***

## 4. Fine-Tuning (Supervised Tuning)

**What:**

- Updates some or all of a model’s internal weights, deeply adapting it to your custom data. This requires more data and compute than prompt tuning.[^8][^2]

**Steps in Practice:**

1. **Build a Large, Quality Labeled Dataset:**
    - Get 100s–1000s of input/response or Q\&A pairs for your domain/task.
2. **Configure a Fine-Tuning Job:**
    - In Vertex AI, select *Supervised Fine-Tuning* (SFT) instead of prompt tuning for Gemini or similar models.
3. **Train:**
    - The system optimizes many (or all) model parameters using your data—adjusting internal knowledge, logic, and style.
    - Much heavier than prompt tuning. Takes longer, costs more.
4. **Deploy:**
    - Use your fine-tuned model endpoint—fast, deeply customized answers.

**Example:**

- Domain-specific medical bot handling complex consultations, after training on thousands of real medical conversations.

**When To Use:**

- When prompt tuning isn’t precise enough; you need deep expertise or new skills the base model doesn’t have.

***

## 5. Retrieval-Augmented Generation (RAG)

**What:**

- The model, before generating its answer, searches documents (vector database or keyword search) for relevant info—and injects those snippets as extra context.[^10]
- No training or model change—just smarter data delivery per query.

**Steps:**

1. Index your knowledge base (documents, web pages, FAQs) using a vector database or search service.
2. For each user input:
a. Retrieve the most relevant passages/facts using embedding similarity or search.
b. Combine those results with the user's question, then send to the LLM as a long prompt.
3. The LLM answers with awareness of BOTH base model knowledge and retrieved context.

**Example:**

- A chatbot for solar support: When asked about the latest warranty policies, it searches your internal docs for the newest info, then answers accurately, even if not part of its training data.

**When To Use:**

- For up-to-date, document-grounded answers—especially when info changes or is proprietary.

***

## 6. The Difference at a Glance

| Aspect | Prompt Engineering | Prompt Tuning | Fine-Tuning | RAG |
| :-- | :-- | :-- | :-- | :-- |
| *What changes?* | Prompt text | Soft prompt vectors | Model weights | Extra prompt context added at runtime |
| *Skill needed* | None | ML/data setup | Data science/ML ops | System integration |
| *Cost* | None/very low | Low/medium | High | Low/medium (infra setup) |
| *Data needed* | None/low | 50–200+ pairs | 1,000+ examples | Docs, website, or text knowledge base |
| *Speed* | Immediate | Few hours | Days–weeks | Immediate (after setup) |
| *Complexity* | Manual | Automated, managed | Heavily managed | Middleware required |
| *Use case* | Quick optimization, tone | Efficient domain adapt | Deep, robust expertise | Up-to-date or proprietary information |


***

## 7. Practical Scenario (Gemini 2.5 Flash)

Suppose you want to build a solar support chatbot with Gemini 2.5 Flash:

- **Prompt Engineering:** Handwrite helpful, detailed prompts and system messages in your app code. Refine these based on results.[^1][^10]
- **Prompt Tuning:** Collect 100+ customer service Q\&A; use Google Vertex AI Prompt Tuning to train soft prompts for your custom endpoint.[^2]
- **Fine-Tuning:** Gather 1,000+ actual chats; request Supervised Fine-Tuning for the model to fully learn your support flow language and technical policies.[^8][^2]
- **RAG:** Set up a vector database with your latest manuals and policies; retrieve top passages per question and inject them before calling Gemini.

***

## 8. Additional References

- [Gemini Prompting Strategies][^1]
- [Vertex AI Supervised Tuning Guide][^2]
- [Gemini 2.5 Pro and Flash Comparison][^3][^4][^5]
- [Prompt Engineering in Vertex AI][^10]

***

## 9. Quick Recap

- Start with prompt engineering for fastest iteration.
- If you want automation but still low cost, use prompt tuning (soft prompts) with modest data.
- For deep domain behavior, fine-tune with larger data.
- For real-time, accurate, and up-to-date info, add RAG to any of the above.
<span style="display:none">[^6][^7][^9]</span>

<div align="center">⁂</div>

[^1]: https://ai.google.dev/gemini-api/docs/prompting-strategies

[^2]: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning

[^3]: https://www.reddit.com/r/Bard/comments/1k31n3e/i_asked_gemini_25_flash_what_is_the_difference/

[^4]: https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/

[^5]: https://vapi.ai/blog/gemini-flash-vs-pro

[^6]: https://wandb.ai/byyoung3/Generative-AI/reports/Evaluating-the-new-Gemini-2-5-Pro-Experimental-model--VmlldzoxMjAyNDMyOA

[^7]: https://developers.googleblog.com/en/gemini-25-flash-lite-is-now-stable-and-generally-available/

[^8]: https://drlee.io/fine-tune-gemini-2-5-07734306b99b

[^9]: https://www.linkedin.com/posts/dorian-smiley-97a72a14_in-our-testing-of-our-ai-coding-platform-activity-7381808432158388224-alTX

[^10]: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-design-strategies

