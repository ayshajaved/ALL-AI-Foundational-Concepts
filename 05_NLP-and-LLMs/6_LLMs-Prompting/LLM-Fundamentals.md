# LLM Fundamentals

> **The Era of Generative AI** - Scaling Laws, Emergence, and Alignment

---

## 📈 Scaling Laws (Kaplan et al., 2020)

Performance of LLMs depends on three factors via a power law:
1.  **N:** Number of Parameters.
2.  **D:** Dataset Size.
3.  **C:** Compute Budget.

**Key Finding:** Bigger is better. We haven't hit the plateau yet.
**Chinchilla Scaling (Hoffmann et al., 2022):** For optimal training, double the model size $\to$ double the data. Most models (like GPT-3) were *undertrained*.

---

## ✨ Emergent Abilities

Capabilities that are **not present** in small models but appear suddenly in large models.
- **Arithmetic:** Small models guess; large models calculate.
- **Coding:** Requires reasoning chains.
- **In-Context Learning:** Learning from prompt examples.

---

## 🛡️ Alignment: Making Models Helpful & Harmless

Raw LLMs (Base Models) just predict the next token. They are chaotic and can be toxic.
We need to align them to human intent.

### 1. SFT (Supervised Fine-Tuning)
Train on high-quality Instruction-Response pairs.
- *Input:* "Write a poem."
- *Output:* "Roses are red..."
- **Result:** Instruction-tuned model (e.g., Vicuna, Alpaca).

### 2. RLHF (Reinforcement Learning from Human Feedback)
The secret sauce of ChatGPT.
1.  **Collect Comparison Data:** Human ranks 2 model outputs (A > B).
2.  **Train Reward Model (RM):** Predicts the human score.
3.  **PPO (Proximal Policy Optimization):** Optimize the LLM to maximize the Reward Model's score while staying close to the original model (KL penalty).

### 3. DPO (Direct Preference Optimization) - 2023
**New Standard:** Skips the Reward Model and PPO entirely.
Optimizes the policy directly on the preference data ($A > B$).
Mathematically equivalent to RLHF but stable and simple (like standard classification).

---

## 🎓 Interview Focus

1.  **What is the difference between a Base Model and a Chat Model?**
    - **Base:** Completes text (e.g., "The capital of France is" $\to$ "Paris").
    - **Chat:** Follows instructions (e.g., "What is the capital of France?" $\to$ "The capital of France is Paris.").

2.  **Why is RLHF hard?**
    - Training stability (RL is finicky).
    - Reward hacking (Model finds loopholes to get high scores without being helpful).

3.  **Explain Chinchilla Scaling.**
    - It proved that for a fixed compute budget, you should train a *smaller* model on *more* data than previously thought. (Llama is a "Chinchilla-optimal" model).

---

**Fundamentals: Understanding the giants!**
