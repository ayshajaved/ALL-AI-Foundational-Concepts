# Swarm Intelligence

> **Emergent Behavior** - Simple Agents, Complex Results

---

## 🐝 The Concept

In nature (Ants, Bees), no single individual is smart. The **Colony** is smart.
In AI, **Swarm Intelligence** involves many simple (weak) agents interacting to solve complex problems.

---

## 🗳️ Voting & Consensus

How do 10 agents agree on an answer?
1.  **Majority Vote:** Everyone outputs an answer. Most common wins.
2.  **Debate:** Agents critique each other's answers.
    - Agent A: "The answer is 5."
    - Agent B: "You forgot the carry. It's 6."
    - Agent A: "You are right. It is 6."

**Paper:** *Improving Factuality and Reasoning in Language Models through Multiagent Debate* (MIT/Google).

---

## 🧬 Evolution (Genetic Algorithms)

We can evolve the **Prompts** or the **Agent Personalities**.
1.  Initialize 100 agents with random prompt variations.
2.  Run them on a task.
3.  Keep the top 10.
4.  Mutate/Crossover their prompts.
5.  Repeat.

---

## 🎓 Interview Focus

1.  **Ensemble vs Swarm?**
    - **Ensemble:** Run model 10 times, average the logits. (Passive).
    - **Swarm:** Agents *communicate* and *influence* each other. (Active).

2.  **Cost?**
    - Swarms are expensive ($N$ times the cost).
    - Use smaller models (Llama-7B) for the swarm, and maybe one GPT-4 as the "Judge".

---

**Swarm Intelligence: Quantity has a quality all its own!**
