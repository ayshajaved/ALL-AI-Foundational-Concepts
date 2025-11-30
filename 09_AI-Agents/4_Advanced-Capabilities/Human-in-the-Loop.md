# Human-in-the-Loop (HITL)

> **Supervision** - Keeping Agents on the Rails

---

## 🛑 Why HITL?

Agents are probabilistic. They hallucinate. They make mistakes.
For critical tasks (deploying code, sending emails, transferring money), **Human Approval** is mandatory.

---

## 🚦 Interaction Patterns

1.  **Approval (Gatekeeper):**
    - Agent: "I plan to delete `production_db`. Proceed?"
    - Human: "NO!"
    - Agent: "Understood. Aborting."

2.  **Feedback (Guidance):**
    - Agent: "Here is the draft email."
    - Human: "Make it more professional."
    - Agent: "Updated draft..."

3.  **Interrupt (Emergency Stop):**
    - The human watches the agent's live trace.
    - If the agent starts looping or going off-track, the human hits "Stop" or injects a new prompt.

---

## 💻 Implementation (LangGraph)

LangGraph allows defining "Breakpoints".

```python
from langgraph.graph import StateGraph

# Define Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)

# Add Edge with Interrupt
workflow.add_edge("agent", "tool")

# Compile with Interrupt
app = workflow.compile(interrupt_before=["tool"])

# Run
thread = {"configurable": {"thread_id": "1"}}
for event in app.stream(inputs, thread):
    pass

# The graph pauses before 'tool'.
# We can inspect the state.
snapshot = app.get_state(thread)
print(snapshot.values["next_action"])

# Resume (if approved)
app.stream(None, thread)
```

---

## 🎓 Interview Focus

1.  **RLHF vs HITL?**
    - **RLHF:** Offline training. Teaching the model *general* alignment.
    - **HITL:** Online inference. Guiding a *specific* task execution.

2.  **What is "Steerability"?**
    - The ability of an agent to change direction based on user feedback mid-task without losing context.

---

**HITL: Trust, but verify!**
