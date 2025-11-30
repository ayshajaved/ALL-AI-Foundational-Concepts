# Tool Use & Function Calling

> **Hands and Feet** - Connecting LLMs to the World

---

## 🛠️ The Problem

LLMs output text. APIs expect JSON.
Regex parsing (`Action: Search`) is brittle. If the model forgets a quote or a bracket, the agent crashes.

---

## 🧩 OpenAI Function Calling (Tool Use)

Fine-tuned models (GPT-4, Llama 3) natively understand **Schemas**.
You pass a list of function definitions (JSON Schema) along with the prompt.
The model outputs a structured JSON object to call the function.

**The Flow:**
1.  **User:** "What's the weather in Tokyo?"
2.  **System:** Sends Prompt + `get_weather` schema.
3.  **Model:** Returns `tool_calls=[{name: "get_weather", arguments: "{\"location\": \"Tokyo\"}"}]`.
4.  **System:** Executes API. Gets "25°C".
5.  **System:** Sends "25°C" back to Model.
6.  **Model:** "The weather in Tokyo is 25°C."

---

## 📜 Defining Tools (Pydantic)

Modern libraries (LangChain, LlamaIndex) use Pydantic to define schemas automatically.

```python
from pydantic import BaseModel, Field
from typing import Optional

class WeatherInput(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: Optional[str] = Field(default="celsius", description="celsius or fahrenheit")

# The library converts this class to:
# {
#   "name": "WeatherInput",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "location": {"type": "string", ...},
#       "unit": {"type": "string", ...}
#     },
#     "required": ["location"]
#   }
# }
```

---

## 🛡️ Safety & Permissions

Giving an agent `os.system` or `delete_file` is dangerous.
- **Human-in-the-loop:** Require user approval before executing sensitive tools.
- **Read-only:** Restrict agents to `read` operations initially.
- **Sandboxing:** Run code execution tools in a Docker container (e.g., E2B).

---

## 🎓 Interview Focus

1.  **Function Calling vs ReAct?**
    - **ReAct:** Uses text prompting ("Action: Search"). Works on any model.
    - **Function Calling:** Uses fine-tuned tokens/headers. More reliable, handles complex JSON arguments better.

2.  **Hallucinated Tool Calls?**
    - The model might try to call a tool that doesn't exist (`get_stock_price` when you only gave it `get_weather`).
    - **Solution:** System prompt: "Only use the provided tools."

---

**Tool Use: Turning text generators into operators!**
