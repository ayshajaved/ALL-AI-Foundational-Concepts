# Structured Output

> **Beyond Chat** - Forcing JSON Reliability

---

## 🧱 The Challenge

Agents often need to output data for *other* programs to use, not just for humans to read.
- **Bad:** "Sure! Here is the JSON you asked for: ```json { ... } ```"
- **Good:** `{ ... }` (Pure JSON).

---

## 🏗️ Techniques

1.  **JSON Mode:**
    - A flag in OpenAI/Anthropic APIs (`response_format={"type": "json_object"}`).
    - Guarantees valid JSON syntax (brackets closed).
    - *Cons:* Doesn't validate the *schema* (keys might be wrong).

2.  **Instructor (Library):**
    - Patches the OpenAI client to return Pydantic objects directly.
    - Uses Function Calling under the hood to force the schema.
    - **Retries:** If validation fails, it sends the error back to the LLM ("You forgot the 'age' field") and asks it to fix it.

3.  **Constrained Decoding (Local Models):**
    - Libraries like `llama.cpp` or `guidance` use **Grammars (GBNF)**.
    - They mask the logits so the model *cannot* output a token that violates the JSON schema.
    - 100% reliability.

---

## 💻 Implementation (Instructor)

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Patch the client
client = instructor.patch(OpenAI())

class UserDetail(BaseModel):
    name: str
    age: int
    hobbies: list[str]

user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail, # The magic
    messages=[
        {"role": "user", "content": "Extract info: John is 30 and loves coding."}
    ]
)

print(user.name) # John
print(user.age)  # 30
```

---

## 🎓 Interview Focus

1.  **Why is Constrained Decoding better than Prompting?**
    - Prompting ("Please output JSON") relies on the model obeying.
    - Constrained Decoding modifies the probability distribution. The probability of an invalid token becomes 0. It is mathematically impossible to generate invalid syntax.

2.  **Streaming Structured Output?**
    - Harder. You need a partial JSON parser to validate the stream as it arrives.
    - Vercel AI SDK and Instructor support partial streaming.

---

**Structured Output: The glue between AI and Code!**
