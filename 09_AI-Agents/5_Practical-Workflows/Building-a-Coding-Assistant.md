# Building a Coding Assistant

> **The Pair Programmer** - File I/O and Linter Integration

---

## 💻 The Goal

Build a CLI agent that can:
1.  Navigate a local codebase.
2.  Read files.
3.  Implement a feature request.
4.  Run the linter to ensure quality.

---

## 🛠️ The Tools

1.  `list_files(path)`: See directory structure.
2.  `read_file(path)`: Read content.
3.  `write_file(path, content)`: Overwrite file.
4.  `run_linter(path)`: Run `flake8` or `pylint`.

---

## 🧠 The Prompt Strategy

We need a **System Prompt** that enforces engineering standards.

```text
You are a Senior Python Engineer.
1. Always read the file before editing it.
2. After editing, ALWAYS run the linter.
3. If the linter fails, fix the errors.
4. Do not delete existing code unless necessary.
```

---

## 💻 Implementation (Simple Loop)

```python
import os
import subprocess

def run_linter(filepath):
    result = subprocess.run(["flake8", filepath], capture_output=True, text=True)
    return result.stdout

def agent_loop(user_request):
    messages = [{"role": "user", "content": user_request}]
    
    while True:
        response = llm.chat(messages, tools=tools)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Execute Tool
                result = execute(tool_call)
                messages.append(tool_result_message(result))
                
                # Special Logic: Auto-Lint
                if tool_call.function.name == "write_file":
                    lint_output = run_linter(tool_call.arguments['path'])
                    if lint_output:
                        messages.append({"role": "user", "content": f"Linter failed: {lint_output}. Fix it."})
        else:
            print(response.content)
            break
```

---

## 🎓 Interview Focus

1.  **Context Window Management?**
    - You can't read the whole repo.
    - **Repo Map:** Generate a tree structure of the repo (files + classes) to help the agent decide which file to read.

2.  **Safety?**
    - What if the agent overwrites `main.py` with empty text?
    - **Backup:** Always create a `.bak` copy before writing.

---

**Coding Assistant: Automating the boring stuff!**
