# Coding Agents

> **Software Engineering 2.0** - Writing, Running, and Fixing Code

---

## 💻 The Loop

A Coding Agent is not just a code generator (Copilot).
It is a **Loop**:
1.  **Write:** Generate code.
2.  **Run:** Execute code (in sandbox).
3.  **Debug:** Read stderr. If error, fix code.
4.  **Test:** Write unit tests. Verify pass.

---

## 🛠️ Tools

- **File System:** Read/Write files.
- **Terminal:** Run shell commands (`ls`, `pip install`, `pytest`).
- **Linter:** Static analysis (`pylint`, `flake8`) to catch syntax errors before running.
- **LSP (Language Server Protocol):** Go to definition, Find references.

---

## 🛡️ Sandboxing (E2B / Docker)

**Never** run agent-generated code on your host machine.
- `rm -rf /`
- Infinite loops.
- Reverse shells.

**E2B (Code Interpreter SDK):**
Provides secure, ephemeral cloud sandboxes for agents.

---

## 💻 Implementation (Swe-agent)

**Swe-agent (Princeton):**
- Converts GitHub Issues $\to$ Pull Requests.
- Uses a custom "Agent-Computer Interface" (ACI).
- Instead of full bash, it gives the agent specialized commands:
    - `search_file <query>`
    - `edit_file <start_line> <end_line>`
    - `run_test <test_file>`

---

## 🎓 Interview Focus

1.  **Why is "edit_file" hard?**
    - LLMs are bad at counting lines.
    - **Search-and-Replace** is often better: "Replace this block of code with that block."

2.  **Test-Driven Development (TDD) for Agents?**
    - The agent should write the test *first*.
    - Then write code until the test passes.
    - This provides a clear "Definition of Done".

---

**Coding Agents: The future of development!**
