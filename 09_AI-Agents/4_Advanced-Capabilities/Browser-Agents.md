# Browser Agents

> **Navigating the Web** - Selenium and Playwright

---

## 🌐 The Challenge

The web is designed for humans (HTML/CSS), not LLMs.
- **DOM Tree:** Too large for context window.
- **Dynamic Content:** JavaScript, Popups, Infinite Scroll.
- **Visuals:** Buttons that are just images.

---

## 🛠️ Approaches

1.  **HTML Parsing (Text-based):**
    - Strip tags, keep text and links.
    - Feed simplified HTML to LLM.
    - *Pros:* Fast, cheap.
    - *Cons:* Fails on complex JS apps (React/Vue).

2.  **Visual Grounding (Multimodal):**
    - Screenshot the page.
    - Overlay a **Set-of-Marks (SoM)** grid (bounding boxes with IDs).
    - Ask GPT-4V: "Click on box 12".
    - *Pros:* Works on anything visible.
    - *Cons:* Slow, expensive.

---

## 🤖 Multi-On / Adept

Commercial agents that specialize in browsing.
They handle auth, cookies, and navigation robustly.

---

## 💻 Implementation (Playwright + LLM)

```python
from playwright.sync_api import sync_playwright

def run_browser_agent(goal):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://google.com")
        
        while True:
            # 1. Get State
            content = page.content() # Simplify this!
            
            # 2. Ask LLM
            action = llm.predict(f"Goal: {goal}. Page: {content}")
            
            # 3. Execute
            if "CLICK" in action:
                selector = extract_selector(action)
                page.click(selector)
            elif "TYPE" in action:
                text = extract_text(action)
                page.keyboard.type(text)
```

---

## 🎓 Interview Focus

1.  **Accessibility Tree?**
    - Instead of the raw DOM, browsers generate an Accessibility Tree for screen readers.
    - This is a much cleaner representation for LLMs (contains Roles, Labels, States).

2.  **Context Window Optimization?**
    - You cannot feed the whole HTML.
    - **RAG for DOM:** Embed the HTML nodes. Retrieve only the nodes relevant to the user's query ("Find the login button").

---

**Browser Agents: The interface to the internet!**
