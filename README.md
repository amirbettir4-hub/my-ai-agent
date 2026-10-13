# 🤖 Super Agent

A multi-model, tool-using AI assistant with a persistent skill library, long-term memory, and a modern web UI. Built to survive provider outages with automatic fallback across 9+ language models.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

---

## ✨ Features

- **Multi-provider fallback chain** — if one model dies, the next takes over automatically
- **22 built-in tools** — image generation, TTS, transcription, web search, code execution, video analysis, file management
- **Local Whisper transcription** — offline speech-to-text, no API keys, no rate limits
- **Skill library** — the agent saves reusable procedures and recalls them semantically
- **Long-term memory** — persistent across sessions via Chroma vector store
- **Modern web UI** — streaming responses, 5 themes, chat search, export, keyboard shortcuts
- **CLI mode** — same brain, terminal interface

---

## 🎯 What it can do

| Category | Capability |
|---|---|
| **Code** | Write, run, and iterate on Python in a local sandbox |
| **Web** | Search for current information via DuckDuckGo |
| **Media** | Generate 2D images, text-to-speech (4-tier fallback) |
| **Video** | Transcribe YouTube, analyze video frames with vision models |
| **Files** | Read/write inside a workspace, scan project structure |
| **Memory** | Save facts long-term, recall across sessions |
| **Skills** | Save procedures, recall them by meaning, not keywords |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Web UI (index.html)                 │
│         or CLI (agent.py main)              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│      FastAPI Server (web_server.py)         │
│      • Session management                   │
│      • Server-Sent Events streaming         │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         Agent Core (agent.py)               │
│      • run_turn() — event stream            │
│      • Manual model fallback chain          │
│      • Tool execution + result truncation   │
└──┬────────────┬────────────┬────────────────┘
   │            │            │
   ▼            ▼            ▼
┌──────┐   ┌────────┐   ┌─────────────┐
│ LLMs │   │ Tools  │   │ Memory      │
│      │   │        │   │ + Skills    │
│ 9    │   │ 22     │   │ Chroma      │
│models│   │ tools  │   │ vector DB   │
└──────┘   └────────┘   └─────────────┘
```

### Provider fallback chain

The agent tries models in this order and uses the first one that responds:

1. **Mistral** (if `MISTRAL_API_KEY` set — free tier, no country restrictions)
2. **OpenRouter** — 50 free requests/day
3. **Pollinations.AI** — no key required
4. **Groq** — if the network allows
5. **Cerebras** — if on a paid tier

If a model fails *before* emitting any tokens, the next is tried. If it fails *mid-stream*, the error is reported honestly.

---

## 🚀 Quickstart

### Prerequisites

- Python 3.11+
- FFmpeg on PATH ([Windows build](https://www.gyan.dev/ffmpeg/builds/))
- At least one LLM API key (see below)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/my-ai-agent.git
cd my-ai-agent
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
# At least one of these is required
OPENROUTER_API_KEY=sk-or-v1-...    # Free: https://openrouter.ai/keys
MISTRAL_API_KEY=...                # Free: https://console.mistral.ai/

# Optional — enables image generation
HF_TOKEN=hf_...                    # Free: https://huggingface.co/settings/tokens
```

### Run

**Web UI:**
```bash
python web_server.py
# Open http://127.0.0.1:8000
```

**CLI:**
```bash
python agent.py
```

### Test setup

```bash
python test_setup.py
```

Prints a checklist of which providers and services are working.

---

## 🎨 Web UI

The browser interface includes:

- **Streaming responses** — see tokens as they arrive
- **5 themes** — Violet, Ocean, Crimson, Emerald, Noir
- **Inline images** — generated images render directly in chat
- **Chat search** (`Ctrl+F`) — find any message
- **Export chat** (`Ctrl+S`) — download as Markdown
- **Session management** — persistent conversation history
- **Voice input** — speech-to-text via Web Speech API
- **Keyboard shortcuts** — press `?` for the full list

---

## 🛠️ Tools

| Tool | What it does |
|---|---|
| `execute_python` | Run Python in a sandboxed subprocess |
| `search_web` | DuckDuckGo search |
| `create_image` | Generate images (HuggingFace → Pollinations fallback) |
| `speak` | Text-to-speech (edge-tts → gTTS → pyttsx3 → ttsfm) |
| `transcribe_youtube` | Speech-to-text via local Whisper |
| `watch_youtube_video` | Full video analysis: frames + audio |
| `write_project` / `read_project` | Files inside the workspace |
| `set_project_workspace` | Lock file operations to a folder |
| `scan_project` | Detect project type and count files |
| `learn_skill` / `recall_skill` | The self-improvement loop |
| `save_memory` | Long-term memory storage |

Full list: 22 tools. See `agent.py`.

---

## 🧠 How the skill library works

**Before** any complex task, the agent calls `recall_skill(query)` to check if it already knows how to do it. If a relevant skill exists, it loads the steps and follows them.

**After** solving something non-trivial, the agent calls `learn_skill(...)` to save the procedure. Skills persist forever and are searchable by meaning, not just keyword match.

Over time, the agent gets measurably better at recurring tasks.

---

## 📁 Project structure

```
my-ai-agent/
├── agent.py            # Core agent logic + CLI
├── web_server.py       # FastAPI server
├── index.html          # Web UI
├── tools.py            # All 22 tool implementations
├── skills.py           # Skill library (Chroma + JSON)
├── prompts.py          # System prompts
├── config.py           # Central configuration
├── requirements.txt    # Dependencies
├── test_setup.py       # Provider health check
├── tests/              # Unit tests
├── ai_creations/       # Generated files (images, audio)
├── chroma_db/          # Memory vector store
└── skills_db/          # Skill vector store
```

---

## ⚠️ Known limitations

- **Free API tiers change frequently.** Three providers used in early development have since been retired or paywalled. The fallback chain handles this, but expect to add new keys occasionally.
- **3D generation** relies on three.ws, which has intermittent free-tier availability.
- **Local Whisper** is CPU-bound. Transcription of long audio on old hardware is slow.

---

## 🗺️ Roadmap

- [ ] Local LLM support via llama.cpp
- [ ] RAG over user documents
- [ ] Multi-step planning before tool execution
- [ ] Cost tracking per provider

---

## 📄 License

MIT — see `LICENSE`.

---

## 🙏 Built with

- [LangChain](https://python.langchain.com/) — agent framework
- [ChromaDB](https://www.trychroma.com/) — vector memory
- [FastAPI](https://fastapi.tiangolo.com/) — web server
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local transcription
- [OpenRouter](https://openrouter.ai/) + others — LLM access