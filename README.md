# 🤖 Super Agent

A multi-model, tool-using AI assistant with long-term memory, a self-improving skill library, document RAG, disk caching, and a background job queue — built to run on a 2014 PC with 4–8 GB RAM, no GPU, and free APIs only.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)
![CPU-only](https://img.shields.io/badge/hardware-CPU--only-lightgrey)
![Free APIs](https://img.shields.io/badge/cost-%240-brightgreen)

---

## ✨ Features

- **Multi-provider fallback chain** — if one model dies, the next takes over automatically
- **Long-term memory** — extracts durable facts about you across sessions
- **Rolling summarization** — compresses old messages so context never floods
- **Document RAG with upload UI** — drag-and-drop files, then ask questions about them
- **Auto-skill capture** — the agent saves and reuses successful tool patterns
- **Tool result cache** — identical calls return instantly, saving tokens
- **Context budget** — old tool results compacted before being re-sent to the LLM
- **Background job queue** — long tools run without freezing the UI
- **Local Whisper transcription** — offline speech-to-text, no API keys needed
- **Image prompt rewriting** — vivid prompts for better generations
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
| **Docs** | Upload files, semantic search over their contents |
| **Memory** | Save facts long-term, recall across sessions |
| **Skills** | Save procedures, recall them by meaning, not keywords |
| **Jobs** | Run long tools in the background, poll or cancel them |

---

## 🏗️ Architecture
┌─────────────────────────────────────────────┐
│ Web UI (index.html) │
│ or CLI (agent.py main) │
└──────────────────┬──────────────────────────┘
│
┌──────────────────▼──────────────────────────┐
│ FastAPI Server (web_server.py) │
│ • Session management │
│ • SSE streaming │
│ • Upload / docs / jobs endpoints │
└──────────────────┬──────────────────────────┘
│
┌──────────────────▼──────────────────────────┐
│ Agent Core (agent.py) │
│ • run_turn() — event stream │
│ • Manual model fallback chain │
│ • Tool execution + result truncation │
│ • Auto-memory, skills, caching │
└──┬────────────┬────────────┬────────────────┘
│ │ │
▼ ▼ ▼
┌──────┐ ┌────────┐ ┌─────────────┐
│ LLMs │ │ Tools │ │ Memory │
│ │ │ + Jobs │ │ + Skills │
│ 9 │ │ 26 │ │ Chroma │
│models│ │ tools │ │ vector DB │
└──────┘ └────────┘ └─────────────┘

### Provider fallback chain

Models are tried in this order and the first one that responds is used:

1. **Mistral** (if `MISTRAL_API_KEY` set — free tier, no country restrictions)
2. **OpenRouter** — 50 free requests/day
3. **Pollinations.AI** — no key required
4. **Groq** — if the network allows
5. **Cerebras** — if on a paid tier

If a model fails **before** emitting any tokens, the next is tried. If it fails **mid-stream**, the error is reported honestly.

---

## ✨ Recent upgrades

Everything below was added after the initial release:

- **Auto-memory** — after a turn, a small LLM extracts durable facts ("User is from Algeria") and stores them in Chroma. Future turns retrieve them semantically.
- **Rolling summarization** — once a chat crosses ~16 messages, older ones get compressed into a running summary. The LLM never sees the full history, but nothing is lost.
- **Document RAG** — drag-and-drop files into the Documents modal (Ctrl+D). Each file is chunked and embedded. Ask questions, the agent searches your docs.
- **Auto-skill capture** — when a turn uses ≥2 tools successfully, the pattern is extracted and saved as a skill. Next time a similar request comes in, the skill is recalled and injected into the prompt.
- **Tool result cache** — deterministic tool calls (file reads, doc searches, web searches for the same query) are cached to disk. Same args → instant reply.
- **Context budget** — before each LLM call, older tool results are replaced with one-line summaries so huge dumps don't flood the window.
- **Background jobs** — heavy tools (transcription, image gen, video analysis) run through a single-worker queue. The UI stays responsive, jobs can be polled or cancelled, and only one heavy job runs at a time (RAM-safe).
- **Image prompt rewriting** — short prompts get rewritten by an LLM into vivid, detailed descriptions before hitting the image API.

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
