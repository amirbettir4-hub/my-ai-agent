"""test_setup.py — check every key and service before running the agent."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

PASS, FAIL = "✅", "❌"
results = []


def check(name, fn):
    try:
        result = fn()
        results.append((PASS, name, result))
    except Exception as e:
        results.append((FAIL, name, f"{type(e).__name__}: {e}"))


# --- 1. Groq ---
def test_groq():
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    c = Groq(api_key=key)
    r = c.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "reply with just: ok"}],
        max_tokens=5,
    )
    return f"replied: {r.choices[0].message.content.strip()[:40]}"


# --- 2. OpenRouter vision ---
def test_openrouter():
    import requests
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.getenv(
                "VISION_MODEL",
                "meta-llama/llama-3.2-11b-vision-instruct:free",
            ),
            "messages": [{"role": "user", "content": "reply with just: ok"}],
            "max_tokens": 5,
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
    return f"replied: {r.json()['choices'][0]['message']['content'].strip()[:40]}"


# --- 3. Hugging Face ---
def test_hf():
    import requests
    key = os.getenv("HF_TOKEN")
    if not key:
        raise RuntimeError("HF_TOKEN not set")
    r = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {key}"},
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return f"user: {r.json().get('name', '?')}"


# --- 4. Cerebras ---
def test_cerebras():
    import requests
    key = os.getenv("CEREBRAS_API_KEY")
    if not key:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    r = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
        timeout=20,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
    return "ok"


# --- 5. DuckDuckGo ---
def test_ddg():
    from duckduckgo_search import DDGS
    with DDGS() as d:
        r = list(d.text("hello world", max_results=1))
    if not r:
        raise RuntimeError("no results")
    return f"got: {r[0].get('title', '')[:40]}"


# --- 6. ffmpeg installed? ---
def test_ffmpeg():
    import subprocess
    r = subprocess.run(["ffmpeg", "-version"],
                       capture_output=True, text=True, timeout=5)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg not found in PATH")
    return r.stdout.splitlines()[0][:60]


check("Groq (brain + whisper)", test_groq)
check("OpenRouter (vision)", test_openrouter)
check("Hugging Face (images)", test_hf)
check("Cerebras (fallback)", test_cerebras)
check("DuckDuckGo (web search)", test_ddg)
check("FFmpeg (video/audio)", test_ffmpeg)

print("\n" + "=" * 60)
for status, name, msg in results:
    print(f"{status} {name:<30} {msg}")
print("=" * 60)

fails = sum(1 for s, _, _ in results if s == FAIL)
sys.exit(1 if fails else 0)