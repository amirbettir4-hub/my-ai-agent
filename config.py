"""config.py — single place to tune the agent."""
import os

from dotenv import load_dotenv
load_dotenv()


# ── Brain models (OpenRouter primary + fallback chain) ───────
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free",
)
OPENROUTER_MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_MODEL_FALLBACKS",
        "nvidia/nemotron-3-ultra-550b-a55b:free,"
        "google/gemma-4-31b-it:free",
    ).split(",")
    if m.strip()
]

# Other providers (fallbacks if OpenRouter dies entirely)
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
COHERE_MODEL   = os.getenv("COHERE_MODEL", "command-a-03-2025")


# ── Vision models (OpenRouter) ────────────────────────────────
VISION_MODEL = os.getenv("VISION_MODEL", "google/gemma-4-31b-it:free")
VISION_MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "VISION_MODEL_FALLBACKS",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ).split(",")
    if m.strip()
]


# ── Agent loop ────────────────────────────────────────────────
MAX_STEPS = 6        # max tool-call iterations per turn
MEMORY_K  = 2        # how many memories to pull per turn


# ── Memory ────────────────────────────────────────────────────
CHROMA_DIR        = "./chroma_db"
CHROMA_COLLECTION = "my_ai_memory"


# ── UI ────────────────────────────────────────────────────────
VERBOSE = True       # set False to hide status noise