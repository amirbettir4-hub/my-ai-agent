"""config.py — single place to tune the agent."""
import os

from dotenv import load_dotenv
load_dotenv()


# ── OpenRouter ───────────────────────────────────────────────
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv(
        "OPENROUTER_MODEL_FALLBACKS",
        "nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-31b-it:free",
    ).split(",") if m.strip()
]

# ── Mistral ──────────────────────────────────────────────────
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv(
        "MISTRAL_MODEL_FALLBACKS", "open-mistral-nemo"
    ).split(",") if m.strip()
]

# ── Pollinations (no key) ────────────────────────────────────
POLLINATIONS_ENABLED = os.getenv("POLLINATIONS_ENABLED", "true").lower() == "true"
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "openai")
POLLINATIONS_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv(
        "POLLINATIONS_MODEL_FALLBACKS", "mistral"
    ).split(",") if m.strip()
]

# ── Vision ───────────────────────────────────────────────────
VISION_MODEL = os.getenv("VISION_MODEL", "google/gemma-4-31b-it:free")
VISION_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv(
        "VISION_MODEL_FALLBACKS",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ).split(",") if m.strip()
]

# ── Agent loop ───────────────────────────────────────────────
MAX_STEPS = 6
MEMORY_K  = 2
TOOL_RESULT_MAX_CHARS = 4000

# ── Smart mode toggles ───────────────────────────────────────
SMART_PLANNING   = os.getenv("SMART_PLANNING",   "true").lower() == "true"
SMART_REFLECTION = os.getenv("SMART_REFLECTION", "true").lower() == "true"
SMART_SUBAGENTS  = os.getenv("SMART_SUBAGENTS",  "true").lower() == "true"

# Reflection only runs when the answer is longer than this
REFLECTION_MIN_CHARS = 600
# Planning only runs when the request is longer than this
PLANNING_MIN_CHARS   = 80

# ── Document RAG ─────────────────────────────────────────────
RAG_DIR           = os.getenv("RAG_DIR", "./documents")
RAG_CHUNK_SIZE    = 800
RAG_CHUNK_OVERLAP = 100
RAG_TOP_K         = 3

# ── Multi-session projects ───────────────────────────────────
PROJECTS_FILE = "./projects.json"

# ── Memory (Chroma) ──────────────────────────────────────────
CHROMA_DIR            = "./chroma_db"
CHROMA_COLLECTION     = "my_ai_memory"
CHROMA_RAG_COLLECTION = "user_documents"

# ── UI ───────────────────────────────────────────────────────
VERBOSE = True