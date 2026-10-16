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
MEMORY_K  = 3
TOOL_RESULT_MAX_CHARS = 4000

# ── Smart mode toggles ───────────────────────────────────────
SMART_PLANNING   = os.getenv("SMART_PLANNING",   "true").lower() == "true"
SMART_REFLECTION = os.getenv("SMART_REFLECTION", "true").lower() == "true"
SMART_SUBAGENTS  = os.getenv("SMART_SUBAGENTS",  "true").lower() == "true"

REFLECTION_MIN_CHARS = 600
PLANNING_MIN_CHARS   = 80

# ── Auto-memory + summarization (upgrade #1) ─────────────────
AUTO_MEMORY_ENABLED   = os.getenv("AUTO_MEMORY_ENABLED", "true").lower() == "true"
AUTO_MEMORY_MIN_CHARS = int(os.getenv("AUTO_MEMORY_MIN_CHARS", "40"))
AUTO_MEMORY_MAX_FACTS = int(os.getenv("AUTO_MEMORY_MAX_FACTS", "4"))
MEMORY_DEDUP_DISTANCE = float(os.getenv("MEMORY_DEDUP_DISTANCE", "0.35"))

SESSION_SUMMARY_ENABLED     = os.getenv("SESSION_SUMMARY_ENABLED", "true").lower() == "true"
SESSION_SUMMARY_TRIGGER     = int(os.getenv("SESSION_SUMMARY_TRIGGER", "16"))
SESSION_SUMMARY_KEEP_RECENT = int(os.getenv("SESSION_SUMMARY_KEEP_RECENT", "8"))
SESSION_SUMMARY_MAX_CHARS   = int(os.getenv("SESSION_SUMMARY_MAX_CHARS", "1800"))

# ── Document RAG + upload (upgrade #2) ───────────────────────
RAG_DIR           = os.getenv("RAG_DIR", "./documents")
RAG_CHUNK_SIZE    = 800
RAG_CHUNK_OVERLAP = 100
RAG_TOP_K         = 3

RAG_MAX_DISTANCE  = float(os.getenv("RAG_MAX_DISTANCE", "1.3"))

RAG_MAX_FILE_MB   = int(os.getenv("RAG_MAX_FILE_MB", "25"))

RAG_ALLOWED_EXTS  = [
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".html", ".css", ".csv", ".log",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".pdf",
]

# ── Auto-skill capture (upgrade #3) ──────────────────────────
AUTO_SKILL_ENABLED = os.getenv("AUTO_SKILL_ENABLED", "true").lower() == "true"
AUTO_SKILL_MIN_TOOLS = int(os.getenv("AUTO_SKILL_MIN_TOOLS", "2"))
AUTO_SKILL_MAX_PER_SESSION = int(os.getenv("AUTO_SKILL_MAX_PER_SESSION", "5"))
SKILL_DEDUP_DISTANCE = float(os.getenv("SKILL_DEDUP_DISTANCE", "0.40"))
SKILL_MATCH_MAX_DISTANCE = float(os.getenv("SKILL_MATCH_MAX_DISTANCE", "1.10"))
SKILL_INJECT_K = int(os.getenv("SKILL_INJECT_K", "2"))

# ── Tool result cache (upgrade #4) ───────────────────────────
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_DIR     = os.getenv("CACHE_DIR", "./cache")

CACHEABLE_TOOLS = {
    "read_file":            300,
    "read_project":         300,
    "list_files":            60,
    "list_project":          60,
    "scan_project":         120,
    "get_project_workspace": None,
    "open_skill":            None,
    "list_all_skills":       60,
    "list_projects":         60,
    "get_project_details":  120,
    "search_my_documents":  300,
    "recall_skill":         300,
    "search_web":           180,
    # Transcription is expensive on CPU — cache for 24h
    "transcribe_youtube":   86400,
    "watch_youtube_video":  86400,
}

# ── Context budget manager (upgrade #4) ──────────────────────
CONTEXT_BUDGET_ENABLED     = os.getenv("CONTEXT_BUDGET_ENABLED", "true").lower() == "true"
CONTEXT_BUDGET_KEEP_RECENT = int(os.getenv("CONTEXT_BUDGET_KEEP_RECENT", "2"))
CONTEXT_BUDGET_PREVIEW_CHARS = int(os.getenv("CONTEXT_BUDGET_PREVIEW_CHARS", "120"))

# ── Background job queue (upgrade #5) ────────────────────────
JOBS_ENABLED      = os.getenv("JOBS_ENABLED", "true").lower() == "true"
JOBS_DIR          = os.getenv("JOBS_DIR", "./jobs")

LONG_RUNNING_TOOLS = {
    "transcribe_youtube",
    "watch_youtube_video",
    "create_image",
    "create_3d_model",
    "index_my_documents",
    "execute_python",
}

JOB_SYNC_TIMEOUT = int(os.getenv("JOB_SYNC_TIMEOUT", "600"))

JOB_HISTORY_KEEP = int(os.getenv("JOB_HISTORY_KEEP", "50"))

# ── Image prompt rewriting (bonus #1) ────────────────────────
# Send the user's prompt through a small LLM first to make it vivid.
IMAGE_REWRITE_ENABLED   = os.getenv("IMAGE_REWRITE_ENABLED", "true").lower() == "true"
# Skip rewriting for prompts below this length (avoids noise).
IMAGE_REWRITE_MIN_CHARS = int(os.getenv("IMAGE_REWRITE_MIN_CHARS", "3"))
# Cap the rewritten prompt so it doesn't blow up the image API.
IMAGE_REWRITE_MAX_CHARS = int(os.getenv("IMAGE_REWRITE_MAX_CHARS", "500"))

# ── Multi-session projects ───────────────────────────────────
PROJECTS_FILE = "./projects.json"

# ── Memory (Chroma) ──────────────────────────────────────────
CHROMA_DIR            = "./chroma_db"
CHROMA_COLLECTION     = "my_ai_memory"
CHROMA_RAG_COLLECTION = "user_documents"

# ── UI ───────────────────────────────────────────────────────
VERBOSE = True