"""
agent.py — The Brain.

Multi-model agent with tool calling, memory, streaming, and a persistent
skill library. Exposes:
    - run_turn(user_input, messages) -> generator of events
    - sessions: in-memory dict for web sessions
    - main(): CLI loop
"""

import os
import sys
import traceback

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

try:
    from langchain_cerebras import ChatCerebras
    HAS_CEREBRAS = True
except ImportError:
    HAS_CEREBRAS = False

try:
    from langchain_cohere import ChatCohere
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

from tools import (
    youtube_to_text, web_search, generate_image, text_to_speech,
    run_python_code, set_workspace, get_workspace,
    write_project_file, read_project_file, list_project_files,
    scan_workspace, generate_3d_model, watch_video,
)

from skills import (
    save_skill, find_skills, get_skill, list_skills, delete_skill,
)

import config
from prompts import build_system_prompt

load_dotenv()


def log(msg: str) -> None:
    if config.VERBOSE:
        print(msg)


# ═══════════════════════════════════════════════════════════
#  1. MEMORY
# ═══════════════════════════════════════════════════════════
log("🧠 Loading memory...")
try:
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_DIR,
    )
    _memory_ok = True
except Exception as e:
    log(f"⚠️  Memory store failed to load: {e}")
    log("   → Continuing without long-term memory.")
    vectorstore = None
    _memory_ok = False


# ═══════════════════════════════════════════════════════════
#  2. BRAIN
#  Build an ordered list of models. LangChain's .with_fallbacks()
#  tries each one in turn if the previous fails at runtime — so we
#  never have to guess which model is "live" at any moment.
# ═══════════════════════════════════════════════════════════
log("🧠 Loading brain...")


def _try_build(name: str, factory):
    """Try to construct a chat model. Return None on any failure."""
    try:
        model = factory()
        log(f"   ✓ {name} ready")
        return model
    except Exception as e:
        log(f"   ✗ {name} failed: {type(e).__name__}: {e}")
        return None


def _openrouter_key():
    return os.getenv("OPENROUTER_API_KEY")


def build_model():
    """Return (primary, [fallbacks]) in priority order.

    Order:
      1. OpenRouter primary model
      2. Each OpenRouter fallback model
      3. Groq (network may be blocked — skip silently if so)
      4. Cerebras (trial may be expired — skip silently if so)
      5. Cohere (only if a key is set)

    At runtime, LangChain will try each in order if the previous fails.
    """
    models = []

    # 1 + 2. OpenRouter chain
    if _openrouter_key():
        or_models = [config.OPENROUTER_MODEL] + config.OPENROUTER_MODEL_FALLBACKS
        for model_id in or_models:
            m = _try_build(
                f"OpenRouter ({model_id})",
                lambda mid=model_id: ChatOpenAI(
                    model=mid,
                    openai_api_key=_openrouter_key(),
                    openai_api_base="https://openrouter.ai/api/v1",
                ),
            )
            if m:
                models.append(m)
    else:
        log("   ⚠️  OPENROUTER_API_KEY not set — skipping OpenRouter")

    # 3. Groq
    if os.getenv("GROQ_API_KEY"):
        m = _try_build(
            f"Groq ({config.GROQ_MODEL})",
            lambda: ChatGroq(model=config.GROQ_MODEL),
        )
        if m:
            models.append(m)

    # 4. Cerebras
    if HAS_CEREBRAS and os.getenv("CEREBRAS_API_KEY"):
        m = _try_build(
            f"Cerebras ({config.CEREBRAS_MODEL})",
            lambda: ChatCerebras(model=config.CEREBRAS_MODEL),
        )
        if m:
            models.append(m)

    # 5. Cohere
    if HAS_COHERE and os.getenv("COHERE_API_KEY"):
        m = _try_build(
            f"Cohere ({config.COHERE_MODEL})",
            lambda: ChatCohere(model=config.COHERE_MODEL),
        )
        if m:
            models.append(m)

    if not models:
        raise RuntimeError(
            "\n\n❌ No AI model could be loaded.\n"
            "   Open .env and make sure at least one of these is set:\n"
            "     • OPENROUTER_API_KEY  (free: https://openrouter.ai/keys)\n"
            "     • GROQ_API_KEY        (free: https://console.groq.com/keys)\n"
            "     • CEREBRAS_API_KEY    (free: https://cloud.cerebras.ai)\n"
        )

    log(f"🧠 Brain chain: {len(models)} model(s) queued")
    return models[0], models[1:]


_primary, _fallbacks = build_model()


# ═══════════════════════════════════════════════════════════
#  3. TOOLS — Local file tools
# ═══════════════════════════════════════════════════════════

@tool
def write_file(filename: str, content: str):
    """Writes a file to the ai_creations folder."""
    os.makedirs("ai_creations", exist_ok=True)
    filepath = os.path.join("ai_creations", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Wrote {filepath}"


@tool
def read_file(filename: str):
    """Reads a file from the ai_creations folder."""
    filepath = os.path.join("ai_creations", filename)
    if not os.path.exists(filepath):
        return f"❌ Not found: {filepath}"
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


@tool
def list_files():
    """Lists files in the ai_creations folder."""
    if not os.path.exists("ai_creations"):
        return "No files yet."
    return "Files: " + ", ".join(os.listdir("ai_creations"))


@tool
def save_memory(fact: str):
    """Saves an important fact to long-term memory."""
    if vectorstore is None:
        return "⚠️ Memory store unavailable — fact not saved."
    try:
        vectorstore.add_texts([fact])
        return f"💾 Remembered: {fact}"
    except Exception as e:
        return f"❌ Memory save failed: {e}"


@tool
def transcribe_youtube(url: str):
    """Transcribes audio from a YouTube video."""
    return youtube_to_text(url)


@tool
def watch_youtube_video(url: str, frame_interval: int = 10):
    """Watches a video by extracting frames and audio."""
    return watch_video(url, frame_interval=frame_interval)


@tool
def search_web(query: str):
    """Searches the web for current information."""
    return web_search(query)


@tool
def create_image(prompt: str):
    """Generates a 2D image from prompt."""
    return generate_image(prompt)


@tool
def create_3d_model(prompt: str):
    """Generates a 3D GLB model."""
    return generate_3d_model(prompt)


@tool
def speak(text: str, voice: str = "en-US-GuyNeural", rate: str = "+0%"):
    """Converts text to speech MP3."""
    return text_to_speech(text, voice=voice, rate=rate)


@tool
def execute_python(code: str):
    """Runs Python code in a local sandboxed subprocess."""
    return run_python_code(code)


@tool
def set_project_workspace(path: str):
    """Sets active project folder."""
    return set_workspace(path)


@tool
def get_project_workspace():
    """Returns active workspace path."""
    return get_workspace()


@tool
def scan_project():
    """Scans workspace for project stats."""
    return scan_workspace()


@tool
def write_project(relative_path: str, content: str):
    """Writes code file into active workspace."""
    return write_project_file(relative_path, content)


@tool
def read_project(relative_path: str):
    """Reads file from active workspace."""
    return read_project_file(relative_path)


@tool
def list_project(subpath: str = ""):
    """Lists files in workspace."""
    return list_project_files(subpath)


# ═══════════════════════════════════════════════════════════
#  4. TOOLS — Skill library
# ═══════════════════════════════════════════════════════════

@tool
def learn_skill(
    name: str,
    description: str,
    when_to_use: str,
    steps: str,
    code_template: str = "",
    tags: str = "",
):
    """Save a new skill to your permanent library.

    Use when: (1) you figure out a complex reusable procedure,
    (2) the user teaches you something, (3) you want to remember
    how you solved a problem.

    Args:
        name: Short snake_case unique name.
        description: One sentence — what does this skill do?
        when_to_use: Trigger condition.
        steps: Newline-separated numbered list.
        code_template: Optional reusable code snippet.
        tags: Comma-separated tags.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    return save_skill(name, description, when_to_use, steps, code_template, tag_list)


@tool
def recall_skill(query: str):
    """Search your skill library semantically.

    REQUIRED: You MUST pass a `query` string. Never call this without a query.
    Example: recall_skill(query="roblox kill brick")

    ALWAYS call this BEFORE starting a complex task.
    """
    return find_skills(query, k=3)


@tool
def open_skill(name: str):
    """Open a specific skill to see its full steps and code template."""
    return get_skill(name)


@tool
def list_all_skills():
    """List every skill you've ever learned."""
    return list_skills()


@tool
def forget_skill(name: str):
    """Permanently delete a skill from the library."""
    return delete_skill(name)


# ═══════════════════════════════════════════════════════════
#  5. ASSEMBLE
# ═══════════════════════════════════════════════════════════

tools = [
    write_file, read_file, list_files,
    save_memory,
    transcribe_youtube, watch_youtube_video,
    search_web, create_image, create_3d_model, speak,
    execute_python,
    set_project_workspace, get_project_workspace, scan_project,
    write_project, read_project, list_project,
    learn_skill, recall_skill, open_skill, list_all_skills, forget_skill,
]

_bound = _primary.bind_tools(tools)
if _fallbacks:
    model_with_tools = _bound.with_fallbacks(
        [m.bind_tools(tools) for m in _fallbacks]
    )
else:
    model_with_tools = _bound

log(f"✅ Brain ready! ({len(tools)} tools, {len(_fallbacks)} fallbacks)")


# ═══════════════════════════════════════════════════════════
#  6. CORE: run one turn as a stream of events
# ═══════════════════════════════════════════════════════════

sessions: dict = {}


def get_memory_context(user_input: str) -> str:
    if vectorstore is None:
        return "Memory unavailable."
    try:
        memories = vectorstore.similarity_search(user_input, k=config.MEMORY_K)
        return "\n".join(m.page_content for m in memories) if memories else "No memories yet."
    except Exception:
        return "Memory unavailable."


def run_tool(tool_call: dict) -> str:
    t_name = tool_call["name"]
    t_args = tool_call["args"]
    selected = next((t for t in tools if t.name == t_name), None)
    if selected is None:
        return f"❌ Unknown tool: {t_name}"
    try:
        return str(selected.invoke(t_args))
    except Exception as e:
        traceback.print_exc()
        return f"❌ Tool error: {e}"


def run_turn(user_input: str, messages: list):
    """Generator. Yields events for one user turn. Mutates `messages` in place.

    Events:
      {"type": "token",       "text": "..."}
      {"type": "tool_call",   "id": "...", "name": "...", "args": {...}}
      {"type": "tool_result", "id": "...", "preview": "..."}
      {"type": "error",       "message": "..."}
      {"type": "max_steps"}
      {"type": "done",        "answered": True/False}
    """
    try:
        workspace = get_workspace()
    except Exception:
        workspace = "Not set"

    system_prompt = build_system_prompt(workspace, get_memory_context(user_input))

    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_prompt)
    else:
        messages.insert(0, SystemMessage(content=system_prompt))

    messages.append(HumanMessage(content=user_input))

    for step in range(config.MAX_STEPS):
        aggregated = None
        try:
            for chunk in model_with_tools.stream(messages):
                aggregated = chunk if aggregated is None else aggregated + chunk
                if isinstance(chunk.content, str) and chunk.content:
                    yield {"type": "token", "text": chunk.content}
        except Exception as e:
            traceback.print_exc()
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
            return

        if aggregated is None:
            yield {"type": "error", "message": "Empty response from model."}
            return

        messages.append(aggregated)

        if not aggregated.tool_calls:
            yield {"type": "done", "answered": True}
            return

        for tc in aggregated.tool_calls:
            yield {
                "type": "tool_call",
                "id": tc["id"],
                "name": tc["name"],
                "args": tc["args"],
            }
            result = run_tool(tc)
            yield {"type": "tool_result", "id": tc["id"], "preview": str(result)[:300]}
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    yield {"type": "max_steps"}
    yield {"type": "done", "answered": False}


# ═══════════════════════════════════════════════════════════
#  7. CLI LOOP
# ═══════════════════════════════════════════════════════════

def main() -> None:
    messages: list = []
    print("\n🤖 SUPER AGENT ready!  (type 'exit' to quit)\n")
    if not _memory_ok:
        print("⚠️  Memory disabled — check the log above.\n")

    while True:
        try:
            user_input = input("🧑 You: ")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if user_input.lower() in ("exit", "quit", "bye"):
            print("👋 Goodbye!")
            break
        if not user_input.strip():
            continue

        print("🤖 AI: ", end="", flush=True)

        try:
            for event in run_turn(user_input, messages):
                t = event["type"]
                if t == "token":
                    print(event["text"], end="", flush=True)
                elif t == "tool_call":
                    print(f"\n🛠️  {event['name']}({list(event['args'].keys())})", flush=True)
                elif t == "tool_result":
                    print(f"   ↳ {event['preview']}", flush=True)
                elif t == "error":
                    print(f"\n❌ {event['message']}", flush=True)
                elif t == "max_steps":
                    print("\n⚠️  Reached max tool steps.", flush=True)
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted.", flush=True)
        except Exception as e:
            print(f"\n❌ Unexpected error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

        print("\n")


if __name__ == "__main__":
    main()