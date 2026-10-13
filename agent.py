"""
agent.py — The Brain.

Multi-model agent with tool calling, memory, streaming, and a persistent
skill library.

Manual fallback chain: if a model errors BEFORE yielding any token,
the next model is tried.
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

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from langchain_cerebras import ChatCerebras
    HAS_CEREBRAS = True
except ImportError:
    HAS_CEREBRAS = False

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


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def log(msg: str) -> None:
    if config.VERBOSE:
        print(msg)


def truncate_tool_result(result: str, max_chars: int = 4000) -> str:
    """Trim overly long tool results before adding to context.

    Prevents a single transcription or search from blowing the token budget.
    Preserves the head; notes how much was cut.
    """
    if result is None:
        return ""
    s = str(result)
    if len(s) <= max_chars:
        return s
    omitted = len(s) - max_chars
    return s[:max_chars] + f"\n\n... (truncated, {omitted:,} more chars omitted)"


def extract_usage(msg):
    """Extract token usage from an aggregated message, if available."""
    try:
        um = getattr(msg, "usage_metadata", None)
        if um and isinstance(um, dict):
            return {
                "input": um.get("input_tokens", 0),
                "output": um.get("output_tokens", 0),
                "total": um.get("total_tokens", 0),
            }
    except Exception:
        pass
    try:
        rm = getattr(msg, "response_metadata", None)
        if rm and isinstance(rm, dict):
            tu = rm.get("token_usage") or rm.get("usage") or {}
            if tu:
                return {
                    "input": tu.get("prompt_tokens", 0),
                    "output": tu.get("completion_tokens", 0),
                    "total": tu.get("total_tokens", 0),
                }
    except Exception:
        pass
    return None


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
#  2. TOOLS
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


@tool
def learn_skill(
    name: str,
    description: str,
    when_to_use: str,
    steps: str,
    code_template: str = "",
    tags: str = "",
):
    """Save a new skill to your permanent library."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    return save_skill(name, description, when_to_use, steps, code_template, tag_list)


@tool
def recall_skill(query: str):
    """Search your skill library semantically.

    REQUIRED: You MUST pass a `query` string. Never call this without a query.
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


# ═══════════════════════════════════════════════════════════
#  3. BUILD MODEL CHAIN
# ═══════════════════════════════════════════════════════════

log("🧠 Loading brain...")


def _try_build(name: str, factory):
    try:
        model = factory()
        log(f"   ✓ {name} ready")
        return model
    except Exception as e:
        log(f"   ✗ {name} failed: {type(e).__name__}: {str(e)[:100]}")
        return None


def _openrouter_key():
    return os.getenv("OPENROUTER_API_KEY")


def _mistral_key():
    return os.getenv("MISTRAL_API_KEY")


def build_model_chain():
    """Return list of (name, bound_model) in priority order."""
    raw_models = []

    # ── 1. Mistral (if key present — free tier, no country wall) ──
    mk = _mistral_key()
    if mk and mk.strip():
        try:
            from langchain_mistralai import ChatMistralAI
            for model_id in [config.MISTRAL_MODEL] + config.MISTRAL_MODEL_FALLBACKS:
                m = _try_build(
                    f"Mistral ({model_id})",
                    lambda mid=model_id, key=mk: ChatMistralAI(
                        model=mid, api_key=key, timeout=60,
                    ),
                )
                if m:
                    raw_models.append((f"Mistral/{model_id}", m))
        except ImportError:
            log("   ⚠️  MISTRAL_API_KEY set but langchain-mistralai not installed")

    # ── 2. OpenRouter ────────────────────────────────────────
    if _openrouter_key():
        or_models = [config.OPENROUTER_MODEL] + config.OPENROUTER_MODEL_FALLBACKS
        for model_id in or_models:
            m = _try_build(
                f"OpenRouter ({model_id})",
                lambda mid=model_id: ChatOpenAI(
                    model=mid,
                    openai_api_key=_openrouter_key(),
                    openai_api_base="https://openrouter.ai/api/v1",
                    timeout=60,
                ),
            )
            if m:
                raw_models.append((f"OpenRouter/{model_id}", m))

    # ── 3. Pollinations (no key) ─────────────────────────────
    if config.POLLINATIONS_ENABLED:
        poll_models = [config.POLLINATIONS_MODEL] + config.POLLINATIONS_MODEL_FALLBACKS
        for model_id in poll_models:
            m = _try_build(
                f"Pollinations ({model_id})",
                lambda mid=model_id: ChatOpenAI(
                    model=mid,
                    openai_api_key="pollinations-no-key",
                    openai_api_base="https://text.pollinations.ai/openai",
                    timeout=90,
                ),
            )
            if m:
                raw_models.append((f"Pollinations/{model_id}", m))

    # ── 4. Groq (if ever fixed) ──────────────────────────────
    if HAS_GROQ and os.getenv("GROQ_API_KEY"):
        m = _try_build(
            "Groq (llama-3.3-70b-versatile)",
            lambda: ChatGroq(model="llama-3.3-70b-versatile"),
        )
        if m:
            raw_models.append(("Groq/llama-3.3-70b", m))

    # ── 5. Cerebras (if ever paid) ───────────────────────────
    if HAS_CEREBRAS and os.getenv("CEREBRAS_API_KEY"):
        m = _try_build(
            "Cerebras (gpt-oss-120b)",
            lambda: ChatCerebras(model="gpt-oss-120b"),
        )
        if m:
            raw_models.append(("Cerebras/gpt-oss-120b", m))

    if not raw_models:
        raise RuntimeError(
            "\n\n❌ No AI model could be loaded.\n"
            "   Open .env and set at least one of:\n"
            "     • OPENROUTER_API_KEY   (free: https://openrouter.ai/keys)\n"
            "     • MISTRAL_API_KEY      (free: https://console.mistral.ai/)\n"
        )

    bound = []
    for name, m in raw_models:
        try:
            bound.append((name, m.bind_tools(tools)))
        except Exception as e:
            log(f"   ⚠️  {name} can't bind tools ({e}) — adding without tools")
            bound.append((name + " [no tools]", m))

    return bound


_model_chain = build_model_chain()
log(f"🧠 Brain chain: {len(_model_chain)} model(s) queued")
log(f"✅ Brain ready! ({len(tools)} tools, {len(_model_chain)} models)")


# ═══════════════════════════════════════════════════════════
#  4. CORE
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
        result = selected.invoke(t_args)
        return truncate_tool_result(str(result), max_chars=config.TOOL_RESULT_MAX_CHARS)
    except Exception as e:
        traceback.print_exc()
        return f"❌ Tool error: {e}"


def _stream_with_fallback(messages: list):
    """Try each model in the chain until one streams successfully.

    Yields:
      {"type": "token", "text": "..."}
      {"type": "usage", "input": N, "output": N, "total": N}

    Returns the aggregated message via StopIteration.value.
    """
    errors = []

    for name, model in _model_chain:
        has_tool_history = any(
            isinstance(m, ToolMessage) or
            (isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
            for m in messages
        )
        if has_tool_history and "[no tools]" in name:
            log(f"   ⏭️  Skipping {name} (tool history present)")
            continue

        aggregated = None
        yielded_any = False

        try:
            log(f"   → trying {name}")
            for chunk in model.stream(messages):
                aggregated = chunk if aggregated is None else aggregated + chunk
                if isinstance(chunk.content, str) and chunk.content:
                    yielded_any = True
                    yield {"type": "token", "text": chunk.content}

            # Extract usage if available
            if aggregated is not None:
                usage = extract_usage(aggregated)
                if usage:
                    yield {"type": "usage", **usage}

            return aggregated

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            errors.append(f"{name}: {err_msg}")

            if yielded_any:
                log(f"   ✗ {name} failed mid-stream, giving up")
                yield {"type": "error", "message": f"{name} failed mid-stream: {err_msg}"}
                return None

            log(f"   ✗ {name} failed: {err_msg[:100]} — trying next")

    yield {
        "type": "error",
        "message": "All models failed. Last 3 errors:\n" + "\n".join(errors[-3:]),
    }
    return None


def run_turn(user_input: str, messages: list):
    """Generator. Yields events for one user turn. Mutates `messages` in place.

    Events:
      {"type": "token",       "text": "..."}
      {"type": "tool_call",   "id": "...", "name": "...", "args": {...}}
      {"type": "tool_result", "id": "...", "preview": "..."}
      {"type": "usage",       "input": N, "output": N, "total": N}
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

    total_input = 0
    total_output = 0

    for step in range(config.MAX_STEPS):
        aggregated = None
        try:
            gen = _stream_with_fallback(messages)
            while True:
                try:
                    event = next(gen)
                    if event.get("type") == "usage":
                        total_input += event.get("input", 0)
                        total_output += event.get("output", 0)
                    yield event
                except StopIteration as si:
                    aggregated = si.value
                    break
        except Exception as e:
            yield {"type": "error", "message": f"Model chain error: {e}"}
            return

        if aggregated is None:
            return

        messages.append(aggregated)

        if not aggregated.tool_calls:
            if total_input or total_output:
                yield {
                    "type": "usage_total",
                    "input": total_input,
                    "output": total_output,
                    "total": total_input + total_output,
                }
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
#  5. CLI LOOP
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

        turn_usage = None

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
                elif t == "usage_total":
                    turn_usage = event
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted.", flush=True)
        except Exception as e:
            print(f"\n❌ Unexpected error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

        if turn_usage:
            print(
                f"\n   [tokens: {turn_usage['input']} in / "
                f"{turn_usage['output']} out / "
                f"{turn_usage['total']} total]",
                flush=True,
            )

        print("\n")


if __name__ == "__main__":
    main()