"""
agent.py — The Brain (Smart Edition).
"""

import os
import re
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
import rag
import projects as proj
import config
from prompts import (
    build_system_prompt, PLANNING_SYSTEM, REFLECTION_SYSTEM,
)

load_dotenv()


def log(msg: str) -> None:
    if config.VERBOSE:
        print(msg)


def truncate_tool_result(result: str, max_chars: int = None) -> str:
    max_chars = max_chars or config.TOOL_RESULT_MAX_CHARS
    if result is None:
        return ""
    s = str(result)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n\n... (truncated, {len(s)-max_chars:,} more chars omitted)"


def extract_usage(msg):
    try:
        um = getattr(msg, "usage_metadata", None)
        if um and isinstance(um, dict):
            return {"input": um.get("input_tokens", 0),
                    "output": um.get("output_tokens", 0),
                    "total": um.get("total_tokens", 0)}
    except Exception:
        pass
    try:
        rm = getattr(msg, "response_metadata", None)
        if rm and isinstance(rm, dict):
            tu = rm.get("token_usage") or rm.get("usage") or {}
            if tu:
                return {"input": tu.get("prompt_tokens", 0),
                        "output": tu.get("completion_tokens", 0),
                        "total": tu.get("total_tokens", 0)}
    except Exception:
        pass
    return None


# ═══ MEMORY ═══
log("🧠 Loading memory...")
try:
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_DIR,
    )
    _memory_ok = True
except Exception as e:
    log(f"⚠️  Memory failed: {e}")
    vectorstore = None
    _memory_ok = False


# ═══ TOOLS ═══

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
        return "⚠️ Memory unavailable."
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
def learn_skill(name: str, description: str, when_to_use: str,
                steps: str, code_template: str = "", tags: str = ""):
    """Save a new skill to your permanent library."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    return save_skill(name, description, when_to_use, steps, code_template, tag_list)


@tool
def recall_skill(query: str):
    """Search your skill library semantically. MUST pass a query."""
    return find_skills(query, k=3)


@tool
def open_skill(name: str):
    """Open a specific skill."""
    return get_skill(name)


@tool
def list_all_skills():
    """List every skill you've ever learned."""
    return list_skills()


@tool
def forget_skill(name: str):
    """Delete a skill."""
    return delete_skill(name)


@tool
def index_my_documents(force: bool = False):
    """Index files in ./documents/ for local search."""
    return rag.index_documents(force=force)


@tool
def search_my_documents(query: str):
    """Search inside your indexed documents."""
    return rag.search_documents(query)


@tool
def create_project(name: str, goal: str = ""):
    """Create a multi-session project."""
    return proj.create_project(name, goal)


@tool
def list_projects(include_done: bool = False):
    """List all projects."""
    return proj.list_projects(include_done=include_done)


@tool
def get_project_details(project_id: str):
    """Get project details."""
    return proj.get_project(project_id)


@tool
def update_project_notes(project_id: str, notes: str = None,
                         status: str = None, goal: str = None):
    """Update project fields."""
    return proj.update_project(project_id, notes=notes, status=status, goal=goal)


@tool
def add_project_step(project_id: str, text: str):
    """Add a step to a project."""
    return proj.add_project_step(project_id, text)


@tool
def complete_project_step(project_id: str, step_number: int):
    """Mark a step done."""
    return proj.complete_project_step(project_id, step_number)


@tool
def delete_project(project_id: str):
    """Delete a project."""
    return proj.delete_project(project_id)


tools = [
    write_file, read_file, list_files,
    save_memory,
    transcribe_youtube, watch_youtube_video,
    search_web, create_image, create_3d_model, speak,
    execute_python,
    set_project_workspace, get_project_workspace, scan_project,
    write_project, read_project, list_project,
    learn_skill, recall_skill, open_skill, list_all_skills, forget_skill,
    index_my_documents, search_my_documents,
    create_project, list_projects, get_project_details,
    update_project_notes, add_project_step, complete_project_step,
    delete_project,
]


# ═══ BRAIN ═══
log("🧠 Loading brain...")


def _try_build(name, factory):
    try:
        m = factory()
        log(f"   ✓ {name} ready")
        return m
    except Exception as e:
        log(f"   ✗ {name} failed: {type(e).__name__}: {str(e)[:100]}")
        return None


def _openrouter_key():
    return os.getenv("OPENROUTER_API_KEY")


def _mistral_key():
    return os.getenv("MISTRAL_API_KEY")


def build_model_chain():
    raw = []

    mk = _mistral_key()
    if mk and mk.strip():
        try:
            from langchain_mistralai import ChatMistralAI
            for model_id in [config.MISTRAL_MODEL] + config.MISTRAL_MODEL_FALLBACKS:
                m = _try_build(
                    f"Mistral ({model_id})",
                    lambda mid=model_id, key=mk: ChatMistralAI(
                        model=mid, api_key=key, timeout=60),
                )
                if m:
                    raw.append((f"Mistral/{model_id}", m))
        except ImportError:
            log("   ⚠️  langchain-mistralai not installed")

    if _openrouter_key():
        for model_id in [config.OPENROUTER_MODEL] + config.OPENROUTER_MODEL_FALLBACKS:
            m = _try_build(
                f"OpenRouter ({model_id})",
                lambda mid=model_id: ChatOpenAI(
                    model=mid,
                    openai_api_key=_openrouter_key(),
                    openai_api_base="https://openrouter.ai/api/v1",
                    timeout=60),
            )
            if m:
                raw.append((f"OpenRouter/{model_id}", m))

    if config.POLLINATIONS_ENABLED:
        for model_id in [config.POLLINATIONS_MODEL] + config.POLLINATIONS_MODEL_FALLBACKS:
            m = _try_build(
                f"Pollinations ({model_id})",
                lambda mid=model_id: ChatOpenAI(
                    model=mid,
                    openai_api_key="pollinations-no-key",
                    openai_api_base="https://text.pollinations.ai/openai",
                    timeout=90),
            )
            if m:
                raw.append((f"Pollinations/{model_id}", m))

    if HAS_GROQ and os.getenv("GROQ_API_KEY"):
        m = _try_build("Groq (llama-3.3-70b-versatile)",
                       lambda: ChatGroq(model="llama-3.3-70b-versatile"))
        if m:
            raw.append(("Groq/llama-3.3-70b", m))

    if HAS_CEREBRAS and os.getenv("CEREBRAS_API_KEY"):
        m = _try_build("Cerebras (gpt-oss-120b)",
                       lambda: ChatCerebras(model="gpt-oss-120b"))
        if m:
            raw.append(("Cerebras/gpt-oss-120b", m))

    if not raw:
        raise RuntimeError("❌ No AI model could be loaded. Check .env")

    bound = []
    for name, m in raw:
        try:
            bound.append((name, m.bind_tools(tools)))
        except Exception:
            bound.append((name + " [no tools]", m))
    return bound, raw


_model_chain, _raw_model_chain = build_model_chain()
log(f"🧠 Brain chain: {len(_model_chain)} tool-capable + {len(_raw_model_chain)} raw")


# ═══ MODE DETECTION ═══

RESEARCH_VERBS = [
    "search", "find", "look up", "lookup", "research",
    "google", "search for", "find me", "what's new",
]
CODER_KW = ["code", "script", "function", "bug", "debug", "implement",
            "compile", "refactor", "algorithm", "syntax"]
RESEARCHER_KW = ["what is", "what's", "explain", "compare", "latest",
                 "news", "how does", "tell me about", "who is", "tutorial"]
PLANNER_KW = ["plan", "steps", "strategy", "roadmap", "break down",
              "how to build", "how do i"]


def detect_mode(user_input: str) -> str:
    text = user_input.lower().strip()
    # Priority 1: any research verb anywhere in the first 60 chars
    head = text[:60]
    if any(v in head for v in RESEARCH_VERBS):
        return "researcher"

    c = sum(1 for k in CODER_KW if k in text)
    r = sum(1 for k in RESEARCHER_KW if k in text)
    p = sum(1 for k in PLANNER_KW if k in text)

    if p > 0 and p >= c and p >= r:
        return "planner"
    if c > r and c > 0:
        return "coder"
    if r > 0:
        return "researcher"
    return "general"


# ═══ FORCED TOOL DETECTION ═══
# If the message clearly wants a specific tool, we inject a system message
# demanding the model call it. This prevents the "answer from memory" bug.

FORCED_TOOL_PATTERNS = [
    (["search", "google", "look up", "find online", "research"],
     "search_web",
     "The user wants a web search. You MUST call `search_web` before answering. "
     "Do NOT answer from memory. Do NOT invent URLs."),
    (["search my documents", "search my docs", "search my files"],
     "search_my_documents",
     "The user wants to search their local documents. You MUST call `search_my_documents`."),
    (["transcribe"],
     "transcribe_youtube",
     "The user wants a transcription. You MUST call `transcribe_youtube` with the URL."),
    (["generate image", "create image", "draw", "make an image"],
     "create_image",
     "The user wants an image. You MUST call `create_image`."),
]


def detect_forced_tool(user_input: str):
    """Return (tool_name, reminder) if the message clearly needs a tool."""
    text = user_input.lower()
    for keywords, tool_name, reminder in FORCED_TOOL_PATTERNS:
        if any(k in text for k in keywords):
            return tool_name, reminder
    return None, None


# ═══ SMART STAGES ═══

ACTION_VERBS = ["build", "create", "make", "implement", "design",
                "develop", "set up", "setup", "write a", "generate",
                "produce", "fix"]


def needs_planning(user_input: str) -> bool:
    text = user_input.lower()
    if len(user_input) < config.PLANNING_MIN_CHARS:
        if not any(v in text for v in ACTION_VERBS):
            return False
    has_action = any(v in text for v in ACTION_VERBS)
    is_long = len(user_input) > 120
    has_multi = (text.count(" and ") >= 1 or text.count(",") >= 2
                 or text.count("\n") >= 1)
    return has_action or is_long or has_multi


def _quick_call(system_prompt: str, user_prompt: str) -> str | None:
    msgs = [SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)]
    for name, model in _raw_model_chain:
        try:
            log(f"      one-shot via {name}")
            r = model.invoke(msgs)
            content = getattr(r, "content", r)
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            return str(content).strip()
        except Exception as e:
            log(f"      ✗ {name}: {type(e).__name__}")
            continue
    return None


def _extract_plan(text: str) -> list[str]:
    if not text:
        return []
    steps = []
    for line in text.splitlines():
        m = re.match(r'^\s*(?:\d+[\.\)]|[\-\*])\s+(.+)$', line.strip())
        if m:
            steps.append(m.group(1).strip())
    return steps[:6]


sessions: dict = {}


def get_memory_context(user_input: str) -> str:
    if vectorstore is None:
        return "unavailable"
    try:
        memories = vectorstore.similarity_search(user_input, k=config.MEMORY_K)
        return "\n".join(m.page_content for m in memories) if memories else "none"
    except Exception:
        return "unavailable"


def run_tool(tool_call: dict) -> str:
    t_name = tool_call["name"]
    t_args = tool_call["args"]
    selected = next((t for t in tools if t.name == t_name), None)
    if selected is None:
        return f"❌ Unknown tool: {t_name}"
    try:
        result = selected.invoke(t_args)
        return truncate_tool_result(str(result))
    except Exception as e:
        traceback.print_exc()
        return f"❌ Tool error: {e}"


def _stream_with_fallback(messages: list):
    errors = []
    for name, model in _model_chain:
        has_tool_history = any(
            isinstance(m, ToolMessage) or
            (isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
            for m in messages
        )
        if has_tool_history and "[no tools]" in name:
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
            if aggregated is not None:
                usage = extract_usage(aggregated)
                if usage:
                    yield {"type": "usage", **usage}
            return aggregated
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            errors.append(f"{name}: {err_msg}")
            if yielded_any:
                yield {"type": "error", "message": f"{name} failed mid-stream: {err_msg}"}
                return None
            log(f"   ✗ {name}: {err_msg[:100]}")

    yield {"type": "error",
           "message": "All models failed. Last 3 errors:\n" + "\n".join(errors[-3:])}
    return None


def run_turn(user_input: str, messages: list):
    try:
        workspace = get_workspace()
    except Exception:
        workspace = "Not set"

    mode = "general"
    if config.SMART_SUBAGENTS:
        mode = detect_mode(user_input)
        yield {"type": "mode", "mode": mode}

    system_prompt = build_system_prompt(workspace, get_memory_context(user_input), mode=mode)
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_prompt)
    else:
        messages.insert(0, SystemMessage(content=system_prompt))

    # ── FORCED TOOL INJECTION ────────────────────────────────
    forced_tool, reminder = detect_forced_tool(user_input)
    if forced_tool:
        messages.append(SystemMessage(
            content=f"⚠️ ACTION REQUIRED: {reminder}\n"
                    f"Call `{forced_tool}` NOW before generating any final answer."
        ))
        yield {"type": "status", "text": f"🎯 forcing tool: {forced_tool}"}

    messages.append(HumanMessage(content=user_input))

    # Planning
    if config.SMART_PLANNING and needs_planning(user_input) and not forced_tool:
        yield {"type": "status", "text": "🧠 Planning..."}
        plan_text = _quick_call(PLANNING_SYSTEM, user_input)
        steps = _extract_plan(plan_text or "")
        if steps:
            yield {"type": "plan", "steps": steps}
            messages.append(SystemMessage(
                content="Plan:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
            ))

    total_in = 0
    total_out = 0
    final_text = ""
    hit_max = True
    any_tool_used = False

    for step in range(config.MAX_STEPS):
        aggregated = None
        try:
            gen = _stream_with_fallback(messages)
            while True:
                try:
                    ev = next(gen)
                    if ev.get("type") == "usage":
                        total_in += ev.get("input", 0)
                        total_out += ev.get("output", 0)
                    yield ev
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
            final_text = aggregated.content if isinstance(aggregated.content, str) else ""
            hit_max = False
            break

        for tc in aggregated.tool_calls:
            any_tool_used = True
            yield {"type": "tool_call", "id": tc["id"],
                   "name": tc["name"], "args": tc["args"]}
            result = run_tool(tc)
            yield {"type": "tool_result", "id": tc["id"], "preview": str(result)[:300]}
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    if hit_max:
        yield {"type": "max_steps"}

    # Reflection — only on pure-chat turns (no tool ran)
    if (config.SMART_REFLECTION
            and final_text
            and not any_tool_used
            and len(final_text) >= config.REFLECTION_MIN_CHARS
            and "❌" not in final_text):
        yield {"type": "status", "text": "🔍 Reflecting..."}
        critique = _quick_call(
            REFLECTION_SYSTEM,
            f"Q: {user_input}\n\nA: {final_text}",
        )
        if critique:
            c = critique.strip()
            if c.upper() not in ("OK", "OK.", "OKAY") and len(c) > 10:
                first_line = c.split("\n")[0].strip()
                yield {"type": "reflection", "text": first_line}

    if total_in or total_out:
        yield {"type": "usage_total", "input": total_in,
               "output": total_out, "total": total_in + total_out}

    yield {"type": "done", "answered": not hit_max}


def main() -> None:
    messages: list = []
    print("\n🤖 SUPER AGENT (Smart) ready!  (type 'exit' to quit)\n")
    if not _memory_ok:
        print("⚠️  Memory disabled.\n")

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
                elif t == "mode":
                    if event["mode"] != "general":
                        print(f"\n🎯 mode: {event['mode']}", flush=True)
                elif t == "status":
                    print(f"\n{event['text']}", flush=True)
                elif t == "plan":
                    print("📋 Plan:", flush=True)
                    for i, s in enumerate(event["steps"], 1):
                        print(f"   {i}. {s}", flush=True)
                elif t == "tool_call":
                    print(f"\n🛠️  {event['name']}({list(event['args'].keys())})", flush=True)
                elif t == "tool_result":
                    print(f"   ↳ {event['preview']}", flush=True)
                elif t == "reflection":
                    print(f"\n💭 {event['text']}", flush=True)
                elif t == "error":
                    print(f"\n❌ {event['message']}", flush=True)
                elif t == "max_steps":
                    print("\n⚠️  Reached max tool steps.", flush=True)
                elif t == "usage_total":
                    turn_usage = event
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted.", flush=True)
        except Exception as e:
            print(f"\n❌ {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

        if turn_usage:
            print(f"\n   [tokens: {turn_usage['input']} in / "
                  f"{turn_usage['output']} out / {turn_usage['total']} total]", flush=True)
        print("\n")


if __name__ == "__main__":
    main()