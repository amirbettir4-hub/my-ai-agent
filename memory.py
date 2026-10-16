"""memory.py — auto-memory, session summaries, and context budget.

Upgrades bundled here:
  #1 — auto-memory extraction + rolling session summary
  #4 — compact_tool_results: replace old ToolMessages with short previews
       before sending the conversation to the LLM.
"""

import re
from typing import Callable, Optional

import config
from langchain_core.messages import ToolMessage


LLMCall = Callable[[str, str], Optional[str]]

_session_state: dict = {}


# ── Heuristic gate: only extract when the user said something personal ──

_PERSONAL_PATTERNS = [
    r"\bi am\b", r"\bi'm\b", r"\bmy name\b", r"\bi like\b", r"\bi love\b",
    r"\bi hate\b", r"\bi prefer\b", r"\bi want\b", r"\bi need\b",
    r"\bremember\b", r"\bnote that\b", r"\bkeep in mind\b",
    r"\bmy (project|goal|dream|plan|pc|computer|setup|stack|name|age|country|city)\b",
    r"\bi'?m (working|building|making|learning|studying|from|using)\b",
    r"\bi live\b", r"\bi use\b", r"\bi don'?t\b", r"\bi can'?t\b",
    r"\bdon'?t forget\b", r"\balways\b", r"\bnever\b",
    r"\bfor future\b", r"\bfrom now on\b",
    r"\bmy (os|editor|language|framework|hardware|ram|cpu|gpu)\b",
]
_PERSONAL_RE = re.compile("|".join(_PERSONAL_PATTERNS), re.IGNORECASE)


def should_extract(user_msg: str) -> bool:
    if not user_msg:
        return False
    if len(user_msg) < config.AUTO_MEMORY_MIN_CHARS:
        return False
    return bool(_PERSONAL_RE.search(user_msg))


# ── Quality gate for extracted facts ─────────────────────────

_SUBJECT_RE = re.compile(
    r"\b(user|i|me|my|mine|he|she|they|them|their)\b",
    re.IGNORECASE,
)


def _is_well_formed_fact(s: str) -> bool:
    if not s:
        return False
    if not _SUBJECT_RE.search(s):
        return False
    if len(s.split()) < 3:
        return False
    return True


# ── Fact extraction ──────────────────────────────────────────

def extract_facts(user_msg: str, assistant_msg: str, llm_call: LLMCall) -> list:
    from prompts import MEMORY_EXTRACTION_SYSTEM

    combined = (
        f"USER: {user_msg.strip()[:1200]}\n\n"
        f"ASSISTANT: {(assistant_msg or '').strip()[:800]}"
    )
    out = llm_call(MEMORY_EXTRACTION_SYSTEM, combined)
    if not out:
        return []

    text = str(out).strip()
    if not text or text.upper().startswith("NONE"):
        return []

    facts = []
    for line in text.splitlines():
        s = line.strip().lstrip("-*•0123456789.) ").strip().strip('"').strip("'")
        if not s or s.upper() == "NONE":
            continue
        if len(s) < 12 or len(s) > 300:
            continue
        if not _is_well_formed_fact(s):
            continue
        facts.append(s)
        if len(facts) >= config.AUTO_MEMORY_MAX_FACTS:
            break
    return facts


def save_facts(vectorstore, facts: list) -> int:
    if vectorstore is None or not facts:
        return 0
    saved = 0
    for f in facts:
        try:
            existing = vectorstore.similarity_search_with_score(f, k=1)
            if existing:
                _, score = existing[0]
                if score < config.MEMORY_DEDUP_DISTANCE:
                    continue
            vectorstore.add_texts([f], metadatas=[{"source": "auto_memory"}])
            saved += 1
        except Exception:
            pass
    return saved


# ── Session summaries ────────────────────────────────────────

def get_session_summary(sid: Optional[str]) -> str:
    if not sid:
        return ""
    return _session_state.get(sid, {}).get("summary", "")


def get_summarized_count(sid: Optional[str]) -> int:
    if not sid:
        return 0
    return _session_state.get(sid, {}).get("summarized_count", 0)


def clear_session(sid: Optional[str]) -> None:
    if sid:
        _session_state.pop(sid, None)


def _flatten_content(content) -> str:
    if isinstance(content, list):
        try:
            return " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        except Exception:
            return str(content)
    return str(content)


def maybe_summarize(messages: list, llm_call: LLMCall,
                    sid: Optional[str]) -> int:
    if not sid or not config.SESSION_SUMMARY_ENABLED:
        return 0

    n = len(messages)
    if n < config.SESSION_SUMMARY_TRIGGER:
        return 0

    from langchain_core.messages import HumanMessage

    prev_count = get_summarized_count(sid)
    skip_start = 1 + prev_count
    keep_recent = config.SESSION_SUMMARY_KEEP_RECENT
    target_cut = n - keep_recent

    if target_cut <= skip_start:
        return 0

    cut = target_cut
    while cut > skip_start and not isinstance(messages[cut], HumanMessage):
        cut -= 1
    if cut <= skip_start:
        return 0

    chunk = messages[skip_start:cut]
    if not chunk:
        return 0

    lines = []
    for m in chunk:
        role = getattr(m, "type", "?")
        content = _flatten_content(getattr(m, "content", "")).strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content[:900]}")
    chunk_text = "\n".join(lines)[:6000]

    prior = get_session_summary(sid)

    if len(chunk_text) < 150:
        _session_state[sid] = {
            "summary": prior,
            "summarized_count": prev_count + len(chunk),
        }
        return len(chunk)

    from prompts import SESSION_SUMMARY_SYSTEM

    prompt = chunk_text
    if prior:
        prompt = (
            f"PREVIOUS SUMMARY:\n{prior}\n\n"
            f"NEW MESSAGES TO ADD:\n{chunk_text}"
        )

    out = llm_call(SESSION_SUMMARY_SYSTEM, prompt)
    new_summary = (out or "").strip()

    if not new_summary or new_summary.upper().startswith("NONE"):
        new_summary = prior or "(earlier messages omitted)"

    if len(new_summary) > config.SESSION_SUMMARY_MAX_CHARS:
        new_summary = new_summary[:config.SESSION_SUMMARY_MAX_CHARS] + "…"

    _session_state[sid] = {
        "summary": new_summary,
        "summarized_count": prev_count + len(chunk),
    }
    return len(chunk)


# ═══════════════════════════════════════════════════════════
#  UPGRADE #4 — context budget: compact old ToolMessages
# ═══════════════════════════════════════════════════════════

_COMPACT_MARKER = "[old tool result"


def compact_tool_results(convo: list, keep_recent: int = 2,
                         preview_chars: int = 120) -> list:
    """Return a new list where every ToolMessage except the last N has its
    content replaced with a short summary. Non-ToolMessages pass through
    unchanged. Safe to call every step — already-compacted messages are
    left alone.

    The original objects are never mutated; new ToolMessage instances are
    created for the compacted ones.
    """
    if not convo:
        return convo

    tool_idx = [i for i, m in enumerate(convo) if isinstance(m, ToolMessage)]
    if len(tool_idx) <= keep_recent:
        return convo

    to_compact = set(tool_idx[:-keep_recent])
    out = []
    for i, m in enumerate(convo):
        if i not in to_compact:
            out.append(m)
            continue

        content = m.content if isinstance(m.content, str) else str(m.content)

        # Already compacted → pass through (avoids re-summarizing a summary)
        if content.lstrip().startswith(_COMPACT_MARKER):
            out.append(m)
            continue

        preview = content[:preview_chars].replace("\n", " ").strip()
        summary = (
            f"{_COMPACT_MARKER}, {len(content)} chars — preview: "
            f"{preview}...]"
        )
        out.append(ToolMessage(content=summary, tool_call_id=m.tool_call_id))

    return out