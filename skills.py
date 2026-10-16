"""skills.py — persistent skill library backed by Chroma + JSON.

Upgrade #3:
  - Auto-extract reusable procedures from successful multi-tool turns.
  - Dedup against existing skills on save.
  - Return scored matches so the agent can gate what it injects.
"""

import os
import re
import json
from datetime import datetime
from typing import Callable, Optional

from langchain_chroma import Chroma

import config

SKILLS_DIR = "./skills_db"
SKILLS_JSON = "./skills.json"
COLLECTION = "agent_skills"


vectorstore = Chroma(
    collection_name=COLLECTION,
    persist_directory=SKILLS_DIR,
)


LLMCall = Callable[[str, str], Optional[str]]


# ═══════════════════════════════════════════════════════════
#  Persistence
# ═══════════════════════════════════════════════════════════

def _load_all() -> dict:
    if not os.path.exists(SKILLS_JSON):
        return {}
    try:
        with open(SKILLS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all(data: dict) -> None:
    with open(SKILLS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════
#  Save / read
# ═══════════════════════════════════════════════════════════

def save_skill(name, description, when_to_use, steps,
               code_template="", tags=None, auto=False) -> str:
    """Save or update a skill."""
    tags = tags or []
    skills = _load_all()

    skills[name] = {
        "name": name,
        "description": description,
        "when_to_use": when_to_use,
        "steps": steps,
        "code_template": code_template,
        "tags": tags,
        "auto": bool(auto),
        "updated_at": datetime.now().isoformat(),
    }
    _save_all(skills)

    summary = (
        f"SKILL: {name}\n"
        f"WHAT: {description}\n"
        f"WHEN: {when_to_use}\n"
        f"TAGS: {', '.join(tags)}"
    )

    # Remove any prior copy of this skill from the vector store
    try:
        existing = vectorstore.get(where={"name": name})
        if existing and existing.get("ids"):
            vectorstore.delete(ids=existing["ids"])
    except Exception:
        pass

    try:
        vectorstore.add_texts(
            [summary],
            metadatas=[{"name": name, "auto": bool(auto)}],
        )
    except Exception:
        pass

    return f"✅ Skill saved: {name}"


def get_skill_dict(name: str) -> Optional[dict]:
    return _load_all().get(name)


def count_auto_skills() -> int:
    return sum(1 for s in _load_all().values() if s.get("auto"))


# ═══════════════════════════════════════════════════════════
#  Search
# ═══════════════════════════════════════════════════════════

def find_skills_scored(query: str, k: int = 3) -> list:
    """Return list of (skill_dict, score). Lower score = closer."""
    try:
        pairs = vectorstore.similarity_search_with_score(query, k=k)
    except Exception:
        return []

    skills = _load_all()
    out = []
    for doc, score in pairs:
        name = (doc.metadata or {}).get("name")
        skill = skills.get(name)
        if skill:
            out.append((skill, score))
    return out


def find_skills(query: str, k: int = 3) -> str:
    """Semantic search over skills (formatted string)."""
    matches = find_skills_scored(query, k=k)
    if not matches:
        return "No matching skills found."

    lines = []
    for skill, _score in matches:
        lines.append(f"• {skill['name']} — {skill['description']}")
    return "Matching skills:\n" + "\n".join(lines)


def get_skill(name: str) -> str:
    skills = _load_all()
    skill = skills.get(name)
    if not skill:
        return f"❌ No skill named '{name}'."
    auto_tag = " (auto-learned)" if skill.get("auto") else ""
    return (
        f"📘 SKILL: {skill['name']}{auto_tag}\n"
        f"📝 {skill['description']}\n"
        f"🎯 When to use: {skill['when_to_use']}\n"
        f"🏷️  Tags: {', '.join(skill.get('tags', []))}\n\n"
        f"STEPS:\n{skill['steps']}\n\n"
        f"CODE TEMPLATE:\n{skill.get('code_template') or '(none)'}"
    )


def list_skills() -> str:
    skills = _load_all()
    if not skills:
        return "No skills learned yet."
    lines = []
    for n, s in skills.items():
        tag = " (auto)" if s.get("auto") else ""
        lines.append(f"• {n}{tag} — {s['description']}")
    return f"🧠 {len(skills)} skill(s):\n" + "\n".join(lines)


def delete_skill(name: str) -> str:
    skills = _load_all()
    if name not in skills:
        return f"❌ No skill named '{name}'."
    del skills[name]
    _save_all(skills)

    try:
        existing = vectorstore.get(where={"name": name})
        if existing and existing.get("ids"):
            vectorstore.delete(ids=existing["ids"])
    except Exception:
        pass

    return f"🗑️ Deleted skill: {name}"


# ═══════════════════════════════════════════════════════════
#  UPGRADE #3 — auto-extraction from a successful turn
# ═══════════════════════════════════════════════════════════

def _parse_skill_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    if t.upper().startswith("NONE"):
        return None

    # strip code fences if the model added them
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    name = str(obj.get("name", "")).strip()
    if not name:
        return None
    # sanitize
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_").lower()
    if not name or len(name) > 60:
        return None

    description = str(obj.get("description", "")).strip()[:280]
    when_to_use = str(obj.get("when_to_use", "")).strip()[:280]
    steps = str(obj.get("steps", "")).strip()[:1200]
    code_template = str(obj.get("code_template", "")).strip()[:1200]

    if not description or not when_to_use or not steps:
        return None

    tags_raw = obj.get("tags") or []
    if isinstance(tags_raw, str):
        tags = [x.strip().lower() for x in tags_raw.split(",") if x.strip()]
    else:
        tags = [str(x).strip().lower() for x in tags_raw if str(x).strip()]
    tags = tags[:4]

    return {
        "name": name,
        "description": description,
        "when_to_use": when_to_use,
        "steps": steps,
        "code_template": code_template,
        "tags": tags,
    }


def extract_skill_from_turn(user_input: str,
                            tool_calls: list,
                            final_answer: str,
                            llm_call: LLMCall) -> Optional[dict]:
    """Ask the LLM whether this turn produced a reusable skill."""
    from prompts import SKILL_EXTRACTION_SYSTEM

    tool_lines = []
    for i, tc in enumerate(tool_calls, 1):
        args = tc.get("args") or {}
        arg_str = ", ".join(
            f"{k}={str(v)[:60]}" for k, v in args.items()
        )
        tool_lines.append(f"{i}. {tc.get('name', '?')}({arg_str})")
        preview = tc.get("result_preview") or ""
        if preview:
            tool_lines.append(f"   → {str(preview)[:160]}")

    combined = (
        f"USER REQUEST:\n{user_input.strip()[:800]}\n\n"
        f"TOOL CALLS MADE:\n" + "\n".join(tool_lines) + "\n\n"
        f"FINAL ANSWER (excerpt):\n{final_answer.strip()[:600]}"
    )

    out = llm_call(SKILL_EXTRACTION_SYSTEM, combined)
    if not out:
        return None
    return _parse_skill_json(out)


def save_auto_skill(skill: dict) -> bool:
    """Save an auto-extracted skill, with dedup. Returns True if saved."""
    if not skill:
        return False
    name = skill.get("name")
    if not name:
        return False

    # Dedup: is there a near-identical skill already?
    try:
        probe = f"{skill.get('description', '')} {skill.get('when_to_use', '')}"
        matches = find_skills_scored(probe, k=1)
        if matches:
            _, score = matches[0]
            if score <= config.SKILL_DEDUP_DISTANCE:
                return False
    except Exception:
        pass

    save_skill(
        name=name,
        description=skill.get("description", ""),
        when_to_use=skill.get("when_to_use", ""),
        steps=skill.get("steps", ""),
        code_template=skill.get("code_template", ""),
        tags=skill.get("tags", []),
        auto=True,
    )
    return True