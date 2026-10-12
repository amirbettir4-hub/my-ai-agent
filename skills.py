"""skills.py — persistent skill library backed by Chroma + JSON."""

import os
import json
from datetime import datetime

from langchain_chroma import Chroma

SKILLS_DIR = "./skills_db"
SKILLS_JSON = "./skills.json"
COLLECTION = "agent_skills"


vectorstore = Chroma(
    collection_name=COLLECTION,
    persist_directory=SKILLS_DIR,
)


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


def save_skill(name, description, when_to_use, steps, code_template="", tags=None) -> str:
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
        "updated_at": datetime.now().isoformat(),
    }
    _save_all(skills)

    # Store a searchable summary in the vector DB
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

    vectorstore.add_texts([summary], metadatas=[{"name": name}])

    return f"✅ Skill saved: {name}"


def find_skills(query: str, k: int = 3) -> str:
    """Semantic search over skills."""
    try:
        results = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        return f"❌ Skill search failed: {e}"

    if not results:
        return "No matching skills found."

    lines = []
    for r in results:
        name = r.metadata.get("name", "unknown")
        # The second line of the summary is "WHAT: ..."
        parts = r.page_content.splitlines()
        what = parts[1] if len(parts) > 1 else ""
        lines.append(f"• {name} — {what}")
    return "Matching skills:\n" + "\n".join(lines)


def get_skill(name: str) -> str:
    skills = _load_all()
    skill = skills.get(name)
    if not skill:
        return f"❌ No skill named '{name}'."
    return (
        f"📘 SKILL: {skill['name']}\n"
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
    lines = [f"• {n} — {s['description']}" for n, s in skills.items()]
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