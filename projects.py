"""projects.py — persistent multi-session project tracking.

Projects are like long-term goals that span multiple conversations.
Example: "build my website", "learn Rust", "make a Roblox game".
"""

import os
import json
import time
import uuid

import config


def _load():
    if not os.path.exists(config.PROJECTS_FILE):
        return {}
    try:
        with open(config.PROJECTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data):
    with open(config.PROJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_project(name: str, goal: str = "") -> str:
    """Create a new project."""
    projects = _load()
    pid = str(uuid.uuid4())[:8]
    now = time.time()
    projects[pid] = {
        "id": pid,
        "name": name,
        "goal": goal,
        "status": "active",
        "steps": [],
        "notes": "",
        "created": now,
        "updated": now,
    }
    _save(projects)
    return f"✅ Created project `{name}` (id: {pid})"


def list_projects(include_done: bool = False) -> str:
    """List all projects."""
    projects = _load()
    if not projects:
        return "No projects yet. Use `create_project` to start one."

    items = list(projects.values())
    if not include_done:
        items = [p for p in items if p.get("status") != "done"]

    if not items:
        return "No active projects."

    items.sort(key=lambda p: p.get("updated", 0), reverse=True)
    lines = []
    for p in items:
        total = len(p.get("steps", []))
        done = sum(1 for s in p.get("steps", []) if s.get("done"))
        status = p.get("status", "active")
        lines.append(
            f"• **{p['name']}** [{status}] — {done}/{total} steps — id: `{p['id']}`"
        )
    return f"📂 {len(items)} project(s):\n" + "\n".join(lines)


def get_project(pid: str) -> str:
    """Get full project details by id or name."""
    projects = _load()
    # Try id first, then name
    p = projects.get(pid)
    if not p:
        for cand in projects.values():
            if cand["name"].lower() == pid.lower():
                p = cand
                break
    if not p:
        return f"❌ No project with id or name `{pid}`."

    steps_txt = "\n".join(
        f"  {'✅' if s.get('done') else '⬜'} {i+1}. {s['text']}"
        for i, s in enumerate(p.get("steps", []))
    ) or "  (no steps yet)"

    return (
        f"📂 **{p['name']}** [{p.get('status', 'active')}]\n"
        f"🎯 Goal: {p.get('goal') or '(not set)'}\n"
        f"🆔 id: `{p['id']}`\n\n"
        f"**Steps:**\n{steps_txt}\n\n"
        f"**Notes:**\n{p.get('notes') or '(none)'}"
    )


def update_project(pid: str, notes: str = None, status: str = None,
                   goal: str = None) -> str:
    """Update project fields."""
    projects = _load()
    p = projects.get(pid)
    if not p:
        for cand in projects.values():
            if cand["name"].lower() == pid.lower():
                p = cand
                break
    if not p:
        return f"❌ No project with id `{pid}`."

    if notes is not None:
        p["notes"] = notes
    if status is not None:
        p["status"] = status
    if goal is not None:
        p["goal"] = goal
    p["updated"] = time.time()
    _save(projects)
    return f"✅ Updated project `{p['name']}`"


def add_project_step(pid: str, text: str) -> str:
    """Add a step to a project."""
    projects = _load()
    p = projects.get(pid)
    if not p:
        for cand in projects.values():
            if cand["name"].lower() == pid.lower():
                p = cand
                break
    if not p:
        return f"❌ No project with id `{pid}`."

    p.setdefault("steps", []).append({"text": text, "done": False})
    p["updated"] = time.time()
    _save(projects)
    return f"✅ Added step {len(p['steps'])} to `{p['name']}`"


def complete_project_step(pid: str, step_number: int) -> str:
    """Mark a step (1-indexed) as done."""
    projects = _load()
    p = projects.get(pid)
    if not p:
        for cand in projects.values():
            if cand["name"].lower() == pid.lower():
                p = cand
                break
    if not p:
        return f"❌ No project with id `{pid}`."

    steps = p.get("steps", [])
    if step_number < 1 or step_number > len(steps):
        return f"❌ Project has {len(steps)} steps."

    steps[step_number - 1]["done"] = True
    p["updated"] = time.time()
    _save(projects)
    return f"✅ Marked step {step_number} done in `{p['name']}`"


def delete_project(pid: str) -> str:
    """Delete a project."""
    projects = _load()
    if pid not in projects:
        for k, v in list(projects.items()):
            if v["name"].lower() == pid.lower():
                pid = k
                break
    if pid not in projects:
        return f"❌ No project with id `{pid}`."
    name = projects[pid]["name"]
    del projects[pid]
    _save(projects)
    return f"🗑️ Deleted project `{name}`"