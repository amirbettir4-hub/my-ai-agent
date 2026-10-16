"""web_server.py — FastAPI wrapper around the agent.

Endpoints:
  - /chat                — SSE streaming chat
  - /sessions            — CRUD for chat sessions
  - /upload              — multipart upload → ./documents/ → Chroma
  - /documents           — list / reindex / delete indexed docs
  - /docs-files/*        — static download of raw documents
  - /jobs                — list / clear background jobs
  - /jobs/{id}           — get status / cancel a job
  - /files/*             — static generated artifacts
"""

import os
import json
import time
import uuid
import traceback

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_turn
import memory
import rag
import jobs
import config
from langchain_core.messages import HumanMessage, AIMessage


app = FastAPI(title="Super Agent")

os.makedirs("ai_creations", exist_ok=True)
os.makedirs(config.RAG_DIR, exist_ok=True)
os.makedirs(config.JOBS_DIR, exist_ok=True)

app.mount("/files",      StaticFiles(directory="ai_creations"),  name="files")
app.mount("/docs-files", StaticFiles(directory=config.RAG_DIR),  name="docs-files")


sessions: dict = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


# ═══════════════════════════════════════════════════════════
#  Health & index
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>index.html not found</h1>", status_code=500)
    return FileResponse(path)


# ═══════════════════════════════════════════════════════════
#  Sessions
# ═══════════════════════════════════════════════════════════

@app.get("/sessions")
def list_sessions():
    items = []
    for sid, data in sessions.items():
        if len(data.get("messages", [])) == 0:
            continue
        items.append({
            "id": sid,
            "title": data.get("title", "New chat"),
            "created": data.get("created", 0),
        })
    items.sort(key=lambda x: x["created"], reverse=True)
    return {"sessions": items}


@app.get("/sessions/{sid}")
def get_session(sid: str):
    if sid not in sessions:
        raise HTTPException(404, "Session not found")

    data = sessions[sid]
    out = []
    for m in data["messages"]:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            if isinstance(m.content, str) and m.content.strip():
                out.append({"role": "assistant", "content": m.content})

    return {
        "id": sid,
        "title": data.get("title", "New chat"),
        "messages": out,
    }


@app.delete("/sessions/{sid}")
def delete_session(sid: str):
    if sid in sessions:
        del sessions[sid]
    memory.clear_session(sid)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════
#  Chat (SSE)
# ═══════════════════════════════════════════════════════════

@app.post("/chat")
def chat(req: ChatRequest):
    sid = req.session_id

    if not sid or sid not in sessions:
        sid = str(uuid.uuid4())
        raw = req.message.strip()
        title = raw[:48] + ("…" if len(raw) > 48 else "")
        sessions[sid] = {
            "title": title or "New chat",
            "created": time.time(),
            "messages": [],
        }

    messages = sessions[sid]["messages"]

    def event_stream():
        yield f"data: {json.dumps({'type': 'session', 'session_id': sid})}\n\n"
        try:
            for event in run_turn(req.message, messages, session_id=sid):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            traceback.print_exc()
            payload = {"type": "error", "message": f"{type(e).__name__}: {e}"}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════
#  Document upload (upgrade #2)
# ═══════════════════════════════════════════════════════════

def _safe_name(raw: str) -> str:
    name = os.path.basename((raw or "").strip())
    name = "".join(c for c in name if c.isalnum() or c in "._- ()[]").strip()
    return name


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    os.makedirs(config.RAG_DIR, exist_ok=True)
    results = []

    for f in files:
        name = _safe_name(f.filename or "")
        if not name:
            results.append({"name": f.filename, "ok": False,
                            "error": "invalid filename"})
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in config.RAG_ALLOWED_EXTS:
            results.append({"name": name, "ok": False,
                            "error": f"unsupported extension: {ext}"})
            continue

        try:
            content = await f.read()
        except Exception as e:
            results.append({"name": name, "ok": False, "error": str(e)})
            continue

        size_mb = len(content) / (1024 * 1024)
        if size_mb > config.RAG_MAX_FILE_MB:
            results.append({
                "name": name, "ok": False,
                "error": f"too large ({size_mb:.1f} MB > {config.RAG_MAX_FILE_MB} MB)",
            })
            continue

        dest = os.path.join(config.RAG_DIR, name)
        try:
            with open(dest, "wb") as out:
                out.write(content)
        except Exception as e:
            results.append({"name": name, "ok": False,
                            "error": f"save failed: {e}"})
            continue

        try:
            msg = rag.index_single_file(dest, force=True)
        except Exception as e:
            results.append({"name": name, "ok": False,
                            "error": f"index failed: {e}"})
            continue

        ok = msg.startswith("✅")
        results.append({
            "name": name,
            "ok": ok,
            "message": msg,
            "size": len(content),
        })

    return {"results": results}


@app.get("/documents")
def list_documents():
    docs = rag.list_documents()
    stats = rag.get_index_stats()
    return {
        "documents": docs,
        "stats": stats,
        "max_mb": config.RAG_MAX_FILE_MB,
        "allowed": config.RAG_ALLOWED_EXTS,
    }


@app.post("/documents/reindex")
def reindex_all():
    try:
        msg = rag.index_documents(force=True)
        return {"ok": True, "message": msg, "documents": rag.list_documents()}
    except Exception as e:
        return {"ok": False, "message": str(e)}


@app.delete("/documents/{name}")
def delete_document(name: str):
    safe = _safe_name(name)
    if not safe:
        raise HTTPException(400, "invalid name")
    msg = rag.delete_document(safe)
    return {"ok": msg.startswith("🗑️"), "message": msg,
            "documents": rag.list_documents()}


# ═══════════════════════════════════════════════════════════
#  Background jobs (upgrade #5)
# ═══════════════════════════════════════════════════════════

def _job_view(j: dict) -> dict:
    """Trim results for list responses; keep detail only for single fetch."""
    if not j:
        return None
    return j


@app.get("/jobs")
def list_jobs(status: str = None):
    items = jobs.list_jobs(status=status or None)
    return {
        "jobs": [_job_view(j) for j in items],
        "counts": jobs.counts(),
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return _job_view(j)


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    ok = jobs.cancel(job_id)
    return {"ok": ok, "job": jobs.get(job_id)}


@app.delete("/jobs")
def clear_finished_jobs():
    n = jobs.clear_finished()
    return {"ok": True, "removed": n}


# ═══════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("  🌐 Web UI starting...")
    print("  → Open http://127.0.0.1:8000 in your browser")
    print("  → Press CTRL+C here to stop the server")
    print("=" * 55 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")