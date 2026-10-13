"""web_server.py — FastAPI wrapper around the agent with session management."""

import os
import json
import time
import uuid
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_turn
from langchain_core.messages import HumanMessage, AIMessage


app = FastAPI(title="Super Agent")

os.makedirs("ai_creations", exist_ok=True)
app.mount("/files", StaticFiles(directory="ai_creations"), name="files")


sessions: dict = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


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
    return {"ok": True}


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
            for event in run_turn(req.message, messages):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            traceback.print_exc()
            payload = {"type": "error", "message": f"{type(e).__name__}: {e}"}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════
#  STARTUP — this is the part that was missing
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("  🌐 Web UI starting...")
    print("  → Open http://127.0.0.1:8000 in your browser")
    print("  → Press CTRL+C here to stop the server")
    print("=" * 55 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")