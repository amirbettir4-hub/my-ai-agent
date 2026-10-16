"""jobs.py — single-worker background job queue.

Upgrade #5:
  - Long-running tools run through one worker thread, serialized so a
    4-8 GB PC never has two heavy jobs competing for RAM.
  - Every job has an id, status, timestamps, and (on completion) a result.
  - Synchronous mode: agent.run_tool() blocks on wait() for its turn.
  - Fire-and-forget mode: agent can call queue_job() and poll later.

Jobs are held in memory only — they disappear on server restart.
That keeps things simple and safe on slow disks.
"""

import time
import uuid
import threading
import queue

import config


# ── In-memory job registry ───────────────────────────────────

_JOBS: dict = {}            # job_id -> dict
_JOBS_LOCK = threading.RLock()
_QUEUE: "queue.Queue[str]" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()

# Runner function — set by agent.py after tools are defined.
# Signature: fn(tool_name: str, args: dict) -> any
_tool_runner = None


# ── Public: register the tool runner (called once from agent.py) ──

def register_tool_runner(fn) -> None:
    global _tool_runner
    _tool_runner = fn


# ── Worker thread ────────────────────────────────────────────

def _ensure_worker():
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        t = threading.Thread(target=_worker_loop, name="job-worker", daemon=True)
        t.start()
        _WORKER_STARTED = True


def _worker_loop():
    while True:
        job_id = _QUEUE.get()
        try:
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
            if job is None:
                continue
            if job.get("status") == "cancelled":
                continue

            job["status"] = "running"
            job["started"] = time.time()

            if _tool_runner is None:
                job["status"] = "error"
                job["error"] = "no tool runner registered"
                job["finished"] = time.time()
                continue

            try:
                result = _tool_runner(job["tool"], job["args"])
                # If it was cancelled mid-flight, discard the result.
                if job.get("status") == "cancelled":
                    continue
                job["result"] = str(result)
                job["status"] = "done"
                job["finished"] = time.time()
            except Exception as e:
                job["error"] = f"{type(e).__name__}: {e}"
                job["status"] = "error"
                job["finished"] = time.time()
        finally:
            _QUEUE.task_done()
            _prune_history()


def _prune_history():
    """Keep the JOBS dict from growing without bound."""
    try:
        keep = config.JOB_HISTORY_KEEP
    except Exception:
        keep = 50
    with _JOBS_LOCK:
        finished = [
            (jid, j) for jid, j in _JOBS.items()
            if j.get("status") in ("done", "error", "cancelled")
        ]
        if len(finished) <= keep:
            return
        finished.sort(key=lambda kv: kv[1].get("finished") or 0)
        to_remove = finished[: len(finished) - keep]
        for jid, _ in to_remove:
            _JOBS.pop(jid, None)


# ── Public API ───────────────────────────────────────────────

def enqueue(tool_name: str, args: dict) -> str:
    """Queue a job. Returns the job id."""
    if not config.JOBS_ENABLED:
        raise RuntimeError("jobs disabled in config")

    _ensure_worker()

    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "tool": tool_name,
            "args": args or {},
            "status": "queued",
            "created": now,
            "started": None,
            "finished": None,
            "result": None,
            "error": None,
        }
    _QUEUE.put(job_id)
    return job_id


def get(job_id: str) -> dict:
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        return dict(j) if j else None


def list_jobs(status: str = None) -> list:
    """Return job dicts, most recent first."""
    with _JOBS_LOCK:
        items = list(_JOBS.values())
    if status:
        items = [j for j in items if j.get("status") == status]
    items.sort(key=lambda j: j.get("created", 0), reverse=True)
    return [dict(j) for j in items]


def cancel(job_id: str) -> bool:
    """Mark a queued or running job as cancelled.

    If it's still queued, the worker skips it. If it's running, we can't
    actually interrupt it (Python threads), but we'll discard its result.
    """
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if not j:
            return False
        if j.get("status") in ("done", "error", "cancelled"):
            return False
        j["status"] = "cancelled"
        j["finished"] = time.time()
    return True


def wait(job_id: str, timeout: float = None) -> dict:
    """Block until the job finishes, or the timeout elapses.

    Returns the (possibly still running) job dict.
    """
    deadline = time.time() + (timeout if timeout is not None else 1e9)
    while True:
        j = get(job_id)
        if j is None:
            return None
        if j.get("status") in ("done", "error", "cancelled"):
            return j
        if time.time() >= deadline:
            return j
        time.sleep(0.4)


def counts() -> dict:
    """Return {'queued':N,'running':N,'done':N,'error':N,'cancelled':N}."""
    out = {"queued": 0, "running": 0, "done": 0, "error": 0, "cancelled": 0}
    with _JOBS_LOCK:
        for j in _JOBS.values():
            s = j.get("status")
            if s in out:
                out[s] += 1
    return out


def clear_finished() -> int:
    """Remove every finished job from the registry. Returns count removed."""
    with _JOBS_LOCK:
        to_remove = [
            jid for jid, j in _JOBS.items()
            if j.get("status") in ("done", "error", "cancelled")
        ]
        for jid in to_remove:
            _JOBS.pop(jid, None)
    return len(to_remove)