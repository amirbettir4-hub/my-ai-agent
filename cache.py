"""cache.py — disk-based cache for deterministic tool results.

Upgrade #4:
  - Same tool + same args → return the previous result without re-running.
  - Per-tool TTL: reads/lists cache for a few minutes, search_web for 3 min.
  - Failed results (starting with ❌ or ⚠️) are never cached.
  - Cache is plain JSON files in ./cache/. Safe to delete anytime.
"""

import os
import json
import time
import hashlib

import config


def _key(tool: str, args: dict) -> str:
    blob = json.dumps({"tool": tool, "args": args}, sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _path(key: str) -> str:
    return os.path.join(config.CACHE_DIR, f"{key}.json")


def get(tool: str, args: dict, ttl):
    """Return the cached result string, or None if missing/expired."""
    if not config.CACHE_ENABLED:
        return None
    key = _key(tool, args)
    path = _path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    ts = data.get("ts", 0)
    if ttl is not None and (time.time() - ts) > ttl:
        try:
            os.remove(path)
        except Exception:
            pass
        return None
    return data.get("result")


def set(tool: str, args: dict, result: str) -> None:
    """Store a result. Never stores failures."""
    if not config.CACHE_ENABLED:
        return
    if not isinstance(result, str):
        result = str(result)
    if result.lstrip().startswith(("❌", "⚠️")):
        return  # don't cache failures
    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        key = _key(tool, args)
        path = _path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "ts": time.time(),
                "tool": tool,
                "args": args,
                "result": result,
            }, f, ensure_ascii=False)
    except Exception:
        pass


def clear() -> int:
    """Delete every cached entry. Returns count removed."""
    if not os.path.isdir(config.CACHE_DIR):
        return 0
    n = 0
    for name in os.listdir(config.CACHE_DIR):
        if name.endswith(".json"):
            try:
                os.remove(os.path.join(config.CACHE_DIR, name))
                n += 1
            except Exception:
                pass
    return n


def stats() -> dict:
    """Return {'files': N, 'bytes': N} for the cache dir."""
    if not os.path.isdir(config.CACHE_DIR):
        return {"files": 0, "bytes": 0}
    files = 0
    total = 0
    for name in os.listdir(config.CACHE_DIR):
        if not name.endswith(".json"):
            continue
        try:
            total += os.path.getsize(os.path.join(config.CACHE_DIR, name))
            files += 1
        except Exception:
            pass
    return {"files": files, "bytes": total}