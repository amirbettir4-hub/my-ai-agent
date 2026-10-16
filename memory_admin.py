"""memory_admin.py — inspect / clean long-term Chroma memory.

Usage:
  python memory_admin.py list
  python memory_admin.py search "your name"
  python memory_admin.py delete-stale       # delete entries flagged as test data
  python memory_admin.py wipe               # delete EVERYTHING (fresh start)
"""

import sys
import config
from langchain_chroma import Chroma


def _store():
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_DIR,
    )


def cmd_list():
    store = _store()
    data = store.get()
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    print(f"📦 {len(ids)} entries in '{config.CHROMA_COLLECTION}'\n")
    for i, (id_, doc, meta) in enumerate(zip(ids, docs, metas)):
        src = (meta or {}).get("source", "?")
        print(f"[{i:03d}] ({src}) {doc[:160]}")
    if not ids:
        print("   (empty)")


def cmd_search(query):
    store = _store()
    results = store.similarity_search_with_score(query, k=10)
    print(f"🔎 top 10 for: {query!r}\n")
    for doc, score in results:
        # Lower score = more similar in Chroma's default L2 space.
        print(f"  score={score:8.3f}  {doc.page_content[:160]}")


def cmd_delete_stale():
    store = _store()
    data = store.get()
    ids = data.get("ids", [])
    docs = data.get("documents", [])

    # Any entry containing obvious test markers.
    markers = ["king", "blue", "favorite color", "test", "lorem",
               "space cat", "foo", "bar", "baz"]

    to_delete = []
    for id_, doc in zip(ids, docs):
        low = (doc or "").lower()
        if any(m in low for m in markers):
            to_delete.append(id_)

    if not to_delete:
        print("✨ Nothing looked like test data. Use 'wipe' for a full reset.")
        return

    print(f"🗑  Deleting {len(to_delete)} entry/entries:")
    for id_, doc in zip(ids, docs):
        if id_ in to_delete:
            print(f"   - {doc[:120]}")
    store.delete(ids=to_delete)
    print("✅ Done.")


def cmd_wipe():
    store = _store()
    data = store.get()
    ids = data.get("ids", [])
    if not ids:
        print("Already empty.")
        return
    confirm = input(f"Delete ALL {len(ids)} entries? type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Cancelled.")
        return
    store.delete(ids=ids)
    print(f"🗑  Deleted {len(ids)} entries. Memory is now empty.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "list":
        cmd_list()
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("usage: python memory_admin.py search \"query\"")
            sys.exit(1)
        cmd_search(" ".join(sys.argv[2:]))
    elif cmd == "delete-stale":
        cmd_delete_stale()
    elif cmd == "wipe":
        cmd_wipe()
    else:
        print(__doc__)