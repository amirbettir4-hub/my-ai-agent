"""rag.py — local document indexing and retrieval.

Drop PDFs, .txt, .md, or code files into ./documents/
Or upload them through the web UI (see /upload in web_server.py).
"""

import os
import hashlib

import config


_rag_store = None


def _get_store():
    """Lazy-init Chroma collection for user documents."""
    global _rag_store
    if _rag_store is None:
        from langchain_chroma import Chroma
        os.makedirs(config.CHROMA_DIR, exist_ok=True)
        _rag_store = Chroma(
            collection_name=config.CHROMA_RAG_COLLECTION,
            persist_directory=config.CHROMA_DIR,
        )
    return _rag_store


def _hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _read_file(path):
    """Return the text content of a file, or None if unreadable."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in config.RAG_ALLOWED_EXTS:
        return None
    try:
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                return "\n".join(
                    (p.extract_text() or "") for p in reader.pages
                )
            except ImportError:
                print("   ⚠️ pypdf not installed — skipping PDF")
                return None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"   ⚠️ Skipping {path}: {e}")
        return None


def _iter_files(folder):
    """Yield (path, text_content) for readable files in folder."""
    if not os.path.isdir(folder):
        return
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            path = os.path.join(root, name)
            text = _read_file(path)
            if text and text.strip():
                yield path, text


def _chunk(text, size=None, overlap=None):
    size = size or config.RAG_CHUNK_SIZE
    overlap = overlap or config.RAG_CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ── Bulk index (used by the `index_my_documents` tool) ──────

def index_documents(force: bool = False) -> str:
    """Index every file in ./documents/. Skips files already indexed."""
    folder = config.RAG_DIR
    if not os.path.isdir(folder):
        return (
            f"⚠️ No documents folder at `{folder}`.\n"
            f"   Create it and drop .txt, .md, .pdf, or code files inside."
        )

    store = _get_store()

    existing_ids = set()
    try:
        existing = store.get()
        existing_ids = set(existing.get("ids", []))
    except Exception:
        pass

    added = 0
    skipped = 0
    chunks_total = 0

    for path, text in _iter_files(folder):
        chunks = _chunk(text)
        if not chunks:
            continue
        file_id = _hash(path + text[:100])
        file_chunk_ids = [f"{file_id}:{i}" for i in range(len(chunks))]
        if not force and all(cid in existing_ids for cid in file_chunk_ids):
            skipped += 1
            continue

        texts = []
        metadatas = []
        ids = []
        for i, c in enumerate(chunks):
            cid = f"{file_id}:{i}"
            if cid in existing_ids and not force:
                continue
            texts.append(c)
            metadatas.append({
                "source": path,
                "name": os.path.basename(path),
                "chunk": i,
            })
            ids.append(cid)

        if texts:
            try:
                store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                added += 1
                chunks_total += len(texts)
            except Exception as e:
                print(f"   ⚠️ Failed to index {path}: {e}")

    if added == 0 and skipped == 0:
        return f"⚠️ No readable files found in `{folder}`."
    return (
        f"✅ Indexed {added} file(s) ({chunks_total} chunks). "
        f"Skipped {skipped} already-indexed file(s)."
    )


# ── Single-file ops for the upload UI ───────────────────────

def index_single_file(path: str, force: bool = True) -> str:
    """Index or re-index one file. Returns a status message."""
    if not os.path.isfile(path):
        return f"⚠️ Not a file: {path}"

    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    if ext not in config.RAG_ALLOWED_EXTS:
        return f"⚠️ Unsupported extension: {ext}"

    text = _read_file(path)
    if not text or not text.strip():
        return f"⚠️ No readable text in {name}"

    store = _get_store()

    existing_ids = []
    try:
        existing = store.get(where={"name": name})
        existing_ids = existing.get("ids", []) or []
    except Exception:
        existing_ids = []

    if existing_ids and not force:
        return f"⏭️ Already indexed: {name} ({len(existing_ids)} chunks)"

    if existing_ids and force:
        try:
            store.delete(ids=existing_ids)
        except Exception:
            pass

    chunks = _chunk(text)
    if not chunks:
        return f"⚠️ Nothing to chunk in {name}"

    file_id = _hash(path + text[:100])
    texts, metadatas, ids = [], [], []
    for i, c in enumerate(chunks):
        texts.append(c)
        metadatas.append({"source": path, "name": name, "chunk": i})
        ids.append(f"{file_id}:{i}")

    try:
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    except Exception as e:
        return f"❌ Indexing failed: {e}"

    return f"✅ Indexed {name}: {len(chunks)} chunks"


def list_documents() -> list:
    """Return every indexable file in RAG_DIR with its index status."""
    folder = config.RAG_DIR
    if not os.path.isdir(folder):
        return []

    store = _get_store()
    out = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in config.RAG_ALLOWED_EXTS:
            continue
        try:
            size = os.path.getsize(full)
        except Exception:
            size = 0
        chunks = 0
        try:
            data = store.get(where={"name": name})
            chunks = len(data.get("ids", []) or [])
        except Exception:
            chunks = 0
        out.append({
            "name": name,
            "size": size,
            "ext": ext,
            "chunks": chunks,
            "indexed": chunks > 0,
        })
    return out


def delete_document(name: str) -> str:
    """Remove a document from disk and drop all its chunks from Chroma."""
    folder = os.path.abspath(config.RAG_DIR)
    full = os.path.abspath(os.path.join(folder, name))

    # Safety: must stay inside RAG_DIR.
    if not full.startswith(folder + os.sep) and full != folder:
        return "❌ Invalid path"

    store = _get_store()
    removed_chunks = 0
    try:
        existing = store.get(where={"name": name})
        ids = existing.get("ids", []) or []
        if ids:
            store.delete(ids=ids)
            removed_chunks = len(ids)
    except Exception:
        pass

    removed_file = False
    if os.path.isfile(full):
        try:
            os.remove(full)
            removed_file = True
        except Exception as e:
            return f"❌ Could not delete file: {e}"

    if removed_file or removed_chunks:
        return f"🗑️ Deleted {name} ({removed_chunks} chunks)"
    return f"⚠️ Nothing to delete: {name}"


def get_index_stats() -> dict:
    """Quick stats for the UI header."""
    store = _get_store()
    total = 0
    files = 0
    try:
        data = store.get()
        total = len(data.get("ids", []) or [])
        names = {
            (m or {}).get("name")
            for m in (data.get("metadatas", []) or [])
        }
        files = len([n for n in names if n])
    except Exception:
        pass
    return {"total_chunks": total, "total_files": files}


def search_documents(query: str, k: int = None) -> str:
    """Semantic search over indexed documents. Skips weak matches."""
    k = k or config.RAG_TOP_K
    try:
        store = _get_store()
        pairs = store.similarity_search_with_score(query, k=k)
    except Exception as e:
        return f"❌ Document search failed: {e}"

    if not pairs:
        return "No matching content in your documents. Did you index them first?"

    # Drop weak matches. Chroma L2 distance: lower = closer.
    kept = [(doc, score) for doc, score in pairs
            if score <= config.RAG_MAX_DISTANCE]

    if not kept:
        return (
            f"⚠️ No strong match in your documents for: {query!r}\n"
            f"   (best distance was {pairs[0][1]:.2f}, threshold is "
            f"{config.RAG_MAX_DISTANCE}). Try a different query, or make sure "
            f"the right files are indexed."
        )

    out = []
    for doc, score in kept:
        src = doc.metadata.get("source", "unknown")
        out.append(f"📄 {src}\n{doc.page_content[:500]}")
    return "Found in your documents:\n\n" + "\n\n---\n\n".join(out)