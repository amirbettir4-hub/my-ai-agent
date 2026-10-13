"""rag.py — local document indexing and retrieval.

Drop PDFs, .txt, .md, or code files into ./documents/
Then call index_documents() once. Ask questions after.
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


def _iter_files(folder):
    """Yield (path, text_content) for readable files in folder."""
    if not os.path.isdir(folder):
        return
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext == ".pdf":
                    # Optional: needs pypdf. Skip if not installed.
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(path)
                        text = "\n".join(
                            (p.extract_text() or "") for p in reader.pages
                        )
                    except ImportError:
                        continue
                elif ext in (".txt", ".md", ".py", ".js", ".json",
                             ".html", ".css", ".csv", ".log"):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                else:
                    continue

                if text.strip():
                    yield path, text
            except Exception as e:
                print(f"   ⚠️ Skipping {path}: {e}")


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


def _hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def index_documents(force: bool = False) -> str:
    """Index every file in ./documents/. Skips files already indexed."""
    folder = config.RAG_DIR
    if not os.path.isdir(folder):
        return (
            f"⚠️ No documents folder at `{folder}`.\n"
            f"   Create it and drop .txt, .md, .pdf, or code files inside."
        )

    store = _get_store()

    # Existing IDs
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
        # Deduplicate per-file via a prefix id
        file_id = _hash(path + text[:100])
        # If every chunk already indexed, skip file
        file_chunk_ids = [f"{file_id}:{i}" for i in range(len(chunks))]
        if not force and all(cid in existing_ids for cid in file_chunk_ids):
            skipped += 1
            continue

        # Index new chunks
        texts = []
        metadatas = []
        ids = []
        for i, c in enumerate(chunks):
            cid = f"{file_id}:{i}"
            if cid in existing_ids and not force:
                continue
            texts.append(c)
            metadatas.append({"source": path, "chunk": i})
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


def search_documents(query: str, k: int = None) -> str:
    """Semantic search over indexed documents."""
    k = k or config.RAG_TOP_K
    try:
        store = _get_store()
        results = store.similarity_search(query, k=k)
    except Exception as e:
        return f"❌ Document search failed: {e}"

    if not results:
        return "No matching content in your documents. Did you index them first?"

    out = []
    for r in results:
        src = r.metadata.get("source", "unknown")
        out.append(f"📄 {src}\n{r.page_content[:500]}")
    return "Found in your documents:\n\n" + "\n\n---\n\n".join(out)