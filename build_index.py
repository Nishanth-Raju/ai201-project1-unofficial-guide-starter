"""
build_index.py — Milestone 3/4: chunk the documents, embed them, store in ChromaDB.

Pipeline:
  documents/*.txt  ->  parse SOURCE_URL/MODEL/SUBTOPIC header  ->  chunk body
  ->  embed (all-MiniLM-L6-v2)  ->  ChromaDB persistent collection "used_cars"

Each chunk keeps its source metadata (url, model, subtopic) so retrieval can cite
where an answer came from — that's what the grounding/attribution requirement needs.

Run:  python build_index.py
Needs: pip install -r requirements.txt   (sentence-transformers, chromadb)
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
DOCS_DIR = BASE / "documents"
DB_DIR = BASE / "chroma_db"
COLLECTION = "used_cars"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800   # characters (~200 tokens)
OVERLAP = 150      # characters


def parse_doc(path):
    """Split a document into (metadata dict, body text). Header ends at the '=' rule."""
    text = path.read_text(encoding="utf-8")
    meta = {"source_url": "", "model": "", "subtopic": "", "doc": path.stem}
    body = text
    if "=" * 10 in text:
        head, body = text.split("=" * 10, 1)
        for line in head.splitlines():
            if line.startswith("SOURCE_URL:"):
                meta["source_url"] = line.split(":", 1)[1].strip()
            elif line.startswith("MODEL:"):
                meta["model"] = line.split(":", 1)[1].strip()
            elif line.startswith("SUBTOPIC:"):
                meta["subtopic"] = line.split(":", 1)[1].strip()
    return meta, body.strip()


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Sliding window over characters, snapped to whitespace so words aren't cut."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            sp = text.rfind(" ", start, end)
            if sp > start:
                end = sp
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, 0)
    return chunks


def main():
    docs = sorted(DOCS_DIR.glob("*.txt"))
    if not docs:
        raise SystemExit(f"No documents in {DOCS_DIR}. Run ingest_urls.py first.")

    ids, texts, metas = [], [], []
    per_doc = {}
    for path in docs:
        meta, body = parse_doc(path)
        chunks = chunk_text(body)
        per_doc[path.name] = len(chunks)
        for i, chunk in enumerate(chunks):
            ids.append(f"{meta['doc']}::{i}")
            texts.append(chunk)
            metas.append({**meta, "chunk_index": i})

    print(f"Chunked {len(docs)} documents into {len(texts)} chunks:")
    for name, count in per_doc.items():
        print(f"  {name:40} {count:>4} chunks")

    print(f"\nEmbedding with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64).tolist()

    print(f"Writing to ChromaDB at {DB_DIR} ...")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)  # rebuild fresh each run
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    # add in batches to stay well under Chroma's per-call limit
    B = 256
    for s in range(0, len(ids), B):
        col.add(ids=ids[s:s + B], documents=texts[s:s + B],
                embeddings=embeddings[s:s + B], metadatas=metas[s:s + B])

    print(f"\nDone. {col.count()} chunks indexed in collection '{COLLECTION}'.")


if __name__ == "__main__":
    main()
