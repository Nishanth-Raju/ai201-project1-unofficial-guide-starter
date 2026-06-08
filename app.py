"""
app.py — Milestone 5: grounded generation + query interface.

Flow:  question -> embed -> retrieve top-k chunks from ChromaDB -> build a
context block with numbered, attributed sources -> ask Groq to answer USING
ONLY that context -> return the answer plus the sources it drew from.

Run the web UI:   python app.py
Quick CLI test:   python app.py --cli "Does the Subaru Outback have head gasket problems?"

Needs: pip install -r requirements.txt  +  gradio   (and a GROQ_API_KEY in .env)
Build the index first:  python build_index.py
"""

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
DB_DIR = BASE / "chroma_db"
COLLECTION = "used_cars"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

load_dotenv()

SYSTEM_PROMPT = """You are an assistant that answers questions about owning used Honda Civic,
Toyota Camry, and Subaru Outback cars, based ONLY on the owner reviews and forum posts provided
in the CONTEXT below.

Rules:
- Use ONLY information found in the CONTEXT. Do not use outside knowledge.
- If the CONTEXT does not contain the answer, say: "The documents I have don't cover that."
  Do not guess or fill gaps with general knowledge.
- Cite the sources you used inline with their bracket numbers, e.g. [Source 2].
- When owners disagree, report the disagreement rather than picking one side.
- Be concise and specific (mention mileage, model years, or symptoms when the context gives them).
"""

# Loaded once and reused.
_embedder = None
_collection = None
_groq = None


def _embedder_model():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _coll():
    global _collection
    if _collection is None:
        _collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
    return _collection


def _client():
    global _groq
    if _groq is None:
        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise SystemExit("Set GROQ_API_KEY in .env (copy .env.example). Get one free at console.groq.com")
        _groq = Groq(api_key=key)
    return _groq


def retrieve(question, k=TOP_K):
    emb = _embedder_model().encode([question]).tolist()
    res = _coll().query(query_embeddings=emb, n_results=k)
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "meta": meta, "dist": dist})
    return out


def build_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c["meta"]
        blocks.append(f"[Source {i}] ({m.get('model','?')} — {m.get('subtopic','?')})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(question, k=TOP_K):
    chunks = retrieve(question, k)
    context = build_context(chunks)
    resp = _client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
    )
    reply = resp.choices[0].message.content
    sources = []
    for i, c in enumerate(chunks, 1):
        m = c["meta"]
        sources.append(f"[Source {i}] {m.get('model','?')} — {m.get('subtopic','?')}  ({m.get('source_url','')})")
    return reply, "\n".join(sources)


def run_cli(question):
    reply, sources = answer(question)
    print("\n=== ANSWER ===\n" + reply)
    print("\n=== SOURCES RETRIEVED ===\n" + sources)


def run_web():
    import gradio as gr

    def respond(q):
        if not q.strip():
            return "Ask a question about a used Civic, Camry, or Outback.", ""
        return answer(q)

    with gr.Blocks(title="The Unofficial Used-Car Guide") as demo:
        gr.Markdown("# The Unofficial Used-Car Guide\n"
                    "Owner-sourced answers for the **Honda Civic**, **Toyota Camry**, and "
                    "**Subaru Outback**. Answers are grounded only in collected reviews & forum posts.")
        q = gr.Textbox(label="Your question", placeholder="Does the 2010 Outback have head gasket issues?")
        btn = gr.Button("Ask", variant="primary")
        ans = gr.Markdown(label="Answer")
        src = gr.Textbox(label="Sources retrieved", lines=6)
        btn.click(respond, inputs=q, outputs=[ans, src])
        q.submit(respond, inputs=q, outputs=[ans, src])
    demo.launch()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli(" ".join(sys.argv[2:]) or "Does the Subaru Outback have head gasket problems?")
    else:
        run_web()
