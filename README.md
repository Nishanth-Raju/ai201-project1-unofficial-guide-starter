# The Unofficial Guide — Project 1

A retrieval-augmented generation (RAG) system that answers used-car ownership questions for the
**Honda Civic**, **Toyota Camry**, and **Subaru Outback**, grounded only in collected owner
reviews and forum posts.

## Setup & Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Groq API key
cp .env.example .env        # then edit .env and set GROQ_API_KEY (free at console.groq.com)

# 3. (Optional) Re-collect the source documents into documents/
python ingest_urls.py            # fetch URLs; blocked sites are listed for manual saving
python ingest_urls.py --manual   # process pages saved into documents/manual/

# 4. Build the vector index (chunk -> embed -> store in ChromaDB)
python build_index.py

# 5. Ask questions
python app.py                                   # launch the Gradio web UI
python app.py --cli "Does the Outback have head gasket problems?"   # one-off CLI query

# 6. Run the evaluation question set
python evaluate.py
```

**Pipeline files:** `ingest_urls.py` (ingestion) → `build_index.py` (chunk + embed + store) →
`app.py` (retrieval + grounded generation + interface). `evaluate.py` runs the 5 test questions.
The 14 source documents are in `documents/`; the ChromaDB index is rebuilt locally by step 4.


---

## Domain


Real-world ownership knowledge for three popular used cars — the **Honda Civic (2012–2018)**,
**Toyota Camry (2012–2018)**, and **Subaru Outback (2010–2014)**: what actually breaks, at what
mileage, repair costs, and what owners wish they'd known before buying. This is valuable because
a used-car buyer's biggest risk is a model's known weak points, and getting it wrong costs
thousands. It's hard to find officially because manufacturer pages and dealer listings are sales
material — they never list common failures. The real, mileage-specific truth lives scattered
across owner forums, mechanic subreddits, and long-term review threads. The three models were
chosen for *distinct* failure profiles (Outback → head gasket, Camry → oil consumption,
Civic → AC/infotainment) so retrieval quality is observable.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

14 sources, balanced across the three models (Civic ×5, Camry ×4, Outback ×5):

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | r/subaru — Civic→Outback thread | Reddit | reddit.com/r/subaru/comments/1muf89g/ |
| 2 | fs1inc — 2012 Camry problems | Blog | fs1inc.com/blog/2012-toyota-camry-problems-reliability/ |
| 3 | ToyotaNation — Honda owner switching | Forum | toyotanation.com/threads/...1698262/ |
| 4 | RepairPal — 2012 Camry | Repair DB | repairpal.com/2012-toyota-camry/problems |
| 5 | WhatCar — Outback used reliability | Review | whatcar.com/subaru/outback/.../reliability |
| 6 | John Kennedy Subaru — best years | Blog | johnkennedysubaru.com/blogs/5362/ |
| 7 | AATheShop — Subaru CVT problems | Blog | aatheshop.com/subaru-cvt-transmission-problems... |
| 8 | KBB — 2010 Outback reviews | Owner reviews | kbb.com/subaru/outback/2010/consumer-reviews/ |
| 9 | CarBrain — Camry problems | Blog | carbrain.com/blog/toyota-camry-problems... |
| 10 | Samarins — 2015 Civic review | Review | samarins.com/reviews/civic-2015.html |
| 11 | r/Honda thread | Reddit | reddit.com/r/Honda/comments/1j5fkh9/ |
| 12 | CopilotSearch — Civic years to avoid | Blog | copilotsearch.com/posts/honda-civic-years-to-avoid/ |
| 13 | r/civic — 5 years ownership | Reddit | reddit.com/r/civic/comments/114q2fo/ |
| 14 | RepairPal — 2012 Civic | Repair DB | repairpal.com/2012-honda-civic/problems |

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** ~800 characters (~200 tokens).

**Overlap:** 150 characters.

**Why these choices fit your documents:** The corpus mixes short structured pages (RepairPal,
WhatCar — one topic each) with long multi-voice threads (the ToyotaNation thread is ~43k chars
of separate owner posts). An ~800-char chunk keeps a single owner's complaint or one repair-cost
entry intact while breaking a 43k-char thread into ~50 retrievable nuggets instead of one
undifferentiated blob. The 150-char overlap prevents a symptom and its mileage/cause from
splitting across a boundary ("started burning oil… around 90k"). Preprocessing: ingestion
stripped HTML/forum boilerplate (nav, scripts, sidebars) and prepended a
`SOURCE_URL/MODEL/SUBTOPIC` header; chunking snaps cut points to whitespace so words aren't broken.

**Final chunk count:** 244 chunks across 14 documents.

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** `all-MiniLM-L6-v2` (sentence-transformers) — fast, local, zero-cost, 384-dim,
strong on short informal English (exactly what reviews and forum posts are).

**Production tradeoff reflection:** If cost were no object, I'd weigh a larger hosted model
(e.g. OpenAI `text-embedding-3-large` or a domain-tuned option). Tradeoffs: **(1) context
length** — MiniLM truncates at 256 tokens, so a long post loses its tail; a bigger window embeds
full posts intact. **(2) Domain accuracy** — jargon like "CVT judder" or "P0420" may embed poorly
in a small general model; a larger/tuned model separates these better (and would likely have
reduced the Q3 single-source-dominance problem). **(3) Latency vs. cost** — local MiniLM is free
and instant but lower quality; a hosted model adds per-query cost and network latency for sharper
retrieval. For a small class corpus, MiniLM's speed and zero cost win.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:** The model is told to "Use ONLY information found in the
CONTEXT. Do not use outside knowledge," and explicitly: *"If the CONTEXT does not contain the
answer, say: 'The documents I have don't cover that.' Do not guess or fill gaps with general
knowledge."* It's also instructed to cite sources inline as `[Source N]` and to report
disagreement when owners conflict rather than picking a side. Temperature is set low (0.2) to
discourage improvisation. Structurally, retrieved chunks are formatted as a numbered, labeled
block — `[Source N] (Model — Subtopic)` followed by the chunk text — and these five chunks are
the *only* context provided; the model never sees the full corpus.

**How source attribution is surfaced in the response:** The answer cites `[Source N]` inline,
and the interface separately lists each source's model, subtopic, and original URL beneath the
answer — verified working in the evaluation (e.g. the Outback head-gasket answer cited
`[Source 1]`/`[Source 3]`, mapped to the John Kennedy Subaru and WhatCar pages).

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

Run with `python evaluate.py`. Results below.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | 2010 Outback head gasket problems & mileage? | Yes, ~100k miles | Head gasket leaks develop around 100k miles; cites the 2010 as a year to avoid; notes other years (2011) failing later. All sources Outback. | Relevant | Accurate |
| 2 | Camry oil consumption, which engine/years? | Yes, 2.5L on 2007/2012 | Confirms excessive oil consumption on 2007 and 2012; warns to check oil regularly. | Relevant | Accurate |
| 3 | Civic common problems & years to avoid? | 2016 AC/Bluetooth, 2006–09 engine blocks; 2012–15 & 2017–20 good | Lists cracked engine blocks (2006–09), AC/Bluetooth (2016), transmission (2001); names 2012–15 and 2017–20 as reliable. | Partially relevant (4 of 5 chunks came from one article) | Accurate |
| 4 | Subaru CVT reliability? | CVT failures reported (~115k, ~$5k) | Reports failures incl. one at 115k miles costing $5k; notes overheating/wear on 2010–13. | Partially relevant (4 of 5 chunks from one article) | Accurate |
| 5 | Civic vs Tesla Model 3 reliability? (out of corpus) | Should refuse | "The documents I have don't cover that… does not mention Tesla Model 3." | Off-target (no Tesla docs exist; retrieved Civic chunks anyway) | Accurate (correctly refused) |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

**Summary:** 5/5 responses accurate. Notably, Q5 confirms the grounding guardrail works — the
system refused an out-of-corpus question instead of hallucinating, even though retrieval still
returned (irrelevant) Civic chunks because top-k has no similarity threshold.

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:** Q3 — "What common problems do owners report on the Honda Civic, and
are there model years to avoid?" (Retrieval, not generation, is what underperformed — the final
answer was accurate, but it leaned almost entirely on one source.)

**What the system returned:** A correct answer, but **4 of the 5 retrieved chunks came from a
single article** (the CopilotSearch "years to avoid" page). The owner-voice sources — the Reddit
"5 years owning a Civic" thread and the r/Honda thread — did not surface at all, even though they
contain first-person ownership problems that are exactly what the question asks for. Q4 (CVT)
showed the same pattern: 4 of 5 chunks from one article.

**Root cause (tied to a specific pipeline stage):** The **retrieval stage**. Top-k is pure
chunk-level cosine similarity with no per-document cap. A long, keyword-dense article like
"Honda Civic years to avoid" produces many chunks that each score highly against a "common
problems / years to avoid" query, so they monopolize all 5 slots and crowd out the more
conversational forum chunks (which phrase the same issues in looser, less keyword-matching
language). The embedding model also favors the article's explicit problem-list phrasing over a
Redditor's anecdote, compounding the effect.

**What you would change to fix it:** Add **source diversity to retrieval** — either cap the
number of chunks returned per `source_url` (e.g. max 2 per document), or apply **MMR (maximal
marginal relevance)** so the top-k balances similarity against diversity. Either change would
force at least one owner-thread chunk into the context, giving the answer a genuine
"unofficial voice" instead of paraphrasing one listicle. A secondary fix: add a minimum
similarity threshold so genuinely irrelevant chunks (like the Civic chunks returned for the
Tesla question) are dropped before they reach the LLM.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:** Writing the Chunking Strategy *before*
coding meant the chunk size (800/150) was a deliberate response to the corpus's two shapes, not a
guess. When I implemented `build_index.py`, I already knew *why* 800 chars fit — keeping one
owner's post intact while splitting the 43k-char thread — so the parameters didn't need
second-guessing or rework.

**One way your implementation diverged from the spec, and why:** The plan assumed all sources
would fetch automatically, but Reddit, KBB, and ToyotaNation blocked scraping, so I added a
manual-save path (`documents/manual/` + a `--manual` mode) that wasn't in the original spec. I
also had to add a "densest text block" heuristic to the HTML extractor after a blog returned only
a 256-char junk snippet — the spec didn't anticipate JS-rendered pages.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1 — Ingestion script**

- *What I gave the AI:* My Documents table of 14 URLs and the requirement that each saved file carry a source header for later citation.
- *What it produced:* `ingest_urls.py`, which routed Reddit links through the `.json` API and other pages through HTML extraction, with a manual-recovery mode for blocked sites.
- *What I changed or overrode:* When one blog extracted only 256 chars of junk, I had the extraction logic rewritten to pick the densest text block instead of trusting `<main>`/`<article>` — which recovered the full 14k-char article and a 43k-char forum thread.

**Instance 2 — Grounded generation prompt**

- *What I gave the AI:* My Retrieval Approach (top-k=5, MiniLM) and the requirement that the system must not answer beyond the retrieved documents.
- *What it produced:* `app.py` with a strict grounding system prompt, inline `[Source N]` citations, and a Gradio interface.
- *What I changed or overrode:* I added an explicit out-of-corpus test question (Civic vs. Tesla) to `evaluate.py` to confirm the guardrail refuses rather than hallucinates — which it did.
