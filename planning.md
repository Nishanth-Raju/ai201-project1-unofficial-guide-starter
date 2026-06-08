# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

**Domain:** Real-world ownership of three popular used cars — the **Honda Civic (2012–2018)**,
**Toyota Camry (2012–2018)**, and **Subaru Outback (2010–2014)**. The system answers questions
about what actually breaks, how much repairs cost, mileage-specific problems, and what owners
wish they'd known before buying.

**Why this knowledge is valuable:** A used-car buyer's biggest risk is a model's *known* weak
points — the head gasket on early Outbacks, AC condenser failures on Civics, or excessive oil
consumption on certain Camry years. Knowing these before purchase changes which car you buy,
what you inspect, and how you negotiate. Getting it wrong costs thousands of dollars.

**Why it's hard to find through official channels:** Manufacturer pages and dealer listings are
sales material — they never list common failures. Official maintenance schedules tell you *when*
to change oil, not that *this engine burns oil between changes*. The real, mileage-specific
ownership truth lives in owner forums, mechanic subreddits, and long-term review threads written
by people who actually drove these cars past 100k miles. This system surfaces that scattered,
unofficial knowledge in one place.

**Why three models with different failure profiles:** Each car has a distinct, well-documented
weak spot (Outback → head gasket, Camry → oil consumption on some years, Civic → AC/infotainment).
That contrast makes retrieval testable — a good system should pull the *Outback* head-gasket
threads for an Outback question and not bleed Camry oil-consumption results into it.

---

## Documents

| #  | Model | Subtopic | Source / URL | Fetch method |
|----|-------|----------|--------------|--------------|
| 1  | Outback | Buying / switching opinions | r/subaru — "from Honda Civic to Subaru Outback 3.6L" — https://www.reddit.com/r/subaru/comments/1muf89g/from_honda_civic_to_subaru_outback_36l_or_maybe/ | Reddit `.json` |
| 2  | Camry | Common problems & reliability | fs1inc blog — 2012 Camry problems — https://www.fs1inc.com/blog/2012-toyota-camry-problems-reliability/ | Static page |
| 3  | Camry | Owner opinions (forum) | ToyotaNation — long-time Honda owner switching to Camry — https://www.toyotanation.com/threads/long-time-honda-owner-switching-to-toyota-camry-suggestions-for-these-two-cars.1698262/ | Forum (may need manual) |
| 4  | Camry | Repair costs & common problems | RepairPal — 2012 Camry problems — https://repairpal.com/2012-toyota-camry/problems | Static page |
| 5  | Outback | Used reliability review | WhatCar — Outback used reliability — https://www.whatcar.com/subaru/outback/estate/used-review/n760/reliability | Static page |
| 6  | Outback | Best / worst model years | John Kennedy Subaru — which Outback years to buy — https://www.johnkennedysubaru.com/blogs/5362/which-subaru-outback-model-years-are-best-to-buy-in-2025/ | Static page |
| 7  | Outback | CVT transmission problems | AATheShop — Subaru CVT problems — https://aatheshop.com/subaru-cvt-transmission-problems-what-you-need-to-know/ | Static page |
| 8  | Outback | Owner consumer reviews | KBB — 2010 Outback consumer reviews — https://www.kbb.com/subaru/outback/2010/consumer-reviews/ | Static (may block) |
| 9  | Camry | Common problems | CarBrain — Camry problems — https://carbrain.com/blog/toyota-camry-problems-you-might-encounter | Static page |
| 10 | Civic | Reliability review | Samarins — 2015 Civic review — https://www.samarins.com/reviews/civic-2015.html | Static page |
| 11 | Civic | Owner opinion | r/Honda — "how badly the…" — https://www.reddit.com/r/Honda/comments/1j5fkh9/its_kind_of_hilarious_how_badly_the/ | Reddit `.json` |
| 12 | Civic | Years to avoid | CopilotSearch — Civic years to avoid — https://www.copilotsearch.com/posts/honda-civic-years-to-avoid/ | Static page |
| 13 | Civic | Long-term ownership | r/civic — "almost 5 years owning a 2018 Civic" — https://www.reddit.com/r/civic/comments/114q2fo/almost_5_years_since_i_owned_a_honda_civic_2018/ | Reddit `.json` |
| 14 | Civic | Repair costs & common problems | RepairPal — 2012 Civic problems — https://repairpal.com/2012-honda-civic/problems | Static page |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** ~800 characters (roughly 200 tokens).

**Overlap:** 150 characters.

**Reasoning:** The corpus has two shapes. Short structured pages (RepairPal, WhatCar — a
few thousand chars each) cover one topic, while the forum and Reddit threads are long and
multi-voiced (the ToyotaNation thread is ~43k chars of separate owner posts). An ~800-char
chunk is large enough to keep a single owner's complaint or one repair-cost entry intact,
but small enough that a 43k-char thread becomes ~50 retrievable nuggets instead of one
undifferentiated blob. The 150-char overlap stops a symptom and its cause/mileage from being
split across a boundary — common in these posts ("started burning oil… around 90k miles").
Preprocessing: each document already carries a `SOURCE_URL / MODEL / SUBTOPIC` header from
ingestion, and HTML/forum boilerplate (nav, scripts, sidebars) was stripped during ingestion.
Final chunk count is reported by `build_index.py` when it runs.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** `all-MiniLM-L6-v2` via sentence-transformers. It is fast, runs locally
with no API cost, produces 384-dim vectors, and performs well on short, informal English text —
which is exactly what owner reviews and forum posts are.

**Top-k:** 5 chunks per query. Enough to gather corroborating opinions from multiple owners
without flooding the LLM context with off-topic or contradictory chunks.

**Production tradeoff reflection:** If cost were no object and this served real buyers, I'd
weigh a larger hosted model (e.g. OpenAI `text-embedding-3-large` or Voyage's automotive-tuned
embeddings). The tradeoffs: (1) **context length** — MiniLM truncates at 256 tokens, so a long
owner post can lose its tail; a larger model with a bigger window would embed full posts intact.
(2) **domain accuracy** — model-specific jargon ("CVT judder", "P0420", "head gasket weep")
may embed poorly in a general small model; a larger or domain-tuned model would separate these
better. (3) **latency vs. cost** — local MiniLM is free and instant but lower quality; a hosted
model adds per-query cost and network latency in exchange for sharper retrieval. For a class
project with a small corpus, MiniLM's speed and zero cost outweigh the accuracy gain.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Does the 2010 Subaru Outback have head gasket problems, and around what mileage? | Yes — head gasket leaks are a known weak point, typically around ~100k miles. |
| 2 | Do owners report the Toyota Camry consuming or burning oil, and on which engine or years? | Yes — excessive oil consumption is reported on the 2.5L, notably 2007 and 2012 model years. |
| 3 | What common problems do owners report on the Honda Civic, and are there model years to avoid? | AC/Bluetooth issues on 2016, cracked engine blocks on 2006–2009; 2012–2015 and 2017–2020 are considered reliable. |
| 4 | What do owners say about the reliability of the Subaru CVT transmission? | Owners report CVT failures (one at ~115k miles, ~$5k repair); the CVT is a known reliability concern. |
| 5 | How does the Honda Civic compare to a Tesla Model 3 for reliability? | Out of corpus — the system should refuse ("documents don't cover that"), NOT invent an answer. This tests the grounding guardrail. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **Source bias toward listicle/blog content over real owner voice.** Several sources
   (RepairPal, "years to avoid" blogs) are SEO articles that summarize problems in dense,
   keyword-rich language, while the genuine owner experience lives in the forum and Reddit
   threads. The risk is that retrieval favors the keyword-dense blog chunks and the
   "unofficial" voice — the whole point of the project — never surfaces. *(This risk
   materialized — see the Failure Case Analysis in README.md.)*

2. **Cross-model contamination in retrieval.** Because all three cars share the same
   vocabulary (head gasket, oil consumption, transmission), a query about one model could
   pull chunks about another. Mitigation: each chunk carries a `MODEL` metadata tag, and the
   three cars were deliberately chosen for *distinct* failure profiles so correct retrieval is
   observable. *(In testing this held up — Outback queries returned only Outback chunks.)*

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

```
┌──────────────────┐   ingest_urls.py  (requests + BeautifulSoup, Reddit .json)
│ Document          │   14 URLs → documents/*.txt  (+ SOURCE_URL/MODEL/SUBTOPIC header)
│ Ingestion         │
└────────┬─────────┘
         ▼
┌──────────────────┐   build_index.py → chunk_text()
│ Chunking          │   ~800-char sliding window, 150 overlap, snapped to whitespace
└────────┬─────────┘   → 244 chunks
         ▼
┌──────────────────┐   sentence-transformers  all-MiniLM-L6-v2  (384-dim)
│ Embedding +       │   stored in ChromaDB (PersistentClient, cosine), metadata per chunk
│ Vector Store      │
└────────┬─────────┘
         ▼
┌──────────────────┐   app.py retrieve() → top-k = 5 by cosine similarity
│ Retrieval         │
└────────┬─────────┘
         ▼
┌──────────────────┐   Groq (llama-3.3-70b-versatile), grounding system prompt
│ Generation        │   context = numbered attributed chunks → answer + [Source N] cites
└──────────────────┘   Interface: Gradio web UI + --cli mode
```

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:** Used Claude (Claude Code) to generate
`ingest_urls.py`. Input: the Documents table above plus the requirement that each saved file
carry a source header. Expected output: a script that fetches each URL, handles Reddit's
`.json` endpoint separately from regular HTML, and strips boilerplate. Verified by running it
and checking character counts — caught and fixed a thin-extraction bug (added a densest-block
heuristic) when one blog returned only 256 chars.

**Milestone 4 — Embedding and retrieval:** Gave Claude the Chunking Strategy and Retrieval
Approach sections and asked it to implement `build_index.py` using `all-MiniLM-L6-v2` and
ChromaDB, carrying metadata through. Verified by running two sanity queries and confirming
model-clean retrieval (Outback query → only Outback chunks).

**Milestone 5 — Generation and interface:** Asked Claude to implement `app.py` with a strict
grounding prompt and source attribution, plus a Gradio UI. Verified with `evaluate.py` against
the 5 test questions, specifically checking that the out-of-corpus question forced a refusal.
