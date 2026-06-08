"""
ingest_urls.py — Milestone 3 helper: fetch the planning.md source URLs into documents/.

For each URL:
  * Reddit threads      -> uses Reddit's .json endpoint, saves post + top comments
  * Everything else     -> fetches the HTML and strips it down to readable text

Each saved file starts with a metadata header (source URL, model, subtopic) so the
origin survives into chunking — that's what lets the RAG system cite its sources later.

Run:
  python ingest_urls.py            fetch every URL from the web (skips ones already saved)
  python ingest_urls.py --manual   process files you saved into documents/manual/

Manual recovery for blocked sites (Reddit, KBB, ToyotaNation, JS-rendered blogs):
  1. Open the URL in your browser. For Reddit, add ".json" to the end of the URL.
  2. Ctrl+S -> "Webpage, HTML Only"  (or for the .json view, save as .json/.txt)
  3. Save it into documents/manual/ named with the matching slug, e.g.
        01-outback-civic-to-outback.html   (or .json)
  4. Run:  python ingest_urls.py --manual
The script applies the SAME extraction + source header, so all 14 end up identical.

Needs: pip install requests beautifulsoup4
"""

import re
import sys
import time
import json
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Missing deps. Run:  pip install requests beautifulsoup4")

DOCS_DIR = Path(__file__).parent / "documents"
DOCS_DIR.mkdir(exist_ok=True)

# Anything shorter than this is almost certainly a junk snippet (JS-rendered page,
# blocked stub, or sidebar) rather than the real article — flag it for manual recovery.
MIN_CHARS = 500

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# (filename slug, model, subtopic, url) — mirrors the planning.md Documents table.
SOURCES = [
    ("01-outback-civic-to-outback", "Outback", "Buying / switching opinions",
     "https://www.reddit.com/r/subaru/comments/1muf89g/from_honda_civic_to_subaru_outback_36l_or_maybe/"),
    ("02-camry-problems-reliability", "Camry", "Common problems & reliability",
     "https://www.fs1inc.com/blog/2012-toyota-camry-problems-reliability/"),
    ("03-camry-honda-owner-switching", "Camry", "Owner opinions (forum)",
     "https://www.toyotanation.com/threads/long-time-honda-owner-switching-to-toyota-camry-suggestions-for-these-two-cars.1698262/"),
    ("04-camry-repairpal-2012", "Camry", "Repair costs & common problems",
     "https://repairpal.com/2012-toyota-camry/problems"),
    ("05-outback-whatcar-reliability", "Outback", "Used reliability review",
     "https://www.whatcar.com/subaru/outback/estate/used-review/n760/reliability"),
    ("06-outback-best-years", "Outback", "Best / worst model years",
     "https://www.johnkennedysubaru.com/blogs/5362/which-subaru-outback-model-years-are-best-to-buy-in-2025/"),
    ("07-outback-cvt-problems", "Outback", "CVT transmission problems",
     "https://aatheshop.com/subaru-cvt-transmission-problems-what-you-need-to-know/"),
    ("08-outback-kbb-2010-reviews", "Outback", "Owner consumer reviews",
     "https://www.kbb.com/subaru/outback/2010/consumer-reviews/"),
    ("09-camry-carbrain-problems", "Camry", "Common problems",
     "https://carbrain.com/blog/toyota-camry-problems-you-might-encounter"),
    ("10-civic-samarins-2015", "Civic", "Reliability review",
     "https://www.samarins.com/reviews/civic-2015.html"),
    ("11-civic-reddit-honda", "Civic", "Owner opinion",
     "https://www.reddit.com/r/Honda/comments/1j5fkh9/its_kind_of_hilarious_how_badly_the/"),
    ("12-civic-years-to-avoid", "Civic", "Years to avoid",
     "https://www.copilotsearch.com/posts/honda-civic-years-to-avoid/"),
    ("13-civic-5-years-ownership", "Civic", "Long-term ownership",
     "https://www.reddit.com/r/civic/comments/114q2fo/almost_5_years_since_i_owned_a_honda_civic_2018/"),
    ("14-civic-repairpal-2012", "Civic", "Repair costs & common problems",
     "https://repairpal.com/2012-honda-civic/problems"),
]


def header(model, subtopic, url):
    return f"SOURCE_URL: {url}\nMODEL: {model}\nSUBTOPIC: {subtopic}\n{'=' * 60}\n\n"


def clean(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def parse_reddit_json(data):
    """Reddit .json payload -> title + selftext + up to 50 comment bodies."""
    post = data[0]["data"]["children"][0]["data"]
    parts = [post.get("title", ""), "", post.get("selftext", ""), "", "--- COMMENTS ---", ""]
    count = 0
    for c in data[1]["data"]["children"]:
        body = c.get("data", {}).get("body")
        if body:
            parts.append(body)
            parts.append("")
            count += 1
            if count >= 50:
                break
    return clean("\n".join(parts))


def fetch_reddit(url):
    json_url = url.rstrip("/") + "/.json?limit=100"
    r = requests.get(json_url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return parse_reddit_json(r.json())


def extract_html(html):
    """Generic HTML string -> visible text, boilerplate stripped.

    Picks the container with the most paragraph/list text rather than trusting
    <main>/<article> (some blogs bury the body in a plain <div>). Falls back to
    the whole <body> if no dense container stands out.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    best, best_len = None, 0
    for cand in soup.find_all(["main", "article", "section", "div"]):
        text = " ".join(p.get_text(" ") for p in cand.find_all(["p", "li"], recursive=False)
                         ) or " ".join(p.get_text(" ") for p in cand.find_all(["p", "li"]))
        if len(text) > best_len:
            best, best_len = cand, len(text)

    chosen = best if best_len >= MIN_CHARS else (soup.body or soup)
    return clean(chosen.get_text("\n"))


def fetch_html(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return extract_html(r.text)


MANUAL_DIR = DOCS_DIR / "manual"
BY_SLUG = {slug: (model, subtopic, url) for slug, model, subtopic, url in SOURCES}


def fetch_from_web():
    ok, skipped, blocked = 0, 0, []
    for slug, model, subtopic, url in SOURCES:
        out = DOCS_DIR / f"{slug}.txt"
        if out.exists():
            print(f"  skip {slug}.txt  (already saved)")
            skipped += 1
            continue
        try:
            body = fetch_reddit(url) if "reddit.com" in url else fetch_html(url)
            if len(body) < MIN_CHARS:
                raise ValueError(f"only {len(body)} chars — likely JS-rendered or blocked")
            out.write_text(header(model, subtopic, url) + body, encoding="utf-8")
            print(f"  OK   {slug}.txt  ({len(body):,} chars)")
            ok += 1
        except Exception as e:
            print(f"  FAIL {slug}  -> {e}")
            blocked.append((slug, url))
        time.sleep(1)  # be polite

    print(f"\n{ok} fetched, {skipped} already present, {len(blocked)} blocked.")
    if blocked:
        print("\nRecover these manually — save into documents/manual/ then run "
              "`python ingest_urls.py --manual`:")
        for slug, url in blocked:
            tip = "  (add .json to the URL in your browser first)" if "reddit.com" in url else ""
            print(f"  - {slug}.html/.json   {url}{tip}")


def process_manual():
    if not MANUAL_DIR.exists() or not any(MANUAL_DIR.iterdir()):
        print(f"Nothing in {MANUAL_DIR}. Save blocked pages there first (see header docs).")
        return
    ok = 0
    for f in sorted(MANUAL_DIR.iterdir()):
        if f.suffix.lower() not in (".html", ".htm", ".json", ".txt"):
            continue
        slug = f.stem
        if slug not in BY_SLUG:
            print(f"  ?    {f.name}  -> no slug match in SOURCES; rename to a known slug")
            continue
        model, subtopic, url = BY_SLUG[slug]
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
            if f.suffix.lower() == ".json":
                body = parse_reddit_json(json.loads(raw))
            else:
                body = extract_html(raw)
            if len(body) < MIN_CHARS:
                raise ValueError(f"only {len(body)} chars extracted")
            (DOCS_DIR / f"{slug}.txt").write_text(header(model, subtopic, url) + body, encoding="utf-8")
            print(f"  OK   {slug}.txt  ({len(body):,} chars, from {f.name})")
            ok += 1
        except Exception as e:
            print(f"  FAIL {f.name}  -> {e}")
    print(f"\n{ok} recovered from {MANUAL_DIR}")


if __name__ == "__main__":
    if "--manual" in sys.argv:
        process_manual()
    else:
        fetch_from_web()
