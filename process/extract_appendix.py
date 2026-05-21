#!/usr/bin/env python3
"""Vision-LLM extraction of the Volume VII Appendix concordance (toponym variants).

The Appendix is a two-column index, scanned with NO text layer, mapping place-name
variants across eras:
  * Article I  (printed p.659+, PDF idx ~711): ancient/mediaeval → modern
  * Article II (printed p.745+, PDF idx ~797): modern → ancient/mediaeval ("Reversed Index")
(printed page = PDF index − 52). This sends each page IMAGE to a vision model and
parses structured rows {headword, equivalents[], note}. Results feed WHG's name-variant
indices and this project's own toponym matching.

Every page result is cached in `llm_cache` (keyed by provider/model/prompt-sig/page) so
re-runs cost nothing. Rows are written to `name_variant`.

Usage (keys from .env; Gemini reachable via the DoH shim):
  python3 process/extract_appendix.py --pages 745,852 --direction a2m   # prototype 2 pages
  python3 process/extract_appendix.py --start 711 --end 796 --direction a2m   # Article I
  python3 process/extract_appendix.py --start 797 --end 896 --direction m2a   # Article II
  python3 process/extract_appendix.py --pages 745 --dry-run             # render only, no API
"""
from __future__ import annotations
import argparse, hashlib, json, socket, sqlite3, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
import fitz

PDF = "data/pdf/agazetteerworld00unkngoog.pdf"
RENDER_MATRIX = 3.0          # ~216 dpi — dense small two-column print needs resolution
MAX_OUTPUT = 32768           # a dense two-column index page is ~100-150 rows of JSON
PROVIDER_MODELS = {"gemini": "gemini-2.5-flash", "claude": "claude-sonnet-4-6"}
# Assumed list prices USD/1M (in, out) — CONFIRM; batch halves.
RATES = {"gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40),
         "claude-sonnet-4-6": (3.00, 15.00), "claude-haiku-4-5": (1.00, 5.00)}


class Row(BaseModel):
    headword: str                  # the bold lead term
    equivalents: List[str]         # the other-era name(s) given for it
    note: Optional[str]            # locating qualifier, e.g. "in co. Carlow", "a river of France"


class Page(BaseModel):
    rows: List[Row]


Page.model_rebuild()        # resolve forward refs eagerly (import-safe)


def _strict(model):
    s = model.model_json_schema()
    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object" and "properties" in n:
                n["additionalProperties"] = False
            for v in n.values(): walk(v)
        elif isinstance(n, list):
            for v in n: walk(v)
    walk(s); return s


SCHEMA = _strict(Page)

PROMPT = """This is a scanned page from the Appendix of a 19th-century gazetteer — a two-column
index of place-name equivalents across historical eras. Each entry is a row: a headword (the bold
lead term) followed by one or more equivalent names from another era, sometimes with a short
locating note (e.g. "in co. Carlow", "a river of France", "near Rome").

Transcribe EVERY entry on the page. Read the LEFT column fully top-to-bottom, then the RIGHT column.
For each entry return: headword; equivalents (the listed equivalent name(s), split on commas/semicolons);
note (any locating phrase, else null). Preserve diacritics, but normalise the archaic long-s (ſ) to a
modern 's' (e.g. "Dionyſiopolis" → "Dionysiopolis", "Caſtaghora" → "Castaghora"). Silently correct
obvious OCR errors in the printed text. Ignore the running header, page number, and column rules.
Do not invent entries."""

PROMPT_SIG = hashlib.sha256((PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:16]

CACHE_DDL = """
CREATE TABLE IF NOT EXISTS llm_cache (
  key TEXT PRIMARY KEY, provider TEXT, model TEXT, entry_id INTEGER,
  response_json TEXT, usage_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS name_variant (
  id INTEGER PRIMARY KEY, page_idx INTEGER, direction TEXT,
  headword TEXT, equivalents TEXT, note TEXT, created_at TEXT);
"""


def ensure(con):
    con.executescript(CACHE_DDL); con.commit()


def cache_key(provider, model, page_idx):
    return hashlib.sha256(f"{provider}\0{model}\0{PROMPT_SIG}\0appendix:{page_idx}".encode()).hexdigest()


def render_jpeg(idx):
    d = fitz.open(PDF)
    return d[idx].get_pixmap(matrix=fitz.Matrix(RENDER_MATRIX, RENDER_MATRIX)).tobytes("jpeg")


_DNS_DONE = False
_GENAI = None
_ANTHROPIC = None


def _ensure_genai_dns():
    global _DNS_DONE
    if _DNS_DONE:
        return
    _DNS_DONE = True
    host = "generativelanguage.googleapis.com"
    try:
        socket.getaddrinfo(host, 443); return
    except OSError:
        pass
    for doh in ("https://1.1.1.1/dns-query", "https://8.8.8.8/resolve"):
        try:
            req = urllib.request.Request(f"{doh}?name={host}&type=A", headers={"accept": "application/dns-json"})
            ips = [a["data"] for a in json.load(urllib.request.urlopen(req, timeout=8))["Answer"] if a.get("type") == 1]
            orig = socket.getaddrinfo
            socket.getaddrinfo = lambda h, *a, **k: orig(ips[0] if h == host else h, *a, **k)
            print(f"[dns] pinned {host} -> {ips[0]} via DoH"); return
        except Exception:
            continue


def gen_gemini(model, jpeg):
    global _GENAI
    if _GENAI is None:
        _ensure_genai_dns()
        from google import genai
        _GENAI = genai.Client()                 # create once and reuse
    from google.genai import types
    r = _GENAI.models.generate_content(
        model=model,
        contents=[PROMPT, types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Page,
            max_output_tokens=MAX_OUTPUT,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))  # transcription: no thinking
    m = r.usage_metadata
    return r.text, {"input": m.prompt_token_count, "output": m.candidates_token_count}


def gen_claude(model, jpeg):
    global _ANTHROPIC
    import anthropic, base64
    if _ANTHROPIC is None:
        _ANTHROPIC = anthropic.Anthropic()
    msg = _ANTHROPIC.messages.create(
        model=model, max_tokens=MAX_OUTPUT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(jpeg).decode()}},
            {"type": "text", "text": PROMPT}]}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    t = next(b.text for b in msg.content if b.type == "text")
    return t, {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--provider", choices=["gemini", "claude"], default="gemini")
    ap.add_argument("--model")
    ap.add_argument("--pages", help="comma-separated PDF page indices")
    ap.add_argument("--start", type=int); ap.add_argument("--end", type=int)
    ap.add_argument("--direction", choices=["a2m", "m2a"], default="a2m")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.pages:
        idxs = [int(x) for x in args.pages.split(",")]
    elif args.start is not None and args.end is not None:
        idxs = list(range(args.start, args.end + 1))
    else:
        sys.exit("give --pages or --start/--end")
    model = args.model or PROVIDER_MODELS[args.provider]
    direction = "ancient_to_modern" if args.direction == "a2m" else "modern_to_ancient"

    con = sqlite3.connect(args.db); ensure(con)
    gen = gen_gemini if args.provider == "gemini" else gen_claude
    tot_in = tot_out = total_rows = calls = 0

    for idx in idxs:
        jpeg = render_jpeg(idx)
        if args.dry_run:
            print(f"  idx {idx}: rendered {len(jpeg):,} bytes JPEG (no API)")
            continue
        key = cache_key(args.provider, model, idx)
        row = con.execute("SELECT response_json,usage_json FROM llm_cache WHERE key=?", (key,)).fetchone()
        if row:
            text, usage, tag = row[0], json.loads(row[1] or "{}"), "cache"
        else:
            for attempt in range(4):
                try:
                    text, usage = gen(model, jpeg)
                    Page.model_validate_json(text); break
                except Exception as e:
                    if attempt == 3:
                        print(f"  idx {idx}: FAILED {type(e).__name__}: {str(e)[:70]}"); text = None
                    else:
                        time.sleep(2 ** attempt)
            if text is None:
                continue
            con.execute("INSERT OR REPLACE INTO llm_cache(key,provider,model,entry_id,response_json,usage_json,created_at)"
                        " VALUES(?,?,?,?,?,?,?)", (key, args.provider, model, idx, text, json.dumps(usage),
                        datetime.now(timezone.utc).isoformat(timespec="seconds")))
            con.commit(); calls += 1; tag = model.split("-")[-1]
        page = Page.model_validate_json(text)
        con.execute("DELETE FROM name_variant WHERE page_idx=?", (idx,))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r in page.rows:
            con.execute("INSERT INTO name_variant(page_idx,direction,headword,equivalents,note,created_at)"
                        " VALUES(?,?,?,?,?,?)", (idx, direction, r.headword, json.dumps(r.equivalents), r.note, now))
        con.commit()
        tot_in += usage.get("input") or 0; tot_out += usage.get("output") or 0; total_rows += len(page.rows)
        print(f"  [{tag:>6}] idx {idx} (printed {idx-52}): {len(page.rows)} rows")

    if args.dry_run or calls == 0 and total_rows == 0:
        return
    n = max(1, len([i for i in idxs]))
    ri, ro = RATES.get(model, (0, 0))
    per_page_cost = (tot_in / n / 1e6 * ri) + (tot_out / n / 1e6 * ro)
    APPENDIX_PAGES = 185           # concordance extent (idx ~711–896)
    print(f"\nsample: {total_rows} variant rows from {n} page(s); "
          f"avg {tot_in//n:,} in / {tot_out//n:,} out tokens/page")
    print(f"per-page cost ({model}): ${per_page_cost:.4f}  ->  full Appendix (~{APPENDIX_PAGES} pages): "
          f"${per_page_cost*APPENDIX_PAGES:,.2f}  (batch ${per_page_cost*APPENDIX_PAGES*0.5:,.2f})")
    print("rates ASSUMED list prices (see RATES); image-token counts are real (measured). Re-runs free (cached).")


if __name__ == "__main__":
    main()
