#!/usr/bin/env python3
"""Digitise statistical tables from the volume scans by vision-LLM, into table_data.

Surya's layout model does NOT detect these 1856 tables (no ruling lines), so we can't
route them out by layout — and plain OCR linearises their cells into scrambled text. The
robust answer is a **vision-LLM**, which both *finds* and *structures* each table from the
page image. To bound cost we don't call it on every page: we score each OCR'd page by
**digit density** (tables leave many number-heavy lines even when linearised) and send only
the candidates. Each table is stored as {subject(title), header, rows} in table_data
(source='vision'), keyed by volume + printed page. Cached per image, idempotent per page.

    # corpus path: detect candidates from the merged OCR text, digitise from the page images
    python3 process/extract_tables.py --img-dir /vast/ishi/gotw/img/v5 --ocr data/txt/gotw-v5-ocr.txt --volume v5
    python3 process/extract_tables.py --img-dir /vast/ishi/gotw/img/v5 --ocr data/txt/gotw-v5-ocr.txt --volume v5 --list-candidates
    # single page from a PDF (spot check / no OCR text yet)
    python3 process/extract_tables.py --pdf data/pdf/gotw-v5.pdf --page 32 --volume v5
"""
from __future__ import annotations
import argparse, base64, hashlib, importlib.util, json, os, re, socket, sqlite3, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel

MODEL = "gemini-2.5-flash"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
HEAD = re.compile(r"## p\. (\S+) \(#(\d+)\)")   # not ^-anchored: \f-split chunks start with "\n"


class Column(BaseModel):
    label: str                            # the column heading as printed
    group: Optional[str]                  # spanning super-heading, e.g. "Population" over "1831"/"1841"
    unit: Optional[str]                   # "£", "acres", "miles", "persons", … else null
    type: Literal["place", "year", "number", "text"]   # 'place' = the reconcilable row-label column


class Table(BaseModel):
    title: Optional[str]                  # caption / subject of the table
    columns: List[Column]                 # one per column, left-to-right (data-first; render to HTML client-side)
    rows: List[List[str]]                 # body rows, one cell per column, exactly as printed
    source_note: Optional[str]            # attribution in the caption, e.g. "[Vigne]" / "[King]"
    footnotes: List[str]                  # footnotes / caveats, else []


class TableSet(BaseModel):
    tables: List[Table]                   # every statistical table on the page (empty if none)


for _m in (Column, Table, TableSet):   # `from __future__ annotations` defers — rebuild each
    _m.model_rebuild()
SCHEMA = TableSet.model_json_schema()
for _n in (SCHEMA, *SCHEMA.get("$defs", {}).values()):
    if _n.get("type") == "object":
        _n["additionalProperties"] = False

PROMPT = ("This is a scanned page from a 19th-century gazetteer. Transcribe EVERY statistical table on the "
          "page (ignore running prose). For each table return:\n"
          "- title: the caption or subject of the table (else null);\n"
          "- columns: one entry per column, left-to-right, each {label (the column heading as printed); "
          "group (a spanning super-heading above this column, e.g. 'Population' over sub-columns '1831'/'1841', "
          "else null); unit (e.g. '£', 'acres', 'miles', 'persons', else null); type (one of: 'place' for the "
          "column of place-names that labels each row; 'year'; 'number'; 'text')};\n"
          "- rows: each body row as a list of cells, ONE cell per column in column order, exactly as printed, "
          "blank cells as empty strings;\n"
          "- source_note: any attribution in the caption, e.g. '[Vigne]' / '[King]' (else null);\n"
          "- footnotes: any footnotes or caveats below the table (else []).\n"
          "Correct obvious OCR digit/letter errors. If the page has no table, return an empty list.")
SIG = hashlib.sha256((PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]
_GENAI = None


def _dns():
    host = "generativelanguage.googleapis.com"
    try:
        socket.getaddrinfo(host, 443); return
    except OSError:
        pass
    for doh in ("https://1.1.1.1/dns-query", "https://8.8.8.8/resolve"):
        try:
            r = urllib.request.Request(f"{doh}?name={host}&type=A", headers={"accept": "application/dns-json"})
            ips = [a["data"] for a in json.load(urllib.request.urlopen(r, timeout=8))["Answer"] if a.get("type") == 1]
            orig = socket.getaddrinfo
            socket.getaddrinfo = lambda h, *a, **k: orig(ips[0] if h == host else h, *a, **k); return
        except Exception:
            continue


# Backend: 'gemini' (API) or 'vllm' (self-hosted Qwen2.5-VL via OpenAI-compatible server).
BACKEND = os.environ.get("TABLE_BACKEND", "gemini")
VL_MODEL = os.environ.get("TABLE_VL_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
VL_BASE = os.environ.get("TABLE_VL_BASE", "http://localhost:8000/v1")


def _model_name():
    return VL_MODEL if BACKEND == "vllm" else MODEL


def _gen_gemini(jpeg):
    global _GENAI
    if _GENAI is None:
        _dns()
        from google import genai
        _GENAI = genai.Client()
    from google.genai import types
    r = _GENAI.models.generate_content(
        model=MODEL, contents=[PROMPT, types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=TableSet,
            max_output_tokens=16384, thinking_config=types.ThinkingConfig(thinking_budget=0)))
    return r.text, {"input": r.usage_metadata.prompt_token_count, "output": r.usage_metadata.candidates_token_count}


def _gen_vllm(jpeg):
    """Qwen2.5-VL via vLLM OpenAI-compatible chat: base64 image + schema-guided JSON."""
    b64 = base64.b64encode(jpeg).decode()
    body = json.dumps({
        "model": VL_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
        "max_tokens": 16384, "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "TableSet", "schema": SCHEMA, "strict": True}},
    }).encode()
    req = urllib.request.Request(f"{VL_BASE.rstrip('/')}/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    resp = json.load(urllib.request.urlopen(req, timeout=900))
    u = resp.get("usage", {})
    return resp["choices"][0]["message"]["content"], {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens")}


def extract(jpeg, con):
    """Vision-LLM a single page image (jpeg bytes) -> TableSet. Cached by (model, image)."""
    model = _model_name()
    key = hashlib.sha256(f"table\0{model}\0{SIG}\0{hashlib.sha256(jpeg).hexdigest()}".encode()).hexdigest()
    row = con.execute("SELECT response_json FROM llm_cache WHERE key=?", (key,)).fetchone()
    if row:
        return TableSet.model_validate_json(row[0])
    gen = _gen_vllm if BACKEND == "vllm" else _gen_gemini
    for attempt in range(4):
        try:
            text, usage = gen(jpeg)
            ts = TableSet.model_validate_json(text)              # validate before caching
            con.execute("INSERT OR REPLACE INTO llm_cache(key,provider,model,response_json,usage_json,created_at)"
                        " VALUES(?,?,?,?,?,?)", (key, BACKEND, model, text, json.dumps(usage),
                        datetime.now(timezone.utc).isoformat(timespec="seconds")))
            con.commit()
            return ts
        except Exception as e:
            if attempt == 3:
                print(f"  {BACKEND} failed: {type(e).__name__}: {str(e)[:70]}"); return None
            time.sleep(2 ** attempt)


# ── candidate detection ──────────────────────────────────────────────────────
def run_score(text: str) -> int:
    """Longest run of consecutive number-heavy lines = table rows.

    Page-level digit density fails here (the gazetteer is number-dense everywhere), but in
    our *column-ordered* OCR a table's rows stay consecutive, so a run of number-heavy lines
    is a strong, specific signal: prose tops out ~2, tables run 6-9 (validated on v1 — e.g.
    BARBADOS p.574 climate+trade tables scored 9). Needs column-aware OCR; it does NOT work
    on cross-column-scrambled text layers."""
    run = best = 0
    for l in text.splitlines():
        if l.startswith("##") or l.startswith("<!--"):
            continue
        toks = l.split()
        num_toks = sum(1 for t in toks if sum(c.isdigit() for c in t) >= 2)
        if num_toks >= 2 and len(toks) >= 2:
            run += 1; best = max(best, run)
        else:
            run = 0
    return best


TABLE_NOTE = re.compile(r"<!-- table bbox=(\[[^\]]*\]) -->")


def table_bboxes(text: str):
    """Table-region bboxes recorded at OCR time (process/ocr_pages.py) for this page. Since OCR now
    routes table cells OUT of the body text, these annotations — not text numeric-runs — are the
    reliable signal for which pages carry tables (Surya's ruled tables + the unruled-table runs)."""
    out = []
    for m in TABLE_NOTE.finditer(text):
        try:
            out.append([float(v) for v in json.loads(m.group(1))])
        except Exception:
            pass
    return out


def parse_ocr(path: str):
    """Yield (printed_page|None, image_index, text) per page from a merged OCR .txt."""
    for chunk in Path(path).read_text(encoding="utf-8").split("\f"):
        m = HEAD.search(chunk)
        if not m:
            continue
        page = int(m.group(1)) if m.group(1).isdigit() else None
        yield page, int(m.group(2)) - 1, chunk


# ── storage ──────────────────────────────────────────────────────────────────
def ensure_schema(con):
    # Data-first storage: `columns` is the JSON column-spec ({label,group,unit,type}), `rows` the cells.
    # HTML is rendered from this client-side — never stored. `header` is retained for legacy rows only.
    con.execute("""CREATE TABLE IF NOT EXISTS table_data(
        id INTEGER PRIMARY KEY, entry_id INTEGER, table_no INTEGER, headword TEXT, page_start INTEGER,
        n_rows INTEGER, n_cols INTEGER, subject TEXT, header TEXT, columns TEXT, rows TEXT,
        source_note TEXT, footnotes TEXT, created_at TEXT, volume TEXT, source TEXT DEFAULT 'html')""")
    cols = {r[1] for r in con.execute("PRAGMA table_info(table_data)")}
    for c, d in (("volume", "TEXT"), ("source", "TEXT DEFAULT 'html'"),   # legacy DBs predate these
                 ("columns", "TEXT"), ("source_note", "TEXT"), ("footnotes", "TEXT")):
        if c not in cols:
            con.execute(f"ALTER TABLE table_data ADD COLUMN {c} {d}")
    con.commit()


def store(con, ts, *, volume, page, headword=None, entry_id=None):
    """Replace any prior vision tables for this (volume, page), then insert the new set."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute("DELETE FROM table_data WHERE source='vision' AND volume=? AND page_start IS ?", (volume, page))
    for i, t in enumerate(ts.tables, 1):
        cols_json = json.dumps([c.model_dump() for c in t.columns], ensure_ascii=False)
        labels_json = json.dumps([c.label for c in t.columns], ensure_ascii=False)   # legacy `header`
        con.execute("INSERT INTO table_data(entry_id,table_no,headword,page_start,n_rows,n_cols,subject,"
                    "header,columns,rows,source_note,footnotes,created_at,volume,source) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,'vision')",
                    (entry_id, i, headword, page, len(t.rows), len(t.columns), t.title,
                     labels_json, cols_json, json.dumps(t.rows, ensure_ascii=False),
                     t.source_note, json.dumps(t.footnotes, ensure_ascii=False), now, volume))
    con.commit()


def show(ts):
    print(f"  -> {len(ts.tables)} table(s)")
    for t in ts.tables:
        cols = " | ".join((f"{c.group}/" if c.group else "") + c.label + (f" ({c.unit})" if c.unit else "")
                          + f" [{c.type}]" for c in t.columns)
        print(f"     title: {t.title} | {len(t.rows)} rows" + (f" | src: {t.source_note}" if t.source_note else ""))
        print(f"     cols: {cols}")
        for r in t.rows[:3]:
            print(f"       {r}")
        if t.footnotes:
            print(f"     notes: {t.footnotes}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--volume", required=True, help="volume tag, e.g. v5")
    ap.add_argument("--img-dir", dest="img_dir", help="page-image directory (corpus path)")
    ap.add_argument("--ocr", help="merged OCR .txt for candidate detection (with --img-dir)")
    ap.add_argument("--pdf", help="PDF source for single-page --page mode")
    ap.add_argument("--page", type=int, help="single printed page (uses --pdf)")
    ap.add_argument("--thresh", type=int, default=1, help="min table-bbox annotations to flag a table page")
    ap.add_argument("--from-triage", action="store_true",
                    help="select candidate pages from VLM page_triage (n_tables>0) instead of geometry table-bboxes")
    ap.add_argument("--triage-db", help="sqlite holding page_triage (default: --db)")
    ap.add_argument("--limit", type=int, help="cap candidates (testing)")
    ap.add_argument("--list-candidates", action="store_true", help="print candidates + scores, no API")
    ap.add_argument("--backend", choices=["gemini", "vllm"], help="table vision backend (default env/gemini)")
    ap.add_argument("--vl-model", help="vLLM vision model name (served-model-name)")
    ap.add_argument("--vl-base-url", help="vLLM OpenAI-compatible base url")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    global BACKEND, VL_MODEL, VL_BASE
    if args.backend:
        BACKEND = args.backend
    if args.vl_model:
        VL_MODEL = args.vl_model
    if args.vl_base_url:
        VL_BASE = args.vl_base_url
    con = sqlite3.connect(args.db)
    con.executescript("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, provider TEXT, model TEXT,"
                      " entry_id INTEGER, response_json TEXT, usage_json TEXT, created_at TEXT);")
    ensure_schema(con)

    # ── single page from a PDF ──
    if args.page is not None:
        import fitz
        pp = importlib.util.module_from_spec(
            importlib.util.spec_from_file_location("pp", Path(__file__).with_name("pdf_pages.py")))
        pp.__spec__.loader.exec_module(pp)
        idx = pp.page_index(args.pdf).get(args.page)
        if idx is None:
            print(f"printed page {args.page} not found in index"); return
        jpeg = fitz.open(args.pdf)[idx].get_pixmap(matrix=fitz.Matrix(3.0, 3.0)).tobytes("jpeg")
        print(f"printed p.{args.page} -> PDF idx {idx} ({len(jpeg):,} byte image)")
        if args.dry_run:
            print("(dry run)"); return
        ts = extract(jpeg, con)
        if ts:
            store(con, ts, volume=args.volume, page=args.page); show(ts)
        return

    # ── corpus: candidate pages ──
    if not args.img_dir or not args.ocr:
        ap.error("corpus mode needs --img-dir and --ocr (or use --page with --pdf)")
    files = sorted(p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.from_triage:
        # candidates from the VLM page-triage (n_tables>0) — better recall than the geometry detector,
        # which misses ~1/3 of table pages. --ocr is still used only for the image-idx -> printed-page map.
        tdb = sqlite3.connect(args.triage_db or args.db)
        tset = {r[0]: r[1] for r in tdb.execute(
            "SELECT idx, n_tables FROM page_triage WHERE volume=? AND n_tables>0", (args.volume,))}
        cands = [(p, i, tset[i]) for p, i, t in parse_ocr(args.ocr) if i in tset]
    else:
        cands = [(p, i, len(table_bboxes(t))) for p, i, t in parse_ocr(args.ocr)]
        cands = [(p, i, s) for p, i, s in cands if s >= args.thresh]   # >=1 table region on the page
    cands.sort(key=lambda x: -x[2])
    if args.limit:
        cands = cands[:args.limit]
    print(f"{len(cands)} candidate table pages (thresh {args.thresh}) of {len(files)} images")
    if args.list_candidates:
        for p, i, s in sorted(cands, key=lambda x: x[1]):
            print(f"  p.{p} (img #{i}): score {s:.2f}")
        return
    if args.dry_run:
        return
    n_tables = 0
    for p, i, s in sorted(cands, key=lambda x: x[1]):
        if i >= len(files):
            print(f"  p.{p}: image #{i} out of range, skip"); continue
        ts = extract(files[i].read_bytes(), con)
        if ts:
            store(con, ts, volume=args.volume, page=p)
            n_tables += len(ts.tables)
            print(f"  p.{p} (img #{i}, score {s:.2f}): {len(ts.tables)} table(s)", flush=True)
    print(f"\nstored {n_tables} tables from {len(cands)} candidate pages into table_data (volume {args.volume})")


if __name__ == "__main__":
    main()
