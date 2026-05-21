#!/usr/bin/env python3
"""Recover/verify statistical tables from the Volume 5 PDF by vision-LLM.

Southall's HTML transcription dropped many embedded tables; now that gotw-v5.pdf
tallies with the transcript (and process/pdf_pages maps printed page → PDF index,
handling plate drift), we can digitise any table straight from the scan. Given a
printed page number, this renders the page and extracts every statistical table as
structured {title, header, rows}, storing them in table_data (source='pdf'). Cached.

    python3 process/extract_tables.py --page 32         # Madras Presidency table
    python3 process/extract_tables.py --page 14 --dry-run
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, socket, sqlite3, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
import fitz

pp = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("pp", Path(__file__).with_name("pdf_pages.py")))
pp.__spec__.loader.exec_module(pp)

MODEL = "gemini-2.5-flash"


class Table(BaseModel):
    title: Optional[str]      # caption / what the table is about
    header: List[str]         # column labels (top row)
    rows: List[List[str]]     # body rows, cells as printed


class TableSet(BaseModel):
    tables: List[Table]       # every statistical table on the page (empty if none)


TableSet.model_rebuild()
SCHEMA = TableSet.model_json_schema()
for _n in (SCHEMA, *SCHEMA.get("$defs", {}).values()):
    if _n.get("type") == "object":
        _n["additionalProperties"] = False

PROMPT = ("This is a scanned page from a 19th-century gazetteer. Transcribe EVERY statistical table on "
          "the page (ignore running prose). For each table give: title (the caption or subject, else null); "
          "header (the column labels); rows (each body row as a list of cells, exactly as printed, keeping "
          "blank cells as empty strings). Correct obvious OCR digit/letter errors. If the page has no "
          "table, return an empty list.")
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


def extract(jpeg, con):
    global _GENAI
    key = hashlib.sha256(f"table\0{MODEL}\0{SIG}\0{hashlib.sha256(jpeg).hexdigest()}".encode()).hexdigest()
    row = con.execute("SELECT response_json FROM llm_cache WHERE key=?", (key,)).fetchone()
    if row:
        return TableSet.model_validate_json(row[0])
    if _GENAI is None:
        _dns()
        from google import genai
        _GENAI = genai.Client()
    from google.genai import types
    for attempt in range(4):
        try:
            r = _GENAI.models.generate_content(
                model=MODEL, contents=[PROMPT, types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=TableSet,
                    max_output_tokens=16384, thinking_config=types.ThinkingConfig(thinking_budget=0)))
            ts = TableSet.model_validate_json(r.text)
            con.execute("INSERT OR REPLACE INTO llm_cache(key,provider,model,response_json,usage_json,created_at)"
                        " VALUES(?,?,?,?,?,?)", (key, "gemini", MODEL, r.text, json.dumps(
                        {"input": r.usage_metadata.prompt_token_count, "output": r.usage_metadata.candidates_token_count}),
                        datetime.now(timezone.utc).isoformat(timespec="seconds")))
            con.commit()
            return ts
        except Exception as e:
            if attempt == 3:
                print(f"  failed: {type(e).__name__}: {str(e)[:70]}"); return None
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/pdf/gotw-v5.pdf")
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--page", type=int, required=True, help="printed page number")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    con.executescript("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, provider TEXT, model TEXT,"
                      " entry_id INTEGER, response_json TEXT, usage_json TEXT, created_at TEXT);")
    idx = pp.page_index(args.pdf).get(args.page)
    if idx is None:
        print(f"printed page {args.page} not found in index"); return
    d = fitz.open(args.pdf)
    jpeg = d[idx].get_pixmap(matrix=fitz.Matrix(3.0, 3.0)).tobytes("jpeg")
    print(f"printed p.{args.page} -> PDF idx {idx} ({len(jpeg):,} byte image)")
    if args.dry_run:
        print("(dry run: no API)"); return
    ts = extract(jpeg, con)
    if not ts:
        return
    print(f"extracted {len(ts.tables)} table(s):")
    for t in ts.tables:
        print(f"\n  title: {t.title}")
        print(f"  header: {t.header}")
        for r in t.rows[:5]:
            print(f"  row:   {r}")
        if len(t.rows) > 5:
            print(f"  … {len(t.rows)-5} more rows ({len(t.rows)} total)")


if __name__ == "__main__":
    main()
