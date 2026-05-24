#!/usr/bin/env python3
"""Build a static SQLite + FTS5 full-text index of the whole transcribed gazetteer, to be queried in
the browser via HTTP range requests (sql.js-httpvfs) — no server, GitHub-Pages-friendly. Only the DB
pages a query touches are fetched, so a multi-MB index loads nothing up front.

Generalises to any ingested gazetteer: it reads the `entry`/`source` tables and writes, per entry, a
**reader locator** (volume, printed page, reader-chunk index) alongside an FTS5 index over the headword
and body, so a search hit deep-links straight into the reader modal. The body is kept in `doc` as FTS5
external content, so `snippet()`/`highlight()` work for result previews.

  python3 process/build_search_db.py            # data/gotw_seg.sqlite -> docs/search/gotw-fts.sqlite
"""
from __future__ import annotations
import argparse, re, sqlite3
from pathlib import Path

VOL = re.compile(r"v(\w+?)(?:-ocr)?\.txt$|-v(\w+?)[-.]", re.I)


def vtag(fn):
    m = VOL.search(fn or "")
    return "v" + ((m.group(1) or m.group(2)) if m else "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw_seg.sqlite")
    ap.add_argument("--out", default="docs/search/gotw-fts.sqlite")
    ap.add_argument("--chunk-size", type=int, default=150, help="MUST match export_reader (for rc -> reader chunk)")
    ap.add_argument("--page-size", type=int, default=4096, help="small pages = less over-fetch per range request")
    args = ap.parse_args()
    src = sqlite3.connect(args.db); src.row_factory = sqlite3.Row
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    o = sqlite3.connect(out)
    o.execute(f"PRAGMA page_size={args.page_size}")   # must precede table creation on an empty DB
    o.execute("PRAGMA journal_mode=DELETE")            # no -wal sidecar; the served file is self-contained
    o.executescript("""
        CREATE TABLE doc(eid INTEGER PRIMARY KEY, vol TEXT, page INTEGER, rc INTEGER, headword TEXT, body TEXT);
        CREATE VIRTUAL TABLE fts USING fts5(headword, body, content='doc', content_rowid='eid',
            tokenize='unicode61 remove_diacritics 2');
        -- trigram index over headwords only (small): powers FUZZY name search — the client splits a
        -- query into overlapping 3-grams and ranks by overlap, so OCR/typo variants still match.
        CREATE VIRTUAL TABLE hw USING fts5(headword, content='doc', content_rowid='eid', tokenize='trigram');
    """)
    n = 0
    for s in src.execute("SELECT source_id, filename FROM source ORDER BY filename"):
        v, i = vtag(s["filename"]), 0
        for r in src.execute("SELECT entry_id, headword_disp, page_start, text FROM entry "
                             "WHERE source_id=? AND kind IN('entry','crossref') AND text IS NOT NULL ORDER BY seq",
                             (s["source_id"],)):
            o.execute("INSERT INTO doc(eid,vol,page,rc,headword,body) VALUES(?,?,?,?,?,?)",
                      (r["entry_id"], v, r["page_start"], i // args.chunk_size, r["headword_disp"], r["text"]))
            i += 1; n += 1
    o.execute("INSERT INTO fts(fts) VALUES('rebuild')")   # build the indexes from doc's external content
    o.execute("INSERT INTO hw(hw) VALUES('rebuild')")
    o.commit()
    o.execute("VACUUM")
    o.commit(); o.close()
    print(f"{n} docs -> {out}  ({out.stat().st_size / 1e6:.1f} MB, page_size {args.page_size})")


if __name__ == "__main__":
    main()
