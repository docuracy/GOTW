#!/usr/bin/env python3
"""Export the full transcribed corpus as a chunked, lazy-loadable reader store for the map's modal.

Per volume, entries are taken in reading order (`seq`) and split into fixed-size JSON chunks; a
manifest gives the chunk count + a page->chunk index so the reader can jump to any printed page. The
map's "Read full entry" button opens the modal at the clicked place's volume+page and scrolls to it,
then lazy-loads neighbouring chunks as you scroll — a whole-volume reader. Text is our own Surya OCR
(public domain); tables (once digitised into table_data) render inline from the structured columns/rows.

Static + GitHub-Pages-friendly:
  docs/reader/<vol>/manifest.json  {vol, title, chunks, chunk_size, count, page_index:{page:chunk}}
  docs/reader/<vol>/<c>.json       [ {eid, hw, p, k, text, tables?}, … ]   (k: 'e'ntry | 'c'rossref)

  python3 process/export_reader.py                 # all volumes
  python3 process/export_reader.py --vol v5        # one volume (e.g. the current demo)
"""
from __future__ import annotations
import argparse, json, re, sqlite3
from pathlib import Path

VOLNUM = re.compile(r"v(\w+?)(?:-ocr)?\.txt$|-v(\w+?)[-.]", re.I)
ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII"}


def vol_tag(filename):
    m = VOLNUM.search(filename or "")
    return (m.group(1) or m.group(2)) if m else None


def load_tables(con, source_filename):
    """entry_id -> [ {title, columns, rows, source_note, footnotes} ], or {} if no table_data yet."""
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='table_data'").fetchone():
        return {}
    cols = {r[1] for r in con.execute("PRAGMA table_info(table_data)")}
    if not {"entry_id", "subject", "columns", "rows", "source_note", "footnotes"}.issubset(cols):
        return {}            # no digitised tables in this DB (table-vision pass not run, or pre-TableSet schema)
    out: dict[int, list] = {}
    for r in con.execute("SELECT entry_id, subject, columns, rows, source_note, footnotes FROM table_data "
                         "WHERE entry_id IS NOT NULL"):
        eid, subject, columns, rows, src, foot = r
        out.setdefault(eid, []).append({
            "title": subject,
            "columns": json.loads(columns) if columns else [],
            "rows": json.loads(rows) if rows else [],
            "source_note": src,
            "footnotes": json.loads(foot) if foot else [],
        })
    return out


def load_page_tables(con, vtag):
    """page(int) -> [table objs] for vision tables stored by volume+page (entry_id NULL). These are
    spliced into reading order by page (the VLM digitises whole pages, not per-entry), parallel to plates."""
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='table_data'").fetchone():
        return {}
    cols = {r[1] for r in con.execute("PRAGMA table_info(table_data)")}
    if not {"volume", "page_start", "columns", "rows", "source_note", "footnotes"}.issubset(cols):
        return {}
    out: dict[int, list] = {}
    for r in con.execute("SELECT page_start, subject, columns, rows, source_note, footnotes FROM table_data "
                         "WHERE source='vision' AND volume=? AND page_start IS NOT NULL", (vtag,)):
        p, subject, columns, rows, src, foot = r
        out.setdefault(p, []).append({
            "title": subject,
            "columns": json.loads(columns) if columns else [],
            "rows": json.loads(rows) if rows else [],
            "source_note": src,
            "footnotes": json.loads(foot) if foot else [],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw_seg.sqlite")
    ap.add_argument("--out-dir", default="docs/reader")
    ap.add_argument("--vol", help="single volume tag, e.g. v5 (default: all)")
    ap.add_argument("--chunk-size", type=int, default=150)
    ap.add_argument("--plates-manifest", default="docs/plates/manifest.json",
                    help="illustration-plate manifest (from export_plates.py) to embed in the reader")
    args = ap.parse_args()
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row
    cs = args.chunk_size
    plates = json.loads(Path(args.plates_manifest).read_text()) if Path(args.plates_manifest).exists() else {}

    sources = con.execute("SELECT source_id, filename FROM source ORDER BY filename").fetchall()
    for s in sources:
        vtag = "v" + (vol_tag(s["filename"]) or "?")
        if args.vol and vtag != args.vol:
            continue
        tables = load_tables(con, s["filename"])
        page_tables = load_page_tables(con, vtag)        # vision tables (entry_id NULL), keyed by page
        rows = con.execute(
            "SELECT entry_id, headword_disp, page_start, kind, text FROM entry "
            "WHERE source_id=? AND kind IN('entry','crossref') AND text IS NOT NULL ORDER BY seq",
            (s["source_id"],)).fetchall()
        if not rows:
            continue
        voldir = Path(args.out_dir) / vtag
        voldir.mkdir(parents=True, exist_ok=True)

        def flush(items, c):
            (voldir / f"{c}.json").write_text(json.dumps(items, ensure_ascii=False))

        entries = []
        for r in rows:
            e = {"eid": r["entry_id"], "hw": r["headword_disp"], "p": r["page_start"],
                 "k": "c" if r["kind"] == "crossref" else "e", "text": r["text"]}
            if r["entry_id"] in tables:
                e["tables"] = tables[r["entry_id"]]
            entries.append(e)

        # Attach page-keyed vision tables (entry_id NULL) to the entry whose page-RANGE covers them — the
        # last entry that started on or before the table's page. A table on page 405 of a long FRANCE
        # essay belongs to FRANCE, whose headword is back on page 400; matching only the table's OWN page
        # orphans every table on a non-headword page (i.e. most of them, since the stat tables cluster in
        # the multi-page country essays). They render inline via the same e["tables"] path as entry tables.
        if page_tables:
            import bisect
            paged = sorted((e for e in entries if e.get("p") is not None), key=lambda e: e["p"])
            starts = [e["p"] for e in paged]
            for pg, ts in page_tables.items():
                j = bisect.bisect_right(starts, pg) - 1
                target = paged[j] if j >= 0 else (paged[0] if paged else None)
                if target is not None:
                    target.setdefault("tables", []).extend(ts)

        # Splice illustration plates into reading order: each goes after the last entry of its
        # `after_page`; any that don't match a page (e.g. front-matter plates) trail at the volume end.
        by_page = {}
        for pl in plates.get(vtag, []):
            by_page.setdefault(str(pl["after_page"]), []).append(
                {"k": "plate", "img": pl["img"], "kind": pl.get("kind"), "title": pl.get("title"), "p": pl["after_page"]})
        if by_page:
            spliced = []
            for i, e in enumerate(entries):
                spliced.append(e)
                nextp = entries[i + 1]["p"] if i + 1 < len(entries) else None
                if e["p"] is not None and e["p"] != nextp:
                    spliced += by_page.pop(str(e["p"]), [])
            for rem in by_page.values():                 # unmatched (after_page None / out of range)
                spliced += rem
            entries = spliced

        page_index = {}                                   # page -> first chunk it appears in (post-splice)
        for i, e in enumerate(entries):
            if e.get("p") is not None:
                page_index.setdefault(str(e["p"]), i // cs)
        nchunks = (len(entries) + cs - 1) // cs
        # Dominant first-letter per chunk — the alphabetical backbone for the A–Z jump. Using the per-chunk
        # majority letter (not the first occurrence) is robust to the out-of-order headwords that "—Also"
        # sub-entries and cross-references scatter through reading order; the result is monotonic A→Z.
        chunk_letters = []
        for c in range(nchunks):
            cnt = {}
            for e in entries[c * cs:(c + 1) * cs]:
                L = next((ch for ch in (e.get("hw") or "").upper() if "A" <= ch <= "Z"), None)
                if L:
                    cnt[L] = cnt.get(L, 0) + 1
            chunk_letters.append(max(cnt, key=cnt.get) if cnt else "")
            flush(entries[c * cs:(c + 1) * cs], c)
        vnum = vtag[1:]
        (voldir / "manifest.json").write_text(json.dumps({
            "vol": vtag, "title": f"A Gazetteer of the World — Vol. {ROMAN.get(vnum, vnum)}",
            "chunks": nchunks, "chunk_size": cs, "count": len(entries),
            "page_index": page_index, "chunk_letters": chunk_letters},
            ensure_ascii=False))
        print(f"{vtag}: {len(entries)} entries -> {nchunks} chunks ({len(page_index)} pages) in {voldir}/")

    # (re)write the top-level index of volumes present, in order — the continuous reader chains through it
    mans = []
    for md in Path(args.out_dir).glob("v*/manifest.json"):
        m = json.loads(md.read_text())
        mans.append({"vol": m["vol"], "title": m["title"], "chunks": m["chunks"], "count": m["count"]})
    mans.sort(key=lambda x: int(re.sub(r"\D", "", x["vol"]) or 0))
    (Path(args.out_dir) / "index.json").write_text(json.dumps(mans, ensure_ascii=False))
    print(f"index.json: {len(mans)} volume(s) -> {sum(x['count'] for x in mans)} entries total")


if __name__ == "__main__":
    main()
