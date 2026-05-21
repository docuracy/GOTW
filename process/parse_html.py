#!/usr/bin/env python3
"""Parse a Gazetteer-of-the-World HTML transcript into the SQLite working store.

The transcripts are a single long flow of top-level <p> elements with:
  * headword entries          e.g. "<p>LUSBY, a parish of Lincolnshire ...</p>"
  * cross-references          e.g. "<p>LUTTICH. See LIEGE.</p>"
  * continuation paragraphs   prose / <em>section</em> headers belonging to the
                              preceding place (merged into it here)
  * statistical <table>s      attached to the preceding place
  * page markers "[[N]]"      inside <br/><br/> runs; these can fall MID-paragraph,
                              gluing the tail of one place onto a new headword, so
                              we split on them rather than trusting <p> boundaries.

One row per logical *entry* (headword block or cross-ref). The `place` table is
left for the LLM extraction stage, which splits multi-place entries ("—Also ...").

Usage: python3 process/parse_html.py data/html/gotw_vol5_all_web_v1.html [--db data/gotw.sqlite]
"""
from __future__ import annotations
import argparse, hashlib, re, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup
import tiktoken

PAGE_MARKER = re.compile(r"\[\[(\d+)\]\]")
# Headword: leading run of caps/spaces/punct, optional (PARENTHETICAL), then , or .
HEADWORD = re.compile(r"^\s*([A-ZÀ-Þ][A-ZÀ-Þ0-9.'\- ]*(?:\([^)]*\))?)\s*[,.]")
# Two consecutive caps => a real headword (rejects abbreviations like "M." / "L.")
TWO_CAPS = re.compile(r"[A-ZÀ-Þ]{2}")
# Cross-ref: the *whole* fragment is "<headword>. See <TARGET>." (anchored, short).
SEE = re.compile(r"\s*See\s+(?:also\s+)?([A-ZÀ-Þ][A-ZÀ-Þ .'()\-,&]*?)\.?\s*$")
ALSO = re.compile(r"(?:—|&mdash;)\s*Also\b", re.IGNORECASE)
# Text that promises a table — used to spot tables omitted during transcription.
TABLE_REF = re.compile(r"\b(?:following|subjoined|annexed|above|below)\s+table\b"
                       r"|\btable\s+(?:exhibits|shows|gives|of the)\b", re.IGNORECASE)

ENC = tiktoken.get_encoding("cl100k_base")

# Connecting particles lower-cased when not the first word of a toponym
# (French/Spanish/Portuguese/Italian/Dutch/German/English).
PARTICLES = {
    "de", "des", "du", "d", "da", "das", "do", "dos", "del", "della", "delle",
    "dei", "degli", "di", "la", "le", "les", "l", "el", "lo", "los", "las",
    "van", "von", "der", "den", "ten", "ter", "het", "op", "aan",
    "am", "an", "im", "in", "ob", "zu", "zur", "zum", "auf",
    "of", "on", "upon", "the", "and", "by", "sur", "sous", "y", "e",
    "au", "aux", "a", "à", "lès",
}


def _cap(word: str) -> str:
    """Capitalise one alphabetic token, handling elided apostrophes (L'Aquila)."""
    if "'" in word:
        head, _, tail = word.partition("'")
        h = head if head.lower() in {"l", "d", "dell", "all", "sant", "o", "mc"} else head.capitalize()
        return f"{h.capitalize()}'{tail.capitalize()}"
    return word.capitalize()


def normalise_toponym(raw: str) -> str:
    """UPPERCASE gazetteer headword -> readable title case.

    Lower-cases connecting particles mid-name, preserves hyphenation and any
    (PARENTHETICAL) qualifier. The LLM's extracted `name` remains canonical;
    this is for display/indexing.
    """
    def fix(segment: str) -> str:
        words = segment.split()
        out = []
        for wi, w in enumerate(words):
            comps = w.split("-")
            nc = []
            for ci, c in enumerate(comps):
                first = wi == 0 and ci == 0
                low = c.lower()
                nc.append(_cap(c) if first or low not in PARTICLES else low)
            out.append("-".join(nc))
        return " ".join(out)

    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", raw)
    if m:
        return f"{fix(m.group(1))} ({fix(m.group(2))})"
    return fix(raw)

SCHEMA = """
CREATE TABLE IF NOT EXISTS source (
  source_id   INTEGER PRIMARY KEY,
  filename    TEXT UNIQUE,
  sha256      TEXT,
  n_entries   INTEGER,
  n_crossrefs INTEGER,
  page_min    INTEGER,
  page_max    INTEGER,
  imported_at TEXT
);
CREATE TABLE IF NOT EXISTS entry (
  entry_id      INTEGER PRIMARY KEY,
  source_id     INTEGER REFERENCES source(source_id),
  seq           INTEGER,        -- order within source
  kind          TEXT,           -- 'entry' | 'crossref'
  headword      TEXT,           -- UPPERCASE leading token (no parenthetical)
  headword_raw  TEXT,           -- as printed, incl (PARENTHETICAL)
  headword_disp TEXT,           -- title-cased for display (e.g. Lus-la-Croix-Haute)
  page_start    INTEGER,
  page_end      INTEGER,
  raw_html      TEXT,           -- headword fragment + continuations + tables
  text          TEXT,           -- plain text
  n_tables      INTEGER,        -- tables present in HTML
  n_also        INTEGER,        -- '—Also' boundaries => candidate extra places
  table_missing INTEGER,        -- 1 if text promises a table but none present
  see_target    TEXT,           -- cross-ref destination headword(s)
  tokens        INTEGER,        -- tiktoken cl100k estimate
  UNIQUE(source_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_entry_headword ON entry(headword);
CREATE INDEX IF NOT EXISTS idx_entry_kind ON entry(kind);

-- Populated by the later LLM extraction stage; the Place is the unit of interest.
CREATE TABLE IF NOT EXISTS place (
  place_id     INTEGER PRIMARY KEY,
  entry_id     INTEGER REFERENCES entry(entry_id),
  ordinal      INTEGER,         -- 1-based position within the entry
  name         TEXT,
  extraction   TEXT,            -- JSON: LLM structured output
  aat_type_id  TEXT,            -- Getty AAT concept id (feature type)
  status       TEXT DEFAULT 'pending',  -- pending|extracted|reconciled|failed
  whg_match_id TEXT,
  lat          REAL,
  lon          REAL,
  created_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_place_entry ON place(entry_id);
CREATE INDEX IF NOT EXISTS idx_place_status ON place(status);
"""


def iter_fragments(body):
    """Yield (kind, page, html, node) in document order.

    Splits each <p> on [[N]] page markers so mid-paragraph headwords surface as
    their own fragments. Tables are emitted as ('table', page, html, node).
    """
    page = None
    for child in body.children:
        name = getattr(child, "name", None)
        if name == "table":
            yield "table", page, str(child), child
        elif name == "p":
            inner = child.decode_contents()
            parts = PAGE_MARKER.split(inner)  # [text, num, text, num, ...]
            # parts[0] is text before first marker; odd indices are page numbers
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    page = int(part)
                    continue
                if part.strip():
                    yield "frag", page, part, None


def clean_text(html: str) -> str:
    return BeautifulSoup(html, "lxml").get_text(" ", strip=True)


def parse(path: Path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    entries = []           # list of dicts
    cur = None             # current entry being accumulated

    def flush():
        if cur is None:
            return
        text = clean_text(cur["raw_html"])
        cur["text"] = text
        cur["tokens"] = len(ENC.encode(text))
        cur["n_also"] = len(ALSO.findall(cur["raw_html"]))
        cur["table_missing"] = int(bool(TABLE_REF.search(text)) and cur["n_tables"] == 0)
        cur["page_end"] = cur["_page"]
        entries.append(cur)

    for kind, page, html, _ in iter_fragments(soup.body):
        if kind == "table":
            if cur is not None:
                cur["raw_html"] += "\n" + html
                cur["n_tables"] += 1
                cur["_page"] = page if page is not None else cur["_page"]
            continue

        text = clean_text(html)
        if not text:
            continue
        head = HEADWORD.match(text)
        # A real headword has >=2 consecutive caps (excludes "M." / "L." abbreviations)
        if head and TWO_CAPS.search(head.group(1)):
            rest = text[head.end():]
            see = SEE.match(rest) if len(rest) < 80 else None
            flush()
            headword_raw = head.group(1).strip()
            headword = re.sub(r"\s*\([^)]*\)", "", headword_raw).strip(" .,")
            cur = {
                "seq": len(entries),
                "kind": "crossref" if see else "entry",
                "headword": headword,
                "headword_raw": headword_raw,
                "headword_disp": normalise_toponym(headword_raw),
                "page_start": page,
                "_page": page,
                "raw_html": html,
                "n_tables": 0,
                "see_target": see.group(1).strip() if see else None,
            }
        else:
            # continuation prose / section header -> belongs to the current place
            if cur is None:
                continue
            cur["raw_html"] += "\n" + html
            cur["_page"] = page if page is not None else cur["_page"]
    flush()
    return entries


def load(db_path: Path, src_path: Path, entries):
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    sha = hashlib.sha256(src_path.read_bytes()).hexdigest()
    pages = [e["page_start"] for e in entries if e["page_start"] is not None]
    n_cross = sum(1 for e in entries if e["kind"] == "crossref")
    con.execute("DELETE FROM entry WHERE source_id IN "
                "(SELECT source_id FROM source WHERE filename=?)", (src_path.name,))
    con.execute("DELETE FROM source WHERE filename=?", (src_path.name,))
    cur = con.execute(
        "INSERT INTO source(filename,sha256,n_entries,n_crossrefs,page_min,page_max,imported_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (src_path.name, sha, len(entries) - n_cross, n_cross,
         min(pages) if pages else None, max(pages) if pages else None,
         datetime.now(timezone.utc).isoformat(timespec="seconds")))
    sid = cur.lastrowid
    con.executemany(
        "INSERT INTO entry(source_id,seq,kind,headword,headword_raw,headword_disp,page_start,page_end,"
        "raw_html,text,n_tables,n_also,table_missing,see_target,tokens) "
        "VALUES(:sid,:seq,:kind,:headword,:headword_raw,:headword_disp,:page_start,:page_end,"
        ":raw_html,:text,:n_tables,:n_also,:table_missing,:see_target,:tokens)",
        [{**e, "sid": sid} for e in entries])
    con.commit()
    return con, sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html", type=Path)
    ap.add_argument("--db", type=Path, default=Path("data/gotw.sqlite"))
    args = ap.parse_args()
    if not args.html.exists():
        sys.exit(f"not found: {args.html}")
    args.db.parent.mkdir(parents=True, exist_ok=True)

    entries = parse(args.html)
    con, sid = load(args.db, args.html, entries)

    q = lambda s: con.execute(s, (sid,)).fetchone()
    n_entry = q("SELECT COUNT(*) FROM entry WHERE source_id=? AND kind='entry'")[0]
    n_cross = q("SELECT COUNT(*) FROM entry WHERE source_id=? AND kind='crossref'")[0]
    n_tbl = q("SELECT COALESCE(SUM(n_tables),0) FROM entry WHERE source_id=?")[0]
    n_miss = q("SELECT COUNT(*) FROM entry WHERE source_id=? AND table_missing=1")[0]
    n_multi = q("SELECT COUNT(*) FROM entry WHERE source_id=? AND n_also>0")[0]
    extra = q("SELECT COALESCE(SUM(n_also),0) FROM entry WHERE source_id=?")[0]
    toks = q("SELECT COALESCE(SUM(tokens),0) FROM entry WHERE source_id=?")[0]
    pg = q("SELECT page_min,page_max FROM source WHERE source_id=?")
    print(f"DB: {args.db}")
    print(f"  entries (headword) : {n_entry:,}")
    print(f"  cross-references   : {n_cross:,}")
    print(f"  pages covered      : {pg[0]}–{pg[1]}")
    print(f"  tables captured    : {n_tbl:,}")
    print(f"  tables MISSING*    : {n_miss:,}  (text promises a table, none in HTML)")
    print(f"  multi-place entries: {n_multi:,}  (+{extra:,} extra places via '—Also')")
    print(f"  est. places total  : {n_entry + extra:,}")
    print(f"  body tokens (cl100k): {toks:,}")
    con.close()


if __name__ == "__main__":
    main()
