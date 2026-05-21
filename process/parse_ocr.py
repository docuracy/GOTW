#!/usr/bin/env python3
"""Parse our Surya OCR text stream into the `entry` table — the public-domain pipeline's parser.

Input is a merged volume .txt from process/ocr_pages.py: per page a `## p. N (#idx) ####`
marker, optional `<!-- table/figure -->` annotations, then reading-order body lines, with a
form-feed between pages. There is no markup to lean on — entries are delimited only by the
print convention: each begins with an ALL-CAPS headword ("MALABAR, a district of …";
cross-references "LUTTICH. See LIEGE."). We reconstruct the prose flow (dropping page markers
+ running heads, de-hyphenating line wraps) and segment on headword starts.

One row per entry, in the schema the rest of the pipeline (OCR correction → extraction →
reconciliation) expects. This is the project's only parser: we OCR the public-domain scans
ourselves and use no external transcript.

    python3 process/parse_ocr.py data/txt/gotw-v1-ocr.txt --volume v1 --db data/gotw.sqlite
    python3 process/parse_ocr.py data/txt/gotw-v1-ocr.txt --volume v1 --dry-run   # stats only
"""
from __future__ import annotations
import argparse, re, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

# Two consecutive caps => a real headword (rejects abbreviations like "M." / "L.").
TWO_CAPS = re.compile(r"[A-ZÀ-Þ]{2}")
# Cross-ref tail: "… See <TARGET>."
SEE = re.compile(r"\bSee\s+(?:also\s+)?([A-ZÀ-Þ][A-ZÀ-Þ .'()\-,&]*?)\.?\s*$")
ALSO = re.compile(r"—\s*Also\b", re.IGNORECASE)
# Text promising a table (so extraction/QA knows a table belongs here, digitised separately).
TABLE_REF = re.compile(r"\b(?:following|subjoined|annexed|above|below)\s+table\b"
                       r"|\btable\s+(?:exhibits|shows|gives|of the)\b", re.IGNORECASE)
PAGE_MARK = re.compile(r"^## p\. (\S+) \(#(\d+)\)")
# Headword line: leading ALL-CAPS run (multi-word, optional (PARENTHETICAL)), then , or .
HEAD = re.compile(r"^([A-ZÀ-Þ][A-ZÀ-Þ0-9 .'’\-]*?(?: \([^)]*\))?)\s*([,.])\s+(.+)$")

# Connecting particles lower-cased when not the first word of a toponym.
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
    """UPPERCASE gazetteer headword -> readable title case (display/indexing only)."""
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
  seq           INTEGER,
  kind          TEXT,
  headword      TEXT,
  headword_raw  TEXT,
  headword_disp TEXT,
  page_start    INTEGER,
  page_end      INTEGER,
  raw_html      TEXT,
  text          TEXT,
  n_tables      INTEGER,
  n_also        INTEGER,
  table_missing INTEGER,
  see_target    TEXT,
  tokens        INTEGER,
  UNIQUE(source_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_entry_headword ON entry(headword);
CREATE INDEX IF NOT EXISTS idx_entry_kind ON entry(kind);
CREATE TABLE IF NOT EXISTS place (
  place_id     INTEGER PRIMARY KEY,
  entry_id     INTEGER REFERENCES entry(entry_id),
  ordinal      INTEGER,
  name         TEXT,
  extraction   TEXT,
  aat_type_id  TEXT,
  status       TEXT DEFAULT 'pending',
  whg_match_id TEXT,
  lat          REAL,
  lon          REAL,
  created_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_place_entry ON place(entry_id);
CREATE INDEX IF NOT EXISTS idx_place_status ON place(status);
"""


def classify(line: str):
    """Return (headword_raw, delim, rest, kind) if the line starts an entry, else None."""
    m = HEAD.match(line)
    if not m:
        return None
    hw, delim, rest = m.group(1).strip(), m.group(2), m.group(3)
    if not TWO_CAPS.search(hw) or sum(c.isalpha() for c in hw) < 3:
        return None                                 # excludes initialisms like "A.M." / "S.W."
    rl = rest.lstrip()
    if delim == ".":                                # period-led head is only valid as a cross-ref
        if not re.match(r"See\b", rl):
            return None
        return hw, delim, rest, "crossref"
    if not rl[:1].islower():                         # a real descriptor follows the comma in lower-case
        return None
    return hw, delim, rest, "entry"


def pages(text: str):
    """Yield (printed_page|None, [body lines]) per page; drop markers/annotations/running head."""
    for chunk in text.split("\f"):
        page = None
        body = []
        for ln in chunk.splitlines():
            m = PAGE_MARK.match(ln)
            if m:
                page = int(m.group(1)) if m.group(1).isdigit() else None
                continue
            if ln.startswith("<!--") or not ln.strip():
                continue
            body.append(ln.rstrip())
        while body and (re.fullmatch(r"\d{1,4}", body[0].strip())
                        or (classify(body[0]) is None
                            and re.fullmatch(r"[A-ZÀ-Þ][A-ZÀ-Þ.'’\- ]{1,30}", body[0].strip()))):
            body.pop(0)                              # running-head: bare number or lone ALL-CAPS token
        yield page, body


def join_lines(lines):
    """Merge wrapped lines into flowing text, healing end-of-line hyphenation."""
    s = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if s.endswith("-") and ln[:1].islower():
            s = s[:-1] + ln
        else:
            s = (s + " " + ln) if s else ln
    return s


def parse(text: str):
    entries = []
    cur = None

    def flush():
        if cur is None:
            return
        body = join_lines(cur.pop("_lines"))
        full = (cur["headword_raw"] + cur.pop("_delim") + " " + body).strip()
        cur["text"] = full
        cur["tokens"] = len(ENC.encode(full))
        cur["n_also"] = len(ALSO.findall(full))
        cur["n_tables"] = 0                          # tables digitised separately (extract_tables.py)
        cur["table_missing"] = int(bool(TABLE_REF.search(full)))
        cur["page_end"] = cur.pop("_page")
        if cur["kind"] == "crossref":
            mt = SEE.search(body)
            cur["see_target"] = mt.group(1).strip() if mt else None
        entries.append(cur)

    for page, body in pages(text):
        for ln in body:
            c = classify(ln)
            if c:
                hw, delim, rest, kind = c
                flush()
                headword = re.sub(r"\s*\([^)]*\)", "", hw).strip(" .,")
                cur = {
                    "seq": len(entries), "kind": kind, "headword": headword, "headword_raw": hw,
                    "headword_disp": normalise_toponym(hw), "page_start": page, "_page": page,
                    "_delim": delim, "_lines": [rest], "see_target": None,
                }
            elif cur is not None:
                cur["_lines"].append(ln)
                cur["_page"] = page if page is not None else cur["_page"]
    flush()
    return entries


def load(db_path: Path, src_name: str, entries):
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    pgs = [e["page_start"] for e in entries if e["page_start"] is not None]
    n_cross = sum(1 for e in entries if e["kind"] == "crossref")
    con.execute("DELETE FROM entry WHERE source_id IN (SELECT source_id FROM source WHERE filename=?)", (src_name,))
    con.execute("DELETE FROM source WHERE filename=?", (src_name,))
    cur = con.execute(
        "INSERT INTO source(filename,sha256,n_entries,n_crossrefs,page_min,page_max,imported_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (src_name, "", len(entries) - n_cross, n_cross,
         min(pgs) if pgs else None, max(pgs) if pgs else None,
         datetime.now(timezone.utc).isoformat(timespec="seconds")))
    sid = cur.lastrowid
    con.executemany(
        "INSERT INTO entry(source_id,seq,kind,headword,headword_raw,headword_disp,page_start,page_end,"
        "raw_html,text,n_tables,n_also,table_missing,see_target,tokens) "
        "VALUES(:sid,:seq,:kind,:headword,:headword_raw,:headword_disp,:page_start,:page_end,"
        ":raw_html,:text,:n_tables,:n_also,:table_missing,:see_target,:tokens)",
        [{**e, "sid": sid, "raw_html": None} for e in entries])
    con.commit()
    return con, sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ocr", type=Path, help="merged OCR .txt from ocr_pages.py")
    ap.add_argument("--volume", required=True, help="volume tag, e.g. v1 (used as source name)")
    ap.add_argument("--db", type=Path, default=Path("data/gotw.sqlite"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.ocr.exists():
        sys.exit(f"not found: {args.ocr}")

    entries = parse(args.ocr.read_text(encoding="utf-8"))
    n_entry = sum(1 for e in entries if e["kind"] == "entry")
    n_cross = sum(1 for e in entries if e["kind"] == "crossref")
    extra = sum(e["n_also"] for e in entries)
    pgs = [e["page_start"] for e in entries if e["page_start"] is not None]
    print(f"{args.ocr.name}: {n_entry:,} entries · {n_cross:,} cross-refs · "
          f"pages {min(pgs) if pgs else '?'}–{max(pgs) if pgs else '?'} · "
          f"+{extra:,} '—Also' places · {sum(e['tokens'] for e in entries):,} tokens")
    print("  sample headwords:", ", ".join(e["headword_disp"] for e in entries[:8]))
    if args.dry_run:
        return
    src_name = f"gotw-{args.volume}-ocr.txt"
    con, _ = load(args.db, src_name, entries)
    print(f"loaded -> {args.db} (source '{src_name}')")
    con.close()


if __name__ == "__main__":
    main()
