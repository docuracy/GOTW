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

    # canonical (stale-proof): build straight from the per-page OCR dir — the single source of truth
    python3 process/parse_ocr.py --ocr-dir ocr/v1 --volume v1 --db data/gotw.sqlite
    # legacy: a pre-merged .txt (parse warns unless it carries an ocr-merge provenance header)
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
SEE = re.compile(r"\bSee\s+(?:also\s+)?([A-ZÀ-Þ][A-Za-zÀ-ÿ .'()\-,&]*?)\.?\s*$")
ALSO = re.compile(r"—\s*Also\b", re.IGNORECASE)
# Text promising a table (so extraction/QA knows a table belongs here, digitised separately).
TABLE_REF = re.compile(r"\b(?:following|subjoined|annexed|above|below)\s+table\b"
                       r"|\btable\s+(?:exhibits|shows|gives|of the)\b", re.IGNORECASE)
PAGE_MARK = re.compile(r"^## p\. (\S+) \(#(\d+)\)")
# Headword line: leading ALL-CAPS run (multi-word, optional (PARENTHETICAL)), then , or .
HEAD = re.compile(r"^([A-ZÀ-Þ][A-ZÀ-Þ0-9 .'’\-]*?(?: \([^)]*\))?)\s*([,.])\s+(.+)$")
# Standalone DISPLAY heading for a major (multi-page) entry: the whole line is an ALL-CAPS
# headword, optionally a (PARENTHETICAL), optionally a trailing comma/period, and NOTHING else.
# (e.g. "AF'GHANISTAN." / "AMERICA," — the description opens on the following line.)
STANDALONE = re.compile(r"^([A-ZÀ-Þ][A-ZÀ-Þ0-9 .'’\-]*?(?: \([^)]*\))?)[,.]?$")
# Page artifacts that interrupt entries at page breaks (never a heading; skipped when seeking prose):
# the Google-scan watermark + library/accession stamps in this particular (ship's-library) copy.
WATERMARK = re.compile(r"digiti[sz]ed by go\w+|^goo\w+$|original from"
                       r"|(?:uni)?versity of min\w*|^y? ?of minnesot\w*"   # incl. OCR-split/truncated
                       r"|received on board|homeward voyage|^(orders|landed|sailed)[.,]?$", re.I)
# Volume title-page / publisher colophon block (between volumes) — never gazetteer entries.
COLOPHON = re.compile(r"fullarton|gazetteer of the world|dictionary of geograph|newgate street"
                      r"|eustace street|leith walk|stead.?s place|\bprinters\b", re.I)
# OCR garble to scrub from entry text (removes inserted noise; does not reinterpret real readings):
JUNK_RUN = re.compile(r"(.)\1{7,}")               # 8+ repeated chars, e.g. "PPPPPPPP"
JUNK_DATE = re.compile(r"(?:[-–]\d{1,4}){3,}")    # corrupt repeated "-01-01-01…" groups


# An inline running-head intruding mid-prose: a lowercase word, then an ALL-CAPS toponym + ".",
# then a lowercase word ("oil and bone NEW ZEALAND. to chance vessels" -> "oil and bone to chance").
INLINE_HEAD = re.compile(r"(?<=[a-zà-ÿ] )[A-ZÀ-Þ]{3,}[A-ZÀ-Þ '’\-]*\.\s+(?=[a-zà-ÿ])")


def clean_text(s: str) -> str:
    """Scrub inserted OCR/scan garble (stamp phrases, repeated-char runs, date corruption,
    running-heads dropped mid-sentence)."""
    s = JUNK_RUN.sub(lambda m: m.group(1), s)
    s = JUNK_DATE.sub("", s)
    s = WATERMARK.sub("", s)
    s = COLOPHON.sub("", s)
    s = INLINE_HEAD.sub("", s)
    return re.sub(r"\s{2,}", " ", s).strip()
# A numbered/lettered SECTION heading inside a long entry ("V. PASHALIK OF MARASH",
# "IV. THE ETHIOPIAN RACE", "6. AFGHANISTAN, …") — never a toponym headword.
SECTION = re.compile(r"^(?:[IVXLCDM]{1,5}|\d{1,3})\.\s")
# Back-matter boundary: the final volume's APPENDIX (the ancient↔modern name concordance, Articles
# I & II) is NOT gazetteer place entries — it is handled separately by extract_appendix.py
# (vision-LLM → name_variant table). Stop the entry parse when it begins, so its ~thousands of
# ancient-name lines never reach the place classifier. (Guarded by entry count so the title-page
# "…AND APPENDIX." in front-matter doesn't trip it.)
APPENDIX_MARK = re.compile(r"^APPENDIX\.?\s*$", re.I)
# False ALL-CAPS "headwords": compass bearings and roman numerals are never toponyms.
COMPASS = {"N", "S", "E", "W", "NE", "NW", "SE", "SW", "NNE", "NNW", "SSE", "SSW",
           "ENE", "ESE", "WNW", "WSW"}
ROMAN = re.compile(r"^[IVXLCDM]+$")
# Common statistical-table column headers (in country essays) — never toponyms; reject as headwords
# so they don't get promoted to entries and swallow the table (tables are handled by extract_tables.py).
STOPWORDS = {"VALUE", "VALUES", "IMPORTS", "EXPORTS", "TOTAL", "TOTALS", "AMOUNT", "AMOUNTS",
             "QUANTITY", "QUANTITIES", "NUMBER", "NUMBERS", "REVENUE", "EXPENDITURE",
             "POPULATION", "TONS", "ARTICLES", "COMMODITIES", "DESCRIPTION"}


def _alpha(s: str) -> str:
    return re.sub(r"[^A-Z]", "", (s or "").upper())


# Strict roman-numeral validator (so genuine words made of I/V/X/L/C/D/M — MILL, DILI, CIVIL —
# are NOT mistaken for numerals; only true numerals like VIII/XII match).
ROMAN_STRICT = re.compile(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def is_false_headword(hw: str) -> bool:
    """Reject compass bearings, roman numerals, and numbered/lettered section headings."""
    k = _alpha(hw)
    if k in COMPASS or k in STOPWORDS or ROMAN.fullmatch(k) or SECTION.match(hw) or JUNK_RUN.search(hw):
        return True                                 # incl. OCR garble like "PPPPPPPP…"
    # section heading whose roman numeral / digit lost its period ("VIII KINGDOM OF KASAN"):
    # first token is a VALID roman numeral or a number, followed by an ALL-CAPS word.
    m = re.match(r"^([IVXLCDM]+|\d{1,3})\.?\s+[A-ZÀ-Þ]", hw)
    return bool(m) and (m.group(1).isdigit() or bool(ROMAN_STRICT.match(m.group(1))))


def is_heading(hw: str, current_hw: str, nextline: str, prev_complete: bool) -> bool:
    """A standalone ALL-CAPS line opens a major entry when it (a) is alphabetically *after* the
    current entry's headword — so the repeating running-heads (equal to the current entry) and
    backward OCR noise are skipped; (b) follows a *completed* entry rather than interrupting one
    mid-sentence — the running-head 'AFGHANISTAN.' that appears inside the earlier AFFENTHAL entry
    breaks the clause '…in the circle of the | Middle Rhine…', whereas a real heading follows
    '…Pop. 657.'; (c) is a plausible headword followed by prose."""
    cand, cur = _alpha(hw), _alpha(current_hw)
    if len(cand) < 4 or is_false_headword(hw):
        return False
    if len(hw) > 40 or len(hw.split()) > 5:         # a real heading is a name, not a sentence
        return False
    if cand <= cur:                                 # == current ⇒ running-head; < current ⇒ noise
        return False
    if not prev_complete:                           # interrupts an entry mid-sentence ⇒ running-head
        return False
    return bool(nextline) and not STANDALONE.match(nextline)   # must be followed by descriptive prose

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
CREATE TABLE IF NOT EXISTS back_matter (
  id          INTEGER PRIMARY KEY,
  source_id   INTEGER REFERENCES source(source_id),
  kind        TEXT,                                  -- 'appendix' (concordance lists + Ethnology essay)
  text        TEXT,                                  -- retained OCR text for later (vision concordance,
  n_lines     INTEGER                                --  ethnonym mining); NOT classified as places
);
"""


def classify(line: str):
    """Return (headword_raw, delim, rest, kind) if the line starts an entry, else None."""
    m = HEAD.match(line)
    if not m:
        return None
    hw, delim, rest = m.group(1).strip(), m.group(2), m.group(3)
    if not TWO_CAPS.search(hw) or sum(c.isalpha() for c in hw) < 3:
        return None                                 # excludes initialisms like "A.M." / "S.W."
    if is_false_headword(hw):
        return None                                 # compass bearings (WNW) / roman numerals (XVI)
    rl = rest.lstrip()
    # cross-reference, incl. multi-variant forms: "[, or VARIANT.] See TARGET." (e.g.
    # "ZALAD, or ZALA. See SZALAD." — comma-led, but still a cross-ref, not a place entry).
    if re.match(r"(or [^.]+\.\s*)?See\s+[A-ZÀ-Þ]", rl):
        return hw, delim, rest, "crossref"
    if delim == ".":
        # variant cross-ref with "See" omitted: "X. Y." where Y is a bare ALL-CAPS toponym
        # (TRAZ-OZ-MONTES. TRAS-OS-MONTES. / BUKKOLZ. BUCHHOLZ. / KALAVRIA. CALAUREIA.).
        rt = rl.rstrip(" .")
        if rt and TWO_CAPS.search(rt) and not re.search(r"[a-zà-ÿ]", rt) and sum(c.isalpha() for c in rt) >= 3:
            return hw, delim, rest, "crossref"
        return None                                 # period-led, otherwise not an entry
    if not rl[:1].islower():                         # a real descriptor follows the comma in lower-case
        return None
    return hw, delim, rest, "entry"


def pages(text: str):
    """Yield (printed_page|None, [body lines]) per page; drop markers/annotations/running heads,
    scan watermarks + colophon, in-body page numbers; heal hyphenated line-wraps (incl. headwords)."""
    for chunk in text.split("\f"):
        page = None
        raw = []
        for ln in chunk.splitlines():
            m = PAGE_MARK.match(ln)
            if m:
                page = int(m.group(1)) if m.group(1).isdigit() else None
                continue
            if ln.strip() and not ln.startswith("<!--"):
                raw.append(ln.rstrip())
        # 1) heal end-of-line hyphenation FIRST — so split stamps ("…UNI-" / "VERSITY OF MINNESOTA")
        #    and wrapped headwords ("…CACHOE-." / "IRA. See …") rejoin before we filter.
        merged = []
        for ln in raw:
            prev = merged[-1] if merged else ""
            if prev and re.search(r"[A-Za-zÀ-ÿ]-\.?$", prev):
                # heal a true word-wrap (lowercase continuation) OR a wrapped uppercase token —
                # a headword/variant that breaks on an ALL-CAPS token+hyphen ("…CACHOE-.", "…or
                # MEISTYRR-", "BANGOR-IS-Y-"); do NOT glue a prose hyphen onto the NEXT headword.
                caps_wrap = bool(re.search(r"[A-ZÀ-Þ][A-ZÀ-Þ0-9'’.\-]*-\.?$", prev))
                if ln[:1].islower() or caps_wrap:
                    merged[-1] = re.sub(r"-\.?$", "", prev) + ln.lstrip()
                    continue
            merged.append(ln)
        # 2) drop scan watermark / library stamp / colophon, and in-body running page numbers
        body = []
        for ln in merged:
            s = ln.strip()
            if WATERMARK.search(s) or COLOPHON.search(s):
                continue
            if re.fullmatch(r"\d{1,4}", s) and page is not None and abs(int(s) - page) <= 1:
                continue
            body.append(ln)
        while body and (re.fullmatch(r"\d{1,4}", body[0].strip())
                        or (classify(body[0]) is None
                            and re.fullmatch(r"[A-ZÀ-Þ][A-ZÀ-Þ.'’\- ]{1,30}", body[0].strip()))):
            body.pop(0)                              # leading running-head: bare number or lone ALL-CAPS
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
        full = clean_text((cur["headword_raw"] + cur.pop("_delim") + " " + body).strip())
        cur["text"] = full
        cur["tokens"] = len(ENC.encode(full))
        cur["n_also"] = len(ALSO.findall(full))
        cur["n_tables"] = 0                          # tables digitised separately (extract_tables.py)
        cur["table_missing"] = int(bool(TABLE_REF.search(full)))
        cur["page_end"] = cur.pop("_page")
        # a headword whose description IS just "See TARGET" (often split across OCR lines, so the
        # headword stood alone and looked like a heading) is a cross-reference, not a place entry.
        if cur["kind"] == "entry" and re.match(r"\s*See\s+[A-ZÀ-Þ]", body):
            cur["kind"] = "crossref"
        if cur["kind"] == "crossref":
            mt = SEE.search(body)
            cur["see_target"] = mt.group(1).strip() if mt else (body.strip(" .") or None)
        entries.append(cur)

    # Flatten to (page, line) so a standalone display heading can look ahead for its echo line.
    flat = [(page, ln) for page, body in pages(text) for ln in body]

    def echo_line(j):
        """First following line that is real prose, skipping page numbers / Google watermarks."""
        while j < len(flat):
            s = flat[j][1].strip()
            if s and not WATERMARK.search(s) and not re.fullmatch(r"\d{1,4}", s):
                return s
            j += 1
        return ""

    def prev_complete():
        """Does the current entry's text so far end a sentence? (heading vs mid-entry running-head)"""
        if cur is None:
            return True
        for ln2 in reversed(cur["_lines"]):
            t = ln2.strip()
            if not t or WATERMARK.search(t) or re.fullmatch(r"\d{1,4}", t):
                continue
            return t.endswith((".", "!", "?", "”", "’"))
        return True                                  # nothing accumulated yet

    def start(hw, delim, first_line, kind, page):
        nonlocal cur
        flush()
        headword = re.sub(r"\s*\([^)]*\)", "", hw).strip(" .,")
        cur = {
            "seq": len(entries), "kind": kind, "headword": headword, "headword_raw": hw,
            "headword_disp": normalise_toponym(hw), "page_start": page, "_page": page,
            "_delim": delim, "_lines": [first_line] if first_line else [], "see_target": None,
        }

    appendix_at = None
    for i, (page, ln) in enumerate(flat):
        if len(entries) > 200 and APPENDIX_MARK.match(ln.strip()):
            appendix_at = i                          # back-matter Appendix begins — stop classifying,
            break                                    # but RETAIN the text (see back_matter / extract_appendix.py)
        c = classify(ln)
        if c:
            hw, delim, rest, kind = c
            start(hw, delim, rest, kind, page)
            continue
        s = ln.strip()
        m = STANDALONE.match(s)
        if m and not WATERMARK.search(s):            # candidate display heading for a major entry
            hw = m.group(1).strip(" ,.")
            if is_heading(hw, cur["headword"] if cur else "", echo_line(i + 1), prev_complete()):
                start(hw, ".", "", "entry", page)
                continue
            if cur is not None and len(_alpha(hw)) >= 3 and _alpha(hw) == _alpha(cur["headword"]):
                continue                             # inline running-head repeating this entry's name -> drop
        if cur is not None:
            cur["_lines"].append(ln)
            cur["_page"] = page if page is not None else cur["_page"]
    flush()
    back_matter = "\n".join(ln for _, ln in flat[appendix_at:]) if appendix_at is not None else ""
    return entries, back_matter


def load(db_path: Path, src_name: str, entries, back_matter=""):
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
    con.execute("DELETE FROM back_matter WHERE source_id=?", (sid,))
    if back_matter.strip():                          # retain the Appendix (lists + essay) for later use
        con.execute("INSERT INTO back_matter(source_id,kind,text,n_lines) VALUES(?,?,?,?)",
                    (sid, "appendix", back_matter, back_matter.count("\n") + 1))
    con.commit()
    return con, sid


def ocr_input_text(args) -> str:
    """OCR text to parse. Prefer --ocr-dir (merge the per-page files in memory — the single source of
    truth, so a stale merged .txt can never silently feed the pipeline). Else read the given .txt, but
    warn if it lacks the ocr-merge provenance header (i.e. it wasn't produced by the current merge)."""
    if args.ocr_dir:
        import ocr_pages                                  # lazy: only this path needs PIL/ocr_pages
        d = Path(args.ocr_dir)
        if not d.is_dir():
            sys.exit(f"not a directory: {d}")
        return ocr_pages.merged_text(d)
    if not args.ocr or not args.ocr.exists():
        sys.exit("provide a merged OCR .txt, or --ocr-dir DIR (recommended)")
    text = args.ocr.read_text(encoding="utf-8")
    if not text.startswith("<!-- ocr-merge:"):
        print(f"WARNING: {args.ocr} has no ocr-merge provenance header — it may be a stale/foreign "
              f"merge. Prefer --ocr-dir, or re-run `ocr_pages.py --merge` then `--verify`.", file=sys.stderr)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ocr", type=Path, nargs="?", help="merged OCR .txt (legacy; prefer --ocr-dir)")
    ap.add_argument("--ocr-dir", help="per-page OCR dir (e.g. ocr/v1) — canonical, stale-proof input")
    ap.add_argument("--volume", required=True, help="volume tag, e.g. v1 (used as source name)")
    ap.add_argument("--db", type=Path, default=Path("data/gotw.sqlite"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries, back_matter = parse(ocr_input_text(args))
    n_entry = sum(1 for e in entries if e["kind"] == "entry")
    n_cross = sum(1 for e in entries if e["kind"] == "crossref")
    extra = sum(e["n_also"] for e in entries)
    pgs = [e["page_start"] for e in entries if e["page_start"] is not None]
    label = args.ocr_dir or (args.ocr.name if args.ocr else "?")
    print(f"{label}: {n_entry:,} entries · {n_cross:,} cross-refs · "
          f"pages {min(pgs) if pgs else '?'}–{max(pgs) if pgs else '?'} · "
          f"+{extra:,} '—Also' places · {sum(e['tokens'] for e in entries):,} tokens")
    print("  sample headwords:", ", ".join(e["headword_disp"] for e in entries[:8]))
    if back_matter.strip():
        print(f"  back-matter (Appendix) retained: {back_matter.count(chr(10)) + 1:,} lines")
    if args.dry_run:
        return
    src_name = f"gotw-{args.volume}-ocr.txt"
    con, _ = load(args.db, src_name, entries, back_matter)
    print(f"loaded -> {args.db} (source '{src_name}')")
    con.close()


if __name__ == "__main__":
    main()
