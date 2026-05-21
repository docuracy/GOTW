#!/usr/bin/env python3
"""Correct Humphrey's OCR using the (cleaner) HathiTrust OCR — deterministic, free, auditable.

Both are OCR of the same Volume V print. Humphrey's HTML has clean structure (reflowed
paragraphs, segmented entries) but n↔u / o↔u-type misreads (Khnsistan, Karon, commnne);
the Hathi .txt reads characters better but has two-column LAYOUT faults (merged columns,
gutter '|', page markers, running heads). So we keep Humphrey's STRUCTURE and import only
Hathi's better CHARACTERS, while staying sceptical of Hathi's faults.

Per entry: clean the Hathi stream, anchor on the headword, align word-tokens with difflib,
and apply ONLY high-confidence single-token swaps (aligned pair differs but alpha-cores are
clearly the same word). Ignore Hathi insertions (column bleed); keep Humphrey on deletes /
multi-token / low-similarity diffs; log every change. Writes entry.text_corrected + a
`corrections` table for QA.

    python3 process/correct_ocr.py --hathi data/txt/gotw-v5.txt
"""
from __future__ import annotations
import argparse, difflib, re, sqlite3
from collections import defaultdict

TOKEN = re.compile(r"\S+")
ALPHA = re.compile(r"[^0-9A-Za-zÀ-ÿ]")
NUMERIC = re.compile(r"[0-9½¼¾⅓⅔⅛⅜⅝⅞]")
DICT_PATHS = ["dict/words", "/usr/share/dict/words"]


def load_words():
    for p in DICT_PATHS:
        try:
            return {w.strip().lower() for w in open(p, encoding="utf-8", errors="replace") if w.strip()}
        except OSError:
            continue
    return set()


WORDS = load_words()


def alpha(t):
    return ALPHA.sub("", t)


def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def is_word(core):
    c = core.lower()
    return bool(c) and (c in WORDS or (c.endswith("s") and c[:-1] in WORDS))


def clean_hathi(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    t = re.sub(r"##\s*p\.[^\n]*\n", "\n", t)               # page markers
    t = re.sub(r"([A-Za-zÀ-ÿ])-\n([a-zà-ÿ])", r"\1\2", t)   # de-hyphenate end-of-line
    t = re.sub(r"(?m)^[A-Z][A-Z .'\-]{1,30}$", "", t)       # running-head ALLCAPS lines
    t = t.replace("|", " ")                                 # inline + standalone gutter marks
    return t


def is_fault(tok):
    return tok.isupper() and len(alpha(tok)) > 2            # a leaked running-head / headword token


def correct_entry(htoks, ktoks):
    """Return (corrected_tokens, applied[(h,new)], flags[(h,k,kind)]).

    Apply ONLY the safe case — Humphrey non-word → Hathi real word (e.g. tho→the,
    commodions→commodious). Never touch numbers. FLAG (don't apply) proper-noun
    disagreements (neither in the dictionary — e.g. Mirebeau/Mirebean) and both-word
    ambiguities (arc/are): those go to the toponym dictionary / LLM, not a blind swap."""
    out = list(htoks)
    applied, flags = [], []
    hk = [alpha(t).lower() for t in htoks]
    kk = [alpha(t).lower() for t in ktoks]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, hk, kk).get_opcodes():
        if tag != "replace" or (i2 - i1) != 1 or (j2 - j1) != 1:
            continue                                        # only 1:1 replaces; ignore ins/del/multi
        h, k = htoks[i1], ktoks[j1]
        ah, ak = alpha(h), alpha(k)
        if not ah or not ak or ah.lower() == ak.lower() or is_fault(k):
            continue
        if NUMERIC.search(h) or NUMERIC.search(k):
            continue                                        # never swap measurements/numbers
        if ratio(ah.lower(), ak.lower()) < 0.7 or abs(len(ah) - len(ak)) > 3:
            continue                                        # not an OCR variant — alignment slip
        hw, kw = is_word(ah), is_word(ak)
        if kw and not hw:                                   # Humphrey misread a real word -> fix
            new = h.replace(ah, ak, 1)
            if new != h:
                out[i1] = new
                applied.append((h, new))
        elif not hw and not kw:
            flags.append((h, k, "toponym"))                 # proper-noun disagreement (toponym candidate)
        elif hw and kw:
            flags.append((h, k, "both-words"))              # ambiguous (e.g. arc/are) — leave to LLM
        # hw and not kw: Humphrey already has the real word -> keep, no flag
    return out, applied, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--hathi", default="data/txt/gotw-v5.txt")
    args = ap.parse_args()
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row
    con.execute("ALTER TABLE entry ADD COLUMN text_corrected TEXT") if "text_corrected" not in \
        {r[1] for r in con.execute("PRAGMA table_info(entry)")} else None
    con.executescript("CREATE TABLE IF NOT EXISTS corrections (entry_id INTEGER, before TEXT, after TEXT);"
                      "CREATE TABLE IF NOT EXISTS ocr_flags (entry_id INTEGER, humphrey TEXT, hathi TEXT, kind TEXT);")
    con.execute("DELETE FROM corrections"); con.execute("DELETE FROM ocr_flags")
    if not WORDS:
        print("⚠ no dictionary found (dict/words or /usr/share/dict/words) — corrections disabled")

    print("cleaning Hathi stream …")
    ktoks_all = TOKEN.findall(clean_hathi(args.hathi))
    klow = [alpha(t).lower() for t in ktoks_all]
    pos = defaultdict(list)                                 # headword token -> positions in Hathi
    for i, w in enumerate(klow):
        if len(w) >= 3:
            pos[w].append(i)

    rows = con.execute("SELECT entry_id, headword, text FROM entry WHERE kind='entry' AND text IS NOT NULL").fetchall()
    aligned = total_changes = no_anchor = 0
    samples = []
    for r in rows:
        htoks = TOKEN.findall(r["text"])
        if not htoks:
            continue
        key = alpha(htoks[0]).lower()
        cands = pos.get(key, [])
        if not cands:
            no_anchor += 1
            con.execute("UPDATE entry SET text_corrected=? WHERE entry_id=?", (r["text"], r["entry_id"]))
            continue
        # pick the Hathi anchor whose following tokens best match the entry's opening
        probe = [alpha(t).lower() for t in htoks[1:5]]
        best = max(cands, key=lambda p: ratio(probe, klow[p + 1:p + 1 + len(probe)]))
        window = ktoks_all[best: best + int(len(htoks) * 1.8) + 10]
        out, applied, flags = correct_entry(htoks, window)
        aligned += 1
        total_changes += len(applied)
        con.execute("UPDATE entry SET text_corrected=? WHERE entry_id=?", (" ".join(out), r["entry_id"]))
        for b, a in applied:
            con.execute("INSERT INTO corrections(entry_id,before,after) VALUES(?,?,?)", (r["entry_id"], b, a))
            if len(samples) < 25:
                samples.append((r["headword"], b, a))
        for h, k, kind in flags:
            con.execute("INSERT INTO ocr_flags(entry_id,humphrey,hathi,kind) VALUES(?,?,?,?)",
                        (r["entry_id"], h, k, kind))
    con.commit()

    nflag = con.execute("SELECT count(*) FROM ocr_flags").fetchone()[0]
    print(f"entries: {len(rows)}  | aligned to Hathi: {aligned}  | no anchor (kept as-is): {no_anchor}")
    print(f"corrections APPLIED (Humphrey non-word → Hathi word): {total_changes}")
    print(f"disagreements FLAGGED (not applied): {nflag}")
    print("\nsample applied corrections:")
    for hw, b, a in samples[:18]:
        print(f"  {hw[:18]:18} {b!r:>15} -> {a!r}")
    print("\ntop recurring applied corrections:")
    for b, a, c in con.execute("SELECT before,after,count(*) c FROM corrections GROUP BY before,after "
                              "ORDER BY c DESC LIMIT 10"):
        print(f"  {c:4}×  {b!r} -> {a!r}")
    print("\ntop FLAGGED toponym disagreements (→ feed the toponym dictionary):")
    for h, k, c in con.execute("SELECT humphrey,hathi,count(*) c FROM ocr_flags WHERE kind='toponym' "
                              "GROUP BY humphrey,hathi ORDER BY c DESC LIMIT 10"):
        print(f"  {c:4}×  Humphrey {h!r}  vs  Hathi {k!r}")


if __name__ == "__main__":
    main()
