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
import argparse, difflib, json, re, sqlite3, unicodedata
from collections import defaultdict
from pathlib import Path

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


def fold(s):
    """accent-folded, lower-cased key — matches process/build_toponym_dict.fold()."""
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z]", "", "".join(c for c in s if not unicodedata.combining(c)).lower())


def load_toponyms(path="dict/toponyms.json"):
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return {}
    return {k: v["canonical"] for k, v in raw.items()}      # folded key -> canonical spelling


def has_diacritic(s):
    return any(ord(c) > 127 for c in s)


def clean_hathi(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    t = re.sub(r"##\s*p\.[^\n]*\n", "\n", t)               # page markers
    t = re.sub(r"([A-Za-zÀ-ÿ])-\n([a-zà-ÿ])", r"\1\2", t)   # de-hyphenate end-of-line
    t = re.sub(r"(?m)^[A-Z][A-Z .'\-]{1,30}$", "", t)       # running-head ALLCAPS lines
    t = t.replace("|", " ")                                 # inline + standalone gutter marks
    return t


def is_fault(tok):
    return tok.isupper() and len(alpha(tok)) > 2            # a leaked running-head / headword token


def correct_entry(htoks, ktoks, topo):
    """Return (corrected_tokens, applied[(h,new,reason)], flags[(h,k,kind)]).

    Decision order per aligned 1:1 difference (after number / similarity guards):
      a. same toponym, spelling variant (folds match, known): prefer the diacritic'd /
         canonical form  → e.g. Isere↔Isère.
      b. one side is a known toponym, the other isn't: the known toponym wins
         → Bastogue→Bastogne (Hathi right) ; Mirebean kept as Mirebeau (Humphrey right,
           because Mirebean isn't a toponym either — falls through to flag, Humphrey kept).
      c. Humphrey non-word → Hathi general-dictionary word  → tho→the.
      d. Humphrey already a real word → keep (ocean over occan).
      e. otherwise FLAG, don't apply (arc/are; neither in any dictionary)."""
    out = list(htoks)
    applied, flags = [], []
    hk = [alpha(t).lower() for t in htoks]
    kk = [alpha(t).lower() for t in ktoks]

    def swap(i, h, ah, new_core, reason):
        new = h.replace(ah, new_core, 1)
        if new != h:
            out[i] = new
            applied.append((h, new, reason))

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, hk, kk).get_opcodes():
        if tag != "replace" or (i2 - i1) != 1 or (j2 - j1) != 1:
            continue
        h, k = htoks[i1], ktoks[j1]
        ah, ak = alpha(h), alpha(k)
        if not ah or not ak or ah.lower() == ak.lower() or is_fault(k):
            continue
        if NUMERIC.search(h) or NUMERIC.search(k):
            continue                                        # never swap measurements/numbers
        if ratio(ah.lower(), ak.lower()) < 0.7 or abs(len(ah) - len(ak)) > 3:
            continue                                        # alignment slip, not an OCR variant
        fh, fk = fold(ah), fold(ak)
        cap = ah[:1].isupper()                              # toponyms-in-text are capitalised
        th, tk = cap and fh in topo, cap and fk in topo
        if fh == fk and (th or tk):                         # (a) same toponym, spelling variant
            if has_diacritic(ak) and not has_diacritic(ah):
                swap(i1, h, ah, ak, "toponym-diacritic")    # accent restored from Hathi (contextual token)
            # else keep Humphrey (no clear improvement)
        elif tk and not th:                                 # (b) Hathi is the known toponym
            swap(i1, h, ah, ak, "toponym")                  # use Hathi's contextual token, not the ALL-CAPS canonical
        elif th and not tk:                                 # (b) Humphrey is the known toponym -> keep
            pass
        elif is_word(ak) and not is_word(ah):               # (c) Humphrey misread a real word
            swap(i1, h, ah, ak, "word")
        elif is_word(ah) and not is_word(ak):               # (d) Humphrey already correct
            pass
        else:                                               # (e) unresolved
            flags.append((h, k, "toponym" if not (is_word(ah) or is_word(ak)) else "both-words"))
    return out, applied, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--hathi", default="data/txt/gotw-v5.txt")
    args = ap.parse_args()
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row
    con.execute("ALTER TABLE entry ADD COLUMN text_corrected TEXT") if "text_corrected" not in \
        {r[1] for r in con.execute("PRAGMA table_info(entry)")} else None
    con.executescript("DROP TABLE IF EXISTS corrections;"
                      "CREATE TABLE corrections (entry_id INTEGER, before TEXT, after TEXT, reason TEXT);"
                      "CREATE TABLE IF NOT EXISTS ocr_flags (entry_id INTEGER, humphrey TEXT, hathi TEXT, kind TEXT);")
    con.execute("DELETE FROM ocr_flags")
    if not WORDS:
        print("⚠ no general dictionary (dict/words or /usr/share/dict/words) — word fixes disabled")
    topo = load_toponyms()
    print(f"toponym authority: {len(topo):,} entries" if topo else "⚠ no dict/toponyms.json — toponym resolution off")

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
        out, applied, flags = correct_entry(htoks, window, topo)
        aligned += 1
        total_changes += len(applied)
        con.execute("UPDATE entry SET text_corrected=? WHERE entry_id=?", (" ".join(out), r["entry_id"]))
        for b, a, reason in applied:
            con.execute("INSERT INTO corrections(entry_id,before,after,reason) VALUES(?,?,?,?)",
                        (r["entry_id"], b, a, reason))
            if reason.startswith("toponym") and len(samples) < 25:
                samples.append((r["headword"], b, a))
        for h, k, kind in flags:
            con.execute("INSERT INTO ocr_flags(entry_id,humphrey,hathi,kind) VALUES(?,?,?,?)",
                        (r["entry_id"], h, k, kind))
    con.commit()

    nflag = con.execute("SELECT count(*) FROM ocr_flags").fetchone()[0]
    print(f"entries: {len(rows)}  | aligned to Hathi: {aligned}  | no anchor (kept as-is): {no_anchor}")
    print(f"corrections APPLIED: {total_changes}  | still FLAGGED (not applied): {nflag}")
    print("  by reason:")
    for reason, c in con.execute("SELECT reason,count(*) FROM corrections GROUP BY reason ORDER BY 2 DESC"):
        print(f"    {reason:18} {c}")
    print("\nsample toponym resolutions (Humphrey -> authority):")
    for hw, b, a in samples[:14]:
        print(f"  {hw[:18]:18} {b!r:>15} -> {a!r}")
    print("\ntop recurring corrections:")
    for b, a, c in con.execute("SELECT before,after,count(*) c FROM corrections GROUP BY before,after "
                              "ORDER BY c DESC LIMIT 8"):
        print(f"  {c:4}×  {b!r} -> {a!r}")
    print("\ntop remaining FLAGGED disagreements (→ corpus tally / LLM):")
    for h, k, c in con.execute("SELECT humphrey,hathi,count(*) c FROM ocr_flags "
                              "GROUP BY humphrey,hathi ORDER BY c DESC LIMIT 8"):
        print(f"  {c:4}×  Humphrey {h!r}  vs  Hathi {k!r}")


if __name__ == "__main__":
    main()
