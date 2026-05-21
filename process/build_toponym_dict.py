#!/usr/bin/env python3
"""Build a toponym authority dictionary, seeded from the Vol VII Appendix concordance.

The Appendix (extracted by process/extract_appendix.py into `name_variant`) is a list of
authoritative place-name spellings — modern and ancient/mediaeval — so it's an ideal seed
for an OCR-correction authority a general dictionary can't provide (Isère, Bastogne, …).

Output `dict/toponyms.json`, keyed by an accent-folded, lower-cased form, each value:
  { canonical, forms: {printed_form: count}, count, eras: [...] }
The accent-fold key groups OCR variants (Isere/Isère); `canonical` is the best-attested
printed form. A flat set of canonical spellings is also written for fast membership tests.

Self-refining (later): `--from-corpus` ingests toponyms harvested from the entry texts of
all volumes, tallying spellings so the most-populous form wins and the dictionary corrects
itself over time.

    python3 process/build_toponym_dict.py            # seed from the Appendix (name_variant)
"""
from __future__ import annotations
import argparse, json, re, sqlite3, unicodedata
from collections import defaultdict, Counter
from pathlib import Path

OUT = Path("dict/toponyms.json")
# a toponym token: starts upper-case, letters/diacritics/hyphens/apostrophes, len>=3
TOPO = re.compile(r"[A-ZÀ-Þ][A-Za-zÀ-ÿ.'\-]{2,}")


def fold(s: str) -> str:
    """accent-folded, lower-cased key (Isère -> isere); groups OCR/diacritic variants."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.lower())


def add(index, printed, era=None):
    printed = printed.strip(" .,;:")
    if len(re.sub(r"[^A-Za-zÀ-ÿ]", "", printed)) < 3:
        return
    k = fold(printed)
    if not k:
        return
    e = index[k]
    e["forms"][printed] += 1
    if era:
        e["eras"].add(era)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    args = ap.parse_args()
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row

    index = defaultdict(lambda: {"forms": Counter(), "eras": set()})
    rows = con.execute("SELECT direction, headword, equivalents FROM name_variant").fetchall()
    if not rows:
        raise SystemExit("name_variant is empty — run process/extract_appendix.py first")
    for r in rows:
        # Article I: headword=ancient/mediaeval, equivalents=modern; Article II: reversed.
        head_era = "ancient" if r["direction"] == "ancient_to_modern" else "modern"
        eq_era = "modern" if r["direction"] == "ancient_to_modern" else "ancient"
        add(index, r["headword"], head_era)
        for eq in json.loads(r["equivalents"] or "[]"):
            # equivalents may pack several names in one string
            for tok in TOPO.findall(eq):
                add(index, tok, eq_era)

    out = {}
    for k, e in index.items():
        canonical = e["forms"].most_common(1)[0][0]
        out[k] = {"canonical": canonical, "forms": dict(e["forms"]),
                  "count": sum(e["forms"].values()), "eras": sorted(e["eras"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0, sort_keys=True))

    diacritic = sum(1 for v in out.values() if any(ord(c) > 127 for c in v["canonical"]))
    print(f"toponym dictionary: {len(out):,} distinct toponyms -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"  with diacritics: {diacritic:,}")
    print("  spot checks (fold -> canonical):")
    for probe in ("Isère", "Mirebeau", "Bastogne", "Aachen", "Carlow", "Würtemberg"):
        v = out.get(fold(probe))
        print(f"    {probe:12} {'✓ ' + v['canonical'] if v else '— not present'}")


if __name__ == "__main__":
    main()
