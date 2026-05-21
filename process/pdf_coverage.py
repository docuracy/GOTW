#!/usr/bin/env python3
"""Report the alphabetical coverage of a Volume scan from its OCR text layer.

Source provenance matters: the 1856 first edition is **volumes 1–7** (use these); the
same HathiTrust record also carries an **undated 8–14 set from another edition** (its
v14 reproduces 1856 v7's "Article II") which must NOT be mixed in. Within the right
edition, still verify coverage by each PDF's actual first/last running-head — confirm
the seven volumes tile A–Z with no overlap or gap, rather than trusting the numbering.

    python3 process/pdf_coverage.py data/pdf/gotw-v5.pdf data/pdf/gotw-v7.pdf
"""
from __future__ import annotations
import importlib.util, re, sys
from pathlib import Path
import fitz

pp = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("pp", Path(__file__).with_name("pdf_pages.py")))
pp.__spec__.loader.exec_module(pp)


def running_head(page):
    """The headword running-head (first all-caps-ish line of the OCR text)."""
    for ln in (l.strip() for l in page.get_text().splitlines() if l.strip()):
        if re.fullmatch(r"[A-Z][A-Z .'\-]{1,}", ln):     # running heads are upper-case words
            return ln
    return "?"


def coverage(pdf_path):
    d = fitz.open(pdf_path)
    pi, plates = pp.scan(pdf_path)
    if not pi:                                            # no OCR text layer (e.g. the v7 scan)
        return {"pages": d.page_count, "plates": len(plates), "no_text": True,
                "first_head": "(no OCR text layer)", "last_head": ""}
    nums = sorted(pi)
    return {"pages": d.page_count, "printed": (nums[0], nums[-1]), "no_text": False,
            "first_head": running_head(d[pi[nums[0]]]), "last_head": running_head(d[pi[nums[-1]]]),
            "plates": len(plates)}


def main():
    pdfs = sys.argv[1:] or ["data/pdf/gotw-v5.pdf", "data/pdf/gotw-v7.pdf"]
    rows = []
    for p in pdfs:
        c = coverage(p)
        rows.append((p, c))
        if c["no_text"]:
            print(f"{Path(p).name:16} {c['pages']} pp, {c['plates']} plates — NO OCR text layer; "
                  f"coverage needs vision OCR of first/last page headers")
        else:
            print(f"{Path(p).name:16} printed {c['printed'][0]}–{c['printed'][1]} "
                  f"({c['pages']} pp, {c['plates']} plates)  covers  "
                  f"{c['first_head']}  …  {c['last_head']}")
    # overlap/gap check on the volumes that have a detectable range
    ranged = sorted([r for r in rows if not r[1]["no_text"]], key=lambda r: r[1]["first_head"])
    if len(ranged) > 1:
        print("\ncoverage check (by leading headword):")
        for i, (p, c) in enumerate(ranged):
            note = ""
            if i + 1 < len(ranged):
                nxt = ranged[i + 1][1]["first_head"]
                note = (f"  ⚠ OVERLAP with {nxt}" if c["last_head"][:3] >= nxt[:3]
                        else f"  → gap before {nxt}?")
            print(f"  {c['first_head']:14} → {c['last_head']:14}  {Path(p).name}{note}")
    print("\nUse 1856 1st-edition volumes 1–7 (NOT the undated 8–14 set on the same record). "
          "Verify they tile A–Z by these head-word ranges, with no overlap/gap.")


if __name__ == "__main__":
    main()
