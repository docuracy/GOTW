#!/usr/bin/env python3
"""Map printed page numbers to PDF page indices for a Volume scan, and find plates.

The Internet Archive scans carry an OCR text layer whose running header includes the
printed page number (e.g. "MALABAR\\nMALABAR\\n60\\n..."). Inserted steel-plate
illustrations are NOT paginated, so a constant offset fails — we read the number off
each page instead. Plates are detected as pages with little/no body text.

    python3 process/pdf_pages.py data/pdf/gotw-v5.pdf            # report coverage + plates
    from pdf_pages import page_index, plate_pages                # use as a library
"""
from __future__ import annotations
import re, sys
import fitz

PLATE_MAX_TEXT = 120   # chars of text below which a page is treated as a plate/illustration


def _header_pageno(text: str):
    """The printed page number from a page's OCR header (first ~6 non-empty lines)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:6]
    for ln in lines:
        if re.fullmatch(r"\d{1,3}", ln):
            n = int(ln)
            if 1 <= n <= 999:
                return n
    return None


def scan(pdf_path):
    """Return (page_index: {printed -> idx}, plates: [idx], texts: {idx: text})."""
    d = fitz.open(pdf_path)
    page_index, plates = {}, []
    for idx in range(d.page_count):
        text = d[idx].get_text()
        if len(text.strip()) < PLATE_MAX_TEXT:
            plates.append(idx)
            continue
        n = _header_pageno(text)
        if n is not None and n not in page_index:   # first occurrence wins
            page_index[n] = idx
    return page_index, plates


def page_index(pdf_path):
    return scan(pdf_path)[0]


def plate_pages(pdf_path):
    return scan(pdf_path)[1]


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/pdf/gotw-v5.pdf"
    d = fitz.open(pdf)
    pi, plates = scan(pdf)
    nums = sorted(pi)
    print(f"{pdf}: {d.page_count} PDF pages")
    print(f"  printed pages mapped : {len(pi)}  (range {nums[0]}–{nums[-1]})")
    print(f"  plates/illustrations : {len(plates)}  e.g. idx {plates[:12]}")
    # offset drift = idx - printed, should grow as plates accumulate
    for p in (nums[0], nums[len(nums)//2], nums[-1]):
        print(f"    printed {p:>4} -> idx {pi[p]:>4}  (drift {pi[p]-p})")
    miss = [n for n in range(nums[0], nums[-1] + 1) if n not in pi]
    print(f"  gaps in 1..{nums[-1]}: {len(miss)}" + (f"  e.g. {miss[:10]}" if miss else ""))


if __name__ == "__main__":
    main()
