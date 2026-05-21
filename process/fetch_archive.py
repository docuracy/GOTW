#!/usr/bin/env python3
"""Fetch public-domain Gazetteer volumes from the Internet Archive (legitimate, no auth).

archive.org permits programmatic download of public-domain items. This avoids scripting
HathiTrust's authenticated reader (against its ToS, and would risk your institutional
access). Three modes:

  probe   identify what a candidate item covers — metadata + first/last head-word, read
          cheaply from the head/tail of its OCR text via HTTP range requests. Use this to
          map identifiers -> volumes (I–VII) before downloading.
  pdf     download an item's full Text-PDF to data/pdf/.
  pages   download a page range as images via the IIIF endpoint (for image-only items).

  python3 process/fetch_archive.py probe dli.ministry.08185 dli.ministry.08186 ...
  python3 process/fetch_archive.py pdf  agazetteerworld00unkngoog --out data/pdf/gotw-v7.pdf
  python3 process/fetch_archive.py pages <id> --from 1 --to 900 --out data/pages/<id>

Be polite: identifies itself, rate-limits, resumes. Public-domain content only.
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
import requests

UA = {"User-Agent": "GOTW-research/0.1 (https://github.com/docuracy/GOTW; public-domain volumes)"}
HEAD_BYTES = 140_000
RUNHEAD = re.compile(r"^[A-Z][A-Z0-9 .,'’\-&]{2,40}$")
SKIP = {"THE GAZETTEER", "GAZETTEER OF THE WORLD", "DICTIONARY OF GEOGRAPHICAL KNOWLEDGE",
        "APPENDIX", "INDEX", "PREFACE", "CONTENTS"}


def meta(idn):
    return requests.get(f"https://archive.org/metadata/{idn}", headers=UA, timeout=30).json()


def _file(m, pred):
    for f in m.get("files", []):
        if pred(f):
            return f
    return None


def _runheads(text):
    out = []
    for ln in (l.strip() for l in text.splitlines()):
        if RUNHEAD.fullmatch(ln) and sum(c.isalpha() for c in ln) >= 3 and ln not in SKIP:
            out.append(ln)
    return out


def _range(url, lo, hi):
    r = requests.get(url, headers={**UA, "Range": f"bytes={lo}-{hi}"}, timeout=60)
    if r.status_code not in (200, 206) or "text/html" in r.headers.get("content-type", ""):
        return None                                  # 404 / access-restricted / not plain text
    return r.text


def probe(ids):
    print(f"{'identifier':30} {'date':5} {'pages':>5}  coverage (first … last head-word)")
    print("-" * 100)
    for idn in ids:
        try:
            m = meta(idn)
        except Exception as e:
            print(f"{idn:30} metadata error: {e}"); continue
        md = m.get("metadata", {})
        date = (md.get("date") or "?")[:4]
        djvu = _file(m, lambda f: f["name"].endswith("_djvu.txt"))
        npages = next((f.get("length") for f in m.get("files", []) if f["name"].endswith("_djvu.xml")), None)
        if not djvu:
            pdf = _file(m, lambda f: "PDF" in f.get("format", ""))
            print(f"{idn:30} {date:5} {'?':>5}  (no OCR text; {'has PDF' if pdf else 'image only'})")
            continue
        url = f"https://archive.org/download/{idn}/{djvu['name']}"
        size = int(djvu.get("size", 0))
        ht, tt = _range(url, 0, HEAD_BYTES), (_range(url, max(0, size - HEAD_BYTES), size) if size else None)
        if ht is None and tt is None:
            print(f"{idn:30} {date:5} {(npages or '?'):>5}  (text not fetchable — access-restricted?)")
            continue
        head = _runheads(ht or "")
        tail = _runheads(tt or "")
        first = " / ".join(dict.fromkeys(head[:3]))
        last = " / ".join(dict.fromkeys(tail[-3:]))
        print(f"{idn:30} {date:5} {(npages or '?'):>5}  {first or '?'}   …   {last or '?'}")
        time.sleep(1)


def download_pdf(idn, out):
    m = meta(idn)
    pdf = _file(m, lambda f: f["name"].lower().endswith(".pdf") and "PDF" in f.get("format", ""))
    if not pdf:
        sys.exit(f"{idn}: no Text-PDF file found")
    url = f"https://archive.org/download/{idn}/{pdf['name']}"
    out = Path(out or f"data/pdf/{idn}.pdf"); out.parent.mkdir(parents=True, exist_ok=True)
    have = out.stat().st_size if out.exists() else 0
    total = int(pdf.get("size", 0))
    if have and have == total:
        print(f"{out} already complete ({have:,} B)"); return
    headers = {**UA, **({"Range": f"bytes={have}-"} if have else {})}
    print(f"downloading {url}\n  -> {out} ({total:,} B){' resuming' if have else ''}")
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out, "ab" if have else "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    print(f"  done: {out.stat().st_size:,} B")


def download_pages(idn, lo, hi, out):
    out = Path(out or f"data/pages/{idn}"); out.mkdir(parents=True, exist_ok=True)
    for leaf in range(lo - 1, hi):                  # IIIF leaf index is 0-based
        dest = out / f"leaf{leaf:04d}.jpg"
        if dest.exists():
            continue
        url = f"https://iiif.archive.org/iiif/{idn}${leaf}/full/full/0/default.jpg"
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                dest.write_bytes(r.content)
            else:
                print(f"  leaf {leaf}: HTTP {r.status_code} (stopping)"); break
        except Exception as e:
            print(f"  leaf {leaf}: {e}"); break
        time.sleep(0.5)
    print(f"saved {len(list(out.glob('*.jpg')))} page images -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("ids", nargs="+")
    p = sub.add_parser("pdf"); p.add_argument("id"); p.add_argument("--out")
    p = sub.add_parser("pages"); p.add_argument("id")
    p.add_argument("--from", dest="lo", type=int, required=True)
    p.add_argument("--to", dest="hi", type=int, required=True); p.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(a.ids)
    elif a.cmd == "pdf":
        download_pdf(a.id, a.out)
    elif a.cmd == "pages":
        download_pages(a.id, a.lo, a.hi, a.out)


if __name__ == "__main__":
    main()
