#!/usr/bin/env python3
"""Column-aware OCR of the public-domain volume scans with Surya (GPU).

We OCR the 1856 RGS scans ourselves — fully public-domain (no third-party transcript
licence), and far better than 2015 OCR. Validated on a CRC L40S: clean diacritics
(São-Pedro, Maranhão), intact coordinate glyphs (S lat. 37° 10′), true two-column
reading order, ~4 s/page (0.4 s layout + 3.5 s detect+recognise).

Two page sources:
  • --img-dir DIR  OCR a directory of page images (HathiTrust 600 dpi `00000NNN.jpg`),
    in filename order. This is the corpus path — no PDF build needed; Surya reads the
    native-resolution scans directly.
  • --pdf FILE     OCR a PDF, rendering each page at --dpi (handy for spot checks).

Per page:
  • LayoutPredictor finds non-text regions (Table, Picture/Figure). Surya treats the
    dense two-column body as ONE 'Text' region, so it does NOT split the columns for us —
    we reconstruct two-column reading order from the recognised line boxes (left column
    top-to-bottom, then right). Table/figure regions are excluded from the body text and
    recorded as `<!-- table bbox=… -->` / `<!-- figure bbox=… -->` annotations, so
    process/extract_tables.py (LLM-vision) and process/extract_maps.py can crop and
    digitise them rather than letting their text pollute the prose.
  • RecognitionPredictor (+DetectionPredictor) reads every line.

Output mimics the page-marker layout the rest of the pipeline already understands: one
`## p. N (#idx) ####` marker per page then reading-order text, form-feed (\\f) between
pages — so process/pdf_pages, the parser, etc. consume it unchanged.

Resumable + shardable for SLURM arrays: one file per page under --out-dir (p<idx>.txt,
written atomically); a page is skipped if its file already exists. `--merge` concatenates
the per-page files (in order) into one volume .txt for the parser.

    # one page to stdout (quick check)
    python3 process/ocr_pages.py --pdf data/pdf/gotw-v5.pdf --start 84 --end 84
    # a shard of a volume's images -> per-page files (what each SLURM array task runs)
    python3 process/ocr_pages.py --img-dir /vast/ishi/gotw/img/v5 --out-dir /vast/ishi/gotw/ocr/v5 --start 0 --end 149
    # stitch the shards into one volume file for the parser
    python3 process/ocr_pages.py --img-dir /vast/ishi/gotw/img/v5 --out-dir /vast/ishi/gotw/ocr/v5 --merge --out data/txt/gotw-v5-ocr.txt
"""
from __future__ import annotations
import argparse, io, os, re, sys
from pathlib import Path
from PIL import Image

_FOUND = _REC = _DET = _LAY = None
TABLE_LABELS = {"Table"}
FIGURE_LABELS = {"Picture", "Figure"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _models():
    global _FOUND, _REC, _DET, _LAY
    if _REC is None:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from surya.layout import LayoutPredictor
        _FOUND = FoundationPredictor()              # device defaults to cuda when available
        _REC = RecognitionPredictor(_FOUND)
        _DET = DetectionPredictor()
        _LAY = LayoutPredictor(_FOUND)
    return _REC, _DET, _LAY


def box_of(obj):
    """Axis-aligned [x0,y0,x1,y1] from a Surya .bbox or .polygon (image pixel coords)."""
    b = getattr(obj, "bbox", None)
    if b:
        return [float(v) for v in b]
    poly = getattr(obj, "polygon", None)
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def in_box(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def reading_order(boxed_lines, page_w):
    """Two-column reading order from line boxes: left col (by y) then right col."""
    mid = page_w / 2
    left = sorted([bl for bl in boxed_lines if center(bl[0])[0] < mid], key=lambda x: x[0][1])
    right = sorted([bl for bl in boxed_lines if center(bl[0])[0] >= mid], key=lambda x: x[0][1])
    return [t for _, t in left] + [t for _, t in right]


def header_pageno(boxed_lines, page_h):
    """Printed page number from the running head: a bare 1-3 digit token in the top band."""
    top = [(b, t) for b, t in boxed_lines if center(b)[1] < 0.08 * page_h]
    for b, t in sorted(top, key=lambda x: x[0][1]):
        m = re.fullmatch(r"\d{1,3}", t.strip())
        if m and 1 <= int(m.group()) <= 999:
            return int(m.group())
    return None


def ocr_image(img):
    """Return (reading-order lines, table bboxes, figure bboxes, printed pageno) for a PIL image."""
    rec, det, lay = _models()
    layout = lay([img])[0]
    tables = [box_of(b) for b in layout.bboxes if b.label in TABLE_LABELS]
    figures = [box_of(b) for b in layout.bboxes if b.label in FIGURE_LABELS]
    regions = tables + figures
    result = rec([img], det_predictor=det)[0]
    boxed = []
    for tl in result.text_lines:
        txt = (tl.text or "").strip()
        if not txt:
            continue
        b = box_of(tl)
        if any(in_box(center(b), r) for r in regions):
            continue                                # inside a table/figure -> routed separately
        boxed.append((b, txt))
    pageno = header_pageno(boxed, img.height)
    return reading_order(boxed, img.width), tables, figures, pageno


def page_block(img, idx):
    lines, tables, figures, pageno = ocr_image(img)
    head = f"## p. {pageno} (#{idx + 1}) ####" if pageno else f"## p. ? (#{idx + 1}) ####"
    notes = [f"<!-- table bbox={[round(v) for v in b]} -->" for b in tables]
    notes += [f"<!-- figure bbox={[round(v) for v in b]} -->" for b in figures]
    return "\n".join([head, *notes, *lines]), pageno, len(tables), len(figures)


def write_atomic(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)                           # atomic; a partial file never looks "done"


def merge(out_dir: Path, out: Path):
    files = sorted(out_dir.glob("p*.txt"), key=lambda p: int(p.stem[1:]))
    blocks = [f.read_text(encoding="utf-8") for f in files]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\f\n".join(blocks), encoding="utf-8")
    print(f"merged {len(blocks)} pages -> {out}")


def page_loader(args):
    """Return (n_pages, get_image(idx)->PIL) for the chosen source."""
    if args.img_dir:
        files = sorted(p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
        if not files:
            sys.exit(f"no page images in {args.img_dir}")
        return len(files), lambda i: Image.open(files[i]).convert("RGB")
    import fitz
    doc = fitz.open(args.pdf)
    mat = fitz.Matrix(args.dpi / 72, args.dpi / 72)
    return doc.page_count, lambda i: Image.open(io.BytesIO(doc[i].get_pixmap(matrix=mat).tobytes("png"))).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", dest="img_dir", help="directory of page images (corpus path)")
    ap.add_argument("--pdf", help="PDF source (alternative to --img-dir)")
    ap.add_argument("--out-dir")
    ap.add_argument("--out")
    ap.add_argument("--start", type=int); ap.add_argument("--end", type=int)
    ap.add_argument("--dpi", type=int, default=220, help="render DPI for --pdf")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:                                  # merge only reads --out-dir; no source needed
        if not args.out_dir or not args.out:
            sys.exit("--merge needs --out-dir and --out")
        merge(Path(args.out_dir), Path(args.out))
        return
    if not args.img_dir and not args.pdf:
        ap.error("one of --img-dir or --pdf is required")

    n, get_image = page_loader(args)
    lo = args.start if args.start is not None else 0
    hi = min(args.end if args.end is not None else n - 1, n - 1)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    for idx in range(lo, hi + 1):
        fp = out_dir / f"p{idx:05d}.txt" if out_dir else None
        if fp and fp.exists():                      # resumable shard
            continue
        block, pageno, nt, nf = page_block(get_image(idx), idx)
        if fp:
            write_atomic(fp, block)
            done += 1
            print(f"  idx {idx} (printed {pageno}): tables={nt} figures={nf}", flush=True)
        else:
            print(block)
    if out_dir:
        print(f"wrote {done} new page file(s) to {out_dir} (range {lo}-{hi})")


if __name__ == "__main__":
    main()
