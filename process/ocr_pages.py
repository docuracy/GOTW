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
import argparse, hashlib, io, json, math, os, re, statistics, sys
from pathlib import Path
from PIL import Image

_FOUND = _REC = _DET = _LAY = None
TABLE_LABELS = {"Table"}
FIGURE_LABELS = {"Picture", "Figure"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
# NOTE (validated at full triage coverage): the geometry TABLE detector below agrees with the VLM page
# triage on ~91% of table-pages (879 geom vs 853 triage, 804 common) — it is NOT redundant; keep it (it
# also routes cell text out of the prose, which a page-level VLM can't). The PLATE detector, however, is
# superseded by triage (which cleanly separates plates from blanks + gives plate_kind) — drive plate
# export from `page_triage` and, if desired, reprocess per-page text from the cached geometry via
# `--from-geom` (no GPU re-OCR). For best table recall, take the UNION of geometry + triage candidates.
# See memory: table-detection-recall-gap, no-external-llms-self-hosted-vlm.
#
# Surya's LayoutPredictor misses the UNRULED 1856 tables. We find them from the line GEOMETRY alone —
# content-agnostic, so it catches statistical, linguistic concordance, and name-list tables alike
# (unlike the old digit-density rule, which only ever caught numeric tables and false-fired on
# number-dense parish prose). A table is a run of >=TABLE_RUN densely-stacked rows split, WITHIN a
# page column, into >=TABLE_BANDS narrow, populated columns by persistent vertical gutters (whitespace
# running several line-heights — the user's heuristic). Prose is one wide column-block (no internal
# gutter); map labels are sparse and often vertical (rejected by the horizontal-text + density gates).
TABLE_RUN = 4            # min consecutive rows to call it a table (4 catches short tables at 0 false-positives
TABLE_BANDS = 2          # min populated sub-columns within a page column (>=1 internal gutter)
TABLE_BANDS_FULL = 3     # min columns for a FULL-WIDTH table (3 avoids firing on 2-column prose)
TABLE_NARROW = 0.45      # a cell's width must be < this fraction of the column width (cells, not prose)
_NBINS = 60              # x-resolution for the per-column coverage profile


def _cluster_rows(idx_boxes, lh):
    """Group (idx, bbox) tuples into rows by y-center proximity; each row sorted by x. Anchors on the
    row's first box (not the previous one) so a slow baseline drift can't chain a whole table into one row."""
    bs = sorted(idx_boxes, key=lambda t: (t[1][1] + t[1][3]) / 2)
    rows, cur, anchor = [], [], None
    for t in bs:
        yc = (t[1][1] + t[1][3]) / 2
        if anchor is None or yc - anchor <= 0.7 * lh:
            cur.append(t); anchor = anchor if anchor is not None else yc
        else:
            rows.append(sorted(cur, key=lambda t: t[1][0])); cur = [t]; anchor = yc
    if cur:
        rows.append(sorted(cur, key=lambda t: t[1][0]))
    return rows


def _band_count(cov, occ):
    """Number of populated column-bands: contiguous bins whose row-coverage reaches `occ`."""
    n = k = 0
    while k < len(cov):
        if cov[k] >= occ:
            n += 1
            while k < len(cov) and cov[k] >= occ:
                k += 1
        else:
            k += 1
    return n


def _merge_regions(regs):
    """Merge half-page table boxes of one full-width table (those overlapping substantially in y)."""
    out = []
    for r in sorted(regs, key=lambda r: (r[1], r[0])):
        for o in out:
            yov = min(o[3], r[3]) - max(o[1], r[1])
            if yov > 0.5 * min(o[3] - o[1], r[3] - r[1]):
                o[0], o[1] = min(o[0], r[0]), min(o[1], r[1])
                o[2], o[3] = max(o[2], r[2]), max(o[3], r[3])
                break
        else:
            out.append(list(r))
    return [[round(v) for v in r] for r in out]


def _detect_range(rows, hx0, hx1, minbands):
    """Flag (bool per row) where a TABLE_RUN window within [hx0,hx1) shows >=minbands narrow, populated,
    densely-multi-cell columns separated by gutters."""
    hw = hx1 - hx0
    occ = math.ceil(0.6 * TABLE_RUN)
    rmask, rcells = [], []
    for row in rows:
        m = [0] * _NBINS
        cells = [(i, b) for i, b in row if hx0 <= (b[0] + b[2]) / 2 < hx1]
        for _, b in cells:
            for k in range(_NBINS):
                x = hx0 + (k + 0.5) * hw / _NBINS
                if b[0] <= x <= b[2]:
                    m[k] = 1
        rmask.append(m); rcells.append(cells)
    flag = [False] * len(rows)
    for s in range(0, max(0, len(rows) - TABLE_RUN + 1)):
        win = range(s, s + TABLE_RUN)
        cov = [sum(rmask[i][k] for i in win) for k in range(_NBINS)]
        if _band_count(cov, occ) < minbands:                        # >=minbands populated columns
            continue
        if sum(1 for i in win if len(rcells[i]) >= minbands) < occ:  # most rows densely multi-cell
            continue
        ws = [(b[2] - b[0]) / hw for i in win for _, b in rcells[i]]
        if not ws or statistics.median(ws) >= TABLE_NARROW:         # cells narrow vs the range (not prose)
            continue
        for i in win:
            flag[i] = True
    return flag


def table_regions(boxed, page_w):
    """Geometry-only table detection. `boxed`: list of (bbox, text), already figure/layout-table
    excluded. Returns (set of indices into boxed lying in a table, merged [x0,y0,x1,y1] region boxes).
    Scans each page half (>=2 columns → within-column tables) AND the full width (>=3 columns → a
    full-width table straddling the mid-line, without firing on ordinary 2-column prose, which is only
    2 bands across the page)."""
    horiz = [(i, b) for i, (b, _) in enumerate(boxed) if (b[2] - b[0]) >= 0.9 * (b[3] - b[1])]
    if len(horiz) < TABLE_RUN:
        return set(), []
    lh = statistics.median((b[3] - b[1]) for _, b in horiz)
    rows = _cluster_rows(horiz, lh)
    narrow_w = TABLE_NARROW * page_w / 2          # "narrow" is column-scale (half-page), for ALL ranges,
    #                                               so a full-width pass never drops wide prose lines.
    ranges = ((0, page_w / 2, TABLE_BANDS), (page_w / 2, page_w, TABLE_BANDS), (0, page_w, TABLE_BANDS_FULL))
    excluded, raw = set(), []
    for hx0, hx1, minbands in ranges:
        flag = _detect_range(rows, hx0, hx1, minbands)
        i = 0
        while i < len(rows):
            if flag[i]:
                j = i
                while j < len(rows) and flag[j]:
                    j += 1
                # route out only NARROW columnar cells; wide lines (prose, captions, long label+value
                # rows) stay in the entry text, so interleaved essay narrative is never lost
                cells = [(ix, b) for r in range(i, j) for ix, b in rows[r]
                         if hx0 <= (b[0] + b[2]) / 2 < hx1 and (b[2] - b[0]) < narrow_w]
                if cells:
                    excluded.update(ix for ix, _ in cells)
                    xs = [v for _, b in cells for v in (b[0], b[2])]
                    ys = [v for _, b in cells for v in (b[1], b[3])]
                    raw.append([min(xs), min(ys), max(xs), max(ys)])
                i = j
            else:
                i += 1
    return excluded, _merge_regions(raw)


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


def ocr_geometry(img):
    """The one Surya pass (layout + recognition) → serialisable page geometry:
    {w, h, layout:[{label,bbox}], lines:[{bbox,text}]}. This is the expensive, GPU-only step and the
    full record of what Surya saw — caching it (`--save-geom`) lets the cheap layout assembly below be
    re-run on CPU after any detector change (`--from-geom`), so the corpus never needs re-OCR for that."""
    rec, det, lay = _models()
    layout = lay([img])[0]
    result = rec([img], det_predictor=det)[0]
    lines = [{"bbox": [round(v) for v in box_of(tl)], "text": (tl.text or "").strip()}
             for tl in result.text_lines if (tl.text or "").strip()]
    return {"w": img.width, "h": img.height,
            "layout": [{"label": b.label, "bbox": [round(v) for v in box_of(b)]} for b in layout.bboxes],
            "lines": lines}


def assemble_page(geom):
    """Geometry → (reading-order lines, table bboxes, figure bboxes, printed pageno). Pure CPU: layout
    region routing + the geometry-only table detector + two-column reading order. No model needed."""
    W, H = geom["w"], geom["h"]
    tables = [r["bbox"] for r in geom["layout"] if r["label"] in TABLE_LABELS]
    figures = [r["bbox"] for r in geom["layout"] if r["label"] in FIGURE_LABELS]
    regions = tables + figures
    boxed = [(ln["bbox"], ln["text"]) for ln in geom["lines"]
             if not any(in_box(center(ln["bbox"]), r) for r in regions)]   # Surya table/figure -> routed
    pageno = header_pageno(boxed, H)
    mid = W / 2
    excluded, tab_boxes = table_regions(boxed, W)        # unruled tables Surya's layout misses (geometry-only)
    detected = list(tables) + tab_boxes
    left = sorted([i for i in range(len(boxed)) if center(boxed[i][0])[0] < mid], key=lambda i: boxed[i][0][1])
    right = sorted([i for i in range(len(boxed)) if center(boxed[i][0])[0] >= mid], key=lambda i: boxed[i][0][1])
    lines = ([boxed[i][1] for i in left if i not in excluded]
             + [boxed[i][1] for i in right if i not in excluded])
    return lines, detected, figures, pageno, len(excluded)


def ocr_image(img):
    """(reading-order lines, table bboxes, figure bboxes, printed pageno) for a PIL image."""
    return assemble_page(ocr_geometry(img))[:4]


def dump_lines(img, idx):
    """Diagnostic: raw page geometry as JSON (for designing/validating the table detector offline)."""
    g = ocr_geometry(img); g["idx"] = idx
    return g


# Inserted-leaf (map plate / blank / frontispiece) detection — a page is a plate iff it carries NEITHER
# substantial prose NOR a substantial table. This keeps real text pages (prose) and real table pages
# (e.g. the Vol VII language concordance — short cells, ~1 prose line but 400+ table cells) while
# dropping maps/blanks whose label-soup would otherwise pollute the entries. Page-geometry only — no
# page-number sequence (numbering drifts across the appendix), so it can't be fooled by mis-OCR'd numbers.
PLATE_PROSE_LEN = 30        # a kept line of >= this many chars counts as "prose"
PLATE_MAX_PROSE = 40        # real text pages have ~120-160; plates ~0-10
PLATE_MIN_TABLE = 60        # a real table page has hundreds of cells; a map's stray labels far fewer


def block_of(geom, idx):
    """Render the page text block for the merged volume file from a page's geometry. An inserted
    map/blank plate (no prose, no table) is reduced to its marker so it can't pollute the entries."""
    lines, tables, figures, pageno, n_tab = assemble_page(geom)
    head = f"## p. {pageno} (#{idx + 1}) ####" if pageno else f"## p. ? (#{idx + 1}) ####"
    prose = sum(1 for t in lines if len(t) >= PLATE_PROSE_LEN)
    if prose < PLATE_MAX_PROSE and n_tab < PLATE_MIN_TABLE:        # plate/blank: omit body
        return "\n".join([head, "<!-- plate: full-page inserted leaf (map/blank), text omitted -->"]), pageno, 0, 0
    notes = [f"<!-- table bbox={[round(v) for v in b]} -->" for b in tables]
    notes += [f"<!-- figure bbox={[round(v) for v in b]} -->" for b in figures]
    return "\n".join([head, *notes, *lines]), pageno, len(tables), len(figures)


def page_block(img, idx):
    return block_of(ocr_geometry(img), idx)


def write_atomic(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)                           # atomic; a partial file never looks "done"


MERGE_HEADER = "<!-- ocr-merge:"            # provenance manifest line; the parser ignores <!-- … --> lines


def page_files(out_dir: Path):
    return sorted(out_dir.glob("p*.txt"), key=lambda p: int(p.stem[1:]))


def merged_text(out_dir: Path) -> str:
    """The authoritative OCR text for a volume: per-page files concatenated in page order, prefixed by a
    provenance manifest (page count + content sha). The per-page files under --out-dir are the SINGLE
    SOURCE OF TRUTH — parse_ocr.py can build straight from them (`--ocr-dir`) so no stale merged .txt can
    silently feed the pipeline; `--verify` re-derives this and flags a drifted/foreign merged file.
    (Table-routing and inserted-plate omission already happened per-page in block_of.)"""
    files = page_files(out_dir)
    body = "\n\f\n".join(f.read_text(encoding="utf-8") for f in files)
    plates = body.count("<!-- plate:")
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    return (f"{MERGE_HEADER} pages={len(files)} plates={plates} src={out_dir.name} sha256={sha} -->\n"
            + body)


def merge_body(text: str) -> str:
    """Drop the provenance manifest line so two merges compare on content alone."""
    return text.split("\n", 1)[1] if text.startswith(MERGE_HEADER) else text


def merge(out_dir: Path, out: Path):
    text = merged_text(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"merged {len(page_files(out_dir))} pages -> {out}")


def verify_merge(out_dir: Path, out: Path) -> bool:
    """True iff `out` matches a fresh merge of the current per-page files — catches the stale/foreign
    merged file that silently dropped entries (the Notley bug)."""
    if not out.exists():
        print(f"STALE: {out} is missing — run --merge"); return False
    if merge_body(merged_text(out_dir)) == merge_body(out.read_text(encoding="utf-8")):
        print(f"OK: {out} matches {len(page_files(out_dir))} per-page files in {out_dir}/"); return True
    print(f"STALE: {out} does NOT match a fresh merge of {out_dir}/ — re-run --merge before parsing")
    return False


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
    ap.add_argument("--verify", action="store_true",
                    help="check that --out matches a fresh merge of --out-dir (exit 1 if stale)")
    ap.add_argument("--dump-lines", action="store_true",
                    help="diagnostic: emit raw line geometry JSON (one obj per page) instead of OCR text")
    ap.add_argument("--save-geom", action="store_true",
                    help="also cache each page's Surya geometry (p<idx>.geom.json) for GPU-free re-runs")
    ap.add_argument("--from-geom",
                    help="rebuild per-page .txt from cached p*.geom.json in this dir (CPU only, no Surya) — "
                         "re-applies the current table detector corpus-wide without re-OCR")
    args = ap.parse_args()
    if args.merge or args.verify:                   # merge/verify only read --out-dir; no source needed
        if not args.out_dir or not args.out:
            sys.exit(f"--{'verify' if args.verify else 'merge'} needs --out-dir and --out")
        if args.verify:
            sys.exit(0 if verify_merge(Path(args.out_dir), Path(args.out)) else 1)
        merge(Path(args.out_dir), Path(args.out))
        return
    if args.from_geom:                              # re-assemble text from cached geometry, no GPU
        gdir = Path(args.from_geom)
        out_dir = Path(args.out_dir) if args.out_dir else gdir
        out_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        for gf in sorted(gdir.glob("p*.geom.json"), key=lambda p: int(p.stem.split(".")[0][1:])):
            idx = int(gf.stem.split(".")[0][1:])
            block, _, _, _ = block_of(json.loads(gf.read_text(encoding="utf-8")), idx)
            write_atomic(out_dir / f"p{idx:05d}.txt", block); done += 1
        print(f"re-assembled {done} page(s) from {gdir}/ -> {out_dir}/ (no GPU)")
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
        if args.dump_lines:
            print(json.dumps(dump_lines(get_image(idx), idx), ensure_ascii=False))
            continue
        fp = out_dir / f"p{idx:05d}.txt" if out_dir else None
        if fp and fp.exists():                      # resumable shard
            continue
        geom = ocr_geometry(get_image(idx))         # the one (GPU) Surya pass
        if args.save_geom and out_dir:              # cache it so detector changes never need re-OCR
            write_atomic(out_dir / f"p{idx:05d}.geom.json", json.dumps(geom, ensure_ascii=False))
        block, pageno, nt, nf = block_of(geom, idx)
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
