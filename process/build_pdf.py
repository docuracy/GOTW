#!/usr/bin/env python3
"""Bundle a HathiTrust volume's page images (+ OCR .txt) into a searchable PDF.

HathiTrust lets you download a public-domain volume as 600dpi page images and its
OCR plain-text separately. This stitches them back into one PDF — each page is the
image with the page's OCR text embedded as an INVISIBLE layer — so the existing
tools (`pdf_pages`, `pdf_coverage`, `extract_tables/appendix/maps`) work unchanged.

The text layer is anchored top-down (overflow clipped at the foot), which always
preserves the running header + printed page number that `pdf_pages` keys on; for
full-text work, use the .txt directly. The OCR .txt is split into pages on form-feed
(\\f), HathiTrust's page delimiter, and aligned to the images by order.

    python3 process/build_pdf.py --images ~/Downloads/vol6_jpgs --txt ~/Downloads/vol6.txt \
        --out data/pdf/gotw-v6.pdf
"""
from __future__ import annotations
import argparse, re, shutil, sys, tempfile, zipfile
from pathlib import Path
import fitz

IMG_EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def _du(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size if path.exists() else 0


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="page-images .zip OR a folder of page images")
    ap.add_argument("--txt", help="HathiTrust OCR plain-text (form-feed delimited)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=600, help="scan resolution of the images")
    ap.add_argument("--fontsize", type=float, default=6.0, help="invisible text-layer font size")
    ap.add_argument("--keep", action="store_true", help="keep source .zip/.txt (default: delete on success)")
    args = ap.parse_args()

    src = Path(args.images)
    tmp_dir = None                       # our temp extraction (always removed at the end)
    sources = []                         # user source files to delete on success
    if src.suffix.lower() == ".zip":
        tmp_dir = Path(tempfile.mkdtemp(dir=src.parent, prefix=".extract_"))
        print(f"extracting {src.name} …")
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp_dir)
        img_root, sources = tmp_dir, [src]
    else:
        img_root, sources = src, [src]
    if args.txt:
        sources.append(Path(args.txt))

    imgs = sorted((p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXT), key=natural_key)
    if not imgs:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(f"no page images in {args.images}")

    pages_txt = []
    if args.txt:
        raw = Path(args.txt).read_text(encoding="utf-8", errors="replace")
        pages_txt = raw.split("\f")
        if abs(len(pages_txt) - len(imgs)) > 2:
            print(f"⚠ image/text page-count mismatch: {len(imgs)} images vs {len(pages_txt)} text pages "
                  f"— alignment may drift; check the .txt delimiter")

    doc = fitz.open()
    embedded = 0
    for i, img in enumerate(imgs):
        pix = fitz.Pixmap(str(img))
        w_pt, h_pt = pix.width * 72.0 / args.dpi, pix.height * 72.0 / args.dpi
        page = doc.new_page(width=w_pt, height=h_pt)
        page.insert_image(page.rect, filename=str(img))             # full-res image embedded
        if i < len(pages_txt) and pages_txt[i].strip():
            lines = [ln for ln in pages_txt[i].splitlines() if ln.strip()]
            tw = fitz.TextWriter(page.rect)
            lead = max(0.6, (h_pt - 6) / (len(lines) + 1))          # pack all lines into the page
            fs = min(args.fontsize, lead * 0.9)
            y = lead
            for ln in lines:
                tw.append((4, y), ln[:200], fontsize=fs)            # render_mode set on write
                y += lead
            tw.write_text(page, render_mode=3)                      # 3 = invisible OCR layer
            embedded += 1
        pix = None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.out, deflate=True, garbage=3)
    n_pages = len(doc)
    sample = doc[n_pages // 2].get_text().strip().splitlines()[:3] if n_pages else []
    doc.close()

    # verify the PDF before deleting anything: right page count, real size, text retrievable
    out = Path(args.out)
    size = out.stat().st_size
    v = fitz.open(out)
    ok = (v.page_count == len(imgs) and size > 1_000_000
          and (not args.txt or bool(v[v.page_count // 2].get_text().strip())))
    v.close()
    print(f"wrote {len(imgs)} pages ({embedded} with text) -> {out}  ({size/1e6:.0f} MB)")
    if args.txt:
        print(f"  text-layer check (mid page, first lines): {sample}")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)        # our temp extraction, always
    if not ok:
        sys.exit("⚠ PDF failed verification (page count / size / text layer) — sources NOT deleted")
    if args.keep:
        print("  sources kept (--keep)")
    else:
        freed = sum(_du(s) for s in sources)
        for s in sources:
            if s.exists():
                shutil.rmtree(s) if s.is_dir() else s.unlink()
        print(f"  deleted sources, freed ~{freed/1e9:.2f} GB: {', '.join(s.name for s in sources)}")
    print("  → verify coverage: python3 process/pdf_coverage.py " + str(out))


if __name__ == "__main__":
    main()
