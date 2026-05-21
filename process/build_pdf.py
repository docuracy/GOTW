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
import argparse, re, sys
from pathlib import Path
import fitz

IMG_EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of page images (one per page)")
    ap.add_argument("--txt", help="HathiTrust OCR plain-text (form-feed delimited)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=600, help="scan resolution of the images")
    ap.add_argument("--fontsize", type=float, default=6.0, help="invisible text-layer font size")
    args = ap.parse_args()

    imgs = sorted((p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXT), key=natural_key)
    if not imgs:
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
    size = Path(args.out).stat().st_size

    # quick self-check: does a mid-volume page expose its OCR text (header etc.)?
    sample = doc[len(doc) // 2].get_text().strip().splitlines()[:3] if len(doc) else []
    print(f"wrote {len(imgs)} pages ({embedded} with text) -> {args.out}  ({size/1e6:.0f} MB)")
    if args.txt:
        print(f"  text-layer check (mid page, first lines): {sample}")
        print("  → run: python3 process/pdf_coverage.py " + args.out)
    doc.close()


if __name__ == "__main__":
    main()
