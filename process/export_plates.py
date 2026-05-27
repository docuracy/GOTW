#!/usr/bin/env python3
"""Export illustration plates for the reader — self-hosted (no external LLM).

Full-page inserted leaves (maps, city views, portraits, frontispieces, blanks) are flagged at OCR
time by process/ocr_pages.py with a `<!-- plate -->` marker. Here, per volume, we: read those marked
pages from the merged OCR .txt, drop blanks/endpapers by ink coverage, classify the rest with the
**self-hosted Qwen2.5-VL** already being served for tables (map / city_view / portrait / …), and export
a downscaled JPG + a manifest the reader embeds inline (click → full-screen lightbox). Classifications
are cached in llm_cache, so re-runs are cheap. Reuses TABLE_VL_BASE/TABLE_VL_MODEL (the served VLM).

    TABLE_VL_BASE=http://127.0.0.1:PORT/v1 python3 process/export_plates.py \
        --img-dir img/v5 --ocr txt/gotw-v5-ocr.txt --volume v5 --db data/gotw_seg.sqlite --out docs/plates
"""
from __future__ import annotations
import argparse, base64, hashlib, io, json, os, re, sqlite3, time, urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
HEAD = re.compile(r"## p\. (\S+) \(#(\d+)\)")
# Manual orientation corrections (additional CW degrees per "<vol>/p<idx>.jpg") from explorer lightbox
# reports, applied by process/apply_plate_orientation.py. Read here so a re-run keeps them. {} if absent.
_OVR = Path("data/plate_orientation_overrides.json")
PLATE_ORIENT_OVERRIDES = json.loads(_OVR.read_text()) if _OVR.exists() else {}
VL_BASE = os.environ.get("PLATE_VL_BASE", os.environ.get("TABLE_VL_BASE", "http://localhost:8000/v1"))
VL_MODEL = os.environ.get("PLATE_VL_MODEL", os.environ.get("TABLE_VL_MODEL", "qwen2.5-vl"))
INK_MIN, INK_MAX = 0.02, 0.55          # below = blank/library-stamp; above = marbled endpaper
EXCLUDE = {"blank_or_text"}            # classified-but-not-embedded kinds


class Plate(BaseModel):
    kind: Literal["map", "city_view", "portrait", "scene", "diagram", "blank_or_text", "other"]
    title: Optional[str]               # engraved caption / place name if legible, else null


SCHEMA = Plate.model_json_schema()
SCHEMA["additionalProperties"] = False
PROMPT = ("This is a single full-page illustration plate from an 1856 gazetteer. Classify it: map "
          "(a cartographic map/chart), city_view (scenic view of a town/landscape), portrait (a person), "
          "scene (people/events), diagram, blank_or_text (a blank or text-only leaf), or other. Give the "
          "engraved title/caption if legible, else null.")
SIG = hashlib.sha256((PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]


def ink_fraction(im: Image.Image) -> float:
    """Fraction of dark pixels (fast, via the luminance histogram). blank≈0, illustration high."""
    h = im.convert("L").histogram()
    return sum(h[:110]) / max(1, sum(h))


def classify(jpeg: bytes, con) -> Optional[Plate]:
    """Classify a plate image via the self-hosted Qwen2.5-VL (OpenAI-compatible). Cached by (model, image)."""
    key = hashlib.sha256(f"plate\0{VL_MODEL}\0{SIG}\0{hashlib.sha256(jpeg).hexdigest()}".encode()).hexdigest()
    row = con.execute("SELECT response_json FROM llm_cache WHERE key=?", (key,)).fetchone()
    if row:
        return Plate.model_validate_json(row[0])
    body = json.dumps({
        "model": VL_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]}],
        "max_tokens": 512, "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "Plate", "schema": SCHEMA, "strict": True}},
    }).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(f"{VL_BASE.rstrip('/')}/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
            txt = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
            Plate.model_validate_json(txt)                       # validate before caching
            con.execute("INSERT OR REPLACE INTO llm_cache(key,provider,model,response_json,usage_json,created_at)"
                        " VALUES(?,?,?,?,?,?)", (key, "vllm", VL_MODEL, txt, "{}",
                        datetime.now(timezone.utc).isoformat(timespec="seconds")))
            con.commit()
            return Plate.model_validate_json(txt)
        except Exception as e:
            if attempt == 3:
                print(f"    classify failed: {type(e).__name__}: {str(e)[:60]}"); return None
            time.sleep(2 ** attempt)


def plate_pages(ocr_path: str):
    """Yield (image_index, after_page) per `<!-- plate -->` page in the merged OCR; after_page = the
    nearest preceding numbered printed page (where to slot the plate in the reader's reading order)."""
    last = None
    for chunk in Path(ocr_path).read_text(encoding="utf-8").split("\f"):
        m = HEAD.search(chunk)
        if not m:
            continue
        idx = int(m.group(2)) - 1
        page = int(m.group(1)) if m.group(1).isdigit() else None
        if "<!-- plate:" in chunk:
            yield idx, last
        elif page is not None:
            last = page


def after_page_map(ocr_path: str) -> dict:
    """image_index -> nearest preceding numbered printed page, for EVERY page (used to slot any page,
    e.g. triage-identified plates, into the reader's reading order)."""
    out, last = {}, None
    for chunk in Path(ocr_path).read_text(encoding="utf-8").split("\f"):
        m = HEAD.search(chunk)
        if not m:
            continue
        idx = int(m.group(2)) - 1
        page = int(m.group(1)) if m.group(1).isdigit() else None
        out[idx] = last
        if page is not None:
            last = page
    return out


def best_rotation(im, rec, det, px=1600):
    """Degrees CLOCKWISE to display the plate upright. Only the upright orientation OCRs confidently, so
    score each of the 4 rotations by confidence-weighted recognised-text length and pick the max. (Reliable
    except rare 90/270 near-ties; the VLM can't do this — it reads rotated text fine, so it never flags it.)"""
    s = im.convert("RGB"); s.thumbnail((px, px))
    best_deg, best_sc = 0, -1.0
    for deg in (0, 90, 180, 270):
        r = s.rotate(-deg, expand=True) if deg else s
        res = rec([r], det_predictor=det)[0]
        sc = sum(len((ln.text or "").strip()) * float(ln.confidence or 0.0) for ln in res.text_lines)
        if sc > best_sc:
            best_deg, best_sc = deg, sc
    return best_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True, help="page-image directory for the volume")
    ap.add_argument("--ocr", required=True, help="merged OCR .txt (carries the <!-- plate --> markers)")
    ap.add_argument("--volume", required=True, help="volume tag, e.g. v5")
    ap.add_argument("--db", default="data/gotw_seg.sqlite", help="for the classification cache (llm_cache)")
    ap.add_argument("--out", default="docs/plates", help="output dir (images + manifest.json)")
    ap.add_argument("--max-px", type=int, default=1400, help="longest edge of the exported web image")
    ap.add_argument("--from-triage", action="store_true",
                    help="take plates from VLM page_triage (type='plate', with plate_kind) — no ink filter / re-classify")
    ap.add_argument("--triage-db", help="sqlite holding page_triage (default: --db)")
    ap.add_argument("--orient", action="store_true",
                    help="detect + correct plate orientation with Surya (OCR-confidence over the 4 rotations)")
    ap.add_argument("--dry-run", action="store_true", help="ink filter only, no classification/export")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.executescript("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, provider TEXT, model TEXT,"
                      " entry_id INTEGER, response_json TEXT, usage_json TEXT, created_at TEXT);")
    files = sorted(p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    outdir = Path(args.out) / args.volume
    outdir.mkdir(parents=True, exist_ok=True)
    man_path = Path(args.out) / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}

    # ── triage-driven: triage already classified every page, so just export its plates (no ink/classify) ──
    if args.from_triage:
        tdb = sqlite3.connect(args.triage_db or args.db)
        apm = after_page_map(args.ocr)
        cols = {r[1] for r in tdb.execute("PRAGMA table_info(page_triage)")}
        tcol = "plate_title" if "plate_title" in cols else "NULL"
        plates = [(i, k, t) for i, k, t in tdb.execute(
            f"SELECT idx, plate_kind, {tcol} FROM page_triage WHERE volume=? AND type='plate'", (args.volume,)) if i < len(files)]
        rec = det = None
        if args.orient:                                   # load Surya once for orientation detection
            import sys as _sys; _sys.path.insert(0, "process")
            import ocr_pages as _ocr; rec, det, _ = _ocr._models()
        recs, kinds = [], Counter()
        for idx, kind, title in sorted(plates):
            im = Image.open(files[idx]).convert("RGB")
            # Surya auto-orientation + any manual correction reported via the explorer lightbox and applied by
            # process/apply_plate_orientation.py (recorded as additional CW degrees, so re-runs stay corrected).
            base = best_rotation(im, rec, det) if (args.orient and rec is not None) else 0
            deg = (base + PLATE_ORIENT_OVERRIDES.get(f"{args.volume}/p{idx:05d}.jpg", 0)) % 360
            if deg:
                im = im.rotate(-deg, expand=True)
            web = im.copy(); web.thumbnail((args.max_px, args.max_px))
            web.save(outdir / f"p{idx:05d}.jpg", "JPEG", quality=82)
            recs.append({"idx": idx, "after_page": apm.get(idx), "kind": kind, "title": title,
                         "img": f"plates/{args.volume}/p{idx:05d}.jpg"})
            kinds[kind or "plate"] += 1
        manifest[args.volume] = recs
        man_path.write_text(json.dumps(manifest, ensure_ascii=False))
        print(f"{args.volume}: exported {len(recs)} triage plates {dict(kinds)} -> {outdir}/", flush=True)
        return

    pages = [(i, a) for i, a in plate_pages(args.ocr) if i < len(files)]
    print(f"{args.volume}: {len(pages)} plate-marked pages of {len(files)} images")
    recs, kinds, inked = [], Counter(), 0
    for idx, after in pages:
        im = Image.open(files[idx]).convert("RGB")
        if not (INK_MIN <= ink_fraction(im) <= INK_MAX):        # blank / endpaper -> skip
            continue
        inked += 1
        if args.dry_run:
            print(f"  idx{idx} inked (after p.{after})"); continue
        small = im.copy(); small.thumbnail((1024, 1024))         # small jpeg for classification
        buf = io.BytesIO(); small.save(buf, "JPEG", quality=85)
        pl = classify(buf.getvalue(), con)
        if pl is None or pl.kind in EXCLUDE:
            continue
        kinds[pl.kind] += 1
        web = im.copy(); web.thumbnail((args.max_px, args.max_px))
        web.save(outdir / f"p{idx:05d}.jpg", "JPEG", quality=82)
        recs.append({"idx": idx, "after_page": after, "kind": pl.kind, "title": pl.title,
                     "img": f"plates/{args.volume}/p{idx:05d}.jpg"})
        print(f"  {pl.kind:11} idx{idx} (after p.{after}): {pl.title!r}", flush=True)
    if args.dry_run:
        print(f"{args.volume}: {inked} inked illustration pages (of {len(pages)} marked)"); return
    recs.sort(key=lambda r: r["idx"])
    manifest[args.volume] = recs
    man_path.write_text(json.dumps(manifest, ensure_ascii=False))
    print(f"{args.volume}: exported {len(recs)} plates {dict(kinds)} -> {outdir}/ ; manifest -> {man_path}")


if __name__ == "__main__":
    main()
