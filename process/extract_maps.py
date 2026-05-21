#!/usr/bin/env python3
"""Find and export the illustration plates (esp. maps) from a Volume scan.

Steel-plate illustrations are unpaginated pages with little/no OCR text (see
process/pdf_pages.plate_pages). Many such pages are actually blanks / library
stamps, so we first filter by ink coverage, then vision-classify the real plates
(map / city view / portrait / scene / diagram) and export the maps as image files
with a manifest. Classifications are cached.

    python3 process/extract_maps.py --pdf data/pdf/gotw-v5.pdf
    python3 process/extract_maps.py --pdf data/pdf/gotw-v5.pdf --dry-run   # ink filter only, no API
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, socket, sqlite3, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel
import fitz

pp = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("pp", Path(__file__).with_name("pdf_pages.py")))
pp.__spec__.loader.exec_module(pp)

INK_MIN = 0.02          # below this, a low-text page is a blank / library stamp
INK_MAX = 0.55          # above this, it's a marbled endpaper (dense uniform pattern), not a plate
MODEL = "gemini-2.5-flash"
OUT_DIR = Path("data/maps")


class Plate(BaseModel):
    kind: Literal["map", "city_view", "portrait", "scene", "diagram", "blank_or_text", "other"]
    title: Optional[str]    # caption/place name if legible


SCHEMA = Plate.model_json_schema()
SCHEMA["additionalProperties"] = False
PROMPT = ("This is a single illustration plate from a 19th-century gazetteer. Classify it: "
          "map (a cartographic map/chart of a place or region), city_view (a scenic view of a "
          "town/landscape), portrait (a person), scene (people/events), diagram, blank_or_text, "
          "or other. Give the engraved title/caption if legible, else null.")
SIG = hashlib.sha256((PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]

_GENAI = None


def _dns():
    host = "generativelanguage.googleapis.com"
    try:
        socket.getaddrinfo(host, 443); return
    except OSError:
        pass
    for doh in ("https://1.1.1.1/dns-query", "https://8.8.8.8/resolve"):
        try:
            r = urllib.request.Request(f"{doh}?name={host}&type=A", headers={"accept": "application/dns-json"})
            ips = [a["data"] for a in json.load(urllib.request.urlopen(r, timeout=8))["Answer"] if a.get("type") == 1]
            orig = socket.getaddrinfo
            socket.getaddrinfo = lambda h, *a, **k: orig(ips[0] if h == host else h, *a, **k); return
        except Exception:
            continue


def ink_fraction(page, scale=0.5):
    """Fraction of dark pixels in a low-res grayscale render (blank≈0, illustration high)."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY)
    s = pix.samples
    dark = sum(1 for b in s if b < 110)
    return dark / max(1, len(s))


def classify(jpeg, con):
    global _GENAI
    key = hashlib.sha256(f"map\0{MODEL}\0{SIG}\0{hashlib.sha256(jpeg).hexdigest()}".encode()).hexdigest()
    row = con.execute("SELECT response_json FROM llm_cache WHERE key=?", (key,)).fetchone()
    if row:
        return Plate.model_validate_json(row[0])
    if _GENAI is None:
        _dns()
        from google import genai
        _GENAI = genai.Client()
    from google.genai import types
    for attempt in range(4):
        try:
            r = _GENAI.models.generate_content(
                model=MODEL, contents=[PROMPT, types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")],
                config=types.GenerateContentConfig(response_mime_type="application/json",
                    response_schema=Plate, thinking_config=types.ThinkingConfig(thinking_budget=0)))
            Plate.model_validate_json(r.text)
            con.execute("INSERT OR REPLACE INTO llm_cache(key,provider,model,response_json,usage_json,created_at)"
                        " VALUES(?,?,?,?,?,?)", (key, "gemini", MODEL, r.text, "{}",
                        datetime.now(timezone.utc).isoformat(timespec="seconds")))
            con.commit()
            return Plate.model_validate_json(r.text)
        except Exception as e:
            if attempt == 3:
                print(f"    classify failed: {type(e).__name__}: {str(e)[:60]}"); return None
            time.sleep(2 ** attempt)


def nearest_printed(idx, page_index):
    befores = [p for p, i in page_index.items() if i < idx]
    return max(befores, key=lambda p: page_index[p]) if befores else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/pdf/gotw-v5.pdf")
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    con.executescript("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, provider TEXT,"
                      " model TEXT, entry_id INTEGER, response_json TEXT, usage_json TEXT, created_at TEXT);")
    d = fitz.open(args.pdf)
    page_index, plates = pp.scan(args.pdf)

    candidates = [(i, ink_fraction(d[i])) for i in plates]
    illus = [(i, f) for i, f in candidates if INK_MIN <= f <= INK_MAX]
    print(f"{len(plates)} low-text pages; {len(illus)} are inked illustrations "
          f"({INK_MIN}–{INK_MAX} dark; endpapers/blanks excluded)")
    if args.dry_run:
        for i, f in illus[:20]:
            print(f"  idx {i}: ink {f:.3f}  (near printed p.{nearest_printed(i, page_index)})")
        print("(dry run: no classification)")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    kinds = Counter()
    manifest = []
    for i, f in illus:
        jpeg = d[i].get_pixmap(matrix=fitz.Matrix(0.9, 0.9)).tobytes("jpeg")  # small for classify
        pl = classify(jpeg, con)
        if pl is None:
            continue
        kinds[pl.kind] += 1
        near = nearest_printed(i, page_index)
        rec = {"pdf_idx": i, "near_printed_page": near, "kind": pl.kind, "title": pl.title}
        if pl.kind == "map":
            fn = OUT_DIR / f"v5_idx{i:04d}_p{near}.png"
            d[i].get_pixmap(matrix=fitz.Matrix(2.4, 2.4)).save(str(fn))   # full-res export
            rec["file"] = str(fn)
            print(f"  MAP  idx {i} (near p.{near}): {pl.title!r} -> {fn}")
        manifest.append(rec)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"\nclassified {sum(kinds.values())} plates: {dict(kinds)}")
    print(f"maps exported to {OUT_DIR}/  ·  manifest -> {OUT_DIR}/manifest.json")


if __name__ == "__main__":
    main()
