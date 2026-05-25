#!/usr/bin/env python3
"""Cheap full-corpus page triage by the self-hosted VLM — the trustworthy page-type detector.

Every page is sent ONCE, at low resolution with a minimal schema, so the VLM (Qwen2.5-VL) decides what
each page is: prose / plate / blank, and for prose how many tables and illustrations are embedded. This
replaces the brittle geometry table/plate detectors as the *candidate selector*: the expensive high-res
passes (table transcription, plate export, figure cropping) then run only where triage says they should.
Low-res + concurrency makes the whole corpus tractable; results land in `page_triage`, which doubles as
the resume state (a re-run only triages pages not already recorded).

Reuses the served VLM via TABLE_VL_BASE / TABLE_VL_MODEL (see process/extract_tables.py).

    TABLE_VL_BASE=http://127.0.0.1:PORT/v1 python3 process/triage_pages.py \
        --img-dir img/v5 --volume v5 --db data/gotw_seg.sqlite --concurrency 32
"""
from __future__ import annotations
import argparse, base64, io, json, os, re, sqlite3, sys, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
VL_BASE = os.environ.get("TABLE_VL_BASE", "http://localhost:8000/v1")
VL_MODEL = os.environ.get("TABLE_VL_MODEL", "qwen2.5-vl")
REQ_TIMEOUT = 180          # per-request seconds; raise it when driving the server harder (set in main)


class Triage(BaseModel):
    type: Literal["prose", "plate", "blank"]          # running text / full-page illustration / empty
    n_tables: int                                     # embedded statistical tables (prose only, else 0)
    n_images: int                                     # embedded illustrations/figures (prose only, else 0)
    plate_kind: Optional[Literal["map", "city_view", "portrait", "scene", "diagram", "other"]]


SCHEMA = Triage.model_json_schema()
for _n in (SCHEMA, *SCHEMA.get("$defs", {}).values()):
    if _n.get("type") == "object":
        _n["additionalProperties"] = False
PROMPT = ("This is a full page from an 1856 printed gazetteer, shown at low resolution. Classify the page "
          "type: 'prose' (running text in columns, which may contain embedded tables or illustrations), "
          "'plate' (a full-page illustration, map, or portrait), or 'blank' (empty / endpaper / library "
          "stamp only). If prose, set n_tables = number of distinct statistical tables embedded and "
          "n_images = number of embedded illustrations/figures (0 if none). If plate, set plate_kind. "
          "All counts are 0 when not applicable.")


def jpeg_lowres(path: Path, maxpx: int) -> bytes:
    im = Image.open(path).convert("RGB")
    im.thumbnail((maxpx, maxpx))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def triage_one(jpeg: bytes) -> Optional[Triage]:
    """Pure: image bytes -> Triage (no DB; safe to call from worker threads)."""
    body = json.dumps({
        "model": VL_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]}],
        "max_tokens": 96, "temperature": 0,    # fields come first; a whitespace-loop is cut short + salvaged
        "response_format": {"type": "json_schema", "json_schema": {"name": "Triage", "schema": SCHEMA, "strict": True}},
    }).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(f"{VL_BASE.rstrip('/')}/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
            txt = json.load(urllib.request.urlopen(req, timeout=REQ_TIMEOUT))["choices"][0]["message"]["content"]
            try:
                return Triage.model_validate_json(txt)
            except Exception:
                t = _salvage(txt)                   # Qwen often emits the fields then loops on whitespace
                if t is not None:                   # past max_tokens, truncating the JSON — pull fields by regex
                    return t
                if not _SAMPLED:
                    _SAMPLED.append(1); print(f"  [bad-payload] {txt[:120]!r}", file=sys.stderr, flush=True)
                return None                          # deterministic garbage; retrying won't help
        except Exception:
            if attempt == 3:
                return None
            time.sleep(2 ** attempt)


_SAMPLED: list = []


def _salvage(txt):
    """Recover a Triage from malformed/truncated output (the common whitespace-loop case leaves the
    fields intact but the object unterminated). Returns None only if even `type` is missing."""
    ty = re.search(r'"type"\s*:\s*"(prose|plate|blank)"', txt)
    if not ty:
        return None
    nt = re.search(r'"n_tables"\s*:\s*(\d+)', txt)
    ni = re.search(r'"n_images"\s*:\s*(\d+)', txt)
    pk = re.search(r'"plate_kind"\s*:\s*"(map|city_view|portrait|scene|diagram|other)"', txt)
    return Triage(type=ty.group(1), n_tables=int(nt.group(1)) if nt else 0,
                  n_images=int(ni.group(1)) if ni else 0, plate_kind=pk.group(1) if pk else None)


def ensure_schema(con):
    con.execute("CREATE TABLE IF NOT EXISTS page_triage (volume TEXT, idx INTEGER, type TEXT, n_tables INTEGER,"
                " n_images INTEGER, plate_kind TEXT, created_at TEXT, PRIMARY KEY(volume, idx))")
    con.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--db", default="data/gotw_seg.sqlite")
    ap.add_argument("--maxpx", type=int, default=1024, help="longest edge sent to the VLM (low-res = fast)")
    ap.add_argument("--concurrency", type=int, default=32, help="in-flight requests (vLLM batches them)")
    ap.add_argument("--nshards", type=int, default=1, help="split pages across this many parallel jobs")
    ap.add_argument("--shard", type=int, default=0, help="this job's shard (page index %% nshards == shard)")
    ap.add_argument("--timeout", type=int, default=180, help="per-request seconds")
    args = ap.parse_args()
    global REQ_TIMEOUT
    REQ_TIMEOUT = args.timeout

    con = sqlite3.connect(args.db, timeout=60)            # used ONLY in the main thread
    ensure_schema(con)
    files = sorted(p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    done = {r[0] for r in con.execute("SELECT idx FROM page_triage WHERE volume=?", (args.volume,))}
    todo = [(i, f) for i, f in enumerate(files)            # page_triage is the resume state
            if i not in done and i % args.nshards == args.shard]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{args.volume}: {len(files)} pages, {len(todo)} to triage (concurrency {args.concurrency}, {args.maxpx}px)", flush=True)

    def work(item):
        i, f = item
        return i, triage_one(jpeg_lowres(f, args.maxpx))

    kinds, n, fail = Counter(), 0, 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for fut in as_completed([pool.submit(work, it) for it in todo]):
            i, t = fut.result()
            if t is None:
                fail += 1; continue
            con.execute("INSERT OR REPLACE INTO page_triage(volume,idx,type,n_tables,n_images,plate_kind,created_at)"
                        " VALUES(?,?,?,?,?,?,?)", (args.volume, i, t.type, t.n_tables, t.n_images, t.plate_kind, now))
            kinds[t.type] += 1; n += 1
            if n % 100 == 0:
                con.commit(); print(f"  …{n}/{len(todo)} {dict(kinds)}", flush=True)
    con.commit()
    tab = con.execute("SELECT COALESCE(SUM(n_tables),0), COUNT(*) FROM page_triage WHERE volume=? AND n_tables>0",
                      (args.volume,)).fetchone()
    print(f"{args.volume}: triaged {n} pages {dict(kinds)} ({fail} failed); {tab[1]} prose pages report tables "
          f"({tab[0]} total)", flush=True)


if __name__ == "__main__":
    main()
