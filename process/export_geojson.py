#!/usr/bin/env python3
"""Export reconciled `place` rows for the MapLibre demo's PMTiles build.

Emits LIGHT newline-delimited GeoJSON (id, name, fclass only) for `tippecanoe` to turn into PMTiles
vector tiles, PLUS a sharded detail store holding the full popup record per place, fetched lazily on
click. Keeping the tiles tiny is what lets the map scale to the whole corpus. (Newline-delimited
GeoJSON is a tippecanoe build input — MapLibre cannot render it directly. See process/build_tiles.sh.)

Detail store layout (static, GitHub-Pages-friendly):
  <detail-dir>/manifest.json   {"shards": N, "count": M}   (frontend reads N to locate a place)
  <detail-dir>/<b>.json        {"<id>": {full props}, …}   for b in 0..N-1  (place id % N == b)

    python3 process/export_geojson.py --out /tmp/places.geojsonl   # + docs/detail/ shards
"""
from __future__ import annotations
import argparse, json, math, re, sqlite3
from collections import defaultdict
from pathlib import Path

_CONCEPTS = json.loads(Path("data/aat_shortlist.json").read_text())["concepts"]
AAT_LABEL = {c["aat_id"]: c["label"] for c in _CONCEPTS}
AAT_FCLASS = {c["aat_id"]: c["fclass"] for c in _CONCEPTS}
_VOL = re.compile(r"v(\w+?)(?:-ocr)?\.txt$|-v(\w+?)[-.]", re.I)
LIGHT_KEYS = ("id", "name", "fclass")   # the only attributes baked into the vector tiles
# Per-volume HathiTrust ids (the University of Minnesota scans we OCR'd) -> exact source-page deep links.
HTIDS = {"1": "umn.31951001678068j", "2": "umn.31951001678069h", "3": "umn.31951001678070w",
         "4": "umn.31951001678071u", "5": "umn.31951001678072s", "6": "umn.31951001678073q",
         "7": "umn.31951001678074o"}
_SEQMAP: dict[str, dict[int, int]] = {}   # vol -> {printed page -> HathiTrust image seq}, from OCR markers


def volume_of(filename: str | None) -> str | None:
    """'gotw-v5-ocr.txt' -> '5'."""
    if not filename:
        return None
    m = _VOL.search(filename)
    return (m.group(1) or m.group(2)) if m else None


def population(ext):
    out = []
    for p in ext.get("population", []):
        if p.get("count") is not None:
            out.append(f"{p['count']:,}" + (f" ({p['year']})" if p.get("year") else ""))
    return "; ".join(out)


def latest_pop(ext):
    """Population count for the most recent year (else the last given) — drives marker size; None if absent."""
    rows = [(p.get("year"), p.get("count")) for p in ext.get("population", []) if p.get("count") is not None]
    if not rows:
        return None
    withyear = [(y, c) for y, c in rows if y is not None]
    return max(withyear, key=lambda t: t[0])[1] if withyear else rows[-1][1]


def hathi_url(vol, page):
    """Exact HathiTrust source-page link: seq = the page's image index from the OCR markers."""
    seq = _SEQMAP.get(vol, {}).get(page) if page is not None else None
    return f"https://babel.hathitrust.org/cgi/pt?id={HTIDS[vol]}&seq={seq}" if (vol in HTIDS and seq) else None


def props_of(r):
    """Full popup property dict for a place row (drops empties)."""
    ext = json.loads(r["extraction"]) if r["extraction"] else {}
    vol = volume_of(r["src_file"])
    p = {
        "id": r["pid"],
        "eid": r["eid"],            # entry id -> the reader highlights this exact entry
        "name": r["name"],
        "type": AAT_LABEL.get(r["aat_type_id"], r["aat_type_id"] or "—"),
        "fclass": AAT_FCLASS.get(r["aat_type_id"], "?"),
        "feature_term": ext.get("feature_term"),
        "country": ext.get("country"),
        "country_code": ext.get("country_code"),
        "admin": " › ".join(ext.get("admin_hierarchy", [])),
        "variants": ", ".join(ext.get("variant_names", [])),
        "population": population(ext),
        "notes": ext.get("notes", [])[:4],
        "vol": vol,
        "page": r["page_start"],
        "headword": r["headword_disp"],
        "whg_id": r["whg_match_id"],
        "whg_score": r["whg_score"],
        "src": hathi_url(vol, r["page_start"]),
    }
    return {k: v for k, v in p.items() if v not in (None, "", [])}


def load_seqmap(ocr_dir):
    """vol -> {printed page -> HathiTrust image seq}, parsed from the OCR '## p. N (#seq)' markers."""
    out, d = {}, Path(ocr_dir)
    if not d.is_dir():
        return out
    pat = re.compile(r"## p\. (\d+) \(#(\d+)\)")
    for f in d.glob("gotw-v*-ocr.txt"):
        v, mp = volume_of(f.name), {}
        for m in pat.finditer(f.read_text(encoding="utf-8", errors="ignore")):
            mp.setdefault(int(m.group(1)), int(m.group(2)))    # first occurrence of a printed page wins
        if v:
            out[v] = mp
    return out


def point(lon, lat, props):
    return {"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 5), round(lat, 5)]},
            "properties": props}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw_seg.sqlite")
    ap.add_argument("--out", default="docs/places.geojsonl", help="light NDJSON features for tippecanoe")
    ap.add_argument("--detail-dir", default="docs/detail", help="sharded popup-detail store")
    ap.add_argument("--detail-shards", type=int, default=256, help="number of detail shard files")
    ap.add_argument("--ocr-dir", default="txt", help="merged OCR .txt dir for HathiTrust seq deep-links")
    ap.add_argument("--geocoded-out", default="docs/search/geocoded.json", help="eid -> [lon,lat,placeId] for search/reader routing")
    args = ap.parse_args()
    global _SEQMAP
    _SEQMAP = load_seqmap(args.ocr_dir)
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT p.rowid AS pid, p.entry_id AS eid, p.name, p.aat_type_id, p.lat, p.lon, p.whg_match_id, "
        "       p.whg_score, p.extraction, e.page_start, e.headword_disp, s.filename AS src_file "
        "FROM place p LEFT JOIN entry e ON e.entry_id = p.entry_id "
        "LEFT JOIN source s ON s.source_id = e.source_id "
        "WHERE p.lat IS NOT NULL AND p.lon IS NOT NULL ORDER BY p.name").fetchall()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    # light tile features + sharded detail store
    n = args.detail_shards
    shards: dict[int, dict] = defaultdict(dict)
    geo: dict = {}                       # eid -> [lon,lat,placeId]: a geocoded place opens on the map
    pops = [latest_pop(json.loads(r["extraction"]) if r["extraction"] else {}) for r in rows]
    maxp = max([c for c in pops if c] or [0])
    with out.open("w", encoding="utf-8") as f:
        for r, c in zip(rows, pops):
            p = props_of(r)
            light = {k: p[k] for k in LIGHT_KEYS if k in p}
            if c and maxp:               # population marker scaling: latest-year count -> 1.3×…3× radius (sqrt)
                light["psize"] = round(1.3 + 1.7 * math.sqrt(c / maxp), 2)
            f.write(json.dumps(point(r["lon"], r["lat"], light), ensure_ascii=False) + "\n")
            shards[p["id"] % n][str(p["id"])] = p
            if r["eid"] is not None:
                geo[r["eid"]] = [round(r["lon"], 5), round(r["lat"], 5), p["id"]]
    dd = Path(args.detail_dir); dd.mkdir(parents=True, exist_ok=True)
    for b in range(n):
        (dd / f"{b}.json").write_text(json.dumps(shards.get(b, {}), ensure_ascii=False))
    (dd / "manifest.json").write_text(json.dumps({"shards": n, "count": len(rows)}))
    gp = Path(args.geocoded_out); gp.parent.mkdir(parents=True, exist_ok=True)
    gp.write_text(json.dumps(geo))
    print(f"wrote {len(rows)} light features -> {out}; detail -> {dd}/ ({n} shards); geocoded -> {gp}")


if __name__ == "__main__":
    main()
