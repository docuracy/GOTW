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
import argparse, json, re, sqlite3
from collections import defaultdict
from pathlib import Path

_CONCEPTS = json.loads(Path("data/aat_shortlist.json").read_text())["concepts"]
AAT_LABEL = {c["aat_id"]: c["label"] for c in _CONCEPTS}
AAT_FCLASS = {c["aat_id"]: c["fclass"] for c in _CONCEPTS}
_VOL = re.compile(r"v(\w+?)(?:-ocr)?\.txt$|-v(\w+?)[-.]", re.I)
LIGHT_KEYS = ("id", "name", "fclass")   # the only attributes baked into the vector tiles


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


def props_of(r):
    """Full popup property dict for a place row (drops empties)."""
    ext = json.loads(r["extraction"]) if r["extraction"] else {}
    p = {
        "id": r["pid"],
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
        "vol": volume_of(r["src_file"]),
        "page": r["page_start"],
        "headword": r["headword_disp"],
        "whg_id": r["whg_match_id"],
        "whg_score": r["whg_score"],
    }
    return {k: v for k, v in p.items() if v not in (None, "", [])}


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
    args = ap.parse_args()
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT p.rowid AS pid, p.name, p.aat_type_id, p.lat, p.lon, p.whg_match_id, p.whg_score, "
        "       p.extraction, e.page_start, e.headword_disp, s.filename AS src_file "
        "FROM place p LEFT JOIN entry e ON e.entry_id = p.entry_id "
        "LEFT JOIN source s ON s.source_id = e.source_id "
        "WHERE p.lat IS NOT NULL AND p.lon IS NOT NULL ORDER BY p.name").fetchall()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    # light tile features + sharded detail store
    n = args.detail_shards
    shards: dict[int, dict] = defaultdict(dict)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            p = props_of(r)
            f.write(json.dumps(point(r["lon"], r["lat"], {k: p[k] for k in LIGHT_KEYS if k in p}),
                               ensure_ascii=False) + "\n")
            shards[p["id"] % n][str(p["id"])] = p
    dd = Path(args.detail_dir); dd.mkdir(parents=True, exist_ok=True)
    for b in range(n):
        (dd / f"{b}.json").write_text(json.dumps(shards.get(b, {}), ensure_ascii=False))
    (dd / "manifest.json").write_text(json.dumps({"shards": n, "count": len(rows)}))
    print(f"wrote {len(rows)} light features -> {out}; detail -> {dd}/ ({n} shards)")


if __name__ == "__main__":
    main()
