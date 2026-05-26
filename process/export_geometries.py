#!/usr/bin/env python3
"""Export the WHG line/polygon geometries for reconciled places as newline-delimited GeoJSON.

A showcase of WHG's geometry holdings: most reconciled matches are points (rendered as circles on the
map), but a few thousand resolve to OSM/OHM relations & ways, Wikidata/TGN areas, etc. that carry real
boundaries (admin areas, water bodies, coasts). The full geometry does NOT live in the ES `_source`
(only `repr_point`/`bounds`/`geometry_index`) — it lives in the consolidated `/vast` geom-store as WKB,
keyed `"{place_id}_{geometry_index}"`. We read it directly (compute nodes can reach `/vast` but not the
gateway's localhost ES), keep only non-point geometries, and tag each with its `whg_id` so the map can
bold-highlight the SELECTED place's geometry.

    python3 process/export_geometries.py --db data/gotw_seg.sqlite --store /vast/ishi/geom \
        --out /tmp/gotw_geometry.geojsonl

Then tippecanoe turns it into docs/geometry.pmtiles (see process/build_tiles.sh). Requires shapely.
Emits NOTHING (and exits 0) if the geom-store is absent — so the pipeline is a no-op off the CRC.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter, defaultdict

NONPOINT = {"Polygon", "MultiPolygon", "LineString", "MultiLineString"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/gotw_seg.sqlite", help="reconciled place DB (reads whg_match_id)")
    ap.add_argument("--store", default=os.environ.get("GEOM_STORE", "/vast/ishi/geom"),
                    help="consolidated geom-store dir (index.json + geom_shard_*.bin)")
    ap.add_argument("--out", default="/tmp/gotw_geometry.geojsonl", help="output newline-delimited GeoJSON")
    args = ap.parse_args()

    index_path = os.path.join(args.store, "index.json")
    if not os.path.exists(index_path):
        # Off-CRC (or store not built): emit an empty file so the tippecanoe step can skip cleanly.
        open(args.out, "w").close()
        print(f"geom-store index not found ({index_path}) — wrote empty {args.out}; skipping geometry export")
        return 0

    try:
        from shapely import wkb as shapely_wkb
    except ImportError:
        print("ERROR: shapely required for geometry export (conda activate the whg env)", file=sys.stderr)
        return 1

    import sqlite3
    idx = json.load(open(index_path))
    print(f"geom-store index entries: {len(idx)}", flush=True)

    # distinct reconciled match ids + a representative GOTW name per match (most common headword/name)
    con = sqlite3.connect(args.db)
    names: dict[str, Counter] = defaultdict(Counter)
    for wid, nm in con.execute(
            "SELECT whg_match_id, name FROM place WHERE whg_match_id IS NOT NULL AND whg_match_id!=''"):
        if nm:
            names[wid][nm] += 1
    want = set(names)
    print(f"distinct reconciled matches: {len(want)}", flush=True)

    # store keys are "{place_id}_{geometry_index}"; place_ids (osm:/wd:/tm:…) never contain '_', so split once
    keys_by_pid: dict[str, list] = defaultdict(list)
    for k in idx:
        pid = k.rsplit("_", 1)[0]
        if pid in want:
            keys_by_pid[pid].append(k)
    print(f"matches present in store: {len(keys_by_pid)}", flush=True)

    handles: dict[str, object] = {}
    def read_wkb(entry):
        fh = handles.get(entry["file"])
        if fh is None:
            fh = handles[entry["file"]] = open(os.path.join(args.store, entry["file"]), "rb")
        fh.seek(entry["offset"])
        return fh.read(entry["length"])

    tc: Counter = Counter()
    nfeat = 0
    with open(args.out, "w") as out:
        for pid, keys in keys_by_pid.items():
            nm = names[pid].most_common(1)[0][0]
            for k in keys:
                try:
                    g = shapely_wkb.loads(read_wkb(idx[k]))
                except Exception:
                    tc["ERR"] += 1
                    continue
                tc[g.geom_type] += 1
                if g.geom_type in NONPOINT and not g.is_empty:
                    out.write(json.dumps({"type": "Feature",
                                          "properties": {"whg_id": pid, "name": nm},
                                          "geometry": g.__geo_interface__}))
                    out.write("\n")
                    nfeat += 1
    print(f"geom type counts: {dict(tc)}", flush=True)
    print(f"non-point features written: {nfeat} -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
