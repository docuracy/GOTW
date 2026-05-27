#!/usr/bin/env python3
"""Cross-check printed coordinates against the place's resolved admin parent and flag mismatches.

A point-in-polygon test: does the PRINTED point (extraction latitude/longitude) fall inside the narrowest
geometry-bearing admin parent recorded in the reconcile `relations`? If not, the coordinates disagree with
the extracted region — likely an OCR/extraction error in one or the other — so we add `containment_fail` to
the place's `reconciliation.flags`. Parent polygons come from the /vast geom-store (same source as the map
geometries). Runs after reconcile, before export. Requires shapely (the `whg` env). No-op off the CRC.

    python3 process/flag_coord_containment.py --db data/gotw_seg.sqlite --store /vast/ishi/geom
"""
from __future__ import annotations
import argparse, json, os, sqlite3
from collections import defaultdict

TOL_DEG = 0.02   # ~2km boundary/precision tolerance (a point just outside a polygon edge isn't flagged)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw_seg.sqlite")
    ap.add_argument("--store", default=os.environ.get("GEOM_STORE", "/vast/ishi/geom"))
    args = ap.parse_args()
    index_path = os.path.join(args.store, "index.json")
    if not os.path.exists(index_path):
        print(f"geom-store index not found ({index_path}) — skipping containment cross-check")
        return 0
    from shapely import wkb as shapely_wkb
    from shapely.geometry import Point
    from shapely.ops import unary_union

    idx = json.load(open(index_path))
    keys_by_pid = defaultdict(list)
    for k in idx:
        keys_by_pid[k.rsplit("_", 1)[0]].append(k)
    handles: dict[str, object] = {}
    geom_cache: dict[str, object] = {}

    def parent_geom(pid):
        if pid in geom_cache:
            return geom_cache[pid]
        gs = []
        for k in keys_by_pid.get(pid, []):
            e = idx[k]
            fh = handles.get(e["file"]) or handles.setdefault(e["file"], open(os.path.join(args.store, e["file"]), "rb"))
            fh.seek(e["offset"])
            try:
                gs.append(shapely_wkb.loads(fh.read(e["length"])))
            except Exception:
                pass
        g = None if not gs else (gs[0] if len(gs) == 1 else unary_union(gs))
        geom_cache[pid] = g
        return g

    con = sqlite3.connect(args.db)
    rows = con.execute("SELECT place_id, extraction, reconciliation FROM place "
                       "WHERE extraction IS NOT NULL AND reconciliation IS NOT NULL").fetchall()
    tested = failed = updates_n = 0
    updates = []
    for pid, ext, rec in rows:
        try:
            e = json.loads(ext); j = json.loads(rec)
        except Exception:
            continue
        lat, lng = e.get("latitude"), e.get("longitude")
        if lat is None or lng is None:
            continue
        parents = [r.get("relationTo") for r in (j.get("relations") or []) if r.get("relationTo")]
        # narrowest first (relations are recorded broadest-first); test the first parent that has geometry
        g = None
        for par in reversed(parents):
            g = parent_geom(par)
            if g is not None:
                break
        if g is None:
            continue                      # no geometry-bearing parent -> can't test
        tested += 1
        pt = Point(lng, lat)
        inside = g.contains(pt) or g.distance(pt) <= TOL_DEG
        flags = set(j.get("flags") or [])
        before = "containment_fail" in flags
        if inside:
            flags.discard("containment_fail")
        else:
            flags.add("containment_fail"); failed += 1
        if ("containment_fail" in flags) != before:
            j["flags"] = sorted(flags)
            updates.append((json.dumps(j), pid)); updates_n += 1
    if updates:
        con.executemany("UPDATE place SET reconciliation=? WHERE place_id=?", updates)
        con.commit()
    print(f"containment cross-check: tested {tested} coord-places against their parent; "
          f"{failed} fell outside (flagged containment_fail); {updates_n} rows updated")
    print("CONTAINMENT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
