#!/usr/bin/env python3
"""Export reconciled `place` rows to GeoJSON for the MapLibre demo (docs/places.geojson).

Each located place becomes a Point feature whose properties drive a rich popup:
canonical name, AAT feature type, country, administrative hierarchy, population,
notes, source page, and the WHG match (id + score).

    python3 process/export_geojson.py [--db data/gotw.sqlite] [--out docs/places.geojson]
"""
from __future__ import annotations
import argparse, json, sqlite3
from pathlib import Path

_CONCEPTS = json.loads(Path("data/aat_shortlist.json").read_text())["concepts"]
AAT_LABEL = {c["aat_id"]: c["label"] for c in _CONCEPTS}
AAT_FCLASS = {c["aat_id"]: c["fclass"] for c in _CONCEPTS}


def population(ext):
    out = []
    for p in ext.get("population", []):
        if p.get("count") is not None:
            out.append(f"{p['count']:,}" + (f" ({p['year']})" if p.get("year") else ""))
    return "; ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--out", default="docs/places.geojson")
    args = ap.parse_args()
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT p.name, p.aat_type_id, p.lat, p.lon, p.whg_match_id, p.whg_score, "
        "       p.extraction, e.page_start, e.headword_disp "
        "FROM place p LEFT JOIN entry e ON e.entry_id = p.entry_id "
        "WHERE p.lat IS NOT NULL AND p.lon IS NOT NULL ORDER BY p.name").fetchall()

    feats = []
    for r in rows:
        ext = json.loads(r["extraction"]) if r["extraction"] else {}
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(r["lon"], 5), round(r["lat"], 5)]},
            "properties": {
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
                "page": r["page_start"],
                "headword": r["headword_disp"],
                "whg_id": r["whg_match_id"],
                "whg_score": r["whg_score"],
            },
        })
    fc = {"type": "FeatureCollection",
          "metadata": {"source": "A Gazetteer of the World (c.1856), Vol. 5",
                       "extraction": "Gemini 2.5 Flash", "reconciliation": "WHG Reconciliation API",
                       "count": len(feats)},
          "features": feats}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(fc, ensure_ascii=False, indent=1))
    print(f"wrote {len(feats)} located places -> {args.out}")


if __name__ == "__main__":
    main()
