#!/usr/bin/env python3
"""Curated Getty AAT shortlist for typing Gazetteer-of-the-World places.

Derived from the feature-type descriptors mined from Volume 5 (each entry opens
"<Name>, a town/parish/river/island... of <place>"). IDs are resolved against
the local AAT dump and validated here, so a wrong/stale id fails loudly.

fclass follows WHG's scheme (placetypes/aat_config.py):
  P populated places · A administrative/political · H water bodies
  T terrestrial landforms · S structures/sites · X non-place agent (people)

    python3 process/build_aat_shortlist.py          # validate + write JSON
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path

INDEX = Path("data/aat_index.pkl")
OUT = Path("data/aat_shortlist.json")

# (aat_id, fclass, [gazetteer descriptors this concept covers])
SHORTLIST = [
    # --- settlements / populated places (P) ---
    ("300008347", "P", ["settlement", "place", "(generic populated place)"]),
    ("300008375", "P", ["town", "market-town", "walled town", "fortified town"]),
    ("300008372", "P", ["village"]),
    ("300008369", "P", ["hamlet"]),
    ("300008389", "P", ["city"]),
    ("300008423", "P", ["market town"]),
    ("300387218", "P", ["capital", "seat of government"]),
    ("300120599", "P", ["port", "seaport", "harbour (as a port town)"]),
    # --- administrative / political divisions (A) ---
    ("300236157", "A", ["(generic admin division)", "circle", "partido",
                         "government (guberniya)", "intendency", "comarca",
                         "bailiwick", "division", "territory"]),
    ("300000773", "A", ["parish", "chapelry"]),
    ("300000792", "A", ["township"]),
    ("300387330", "A", ["commune"]),
    ("300000769", "A", ["canton"]),
    ("300000705", "A", ["district"]),
    ("300000772", "A", ["department"]),
    ("300000771", "A", ["county"]),
    ("300000774", "A", ["province"]),
    ("300000759", "A", ["arrondissement"]),
    ("300000778", "A", ["borough"]),
    ("300000776", "A", ["state (political division)"]),
    ("300232420", "A", ["state", "kingdom", "empire", "principality (as polity)"]),
    ("300387506", "A", ["country"]),
    # --- water bodies (H) ---
    ("300008707", "H", ["river"]),
    ("300008699", "H", ["rivulet", "stream", "brook"]),
    ("300008680", "H", ["lake"]),
    ("300132316", "H", ["bay"]),
    ("300132315", "H", ["gulf"]),
    ("300266559", "H", ["strait"]),
    ("300008706", "H", ["creek"]),
    ("300185707", "H", ["inlet", "indentation"]),
    ("300008713", "H", ["channel"]),
    # --- terrestrial landforms (T) ---
    ("300008791", "T", ["island"]),
    ("300386852", "T", ["islet"]),
    ("300386854", "T", ["group", "cluster", "archipelago"]),
    ("300008850", "T", ["cape", "headland", "point"]),
    ("300008853", "T", ["promontory"]),
    ("300008804", "T", ["peninsula"]),
    ("300008795", "T", ["mountain"]),
    ("300008777", "T", ["hill"]),
    ("300008798", "T", ["summit", "peak"]),
    ("300386831", "T", ["range", "chain", "mountain range"]),
    ("300008761", "T", ["valley"]),
    ("300132325", "T", ["volcano"]),
    ("300008805", "T", ["plain"]),
    # --- structures / sites (S) ---
    ("300006894", "S", ["fortress"]),
    ("300006909", "S", ["fort"]),
    # --- non-place agents (X): peoples, included for completeness ---
    ("300191997", "X", ["tribe", "people", "ethnic group"]),
]

REVIEW = {  # ids whose sense is approximate for this corpus
    "300000773": "AAT only has 'parishes (religious divisions)'; gazetteer parishes "
                 "are often civil/territorial. Acceptable, flag if precision matters.",
    "300120599": "harbour is usually an attribute of a port town, not a standalone place.",
    "300008850": "no AAT 'points (coastal landforms)'; folded into capes/headlands.",
    "300191997": "an Agent, not a geographic place; peoples/tribes may be excluded "
                 "from the gazetteer or modelled separately.",
}


def main():
    if not INDEX.exists():
        sys.exit("run process/aat_resolve.py first to build data/aat_index.pkl")
    _, id2label, _ = pickle.loads(INDEX.read_bytes())
    rows, bad = [], []
    for cid, fclass, terms in SHORTLIST:
        label = id2label.get(cid)
        if not label:
            bad.append(cid)
            continue
        rows.append({"aat_id": cid, "label": label, "fclass": fclass,
                     "gazetteer_terms": terms,
                     "uri": f"http://vocab.getty.edu/aat/{cid}",
                     **({"review": REVIEW[cid]} if cid in REVIEW else {})})
    if bad:
        sys.exit(f"VALIDATION FAILED — ids not in AAT index: {bad}")
    OUT.write_text(json.dumps({"concepts": rows}, indent=2, ensure_ascii=False))
    by = {}
    for r in rows:
        by.setdefault(r["fclass"], []).append(r)
    names = {"P": "Populated places", "A": "Administrative/political",
             "H": "Water bodies", "T": "Terrestrial landforms",
             "S": "Structures/sites", "X": "Agents (people)"}
    print(f"validated {len(rows)} concepts -> {OUT}\n")
    for f in "PAHTSX":
        print(f"{names[f]} ({f}):")
        for r in by.get(f, []):
            print(f"  {r['aat_id']}  {r['label']:30} <- {', '.join(r['gazetteer_terms'])}")
        print()


if __name__ == "__main__":
    main()
