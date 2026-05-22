#!/usr/bin/env python3
"""Scrutiny / critic step: verify Llama's extractions and flag low-confidence records.

The free pipeline's QA layer (see memory free-extraction-pipeline). For each entry it combines:
  1. DETERMINISTIC rules — cheap, exact checks that don't need an LLM. The first and most
     important: a latitude/longitude is only legitimate if the SOURCE TEXT prints it. Llama
     sometimes supplies a coordinate from world knowledge when the gazetteer gave only the
     other axis (e.g. a bay with "N lat. 55° 27'" but no longitude). Such coords are flagged
     and stripped (or sent for repair) — we never keep an un-sourced coordinate.
  2. gpt-oss-120B CRITIC (added with the full build) — given the source + Llama's record, it
     flags anything that looks off (wrong type, historical→present-day country, missed
     —Also place). Only the flagged minority (~7-11%) goes to Qwen3-thinking for repair.

This module currently implements the deterministic rules; the gpt-oss critic hooks in next.
"""
from __future__ import annotations
import re

# The 1856 gazetteer prints coordinates as e.g. "N lat. 66° 23', E long. 12° 55'".
LAT_RE = re.compile(r"\blat\b", re.I)
LONG_RE = re.compile(r"\blong\b", re.I)


def coord_flags(source_text: str, latitude, longitude):
    """Flag coordinates not actually printed in the source (Llama inferring un-stated axes)."""
    flags = []
    if latitude is not None and not LAT_RE.search(source_text):
        flags.append("latitude_not_in_source")
    if longitude is not None and not LONG_RE.search(source_text):
        flags.append("longitude_not_in_source")
    return flags


def deterministic_flags(source_text: str, place) -> list[str]:
    """All non-LLM checks for one extracted place. Returns flag strings (empty = clean)."""
    return coord_flags(source_text, place.get("latitude"), place.get("longitude"))


if __name__ == "__main__":
    # quick self-test on the sampled Llama outputs vs their source entries
    import json, sqlite3, sys
    from pathlib import Path
    sample = sys.argv[1] if len(sys.argv) > 1 else "/tmp/llama_sample.jsonl"
    con = sqlite3.connect("data/gotw.sqlite")
    flagged = clean = 0
    for line in Path(sample).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        src = con.execute("SELECT text FROM entry WHERE entry_id=?", (rec["entry_id"],)).fetchone()
        if not src:
            continue
        ext = json.loads(rec["response_json"])
        for p in ext.get("places", []):
            fl = coord_flags(src[0], p.get("latitude"), p.get("longitude"))
            if fl:
                flagged += 1
                print(f"  FLAG {p['name']}: {fl}  (lat={p.get('latitude')}, lon={p.get('longitude')})")
                print(f"       src: {src[0][:90]}")
            else:
                clean += 1
    print(f"\n{flagged} coordinate flags, {clean} clean places")
