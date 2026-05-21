#!/usr/bin/env python3
"""LLM extraction stage: gazetteer `entry` rows -> structured `place` rows.

For each headword entry, an LLM extracts one record PER PLACE (multi-place
entries split on "—Also …"), typed with a Getty AAT concept from the validated
shortlist, plus the geographic context needed to disambiguate the toponym
against the WHG Reconciliation API (country, admin hierarchy, nearby places with
bearings/distances, coordinates, population, area).

Design (see process/estimate_cost.py for the costed model):
  * Length-routed models: short/standard entries -> Haiku, long dense -> Sonnet.
  * Prompt caching: the instructions + AAT shortlist + schema form a stable
    cached prefix; only the per-entry text varies after the cache breakpoint.
  * Structured output via output_config json_schema; responses validated with
    Pydantic. Same request shape for sync and Batch API.

Usage:
  python3 process/extract.py --dry-run               # no API key: show prompt+schema+1 request
  python3 process/extract.py --limit 20              # sync-process 20 pending entries
  python3 process/extract.py --batch                 # submit all pending entries to the Batch API
  python3 process/extract.py --collect <batch_id>    # write results of a finished batch

Requires ANTHROPIC_API_KEY for anything but --dry-run.
"""
from __future__ import annotations
import argparse, json, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel

SHORTLIST = Path("data/aat_shortlist.json")

# --- model routing (matches process/estimate_cost.py; override here to use Opus) ---
MODEL_SHORT = "claude-haiku-4-5"     # entries <= LONG_THRESHOLD tokens
MODEL_LONG = "claude-sonnet-4-6"     # long, dense entries
LONG_THRESHOLD = 800
MAX_TOKENS = {"claude-haiku-4-5": 4096, "claude-sonnet-4-6": 8192}

# ---------------------------------------------------------------------------
# AAT shortlist -> the closed set of feature-type ids the model may choose
# ---------------------------------------------------------------------------
_concepts = json.loads(SHORTLIST.read_text())["concepts"]
AAT_LABEL = {c["aat_id"]: c["label"] for c in _concepts}
AAT_VALUES = tuple(AAT_LABEL) + ("other",)   # "other" = feature type outside the shortlist
AATTypeId = Literal[AAT_VALUES]  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# Output schema (one record per place). All fields required; nullable via None.
# ---------------------------------------------------------------------------
class Population(BaseModel):
    year: Optional[int]
    count: Optional[int]


class SpatialRelation(BaseModel):
    reference: str                 # the place referred to, e.g. "Lisbon"
    bearing: Optional[str]         # compass bearing as printed, e.g. "NW"
    distance_value: Optional[float]
    distance_unit: Optional[str]   # e.g. "miles"


class Place(BaseModel):
    name: str                      # canonical toponym, title-cased
    variant_names: List[str]       # alternative/foreign spellings printed in the entry
    feature_term: str              # the gazetteer's own word(s), e.g. "village", "canton, commune, and town"
    aat_type_id: AATTypeId         # Getty AAT concept id from the shortlist (or "other")
    country: Optional[str]         # modern-or-period country/state named in the entry
    admin_hierarchy: List[str]     # containing units, largest -> smallest (province, dep., canton, …)
    spatial_relations: List[SpatialRelation]
    latitude: Optional[float]      # decimal degrees if coordinates are printed (S/W negative)
    longitude: Optional[float]
    population: List[Population]
    area: Optional[str]            # as printed, e.g. "5,400 acres", "33 sq. m."
    notes: List[str]               # salient facts useful for disambiguation/typing


class OCRCorrection(BaseModel):
    original: str                  # garbled form as printed (e.g. "commnne", "villnge")
    corrected: str                 # the intended reading (e.g. "commune", "village")


class Extraction(BaseModel):
    places: List[Place]            # one entry may describe several places ("—Also …")
    ocr_corrections: List[OCRCorrection]  # OCR errors the model fixed while reading (QA trail)


Extraction.model_rebuild()         # resolve forward refs eagerly (import-safe)


def _strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema -> structured-outputs-compatible (additionalProperties:false)."""
    schema = model.model_json_schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


SCHEMA = _strict_schema(Extraction)


# ---------------------------------------------------------------------------
# Cached system prompt: reading rules + AAT shortlist + task framing
# ---------------------------------------------------------------------------
def _shortlist_block() -> str:
    by = {}
    for c in _concepts:
        by.setdefault(c["fclass"], []).append(c)
    names = {"P": "Populated places", "A": "Administrative/political divisions",
             "H": "Water bodies", "T": "Terrestrial landforms",
             "S": "Structures/sites", "X": "Peoples (agents, not places)"}
    out = []
    for f in "PAHTSX":
        out.append(f"  [{f}] {names[f]}")
        for c in by.get(f, []):
            out.append(f"    {c['aat_id']}  {c['label']:30}  ← {', '.join(c['gazetteer_terms'])}")
    return "\n".join(out)


SYSTEM_PROMPT = f"""You extract structured place records from entries in *A Gazetteer of the World* (c. 1856).
Each entry begins with a headword and describes one or more geographic places in dense, abbreviated prose.

Return one record per PLACE. A single entry often describes several places:
- An "—Also …" clause introduces a SEPARATE place that shares the headword — emit a distinct record for each.
- Distinct senses or sub-places (e.g. "(GRANDE and PICCOLO)", "L.-Kuchuk … L.-Buzurg") are separate places.
Do NOT split a single place's sub-sections (Climate, History, Population) into multiple records.

Your priority is the geographic context needed to DISAMBIGUATE the toponym for reconciliation:
country, the administrative hierarchy that contains it, nearby reference places with their bearing and
distance, and coordinates. Capture these faithfully from the text; never invent them.

The text is OCR'd and not fully corrected, so it contains scanning errors — e.g. "commnne"→commune,
"villnge"/"tillage"→village, "fortres"→fortress, "Wnrteniberg"→Württemberg, "in."→"m." (miles),
"bo"→be, "arc"→are. Silently correct obvious OCR garbling when reading values and names, but be
conservative with proper nouns — only fix a toponym when the intended reading is unambiguous. Record
every correction you make in `ocr_corrections` as an {{original, corrected}} pair, for human QA.

Reading the abbreviations:
  m. = miles · prov. = province · dep./dept. = department · cant. = canton · arrond. = arrondissement
  co. = county · p. = parish · dio. = diocese · bail. = bailiwick · gov. = government (province) · div. = division
  reg. = region · comarca/intendency/partido = administrative divisions · Pop. = population
  r. bank / l. bank = right / left bank · cap. = capital
  Coordinates print as e.g. "N lat. 66° 23', E long. 12° 55'". Convert to decimal degrees;
  make South latitudes and West longitudes NEGATIVE. Only set latitude/longitude when coordinates are printed.

Field rules:
- name: the toponym in normalised TITLE CASE (the source prints it UPPERCASE), e.g. "Lus-la-Croix-Haute".
- feature_term: the gazetteer's own descriptor verbatim (e.g. "village", "canton, commune, and town").
- aat_type_id: choose the single best-fitting concept id from the shortlist below for THIS place's primary
  type. If a place plays several administrative roles, pick the most specific that applies. Use "other"
  only when no shortlist concept reasonably fits (e.g. a monastery, battlefield, ruin).
- population: one entry per (year, count) pair printed, e.g. "Pop. in 1831, 6,893; in 1841, 7,002".
- notes: short factual phrases that aid disambiguation or typing (industries, rivers, antiquities). Omit prose.
- Use null / empty lists when a field is absent. Do not guess.

Getty AAT feature-type shortlist (id · label · gazetteer terms it covers):
{_shortlist_block()}
"""


def system_blocks():
    """System prompt as a single cached block (stable across all entries)."""
    return [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]


def model_for(tokens: int) -> str:
    return MODEL_LONG if tokens > LONG_THRESHOLD else MODEL_SHORT


def build_params(entry_text: str, model: str) -> dict:
    return {
        "model": model,
        "max_tokens": MAX_TOKENS[model],
        "system": system_blocks(),
        "messages": [{"role": "user", "content":
                      f"Extract the place record(s) from this entry:\n\n{entry_text}"}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
    }


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------
def pending_entries(con, limit=None):
    sql = ("SELECT e.entry_id, e.headword_disp, e.text, e.tokens FROM entry e "
           "WHERE e.kind='entry' AND e.text IS NOT NULL "
           "AND NOT EXISTS (SELECT 1 FROM place p WHERE p.entry_id=e.entry_id) "
           "ORDER BY e.entry_id")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return con.execute(sql).fetchall()


def write_places(con, entry_id: int, extraction: Extraction):
    con.execute("DELETE FROM place WHERE entry_id=?", (entry_id,))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, pl in enumerate(extraction.places, 1):
        aat = None if pl.aat_type_id == "other" else pl.aat_type_id
        con.execute(
            "INSERT INTO place(entry_id,ordinal,name,extraction,aat_type_id,status,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (entry_id, i, pl.name, pl.model_dump_json(), aat, "extracted", now))
    con.commit()


def parse_message(msg) -> Extraction:
    text = next(b.text for b in msg.content if b.type == "text")
    return Extraction.model_validate_json(text)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_sync(con, rows):
    import anthropic
    client = anthropic.Anthropic()
    n_places = 0
    for r in rows:
        params = build_params(r["text"], model_for(r["tokens"]))
        msg = client.messages.create(**params)
        ext = parse_message(msg)
        write_places(con, r["entry_id"], ext)
        n_places += len(ext.places)
        print(f"  entry {r['entry_id']:>6} {r['headword_disp'][:28]:28} -> {len(ext.places)} place(s)"
              f"  [cache_read={msg.usage.cache_read_input_tokens}]")
    print(f"done: {len(rows)} entries -> {n_places} places")


def run_batch_create(con, rows):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    requests = [Request(custom_id=f"e{r['entry_id']}",
                        params=MessageCreateParamsNonStreaming(
                            **build_params(r["text"], model_for(r["tokens"]))))
                for r in rows]
    batch = client.messages.batches.create(requests=requests)
    print(f"submitted batch {batch.id} with {len(requests)} requests (status: {batch.processing_status})")
    print(f"collect when ended:  python3 process/extract.py --collect {batch.id}")
    return batch.id


def run_batch_collect(con, batch_id):
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"batch {batch_id} status={batch.processing_status} "
              f"(processing={batch.request_counts.processing}); not ready")
        return
    ok = err = n_places = 0
    for result in client.messages.batches.results(batch_id):
        entry_id = int(result.custom_id[1:])
        if result.result.type != "succeeded":
            err += 1
            con.execute("INSERT INTO place(entry_id,ordinal,status,created_at) VALUES(?,?,?,?)",
                        (entry_id, 0, f"failed:{result.result.type}",
                         datetime.now(timezone.utc).isoformat(timespec="seconds")))
            con.commit()
            continue
        ext = parse_message(result.result.message)
        write_places(con, entry_id, ext)
        ok += 1
        n_places += len(ext.places)
    print(f"collected: {ok} ok, {err} failed -> {n_places} places")


def dry_run(con, rows):
    print("=" * 80, "\nSYSTEM PROMPT (cached prefix)\n", "=" * 80, sep="")
    print(SYSTEM_PROMPT)
    print("=" * 80, "\nOUTPUT JSON SCHEMA (structured output)\n", "=" * 80, sep="")
    print(json.dumps(SCHEMA, indent=2)[:1500], "\n  …")
    if rows:
        r = rows[0]
        params = build_params(r["text"], model_for(r["tokens"]))
        print("=" * 80, f"\nEXAMPLE REQUEST  entry {r['entry_id']} '{r['headword_disp']}'"
              f"  ({r['tokens']} tok -> {params['model']})\n", "=" * 80, sep="")
        print("user content:\n", params["messages"][0]["content"][:600])
    print("\n(dry run: no API calls made)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--collect", metavar="BATCH_ID")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    if args.collect:
        run_batch_collect(con, args.collect)
        return
    rows = pending_entries(con, args.limit)
    print(f"{len(rows)} pending entries (kind='entry', not yet extracted)")
    if args.dry_run:
        dry_run(con, rows)
    elif args.batch:
        run_batch_create(con, rows)
    else:
        run_sync(con, rows)


if __name__ == "__main__":
    main()
