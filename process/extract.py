#!/usr/bin/env python3
"""LLM extraction stage: gazetteer `entry` rows -> structured `place` rows.

For each headword entry, an LLM extracts one record PER PLACE (multi-place
entries split on "—Also …"), typed with a Getty AAT concept from the validated
shortlist, plus the geographic context needed to disambiguate the toponym
against the WHG Reconciliation API (country + country_code, admin hierarchy,
nearby places with bearings/distances, coordinates, population, area).

Provider-pluggable for cost/quality A/B:
  --provider claude  -> Haiku (short) / Sonnet (long)        [needs ANTHROPIC_API_KEY]
  --provider gemini  -> Flash-Lite (short) / Flash (long)    [needs GEMINI_API_KEY / GOOGLE_API_KEY]
  --model NAME        -> force one model for every entry (overrides length routing)

EVERY successful LLM result is cached in the `llm_cache` table keyed by
(provider, model, prompt+schema signature, entry text). Re-runs and re-collects
never re-hit the API for an input already seen — no quota waste. Changing the
prompt or schema changes the signature and so invalidates the cache deliberately.

Usage:
  python3 process/extract.py --dry-run --limit 1                 # no key: show prompt+schema+request
  python3 process/extract.py --provider gemini --limit 20        # sync, cached
  python3 process/extract.py --provider claude --batch           # Batch API (Claude only), cached on collect
  python3 process/extract.py --provider claude --collect <id>    # write + cache batch results
"""
from __future__ import annotations
import argparse, hashlib, json, os, socket, sqlite3, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel

SHORTLIST = Path("data/aat_shortlist.json")
LONG_THRESHOLD = 800        # entries longer than this (tokens) route to the stronger model

# provider -> (short_model, long_model) for length routing
PROVIDER_MODELS = {
    "claude": ("claude-haiku-4-5", "claude-sonnet-4-6"),
    "gemini": ("gemini-2.5-flash-lite", "gemini-2.5-flash"),
}
MAX_TOKENS = 8192           # output cap (structured JSON per entry stays well under)

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
    reference: str
    bearing: Optional[str]
    distance_value: Optional[float]
    distance_unit: Optional[str]


class Place(BaseModel):
    name: str
    variant_names: List[str]
    feature_term: str
    aat_type_id: AATTypeId
    country: Optional[str]
    country_code: Optional[str]    # present-day ISO 3166-1 alpha-2 (drives WHG `countries` filter)
    admin_hierarchy: List[str]
    spatial_relations: List[SpatialRelation]
    latitude: Optional[float]
    longitude: Optional[float]
    population: List[Population]
    area: Optional[str]
    notes: List[str]


class OCRCorrection(BaseModel):
    original: str
    corrected: str


class Extraction(BaseModel):
    places: List[Place]
    ocr_corrections: List[OCRCorrection]


Extraction.model_rebuild()


def _strict_schema(model: type[BaseModel]) -> dict:
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
- country_code: the PRESENT-DAY ISO 3166-1 alpha-2 code for where this place now lies, mapping historical
  names to modern states (Persia→IR; Naples/Sardinia→IT; Prussia→DE or PL; Illyria→HR/SI). This drives
  reconciliation, so set it whenever the country is clear; use null only when genuinely uncertain.
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

# Signature so any change to prompt or schema invalidates the cache deliberately.
PROMPT_SIG = hashlib.sha256(
    (SYSTEM_PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:16]


def user_text(entry_text: str) -> str:
    return f"Extract the place record(s) from this entry:\n\n{entry_text}"


def model_for(provider: str, tokens: int, override: Optional[str]) -> str:
    if override:
        return override
    short, long = PROVIDER_MODELS[provider]
    return long if tokens > LONG_THRESHOLD else short


# ---------------------------------------------------------------------------
# Providers: each returns (json_text, usage_dict) for one entry
# ---------------------------------------------------------------------------
class ClaudeProvider:
    name = "claude"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def generate(self, model, system, user):
        msg = self.client.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
        text = next(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        return text, {"input": u.input_tokens, "output": u.output_tokens,
                      "cache_read": getattr(u, "cache_read_input_tokens", 0)}


def _ensure_genai_dns():
    """Some networks fail to resolve only `generativelanguage.googleapis.com` (other
    googleapis hosts resolve fine). When that's the case, resolve it via public DoH
    and pin the IP through a getaddrinfo shim — TLS SNI/cert still use the real
    hostname, which Google's front-end serves. No-op when local DNS already works."""
    host = "generativelanguage.googleapis.com"
    try:
        socket.getaddrinfo(host, 443)
        return
    except OSError:
        pass
    for doh in ("https://1.1.1.1/dns-query", "https://8.8.8.8/resolve"):
        try:
            req = urllib.request.Request(f"{doh}?name={host}&type=A",
                                         headers={"accept": "application/dns-json"})
            ips = [a["data"] for a in json.load(urllib.request.urlopen(req, timeout=8))["Answer"]
                   if a.get("type") == 1]
            if not ips:
                continue
            orig = socket.getaddrinfo
            socket.getaddrinfo = lambda h, *a, **k: orig(ips[0] if h == host else h, *a, **k)
            print(f"[dns] {host} unresolved locally; pinned {ips[0]} via DoH ({doh.split('/')[2]})")
            return
        except Exception:
            continue
    print(f"[dns] warning: could not resolve {host} via DoH; gemini calls may fail")


class GeminiProvider:
    name = "gemini"

    def __init__(self):
        _ensure_genai_dns()
        from google import genai
        self.genai = genai
        self.client = genai.Client()   # GEMINI_API_KEY or GOOGLE_API_KEY

    def generate(self, model, system, user):
        from google.genai import types
        resp = self.client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=Extraction,
                max_output_tokens=MAX_TOKENS))
        m = resp.usage_metadata
        return resp.text, {"input": getattr(m, "prompt_token_count", None),
                           "output": getattr(m, "candidates_token_count", None),
                           "cache_read": getattr(m, "cached_content_token_count", 0) or 0}


def make_provider(name):
    return {"claude": ClaudeProvider, "gemini": GeminiProvider}[name]()


# ---------------------------------------------------------------------------
# Response cache (mandatory) + DB I/O
# ---------------------------------------------------------------------------
CACHE_DDL = """
CREATE TABLE IF NOT EXISTS llm_cache (
  key          TEXT PRIMARY KEY,   -- sha256(provider,model,prompt_sig,user_text)
  provider     TEXT, model TEXT, entry_id INTEGER,
  response_json TEXT,              -- raw model JSON (an Extraction)
  usage_json   TEXT,
  created_at   TEXT
);
"""


def ensure_cache(con):
    con.executescript(CACHE_DDL)
    con.commit()


def cache_key(provider, model, utext):
    return hashlib.sha256(f"{provider}\0{model}\0{PROMPT_SIG}\0{utext}".encode()).hexdigest()


def cache_get(con, key):
    row = con.execute("SELECT response_json FROM llm_cache WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def cache_put(con, key, provider, model, entry_id, text, usage):
    con.execute("INSERT OR REPLACE INTO llm_cache"
                "(key,provider,model,entry_id,response_json,usage_json,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (key, provider, model, entry_id, text, json.dumps(usage),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()


def pending_entries(con, limit=None):
    sql = ("SELECT e.entry_id, e.headword_disp, e.text, e.tokens FROM entry e "
           "WHERE e.kind='entry' AND e.text IS NOT NULL "
           "AND NOT EXISTS (SELECT 1 FROM place p WHERE p.entry_id=e.entry_id) "
           "ORDER BY e.entry_id")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return con.execute(sql).fetchall()


def write_places(con, entry_id, extraction: Extraction):
    con.execute("DELETE FROM place WHERE entry_id=?", (entry_id,))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, pl in enumerate(extraction.places, 1):
        aat = None if pl.aat_type_id == "other" else pl.aat_type_id
        con.execute(
            "INSERT INTO place(entry_id,ordinal,name,extraction,aat_type_id,status,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (entry_id, i, pl.name, pl.model_dump_json(), aat, "extracted", now))
    con.commit()


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_from_cache(con, provider_name, override):
    """Materialize place rows from already-cached results — no API calls. Lets you
    promote an A/B run's cached extractions (e.g. Gemini Flash) into the place table."""
    rows = con.execute("SELECT entry_id, text, tokens FROM entry "
                       "WHERE kind='entry' AND text IS NOT NULL ORDER BY entry_id").fetchall()
    n = p = 0
    for r in rows:
        model = model_for(provider_name, r["tokens"], override)
        cached = cache_get(con, cache_key(provider_name, model, user_text(r["text"])))
        if cached is None:
            continue
        ext = Extraction.model_validate_json(cached)
        write_places(con, r["entry_id"], ext)
        n += 1
        p += len(ext.places)
    print(f"materialized {p} places from {n} cached entries ({provider_name}/{override or 'routed'})")


def run_sync(con, rows, provider_name, override):
    provider = None
    n_places = hits = calls = 0
    for r in rows:
        model = model_for(provider_name, r["tokens"], override)
        utext = user_text(r["text"])
        key = cache_key(provider_name, model, utext)
        cached = cache_get(con, key)
        if cached is None:
            if provider is None:
                provider = make_provider(provider_name)   # lazy: only construct if we'll call
            text, usage = provider.generate(model, SYSTEM_PROMPT, utext)
            Extraction.model_validate_json(text)           # validate before caching
            cache_put(con, key, provider_name, model, r["entry_id"], text, usage)
            calls += 1
            tag = f"[{model.split('-')[-1]}]"
        else:
            text = cached
            hits += 1
            tag = "[cache]"
        ext = Extraction.model_validate_json(text)
        write_places(con, r["entry_id"], ext)
        n_places += len(ext.places)
        print(f"  {tag:9} entry {r['entry_id']:>6} {r['headword_disp'][:26]:26} -> {len(ext.places)} place(s)")
    print(f"done: {len(rows)} entries -> {n_places} places  ({calls} API calls, {hits} cache hits)")


def run_batch_create(con, rows, provider, override):
    (_gemini_batch_create if provider == "gemini" else _claude_batch_create)(con, rows, override)


def run_batch_collect(con, batch_id, provider, override):
    (_gemini_batch_collect if provider == "gemini" else _claude_batch_collect)(con, batch_id, override)


def _claude_batch_create(con, rows, override):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    requests = []
    skipped = 0
    for r in rows:
        model = model_for("claude", r["tokens"], override)
        if cache_get(con, cache_key("claude", model, user_text(r["text"]))) is not None:
            skipped += 1
            continue
        requests.append(Request(custom_id=f"e{r['entry_id']}",
                                params=MessageCreateParamsNonStreaming(
                                    model=model, max_tokens=MAX_TOKENS,
                                    system=[{"type": "text", "text": SYSTEM_PROMPT,
                                             "cache_control": {"type": "ephemeral"}}],
                                    messages=[{"role": "user", "content": user_text(r["text"])}],
                                    output_config={"format": {"type": "json_schema", "schema": SCHEMA}})))
    if skipped:
        print(f"{skipped} entries already cached — not resubmitted")
    if not requests:
        print("nothing to submit")
        return
    batch = client.messages.batches.create(requests=requests)
    print(f"submitted batch {batch.id} with {len(requests)} requests (status: {batch.processing_status})")
    print(f"collect when ended:  python3 process/extract.py --provider claude --collect {batch.id}")


def _gemini_batch_create(con, rows, override):
    """One Gemini batch job per routed model (Batch Mode ≈ 50% off). Entry id rides
    on each request's `metadata` so results can be correlated on collect."""
    _ensure_genai_dns()
    from google import genai
    from google.genai import types
    client = genai.Client()
    groups, skipped = {}, 0
    for r in rows:
        model = model_for("gemini", r["tokens"], override)
        if cache_get(con, cache_key("gemini", model, user_text(r["text"]))) is not None:
            skipped += 1
            continue
        groups.setdefault(model, []).append(r)
    if skipped:
        print(f"{skipped} entries already cached — not resubmitted")
    cfg = lambda: types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, response_mime_type="application/json",
        response_schema=Extraction, max_output_tokens=MAX_TOKENS)
    for model, grp in groups.items():
        src = [types.InlinedRequest(contents=user_text(r["text"]), config=cfg(),
                                    metadata={"entry_id": str(r["entry_id"])}) for r in grp]
        job = client.batches.create(model=model, src=src,
                                    config=types.CreateBatchJobConfig(display_name="gotw-extract"))
        print(f"submitted gemini batch {job.name} ({len(src)} reqs, model {model}, state {job.state})")
        print(f"collect when done:  python3 process/extract.py --provider gemini --collect {job.name}")


def _gemini_batch_collect(con, name, override):
    _ensure_genai_dns()
    from google import genai
    client = genai.Client()
    job = client.batches.get(name=name)
    state = str(job.state)
    if "SUCCEEDED" not in state and "PARTIALLY" not in state:
        print(f"gemini batch {name} state={state}; not ready")
        return
    ok = err = n_places = 0
    for ir in (job.dest.inlined_responses or []):
        meta = ir.metadata or {}
        if ir.error or not getattr(ir, "response", None):
            err += 1
            continue
        try:
            entry_id = int(meta["entry_id"])
            text = ir.response.text
            ext = Extraction.model_validate_json(text)
        except Exception:
            err += 1
            continue
        row = con.execute("SELECT text, tokens FROM entry WHERE entry_id=?", (entry_id,)).fetchone()
        model = model_for("gemini", row[1], override)
        cache_put(con, cache_key("gemini", model, user_text(row[0])), "gemini", model, entry_id, text, {})
        write_places(con, entry_id, ext)
        ok += 1
        n_places += len(ext.places)
    print(f"collected: {ok} ok, {err} failed -> {n_places} places (all cached)")


def _claude_batch_collect(con, batch_id, override):
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"batch {batch_id} status={batch.processing_status} "
              f"(processing={batch.request_counts.processing}); not ready")
        return
    ok = err = n_places = 0
    ent = {}  # entry_id -> (tokens) for cache key
    for result in client.messages.batches.results(batch_id):
        entry_id = int(result.custom_id[1:])
        if result.result.type != "succeeded":
            err += 1
            continue
        msg = result.result.message
        text = next(b.text for b in msg.content if b.type == "text")
        try:
            ext = Extraction.model_validate_json(text)
        except Exception:
            err += 1
            continue
        # cache it (recompute key from the entry text + the model the batch used)
        row = con.execute("SELECT text, tokens FROM entry WHERE entry_id=?", (entry_id,)).fetchone()
        model = model_for("claude", row[1], override)
        u = msg.usage
        cache_put(con, cache_key("claude", model, user_text(row[0])), "claude", model, entry_id, text,
                  {"input": u.input_tokens, "output": u.output_tokens})
        write_places(con, entry_id, ext)
        ok += 1
        n_places += len(ext.places)
    print(f"collected: {ok} ok, {err} failed -> {n_places} places (all cached)")


def dry_run(con, rows, provider_name, override):
    print("=" * 80, "\nSYSTEM PROMPT (cached prefix)\n", "=" * 80, sep="")
    print(SYSTEM_PROMPT[:1800], "\n  …(AAT shortlist continues)…")
    print(f"\nprompt+schema signature: {PROMPT_SIG}")
    print("=" * 80, "\nOUTPUT JSON SCHEMA\n", "=" * 80, sep="")
    print(json.dumps(SCHEMA, indent=2)[:900], "\n  …")
    if rows:
        r = rows[0]
        model = model_for(provider_name, r["tokens"], override)
        print("=" * 80, f"\nEXAMPLE  provider={provider_name} entry {r['entry_id']} "
              f"'{r['headword_disp']}' ({r['tokens']} tok -> {model})\n", "=" * 80, sep="")
        print("cache key:", cache_key(provider_name, model, user_text(r["text"])))
        print("user content:\n", user_text(r["text"])[:500])
    print("\n(dry run: no API calls made)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--provider", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", help="force one model for every entry (overrides length routing)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--collect", metavar="BATCH_ID")
    ap.add_argument("--from-cache", action="store_true",
                    help="write place rows from cached results only (no API calls)")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    ensure_cache(con)

    if args.collect:
        run_batch_collect(con, args.collect, args.provider, args.model)
        return
    if args.from_cache:
        run_from_cache(con, args.provider, args.model)
        return
    rows = pending_entries(con, args.limit)
    print(f"{len(rows)} pending entries (kind='entry', not yet extracted)")
    if args.dry_run:
        dry_run(con, rows, args.provider, args.model)
    elif args.batch:
        run_batch_create(con, rows, args.provider, args.model)
    else:
        run_sync(con, rows, args.provider, args.model)


if __name__ == "__main__":
    main()
