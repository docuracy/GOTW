#!/usr/bin/env python3
"""A/B harness: compare extraction models on the same entries, by cost + agreement.

Runs a deterministic sample of entries through several (provider, model) configs,
reusing extract.py's response cache (so re-runs are free and quota-safe) and WITHOUT
writing to the `place` table — this is a comparison, not the canonical extraction.

There is no gold standard, so this does NOT claim absolute accuracy. It reports:
  * cost      — tokens (from cached usage) × assumed rates, projected to the corpus
  * coverage  — how completely each model fills disambiguation-critical fields
  * agreement — pairwise vs the first config (place-count, name overlap, AAT/ccode match)
and dumps per-entry disagreements to a markdown report for human review.

Usage (keys from .env; Gemini needs network the agent sandbox may block):
  python3 process/ab_compare.py --entries 25 \
      --configs claude:claude-haiku-4-5,gemini:gemini-2.5-flash-lite,gemini:gemini-2.5-flash
"""
from __future__ import annotations
import argparse, importlib.util, json, sqlite3, sys, time, unicodedata
from pathlib import Path

ex = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("extract", Path(__file__).with_name("extract.py")))
ex.__spec__.loader.exec_module(ex)  # reuse SYSTEM_PROMPT, SCHEMA, providers, cache, Extraction

# Assumed list prices (USD / 1M tokens) — CONFIRM before quoting. Batch halves these.
RATES = {
    "claude-haiku-4-5":      (1.00, 5.00),
    "claude-sonnet-4-6":     (3.00, 15.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash":      (0.30, 2.50),
}
CORPUS_FILES = 1   # all 7 volumes are now parsed into the DB, so no ×7 extrapolation


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return "".join(c for c in s if c.isalnum())


def sample_entries(con, n):
    rows = con.execute("SELECT entry_id, headword_disp, text, tokens FROM entry "
                       "WHERE kind='entry' AND text IS NOT NULL ORDER BY entry_id").fetchall()
    if n >= len(rows):
        return rows
    step = len(rows) / n                       # evenly spaced -> spans alphabet & lengths
    return [rows[int(i * step)] for i in range(n)]


def get_or_generate(con, provider_name, model, utext, entry_id, providers, tries=4):
    """Cache-first; on miss, generate with backoff. Returns (Extraction|None, usage, cached)."""
    key = ex.cache_key(provider_name, model, utext)
    row = con.execute("SELECT response_json, usage_json FROM llm_cache WHERE key=?", (key,)).fetchone()
    if row:
        return ex.Extraction.model_validate_json(row[0]), json.loads(row[1] or "{}"), True
    if provider_name not in providers:
        providers[provider_name] = ex.make_provider(provider_name)
    for i in range(tries):
        try:
            text, usage = providers[provider_name].generate(model, ex.SYSTEM_PROMPT, utext)
            ext = ex.Extraction.model_validate_json(text)        # validate before caching
            ex.cache_put(con, key, provider_name, model, entry_id, text, usage)
            return ext, usage, False
        except Exception as e:
            if i == tries - 1:
                print(f"    ! {provider_name}:{model} entry {entry_id} failed after {tries}: "
                      f"{type(e).__name__} {str(e)[:80]}")
                return None, {}, False
            time.sleep(2 ** i)


def coverage(extractions):
    places = [p for e in extractions for p in e.places]
    n = len(places) or 1
    return {
        "places": len(places),
        "cc%": round(100 * sum(p.country_code is not None for p in places) / n),
        "coords%": round(100 * sum(p.latitude is not None for p in places) / n),
        "aat≠other%": round(100 * sum(p.aat_type_id != "other" for p in places) / n),
        "notes/pl": round(sum(len(p.notes) for p in places) / n, 1),
        "ocr": sum(len(e.ocr_corrections) for e in extractions),
    }


def agreement(ref, other):
    """ref/other: {entry_id: Extraction}. Pairwise agreement metrics."""
    pc_match = jac = aat_ok = aat_tot = cc_ok = cc_tot = 0
    shared = [eid for eid in ref if eid in other]
    for eid in shared:
        re_, oe = ref[eid], other[eid]
        if len(re_.places) == len(oe.places):
            pc_match += 1
        a = {norm(p.name): p for p in re_.places}
        b = {norm(p.name): p for p in oe.places}
        union = set(a) | set(b)
        jac += (len(set(a) & set(b)) / len(union)) if union else 1.0
        for name in set(a) & set(b):
            aat_tot += 1; aat_ok += a[name].aat_type_id == b[name].aat_type_id
            cc_tot += 1;  cc_ok += a[name].country_code == b[name].country_code
    n = len(shared) or 1
    return {"place_count": round(100 * pc_match / n), "name_jaccard": round(jac / n, 2),
            "aat_match%": round(100 * aat_ok / aat_tot) if aat_tot else None,
            "ccode_match%": round(100 * cc_ok / cc_tot) if cc_tot else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--entries", type=int, default=25)
    ap.add_argument("--configs",
                    default="claude:claude-haiku-4-5,gemini:gemini-2.5-flash-lite,gemini:gemini-2.5-flash")
    ap.add_argument("--report", default="docs/model-comparison.md")
    args = ap.parse_args()
    configs = [tuple(c.split(":", 1)) for c in args.configs.split(",")]

    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row
    ex.ensure_cache(con)
    rows = sample_entries(con, args.entries)
    corpus_entries = con.execute("SELECT COUNT(*) FROM entry WHERE kind='entry'").fetchone()[0] * CORPUS_FILES
    print(f"{len(rows)} sampled entries × {len(configs)} configs "
          f"(corpus projection ×{corpus_entries / len(rows):.0f} to {corpus_entries:,} entries)\n")

    providers = {}
    results = {c: {} for c in configs}   # config -> {entry_id: Extraction}
    usage = {c: [0, 0] for c in configs} # config -> [in, out]
    calls = {c: 0 for c in configs}
    for prov, model in configs:
        for r in rows:
            ext, u, cached = get_or_generate(con, prov, model, ex.user_text(r["text"]),
                                             r["entry_id"], providers)
            if ext is None:
                continue                                   # skip persistent failures; resume free later
            results[(prov, model)][r["entry_id"]] = ext
            usage[(prov, model)][0] += u.get("input") or 0
            usage[(prov, model)][1] += u.get("output") or 0
            calls[(prov, model)] += 0 if cached else 1

    import datetime
    ref_c = configs[0]
    R = []                                              # markdown report lines

    def add(s=""):
        R.append(s)

    add(f"# Extraction model comparison — *Gazetteer of the World*")
    add(f"\n_Generated by `process/ab_compare.py` on "
        f"{datetime.date.today().isoformat()} · {len(rows)} entries sampled evenly across "
        f"Volume 5 · projected ×{corpus_entries/len(rows):.0f} to the ≈{corpus_entries:,}-entry corpus._\n")
    add("This is an evidence base for choosing an extraction model for a historical-PDF → linked-data "
        "pipeline. There is **no gold standard**, so this does not claim absolute accuracy — it measures "
        "**cost**, **field coverage** (how completely each model fills the disambiguation-critical "
        "fields), and **inter-model agreement** (a quality proxy). Costs come from real token usage; "
        "re-running is free because every result is cached.\n")

    # ---- cost + coverage ----
    add("## Cost and coverage\n")
    add("| config | API calls | in/out tokens | $/sample | $/corpus | $/corpus (batch) | "
        "country_code | coords | AAT≠other | places | ocr-fixes |")
    add("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    rowsout = []
    for c in configs:
        model = c[1]; ri, ro = RATES.get(model, (0, 0)); ti, to = usage[c]
        done = len(results[c]) or 1
        cost = ti / 1e6 * ri + to / 1e6 * ro
        proj = cost * corpus_entries / done
        cov = coverage(list(results[c].values()))
        add(f"| `{c[0]}:{model}` | {calls[c]} | {ti:,}/{to:,} | ${cost:.4f} | ${proj:,.0f} | "
            f"${proj*0.5:,.0f} | {cov['cc%']}% | {cov['coords%']}% | {cov['aat≠other%']}% | "
            f"{cov['places']} | {cov['ocr']} |")
        rowsout.append((c, proj, cov))
    add("\n> **`$/corpus` counts raw input tokens with no prompt-cache credit, so it is a ceiling.** In "
        "production the cached AAT-shortlist prefix is ~free and the Batch API halves the rest, so real "
        "spend is far lower — but the *cross-model ratio* holds. Rates are assumed list prices.\n")

    # ---- agreement ----
    add(f"## Agreement vs `{ref_c[0]}:{ref_c[1]}` (quality proxy)\n")
    add("| config | place-count match | name Jaccard | AAT match | country_code match |")
    add("|---|--:|--:|--:|--:|")
    for c in configs[1:]:
        a = agreement(results[ref_c], results[c])
        add(f"| `{c[0]}:{c[1]}` | {a['place_count']}% | {a['name_jaccard']} | "
            f"{a['aat_match%']}% | {a['ccode_match%']}% |")
    add("")

    # ---- Appendix concordance costing (if extract_appendix.py has populated the cache) ----
    app = con.execute("SELECT model, usage_json, response_json FROM llm_cache "
                      "WHERE response_json LIKE '%\"rows\":%'").fetchall()
    if app:
        APP_PAGES = 185   # concordance extent (PDF idx ~711–896)
        by = {}
        for model, uj, rj in app:
            u = json.loads(uj or "{}")
            nrows = len(json.loads(rj).get("rows", []))
            d = by.setdefault(model, [0, 0, 0, 0])     # pages, rows, in_tok, out_tok
            d[0] += 1; d[1] += nrows
            d[2] += u.get("input") or 0; d[3] += u.get("output") or 0
        add("## Appendix concordance (Volume VII) — vision-LLM costing\n")
        add("A separate task: the Vol VII Appendix is a bidirectional ancient↔modern toponym index, "
            f"≈{APP_PAGES} dense two-column scanned pages (~115 rows/page → ~21,000 variant pairs). "
            "`process/extract_appendix.py` sends each page image to a vision model for structured rows. "
            "Costs below are measured from the cached sample page(s).\n")
        add("| model | pages sampled | rows | avg in/out tok/page | $/page | $/full Appendix | batch |")
        add("|---|--:|--:|--:|--:|--:|--:|")
        for model, (pg, nrows, ti, to) in by.items():
            ri, ro = RATES.get(model, (0, 0))
            ppc = (ti / pg / 1e6 * ri) + (to / pg / 1e6 * ro)
            add(f"| `{model}` | {pg} | {nrows} | {ti//pg:,}/{to//pg:,} | ${ppc:.4f} | "
                f"${ppc*APP_PAGES:,.2f} | ${ppc*APP_PAGES*0.5:,.2f} |")
        add("\n> Image input is cheap here — the page counts as only a few hundred input tokens; cost is "
            "output-dominated (the transcribed rows). Vision thinking is disabled (pure transcription).\n")

    # ---- per-entry disagreements ----
    add("## Where the models disagree\n")
    add("Entries where configs differ on place count, AAT type, or country code "
        "(the rest agreed). Useful for spotting failure modes.\n")
    shown = 0
    for r in rows:
        eid = r["entry_id"]
        if not all(eid in results[c] for c in configs):
            continue
        cells = {c: results[c][eid] for c in configs}
        flag = len({len(e.places) for e in cells.values()}) > 1
        if not flag:
            bm = {norm(p.name): p for p in cells[ref_c].places}
            for c in configs[1:]:
                om = {norm(p.name): p for p in cells[c].places}
                if any(bm[nm].aat_type_id != om[nm].aat_type_id or
                       bm[nm].country_code != om[nm].country_code for nm in set(bm) & set(om)):
                    flag = True
        if not flag:
            continue
        shown += 1
        if shown > 15:
            continue
        add(f"### {r['headword_disp']} (entry {eid})")
        for c in configs:
            tags = " · ".join(f"{p.name} [{ex.AAT_LABEL.get(p.aat_type_id, p.aat_type_id)}, "
                              f"{p.country_code}]" for p in cells[c].places)
            add(f"- **{c[0]}:{c[1].split('-')[-1]}** ({len(cells[c].places)}): {tags}")
        add("")
    if shown > 15:
        add(f"_…and {shown-15} more disagreeing entries (re-run for the full set)._\n")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(R))

    # concise console echo
    for c, proj, cov in rowsout:
        print(f"  {c[0]+':'+c[1]:30} ${proj:>6,.0f}/corpus  cc {cov['cc%']}%  aat≠other {cov['aat≠other%']}%  "
              f"{cov['places']} places  {cov['ocr']} ocr-fix")
    print(f"\nfull report -> {args.report}")


if __name__ == "__main__":
    main()
