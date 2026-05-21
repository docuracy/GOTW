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
CORPUS_FILES = 7   # this volume is 1 of 7


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
    ap.add_argument("--report", default="data/ab_report.md")
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

    # ---- cost + coverage table ----
    print(f"{'config':30} {'calls':>5} {'in/out tok':>14} {'$/sample':>9} {'$/corpus':>9} "
          f"{'batch':>8}  coverage")
    print("-" * 118)
    for c in configs:
        model = c[1]; ri, ro = RATES.get(model, (0, 0)); ti, to = usage[c]
        done = len(results[c]) or 1
        cost = ti / 1e6 * ri + to / 1e6 * ro
        proj = cost * corpus_entries / done
        cov = coverage(list(results[c].values()))
        print(f"{c[0]+':'+model:30} {calls[c]:>5} {f'{ti:,}/{to:,}':>14} "
              f"{cost:>9.4f} {proj:>9.0f} {proj*0.5:>8.0f}  "
              f"cc {cov['cc%']}% · coords {cov['coords%']}% · aat≠other {cov['aat≠other%']}% · "
              f"{cov['places']} places · {cov['ocr']} ocr-fix")

    # ---- agreement vs first config ----
    ref_c = configs[0]
    print(f"\nagreement vs {ref_c[0]}:{ref_c[1]} (proxy for quality; no gold standard):")
    for c in configs[1:]:
        a = agreement(results[ref_c], results[c])
        print(f"  {c[0]+':'+c[1]:30} place-count {a['place_count']}% · name-Jaccard {a['name_jaccard']} "
              f"· AAT {a['aat_match%']}% · ccode {a['ccode_match%']}%")

    # ---- disagreement dump ----
    lines = [f"# A/B extraction comparison\n",
             f"{len(rows)} entries · configs: {', '.join(p+':'+m for p,m in configs)}\n",
             "Per-entry places where configs disagree on count, AAT type, or country code.\n"]
    for r in rows:
        eid = r["entry_id"]
        if not all(eid in results[c] for c in configs):
            continue                                   # only dump entries every config produced
        cells = {c: results[c][eid] for c in configs}
        counts = {len(e.places) for e in cells.values()}
        # flag entries where any pair disagrees on count or a shared place's aat/ccode
        flag = len(counts) > 1
        if not flag:
            base = cells[ref_c]
            for c in configs[1:]:
                bm = {norm(p.name): p for p in base.places}
                om = {norm(p.name): p for p in cells[c].places}
                for nm in set(bm) & set(om):
                    if bm[nm].aat_type_id != om[nm].aat_type_id or bm[nm].country_code != om[nm].country_code:
                        flag = True
        if not flag:
            continue
        lines.append(f"\n## {r['headword_disp']} (entry {eid})")
        for c in configs:
            for p in cells[c].places:
                lines.append(f"- `{c[0]}:{c[1].split('-')[-1]}` **{p.name}** · "
                             f"{ex.AAT_LABEL.get(p.aat_type_id, p.aat_type_id)} · {p.country_code}")
    Path(args.report).write_text("\n".join(lines))
    print(f"\ndisagreement dump -> {args.report}")
    print("NOTE: $ counts RAW input tokens with no prompt-cache credit, so $/corpus is a ceiling — in "
          "production the cached AAT-shortlist prefix is ~free and batch halves the rest. The cross-model "
          "RATIO is the signal. Rates are ASSUMED list prices (see RATES). Re-runs are free (cached).")


if __name__ == "__main__":
    main()
