#!/usr/bin/env python3
"""Estimate LLM cost for extracting structured places from the whole corpus.

Reads token counts from the SQLite store (one or more parsed HTML files) and
extrapolates to the full 7-file corpus if only a sample is loaded.

    python3 process/estimate_cost.py --db data/gotw.sqlite

ASSUMED list prices (USD per 1M tokens) — CONFIRM current Anthropic pricing
before committing spend. Cost is dominated by *output* tokens.
"""
from __future__ import annotations
import argparse, sqlite3

PRICE = {  # USD / 1M tokens
    "haiku":  dict(inp=1.00, out=5.00,  cache=0.10),   # Haiku 4.5  (assumed)
    "sonnet": dict(inp=3.00, out=15.00, cache=0.30),   # Sonnet 4.6 (assumed)
}
BATCH_DISCOUNT = 0.5     # Batch API
PREFIX_TOKENS  = 3000    # cached shared prompt: instructions + AAT shortlist + schema
BATCH_SIZE     = 20      # short entries packed per request
OUT_PER_PLACE  = 250     # structured-JSON output tokens per extracted place
LONG_THRESHOLD = 800     # entries longer than this -> Sonnet, sent solo
N_FILES        = 7       # files covering the whole work


def bucket_cost(entry_tokens, places, model, batch_size, scale):
    p = PRICE[model]
    calls = max(1, len(entry_tokens) // batch_size) * scale
    entry_in = sum(entry_tokens) * scale
    prefix_in = calls * PREFIX_TOKENS
    out = places * scale * OUT_PER_PLACE
    cost = (entry_in / 1e6 * p["inp"]
            + prefix_in / 1e6 * p["cache"]
            + out / 1e6 * p["out"]) * BATCH_DISCOUNT
    return cost, calls, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    rows = con.execute("SELECT tokens, n_also FROM entry WHERE kind='entry'").fetchall()
    n_src = con.execute("SELECT COUNT(*) FROM source").fetchone()[0]
    scale = max(1, round(N_FILES / max(1, n_src)))

    short  = [t for t, a in rows if t <= LONG_THRESHOLD]
    long_  = [t for t, a in rows if t >  LONG_THRESHOLD]
    pl_short = sum(1 + a for t, a in rows if t <= LONG_THRESHOLD)
    pl_long  = sum(1 + a for t, a in rows if t >  LONG_THRESHOLD)
    places = sum(1 + a for t, a in rows)

    print(f"loaded files: {n_src}  ->  extrapolating x{scale} to {N_FILES}-file corpus")
    print(f"corpus places ~ {places*scale:,}   input text ~ {sum(t for t,_ in rows)*scale:,} tok\n")

    cs, calls_s, out_s = bucket_cost(short, pl_short, "haiku", BATCH_SIZE, scale)
    cl, calls_l, out_l = bucket_cost(long_, pl_long, "sonnet", 1, scale)
    print(f"Haiku  (<= {LONG_THRESHOLD} tok, batch {BATCH_SIZE}): ${cs:6,.0f}   calls~{calls_s:,}  out~{out_s:,} tok")
    print(f"Sonnet (>  {LONG_THRESHOLD} tok, solo)      : ${cl:6,.0f}   calls~{calls_l:,}  out~{out_l:,} tok")
    print(f"TOTAL  (Batch API, length-routed)  : ${cs+cl:6,.0f}")

    allt = [t for t, _ in rows]
    ah, *_ = bucket_cost(allt, places, "haiku", BATCH_SIZE, scale)
    asn, *_ = bucket_cost(allt, places, "sonnet", BATCH_SIZE, scale)
    print(f"\nrange: all-Haiku ${ah:,.0f}  …  all-Sonnet ${asn:,.0f}   (Batch API; ~2x for realtime)")


if __name__ == "__main__":
    main()
