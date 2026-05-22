#!/usr/bin/env python3
"""Reconciliation stage: geolocate `place` rows against the WHG Reconciliation API.

A **3-pass cascade**, each pass run only on the places the previous one left unmatched,
trading precision for recall in a controlled way:

  Pass 1  EXACT   — mode='exact',    countries=[cc]            (strict text + strict country)
  Pass 2  PHONETIC— mode='phonetic', countries=[cc]            (Symphonym KNN + strict country)
  Pass 3  PROX    — mode='phonetic', no countries, bounds=box  (border changes, but a spatial
                    box around the printed coordinates so matches can't come from the other
                    side of the world). Runs only for places that printed coordinates.

All of exact/phonetic/countries/bounds are honoured server-side by the WHG endpoint (it
proxies to the Pitt ES gateway / Symphonym KNN), so this works over the public API from
anywhere with internet — including CRC compute nodes — with no VM access. Matches get the
candidate id, score, the pass that found it, and the WHG centroid (via data extension).

Learned from the live API (do not change without re-probing):
  * `countries` (ISO alpha-2) is the param that genuinely filters by country (not `ccodes`).
  * Never filter by `types`/`fclasses` — WHG is sparsely typed; that tanks recall.
  * `match` is conservative (a score-100 hit can be match:false) — threshold on `score`.
  * Centroids need a 2nd POST: {"extend":{"ids":[…],"properties":[{"id":"whg:geometry_centroid"}]}}.

Usage (needs WHG_API_TOKEN in .env):
  python3 process/reconcile.py --seed-demo 8        # curated demo places (no extraction needed)
  python3 process/reconcile.py --limit 200 --concurrency 6
  python3 process/reconcile.py --dry-run            # show each pass's query, no API calls
"""
from __future__ import annotations
import argparse, json, os, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

WHG_URL = "https://whgazetteer.org/reconcile"
USER_AGENT = "GOTW-reconcile/0.2 (https://github.com/docuracy/GOTW; stephen@docuracy.co.uk)"
CHUNK = 25                # queries per reconcile POST
DEFAULT_THRESHOLD = 80    # accept the top candidate at/above this score
DEFAULT_RADIUS_KM = 150   # half-extent of the Pass-3 proximity box around printed coords

# (label, mode, use_country, use_bounds)
PASSES = [
    ("1-exact",    "exact",    True,  False),
    ("2-phonetic", "phonetic", True,  False),
    ("3-proximity", "phonetic", False, True),
]


def token() -> str:
    tok = os.getenv("WHG_API_TOKEN")
    if not tok:
        sys.exit("WHG_API_TOKEN not set — add it to .env and `set -a; . ./.env; set +a`")
    return tok


def ensure_columns(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(place)")}
    for col, typ in (("reconciliation", "TEXT"), ("whg_score", "REAL"), ("recon_pass", "TEXT")):
        if col not in cols:
            con.execute(f"ALTER TABLE place ADD COLUMN {col} {typ}")
    con.commit()


def build_query(place_row, mode, use_country, use_bounds, radius_km) -> dict | None:
    """One WHG query for a given pass. Returns None if the pass can't apply to this place
    (proximity pass with no printed coordinates → skip, leave unmatched)."""
    ext = json.loads(place_row["extraction"]) if place_row["extraction"] else {}
    q = {"query": place_row["name"], "mode": mode, "limit": 5}
    cc = ext.get("country_code")
    if use_country and cc:
        q["countries"] = [cc]
    if use_bounds:
        lat, lng = ext.get("latitude"), ext.get("longitude")
        if lat is None or lng is None:
            return None
        d = radius_km / 111.0      # ~deg per km; a box ±d around the printed point
        q["bounds"] = {"type": "Polygon", "coordinates": [[
            [lng - d, lat - d], [lng + d, lat - d], [lng + d, lat + d],
            [lng - d, lat + d], [lng - d, lat - d]]]}
    return q


def post(body, tok, tries=4):
    for i in range(tries):
        try:
            r = requests.post(WHG_URL, params={"token": tok}, json=body,
                              headers={"User-Agent": USER_AGENT}, timeout=120)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if i == tries - 1:
                return {"_error": str(e)}
            time.sleep(min(2 ** i, 15))


def run_pass(rows, tok, mode, use_country, use_bounds, radius_km, threshold, concurrency):
    """Return {place_id: best_candidate>=threshold} for one pass, over `rows`, concurrently."""
    chunks = []
    for i in range(0, len(rows), CHUNK):
        batch = rows[i:i + CHUNK]
        queries = {}
        for r in batch:
            q = build_query(r, mode, use_country, use_bounds, radius_km)
            if q is not None:
                queries[f"q{r['place_id']}"] = q
        if queries:
            chunks.append(queries)
    best = {}
    if not chunks:
        return best

    def work(queries):
        return post({"queries": queries}, tok), queries

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in as_completed([pool.submit(work, c) for c in chunks]):
            resp, queries = fut.result()
            if not resp or "_error" in resp:
                continue
            for qid in queries:
                pid = int(qid[1:])
                res = (resp.get(qid) or {}).get("result") or []
                if res:
                    top = max(res, key=lambda c: c.get("score", 0))
                    if top.get("score", 0) >= threshold:
                        best[pid] = top
    return best


def fetch_centroids(ids, tok) -> dict:
    out = {}
    for i in range(0, len(ids), CHUNK):
        d = post({"extend": {"ids": ids[i:i + CHUNK],
                             "properties": [{"id": "whg:geometry_centroid"}]}}, tok)
        for pid, props in ((d or {}).get("rows") or {}).items():
            vals = props.get("whg:geometry_centroid") or []
            if vals and vals[0].get("str"):
                lat_s, lng_s = vals[0]["str"].split(",")
                out[pid] = (float(lat_s), float(lng_s))
    return out


def reconcile(con, rows, tok, threshold, radius_km, concurrency):
    matched = {}      # place_id -> (pass_label, candidate)
    remaining = list(rows)
    for label, mode, use_country, use_bounds in PASSES:
        todo = [r for r in remaining if r["place_id"] not in matched]
        if not todo:
            break
        got = run_pass(todo, tok, mode, use_country, use_bounds, radius_km, threshold, concurrency)
        for pid, cand in got.items():
            matched[pid] = (label, cand)
        print(f"pass {label:12} on {len(todo):>6}  -> matched {len(got):>5}  "
              f"(cumulative {len(matched)}/{len(rows)})", flush=True)

    centroids = fetch_centroids(list({c['id'] for _, c in matched.values()}), tok)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in rows:
        pid = r["place_id"]
        if pid in matched:
            label, cand = matched[pid]
            lat, lon = centroids.get(cand["id"], (None, None))
            con.execute("UPDATE place SET whg_match_id=?, whg_score=?, lat=?, lon=?, recon_pass=?, "
                        "reconciliation=?, status='reconciled', created_at=? WHERE place_id=?",
                        (cand["id"], cand.get("score"), lat, lon, label,
                         json.dumps({"pass": label, "candidate": cand}), now, pid))
        else:
            con.execute("UPDATE place SET recon_pass='unmatched', status='unmatched', created_at=? "
                        "WHERE place_id=?", (now, pid))
    con.commit()
    print(f"reconciled {len(matched)}/{len(rows)} places "
          f"({sum(1 for v in matched.values() if v[0]=='1-exact')} exact, "
          f"{sum(1 for v in matched.values() if v[0]=='2-phonetic')} phonetic, "
          f"{sum(1 for v in matched.values() if v[0]=='3-proximity')} proximity)")


DEMO = [
    ("Lutterworth", "300008375", "GB", None, None),
    ("Luristan", "300236157", "IR", None, None),
    ("Luton", "300008375", "GB", None, None),
    ("Macao", "300008375", "MO", 22.19, 113.50),
    ("Luroe", "300008791", "NO", 66.39, 12.92),
    ("Lutry", "300008375", "CH", None, None),
    ("Lusatia", "300236157", "DE", None, None),
    ("Lustleigh", "300000773", "GB", None, None),
]


def seed_demo(con, n):
    ensure_columns(con)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute("DELETE FROM place WHERE status='demo'")
    for i, (name, aat, cc, lat, lng) in enumerate(DEMO[:n], 1):
        ext = json.dumps({"name": name, "country_code": cc, "latitude": lat, "longitude": lng})
        con.execute("INSERT INTO place(entry_id,ordinal,name,extraction,aat_type_id,status,created_at)"
                    " VALUES(NULL,?,?,?,?,'demo',?)", (i, name, ext, aat, now))
    con.commit()
    print(f"seeded {min(n, len(DEMO))} demo places (status='demo')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed-demo", type=int, metavar="N")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--retry-unmatched", action="store_true", help="also re-attempt previously unmatched")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    ensure_columns(con)
    if args.seed_demo:
        seed_demo(con, args.seed_demo)

    states = "('extracted','demo','unmatched')" if args.retry_unmatched else "('extracted','demo')"
    sql = (f"SELECT place_id, name, extraction, aat_type_id FROM place "
           f"WHERE whg_match_id IS NULL AND status IN {states} ORDER BY place_id")
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    rows = con.execute(sql).fetchall()
    print(f"{len(rows)} places to reconcile (3-pass cascade, threshold {args.threshold})")

    if args.dry_run:
        for r in rows[:8]:
            qs = {lbl: build_query(r, m, uc, ub, args.radius_km) for lbl, m, uc, ub in PASSES}
            print(f"  {r['name'][:22]:22} {json.dumps(qs)}")
        print("(dry run: no API calls)")
        return
    if rows:
        reconcile(con, rows, token(), args.threshold, args.radius_km, args.concurrency)


if __name__ == "__main__":
    main()
