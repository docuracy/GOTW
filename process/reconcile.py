#!/usr/bin/env python3
"""Reconciliation stage: geolocate `place` rows against the WHG Reconciliation API.

For each extracted place we send the disambiguation context (name + WHG fclass +
coordinates if the gazetteer printed any) to https://whgazetteer.org/reconcile,
pick the best candidate by score, then fetch its centroid via data extension and
write the match (id, score, lat, lon) back to `place`.

Learned from the live API (do not change without re-probing):
  * Response: { "<qid>": {"result": [ {id,name,score,match,alt_names,description,type} ], "geojson": …} }
  * Sending `types: ["aat:…"]` returns ZERO hits — WHG is not indexed by AAT ids.
    AAT typing is OUR enrichment; never use it as a reconciliation filter.
  * `fclasses` is too sparse in WHG to rely on; `ccodes` is accepted but does NOT actually
    constrain (returns out-of-country hits). The parameter that genuinely filters by country is
    `countries` (ISO 3166-1 alpha-2) — that is what we send, from the extracted `country_code`.
  * `match` is conservative (a score-100 hit can be match:false), so we threshold on score.
  * Coordinates need a second POST: {"extend": {"ids":[…], "properties":[{"id":"whg:geometry_centroid"}]}}
    -> rows[id]["whg:geometry_centroid"][0]["str"] == "lat, lng".

Usage (needs WHG_API_TOKEN in .env):
  python3 process/reconcile.py --seed-demo 8     # insert curated demo places (no extraction needed)
  python3 process/reconcile.py --limit 50        # reconcile pending places
  python3 process/reconcile.py --dry-run         # show the query that would be sent for each place
"""
from __future__ import annotations
import argparse, json, os, sqlite3, sys, time
from datetime import datetime, timezone

import requests

WHG_URL = "https://whgazetteer.org/reconcile"
USER_AGENT = "GOTW-reconcile/0.1 (https://github.com/docuracy/GOTW; stephen@docuracy.co.uk)"

CHUNK = 25            # queries per reconcile POST
SCORE_THRESHOLD = 50  # accept the top candidate at/above this score
RADIUS_KM = 50        # locator radius when the gazetteer printed coordinates


def token() -> str:
    tok = os.getenv("WHG_API_TOKEN")
    if not tok:
        sys.exit("WHG_API_TOKEN not set — add it to .env and `set -a; . ./.env; set +a`")
    return tok


def ensure_columns(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(place)")}
    if "reconciliation" not in cols:
        con.execute("ALTER TABLE place ADD COLUMN reconciliation TEXT")
    if "whg_score" not in cols:
        con.execute("ALTER TABLE place ADD COLUMN whg_score REAL")
    con.commit()


def build_query(place_row) -> dict:
    """One WHG reconciliation query from a place row's extraction JSON."""
    ext = json.loads(place_row["extraction"]) if place_row["extraction"] else {}
    q = {"query": place_row["name"], "mode": "fuzzy", "limit": 5}
    cc = ext.get("country_code")
    if cc:
        q["countries"] = [cc]                          # the param that actually filters by country
    lat, lng = ext.get("latitude"), ext.get("longitude")
    if lat is not None and lng is not None:
        q.update(lat=lat, lng=lng, radius=RADIUS_KM)   # strong locator signal
    return q


def post(body, tok, tries=3):
    for i in range(tries):
        try:
            r = requests.post(WHG_URL, params={"token": tok}, json=body,
                              headers={"User-Agent": USER_AGENT}, timeout=90)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def fetch_centroids(ids, tok) -> dict:
    """ids -> (lat, lon) via WHG data extension."""
    out = {}
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        d = post({"extend": {"ids": chunk,
                             "properties": [{"id": "whg:geometry_centroid"}]}}, tok)
        for pid, props in (d.get("rows") or {}).items():
            vals = props.get("whg:geometry_centroid") or []
            if vals and vals[0].get("str"):
                lat_s, lng_s = vals[0]["str"].split(",")
                out[pid] = (float(lat_s), float(lng_s))
    return out


def _best_batch(rows, tok, drop_countries=False):
    """Return {place_id: best_candidate>=threshold} for a list of place rows."""
    best = {}
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        queries = {}
        for r in chunk:
            q = build_query(r)
            if drop_countries:
                q.pop("countries", None)
            queries[f"q{r['place_id']}"] = q
        resp = post({"queries": queries}, tok)
        for r in chunk:
            res = (resp.get(f"q{r['place_id']}") or {}).get("result") or []
            if res:
                top = max(res, key=lambda c: c.get("score", 0))
                if top.get("score", 0) >= SCORE_THRESHOLD:
                    best[r["place_id"]] = top
    return best


def reconcile(con, rows, tok):
    best = _best_batch(rows, tok)
    # Recall fallback: places that got no country-filtered match but DO have coordinates
    # are retried coords-only (handles WHG records missing a country code, e.g. small islands).
    retry = [r for r in rows if r["place_id"] not in best
             and (json.loads(r["extraction"]) if r["extraction"] else {}).get("latitude") is not None]
    if retry:
        best.update(_best_batch(retry, tok, drop_countries=True))

    matched = 0
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        centroids = fetch_centroids([best[r["place_id"]]["id"] for r in chunk
                                     if r["place_id"] in best], tok)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r in chunk:
            pid = r["place_id"]
            cand = best.get(pid)
            if cand:
                lat, lon = centroids.get(cand["id"], (None, None))
                con.execute(
                    "UPDATE place SET whg_match_id=?, whg_score=?, lat=?, lon=?, "
                    "reconciliation=?, status='reconciled', created_at=? WHERE place_id=?",
                    (cand["id"], cand.get("score"), lat, lon, json.dumps(cand), now, pid))
                matched += 1
                print(f"  {r['name'][:24]:24} -> {cand['name'][:24]:24} "
                      f"score={cand.get('score')} {cand['id']}"
                      f"{'  @%.3f,%.3f' % (lat, lon) if lat is not None else '  (no centroid)'}")
            else:
                con.execute("UPDATE place SET status='unmatched', created_at=? WHERE place_id=?",
                            (now, pid))
                print(f"  {r['name'][:24]:24} -> (no match)")
        con.commit()
    print(f"reconciled {matched}/{len(rows)} places")


DEMO = [  # curated real Vol-5 places (name, aat_id, ISO ccode, lat, lng) — exercises the stage w/o extraction
    ("Lutterworth", "300008375", "GB", None, None),     # town
    ("Luristan", "300236157", "IR", None, None),        # region/province, "Persia" -> IR
    ("Luton", "300008375", "GB", None, None),           # parish & market-town
    ("Macao", "300008375", "MO", 22.19, 113.50),        # town/settlement, China coast (coords printed)
    ("Luroe", "300008791", "NO", 66.39, 12.92),         # island, Norway
    ("Lutry", "300008375", "CH", None, None),           # town, Switzerland
    ("Lusatia", "300236157", "DE", None, None),         # old division, Germany
    ("Lustleigh", "300000773", "GB", None, None),       # parish, Devon
]


def seed_demo(con, n):
    ensure_columns(con)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute("DELETE FROM place WHERE status='demo'")
    for i, (name, aat, cc, lat, lng) in enumerate(DEMO[:n], 1):
        ext = json.dumps({"name": name, "country_code": cc, "latitude": lat, "longitude": lng})
        con.execute(
            "INSERT INTO place(entry_id,ordinal,name,extraction,aat_type_id,status,created_at)"
            " VALUES(NULL,?,?,?,?,'demo',?)", (i, name, ext, aat, now))
    con.commit()
    print(f"seeded {min(n, len(DEMO))} demo places (status='demo')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed-demo", type=int, metavar="N")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    ensure_columns(con)

    if args.seed_demo:
        seed_demo(con, args.seed_demo)

    sql = ("SELECT place_id, name, extraction, aat_type_id FROM place "
           "WHERE whg_match_id IS NULL AND status IN ('extracted','demo') ORDER BY place_id")
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    rows = con.execute(sql).fetchall()
    print(f"{len(rows)} places to reconcile")

    if args.dry_run:
        for r in rows[:10]:
            print(f"  {r['name'][:24]:24} -> {json.dumps(build_query(r))}")
        print("(dry run: no API calls)")
        return
    if rows:
        reconcile(con, rows, token())


if __name__ == "__main__":
    main()
