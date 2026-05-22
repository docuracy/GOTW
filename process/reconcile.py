#!/usr/bin/env python3
"""Reconciliation stage: geolocate `place` rows against WHG, as a 3-pass cascade.

Each pass runs only on the previous one's misses (precision first, then recall), thresholding
on `score`:

  Pass 1  EXACT    — mode='exact',    ccodes=[cc]                (strict text + strict country)
  Pass 2  PHONETIC — mode='phonetic', ccodes=[cc]                (Symphonym KNN + strict country)
  Pass 3  PROX     — mode='phonetic', no ccodes, bounds=box      (border/name changes, but a spatial
                     box around the printed coordinates so matches can't come from the other side of
                     the world). Runs only for places that printed coordinates.

Two backends (same cascade):
  * **gateway** (default) — POST directly to the Pitt ES gateway's `/api/reconcile`
    (`gazetteer-clus.crc.pitt.edu:9200`, the cluster-facing interface — a direct local connection
    from CRC compute nodes, no firewall, no token). One `ReconcileRequest` per query, run
    concurrently; the response carries the centroid inline (`geometries[].repr_point` = [lon,lat]),
    so no separate data-extension call. Fastest; use from CRC.
  * **api** — the public endpoint `https://whgazetteer.org/reconcile` (W3C-style batched
    `{queries:{…}}`, `countries`, token from `.env`; centroid via a second `extend` POST). Works
    anywhere with internet; the gateway proxies the same Symphonym-KNN behind it.

Both honour exact/phonetic + country + spatial bounds server-side. Never filter by AAT `types`/
`fclasses` — sparsely populated, tanks recall. Threshold on `score`, not the conservative `match`.
Writes whg_match_id, whg_score, lat, lon, recon_pass, reconciliation(JSON), status.

Usage:
  python3 process/reconcile.py --seed-demo 8                       # demo (no extraction needed)
  python3 process/reconcile.py --limit 200 --concurrency 24        # gateway (default), on CRC
  python3 process/reconcile.py --backend api --concurrency 6       # public API (needs WHG_API_TOKEN)
"""
from __future__ import annotations
import argparse, json, os, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

GATEWAY_URL = os.getenv("WHG_GATEWAY_URL", "http://gazetteer-clus.crc.pitt.edu:9200")
DO_URL = "https://whgazetteer.org/reconcile"
USER_AGENT = "GOTW-reconcile/0.3 (https://github.com/docuracy/GOTW; stephen@docuracy.co.uk)"
CHUNK = 25                # queries per DO-API POST (batched); gateway is one query per POST
DEFAULT_THRESHOLD = 80
DEFAULT_RADIUS_KM = 150

# (label, mode, use_country, use_bounds)
PASSES = [
    ("1-exact",     "exact",    True,  False),
    ("2-phonetic",  "phonetic", True,  False),
    ("3-proximity", "phonetic", False, True),
]


def token() -> str:
    tok = os.getenv("WHG_API_TOKEN")
    if not tok:
        sys.exit("WHG_API_TOKEN not set — add it to .env / ~/.gotw_env (needed for --backend api)")
    return tok


def ensure_columns(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(place)")}
    for col, typ in (("reconciliation", "TEXT"), ("whg_score", "REAL"), ("recon_pass", "TEXT")):
        if col not in cols:
            con.execute(f"ALTER TABLE place ADD COLUMN {col} {typ}")
    con.commit()


def _bounds(lat, lng, radius_km):
    d = radius_km / 111.0
    return {"type": "Polygon", "coordinates": [[
        [lng - d, lat - d], [lng + d, lat - d], [lng + d, lat + d],
        [lng - d, lat + d], [lng - d, lat - d]]]}


def _post(url, body, tok=None, tries=4):
    params = {"token": tok} if tok else None
    for i in range(tries):
        try:
            r = requests.post(url, params=params, json=body,
                              headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
                              timeout=120)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if i == tries - 1:
                return {"_error": str(e)}
            time.sleep(min(2 ** i, 15))


# ── gateway backend (direct /api/reconcile, single query, coords inline) ──────
def _gw_request(place, mode, use_country, use_bounds, radius_km):
    ext = json.loads(place["extraction"]) if place["extraction"] else {}
    body = {"query": place["name"], "mode": mode, "size": 5}
    cc = ext.get("country_code")
    if use_country and cc:
        body["ccodes"] = [cc]
    if use_bounds:
        lat, lng = ext.get("latitude"), ext.get("longitude")
        if lat is None or lng is None:
            return None
        body["bounds"] = _bounds(lat, lng, radius_km)
    return body


def _gw_top(hit_list, threshold):
    if not hit_list:
        return None
    top = max(hit_list, key=lambda h: h.get("score", 0))
    if top.get("score", 0) < threshold:
        return None
    pt = None
    geos = top.get("geometries") or []
    if geos and geos[0].get("repr_point"):
        lon, lat = geos[0]["repr_point"][:2]
        pt = (lat, lon)
    return {"id": top["place_id"], "name": top.get("title"), "score": top.get("score"), "coords": pt}


def run_pass_gateway(rows, pass_cfg, threshold, concurrency):
    _, mode, uc, ub, radius = pass_cfg
    items = [(r, _gw_request(r, mode, uc, ub, radius)) for r in rows]
    items = [(r, b) for r, b in items if b is not None]
    best = {}

    def work(item):
        r, body = item
        return r["place_id"], _post(f"{GATEWAY_URL}/api/reconcile", body)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in as_completed([pool.submit(work, it) for it in items]):
            pid, resp = fut.result()
            if not resp or "_error" in resp:
                continue
            cand = _gw_top(resp.get("hits") or [], threshold)
            if cand:
                best[pid] = cand
    return best


# ── DO public-API backend (batched /reconcile, countries, extend centroids) ──
def _api_request(place, mode, use_country, use_bounds, radius_km):
    ext = json.loads(place["extraction"]) if place["extraction"] else {}
    q = {"query": place["name"], "mode": mode, "limit": 5}
    cc = ext.get("country_code")
    if use_country and cc:
        q["countries"] = [cc]
    if use_bounds:
        lat, lng = ext.get("latitude"), ext.get("longitude")
        if lat is None or lng is None:
            return None
        q["bounds"] = _bounds(lat, lng, radius_km)
    return q


def run_pass_api(rows, pass_cfg, threshold, concurrency, tok):
    _, mode, uc, ub, radius = pass_cfg
    chunks = []
    for i in range(0, len(rows), CHUNK):
        queries = {}
        for r in rows[i:i + CHUNK]:
            q = _api_request(r, mode, uc, ub, radius)
            if q is not None:
                queries[f"q{r['place_id']}"] = q
        if queries:
            chunks.append(queries)
    best = {}

    def work(queries):
        return _post(DO_URL, {"queries": queries}, tok=tok), queries

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
                        best[pid] = {"id": top["id"], "name": top.get("name"),
                                     "score": top.get("score"), "coords": None}
    return best


def fetch_centroids_api(ids, tok):
    out = {}
    for i in range(0, len(ids), CHUNK):
        d = _post(DO_URL, {"extend": {"ids": ids[i:i + CHUNK],
                                      "properties": [{"id": "whg:geometry_centroid"}]}}, tok=tok)
        for pid, props in ((d or {}).get("rows") or {}).items():
            vals = props.get("whg:geometry_centroid") or []
            if vals and vals[0].get("str"):
                lat_s, lng_s = vals[0]["str"].split(",")
                out[pid] = (float(lat_s), float(lng_s))
    return out


def reconcile(con, rows, backend, threshold, radius_km, concurrency, tok=None):
    matched = {}      # place_id -> (pass_label, candidate{id,name,score,coords})
    for label, mode, uc, ub in PASSES:
        todo = [r for r in rows if r["place_id"] not in matched]
        if not todo:
            break
        cfg = (label, mode, uc, ub, radius_km)
        got = (run_pass_gateway(todo, cfg, threshold, concurrency) if backend == "gateway"
               else run_pass_api(todo, cfg, threshold, concurrency, tok))
        for pid, cand in got.items():
            matched[pid] = (label, cand)
        print(f"pass {label:12} on {len(todo):>6}  -> matched {len(got):>5}  "
              f"(cumulative {len(matched)}/{len(rows)})", flush=True)

    if backend == "api":      # gateway already returned coords inline
        need = [c["id"] for _, c in matched.values() if c["coords"] is None]
        cents = fetch_centroids_api(need, tok)
        for _, c in matched.values():
            if c["coords"] is None:
                c["coords"] = cents.get(c["id"], (None, None))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in rows:
        pid = r["place_id"]
        if pid in matched:
            label, cand = matched[pid]
            lat, lon = cand["coords"] or (None, None)
            con.execute("UPDATE place SET whg_match_id=?, whg_score=?, lat=?, lon=?, recon_pass=?, "
                        "reconciliation=?, status='reconciled', created_at=? WHERE place_id=?",
                        (cand["id"], cand.get("score"), lat, lon, label,
                         json.dumps({"pass": label, "candidate": cand}), now, pid))
        else:
            con.execute("UPDATE place SET recon_pass='unmatched', status='unmatched', created_at=? "
                        "WHERE place_id=?", (now, pid))
    con.commit()
    by = lambda lbl: sum(1 for v in matched.values() if v[0] == lbl)
    print(f"reconciled {len(matched)}/{len(rows)} via {backend} "
          f"({by('1-exact')} exact, {by('2-phonetic')} phonetic, {by('3-proximity')} proximity)")


DEMO = [
    ("Lutterworth", "300008375", "GB", None, None), ("Luristan", "300236157", "IR", None, None),
    ("Luton", "300008375", "GB", None, None), ("Macao", "300008375", "MO", 22.19, 113.50),
    ("Luroe", "300008791", "NO", 66.39, 12.92), ("Lutry", "300008375", "CH", None, None),
    ("Lusatia", "300236157", "DE", None, None), ("Lustleigh", "300000773", "GB", None, None),
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
    ap.add_argument("--backend", choices=["gateway", "api"], default="gateway")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed-demo", type=int, metavar="N")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--retry-unmatched", action="store_true")
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
    endpoint = GATEWAY_URL if args.backend == "gateway" else DO_URL
    print(f"{len(rows)} places to reconcile — 3-pass cascade via {args.backend} ({endpoint}), threshold {args.threshold}")

    if args.dry_run:
        for r in rows[:8]:
            req = _gw_request if args.backend == "gateway" else _api_request
            qs = {lbl: req(r, m, uc, ub, args.radius_km) for lbl, m, uc, ub in PASSES}
            print(f"  {r['name'][:22]:22} {json.dumps(qs)}")
        print("(dry run: no calls)")
        return
    if rows:
        tok = token() if args.backend == "api" else None
        reconcile(con, rows, args.backend, args.threshold, args.radius_km, args.concurrency, tok)


if __name__ == "__main__":
    main()
