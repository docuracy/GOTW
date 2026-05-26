#!/usr/bin/env python3
"""Reconciliation stage: geolocate `place` rows against WHG, as a cascade.

Pass 0 (gateway only) first resolves each place's **admin hierarchy** top-down: the LLM-extracted
`admin_hierarchy` levels, broadest first, each resolved to a WHG place id that HAS GEOMETRY (`has_geom`)
and queried `contained_in` the previously-resolved parent — so we descend an actual nested region. The
country level is handled by `ccodes` (a reliable proxy), not an id lookup. The chain is recorded as
Linked-Places `gvp:broaderPartitive` (partOf) relations (stored in `reconciliation.relations`, even when
the leaf stays unmatched).

The leaf then runs the cascade — each pass on the previous one's misses, `ccodes=[cc]` throughout,
thresholding on `score`; precision first, then progressively relaxed recall:

  Pass 1  exact,    contained_in=[narrowest parent], relation=within        (strict text, strict region)
  Pass 2  phonetic, contained_in=[narrowest parent], relation=intersects    (Symphonym within the parent —
                    stops a phonetic look-alike matching on the far side of the country)
  Pass 3  phonetic, contained_in=[next-broader parent]                      (relax the region outward)
  Pass 4  phonetic, ccodes only                                             (parent had no geometry / none)
  Pass 5  phonetic, bounds=box around the printed coordinates               (coordinate-bearing fallback)

`contained_in` is WHG's server-side containment (union of the named places' geometries); far better
than the padded centroid-bbox this previously used. We request `containment="exact"` (precise Shapely
polygon test): the `fuzzy` H3 mode currently returns 0 hits even for genuinely-contained places, while
`exact` correctly admits in-region matches and rejects out-of-region phonetic look-alikes.

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
import argparse, json, os, re, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

GATEWAY_URL = os.getenv("WHG_GATEWAY_URL", "http://gazetteer-clus.crc.pitt.edu:9200")
DO_URL = "https://whgazetteer.org/reconcile"
USER_AGENT = "GOTW-reconcile/0.3 (https://github.com/WorldHistoricalGazetteer/gazetteer-of-the-world; stephen@docuracy.co.uk)"
CHUNK = 25                # queries per DO-API POST (batched); gateway is one query per POST
DEFAULT_THRESHOLD = 80
DEFAULT_RADIUS_KM = 150

# (label, mode, contain, relation, coords) — every pass also applies ccodes=[cc] when known (the reliable
# country-level proxy). `contain`: None | "narrow" (innermost geometry-bearing parent) | "broad" (next
# parent out); `relation`: WHG spatial relation for contained_in. Precision first (exact, within the
# narrowest parent), then recall (phonetic within parent → relax outward → country-only → printed coords).
PASSES = [
    ("1-exact-in",   "exact",    "narrow", "within",     False),
    ("2-phon-in",    "phonetic", "narrow", "intersects", False),
    ("3-phon-broad", "phonetic", "broad",  "intersects", False),
    ("4-phon-cc",    "phonetic", None,     None,         False),
    ("5-coords",     "phonetic", None,     None,         True),
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
def _gw_request(place, pass_cfg, parents, radius_km):
    """Build one gateway query for a leaf place. `parents` = its geometry-bearing parent place ids,
    broadest-first (narrowest last). Returns None when the pass's spatial constraint can't apply (e.g.
    a 'narrow' pass for a place with no resolved parent), so that place falls through to a later pass."""
    _, mode, contain, relation, coords = pass_cfg
    ext = json.loads(place["extraction"]) if place["extraction"] else {}
    body = {"query": place["name"], "mode": mode, "size": 5}
    cc = ext.get("country_code")
    if cc:
        body["ccodes"] = [cc]                              # country-level proxy, always on when known
    if contain == "narrow":
        if not parents:
            return None
        body.update(contained_in=[parents[-1]], containment="exact", relation=relation)
    elif contain == "broad":
        if len(parents) < 2:
            return None
        body.update(contained_in=[parents[-2]], containment="exact", relation=relation)
    if coords:
        lat, lng = ext.get("latitude"), ext.get("longitude")
        if lat is None or lng is None:
            return None
        body["bounds"] = _bounds(lat, lng, radius_km)
    return body


# Non-point tie-break: when the best hit is a point but a POLYGON-bearing candidate (has_geom) sits within
# GEOM_MARGIN of the top score, prefer the polygon — richer geometry + better map display, without reaching
# past a clearly-better match. Tunable via GOTW_GEOM_MARGIN. Set GOTW_MEASURE=1 to log every match's
# top-vs-best-polygon gap (no behaviour change) for impact analysis; dumped on exit to GOTW_MEASURE_OUT.
GEOM_MARGIN = float(os.environ.get("GOTW_GEOM_MARGIN", "1"))
_MEASURE_ON = os.environ.get("GOTW_MEASURE") == "1"
_MEASURE: list = []
if _MEASURE_ON:
    import atexit
    atexit.register(lambda: open(os.environ.get("GOTW_MEASURE_OUT", "/tmp/gotw_measure.json"), "w")
                    .write(json.dumps(_MEASURE)))


def _gw_top(hit_list, threshold, label=None):
    if not hit_list:
        return None
    top = max(hit_list, key=lambda h: h.get("score", 0))
    if top.get("score", 0) < threshold:
        return None
    if not _has_geom(top):
        polys = [h for h in hit_list if _has_geom(h)]
        best_poly = max(polys, key=lambda h: h.get("score", 0)) if polys else None
        if _MEASURE_ON:
            _MEASURE.append({"label": label, "top_id": top.get("place_id"), "top_title": top.get("title"),
                             "top_score": top.get("score"), "top_poly": False,
                             "poly_id": best_poly.get("place_id") if best_poly else None,
                             "poly_title": best_poly.get("title") if best_poly else None,
                             "poly_score": best_poly.get("score") if best_poly else None,
                             "gap": round(top.get("score", 0) - best_poly.get("score", 0), 3) if best_poly else None})
        if best_poly and top.get("score", 0) - best_poly.get("score", 0) <= GEOM_MARGIN:
            top = best_poly                       # prefer the (near-tied) polygon
    elif _MEASURE_ON:
        _MEASURE.append({"label": label, "top_id": top.get("place_id"), "top_title": top.get("title"),
                         "top_score": top.get("score"), "top_poly": True})
    pt = None
    geos = top.get("geometries") or []
    if geos and geos[0].get("repr_point"):
        lon, lat = geos[0]["repr_point"][:2]
        pt = (lat, lon)
    return {"id": top["place_id"], "name": top.get("title"), "score": top.get("score"), "coords": pt}


# ── admin-hierarchy resolution (top-down, server-side containment) ────────────
# Resolve a place's admin parents broadest-first to WHG place ids that HAVE GEOMETRY (`has_geom`), each
# query contained_in the previously-resolved parent so we descend an actual nested region — not a padded
# bbox. The country level is left to `ccodes` (a reliable proxy), so only sub-country admin units are
# resolved to ids. The leaf cascade then constrains matches with `contained_in=[parent id]`, and the
# chain is recorded as Linked-Places `gvp:broaderPartitive` relations. Gateway-only.
_PARENT_CACHE: dict = {}          # (norm_name, ccode, container_ids|None) -> {id,name,score} | None
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")    # drop a trailing type tag, e.g. "Fars (province)" -> "Fars"


def _clean_admin(name):
    return _PAREN.sub("", name or "").strip()


def _norm(s):
    return " ".join((s or "").lower().split())


def _has_geom(hit):
    # has_geom flags a POLYGON-type geometry — the only kind that can contain children (a point/line
    # parent can't). The gateway must surface it in the /api/reconcile geometries (see WHG reconcile API).
    return any(g.get("has_geom") for g in (hit.get("geometries") or []))


def _resolve_one(name, cc, container_ids, threshold):
    """Resolve one admin name to a WHG place id WITH GEOMETRY, within its country (ccodes) and inside the
    already-resolved ancestor (`contained_in`). Cached per (name, cc, container). Returns {id,name,score}|None."""
    key = (_norm(name), cc, tuple(container_ids) if container_ids else None)
    if key in _PARENT_CACHE:
        return _PARENT_CACHE[key]
    res = None
    for mode in ("exact", "phonetic"):
        body = {"query": name, "mode": mode, "size": 8}
        if cc:
            body["ccodes"] = [cc]
        if container_ids:
            body.update(contained_in=list(container_ids), containment="exact", relation="intersects")
        resp = _post(f"{GATEWAY_URL}/api/reconcile", body)
        if not resp or "_error" in resp:
            continue
        # only a geometry-bearing candidate can constrain children / be a usable container
        hits = [h for h in (resp.get("hits") or []) if h.get("score", 0) >= threshold and _has_geom(h)]
        if not hits:
            continue
        top = max(hits, key=lambda h: h.get("score", 0))
        res = {"id": top["place_id"], "name": top.get("title"), "score": top.get("score")}
        break
    _PARENT_CACHE[key] = res
    return res


def resolve_hierarchy(rows, threshold):
    """Per place: resolve its admin_hierarchy (broadest→narrowest) to geometry-bearing parent ids, each
    contained_in the previous. Returns (parents {pid: [ids broadest-first]}, relations {pid: [LPF rels]}).
    Sequential, but the cache collapses shared ancestors (every 'Essex' resolved once)."""
    parents_by_pid, relations = {}, {}
    for r in rows:
        ext = json.loads(r["extraction"]) if r["extraction"] else {}
        cc = ext.get("country_code")
        chain, seen = [], set()
        for name in (_clean_admin(n) for n in (ext.get("admin_hierarchy") or [])):
            if name and _norm(name) not in seen:
                chain.append(name); seen.add(_norm(name))
        ids, rels, container = [], [], []
        for name in chain:
            par = _resolve_one(name, cc, container, threshold)
            if not par:
                continue
            ids.append(par["id"])
            rels.append({"relationType": "gvp:broaderPartitive", "relationTo": par["id"],
                         "label": par["name"], "when": None})
            container = [par["id"]]        # next level must lie within this parent
        if ids:
            parents_by_pid[r["place_id"]] = ids
        if rels:
            relations[r["place_id"]] = rels
    return parents_by_pid, relations


def run_pass_gateway(rows, pass_cfg, radius_km, threshold, concurrency, parents_by_pid):
    items = [(r, _gw_request(r, pass_cfg, parents_by_pid.get(r["place_id"], []), radius_km)) for r in rows]
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
            cand = _gw_top(resp.get("hits") or [], threshold, pass_cfg[0])
            if cand:
                best[pid] = cand
    return best


# ── DO public-API backend (batched /reconcile, countries, extend centroids) ──
# No per-place hierarchy here (the W3C result carries no geometry/has_geom inline), so the containment
# passes collapse: a "narrow" pass runs as country-only, "broad"/country-only passes are skipped as
# redundant, and the coords pass uses bounds. Effectively exact-cc → phonetic-cc → coords.
def _api_request(place, pass_cfg, radius_km):
    _, mode, contain, relation, coords = pass_cfg
    ext = json.loads(place["extraction"]) if place["extraction"] else {}
    q = {"query": place["name"], "mode": mode, "limit": 5}
    cc = ext.get("country_code")
    if cc:
        q["countries"] = [cc]
    if coords:
        lat, lng = ext.get("latitude"), ext.get("longitude")
        if lat is None or lng is None:
            return None
        q["bounds"] = _bounds(lat, lng, radius_km)
    elif contain != "narrow":          # "broad"/country-only passes duplicate the "narrow"→cc query here
        return None
    return q


def run_pass_api(rows, pass_cfg, radius_km, threshold, concurrency, tok):
    chunks = []
    for i in range(0, len(rows), CHUNK):
        queries = {}
        for r in rows[i:i + CHUNK]:
            q = _api_request(r, pass_cfg, radius_km)
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


def reconcile(con, rows, backend, threshold, radius_km, concurrency, tok=None, hierarchy=True):
    # Pass 0: resolve admin hierarchies top-down (gateway only) so the leaf passes can be bounded by the
    # parent footprint, and so we can record the chain as LOD relations.
    parents_by_pid, relations = {}, {}
    use_hier = hierarchy and backend == "gateway"
    if use_hier:
        print("resolving admin hierarchies (top-down, server-side containment) …", flush=True)
        parents_by_pid, relations = resolve_hierarchy(rows, threshold)
        print(f"  geometry-bearing parent chain for {len(parents_by_pid)}/{len(rows)} places; "
              f"relations for {len(relations)}; {len(_PARENT_CACHE)} distinct parents cached", flush=True)

    matched = {}      # place_id -> (pass_label, candidate{id,name,score,coords})
    for cfg in PASSES:
        label, _, contain, _, _ = cfg
        if contain in ("narrow", "broad") and not use_hier:   # containment passes need the resolved parents
            continue
        todo = [r for r in rows if r["place_id"] not in matched]
        if not todo:
            break
        got = (run_pass_gateway(todo, cfg, radius_km, threshold, concurrency, parents_by_pid)
               if backend == "gateway" else run_pass_api(todo, cfg, radius_km, threshold, concurrency, tok))
        for pid, cand in got.items():
            matched[pid] = (label, cand)
        print(f"pass {label:13} on {len(todo):>6}  -> matched {len(got):>5}  "
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
        rels = relations.get(pid) or []      # partOf chain recorded even when the leaf itself is unmatched
        if pid in matched:
            label, cand = matched[pid]
            lat, lon = cand["coords"] or (None, None)
            con.execute("UPDATE place SET whg_match_id=?, whg_score=?, lat=?, lon=?, recon_pass=?, "
                        "reconciliation=?, status='reconciled', created_at=? WHERE place_id=?",
                        (cand["id"], cand.get("score"), lat, lon, label,
                         json.dumps({"pass": label, "candidate": cand, "relations": rels}), now, pid))
        else:
            con.execute("UPDATE place SET recon_pass='unmatched', status='unmatched', reconciliation=?, "
                        "created_at=? WHERE place_id=?",
                        (json.dumps({"relations": rels}) if rels else None, now, pid))
    con.commit()
    by = lambda lbl: sum(1 for v in matched.values() if v[0] == lbl)
    print(f"reconciled {len(matched)}/{len(rows)} via {backend} ("
          + ", ".join(f"{by(c[0])} {c[0]}" for c in PASSES)
          + f"); {sum(len(v) for v in relations.values())} partOf relations")


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
    ap.add_argument("--no-hierarchy", action="store_true",
                    help="skip top-down admin-parent resolution / partOf relations (gateway only)")
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
    hier = (not args.no_hierarchy) and args.backend == "gateway"
    print(f"{len(rows)} places to reconcile — cascade via {args.backend} ({endpoint}), threshold "
          f"{args.threshold}{', hierarchy-aware' if hier else ''}")

    if args.dry_run:
        parents = ({} if (args.no_hierarchy or args.backend != "gateway")
                   else resolve_hierarchy(rows[:8], args.threshold)[0])
        for r in rows[:8]:
            if args.backend == "gateway":
                qs = {c[0]: _gw_request(r, c, parents.get(r["place_id"], []), args.radius_km) for c in PASSES}
            else:
                qs = {c[0]: _api_request(r, c, args.radius_km) for c in PASSES}
            print(f"  {r['name'][:22]:22} {json.dumps(qs)}")
        print("(dry run: no calls)")
        return
    if rows:
        tok = token() if args.backend == "api" else None
        reconcile(con, rows, args.backend, args.threshold, args.radius_km, args.concurrency, tok,
                  hierarchy=not args.no_hierarchy)


if __name__ == "__main__":
    main()
