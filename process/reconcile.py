#!/usr/bin/env python3
"""Reconciliation stage: geolocate `place` rows against WHG, as a cascade.

Pass 0 (gateway only) first resolves each place's **admin hierarchy** top-down — country, then the
LLM-extracted `admin_hierarchy` levels, broadest first — caching each parent's WHG footprint so the
parent's bbox bounds the search for the next level down. This both (a) yields a spatial bound for the
many coordinate-less parishes and (b) records the chain as Linked-Places `gvp:broaderPartitive`
(partOf) relations to the parents' WHG ids (stored in `reconciliation.relations`, even when the leaf
itself stays unmatched).

The leaf then runs the cascade — each pass on the previous one's misses (precision first, then
recall), thresholding on `score`:

  Pass 1  EXACT    — mode='exact',    ccodes=[cc]                (strict text + strict country)
  Pass 2  PHONETIC — mode='phonetic', ccodes=[cc]                (Symphonym KNN + strict country)
  Pass 3  PROX     — mode='phonetic', no ccodes, bounds=box      (border/name changes, but a spatial
                     box around the printed coordinates so matches can't come from the other side of
                     the world). Runs only for places that printed coordinates.
  Pass 4  PARENT   — mode='phonetic', ccodes=[cc], bounds=parent (coordinate-less places: bound by the
                     reconciled immediate-parent footprint from Pass 0). Gateway/hierarchy only.

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
USER_AGENT = "GOTW-reconcile/0.3 (https://github.com/docuracy/GOTW; stephen@docuracy.co.uk)"
CHUNK = 25                # queries per DO-API POST (batched); gateway is one query per POST
DEFAULT_THRESHOLD = 80
DEFAULT_RADIUS_KM = 150

# (label, mode, use_country, bound_src)   bound_src: "none" | "coords" (printed) | "parent" (admin geometry)
PASSES = [
    ("1-exact",     "exact",    True,  "none"),
    ("2-phonetic",  "phonetic", True,  "none"),
    ("3-proximity", "phonetic", False, "coords"),
    ("4-parent",    "phonetic", True,  "parent"),   # coordinate-less places: bound by the reconciled admin parent
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
def _gw_request(place, mode, use_country, bound_src, radius_km, parent_bbox=None):
    ext = json.loads(place["extraction"]) if place["extraction"] else {}
    body = {"query": place["name"], "mode": mode, "size": 5}
    cc = ext.get("country_code")
    if use_country and cc:
        body["ccodes"] = [cc]
    if bound_src == "coords":
        lat, lng = ext.get("latitude"), ext.get("longitude")
        if lat is None or lng is None:
            return None
        body["bounds"] = _bounds(lat, lng, radius_km)
    elif bound_src == "parent":
        if not parent_bbox:                 # only runs for places whose admin parent resolved to a footprint
            return None
        body["bounds"] = _poly_from_bbox(parent_bbox)
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


# ── admin-hierarchy resolution (top-down, geometry-constraining) ──────────────
# Resolve a place's admin parents broadest-first, caching each parent's WHG footprint, so a parent's
# bbox bounds the search for the next level down — decisive for the parish majority that print no
# coordinates — and the chain is recorded as Linked-Places `gvp:broaderPartitive` relations to the
# parents' WHG ids. Gateway-only (the public API path keeps the plain 3-pass cascade).
_PARENT_CACHE: dict = {}          # (norm_name, ccode, ancestor_bbox|None) -> {id,name,bbox,centroid,score} | None


_PAREN = re.compile(r"\s*\([^)]*\)\s*$")    # drop a trailing type tag, e.g. "Fars (province)" -> "Fars"


def _clean_admin(name):
    return _PAREN.sub("", name or "").strip()


def _norm(s):
    return " ".join((s or "").lower().split())


def _iter_lonlat(coords):
    """Yield (lon, lat) from any GeoJSON coordinate nesting (Point/Line/Polygon/Multi*)."""
    if coords and isinstance(coords[0], (int, float)):
        yield coords[0], coords[1]
        return
    for c in coords or []:
        yield from _iter_lonlat(c)


def _bbox_of_hit(hit):
    """[w,s,e,n] extent of a gateway hit — from its geometry if present, else its repr_point.
    NB: confirm the gateway hit's geometry shape against a live response; falls back to centroid."""
    xs, ys = [], []
    for g in hit.get("geometries") or []:
        geom = g.get("geometry") or (g if g.get("type") in ("Point", "Polygon", "MultiPolygon",
                                                             "LineString", "MultiLineString") else None)
        if geom and geom.get("coordinates"):
            for lon, lat in _iter_lonlat(geom["coordinates"]):
                xs.append(lon); ys.append(lat)
        rp = g.get("repr_point")
        if rp:
            xs.append(rp[0]); ys.append(rp[1])
    return [min(xs), min(ys), max(xs), max(ys)] if xs else None


def _poly_from_bbox(bbox, pad_km=10):
    w, s, e, n = bbox
    d = pad_km / 111.0
    return {"type": "Polygon", "coordinates": [[
        [w - d, s - d], [e + d, s - d], [e + d, n + d], [w - d, n + d], [w - d, s - d]]]}


def _resolve_one(name, cc, ancestor_bbox, threshold):
    """Reconcile a single admin name against the gateway (exact then phonetic), inside a country and an
    optional ancestor bbox. Cached per (name, cc, ancestor). Returns {id,name,bbox,centroid,score}|None."""
    key = (_norm(name), cc, tuple(ancestor_bbox) if ancestor_bbox else None)
    if key in _PARENT_CACHE:
        return _PARENT_CACHE[key]
    res = None
    for mode in ("exact", "phonetic"):
        body = {"query": name, "mode": mode, "size": 5}
        if cc:
            body["ccodes"] = [cc]
        if ancestor_bbox:
            body["bounds"] = _poly_from_bbox(ancestor_bbox)
        resp = _post(f"{GATEWAY_URL}/api/reconcile", body)
        if not resp or "_error" in resp:
            continue
        hits = resp.get("hits") or []
        if not hits:
            continue
        top = max(hits, key=lambda h: h.get("score", 0))
        if top.get("score", 0) < threshold:
            continue
        geos = top.get("geometries") or []
        rp = geos[0]["repr_point"][:2] if (geos and geos[0].get("repr_point")) else None
        res = {"id": top["place_id"], "name": top.get("title"), "bbox": _bbox_of_hit(top),
               "centroid": rp, "score": top.get("score")}
        break
    _PARENT_CACHE[key] = res
    return res


def resolve_hierarchy(rows, threshold):
    """For each place resolve country -> admin parents (broadest first), each bounded by the previously
    resolved level's bbox. Returns (parent_bboxes {pid: immediate-parent bbox for leaf bounding},
    relations {pid: [LPF partOf rels]}). Sequential, but the cache collapses the shared parents (every
    'England'/'Essex' is resolved once)."""
    parent_bboxes, relations = {}, {}
    for r in rows:
        ext = json.loads(r["extraction"]) if r["extraction"] else {}
        cc = ext.get("country_code")
        # chain broadest -> narrowest: country, then admin_hierarchy (assumed broad->narrow; verify on data)
        chain, seen = [], set()
        raw = ([ext["country"]] if ext.get("country") else []) + (ext.get("admin_hierarchy") or [])
        for name in (_clean_admin(n) for n in raw):
            if name and _norm(name) not in seen:
                chain.append(name); seen.add(_norm(name))
        rels, bound = [], None
        for name in chain:
            par = _resolve_one(name, cc, bound, threshold)
            if not par:
                continue
            rels.append({"relationType": "gvp:broaderPartitive", "relationTo": par["id"],
                         "label": par["name"], "when": None})
            if par["bbox"]:
                bound = par["bbox"]        # tighten the spatial bound for the next level down
        if bound:
            parent_bboxes[r["place_id"]] = bound
        if rels:
            relations[r["place_id"]] = rels
    return parent_bboxes, relations


def run_pass_gateway(rows, pass_cfg, threshold, concurrency, parent_bboxes=None):
    _, mode, uc, bsrc, radius = pass_cfg
    pb = parent_bboxes or {}
    items = [(r, _gw_request(r, mode, uc, bsrc, radius, pb.get(r["place_id"]))) for r in rows]
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
# Keeps the plain cascade only: the "parent" bound source is gateway-only (no hierarchy here), so a
# "parent" pass produces no queries and is skipped.
def _api_request(place, mode, use_country, bound_src, radius_km):
    ext = json.loads(place["extraction"]) if place["extraction"] else {}
    q = {"query": place["name"], "mode": mode, "limit": 5}
    cc = ext.get("country_code")
    if use_country and cc:
        q["countries"] = [cc]
    if bound_src == "coords":
        lat, lng = ext.get("latitude"), ext.get("longitude")
        if lat is None or lng is None:
            return None
        q["bounds"] = _bounds(lat, lng, radius_km)
    elif bound_src == "parent":
        return None
    return q


def run_pass_api(rows, pass_cfg, threshold, concurrency, tok):
    _, mode, uc, bsrc, radius = pass_cfg
    chunks = []
    for i in range(0, len(rows), CHUNK):
        queries = {}
        for r in rows[i:i + CHUNK]:
            q = _api_request(r, mode, uc, bsrc, radius)
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
    parent_bboxes, relations = {}, {}
    use_hier = hierarchy and backend == "gateway"
    if use_hier:
        print("resolving admin hierarchies (top-down, geometry-constrained) …", flush=True)
        parent_bboxes, relations = resolve_hierarchy(rows, threshold)
        print(f"  parent footprint for {len(parent_bboxes)}/{len(rows)} places; "
              f"relations for {len(relations)}; {len(_PARENT_CACHE)} distinct parents cached", flush=True)

    matched = {}      # place_id -> (pass_label, candidate{id,name,score,coords})
    for label, mode, uc, bsrc in PASSES:
        if bsrc == "parent" and not use_hier:        # parent-bounded pass is hierarchy/gateway-only
            continue
        todo = [r for r in rows if r["place_id"] not in matched]
        if not todo:
            break
        cfg = (label, mode, uc, bsrc, radius_km)
        got = (run_pass_gateway(todo, cfg, threshold, concurrency, parent_bboxes) if backend == "gateway"
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
    print(f"reconciled {len(matched)}/{len(rows)} via {backend} "
          f"({by('1-exact')} exact, {by('2-phonetic')} phonetic, {by('3-proximity')} proximity, "
          f"{by('4-parent')} parent-bounded); {sum(len(v) for v in relations.values())} partOf relations")


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
        for r in rows[:8]:
            req = _gw_request if args.backend == "gateway" else _api_request
            qs = {lbl: req(r, m, uc, bsrc, args.radius_km) for lbl, m, uc, bsrc in PASSES}
            print(f"  {r['name'][:22]:22} {json.dumps(qs)}")
        print("(dry run: no calls)")
        return
    if rows:
        tok = token() if args.backend == "api" else None
        reconcile(con, rows, args.backend, args.threshold, args.radius_km, args.concurrency, tok,
                  hierarchy=not args.no_hierarchy)


if __name__ == "__main__":
    main()
