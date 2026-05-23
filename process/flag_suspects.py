"""QA: flag suspect entries (intrinsic signals) and run a LOOSE reasonableness check of each
suspect's headword against the WHG toponyms index (gateway). Writes a `qa` table — it NEVER
alters OCR text, it only routes entries to ok-in-index vs needs-review.

Run in the whg conda env on a CRC compute node (gateway reachable, no token):
    python3 process/flag_suspects.py data/gotw_seg.sqlite
"""
import sqlite3, re, sys, json, urllib.request, os, difflib

DB = sys.argv[1] if len(sys.argv) > 1 else "data/gotw_seg.sqlite"
GATEWAY = os.getenv("WHG_GATEWAY_URL", "http://gazetteer-clus.crc.pitt.edu:9200") + "/api/reconcile"

VOWEL = re.compile(r"[AEIOUY]")
DESCR = re.compile(r"\b(a|an|the|see|or|is|in|of|on|near|chief|town|vil|river|prov|cap"
                   r"|seaport|port|par|co|dep|gov|dist|reg|isl|lake|mt|cape)\b", re.I)


def alpha(s):
    return re.sub(r"[^A-Z]", "", (s or "").upper())


def intrinsic_flags(hw, text, kind):
    if kind != "entry":
        return []
    f, k = [], alpha(hw)
    if len(k) <= 3:
        f.append("short-headword")
    if k and not VOWEL.search(k):
        f.append("no-vowel")
    if len(text) < 35:
        f.append("short-text")
    if len(text) > 40000:
        f.append("very-long")          # blob candidate (legit continents pass the index check)
    return f


def gw(name, mode, size):
    body = json.dumps({"query": name, "mode": mode, "size": size}).encode()
    req = urllib.request.Request(GATEWAY, data=body, headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=20)).get("hits", [])
    except Exception:
        return None


def assess(disp):
    """Loose reasonableness: exact-match in the index => verbatim known toponym; else best string
    similarity to a fuzzy hit's names (the gateway 'score' is rank, not similarity, so we compute it).
    Returns (verdict, similarity 0..1, best_hit_name)."""
    q = alpha(disp)
    ex = gw(disp, "exact", 1)
    if ex is None:
        return ("review", None, "ERR")          # gateway unreachable
    if ex:
        return ("ok-in-index", 1.0, ex[0].get("title", ""))
    def nstr(nm):
        if isinstance(nm, str):
            return nm
        if isinstance(nm, dict):
            return (nm.get("toponym") or nm.get("name") or nm.get("label")
                    or next((v for v in nm.values() if isinstance(v, str)), ""))
        return ""
    best, bt = 0.0, ""
    for h in (gw(disp, "fuzzy", 5) or []):
        cands = [h.get("title", "")] + [nstr(nm) for nm in (h.get("names") or [])]
        for nm in cands:
            r = difflib.SequenceMatcher(None, q, alpha(nm)).ratio()
            if r > best:
                best, bt = r, nm
    return ("likely-variant" if best >= 0.88 else "review", round(best, 3), bt)


# --- gateway self-test ---
for probe in ("Acre", "Mozambique", "Tna", "Xqzzr", "Maulmain"):
    print(f"  probe {probe!r:14} -> {assess(probe)}")

con = sqlite3.connect(DB)
print("entries:", con.execute("SELECT COUNT(*) FROM entry").fetchone()[0],
      "max-len:", con.execute("SELECT max(length(text)) FROM entry").fetchone()[0])
con.execute("DROP TABLE IF EXISTS qa")
con.execute("CREATE TABLE qa(entry_id INTEGER PRIMARY KEY, flags TEXT, idx_sim REAL, "
            "idx_hit TEXT, verdict TEXT)")

suspects = []
for eid, hw, disp, text, kind in con.execute(
        "SELECT entry_id,headword,headword_disp,text,kind FROM entry"):
    fl = intrinsic_flags(disp, text, kind)       # use the FULL headword (incl. parenthetical, e.g.
    if fl:                                        # "Yvi (Saint)" -> not short) for length-based flags
        suspects.append((eid, disp, fl))
print("intrinsic suspects:", len(suspects))

for eid, disp, fl in suspects:
    verdict, sim, hit = assess(disp)
    con.execute("INSERT INTO qa VALUES(?,?,?,?,?)", (eid, ",".join(fl), sim, hit, verdict))
con.commit()
print("verdicts:", dict(con.execute("SELECT verdict,COUNT(*) FROM qa GROUP BY verdict").fetchall()))
