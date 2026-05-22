"""QA: flag suspect entries (intrinsic signals) and run a LOOSE reasonableness check of each
suspect's headword against the WHG toponyms index (gateway). Writes a `qa` table — it NEVER
alters OCR text, it only routes entries to ok-in-index vs needs-review.

Run in the whg conda env on a CRC compute node (gateway reachable, no token):
    python3 process/flag_suspects.py data/gotw_seg.sqlite
"""
import sqlite3, re, sys, json, urllib.request, os

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


def gw_query(name, mode="fuzzy", size=3):
    body = json.dumps({"query": name, "mode": mode, "size": size}).encode()
    req = urllib.request.Request(GATEWAY, data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=20))
        hits = r.get("hits", [])
        return (hits[0].get("score"), hits[0].get("title", "")) if hits else (0.0, "")
    except Exception as e:
        return (None, f"ERR {e}"[:50])


# --- gateway self-test ---
for probe in ("Acre", "Mozambique", "Tna", "Xqzzr"):
    print(f"  probe {probe!r:14} -> {gw_query(probe)}")

con = sqlite3.connect(DB)
print("entries:", con.execute("SELECT COUNT(*) FROM entry").fetchone()[0],
      "max-len:", con.execute("SELECT max(length(text)) FROM entry").fetchone()[0])
con.execute("DROP TABLE IF EXISTS qa")
con.execute("CREATE TABLE qa(entry_id INTEGER PRIMARY KEY, flags TEXT, idx_score REAL, "
            "idx_hit TEXT, verdict TEXT)")

suspects = []
for eid, hw, disp, text, kind in con.execute(
        "SELECT entry_id,headword,headword_disp,text,kind FROM entry"):
    fl = intrinsic_flags(hw, text, kind)
    if fl:
        suspects.append((eid, disp, fl))
print("intrinsic suspects:", len(suspects))

for eid, disp, fl in suspects:
    score, hit = gw_query(disp)
    verdict = "review" if (score is None or score < 90) else "ok-in-index"
    con.execute("INSERT INTO qa VALUES(?,?,?,?,?)",
                (eid, ",".join(fl), score if isinstance(score, (int, float)) else None, hit, verdict))
con.commit()
print("verdicts:", dict(con.execute("SELECT verdict,COUNT(*) FROM qa GROUP BY verdict").fetchall()))
