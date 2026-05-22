"""Local browser UI to review/resolve suspect gazetteer entries flagged by flag_suspects.py.

Read-mostly over the parsed `entry` table + the `qa` flags. Decisions are recorded in a `review`
table — the OCR / entry text is NEVER mutated here; we store an action + payload that the eventual
re-parse/re-extract step applies. Designed for the irreducible "really difficult cases"; the same
work-list later drives the Django QA module (see memory human-review-and-entity-types).

    .venv/bin/python process/review_ui.py                 # http://127.0.0.1:5000
    .venv/bin/python process/review_ui.py --db data/gotw_seg.sqlite --port 5000
"""
import sqlite3, json, argparse, datetime
from flask import Flask, request, jsonify, Response

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/gotw_seg.sqlite")
ap.add_argument("--port", type=int, default=5000)
ARGS = ap.parse_args()

app = Flask(__name__)


def db():
    c = sqlite3.connect(ARGS.db)
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE IF NOT EXISTS review(entry_id INTEGER PRIMARY KEY, action TEXT, "
              "payload TEXT, ts TEXT)")
    return c


def has_qa(c):
    return c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='qa'").fetchone()


@app.route("/api/list")
def api_list():
    verdict = request.args.get("verdict", "review")
    decided = request.args.get("decided", "undecided")
    c = db()
    qa = has_qa(c)
    where, params = ["e.kind='entry'"], []
    join = ""
    sel = ("e.entry_id, e.headword_disp, s.filename vol, e.page_start, length(e.text) tlen, "
           "r.action decision")
    if qa:
        sel += ", q.flags, q.idx_sim, q.idx_hit, q.verdict"
        join = "LEFT JOIN qa q ON q.entry_id=e.entry_id"
        where.append("q.entry_id IS NOT NULL")
        if verdict != "all":
            where.append("q.verdict=?")
            params.append(verdict)
    sql = (f"SELECT {sel} FROM entry e JOIN source s ON e.source_id=s.source_id "
           f"LEFT JOIN review r ON r.entry_id=e.entry_id {join} WHERE " + " AND ".join(where))
    rows = [dict(x) for x in c.execute(sql, params)]
    if decided == "undecided":
        rows = [x for x in rows if not x.get("decision")]
    elif decided == "decided":
        rows = [x for x in rows if x.get("decision")]
    rows.sort(key=lambda x: (x.get("idx_sim") if x.get("idx_sim") is not None else -1, x["tlen"]))
    return jsonify(total=len(rows), rows=rows[:1000])


@app.route("/api/entry/<int:eid>")
def api_entry(eid):
    c = db()
    e = c.execute("SELECT e.*, s.filename vol FROM entry e JOIN source s ON e.source_id=s.source_id "
                  "WHERE e.entry_id=?", (eid,)).fetchone()
    if not e:
        return jsonify(error="not found"), 404
    e = dict(e)
    nbr = {}
    for label, op, order in (("prev", "<", "DESC"), ("next", ">", "ASC")):
        r = c.execute(f"SELECT headword_disp, substr(text,1,160) snip FROM entry "
                      f"WHERE source_id=? AND seq {op} ? ORDER BY seq {order} LIMIT 1",
                      (e["source_id"], e["seq"])).fetchone()
        nbr[label] = dict(r) if r else None
    if has_qa(c):
        q = c.execute("SELECT flags, idx_sim, idx_hit, verdict FROM qa WHERE entry_id=?", (eid,)).fetchone()
        e["qa"] = dict(q) if q else None
    r = c.execute("SELECT action, payload FROM review WHERE entry_id=?", (eid,)).fetchone()
    e["decision"] = dict(r) if r else None
    e["neighbors"] = nbr
    return jsonify(e)


@app.route("/api/decide", methods=["POST"])
def api_decide():
    d = request.get_json(force=True)
    c = db()
    c.execute("INSERT OR REPLACE INTO review(entry_id, action, payload, ts) VALUES(?,?,?,?)",
              (d["entry_id"], d["action"], json.dumps(d.get("payload", {})),
               datetime.datetime.now().isoformat(timespec="seconds")))
    c.commit()
    return jsonify(ok=True)


@app.route("/api/stats")
def api_stats():
    c = db()
    out = {"reviewed": c.execute("SELECT COUNT(*) FROM review").fetchone()[0]}
    out["by_action"] = dict(c.execute("SELECT action,COUNT(*) FROM review GROUP BY action").fetchall())
    if has_qa(c):
        out["by_verdict"] = dict(c.execute("SELECT verdict,COUNT(*) FROM qa GROUP BY verdict").fetchall())
    return jsonify(out)


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>GOTW segmentation review</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;display:flex;height:100vh}
 #list{width:340px;border-right:1px solid #ccc;overflow:auto;background:#fafafa}
 #list .it{padding:6px 10px;border-bottom:1px solid #eee;cursor:pointer}
 #list .it:hover{background:#eef} #list .it.done{opacity:.45}
 #list .hw{font-weight:600} #list .meta{color:#888;font-size:12px}
 #detail{flex:1;padding:18px 24px;overflow:auto}
 .tag{display:inline-block;background:#fee;color:#900;border-radius:3px;padding:1px 6px;margin:0 4px 0 0;font-size:12px}
 .nbr{background:#f4f4f4;border-left:3px solid #ccc;padding:6px 10px;margin:6px 0;color:#444}
 .txt{white-space:pre-wrap;background:#fff;border:1px solid #ddd;padding:10px;max-height:40vh;overflow:auto}
 button{font:14px system-ui;padding:6px 12px;margin:3px;border:1px solid #aaa;border-radius:5px;background:#fff;cursor:pointer}
 button.act{background:#eef} #bar{margin:10px 0}
 #toolbar{padding:8px 10px;border-bottom:1px solid #ccc;background:#f0f0f0}
 .score{font-weight:600} input[type=text]{padding:5px;width:60%}
</style></head><body>
<div id="list"><div id="toolbar">
  <select id="fverdict" onchange="load()"><option value="review">needs review</option>
   <option value="ok-in-index">ok-in-index</option><option value="all">all flagged</option></select>
  <select id="fdecided" onchange="load()"><option value="undecided">undecided</option>
   <option value="decided">decided</option><option value="all">all</option></select>
  <span id="count"></span></div><div id="items"></div></div>
<div id="detail">Select an entry…</div>
<script>
let cur=null;
async function load(){
  const v=fverdict.value, d=fdecided.value;
  const r=await(await fetch(`/api/list?verdict=${v}&decided=${d}`)).json();
  count.textContent=r.total+" entries";
  items.innerHTML=r.rows.map(x=>`<div class="it ${x.decision?'done':''}" onclick="show(${x.entry_id})">
    <div class="hw">${x.headword_disp||''} ${x.decision?'· '+x.decision:''}</div>
    <div class="meta">${x.vol.slice(5,7)} p${x.page_start??'?'} · ${x.tlen} ch · idx ${x.idx_sim??'–'} · ${x.flags||''}</div></div>`).join('');
}
async function show(id){
  const e=await(await fetch(`/api/entry/${id}`)).json(); cur=e;
  const q=e.qa||{}, n=e.neighbors||{};
  detail.innerHTML=`<h2>${e.headword_disp} <small style="color:#888">[${e.headword_raw}]</small></h2>
   <div>${e.vol.slice(5,7)} · page ${e.page_start??'?'} · ${e.kind} · ${e.text.length} chars
     · <span class="score">index ${q.idx_sim??'–'}</span> ${q.idx_hit?('→ '+q.idx_hit):''}</div>
   <div style="margin:6px 0">${(q.flags||'').split(',').filter(Boolean).map(f=>`<span class="tag">${f}</span>`).join('')}
     ${e.decision?`<b style="color:#070">decided: ${e.decision.action}</b>`:''}</div>
   ${n.prev?`<div class="nbr"><b>↑ prev:</b> ${n.prev.headword_disp} — ${n.prev.snip}…</div>`:''}
   <div class="txt">${e.text.replace(/</g,'&lt;').slice(0,8000)}${e.text.length>8000?' …[truncated]':''}</div>
   ${n.next?`<div class="nbr"><b>↓ next:</b> ${n.next.headword_disp} — ${n.next.snip}…</div>`:''}
   <div id="bar">
     <button class="act" onclick="decide('keep')">✓ Keep (real place)</button>
     <button class="act" onclick="decide('reject')">✗ Reject (spurious/garbage)</button>
     <button class="act" onclick="decide('table')">▦ Table content</button>
     <button class="act" onclick="decide('people')">◐ People/ethnonym</button>
     <button class="act" onclick="decide('merge_prev')">⇧ Merge into prev</button>
     <button class="act" onclick="splitPrompt()">✂ Split…</button>
     <button class="act" onclick="editPrompt()">✎ Edit headword</button>
   </div>
   <div><input type="text" id="note" placeholder="optional note"> </div>`;
}
async function decide(action,payload={}){
  if(!cur)return; const note=document.getElementById('note'); if(note&&note.value)payload.note=note.value;
  await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({entry_id:cur.entry_id,action,payload})});
  await load(); show(cur.entry_id);
}
function splitPrompt(){const at=prompt("Split: paste the headword text where the next entry begins (or char offset):");if(at)decide('split',{at});}
function editPrompt(){const hw=prompt("Corrected headword:",cur.headword_disp);if(hw)decide('edit',{headword:hw});}
load();
</script></body></html>"""

if __name__ == "__main__":
    print(f"GOTW review UI on http://127.0.0.1:{ARGS.port}  (db={ARGS.db})")
    app.run(port=ARGS.port, debug=False)
