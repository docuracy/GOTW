"""Local browser UI to review/resolve suspect gazetteer entries flagged by flag_suspects.py.

Read-mostly over the parsed `entry` table + `qa` flags in gotw_seg.sqlite. Decisions are kept in a
SEPARATE sig-keyed sidecar (data/review.sqlite via review_store) so they survive re-parses / DB
rebuilds, and are re-attached to entries by signature. The OCR / entry text is never mutated here;
keep/reject/table/people/merge/split/edit are recorded for the eventual re-extract to apply.

    .venv/bin/python process/review_ui.py                 # http://127.0.0.1:5000
"""
import sqlite3, json, argparse, datetime, sys
from flask import Flask, request, jsonify, Response
sys.path.insert(0, "process")
import review_store as RS

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/gotw_seg.sqlite")
ap.add_argument("--store", default=RS.DEFAULT_STORE)
ap.add_argument("--port", type=int, default=5000)
ARGS = ap.parse_args()

app = Flask(__name__)


def db():
    c = sqlite3.connect(ARGS.db)
    c.row_factory = sqlite3.Row
    return c


def store():
    return RS.open_store(ARGS.store)


def has_qa(c):
    return c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='qa'").fetchone()


def decisions_map():
    s = store()
    m = {row[0]: {"action": row[1], "payload": row[2]} for row in
         s.execute("SELECT sig, action, payload FROM decisions")}
    s.close()
    return m


@app.route("/api/list")
def api_list():
    verdict = request.args.get("verdict", "review")
    decided = request.args.get("decided", "undecided")
    c, qa = db(), None
    qa = has_qa(c)
    sel = ("e.entry_id, e.headword_disp, e.headword_raw, s.filename vol, e.page_start, "
           "length(e.text) tlen")
    join, where, params = "", ["e.kind='entry'"], []
    if qa:
        sel += ", q.flags, q.idx_sim, q.idx_hit, q.verdict"
        join = "JOIN qa q ON q.entry_id=e.entry_id"
        if verdict != "all":
            where.append("q.verdict=?")
            params.append(verdict)
    sql = (f"SELECT {sel} FROM entry e JOIN source s ON e.source_id=s.source_id {join} "
           f"WHERE " + " AND ".join(where))
    dm = decisions_map()
    rows = []
    for x in c.execute(sql, params):
        x = dict(x)
        x["sig"] = RS.sig(x["vol"], x["headword_raw"], x["page_start"])
        d = dm.get(x["sig"])
        x["decision"] = d["action"] if d else None
        rows.append(x)
    if decided == "undecided":
        rows = [x for x in rows if not x["decision"]]
    elif decided == "decided":
        rows = [x for x in rows if x["decision"]]
    rows.sort(key=lambda x: (x.get("idx_sim") if x.get("idx_sim") is not None else -1, x["tlen"]))
    return jsonify(total=len(rows), rows=rows[:2000])


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
        r = c.execute(f"SELECT entry_id, headword_disp, substr(text,1,200) snip FROM entry "
                      f"WHERE source_id=? AND seq {op} ? ORDER BY seq {order} LIMIT 1",
                      (e["source_id"], e["seq"])).fetchone()
        nbr[label] = dict(r) if r else None
    if has_qa(c):
        q = c.execute("SELECT flags, idx_sim, idx_hit, verdict FROM qa WHERE entry_id=?", (eid,)).fetchone()
        e["qa"] = dict(q) if q else None
    e["sig"] = RS.sig(e["vol"], e["headword_raw"], e["page_start"])
    d = decisions_map().get(e["sig"])
    e["decision"] = d
    e["neighbors"] = nbr
    return jsonify(e)


@app.route("/api/decide", methods=["POST"])
def api_decide():
    d = request.get_json(force=True)
    c = db()
    e = c.execute("SELECT e.headword_raw, e.page_start, s.filename vol FROM entry e "
                  "JOIN source s ON e.source_id=s.source_id WHERE e.entry_id=?", (d["entry_id"],)).fetchone()
    if not e:
        return jsonify(error="not found"), 404
    sg = RS.sig(e["vol"], e["headword_raw"], e["page_start"])
    s = store()
    if d["action"] == "undo":
        s.execute("DELETE FROM decisions WHERE sig=?", (sg,))
    else:
        s.execute("INSERT OR REPLACE INTO decisions(sig, headword, action, payload, ts) VALUES(?,?,?,?,?)",
                  (sg, e["headword_raw"], d["action"], json.dumps(d.get("payload", {})),
                   datetime.datetime.now().isoformat(timespec="seconds")))
    s.commit()
    s.close()
    return jsonify(ok=True, sig=sg)


@app.route("/api/stats")
def api_stats():
    s = store()
    out = {"reviewed": s.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
           "by_action": dict(s.execute("SELECT action,COUNT(*) FROM decisions GROUP BY action").fetchall())}
    s.close()
    c = db()
    if has_qa(c):
        out["by_verdict"] = dict(c.execute("SELECT verdict,COUNT(*) FROM qa GROUP BY verdict").fetchall())
    return jsonify(out)


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>GOTW segmentation review</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;display:flex;height:100vh}
 #list{width:330px;border-right:1px solid #ccc;overflow:auto;background:#fafafa}
 #list .it{padding:6px 10px;border-bottom:1px solid #eee;cursor:pointer}
 #list .it:hover{background:#eef} #list .it.done{opacity:.4} #list .it.cur{background:#dde7ff}
 #list .hw{font-weight:600} #list .meta{color:#888;font-size:12px}
 #detail{flex:1;padding:16px 22px;overflow:auto}
 .tag{display:inline-block;background:#fee;color:#900;border-radius:3px;padding:1px 6px;margin:0 4px 0 0;font-size:12px}
 .nbr{background:#f4f4f4;border-left:3px solid #ccc;padding:6px 10px;margin:6px 0;color:#444;font-size:13px}
 .txt{white-space:pre-wrap;background:#fff;border:1px solid #ddd;padding:10px;max-height:38vh;overflow:auto}
 textarea{width:100%;height:18vh;font:13px ui-monospace,monospace}
 button{font:13px system-ui;padding:6px 11px;margin:3px 2px;border:1px solid #aaa;border-radius:5px;background:#fff;cursor:pointer}
 button.act{background:#eef} button.danger{background:#fee}
 #toolbar{padding:8px 10px;border-bottom:1px solid #ccc;background:#f0f0f0;position:sticky;top:0}
 .score{font-weight:600} kbd{background:#eee;border:1px solid #ccc;border-radius:3px;padding:0 4px}
</style></head><body>
<div id="list"><div id="toolbar">
  <select id="fverdict" onchange="load()"><option value="review">needs review</option>
   <option value="likely-variant">likely-variant</option><option value="ok-in-index">ok-in-index</option>
   <option value="all">all flagged</option></select>
  <select id="fdecided" onchange="load()"><option value="undecided">undecided</option>
   <option value="decided">decided</option><option value="all">all</option></select>
  <span id="count"></span></div><div id="items"></div></div>
<div id="detail">Select an entry… &nbsp; <small>keys: <kbd>k</kbd>eep <kbd>r</kbd>eject <kbd>j</kbd>↓ <kbd>f</kbd>↑</small></div>
<script>
let rows=[], curIdx=-1;
async function load(){
  const r=await(await fetch(`/api/list?verdict=${fverdict.value}&decided=${fdecided.value}`)).json();
  rows=r.rows; count.textContent=r.total+" entries";
  items.innerHTML=rows.map((x,i)=>`<div class="it ${x.decision?'done':''}" id="it${i}" onclick="show(${i})">
    <div class="hw">${x.headword_disp||''} ${x.decision?'· '+x.decision:''}</div>
    <div class="meta">${x.vol.slice(5,7)} p${x.page_start??'?'} · ${x.tlen} ch · idx ${x.idx_sim??'–'} · ${x.flags||''}</div></div>`).join('');
  if(rows.length){curIdx=-1; next();}
}
async function show(i){
  if(i<0||i>=rows.length)return; curIdx=i;
  document.querySelectorAll('#list .it').forEach(el=>el.classList.remove('cur'));
  const el=document.getElementById('it'+i); if(el){el.classList.add('cur'); el.scrollIntoView({block:'nearest'});}
  const e=await(await fetch(`/api/entry/${rows[i].entry_id}`)).json();
  const q=e.qa||{}, n=e.neighbors||{}, dec=e.decision;
  const ov=(dec&&dec.action=='edit'&&dec.payload)?(JSON.parse(dec.payload).text||''):'';
  detail.innerHTML=`<h2>${e.headword_disp} <small style="color:#888">[${e.headword_raw}]</small></h2>
   <div>${e.vol.slice(5,7)} · page ${e.page_start??'?'} · ${e.text.length} chars
     · <span class="score">index ${q.idx_sim??'–'}</span> ${q.idx_hit?('→ '+q.idx_hit):''}
     ${dec?`· <b style="color:#070">decided: ${dec.action}</b>`:''}</div>
   <div style="margin:6px 0">${(q.flags||'').split(',').filter(Boolean).map(f=>`<span class="tag">${f}</span>`).join('')}</div>
   ${n.prev?`<div class="nbr"><b>↑ prev:</b> ${n.prev.headword_disp} — ${esc(n.prev.snip)}…</div>`:''}
   <div class="txt">${esc(e.text).slice(0,9000)}${e.text.length>9000?' …[truncated]':''}</div>
   ${n.next?`<div class="nbr"><b>↓ next:</b> ${n.next.headword_disp} — ${esc(n.next.snip)}…</div>`:''}
   <div style="margin:10px 0">
     <button class="act" onclick="decide('keep')">✓ Keep</button>
     <button class="danger" onclick="decide('reject')">✗ Reject</button>
     <button class="act" onclick="decide('table')">▦ Table</button>
     <button class="act" onclick="decide('people')">◐ People</button>
     <button class="act" onclick="decide('merge_prev')">⇧ Merge ↑prev</button>
     <button class="act" onclick="decide('merge_next',{},true)">⇩ Merge ↓next</button>
     <button class="act" onclick="editHead()">✎ Headword</button>
     ${dec?`<button onclick="decide('undo')">↺ Undo</button>`:''}
   </div>
   <div><b>Edit text</b> (correct garbled OCR; saved as an override, source text untouched):<br>
     <textarea id="edtext">${esc(ov||e.text)}</textarea>
     <button class="act" onclick="saveText()">Save edited text</button></div>`;
}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
async function decide(action,payload={},stay){
  if(curIdx<0)return;
  await post({entry_id:rows[curIdx].entry_id,action,payload});
  rows[curIdx].decision=(action=='undo')?null:action;
  const el=document.getElementById('it'+curIdx);
  if(el){el.classList.toggle('done', action!='undo'); el.querySelector('.hw').innerHTML=`${rows[curIdx].headword_disp} ${action!='undo'?'· '+action:''}`;}
  if(action=='undo'||stay){show(curIdx);} else {next();}   // auto-advance on terminal decisions
}
function next(){let i=curIdx+1; while(i<rows.length&&rows[i].decision)i++; if(i<rows.length)show(i); else detail.innerHTML='<h3>✓ No more undecided entries in this list.</h3>';}
function prev(){let i=curIdx-1; while(i>=0&&rows[i].decision)i--; if(i>=0)show(i);}
async function saveText(){const t=document.getElementById('edtext').value; await post({entry_id:rows[curIdx].entry_id,action:'edit',payload:{text:t}}); rows[curIdx].decision='edit'; show(curIdx);}
function editHead(){const hw=prompt('Corrected headword:',rows[curIdx].headword_disp); if(hw)post({entry_id:rows[curIdx].entry_id,action:'edit',payload:{headword:hw}}).then(()=>{rows[curIdx].decision='edit';show(curIdx);});}
async function post(b){return fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
document.addEventListener('keydown',e=>{if(e.target.tagName=='TEXTAREA'||e.target.tagName=='INPUT')return;
  if(e.key=='k')decide('keep'); else if(e.key=='r')decide('reject'); else if(e.key=='j')next(); else if(e.key=='f')prev();});
load();
</script></body></html>"""

if __name__ == "__main__":
    print(f"GOTW review UI on http://127.0.0.1:{ARGS.port}  (db={ARGS.db}, store={ARGS.store})")
    app.run(port=ARGS.port, debug=False)
