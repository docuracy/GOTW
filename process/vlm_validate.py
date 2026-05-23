"""Production VLM heading-validation pass over a volume's PROSE pages.

Per prose page: Qwen2.5-VL lists the entry headings (per-column), fuzzy-diffed against our parsed
headwords for that printed page. Records merge candidates (VLM sees, we lack) and spurious candidates
(we have, VLM lacks). SKIPS table-bbox / no-printed-page / sparse pages (inconclusive — see the
reliability finding). Shardable by image-index range; resumable via a per-shard JSONL (one line/page).
Ingest the JSONLs into the `vlm_qa` table with --ingest.

  TABLE_VL_BASE=http://127.0.0.1:PORT/v1 python3 process/vlm_validate.py --db data/gotw_seg.sqlite \
      --vol v3 --img-dir img/v3 --ocr txt/gotw-v3-ocr.txt --start 0 --end 9999 --out-jsonl vlm_jsonl/v3.0.jsonl
  python3 process/vlm_validate.py --db data/gotw_seg.sqlite --ingest 'vlm_jsonl/v3.*.jsonl'
"""
import argparse, re, json, glob, difflib, sqlite3, datetime, sys
from pathlib import Path
sys.path.insert(0, "process")
import vlm_headings as V

VLM_QA_DDL = """CREATE TABLE IF NOT EXISTS vlm_qa(
  id INTEGER PRIMARY KEY, volume TEXT, page INTEGER, kind TEXT,   -- 'merge' | 'spurious'
  name TEXT, detail TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS vlm_pages(
  volume TEXT, page INTEGER, n_vlm INTEGER, n_merge INTEGER, n_spurious INTEGER,
  PRIMARY KEY (volume, page));"""


def our_headwords(con, vol):
    """page_start -> [display headwords] for a volume (entries with a known printed page)."""
    by = {}
    for hw, pg in con.execute(
            "SELECT headword_disp, page_start FROM entry e JOIN source s ON e.source_id=s.source_id "
            "WHERE s.filename=? AND e.kind IN('entry','crossref') AND page_start IS NOT NULL", (f"gotw-{vol}-ocr.txt",)):
        if V.alpha(hw):
            by.setdefault(pg, []).append(hw)
    return by


def keys_of(name):
    """Match keys for a heading: per variant form ('X, or Y'), the alpha AND the token-sorted alpha
    — so 'Lake George' == 'George (Lake)' and 'San Giacomo' == 'Giacomo (San)' (gazetteer inverts)."""
    ks = set()
    for part in re.split(r"\s*,?\s+or\s+|\s*;\s*", name or "", flags=re.I):
        words = [w.upper() for w in re.findall(r"[A-Za-zÀ-ÿ]+", part)]
        a = re.sub(r"[^A-Z]", "", "".join(words))
        if len(a) >= 3:
            ks.add(a)
            ks.add(re.sub(r"[^A-Z]", "", "".join(sorted(words))))
    return ks


def prose_pages(ocr_path, ourby):
    """(img_idx, printed_page) for prose pages: digit printed page, no table bbox, dense, real entries."""
    out = []
    for b in Path(ocr_path).read_text(encoding="utf-8").split("\f"):
        L = [x for x in b.splitlines() if x]
        if not L:
            continue
        m = re.search(r"## p\. (\d+) \(#(\d+)\)", L[0])
        if not m or any(x.startswith("<!-- table") for x in L):
            continue
        body = [x for x in L if not x.startswith(("##", "<!--"))]
        pg, idx = int(m.group(1)), int(m.group(2)) - 1
        if len(body) >= 30 and len(ourby.get(pg, set())) >= 10:
            out.append((idx, pg))
    return out


def diff(vlm_names, ours_names, cutoff=0.86):
    """Name-level set-diff: match on token-sorted/variant keys (paren + word-order invariant), with a
    fuzzy fallback for OCR-spelling variants. Returns (vlm_only=merge candidates, ours_only=spurious)."""
    okeys = set().union(*(keys_of(o) for o in ours_names)) if ours_names else set()
    vkeys = set().union(*(keys_of(v) for v in vlm_names)) if vlm_names else set()
    oalpha = [V.alpha(o) for o in ours_names]
    valpha = [V.alpha(v) for v in vlm_names]
    vlm_only = [v for v in vlm_names if not (keys_of(v) & okeys)
                and not difflib.get_close_matches(V.alpha(v), oalpha, 1, cutoff)]
    ours_only = [o for o in ours_names if not (keys_of(o) & vkeys)
                 and not difflib.get_close_matches(V.alpha(o), valpha, 1, cutoff)]
    return vlm_only, ours_only


def do_ingest(con, pattern):
    con.executescript(VLM_QA_DDL)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    n = 0
    for f in glob.glob(pattern):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            vol, pg = r["vol"], r["page"]
            con.execute("DELETE FROM vlm_qa WHERE volume=? AND page=?", (vol, pg))
            con.execute("INSERT OR REPLACE INTO vlm_pages VALUES(?,?,?,?,?)",
                        (vol, pg, r["n_vlm"], len(r["merges"]), len(r["spurious"])))
            for m in r["merges"]:
                con.execute("INSERT INTO vlm_qa(volume,page,kind,name,detail,created_at) VALUES(?,?,'merge',?,?,?)",
                            (vol, pg, m["name"], json.dumps({"variants": m.get("variants", []), "see": m.get("see")}), now))
                n += 1
            for s in r["spurious"]:
                con.execute("INSERT INTO vlm_qa(volume,page,kind,name,detail,created_at) VALUES(?,?,'spurious',?,?,?)",
                            (vol, pg, s, "{}", now))
                n += 1
    con.commit()
    print(f"ingested {n} vlm_qa candidates from {pattern}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw_seg.sqlite")
    ap.add_argument("--vol")
    ap.add_argument("--img-dir")
    ap.add_argument("--ocr")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--out-jsonl")
    ap.add_argument("--ingest", help="glob of shard JSONLs to load into vlm_qa")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    if args.ingest:
        do_ingest(con, args.ingest)
        return

    ourby = our_headwords(con, args.vol)
    pages = [(i, p) for i, p in prose_pages(args.ocr, ourby) if args.start <= i <= args.end]
    files = sorted(p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in V.IMG_EXTS)
    done = set()
    if args.out_jsonl and Path(args.out_jsonl).exists():
        done = {json.loads(l)["idx"] for l in open(args.out_jsonl) if l.strip()}
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    jf = open(args.out_jsonl, "a", encoding="utf-8")
    print(f"{args.vol}: {len(pages)} prose pages in [{args.start},{args.end}], {len(done)} already done")
    for idx, pg in pages:
        if idx in done:
            continue
        heads = []
        for crop in V.column_crops(files[idx].read_bytes()):
            heads += V.vlm_headings(crop)
        seen, uniq = set(), []
        for h in heads:
            k = V.alpha(h.get("name", ""))
            if k and k not in seen:
                seen.add(k)
                uniq.append({"name": h["name"], "k": k, "variants": h.get("variants", []), "see": h.get("see")})
        vlm_names = [h["name"] for h in uniq]
        ours_names = ourby.get(pg, [])
        vlm_only, ours_only = diff(vlm_names, ours_names)
        voset = set(vlm_only)
        merges = [{"name": h["name"], "variants": h["variants"], "see": h["see"]} for h in uniq if h["name"] in voset]
        jf.write(json.dumps({"vol": args.vol, "idx": idx, "page": pg, "n_vlm": len(uniq),
                             "n_ours": len(ours_names), "vlm": vlm_names, "merges": merges,
                             "spurious": ours_only}, ensure_ascii=False) + "\n")
        jf.flush()
    jf.close()
    print("done")


if __name__ == "__main__":
    main()
