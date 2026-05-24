#!/usr/bin/env python3
"""Precompute Symphonym phonetic embeddings for the gazetteer's headwords -> int8 matrix for in-browser
cosine-KNN ("phonetic" search mode). Reads the SAME doc set as the FTS search DB, so a phonetic hit
carries the reader locator (vol, page, rc, eid) and opens the reader exactly like the other modes.

Embeddings are computed with the fp32 reference model (exact), then stored int8 (components of an
L2-normalised vector are in [-1,1] -> scale 127). The browser embeds the query with the int8 ONNX
encoder and ranks by dot(query_fp32, corpus_int8) (scale is irrelevant to ranking).

  python3 process/build_symphonym_index.py --sym /home/stephen/PycharmProjects/indexing/hf
"""
from __future__ import annotations
import argparse, json, sqlite3, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="docs/search/gotw-fts.sqlite.png", help="FTS DB (doc table) for headwords+locator")
    ap.add_argument("--sym", default="/home/stephen/PycharmProjects/indexing/hf")
    ap.add_argument("--out-dir", default="docs/search")
    ap.add_argument("--lang", default="und", help="corpus language conditioning (gazetteer mixes many)")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    sys.path.insert(0, args.sym)
    from inference import SymphonymModel
    sm = SymphonymModel(model_dir=args.sym)

    con = sqlite3.connect(args.db)
    rows = con.execute("SELECT eid, vol, page, rc, headword FROM doc ORDER BY eid").fetchall()
    hws = [r[4] for r in rows]
    print(f"embedding {len(hws)} headwords (lang={args.lang!r}) …")
    embs = []
    for i in range(0, len(hws), args.batch):
        embs.append(sm.batch_embed([(h or "", args.lang) for h in hws[i:i + args.batch]]))
        if i % (args.batch * 40) == 0:
            print(f"  {i}/{len(hws)}", flush=True)
    E = np.vstack(embs).astype(np.float32)                       # (N,128), L2-normalised
    q = np.clip(np.round(E * 127), -127, 127).astype(np.int8)     # int8 store (scale 127)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "symphonym-embeddings.i8").write_bytes(q.tobytes())
    meta = {"hw": [r[4] for r in rows], "vol": [r[1] for r in rows],
            "page": [r[2] for r in rows], "rc": [r[3] for r in rows], "eid": [r[0] for r in rows]}
    (out / "symphonym-meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    (out / "symphonym-manifest.json").write_text(json.dumps(
        {"n": len(rows), "dim": 128, "scale": 127, "lang_corpus": args.lang}))
    print(f"wrote {len(rows)}×128 int8 -> symphonym-embeddings.i8 ({q.nbytes/1e6:.1f} MB) + meta/manifest")

    # --- KNN sanity: fp32 query vs dequantised int8 corpus (typos + cross-script) ---
    Qd = q.astype(np.float32) / 127.0
    print("\nquery -> top phonetic matches:")
    for query in ["Constantinopel", "Edinburg", "Bordaux", "Moskva", "Munich", "Florense"]:
        v = sm.embed(query, "und").astype(np.float32)
        top = np.argsort(-(Qd @ v))[:5]
        print(f"  {query:16} -> " + ", ".join(f"{meta['hw'][j]}({Qd[j]@v:.2f})" for j in top))


if __name__ == "__main__":
    main()
