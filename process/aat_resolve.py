#!/usr/bin/env python3
"""Resolve AAT concept ids for a set of place-type labels from the local dump.

Two-hop SKOS-XL: concept --prefLabel--> term --literalForm--> "label"@en.
Also reads broaderPreferred/broaderGeneric so we can show an ancestor path
(for disambiguating e.g. "capes (landforms)" vs "capes (clothing)").

    python3 process/aat_resolve.py "towns" "villages" "rivers (streams)"

With no args, resolves the built-in DRAFT_TERMS shortlist and prints a table.
Requires the dump at ~/Documents/GitHub/whg3/data/aat/AATOut_*.nt
"""
from __future__ import annotations
import os, re, sys, pickle
from pathlib import Path

AAT_DIR = Path(os.path.expanduser("~/Documents/GitHub/whg3/data/aat"))
TERMS = AAT_DIR / "AATOut_2Terms.nt"
RELS = AAT_DIR / "AATOut_HierarchicalRels.nt"
CACHE = Path("data/aat_index.pkl")

PREF = "http://www.w3.org/2008/05/skos-xl#prefLabel"
LIT = "http://www.w3.org/2008/05/skos-xl#literalForm"
BPREF = "http://vocab.getty.edu/ontology#broaderPreferred"
BGEN = "http://vocab.getty.edu/ontology#broaderGeneric"
AAT = "http://vocab.getty.edu/aat/"

TRIPLE = re.compile(r"^<([^>]+)>\s+<([^>]+)>\s+(.+?)\s*\.\s*$")
LITERAL = re.compile(r'^"((?:[^"\\]|\\.)*)"(?:@(\w+))?')


def _id(uri: str) -> str | None:
    return uri[len(AAT):] if uri.startswith(AAT) else None


def build_index():
    """Return (label2ids, id2label, broader). Cached to data/aat_index.pkl."""
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    if not TERMS.exists():
        sys.exit(f"AAT dump not found at {AAT_DIR}")
    concept_term, term_lit = {}, {}      # concept->termURI ; termURI->label
    with TERMS.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TRIPLE.match(line)
            if not m:
                continue
            s, p, o = m.groups()
            # Concepts carry a prefLabel per language; the English preferred
            # term URI ends in "-en". Selecting it avoids keeping a non-English
            # term whose @en literal we never stored (which silently drops it).
            if p == PREF and o.endswith("-en>"):
                cid = _id(s)
                if cid:
                    concept_term[cid] = o.strip("<>")
            elif p == LIT:
                lm = LITERAL.match(o)
                if lm and (lm.group(2) or "en") == "en":
                    term_lit[s] = lm.group(1)
    id2label, label2ids = {}, {}
    for cid, turi in concept_term.items():
        lab = term_lit.get(turi)
        if lab:
            id2label[cid] = lab
            label2ids.setdefault(lab.lower(), []).append(cid)
    broader = {}
    with RELS.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TRIPLE.match(line)
            if not m:
                continue
            s, p, o = m.groups()
            if p in (BPREF, BGEN) and o.startswith("<"):
                c, par = _id(s), _id(o.strip("<>"))
                if c and par:
                    broader.setdefault(c, []).append(par)
    idx = (label2ids, id2label, broader)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(idx))
    return idx


def path(cid, id2label, broader, depth=6):
    out, seen = [], set()
    while cid and cid not in seen and depth:
        seen.add(cid)
        out.append(id2label.get(cid, cid))
        parents = broader.get(cid)
        cid = parents[0] if parents else None
        depth -= 1
    return " < ".join(out)


# Draft shortlist: gazetteer descriptor -> candidate AAT preferred label(s)
DRAFT_TERMS = [
    "inhabited places", "cities", "towns", "villages", "hamlets", "market towns",
    "boroughs", "capitals (seats of government)", "ports (settlements)",
    "parishes", "townships", "communes", "cantons", "districts", "departments",
    "counties", "provinces", "arrondissements", "sovereign states", "countries",
    "rivers (streams)", "streams", "lakes", "bays (bodies of water)", "gulfs",
    "straits", "creeks (streams)", "inlets", "harbors", "channels (waterways)",
    "islands", "islets", "archipelagos", "capes (landforms)", "promontories",
    "peninsulas", "mountains", "hills", "summits (landforms)",
    "mountain ranges", "valleys", "volcanoes", "plains", "coasts",
    "fortresses", "forts", "ethnic groups",
]


def main():
    label2ids, id2label, broader = build_index()
    queries = sys.argv[1:] or DRAFT_TERMS
    print(f"{'query':32} {'aat id':>10}  preferred label / ancestry")
    print("-" * 100)
    for q in queries:
        ids = label2ids.get(q.lower(), [])
        if not ids:
            # fuzzy: any preferred label containing the query head word
            head = q.split()[0].rstrip("s")
            ids = sorted({c for lab, cs in label2ids.items()
                          if lab.startswith(head) for c in cs})[:3]
            tag = " (fuzzy)" if ids else ""
        else:
            tag = ""
        if not ids:
            print(f"{q:32} {'—':>10}  NOT FOUND")
            continue
        for cid in ids[:3]:
            print(f"{q+tag:32} {cid:>10}  {path(cid, id2label, broader)}")


if __name__ == "__main__":
    main()
