#!/usr/bin/env python3
"""Parse and analyse the statistical tables captured in the HTML transcript.

The parser attached each entry's `<table>`s to it (`entry.n_tables`); here we turn
that markup into structured rows in a `table_data` store and classify each table by
subject (climate, population/area, agriculture, …). This is deterministic — no API,
no PDF — and gives an immediate read on the ~140 tables already in hand. (The v5 PDF
now lets us recover the tables Southall's transcription omitted; see process/extract_tables.py.)

    python3 process/parse_tables.py [--db data/gotw.sqlite]
"""
from __future__ import annotations
import argparse, json, re, sqlite3
from collections import Counter
from bs4 import BeautifulSoup

DDL = """
CREATE TABLE IF NOT EXISTS table_data (
  id INTEGER PRIMARY KEY,
  entry_id INTEGER REFERENCES entry(entry_id),
  table_no INTEGER, headword TEXT, page_start INTEGER,
  n_rows INTEGER, n_cols INTEGER, subject TEXT,
  header TEXT,        -- JSON: first row (column labels)
  rows TEXT,          -- JSON: list of rows (each a list of cells)
  created_at TEXT
);
"""

SUBJECTS = [                                   # (label, header/keyword regex)
    ("climate",       r"temp|barom|therm|rain|fahr|mean height"),
    ("population",    r"\bpop|inhabit|census|souls"),
    ("area/revenue",  r"\barea\b|sq\.? m|acres|revenue|assess|districts"),
    ("agriculture",   r"wheat|barley|oats|maize|crop|produce|cattle|acres of"),
    ("trade",         r"export|import|tonnage|vessels|trade|customs|duties"),
]


def cells(tr):
    return [td.get_text(" ", strip=True).replace("\xa0", "").strip()
            for td in tr.find_all(["td", "th"])]


def classify(header, rows):
    blob = " ".join(header + [c for r in rows[:3] for c in r]).lower()
    for label, rx in SUBJECTS:
        if re.search(rx, blob):
            return label
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw.sqlite")
    args = ap.parse_args()
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row
    con.executescript(DDL)
    con.execute("DELETE FROM table_data")

    n_tables = 0
    subj = Counter()
    dims = []
    for e in con.execute("SELECT entry_id, headword_disp, page_start, raw_html "
                         "FROM entry WHERE n_tables>0"):
        soup = BeautifulSoup(e["raw_html"], "lxml")
        for ti, tbl in enumerate(soup.find_all("table"), 1):
            rows = [c for tr in tbl.find_all("tr") if (c := cells(tr))]
            if not rows:
                continue
            header = rows[0]
            body = rows[1:]
            subject = classify(header, body)
            ncols = max(len(r) for r in rows)
            con.execute(
                "INSERT INTO table_data(entry_id,table_no,headword,page_start,n_rows,n_cols,"
                "subject,header,rows,created_at) VALUES(?,?,?,?,?,?,?,?,?,datetime('now'))",
                (e["entry_id"], ti, e["headword_disp"], e["page_start"], len(rows), ncols,
                 subject, json.dumps(header), json.dumps(body)))
            n_tables += 1
            subj[subject] += 1
            dims.append((len(rows), ncols))
    con.commit()

    print(f"parsed {n_tables} tables -> table_data")
    print("\nby subject:")
    for s, c in subj.most_common():
        print(f"  {s:14} {c}")
    if dims:
        rws = sorted(r for r, _ in dims)
        print(f"\ndimensions: rows median {rws[len(rws)//2]}, max {max(rws)}; "
              f"cols up to {max(c for _, c in dims)}")
    print("\nlargest tables:")
    for r in con.execute("SELECT headword,page_start,n_rows,n_cols,subject FROM table_data "
                         "ORDER BY n_rows*n_cols DESC LIMIT 6"):
        print(f"  {r['headword'][:28]:28} p.{r['page_start']:>3}  {r['n_rows']}×{r['n_cols']}  {r['subject']}")
    # show one parsed table end-to-end
    ex = con.execute("SELECT headword,header,rows FROM table_data WHERE subject='climate' LIMIT 1").fetchone()
    if ex:
        print(f"\nexample — {ex['headword']} (climate):")
        print("  header:", json.loads(ex['header']))
        for row in json.loads(ex['rows'])[:3]:
            print("  row:   ", row)


if __name__ == "__main__":
    main()
