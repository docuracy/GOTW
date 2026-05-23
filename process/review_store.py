"""Sidecar store for human review decisions.

Decisions are keyed by a STABLE signature (source + normalised headword + page), NOT by entry_id,
so they survive re-parses and DB rebuilds — the `gotw_seg.sqlite` entry table is rebuilt repeatedly
during QA, and an scp of a fresh DB would otherwise wipe an in-DB review table. This sidecar lives in
its own file (`data/review.sqlite`) and is re-attached to entries by signature on each load.
"""
import sqlite3, hashlib, re

DEFAULT_STORE = "data/review.sqlite"


def _alpha(s):
    return re.sub(r"[^A-Z]", "", (s or "").upper())


def sig(source, headword_raw, page_start):
    """Stable id for an entry across re-parses (raw headword may keep an OCR spelling we override)."""
    return hashlib.sha1(f"{source}|{_alpha(headword_raw)}|{page_start}".encode()).hexdigest()[:16]


def open_store(path=DEFAULT_STORE):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE IF NOT EXISTS decisions("
              "sig TEXT PRIMARY KEY, headword TEXT, action TEXT, payload TEXT, ts TEXT)")
    return c
