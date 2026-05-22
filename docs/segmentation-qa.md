# Segmentation QA & review workflow

How we validated the OCR→entry segmentation and how to review the residual hard cases. This sits
*before* the LLM re-extraction: we nail segmentation first, then extract.

## What changed in the parser (`process/parse_ocr.py`)

The gazetteer prints entries in **two typographies**, and the parser now handles both:

- **Inline minor entries** — `HEADWORD, descriptor …` (always handled).
- **Standalone display headings** for major multi-page entries (countries, continents, empires) —
  the headword alone on its line ending in `.` or `,` (`AF'GHANISTAN.`, `AMERICA,`), description as the
  next paragraph. Detected by **alphabetical continuity** (must sort after the current headword, so the
  repeating page running-heads are skipped) + a **sentence-boundary** guard (a real heading follows a
  finished entry, not a clause interrupted mid-page).

Rejected as non-headwords: compass bearings, roman numerals, numbered/lettered **section headings**
(incl. period-less `VIII KINGDOM OF KASAN`, via a *strict* roman validator so `MILL HILL`/`DILI` survive),
scan/library stamps (Google watermark, `UNIVERSITY OF MINNESOTA`, the ship's-library stamps in this copy).
The v7 **Appendix** (ancient↔modern concordance + Ethnology essay) is excluded — it stops at `APPENDIX.`
and is handled separately (`extract_appendix.py` → `name_variant`); it must never hit the place LLM.

## Validation (`process/crosscheck_headwords.py`)

Cross-checks our headwords against Humphrey Southall's transcripts (QA only — counts/headwords are facts,
his prose is **not** ingested). Parentheticals captured on both sides; fuzzy bipartite matching cancels OCR
spelling variants; a his-only headword is a genuine **merge** only if it appears in our OCR as an entry-start
(`NAME, <lowercase>`), not a mere prose mention.

**Result:** ~97–98 % headword agreement across all 7 volumes, coverage on par. Genuine residual:
**~35 merges + ~231 over-split candidates**, the latter inflated by real short places (`ACRE`/`AGEN`/`BALI`).

## Suspect flagging + index reasonableness-check (`process/flag_suspects.py`)

Run in the **whg** conda env on a CRC compute node (gateway reachable, no token):

    python3 process/flag_suspects.py data/gotw_seg.sqlite

Flags suspect entries (short/no-vowel headword, very short text, >40 k blob) and checks each loosely
against the WHG gateway toponyms index: **exact match ⇒ `ok-in-index`; else string-similarity to the best
fuzzy hit ⇒ `likely-variant` (≥0.88) or `review`**. (The gateway `score` is a rank, not query similarity,
so similarity is computed here.) Writes a `qa` table — **OCR text is never altered**, only routed.

## Local review UI (`process/review_ui.py`)

    .venv/bin/python process/review_ui.py --db data/gotw_seg.sqlite     # http://127.0.0.1:5000

Work-list sorted worst-first (lowest index similarity), filterable by verdict / decided. Shows the entry
text, its alphabetical neighbours, flags and index hit. Records decisions — **keep / reject / table /
people / merge-into-prev / split / edit** — into a `review` table (entry text untouched). This same
work-list is the input to the future Django QA module.

## Data

- `data/gotw_seg.sqlite` — the **new** segmentation (92,068 entries, 7 vols) + `qa` flags + `review` decisions.
- `data/gotw.sqlite` — the pre-fix DB (to be superseded by the re-extract built on `gotw_seg`).

## Next

1. Work the review UI through the `review`-verdict suspects (a few hundred).
2. Apply decisions, then run the full LLM re-extract on the clean entries (`process/submit_extract_slurm.py`).
3. Reconcile (`process/submit_reconcile_slurm.py`).
