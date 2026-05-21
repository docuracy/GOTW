# Gazetteer of the World → Linked Data

Transforming *A Gazetteer of the World* (c. 1856,
[archive.org/details/agazetteerworld00unkngoog](https://archive.org/details/agazetteerworld00unkngoog))
into structured linked data as an **authority gazetteer** for the
[World Historical Gazetteer](https://whgazetteer.org/) (WHG). Places are typed
with [Getty AAT](https://www.getty.edu/research/tools/vocabularies/aat/) concepts
and geolocated against WHG's indices via its
[Reconciliation API](https://docs.whgazetteer.org/content/technical/apis.html#reconciliation-service-api).

## Sources

| Asset | Notes |
|-------|-------|
| `data/pdf/agazetteerworld00unkngoog.pdf` | 919-page scanned book. Every page is a raster scan (no vector layer); maps are baked into the page images. |
| `data/html/gotw_vol5_all_web_v1.html` | OCR-derived transcript, 1 of **7** files covering the whole work. Volume 5 here, printed pages 1–874. |
| `data/gotw.sqlite` | Working store, generated from the HTML (see below). Not committed; rebuild with the parser. |

The HTML transcription was produced under **Prof. Humphrey Southall** (Director,
GB Historical GIS, University of Portsmouth) — see [Credits](#credits).

## Pipeline

```
HTML transcript ──parse──▶ SQLite (entry) ──LLM extract──▶ SQLite (place)
                                                  │
                                                  ├─ AAT feature typing
                                                  ├─ multi-place splitting ("—Also …")
                                                  ├─ cross-reference resolution ("See …")
                                                  └─ geographic context for disambiguation
                                                            │
                                              WHG Reconciliation API ──▶ coordinates
                                                            │
                                              MapLibre demo (GitHub Pages)
```

### 1. Parsing — `process/parse_html.py` *(done)*

```bash
python3 process/parse_html.py data/html/gotw_vol5_all_web_v1.html --db data/gotw.sqlite
```

The transcript is one long flow of `<p>` elements. The parser handles the
structure that makes naive splitting fail:

- **Page markers `[[N]]`** appear inside `<br/>` runs and can fall *mid-paragraph*,
  gluing the tail of one place onto a new headword. We split on them and record
  page provenance (`page_start`/`page_end`) — essential for recovering content
  from the PDF later.
- **Continuation paragraphs** (overflow prose and `<em>`Climate`</em>.]` /
  `History.]` section headers) are merged back into the place they describe.
- **Cross-references** (`LUTTICH. See LIEGE.`) are classified as `kind='crossref'`
  with the target captured in `see_target`.
- **Statistical tables** are attached to their place (`n_tables`); see the
  [table-recovery](#3-table-recovery-todo) note.
- **Toponym case** — headwords are printed UPPERCASE; `headword_disp` holds a
  title-cased form (`LUS-LA-CROIX-HAUTE` → `Lus-la-Croix-Haute`) with
  multilingual connecting particles (`de`, `la`, `von`, `of`, …) lower-cased.

**Volume 5 result:** 12,477 entries · 730 cross-references · 140 tables ·
1,128 multi-place entries (≈14,800 places once split) · pages 1–874.

#### Schema

- **`source`** — one row per HTML file (sha256, counts, page range).
- **`entry`** — one row per logical headword block. Key columns: `kind`,
  `headword` / `headword_raw` / `headword_disp`, `page_start`/`page_end`,
  `raw_html`, `text`, `n_tables`, `n_also`, `see_target`, `tokens`.
- **`place`** — *the unit of interest*; populated by the LLM stage. One row per
  place (a multi-place entry yields several). Holds `extraction` (JSON),
  `aat_type_id`, reconciliation result (`whg_match_id`, `lat`, `lon`), `status`.

> **Why SQLite (not DuckDB)?** The workload is write-heavy and incremental —
> each place's row is updated as LLM extraction and then reconciliation complete.
> SQLite suits many small single-writer updates; DuckDB shines for columnar
> analytics. We can export to Parquet/DuckDB for analysis at any point; the
> schema translates directly.

### 2. LLM extraction → `place` *(next)*

Each `entry` is sent to an LLM to produce one structured record **per place**:
- canonical `name` (+ variants), feature type as a **Getty AAT** concept id,
- the **geographic context needed to disambiguate** the toponym (containing
  country / province / district, nearby places + bearings/distances, coordinates
  if printed, population, area),
- multi-place splitting on `—Also …` boundaries (`n_also` is a heuristic count),
- cross-reference resolution using `see_target`.

AAT integration follows the approach in
`../London_Customs_Accounts/AAT_INTEGRATION_PLAN.md`: load the Getty AAT bulk
export locally and supply a **shortlist of relevant concepts** to the model.

The shortlist is **drafted and validated** — `data/aat_shortlist.json`, built by
`process/build_aat_shortlist.py` from the feature-type descriptors mined across
Volume 5 and resolved to real AAT ids via `process/aat_resolve.py`. 47 concepts
across WHG fclasses: populated places (town/village/hamlet/city/capital/port),
administrative divisions (parish/commune/canton/district/department/county/
province/state/country…), water bodies (river/lake/bay/strait/creek…),
terrestrial landforms (island/cape/mountain/peninsula/valley/volcano…), and
fortifications. The AAT dump (`~/Documents/GitHub/whg3/data/aat/`, ~59k concepts,
2026-01 release) is complete and current — no re-download needed.

**Cost estimate (full 7-file corpus)** — extrapolated ×7 from Volume 5:

| | value |
|---|---|
| Entries / cross-refs / places | ≈87,000 / 5,100 / **104,000** |
| Input text | ≈13.0M tokens |
| **Total (Batch API, length-routed)** | **≈ $84** |
| Range | $72 (all-Haiku) … $216 (all-Sonnet) |

Routing sends short/standard entries (≤800 tok, ~98%) to Haiku in batches and
the long dense entries to Sonnet. Cost is dominated by *output* (~25M tokens of
structured JSON); prompt caching makes the shared instruction/AAT/schema prefix
near-free. Realtime (non-Batch) ≈ 2×. Assumed list rates are documented in
`process/` and should be reconfirmed before committing spend. Re-derive exactly
once all 7 files are parsed: `SELECT SUM(tokens) FROM entry`.

### 3. Table recovery *(todo)*

Per Humphrey Southall, **many embedded statistical tables were omitted** during
transcription. Volume 5 retains 140 tables; the omitted ones generally left no
textual trace (their introductory sentence was dropped too), so HTML-only
detection is unreliable. Recovery plan: use each place's `page_start`/`page_end`
to locate the relevant **PDF scan pages**, detect table regions, and OCR/crop
them back into the `place` record.

### 4. Map extraction *(todo)*

The PDF carries maps as part of the scanned page images (no vector data). Plan:
detect map regions on the relevant scan pages and render them to linked image
files, associated with the place(s) they illustrate via page provenance.

### 5. MapLibre demo — GitHub Pages *(todo)*

A static demo UI (MapLibre GL) hosted on GitHub Pages, plotting a sample of
~100 extracted places geolocated via the WHG Reconciliation API — to showcase
the pipeline end-to-end.

## Credits

- **Prof. Humphrey Southall** — University of Portsmouth; Director, Great Britain
  Historical GIS. Oversaw the OCR transcription of the seven volumes that this
  project builds upon.
- **World Historical Gazetteer** (University of Pittsburgh; Dir. Prof. Ruth
  Mostern) — reconciliation indices and authority-gazetteer framework.
