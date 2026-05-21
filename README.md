# Gazetteer of the World → Linked Data

Turning a 19th-century printed gazetteer into a structured, AAT-typed, geolocated
**authority gazetteer** for the [World Historical Gazetteer](https://whgazetteer.org/) (WHG).

> **This repo doubles as a worked exemplar** for digital-humanities practitioners who
> want to extract structured data from a historical PDF and geolocate the places it
> names. The [Method](#the-method-reusable) section below is written to be lifted and
> adapted; the rest documents how this project instantiates it.

Source: **Royal Geographical Society**, *A Gazetteer of the World, or Dictionary of Geographical
Knowledge* (Edinburgh: A. Fullarton & Co., 1856), 7 vols. The work was published anonymously —
"edited by a Member of the Royal Geographical Society" — so the **RGS is cited as the corporate
author**.
[archive.org/details/agazetteerworld00unkngoog](https://archive.org/details/agazetteerworld00unkngoog).

> **We OCR the public-domain scans ourselves.** Rather than reuse a third-party transcript, we run
> the **1856 first-edition page images (HathiTrust vols 1–7)** through a modern layout-aware OCR model
> ([Surya](https://github.com/datalab-to/surya)) on a GPU cluster — fully public-domain, no licence
> encumbrance, and cleaner than the 2015 OCR it replaces (see [the OCR stage](#1-ocr-the-scans--processocr_pagespy)).
> Volume VII's **Appendix** is a prize in its own right: a bidirectional ancient↔modern toponym
> concordance — see [the Appendix concordance](#6-volume-vii-appendix--toponym-authority).

---

## The method (reusable)

A general recipe for *historical PDF → geolocated linked data*, with the transferable
lesson at each step:

| Stage | What you do | Transferable lesson |
|------|-------------|---------------------|
| **0. Get text** | OCR the page scans yourself with a **layout-aware** model. | Don't inherit someone else's OCR (licence + quality risk). Modern OCR (Surya) reads diacritics and coordinates cleanly; let it segment tables/figures, but reconstruct multi-column reading order from line geometry — layout models treat a dense column body as one block. |
| **1. Parse to records** | Split the flow into one row per source record, keeping **provenance** (page, raw markup). | Parse defensively: the obvious delimiter lies — page breaks fall mid-record, cross-references and continuations masquerade as entries. Validate against spot-checks, not assumptions. |
| **2. Model the target** | Decide your unit of interest and a controlled vocabulary for typing it. | Mine the *actual* data for its categories before choosing vocabulary terms; resolve them to real authority IDs (here Getty AAT) and **validate every ID** so a typo fails loudly. |
| **3. LLM extraction** | Send each record to an LLM for structured output: the entity, its type, and the **context needed to disambiguate** it. | Use schema-constrained (structured) output, cache the shared prompt prefix, route by length to control cost, and **estimate cost up front** from a real sample. Fold OCR correction into this same pass. |
| **4. Reconcile / geolocate** | Match each place against a gazetteer service to attach coordinates and identifiers. | Disambiguation context (containing region, neighbours, coordinates) is what makes reconciliation accurate — extract it deliberately in step 3. |
| **5. Publish** | Export linked data; show it. | A small interactive demo (here MapLibre on GitHub Pages) communicates the result better than a data dump. |

Everything here runs on the Python standard library plus `surya-ocr`, `pymupdf`,
`beautifulsoup4`, `lxml`, `tiktoken`, `anthropic`, and `pydantic` (see [Setup](#setup)).
OCR runs on a GPU (we use the Pitt CRC cluster); the rest runs anywhere.

---

## This project's pipeline

```
PD scans ──Surya OCR──▶ volume .txt ──parse──▶ entry ──toponym-check──▶ entry.text_corrected ──LLM extract──▶ place
 (vols 1–7)  (GPU/CRC)   (## p. N)                  ▲                                                       │
                                            dict/toponyms.json                  AAT typing · multi-place split ·
                                            (authority + Surya cross-check)      cross-ref · ethnography · context
                                                                                                           │
                                                              WHG Reconciliation API ──▶ coordinates ──▶ MapLibre demo

v7 Appendix ──vision-LLM──▶ name_variant ──▶ dict/toponyms.json   (ancient↔modern toponym authority)
```

| Asset | Notes |
|-------|-------|
| `data/pdf/gotw-v{1..7}.pdf` | Per-volume public-domain scans, built from HathiTrust 600dpi images by `build_pdf.py`. The OCR input. Git-ignored; rebuild from source. |
| `data/txt/ocr/v{N}/p*.txt`, `data/txt/gotw-v{N}-ocr.txt` | Our **Surya OCR** output — one resumable file per page, merged per volume (`## p. N` markers). Produced by `ocr_pages.py`. Git-ignored. |
| `data/gotw.sqlite` | Working store (entry, place, table_data, name_variant, corrections, llm_cache…). Git-ignored; rebuild from source. |
| `data/aat_shortlist.json` | 47 validated Getty AAT feature-type concepts (committed). |
| `dict/toponyms.json` | 27,528-toponym authority built from the Vol VII Appendix (committed). |

Source work: the **Royal Geographical Society** (1856), public domain — see [Credits](#credits).

### Setup

```bash
pip install anthropic google-genai beautifulsoup4 lxml tiktoken pymupdf pydantic requests
pip install surya-ocr          # OCR stage only; needs a CUDA GPU (we run it on the Pitt CRC cluster)
# Secrets go in .env (git-ignored): WHG_API_TOKEN, ANTHROPIC_API_KEY, GEMINI_API_KEY
# The toponym cross-check reads a word list from dict/words or /usr/share/dict/words
```

### 1. OCR the scans — `process/ocr_pages.py`
```bash
# one page to stdout (quick check — needs a CUDA GPU)
python3 process/ocr_pages.py --pdf data/pdf/gotw-v5.pdf --start 84 --end 84
# shard a volume into resumable per-page files, then stitch into one volume .txt
python3 process/ocr_pages.py --pdf data/pdf/gotw-v5.pdf --out-dir data/txt/ocr/v5 --start 0 --end 199
python3 process/ocr_pages.py --pdf data/pdf/gotw-v5.pdf --out-dir data/txt/ocr/v5 --merge --out data/txt/gotw-v5-ocr.txt
```

We OCR the public-domain scans ourselves with **[Surya](https://github.com/datalab-to/surya)**, a
modern layout-aware OCR model. It reads this 1856 print far better than the 2015 transcript or
Tesseract: validated on a CRC L40S GPU it returns clean diacritics (*São-Pedro*, *Maranhão*), intact
coordinate glyphs (*S lat. 37° 10′*), and correct two-column reading order, at **~4 s/page** (0.4 s
layout + 3.5 s detect+recognise).

What generalises:
- **Layout segmentation has limits.** Surya's `LayoutPredictor` reliably finds **tables and figures**
  — we route those to [`extract_tables.py`](#5-tables-maps-and-the-demo)/`extract_maps.py` and exclude
  them from the prose — but it treats the dense two-column body as a *single* text block; it will
  **not** split the columns. So we reconstruct **two-column reading order from the recognised line-box
  geometry** (left column top-to-bottom, then right). The page marker (`## p. N`) is read from the
  running-head number in the top margin.
- **Resumable + shardable.** One file per page (`p<idx>.txt`, written atomically); a page already on
  disk is skipped, so a re-run only fills gaps. `--merge` stitches a volume into one `## p. N`-marked
  `.txt` for the parser — the same format the downstream steps already understand.
- **Cluster submitter** — `process/submit_ocr_slurm.py` shards a volume across the Pitt CRC **GPU
  cluster** as a SLURM array (`--clusters=gpu --partition=l40s --gres=gpu:1`, conda `whg`, models +
  scans on fast `/vast/ishi`). At ~4 s/page the ~6,600-page corpus is ~7 h on one GPU, minutes across
  the array; `--partition preempt` taps the large pre-emptible pool for free since OCR resumes.

> **Why self-OCR (and no external transcript).** An earlier OCR transcript of these volumes exists, but
> it is licensed **CC-BY-SA-NC** with an attribution/co-authorship requirement. To keep this gazetteer's
> output **cleanly public-domain and unencumbered**, the project does **not** use that transcript or any
> derivative of it — we OCR the public-domain 1856 page scans ourselves with Surya. Modern OCR is also
> markedly better. *(The local 6 GB laptop GPU stalls Surya at init on torch 2.9/CUDA 12.8; the cluster
> A100/L40S nodes run it fine.)*

### 1b. Parse to entries — `process/parse_ocr.py`
```bash
python3 process/parse_ocr.py data/txt/gotw-v1-ocr.txt --volume v1 --db data/gotw.sqlite
python3 process/parse_ocr.py data/txt/gotw-v1-ocr.txt --volume v1 --dry-run   # stats only
```

The OCR stream has **no markup** — entries are delimited only by the print convention, so the parser
segments on the typographic signal and heals what OCR/line-wrapping break:

- **Headword segmentation** — each entry opens with an **ALL-CAPS headword** then a descriptor
  (`MALABAR, a district of …`); cross-references are `HEADWORD. See TARGET.`. We start a new entry on
  any line matching that shape, with guards against false positives (initialisms like `A.M.`/`S.W.`,
  bare running heads) — the boundary the markup used to give us for free.
- **Prose flow & de-hyphenation** — wrapped lines are merged back into one text, healing end-of-line
  hyphenation (`table-\nland` → `table-land`, `ot-\nters` → `otters`).
- **Page provenance** — each page's `## p. N` marker (read by [OCR](#1-ocr-the-scans--processocr_pagespy)
  from the running head) gives `page_start`/`page_end`, so every entry traces back to the scan.
- **Cross-references** (`Aarafat. See Arafat.`) are classified `kind='crossref'` with the target in `see_target`.
- **Toponym case** — headwords are printed UPPERCASE; `headword_disp` holds a title-cased form
  (`LUS-LA-CROIX-HAUTE` → `Lus-la-Croix-Haute`) with multilingual particles (`de`, `la`, `von`, `of`, …) lower-cased.

Statistical tables are **not** linearised into the prose — they are digitised separately by a vision-LLM
(see [tables](#5-tables-maps-and-the-demo)). `parse_ocr.py` replaces the earlier `parse_html.py`, which
parsed a third-party HTML transcript no longer used.

**Volume I (our OCR):** 11,393 entries · 813 cross-references · ~1,426 multi-place entries (via `—Also`)
· pages 3–896. The remaining volumes parse as their OCR completes.

**Schema:** `source` (one row per volume) → `entry` (one row per headword block) →
`place` (the unit of interest; populated by step 3). We use **SQLite, not DuckDB**, because
the workload is write-heavy incremental updates as extraction and reconciliation complete;
export to Parquet/DuckDB for analytics whenever you want.

> **Volume acquisition.** We use the **1856 first edition (HathiTrust vols 1–7)** — *not* the
> undated 8–14 set on the same record (a different edition). `process/fetch_archive.py` pulls
> public-domain copies from the Internet Archive (the legitimate, ToS-clean alternative to scripting
> HathiTrust's authenticated reader). HathiTrust 600dpi image zips + OCR `.txt` are stitched into a
> searchable per-volume PDF by `process/build_pdf.py` (image + invisible OCR layer; deletes the bulky
> source on a verified build). `process/pdf_pages.py` maps printed page → PDF index from the OCR
> headers (handling the drift from unpaginated plates); `process/pdf_coverage.py` reports each PDF's
> head-word range so you can confirm the seven tile A–Z without overlap or gap.

### 1c. Toponym cross-check — `process/correct_ocr.py`
```bash
python3 process/build_toponym_dict.py                        # build dict/toponyms.json from the v7 Appendix
python3 process/correct_ocr.py --hathi data/txt/gotw-v5.txt  # apply the toponym authority (+ optional 2nd-OCR cross-check)
```

A single high-quality Surya pass removes the need for the heavy two-transcript diff-correction this step
once did. What stays valuable is the **toponym authority** `dict/toponyms.json` — 27,528 place-name
spellings from the [v7 Appendix](#6-volume-vii-appendix--toponym-authority) that a general dictionary
can't supply. It restores accents and fixes place-name misreads (`Bastogue→Bastogne`, `Isere→Isère`)
that even good OCR makes. `correct_ocr.py` applies it conservatively, and can fold in a **second,
independent OCR** (e.g. the HathiTrust `.txt`) as a cross-check where available — anchoring on the
headword, aligning word-tokens (`difflib`), and applying a swap only when safe:
- a **known toponym wins** (via `dict/toponyms.json`); a name only one source spells correctly is restored;
- **never numbers** (measurements/coordinates untouched), and a real word is never "corrected" into a non-word;
- everything ambiguous (unknown proper nouns, `arc/are`) is **flagged, not applied** (`ocr_flags`), to be
  settled by the corpus variant-tally or the extraction LLM.

These guards came from real corruptions they prevent (`Mirebeau→Mirebean`, `ocean→occan`, `1½→14½`).
Changes write to `entry.text_corrected` with a per-change `reason` in `corrections`; extraction reads the
corrected text, with the LLM's own OCR-correction as a backstop. *(The cross-check needs a second OCR
stream; with Surya alone, the toponym pass is the active path.)*

### 2. Feature-type vocabulary — `process/aat_resolve.py`, `process/build_aat_shortlist.py`
```bash
python3 process/aat_resolve.py            # build/validate the AAT index from the local Getty dump
python3 process/build_aat_shortlist.py    # write + self-validate data/aat_shortlist.json
```

The descriptors that open each entry ("…, a **town/parish/river/island** of …") were mined
to find the categories actually present, then resolved to **real Getty AAT concept ids** and
grouped by WHG fclass: populated places, administrative divisions, water bodies, terrestrial
landforms, fortifications, and (for completeness) peoples. The builder **self-validates every
id** against the AAT index, so a wrong/stale id fails loudly rather than silently mis-typing.

> Gotcha worth knowing: in the Getty `AATOut_2Terms.nt` dump each concept has a `prefLabel`
> *per language*; the English term URI ends `-en`. Keep the `-en` one or you silently drop
> most concepts. The dump (~59k concepts) is complete as of 2026-01 — no re-download needed.

### 3. LLM extraction → `place` — `process/extract.py`
```bash
python3 process/extract.py --dry-run --limit 1                  # prompt + schema + a request (no key)
python3 process/extract.py --provider gemini --limit 20         # sync (Flash-Lite/Flash)
python3 process/extract.py --provider claude --limit 20         # sync (Haiku/Sonnet)
python3 process/extract.py --provider claude --batch            # full corpus via Batch API (≈50% off)
python3 process/extract.py --provider gemini --batch            # Gemini Batch Mode (≈50% off)
python3 process/extract.py --provider <p> --collect <id>        # write results once the batch ends
```

**Provider-pluggable for a cost/quality A/B.** A thin provider interface backs two implementations —
`claude` (Haiku→short / Sonnet→long) and `gemini` (Flash-Lite→short / Flash→long) — selectable with
`--provider`, with `--model NAME` to force one model for every entry. Keys come from `.env`
(`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). **Both providers support the 50%-off batch path** (Claude
Batch API; Gemini Batch Mode, one job per routed model with the entry id carried on each request's
`metadata`). The Gemini SDK only resolves `generativelanguage.googleapis.com`, which some networks
fail to look up — the client auto-pins it via public DoH when local DNS can't.

**Every successful result is cached** in an `llm_cache` table, keyed by `(provider, model,
prompt+schema signature, entry text)`. Re-runs, re-collects, and switching back to a model you've
already used cost **zero** API calls — essential for running the A/B without burning quota. Editing
the prompt or schema changes the signature and so re-extracts deliberately. (Verified live: a 2-entry
Haiku run made 2 calls; the immediate re-run made 0 and served both from cache.)

Each entry yields one **structured record per place** (Pydantic-validated against a JSON
schema): canonical name + variants, the feature type as a **Getty AAT id** (a closed enum —
the model can only pick a shortlist concept or `"other"`), and the disambiguation context —
country (+ present-day ISO code), administrative hierarchy, nearby places with bearing/distance,
coordinates (DMS → decimal), population (a `[{year, count}]` time series), area, and a **`peoples`
ethnography facet** — precise ethnonyms named in the entry (e.g. *Lumris, Numaris, Wends*), typed
AAT 300191997 "ethnic groups". Multi-place entries split on "—Also …"; cross-references resolved
from `see_target`. Extraction reads `entry.text_corrected` (step 1b) when present.

Design choices that generalise:
- **Structured output** (`output_config` json_schema) guarantees parseable, schema-valid records.
- **Prompt caching** — the instructions + AAT shortlist + schema are a stable cached prefix;
  only the per-entry text varies, so the big shared prefix is near-free across thousands of calls.
- **Length-routed models** — short/standard entries → Haiku, long dense entries → Sonnet
  (configurable at the top of `extract.py`; switch to Opus there if you prefer maximum quality).
- **OCR correction in-pass** — even good OCR leaves some misreads (`commnne`→commune,
  `villnge`→village, `fortres`→fortress…). The model fixes obvious garbling while it reads and
  logs each fix to an `ocr_corrections` QA trail — no separate, extra-cost correction pass.

**Choosing a model — `process/ab_compare.py`.** Runs a deterministic sample through several
`provider:model` configs (reusing the cache, never touching `place`) and reports cost, field
coverage, and inter-model agreement (no gold standard, so agreement is a quality proxy), with a
per-entry disagreement dump. On a 20-entry sample, all three models agreed **100% on `country_code`**
(the reconciliation-critical field) and ~80–87% on AAT type and place-count; Gemini Flash-Lite came
in ~12× cheaper than Haiku and Flash ~3× cheaper. It also caught a real quality issue — Flash-Lite
*parroted the prompt's OCR examples* as if it had found them, inflating `ocr_corrections` — so the
cheapest model isn't automatically the right one. (Run it yourself; results are cached, so it's free
to re-run.) **The generated report is published at [`docs/model-comparison.md`](docs/model-comparison.md)**
as a worked example for practitioners weighing extraction models.

**Cost (full 7-file corpus)** — extrapolated ×7 from Volume 5 by `process/estimate_cost.py`:
≈87,000 entries / ≈104,000 places / ≈13M input tokens → **≈ $84 via the Batch API**
(range $72 all-Haiku … $216 all-Sonnet; ~2× for realtime). Dominated by *output* tokens;
prompt caching makes the shared prefix negligible. ⚠️ Rates are assumed list prices — confirm
before committing spend. Re-derive exactly once all 7 files are parsed: `SELECT SUM(tokens) FROM entry`.

### 4. Reconcile against WHG — `process/reconcile.py`
```bash
python3 process/reconcile.py --seed-demo 8   # curated demo places (no extraction needed)
python3 process/reconcile.py --limit 50      # reconcile pending places
```

Geolocates each `place` via WHG's
[Reconciliation API](https://docs.whgazetteer.org/content/technical/apis.html#reconciliation-service-api):
it sends the extracted disambiguation context, picks the best candidate by score, fetches that
match's centroid via a data-extension call, and writes `whg_match_id`, `whg_score`, `lat`, `lon`,
and the full candidate JSON back to `place`. The token is read from `.env` (`WHG_API_TOKEN`).

What the live API actually rewards (probe before you trust the docs):
- **Don't filter by AAT type.** `types: ["aat:…"]` returns *zero* hits — WHG isn't indexed by AAT;
  our typing is enrichment, not a query key.
- **Filter by country, via `countries` (ISO codes), not `fclasses` or `ccodes`.** `fclasses` is too
  sparse to rely on; `ccodes` is accepted but doesn't actually constrain; only `countries` genuinely
  filters. This is why extraction emits a present-day `country_code`. It's decisive: it moved
  *Lusatia* off **Lusaka, Zambia** and *Luton* off a Devon namesake onto Bedfordshire.
- **Threshold on `score`, not `match`** — WHG's `match` flag is conservative (a score-100 hit can be
  `match:false`).
- **Coords-only fallback for recall** — a country filter drops places whose WHG record lacks a country
  code (e.g. small islands); when that happens and the gazetteer printed coordinates, retry without
  `countries` using `lat`/`lng`/`radius`.

On the 8 curated demo places this reconciles **8/8** with correct coordinates.

### 5. Tables, maps, and the demo

A matching scan of **Volume V** (`data/pdf/gotw-v5.pdf`, with an OCR text layer) makes table and map
work tractable. `process/pdf_pages.py` maps printed page → PDF index from the header page numbers,
handling the offset drift (16→80) caused by ~70 unpaginated steel plates that defeat a constant offset.

- **Tables — vision-LLM into `table_data`** — Surya's layout model does **not** detect these unruled
  1856 tables, and prose OCR scrambles their cells, so `process/extract_tables.py` digitises them with a
  vision-LLM. To bound cost it first flags **candidate pages by numeric-row runs** in the column-ordered
  OCR (a table leaves a run of consecutive number-heavy lines; prose tops out ~2, tables score 6–9), then
  sends each candidate to the model for structured `{title, header, rows}` stored in `table_data`.
  Validated: the *Madras* climate + districts tables digitise cleanly, and the detector flags real table
  pages (e.g. *Barbados* meteorology + trade, run 9). Output-dominated and cached, like the Appendix.
- **Maps** — `process/extract_maps.py` finds illustration plates (ink-filtering out blank/stamp pages
  and marbled endpapers) and vision-classifies them. **Volume V contains no cartographic maps**: its 8
  steel plates are all city/landscape views (Magdeburg, Malta, Melrose Abbey, Mytelene, New York Bay,
  Padua, Patras). The tool writes an illustration manifest (titles + page provenance) and would
  crop/export any genuine maps — more likely to appear in other volumes.
- ⚠️ **Other volumes** — we hold scans of Vol V and Vol VII; the remaining five are a prerequisite for
  table/map recovery across the corpus. Source them as the **1856 first edition = volumes 1–7** on
  [HathiTrust 011407465](https://catalog.hathitrust.org/Record/011407465); the **undated 8–14 set** on
  the same record is a *different edition* (its v14 reproduces 1856 v7's Article II) and must not be
  mixed in. Trust each PDF's actual head-word range over its number — `process/pdf_coverage.py` reports
  it from the OCR text layer (Vol I `AAR…BRA`, Vol V `LUX…PERTHSHIRE`; the Vol VII scan lacks a text
  layer) so you can confirm the seven tile A–Z without overlap or gap. HathiTrust 600dpi page-image
  zips + the OCR `.txt` are bundled into a searchable per-volume PDF by `process/build_pdf.py`
  (image + invisible OCR layer; reads HathiTrust's `## p. N` page markers); it deletes the bulky
  `.zip`/`.txt` once the PDF is verified (`--keep` to retain).
- **MapLibre demo** — a static GitHub Pages UI plotting the extracted, reconciled places with rich
  popups, live at [docuracy.github.io/GOTW/map.html](https://docuracy.github.io/GOTW/map.html). Built
  from the cached Gemini 2.5 Flash set via `process/export_geojson.py` → `docs/places.geojson`.

> **Looking ahead — demographic change over time.** Extraction captures population as a structured
> `[{year, count}]` time series (the source carries population figures in ~53% of entries, often for
> multiple census years). Once the full corpus is extracted and reconciled, these mid-19th-century
> figures can be joined to **modern population data** on the WHG/Wikidata/GeoNames identifiers the
> reconciliation step attaches — making it possible to visualise long-run demographic change place by
> place. The structured, ID-linked output is the enabling step; the gazetteer becomes a dated baseline.

### 6. Volume VII Appendix → toponym authority

The held PDF (Volume VII) ends with an **Appendix that is itself a toponym-variant goldmine** for WHG —
a bidirectional concordance of historical and modern place names:

- **Article I** (printed p.659): *"A List of Geographical Names showing the Ancient, Mediæval, and Modern
  designations borne by the same place"* — a Modern↔Ancient table with an abbreviations key, followed by
  an alphabetical ancient/mediæval → modern index (e.g. *Caterlogum → Carlow*).
- **Article II** (printed p.745): *"Reversed Modern, Ancient, and Mediæval Index"* — the inverse,
  modern → ancient/mediæval (e.g. *Aachen → Aquisgranum, Aquae Grani*).

These pages are scanned with no usable column layout in the plain text, so we extract them with a
**vision-LLM** — `process/extract_appendix.py` renders each page (printed page = PDF index − 52) and
returns structured `{headword, equivalents[], note}` rows into `name_variant` (cached per page; vision
thinking off; archaic long-ſ normalised). The **full Appendix is extracted**: 16,884 variant rows from
158 pages for **≈ $3 (≈$1.5 batched)** on Gemini 2.5 Flash (costing alongside the model A/B in
[`docs/model-comparison.md`](docs/model-comparison.md)).

`process/build_toponym_dict.py` then folds those rows into **`dict/toponyms.json` — a 27,528-toponym
authority** (accent-folded key → best-attested canonical spelling + variant forms + era). It serves
two purposes: it's name-variant data for WHG, and it's the authority the [toponym cross-check](#1c-toponym-cross-check--processcorrect_ocrpy)
uses to settle place-name disagreements. (The Appendix's own OCR drops some accents, so a `--from-corpus`
refinement — tallying spellings across all volumes, most-populous wins — will fill those in over time.)

---

## Credits

- **Royal Geographical Society** — corporate author of the source work, *A Gazetteer of the World*
  (A. Fullarton & Co., 1856), published anonymously "edited by a Member of the Royal Geographical Society".
- **World Historical Gazetteer** (University of Pittsburgh; Dir. Prof. Ruth Mostern) —
  reconciliation indices and the authority-gazetteer framework.
- Place types use the Getty **Art & Architecture Thesaurus** (AAT), made available under the
  [ODC Attribution License](https://www.getty.edu/research/tools/vocabularies/license.html).
