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

> **Two different volumes are in play.** The OCR transcript we parse is **Volume V**; the scanned PDF
> in `data/pdf/` is **Volume VII (TA–ZZUBIN, with Appendix)**. They are different volumes of the same
> work — so the HTML's page numbers do **not** index the held PDF (see [table/map recovery](#5-tables-maps-and-the-demo)).
> Volume VII's Appendix, however, is itself a prize: a bidirectional ancient↔modern toponym
> concordance — see [the Appendix concordance](#6-volume-vii-appendix--toponym-authority).

---

## The method (reusable)

A general recipe for *historical PDF → geolocated linked data*, with the transferable
lesson at each step:

| Stage | What you do | Transferable lesson |
|------|-------------|---------------------|
| **0. Get text** | Obtain (or OCR) a text transcript of the PDF. | OCR is rarely clean; plan to correct it, ideally *inside* a step you're already paying for. |
| **1. Parse to records** | Split the flow into one row per source record, keeping **provenance** (page, raw markup). | Parse defensively: the obvious delimiter (here, `<p>`) lies — page breaks fall mid-record, cross-references and continuations masquerade as entries. Validate against spot-checks, not assumptions. |
| **2. Model the target** | Decide your unit of interest and a controlled vocabulary for typing it. | Mine the *actual* data for its categories before choosing vocabulary terms; resolve them to real authority IDs (here Getty AAT) and **validate every ID** so a typo fails loudly. |
| **3. LLM extraction** | Send each record to an LLM for structured output: the entity, its type, and the **context needed to disambiguate** it. | Use schema-constrained (structured) output, cache the shared prompt prefix, route by length to control cost, and **estimate cost up front** from a real sample. Fold OCR correction into this same pass. |
| **4. Reconcile / geolocate** | Match each place against a gazetteer service to attach coordinates and identifiers. | Disambiguation context (containing region, neighbours, coordinates) is what makes reconciliation accurate — extract it deliberately in step 3. |
| **5. Publish** | Export linked data; show it. | A small interactive demo (here MapLibre on GitHub Pages) communicates the result better than a data dump. |

Everything here runs on the Python standard library plus `beautifulsoup4`, `lxml`,
`tiktoken`, `pymupdf`, `anthropic`, and `pydantic` (see [Setup](#setup)).

---

## This project's pipeline

```
HTML transcript ──parse──▶ entry ──OCR-correct──▶ entry.text_corrected ──LLM extract──▶ place
   (Humphrey, Vol V)                  ▲                                                    │
                          Hathi .txt + toponym dict                  AAT typing · multi-place split ·
                          (cleaner chars, authority)                 cross-ref · ethnography · context
                                                                                          │
                                                       WHG Reconciliation API ──▶ coordinates ──▶ MapLibre demo

v7 Appendix ──vision-LLM──▶ name_variant ──▶ dict/toponyms.json   (ancient↔modern toponym authority)
```

| Asset | Notes |
|-------|-------|
| `data/pdf/gotw-v{1,5,7}.pdf` | Per-volume scans (Vol I `AAR…BRA`, Vol V `LUS…PERTHSHIRE`, Vol VII `TA…ZZUBIN`+Appendix), built from HathiTrust images+OCR by `build_pdf.py`. Git-ignored; rebuild from source. |
| `data/html/…html` | OCR transcript of **Volume V** by **Prof. Humphrey Southall** — the *structural* base (clean entry/paragraph segmentation), corrected against the Hathi OCR. |
| `data/jpeg/`, `data/txt/` | HathiTrust per-volume downloads (600dpi image zips + OCR text). Transient — consumed and deleted by `build_pdf.py`. Git-ignored. |
| `data/gotw.sqlite` | Working store (entry, place, table_data, name_variant, corrections, llm_cache…). Git-ignored; rebuild from source. |
| `data/aat_shortlist.json` | 47 validated Getty AAT feature-type concepts (committed). |
| `dict/toponyms.json` | 27,528-toponym authority built from the Vol VII Appendix (committed). |

Source work: the **Royal Geographical Society** (1856); transcription by **Prof. Humphrey Southall** — see [Credits](#credits).

### Setup

```bash
pip install anthropic google-genai beautifulsoup4 lxml tiktoken pymupdf pydantic requests
# Secrets go in .env (git-ignored): WHG_API_TOKEN, ANTHROPIC_API_KEY, GEMINI_API_KEY
# OCR correction reads a word list from dict/words or /usr/share/dict/words
```

### 1. Parse — `process/parse_html.py`
```bash
python3 process/parse_html.py data/html/gotw_vol5_all_web_v1.html --db data/gotw.sqlite
```

The transcript is one long flow of `<p>` elements. The parser handles what breaks
naive splitting:

- **Page markers `[[N]]`** sit inside `<br/>` runs and can fall *mid-paragraph*, gluing
  the tail of one place onto a new headword. We split on them and record page provenance
  (`page_start`/`page_end`) — essential for going back to the PDF later.
- **Continuation paragraphs** (overflow prose, `<em>Climate</em>.]` / `History.]` section
  headers) are merged back into the place they describe.
- **Cross-references** (`Luttich. See Liege.`) are classified as `kind='crossref'` with
  the target in `see_target`.
- **Statistical tables** are attached to their place (`n_tables`) — see [table recovery](#5-tables-maps-and-the-demo).
- **Toponym case** — headwords are printed UPPERCASE; `headword_disp` holds a title-cased
  form (`LUS-LA-CROIX-HAUTE` → `Lus-la-Croix-Haute`) with multilingual particles
  (`de`, `la`, `von`, `of`, …) lower-cased.

**Volume 5 result:** 12,477 entries · 730 cross-references · 140 tables · 1,128 multi-place
entries (≈14,800 places once split) · pages 1–874.

**Schema:** `source` (one row per HTML file) → `entry` (one row per headword block) →
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

### 1b. OCR correction — `process/correct_ocr.py`
```bash
python3 process/build_toponym_dict.py                       # seed dict/toponyms.json from the v7 Appendix
python3 process/correct_ocr.py --hathi data/txt/gotw-v5.txt # write entry.text_corrected + correction log
```

The two OCR sources are complementary: **Humphrey's HTML has clean structure** (reflowed paragraphs,
segmented entries) but `n↔u`/`o↔u` misreads; **HathiTrust's `.txt` reads characters better** but has
two-column *layout* faults (merged columns, gutter `|`, scrambled order). So we keep Humphrey's
structure and import only Hathi's better characters — deterministic, free, and fully logged.

Per entry: clean the Hathi stream, anchor on the headword, align word-tokens (`difflib`), and decide
each difference conservatively — **never numbers**; apply a swap only when it's safe:
- a **known toponym** wins (via `dict/toponyms.json`): `Bastogue→Bastogne`, accent restored
  `Isere→Isère`; and `Mirebeau` is *kept* (Hathi's `Mirebean` isn't a toponym);
- a **Humphrey non-word → general-dictionary word**: `commnne→commune`, `tho→the`;
- everything ambiguous (e.g. `arc/are`, unknown proper nouns) is **flagged, not applied** (`ocr_flags`),
  to be settled by the corpus variant-tally or the extraction LLM.

This avoided the naive-rule corruptions (`Mirebeau→Mirebean`, `ocean→occan`, `1½→14½`). v5: **2,364
corrections applied** (1,572 word · 722 toponym · 70 diacritic), 3,196 flagged; written to
`entry.text_corrected` with a per-change `reason` in `corrections`. Extraction reads the corrected
text; the LLM's own OCR-correction remains a backstop.

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
- **OCR correction in-pass** — the transcript is uncorrected OCR (`commnne`→commune,
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

- **Tables — analysed** — `process/parse_tables.py` parses the **140** tables already in the HTML into
  structured rows (`table_data`) and classifies them by subject: trade 23, population 18, climate 13,
  agriculture 11, area/revenue 9 (up to 74 rows).
- **Tables — recovered from the scan** — `process/extract_tables.py` digitises any table directly from
  the v5 scan with a vision-LLM (printed page → structured `{title, header, rows}`), recovering the
  tables Southall's transcription dropped. Validated on the *Madras Presidency* page (the districts
  table + a climate table, matching the HTML). Output-dominated and cached, like the Appendix.
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
two purposes: it's name-variant data for WHG, and it's the authority the [OCR-correction step](#1b-ocr-correction--processcorrect_ocrpy)
uses to settle place-name disagreements. (The Appendix's own OCR drops some accents, so a `--from-corpus`
refinement — tallying spellings across all volumes, most-populous wins — will fill those in over time.)

---

## Credits

- **Royal Geographical Society** — corporate author of the source work, *A Gazetteer of the World*
  (A. Fullarton & Co., 1856), published anonymously "edited by a Member of the Royal Geographical Society".
- **Prof. Humphrey Southall** — University of Portsmouth; Director, Great Britain Historical
  GIS. Oversaw the OCR transcription of the seven volumes this project builds upon.
- **World Historical Gazetteer** (University of Pittsburgh; Dir. Prof. Ruth Mostern) —
  reconciliation indices and the authority-gazetteer framework.
- Place types use the Getty **Art & Architecture Thesaurus** (AAT), made available under the
  [ODC Attribution License](https://www.getty.edu/research/tools/vocabularies/license.html).
