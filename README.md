# Gazetteer of the World → Linked Data

Turning a 19th-century printed gazetteer into a structured, AAT-typed, geolocated
**authority gazetteer** for the [World Historical Gazetteer](https://whgazetteer.org/) (WHG).

> **Scope note.** This began as a portable, locally-runnable recipe, but the working pipeline now
> depends on **Pitt CRC infrastructure** — a GPU cluster (Slurm + vLLM) for self-hosted OCR and LLMs,
> and WHG's own Elasticsearch gateway at Pitt for reconciliation. It is therefore **not turn-key
> reproducible** by an outside DH researcher, and is best read as a **thoroughly-documented record** of
> how this specific gazetteer is being digitised and ingested into WHG — and as a design reference for
> anyone with comparable infrastructure. The *shape* of the pipeline (OCR → parse → typed LLM extraction
> → reconciliation) generalises; the *implementation* assumes the cluster.

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

## Pipeline overview

The stages of *historical PDF → geolocated linked data*, and how this project does each:

| Stage | What it does | Approach / what we found |
|------|-------------|---------------------|
| **0. OCR the scans** | Self-OCR the public-domain page images with a **layout-aware** model. | No inherited OCR (licence + quality). Surya reads diacritics/coordinates cleanly; it won't split dense columns, so we reconstruct two-column reading order from line geometry. Runs as a GPU Slurm array. |
| **1. Parse to records** | Split the flow into one row per source record, keeping **provenance** (page). | Parse defensively: entries are delimited only by the ALL-CAPS-headword print convention — cross-references and continuations masquerade as entries; validate against spot-checks. |
| **2. Model the target** | Decide the unit of interest and a controlled vocabulary for typing it. | Mine the *actual* descriptors for their categories; resolve to real Getty AAT ids and **validate every id** so a typo fails loudly. |
| **3. Self-hosted LLM extraction** | Run every record through **self-hosted open models** for schema-constrained structured output. | Free (no per-token cost): a **primary extractor** (Llama-3.3-70B) + a **critic** (gpt-oss-120B) + a **repair** pass (Qwen3-thinking) on the flagged minority; closed AAT enum enforced; sharded across GPUs. Chosen by a 7-model A/B ([`docs/model-comparison.md`](docs/model-comparison.md)). |
| **4. Reconcile / geolocate** | Match each place to WHG via a **3-pass cascade** (exact → phonetic → proximity). | Disambiguation context (country, printed coordinates) drives it; precision-first then recall, with a spatial bound so border/name changes don't pull in other-side-of-the-world matches. |
| **5. Publish** | Export linked data; show it. | A small MapLibre demo on GitHub Pages communicates the result better than a data dump. |

**Infrastructure dependency.** Stages 0 and 3 require the **Pitt CRC GPU cluster** (Slurm + vLLM,
conda envs `whg`/`vllm`, fast `/vast/ishi` storage); stage 4 queries the WHG reconciliation service.
The lighter steps (parse, AAT build, toponym dictionary) are plain Python (`pymupdf`, `tiktoken`,
`pydantic`, `requests`, `beautifulsoup4`/`lxml`) and run anywhere. See [Setup](#setup).

---

## This project's pipeline

```
PD scans ─Surya OCR─▶ volume.txt ─parse─▶ entry ─extract─▶ place ─reconcile─▶ WHG id + coords ─▶ MapLibre demo
(vols 1-7) (GPU array) (## p.N)             │     Llama-3.3-70B     │      3-pass: exact → phonetic → proximity
                                   toponym-check   (self-hosted vLLM) │      (WHG gateway via public API)
                                   (toponyms.json) + gpt-oss critic
                                                   + Qwen-thinking repair (flagged ~7-11%)
                                                   AAT type · country · population · ethnonyms · multi-place

v7 Appendix ─Qwen2.5-VL─▶ name_variant ─▶ dict/toponyms.json        tables ─Qwen2.5-VL─▶ table_data
```

All LLM stages are **self-hosted on CRC GPUs** (vLLM) — no per-token cost. Reconciliation uses WHG's
public reconciliation service (which proxies to the same Pitt Elasticsearch gateway).

| Asset | Notes |
|-------|-------|
| `data/pdf/gotw-v{1..7}.pdf` | Per-volume public-domain scans, built from HathiTrust 600dpi images by `build_pdf.py`. The OCR input. Git-ignored; rebuild from source. |
| `data/txt/ocr/v{N}/p*.txt`, `data/txt/gotw-v{N}-ocr.txt` | Our **Surya OCR** output — one resumable file per page, merged per volume (`## p. N` markers). Produced by `ocr_pages.py`. Git-ignored. |
| `data/gotw.sqlite` | Working store (entry, place, table_data, name_variant, corrections, llm_cache…). Git-ignored; rebuild from source. |
| `data/aat_shortlist.json` | 47 validated Getty AAT feature-type concepts (committed). |
| `dict/toponyms.json` | 27,528-toponym authority built from the Vol VII Appendix (committed). |

Source work: the **Royal Geographical Society** (1856), public domain — see [Credits](#credits).

### Setup

The light steps (parse, AAT build, toponym dictionary, reconciliation client) are plain Python and
run anywhere; the heavy steps need the **Pitt CRC cluster**.

```bash
# Light steps — run anywhere:
pip install pymupdf tiktoken pydantic requests beautifulsoup4 lxml

# GPU steps — Pitt CRC, via Slurm + vLLM (conda envs on /vast/ishi):
#   env `whg`  : surya-ocr + pymupdf            -> OCR        (process/submit_ocr_slurm.py)
#   env `vllm` : vllm 0.10.2 + torch 2.8+cu128  -> Llama / gpt-oss / Qwen / Qwen2.5-VL
#                (module load cuda/12.8.0 for FlashInfer JIT; VLLM_ATTENTION_BACKEND=FLASH_ATTN)
#   self-hosted model weights + the working DB live on /vast/ishi; servers bind localhost (no tunnel)

# Secrets in .env (git-ignored): WHG_API_TOKEN  (on CRC: ~/.gotw_env, chmod 600).
#   ANTHROPIC_API_KEY / GEMINI_API_KEY are optional — only for the API providers in the model A/B.
# The toponym cross-check reads a word list from dict/words or /usr/share/dict/words.
```

> **Reproducibility.** Stages 0 (OCR) and 3 (extraction) assume CRC's Slurm/vLLM environment and are
> not runnable as-is elsewhere; the `submit_*_slurm.py` scripts document exactly how they run there.
> Reconciliation (stage 4) needs only internet + a WHG API token.

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

### 3. Self-hosted LLM extraction → `place` — `process/extract.py`, `process/submit_extract_slurm.py`
```bash
# production: shard the corpus across GPUs, each serving the model on localhost via vLLM
python3 process/submit_extract_slurm.py --db /vast/ishi/gotw/data/gotw.sqlite --nshards 8
python3 process/extract.py --ingest 'llama_jsonl/llama.*.jsonl' --db data/gotw.sqlite   # merge -> place rows
# single server + concurrent client (e.g. one model on one GPU):
QWEN_BASE_URL=http://localhost:8000/v1 python3 process/extract.py --provider vllm --model llama-3.3-70b --concurrency 48
# the API providers remain only for the model comparison:
python3 process/extract.py --provider claude --limit 20    # or gemini
```

**Self-hosted and free.** Production extraction runs **open models on the CRC GPUs via vLLM** — no
per-token cost. A thin provider interface backs `vllm` (any local OpenAI-compatible server) plus
`claude`/`gemini` (kept only for the A/B). Schema-constrained JSON (vLLM `response_format`) enforces the
**closed AAT-id enum** — a model can only pick a shortlist concept or `"other"`. `submit_extract_slurm.py`
shards across GPUs (each shard serves the model on `localhost`, runs a concurrent client, writes a
per-shard JSONL); `--ingest` merges them into `place` rows, **skipping any entry no longer in the DB**.
Resumable + idempotent (the `llm_cache`/JSONL skip done work).

**Choosing the models — a 7-way A/B** (`process/ab_compare.py` → [`docs/model-comparison.md`](docs/model-comparison.md)).
A deterministic sample through several configs (cache-reused, never touching `place`), scored on field
coverage and inter-model agreement (no gold standard → agreement is a quality proxy). Findings:
- **`country_code`** (reconciliation-critical) ~95–100% across all models.
- **AAT typing** is the discriminating axis: **Llama-3.3-70B and Qwen3-32B-*thinking* reach 95%**
  (matching Gemini Flash); gpt-oss-120B 86%; non-thinking Qwen / Qwen2.5-72B ~70%. Reasoning is what
  lifts Qwen 70%→95% — but at ~80 s/entry it's too slow as a primary.

So the **generate → critique → repair** design:
- **Llama-3.3-70B = primary extractor** — fast (~8 s/entry), 95% AAT, best name extraction (Jaccard 0.97).
- **gpt-oss-120B = critic** — sees the source entry + Llama's record and flags what looks off (wrong type,
  historical→present-day country, a missed `—Also` place).
- **Qwen3-32B-thinking = repair** — re-does the flagged ~7–11% with full reasoning.
- Two independent model families concurring is the **per-record confidence signal** for the WHG ingest.

**Scrutiny (`process/scrutinise.py`).** Deterministic checks run first — chiefly: a coordinate is kept
only when the source actually printed it (`lat.`/`long.`), so the model can't supply a plausible-but-
unstated coordinate from world knowledge (validated: it flags exactly the inferred longitudes).

Each entry yields one **structured record per place** (Pydantic-validated): canonical name + variants;
the feature type as a **Getty AAT id** (closed enum); and the disambiguation context — country (+ present-
day ISO code), admin hierarchy, nearby places with bearing/distance, coordinates (DMS→decimal, *only when
printed*), population (`[{year,count}]` series), area, and a **`peoples` ethnography facet** (ethnonyms
like *Lumris, Numaris, Wends*, typed AAT 300191997). Multi-place entries split on "—Also …"; cross-refs
from `see_target`. The model fixes obvious OCR garbling in-pass, logged to `ocr_corrections`.

**Cost: $0 per token** — self-hosted (the spend is GPU-hours on our CRC allocation). Every result is
cached in `llm_cache` keyed by `(provider, model, prompt+schema sig, entry text)`, so re-runs and the A/B
never recompute. (For reference, `process/estimate_cost.py` costs the API route: the ~90k-entry /
~104k-place corpus would be ≈$84 batched on Claude/Gemini — which the free stack avoids.)

### 4. Reconcile against WHG — `process/reconcile.py`, `process/submit_reconcile_slurm.py`
```bash
python3 process/reconcile.py --seed-demo 8                      # demo (no extraction needed)
python3 process/reconcile.py --limit 200 --concurrency 24       # gateway backend (default), on CRC
python3 process/reconcile.py --backend api --concurrency 6      # public-API fallback (needs WHG_API_TOKEN)
python3 process/submit_reconcile_slurm.py                       # ingest + cascade as an htc CPU job
```

A **3-pass cascade**, each pass run only on the previous one's misses (precision first, then recall),
thresholding on `score`:

| Pass | `mode` | country | spatial |
|---|---|---|---|
| **1** exact | `exact` | strict (`ccodes=[cc]`) | — |
| **2** phonetic | `phonetic` (Symphonym KNN) | strict | — |
| **3** proximity | `phonetic` | dropped | `bounds` box around the **printed** coords |

Pass 3 lets borders/spellings change but **bounds the search spatially** — a renamed place is found
near its printed coordinates, never on the other side of the world (validated: *Luroe*→*Lurøy* via the
box). Demo: **8/8** (7 exact, 1 proximity). Matches get `whg_match_id`, `whg_score`, the pass that found
them, and a centroid.

**Two backends, same cascade** (`exact`/`phonetic`, country, and `bounds` are all honoured server-side;
never filter by `types`/`fclasses` — sparsely populated, tanks recall; threshold on `score`, not the
conservative `match`):
- **`gateway`** (default) — POST directly to the Pitt ES gateway's `/api/reconcile` on its
  **cluster-facing interface** `gazetteer-clus.crc.pitt.edu:9200` (`10.201.0.185`). That interface is a
  **direct local connection from CRC compute nodes** — no firewall, **no token** — and the response
  carries the centroid inline (`repr_point`), so there's no separate data-extension call. Fastest.
- **`api`** — the public `https://whgazetteer.org/reconcile` (W3C-style batched `{queries}`, `countries`,
  `.env` token, centroid via a second `extend` POST). The gateway proxies the same KNN behind it; works
  anywhere with internet.

**Transport:** the external service — not CRC — is the limiter, so it runs as a **single htc CPU `srun`**
with moderate `--concurrency` (not a GPU array). The gateway's *Internet*-facing interface
(`gazetteer.crcd.pitt.edu`) is firewalled to login nodes, but its *cluster*-facing interface (above)
reaches compute directly, so the fast path needs no firewall change.

### 5. Tables, maps, and the demo

A matching scan of **Volume V** (`data/pdf/gotw-v5.pdf`, with an OCR text layer) makes table and map
work tractable. `process/pdf_pages.py` maps printed page → PDF index from the header page numbers,
handling the offset drift (16→80) caused by ~70 unpaginated steel plates that defeat a constant offset.

- **Tables — vision-LLM into `table_data`** — Surya's layout model does **not** detect these unruled
  1856 tables, and prose OCR scrambles their cells, so `process/extract_tables.py` digitises them with a
  **vision** model — self-hosted **Qwen2.5-VL-72B** on the cluster (`--backend vllm`, free) or Gemini
  Flash (`--backend gemini`, ~$1.50 for the corpus). To bound cost it flags **candidate pages by
  numeric-row runs** in the column-ordered OCR (a table leaves a run of number-heavy lines; prose tops
  out ~2, tables score 6–9), then sends each to the model for structured `{title, header, rows}`.
  Validated: the *Madras* climate + districts tables digitise cleanly (Qwen2.5-VL even keeps the print's
  `·` decimals). ⚠️ Note the two vision models *disagreed on some digits* (e.g. an area `8,700` vs
  `3,700`) — table digit accuracy is error-prone for any single model, so tables warrant the same
  two-reader scrutiny as the prose.
- **Maps** — `process/extract_maps.py` finds illustration plates (ink-filtering out blank/stamp pages
  and marbled endpapers) and vision-classifies them. **Volume V contains no cartographic maps**: its 8
  steel plates are all city/landscape views (Magdeburg, Malta, Melrose Abbey, Mytelene, New York Bay,
  Padua, Patras). The tool writes an illustration manifest (titles + page provenance) and would
  crop/export any genuine maps — more likely to appear in other volumes.
- **All seven volumes acquired + OCR'd.** Source them as the **1856 first edition = volumes 1–7** on
  [HathiTrust 011407465](https://catalog.hathitrust.org/Record/011407465); the **undated 8–14 set** on
  the same record is a *different edition* (its v14 reproduces 1856 v7's Article II) and must not be
  mixed in. `process/pdf_coverage.py` reports each volume's head-word range so the seven tile A–Z without
  overlap or gap. (Tables/maps recover from the 600 dpi page images directly; the searchable per-volume
  PDF built by `process/build_pdf.py` is now mainly an archival artifact, since OCR reads the images.)
- **MapLibre demo** — a static GitHub Pages UI plotting the extracted, reconciled places with rich
  popups, live at [docuracy.github.io/GOTW/map.html](https://docuracy.github.io/GOTW/map.html), built via
  `process/export_geojson.py` → `docs/places.geojson`. (The current demo is from an early sample; it will
  be regenerated from the full self-hosted Llama extraction + 3-pass reconciliation.)

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
  reconciliation indices, the Elasticsearch gateway, and the authority-gazetteer framework.
- **Pitt Center for Research Computing (CRC)** — the GPU cluster that hosts all OCR and LLM stages.
- **Open models** (self-hosted via vLLM): [Surya](https://github.com/datalab-to/surya) OCR,
  Meta **Llama-3.3-70B**, OpenAI **gpt-oss-120B**, Alibaba **Qwen3** / **Qwen2.5-VL**.
- Place types use the Getty **Art & Architecture Thesaurus** (AAT), made available under the
  [ODC Attribution License](https://www.getty.edu/research/tools/vocabularies/license.html).
