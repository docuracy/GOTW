# Gazetteer of the World → WHG: a demonstrator, and what it teaches us

This note accompanies the GOTW demonstrator ([docuracy.github.io/GOTW/map.html](https://docuracy.github.io/GOTW/map.html)).
It is in three parts: **(1)** a guided tour of the UI and how the data behind it was produced, for talking
through at the team meeting; **(2)** lessons for rebuilding the WHG dataset-explorer tool — including running
**Symphonym embeddings client-side via ONNX** to cut server dependency; and **(3)** an honest assessment of how
far this pipeline adapts to other historical print gazetteers, and where it doesn't.

---

## Part 1 — A guided tour

### 1a. The explorer UI

Everything below is **static** — served from GitHub Pages, **no application server**, no database backend.

- **Map (MapLibre + PMTiles).** Reconciled places are vector tiles in a single `.pmtiles` file, read by
  *viewport* over HTTP range requests. At low zoom the map draws a **density heatmap** (weighted by a
  clustered `point_count` baked in at tile-build time); zooming in **cross-fades to individual circles**.
- **Light tiles, detail on demand.** Tiles carry only `id`/`name`/`fclass` (so they stay tiny); clicking a
  place **fetches its full record** from a *sharded* JSON store (`detail/<id%N>.json`, cached per shard). The
  popup shows the extracted record — name, AAT feature type, country, admin hierarchy, population time-series,
  WHG match — plus a **"View source page"** deep-link to the exact HathiTrust scan and a **"Read full entry"**
  button.
- **Whole-corpus reader (modal).** "Read full entry" opens a continuous, **lazy-loaded transcription of all
  seven volumes**, scrolled to the clicked entry; you can keep scrolling across volume boundaries. The DOM is
  **windowed** (only ~7 chunks kept live) so memory stays bounded over 91k entries. Tables render inline from
  structured data.
- **Search — three tiers, all in-browser:**
  - **Full text** — SQLite **FTS5 queried over HTTP range requests** (`sql.js-httpvfs`); only the DB pages a
    query touches are fetched, so an 85 MB index loads nothing up front.
  - **Fuzzy names** — a trigram index queried as overlapping 3-grams (typo/OCR-tolerant: *Bordaux*→Bordeaux).
  - **Phonetic / cross-script** (opt-in toggle) — **Symphonym v7 running in the browser** via onnxruntime-web:
    the query is embedded locally and cosine-matched against precomputed headword embeddings
    (*Moskva*→Moscow, *Constantinopel*→Constantinople).

### 1b. How the data was produced

A six-stage pipeline turns 7 volumes of 1856 print into typed, geolocated, linked records:

1. **OCR** — public-domain HathiTrust page images run through **Surya** (layout-aware) on the Pitt CRC GPU
   cluster. Two-column reading order is reconstructed from line geometry; **unruled statistical tables are
   detected by digit-density and routed out** of the prose stream (recorded as bbox markers) for separate
   vision-LLM digitisation.
2. **Parse / segment** — a rule-based parser splits the OCR stream into one record per headword using the
   *typographic* conventions (ALL-CAPS headwords, cross-reference forms, standalone vs inline headings),
   hardened against real failures (library stamps, hyphen-wrapped headwords, running heads). ~98% headword
   agreement against an independent reference.
3. **Human review** — residual hard cases go to a small local UI; decisions live in a **signature-keyed
   sidecar** so they survive re-parses and DB rebuilds.
4. **Typed extraction** — every entry is run through a **self-hosted Llama-3.3-70B** (vLLM, sharded across
   GPUs) with **schema-constrained output** and a **closed Getty-AAT enum**, producing one structured record
   per place (name + variants, AAT type, country/ISO, admin, coordinates *only when printed*, population
   series, ethnonyms).
5. **Reconcile** — a 3-pass cascade against the WHG gateway: exact → **phonetic (Symphonym KNN)** → proximity
   (spatially bounded), precision-first.
6. **Publish** — export to PMTiles + the sharded detail store + the chunked reader + the FTS/Symphonym search
   indexes; deploy to Pages.

The **Volume VII Appendix** (a bidirectional ancient↔modern concordance) is retained as a 27k-toponym authority.

---

## Part 2 — Lessons for the WHG dataset-explorer rebuild

The demonstrator is, in effect, a **server-less dataset explorer**. Several of its choices transfer directly.

### 2a. Static + range-requested data scales surprisingly far
A rich explorer — map, full-text search, and a whole-corpus reader — runs with **no backend** by leaning on
three "fetch only what you need" formats:
- **PMTiles** for the map (one file, range-read by viewport),
- **SQLite/FTS5 over HTTP range requests** for full-text search (no search server),
- **chunked JSON** for the reader (lazy, windowed).

For WHG-explorer use cases this means **lower ops cost, no query backend to scale, trivial CDN caching**, and
the same artifacts work offline. The pattern handled ~100k features and ~91k full-text docs comfortably.

### 2b. Client-side Symphonym via ONNX — reduce server dependency for queries
The Symphonym **Student encoder exports to an 8 MB int8 ONNX** that runs in the browser (onnxruntime-web),
with **128-d embeddings precomputed for the corpus** (here ~91k headwords → 12 MB int8). The browser embeds
the user's query and does cosine-KNN locally. Implications for the explorer:
- **Phonetic / cross-script name search without a gateway round-trip** — the KNN that currently needs the
  Elasticsearch/Symphonym service can be served **client-side** for explorer-style browsing, reducing load on
  and dependency upon the gateway.
- **Verified faithful**: int8 encoder parity `cos(int8,fp32) = 0.9997`; preprocessing (char-vocab tokenisation,
  Unicode-range script detection, **`navigator.language` conditioning**) ported exactly from the reference
  `inference.py`.
- **Costs/caveats**: a one-time, opt-in, cached download (~30 MB: encoder + embeddings + ort-web wasm); the
  query tokeniser must track the model's preprocessing; and it is a **candidate retriever** — geographic /
  type constraints still belong server-side or in a second pass. Good division of labour: **browser for
  fast fuzzy/phonetic candidate retrieval, server for authoritative reconciliation.**

### 2c. GitHub-Pages / static-hosting gotchas worth codifying
- **Gzip breaks range requests.** Pages gzips `.sqlite`/octet-stream, so byte-ranges index the *compressed*
  stream → garbage. Serving the DB as **`.png`** (Pages never gzips images) keeps ranges raw. Same applies to
  any range-read binary.
- **Keep big data out of git history.** Deploy via the **GitHub Actions Pages artifact**; stage large
  generated files in a **Release** that the workflow pulls at build time. Same-origin, GitHub-hosted, no LFS,
  no history bloat. Regenerate → re-publish, never re-commit.
- **Self-host every library** (maplibre, pmtiles, sql.js-httpvfs, onnxruntime-web) for robustness and offline
  use; the only unavoidable third-party runtime call is the basemap *tile* service.

### 2d. Other transferable UI patterns
- **Light tiles + lazy detail fetch** (don't put rich popups in the tiles).
- **DOM windowing** for any long, continuously-scrolled view.
- **Tiered search** (exact/prefix → fuzzy → phonetic), each cheap and composable, with results sharing one
  locator so any hit can deep-link into the same detail/reader view.
- **Provenance is a first-class feature**: per-record source-page deep links (derived here from the OCR's
  image-sequence markers) and full transcription access.

### 2e. In-context error reporting — closing the curation loop
Every popup and reader entry carries a **⚑ Report** link. In the demonstrator this is implemented with
**zero backend**, leaning on GitHub:

- the link opens a **pre-filled GitHub New-Issue** (an *Issue Form* template) — no token/OAuth/proxy in
  the UI; GitHub handles identity, so **anyone with a GitHub account** can submit;
- the template **auto-applies an `explorer-report` label** (curators filter on it), and a small
  **labeler workflow** reads the ticked error type(s) and adds per-type labels
  (`ocr`, `tables`, `geocoding`, `aat-typing`, `over-split`, `under-split`, `merged`, `other`) — these map
  one-to-one onto pipeline stages, so a flag routes to the right fix;
- each issue embeds a **machine-readable `meta` block** (`{eid, vol, page, headword}`) plus a
  **`?entry=<eid>` deep link** back into the explorer, so (a) a coding agent can pull `label:explorer-report`
  via the API, parse, and **cluster systemic problems** (e.g. OCR flags concentrated on a page range → targeted
  re-OCR; geocoding flags by country → reconciliation bias), and (b) a triager clicks straight back to the
  exact entry. Spot error → report → label/triage → land on the entry, with no moderation service to run.

> **Limitation to flag for site-wide use: it requires a GitHub account.** That's fine — even ideal — for a
> *developer/curator-facing demonstrator* and for technically-minded contributors, because it reuses GitHub's
> auth, notifications, labels, and Projects for free. But it is a real barrier for the **general public** and
> non-technical contributors, who shouldn't need to create a GitHub account to report that a place is in the
> wrong country.

**A WHG-Django site-wide model.** Because WHG is already a Django platform, a production, site-wide reporter
should be **Django-native** rather than GitHub-bound:

- **Lower barrier:** use WHG's own login, or allow **anonymous / email-only** reports (with light anti-spam),
  so no third-party account is required.
- **Integrated data model:** a `Correction`/`Report` model linked to the WHG place/record by its **stable id**,
  carrying the *same error taxonomy and structured fields* as above — so the two systems are interoperable and a
  report means the same thing whichever route it came in by.
- **In-platform triage:** filter/moderate by type, dataset, and status in Django admin or a curator dashboard;
  surface a discreet "reported" badge on affected records; support assignment and resolution states.
- **Same downstream analysis:** expose the reports via an API/export so the **clustering agent** (and, if
  desired, a GitHub-Issues bridge) can still look for systemic fixes — the analysis layer is unchanged.

In short: **the GitHub-Issues route is the zero-infrastructure MVP and stays excellent for the dev/curation
loop; the Django model is the inclusive, site-wide form.** Share the taxonomy and the structured payload between
them and reports flow either way.

---

## Part 3 — Adaptability and limitations for other print gazetteers

### What generalises (the harness)
The **shape** of the pipeline — OCR → parse → typed extraction → reconcile → publish — is reusable, and a lot
of the machinery is genuinely source-agnostic:
- the **self-hosted extraction harness** (sharded vLLM, schema-constrained output, closed-vocabulary typing,
  resumable/idempotent), and the **AAT typing** + **reconciliation cascade**;
- the **human-QA pattern** (signature-keyed decisions, suspect work-list);
- the entire **static explorer stack** (PMTiles export, chunked reader, FTS + trigram + **Symphonym-ONNX**
  search, the Pages/Release deploy);
- the **provenance** plumbing (source-page deep links from page markers).

### What is irreducibly bespoke (layout & content analysis)
**Every print gazetteer has its own typography and structure, and that layer must be re-analysed and re-built
each time.** For GOTW this meant: distinguishing standalone display headings from inline minor entries;
healing hyphen-wrapped headwords; scrubbing recurrent library stamps; classifying several cross-reference
forms; and — because the layout model can't see **unruled** 1856 tables — detecting tables and full-page
plates from the **line geometry** (persistent vertical gutters between narrow, populated columns; sparse/blank
pages by ink + text density) and routing them out of the reading order. None of that transfers verbatim. A
different work will have different column rules, delimiters, running-head formats, plates, abbreviation
systems, and table styles.

> **Lesson learned mid-project — prefer the VLM as the page-type detector.** The geometry heuristics are
> precise and content-agnostic (they replaced an earlier digit-density rule that only ever caught *numeric*
> tables), but they **under-recall**: a cheap **low-resolution full-page pass by a self-hosted VLM**
> (Qwen2.5-VL) classifying each page *prose / plate / blank* + counting embedded tables/figures found tables
> on ~21% of main-volume pages vs the geometry detector's ~14% — i.e. geometry **missed ~⅓ of table pages** —
> and cleanly separated real illustration plates (maps/portraits) from blanks, which the geometry plate
> detector lumped together. The durable pattern: **cache the OCR line-geometry per page** so layout decisions
> can be re-derived on CPU without re-OCR, use the geometry detector only to *route cell/label text out of the
> prose* (which a page-level VLM can't localise), and let the **VLM triage be the authoritative selector** for
> the expensive high-res table-digitisation and plate-export passes. (Tables are digitised to a data-first
> column/row schema by the same self-hosted VLM — no external API.)

**The honest rule of thumb:** budget for a **layout-and-content analysis phase up front** for each new source,
then assemble a *precise* workflow from the reusable components. The pipeline is a **kit, not a turnkey
ingester** — the kit is large and good, but the segmentation/table/typing rules are custom carpentry.

### Known limitations (carry into planning)
- **OCR quality varies by scan**; layout reconstruction and table detection are heuristic and need spot-checks.
- **Extraction has a failure tail** — long essay-style entries occasionally return malformed/truncated JSON
  (~9% on the hardest tranche here); plan a **repair pass** (higher token budget, or a critic + reasoning
  re-do) rather than assuming 100% yield.
- **Reconciliation precision depends on disambiguation context** (country, printed coordinates); phonetic and
  fuzzy matching widen recall but human QA remains necessary for the hard residue. The strongest lever here is
  **hierarchical containment**: resolve each place's admin parents top-down to WHG ids that *have geometry*,
  then constrain the (fuzzy) leaf match to inside that parent via the API's **`contained_in` + `relation`**
  spatial filter — this stops a phonetic look-alike matching on the far side of the country (e.g. an Essex
  parish landing in Ireland). `ccodes` is a reliable proxy for the country level; only sub-country parents
  need id resolution.
- **A cheap full-page VLM triage can fail systematically on a minority of pages** (here ~⅓ failed schema-
  constrained generation and did *not* recover on a lighter-load re-run) — budget a diagnostic/repair loop
  (relax the schema, log the failing responses) before trusting it as the sole detector.
- **Non-Latin / multilingual sources** shift the balance — this is exactly where Symphonym's cross-script
  matching earns its place, and where client-side ONNX embeddings could most reduce gateway load.
- **Infrastructure assumptions**: OCR and LLM stages here assume a Slurm/vLLM GPU cluster; the explorer
  assumes only static hosting.

---

## Part 4 — Extension ideas

Concrete next steps, in rough order of value-to-effort:

1. **Richer AAT typing of attributes and secondary features.** The extraction currently types the *primary*
   place. Prompt it to also categorise, against AAT, the **produce and manufactures** named in an entry
   (e.g. wine, tallow, woollens → AAT material/product concepts) and to emit secondary typed features that the
   prose mentions — **religions and religious buildings, battlefields / battle sites, fortifications, ports
   and harbours**, etc. These become facets for search/filtering and additional reconcilable sub-entities,
   turning each essay into several typed records rather than one.

2. **Fix the low-zoom heatmap "checkerboard".** Tippecanoe quantises point locations to the tile grid, so at
   low zoom the heatmap shows a regular checkerboard rather than smooth density. Mitigate purely in the
   **front-end style** — increase the heatmap `intensity`/`radius` and soften the colour ramp at low zoom
   (zoom-interpolated `heatmap-radius`/`heatmap-weight`), and/or add a small deterministic jitter to point
   positions in the light tiles — so density reads continuously without re-quantising the data.

3. **Condense the top bar into a single top-left control pane.** Replace the wide header with a compact
   panel: a short **title**; an **ⓘ info button** opening a modal with the full project description + source
   (RGS 1856, the self-OCR rationale, credits/origin); the **search input with its `phonetic` toggle**; and a
   **dropdown to jump the reader** to any volume, any index letter within a volume, or any **Appendix**. A
   second dropdown could index the **identified plates** (now that triage classifies them with `plate_kind`
   and titles) and open them in the reader/lightbox — effectively a "list of illustrations/maps" per volume.

---

*Generated from the GOTW project; see `README.md` for the full pipeline and `process/` for the scripts behind
each stage (`ocr_pages.py`, `parse_ocr.py`, `extract.py`, `reconcile.py`, `triage_pages.py`,
`extract_tables.py`, `export_plates.py`, `export_geojson.py`, `build_tiles.sh`, `export_reader.py`,
`build_search_db.py`, `export_symphonym_onnx.py`, `build_symphonym_index.py`).*
