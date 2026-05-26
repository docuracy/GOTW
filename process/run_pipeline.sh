#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — end-to-end GOTW pipeline orchestrator, runnable ENTIRELY on
# the Pitt CRC. Pull the repo to /vast/ishi/gotw, then from a login node:
#
#     tmux new -s gotw                      # so it survives disconnects
#     process/run_pipeline.sh --list        # show stages
#     process/run_pipeline.sh --dry-run     # print every command/sbatch, submit nothing
#     process/run_pipeline.sh               # run all stages, in order
#     process/run_pipeline.sh --from vlm    # resume from a stage
#     process/run_pipeline.sh --only reconcile --reconcile-reset   # re-run one stage
#
# It only SUBMITS jobs and POLLS sacct — no heavy compute on the login node
# (allowed; same as squeue). Every stage runs as a Slurm job (srun/sbatch); it
# waits for each to finish and aborts loudly on failure. Stages:
#
#   ocr  merge  parse  extract  ingest  vlm  reconcile  export  publish
#
# NOTE: assembled from the per-stage scripts + the invocations verified in the
# 2026-05 build. It has NOT been run end-to-end as one script — on first use,
# go stage-by-stage (`--only <stage>`) or start with `--dry-run`. See README +
# WHG-LESSONS.md for the rationale behind each stage.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

# ── config (override via env) ────────────────────────────────────────────────
REPO="${REPO:-/vast/ishi/gotw}"
VOLS="${VOLS:-1 2 3 4 5 6 7}"
DB="${DB:-$REPO/data/gotw_seg.sqlite}"
LOGS="$REPO/logs"
CONDA_SH="${CONDA_SH:-/ihome/ishi/stg135/miniconda3/etc/profile.d/conda.sh}"
WHG_ENV="${WHG_ENV:-/ihome/ishi/stg135/miniconda3/envs/whg}"      # Surya, PIL, tippecanoe, sqlite
VLLM_ENV="${VLLM_ENV:-/vast/ishi/envs/vllm}"                      # vLLM, requests, pydantic
HF_CACHE="${HF_CACHE:-/vast/ishi/hf_cache}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"
LLAMA_OUT="${LLAMA_OUT:-$REPO/llama_seg}"                         # extract shard JSONLs land here
LLAMA_GLOB="${LLAMA_GLOB:-llama_seg/llama.*.jsonl}"               # MUST match LLAMA_OUT naming
SYM_DIR="${SYM_DIR:-}"                                            # Symphonym hf/ dir ON THE CRC; empty → skip phonetic index
RECONCILE_RESET="${RECONCILE_RESET:-0}"                           # 1 = clear all prior matches before reconcile (re-run)
DO_PUBLISH="${DO_PUBLISH:-0}"                                     # publish stage opt-in (needs gh + git creds on CRC)

ALL_STAGES=(ocr merge parse extract ingest vlm reconcile export publish)
DRY=0; FROM=""; TO=""; ONLY=""

# ── helpers ──────────────────────────────────────────────────────────────────
log(){ printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
die(){ printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
run(){ if [ "$DRY" = 1 ]; then echo "  + $*"; else eval "$@"; fi; }

# wait_job <cluster> <jobid> <label> — poll sacct until terminal; nonzero on non-COMPLETED
wait_job(){
  local cl="$1" jid="$2" lbl="$3" st
  [ "$DRY" = 1 ] && { echo "  (dry-run: would wait on $cl job $jid — $lbl)"; return 0; }
  log "waiting on $lbl ($cl job $jid)…"
  while :; do
    st=$(sacct -M "$cl" -j "$jid" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      COMPLETED) echo "  $lbl: COMPLETED"; return 0;;
      FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL) echo "  $lbl: $st"; return 1;;
      *) sleep 30;;
    esac
  done
}

# submit_wait <cluster> <sbatch-file> <label> — sbatch, capture jobid, wait
submit_wait(){
  local cl="$1" sb="$2" lbl="$3" jid
  if [ "$DRY" = 1 ]; then echo "  + sbatch $sb   ($cl — $lbl)"; return 0; fi
  jid=$(sbatch --parsable "$sb" 2>&1 | tail -1 | cut -d';' -f1)
  [[ "$jid" =~ ^[0-9]+$ ]] || die "submit failed for $lbl: $jid"
  wait_job "$cl" "$jid" "$lbl" || die "$lbl failed — see $LOGS"
}

# write an htc CPU sbatch that runs <cmds> in <env>
htc_sbatch(){  # $1=name $2=env $3=time $4=cmds  → echoes path
  local name="$1" env="$2" tm="$3" cmds="$4"
  local f="$LOGS/${name}.sbatch"
  cat > "$f" <<EOF
#!/bin/bash
#SBATCH --job-name=gotw-$name
#SBATCH --output=$LOGS/${name}-%j.out
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --qos=htc-htc-s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=$tm
set -eo pipefail
source $CONDA_SH; conda activate $env
cd $REPO
$cmds
echo "${name}_DONE"
EOF
  echo "$f"
}

# ── stages ───────────────────────────────────────────────────────────────────
stage_ocr(){            # Surya OCR per volume (GPU array jobs), caching geometry
  log "STAGE ocr — Surya OCR (+ --save-geom) per volume"
  local jids=() v jid
  for v in $VOLS; do
    if [ "$DRY" = 1 ]; then echo "  + python3 process/submit_ocr_slurm.py --img-dir img/v$v --vol v$v --save-geom"; continue; fi
    jid=$(python3 process/submit_ocr_slurm.py --img-dir "$REPO/img/v$v" --vol "v$v" --save-geom 2>&1 | grep -oE 'job [0-9]+' | grep -oE '[0-9]+' | head -1)
    [[ "$jid" =~ ^[0-9]+$ ]] || die "ocr submit failed for v$v"
    jids+=("$jid"); echo "  v$v → gpu job $jid"
  done
  for jid in "${jids[@]:-}"; do [ -n "$jid" ] && { wait_job gpu "$jid" "ocr $jid" || die "ocr failed"; }; done
}

stage_merge(){          # per-page OCR → merged volume txt
  log "STAGE merge — per-page OCR → txt/gotw-v<N>-ocr.txt"
  local c=""; for v in $VOLS; do c+="python3 process/ocr_pages.py --merge --out-dir ocr/v$v --out txt/gotw-v$v-ocr.txt"$'\n'; done
  submit_wait htc "$(htc_sbatch merge "$WHG_ENV" 01:00:00 "$c")" "merge"
}

stage_parse(){          # rule-based segmentation → fresh gotw_seg.sqlite
  log "STAGE parse — segment volumes into $DB"
  local c="rm -f $DB"$'\n'; for v in $VOLS; do c+="python3 process/parse_ocr.py txt/gotw-v$v-ocr.txt --volume v$v --db $DB"$'\n'; done
  submit_wait htc "$(htc_sbatch parse "$WHG_ENV" 01:00:00 "$c")" "parse"
}

stage_extract(){        # self-hosted Llama-3.3-70B typed extraction (GPU, sharded)
  log "STAGE extract — Llama typed extraction → $LLAMA_OUT"
  if [ "$DRY" = 1 ]; then echo "  + python3 process/submit_extract_slurm.py --db $DB --out-dir $LLAMA_OUT --nshards 8"; return; fi
  # submit_extract submits N shard jobs; capture all ids and wait
  local out jids
  out=$(python3 process/submit_extract_slurm.py --db "$DB" --out-dir "$LLAMA_OUT" --nshards 8 2>&1); echo "$out"
  jids=$(echo "$out" | grep -oE 'job [0-9]+' | grep -oE '[0-9]+')
  [ -n "$jids" ] || die "extract submit produced no job ids"
  local jid; for jid in $jids; do wait_job gpu "$jid" "extract $jid" || die "extract failed"; done
}

stage_ingest(){         # extraction JSONLs → place rows
  log "STAGE ingest — $LLAMA_GLOB → place rows"
  submit_wait htc "$(htc_sbatch ingest "$VLLM_ENV" 00:30:00 \
    "python3 process/extract.py --ingest '$LLAMA_GLOB' --db $DB")" "ingest"
}

stage_vlm(){            # serve Qwen2.5-VL once → triage, tables, plates
  log "STAGE vlm — serve Qwen2.5-VL → triage + tables + plates"
  local f="$LOGS/vlm.sbatch" port=18994
  cat > "$f" <<EOF
#!/bin/bash
#SBATCH --job-name=gotw-vlm
#SBATCH --output=$LOGS/vlm-%j.out
#SBATCH --clusters=gpu
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -eo pipefail
source $CONDA_SH; conda activate $VLLM_ENV; module load cuda/12.8.0
RUN=/tmp/gotw_vlm_\${SLURM_JOB_ID}; mkdir -p "\$RUN"
export HF_HOME=$HF_CACHE TMPDIR="\$RUN" XDG_CACHE_HOME="\$RUN" VLLM_CACHE_ROOT="\$RUN"
cd $REPO
SRV=$LOGS/vlm-srv-\${SLURM_JOB_ID}.log
vllm serve $VLM_MODEL --host 127.0.0.1 --port $port --max-model-len 32768 \\
  --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":1,"video":0}' --served-model-name qwen2.5-vl > \$SRV 2>&1 &
SRVPID=\$!; trap "kill \\\$SRVPID 2>/dev/null||true" EXIT
for i in \$(seq 1 240); do grep -qa "Application startup complete" \$SRV && break; sleep 5; done
grep -qa "Application startup complete" \$SRV || { echo SERVER_FAILED; tail -25 \$SRV; exit 1; }
export TABLE_BACKEND=vllm TABLE_VL_BASE=http://127.0.0.1:$port/v1 TABLE_VL_MODEL=qwen2.5-vl
echo "--- triage (every page: prose/plate/blank + table/figure counts) ---"
for v in $VOLS; do python -u process/triage_pages.py --img-dir img/v\$v --volume v\$v --db $DB --concurrency 32; done
echo "--- merge page_triage shards into triage.sqlite (single-shard here, but keep the artifact) ---"
python -u - <<'PYEOF'
import sqlite3
src=sqlite3.connect("$DB"); out=sqlite3.connect("data/triage.sqlite")
out.execute("DROP TABLE IF EXISTS page_triage")
out.execute("CREATE TABLE page_triage(volume TEXT,idx INTEGER,type TEXT,n_tables INTEGER,n_images INTEGER,plate_kind TEXT,created_at TEXT,PRIMARY KEY(volume,idx))")
rows=list(src.execute("SELECT volume,idx,type,n_tables,n_images,plate_kind,created_at FROM page_triage"))
out.executemany("INSERT OR IGNORE INTO page_triage VALUES(?,?,?,?,?,?,?)", rows); out.commit()
print("triage.sqlite rows:", out.execute("SELECT COUNT(*) FROM page_triage").fetchone()[0])
PYEOF
echo "--- tables (union via triage; --skip-extracted only does NEW pages on a re-run) ---"
for v in $VOLS; do python -u process/extract_tables.py --from-triage --skip-extracted --triage-db data/triage.sqlite --db $DB --volume v\$v --img-dir img/v\$v --ocr txt/gotw-v\$v-ocr.txt; done
kill \$SRVPID 2>/dev/null||true; sleep 5
echo "--- plates (CPU, from triage) ---"
conda activate $WHG_ENV
rm -rf docs/plates; mkdir -p docs/plates
for v in $VOLS; do python -u process/export_plates.py --from-triage --orient --max-px 2400 --triage-db data/triage.sqlite --db $DB --volume v\$v --img-dir img/v\$v --ocr txt/gotw-v\$v-ocr.txt --out docs/plates; done
echo vlm_DONE
EOF
  submit_wait gpu "$f" "vlm (triage+tables+plates)"
}

stage_reconcile(){      # hierarchy-aware containment cascade (htc, gateway)
  log "STAGE reconcile — containment cascade via WHG gateway$([ "$RECONCILE_RESET" = 1 ] && echo ' (RESET prior matches)')"
  local reset=""
  [ "$RECONCILE_RESET" = 1 ] && reset="python3 -c \"import sqlite3;c=sqlite3.connect('$DB');c.execute(\\\"UPDATE place SET whg_match_id=NULL,whg_score=NULL,lat=NULL,lon=NULL,recon_pass=NULL,reconciliation=NULL,status='extracted' WHERE status IN ('reconciled','unmatched')\\\");c.commit();print('reset',c.total_changes)\""$'\n'
  # backup, then reconcile (no-ingest: ingest is its own stage). submit_reconcile_slurm already does the node-local copy + cascade.
  run "cp $DB $REPO/data/gotw_seg.prebak-recon-\$(date +%s).sqlite"
  if [ -n "$reset" ]; then submit_wait htc "$(htc_sbatch recon-reset "$VLLM_ENV" 00:15:00 "$reset")" "reconcile-reset"; fi
  if [ "$DRY" = 1 ]; then echo "  + python3 process/submit_reconcile_slurm.py --no-ingest"; return; fi
  local jid; jid=$(python3 process/submit_reconcile_slurm.py --no-ingest 2>&1 | grep -oE 'job [0-9]+' | grep -oE '[0-9]+' | head -1)
  [[ "$jid" =~ ^[0-9]+$ ]] || die "reconcile submit failed"
  wait_job htc "$jid" "reconcile $jid" || die "reconcile failed"
}

stage_export(){         # places+geometry pmtiles, detail+geocoded, reader, FTS, (phonetic)
  log "STAGE export — places+geometry pmtiles + detail + reader + search indexes"
  local c="process/build_tiles.sh"$'\n'
  c+="python3 process/export_reader.py --db $DB --out-dir docs/reader --plates-manifest docs/plates/manifest.json"$'\n'
  c+="python3 process/build_search_db.py --db $DB --out docs/search/gotw-fts.sqlite.png"$'\n'
  if [ -n "$SYM_DIR" ]; then
    c+="python3 process/export_symphonym_onnx.py --sym $SYM_DIR --out docs/search/symphonym.onnx"$'\n'
    c+="python3 process/build_symphonym_index.py --sym $SYM_DIR --db docs/search/gotw-fts.sqlite.png --out-dir docs/search"$'\n'
  else
    c+="echo 'SYM_DIR unset → skipping phonetic ONNX/index (set SYM_DIR to a Symphonym hf/ dir on the CRC)'"$'\n'
  fi
  submit_wait htc "$(htc_sbatch export "$WHG_ENV" 01:00:00 "$c")" "export"
}

stage_publish(){        # release assets + push docs (needs gh + git creds on CRC)
  log "STAGE publish — site-assets release + push docs (Pages)"
  [ "$DO_PUBLISH" = 1 ] || { echo "  (publish is opt-in; re-run with --publish / DO_PUBLISH=1 once gh+git creds are set on the CRC)"; return 0; }
  run "cd $REPO && process/publish_assets.sh"
  run "cd $REPO && git add docs/places.pmtiles docs/detail docs/search/geocoded.json && git commit -m 'Rebuild map data' && git push origin main"
}

# ── arg parsing ──────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do case "$1" in
  --list) printf 'stages: %s\n' "${ALL_STAGES[*]}"; exit 0;;
  --dry-run) DRY=1;;
  --from) FROM="$2"; shift;;
  --to) TO="$2"; shift;;
  --only) ONLY="$2"; shift;;
  --reconcile-reset) RECONCILE_RESET=1;;
  --publish) DO_PUBLISH=1;;
  -h|--help) sed -n '2,40p' "$0"; exit 0;;
  *) die "unknown arg: $1";;
esac; shift; done

command -v sbatch >/dev/null || die "sbatch not found — run this on the Pitt CRC"
mkdir -p "$LOGS"; cd "$REPO" || die "no repo at $REPO"

# build the ordered list of stages to run
selected=()
started=0; [ -n "$FROM" ] || started=1
for s in "${ALL_STAGES[@]}"; do
  [ -n "$ONLY" ] && { [ "$s" = "$ONLY" ] && selected+=("$s"); continue; }
  [ "$s" = "$FROM" ] && started=1
  [ "$started" = 1 ] && selected+=("$s")
  [ -n "$TO" ] && [ "$s" = "$TO" ] && break
done
[ ${#selected[@]} -gt 0 ] || die "no stages selected (check --from/--to/--only names; see --list)"

log "GOTW pipeline — repo=$REPO db=$DB $([ "$DRY" = 1 ] && echo '(DRY-RUN)')"
echo "stages to run: ${selected[*]}"
for s in "${selected[@]}"; do "stage_$s"; done
log "pipeline: done (${selected[*]})"
