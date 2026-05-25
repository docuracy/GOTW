#!/usr/bin/env python3
"""Run the hierarchy-aware reconciliation cascade on CRC as a single htc CPU job (Slurm).

Reconciliation queries the external WHG API (internet-reachable from compute), so the API —
not CRC — is the throughput limiter: one CPU node with moderate `--concurrency` is right, and
sharding across nodes would only hammer the API / risk rate-limits. Optionally runs the JSONL
ingest first (extraction shard results → place rows), then the cascade. Resumable: re-running
skips reconciled places; the cache/ingest is idempotent.

Run on a CRC login node, with the DB + repo + token on the cluster:

    python3 process/submit_reconcile_slurm.py --dry-run
    python3 process/submit_reconcile_slurm.py            # ingest llama JSONLs then reconcile
    python3 process/submit_reconcile_slurm.py --no-ingest --limit 500   # reconcile only
"""
from __future__ import annotations
import argparse, os, subprocess
from pathlib import Path

_CONDA_SH = os.environ.get("CONDA_SH", "/ihome/ishi/stg135/miniconda3/etc/profile.d/conda.sh")
_ENV = os.environ.get("RECON_ENV", "/vast/ishi/envs/vllm")   # has requests + pydantic
_TOKEN_FILE = os.environ.get("RECON_TOKEN_FILE", "$HOME/.gotw_env")


def build_sbatch(*, db, repo, ingest_glob, concurrency, threshold, radius_km, limit, wall, no_hierarchy):
    log_dir = Path(repo) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=gotw-reconcile",
        f"#SBATCH --output={log_dir}/reconcile-%j.out",
        f"#SBATCH --error={log_dir}/reconcile-%j.err",
        "#SBATCH --clusters=htc",
        "#SBATCH --partition=htc",
        "#SBATCH --qos=htc-htc-s",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=4", "#SBATCH --mem=16G",
        f"#SBATCH --time={wall}",
        "",
        "set -eo pipefail",
        f"source {_CONDA_SH}",
        f"conda activate {_ENV}",
        f"set -a; source {_TOKEN_FILE}; set +a    # WHG_API_TOKEN",
        f"cd {repo}",
        "",
        "# Work on a NODE-LOCAL copy: SQLite write-locking is unreliable over the /vast network FS",
        "# ('database is locked'). Copy in, mutate locally, copy back on success.",
        f'LDB="/tmp/gotw_recon_${{SLURM_JOB_ID}}.sqlite"',
        f'trap \'rm -f "$LDB"\' EXIT',
        f'echo "copying DB to node-local $LDB"; cp {db} "$LDB"',
    ]
    if ingest_glob:
        lines += ["echo '--- ingest shard JSONLs -> place rows ---'",
                  f'python -u process/extract.py --ingest \'{ingest_glob}\' --db "$LDB"']
    recon = ["echo '--- hierarchy-aware reconciliation cascade ---'",
             "python -u process/reconcile.py \\",
             '    --db "$LDB" \\',
             f"    --concurrency {concurrency} --threshold {threshold} --radius-km {radius_km}"]
    if no_hierarchy:
        recon[-1] += " \\\n    --no-hierarchy"
    if limit:
        recon[-1] += f" \\\n    --limit {limit}"
    lines += recon
    lines += [f'echo "copying reconciled DB back to {db}"; cp "$LDB" {db}']
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/vast/ishi/gotw/data/gotw_seg.sqlite")   # table-aware re-OCR'd dataset
    ap.add_argument("--repo", default="/vast/ishi/gotw")
    ap.add_argument("--ingest-glob", default="llama_seg/llama.*.jsonl",       # fresh re-extract output
                    help="ingest these shard JSONLs first (relative to --repo); '' or --no-ingest to skip")
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--no-hierarchy", action="store_true",
                    help="skip top-down admin-parent resolution / partOf relations")
    ap.add_argument("--concurrency", type=int, default=24)   # gateway is local/fast
    ap.add_argument("--threshold", type=float, default=80)
    ap.add_argument("--radius-km", type=float, default=150)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--time", default="06:00:00")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ingest = "" if args.no_ingest else args.ingest_glob
    script = build_sbatch(db=args.db, repo=args.repo, ingest_glob=ingest, concurrency=args.concurrency,
                          threshold=args.threshold, radius_km=args.radius_km, limit=args.limit, wall=args.time,
                          no_hierarchy=args.no_hierarchy)
    sb = Path(args.repo) / "logs" / "gotw-reconcile.sbatch"
    sb.parent.mkdir(parents=True, exist_ok=True)
    sb.write_text(script)
    print(f"reconcile job: 1 htc CPU node, concurrency {args.concurrency}, threshold {args.threshold}"
          f"{', ingest first' if ingest else ', no ingest'}")
    print(f"sbatch -> {sb}")
    if args.dry_run:
        print("\n--- DRY RUN ---\n" + script + "--- END ---")
    else:
        out = subprocess.run(["sbatch", str(sb)], capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())


if __name__ == "__main__":
    main()
