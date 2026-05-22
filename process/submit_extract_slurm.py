#!/usr/bin/env python3
"""Parallel Llama extraction over the corpus as a SLURM array on the CRC GPU cluster.

Each array task = one GPU shard: it serves Llama-3.3-70B on localhost (no tunnel), copies
the entries DB to node-local /tmp (so shards never write the shared DB → zero contention),
runs the concurrent extractor over its slice (entry_id %% nshards == shard), and writes
results to a per-shard JSONL on /vast. Resumable (the JSONL is append/skip). After the array
drains, merge with:  python3 process/extract.py --ingest '<out-dir>/llama.*.jsonl'

Run on a CRC login node, with the DB + repo on /vast and Llama-AWQ in the HF cache:

    python3 process/submit_extract_slurm.py --db /vast/ishi/gotw/data/gotw.sqlite \
        --repo /vast/ishi/gotw --out-dir /vast/ishi/gotw/llama_jsonl --nshards 8 --dry-run
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

_CONDA_SH = os.environ.get("CONDA_SH", "/ihome/ishi/stg135/miniconda3/etc/profile.d/conda.sh")
_VLLM_ENV = os.environ.get("VLLM_ENV", "/vast/ishi/envs/vllm")
_HF_HOME = os.environ.get("HF_HOME", "/vast/ishi/hf_cache")


def build_sbatch(*, db, repo, out_dir, nshards, partition, conc, hf_model, served, port_base, wall, log_dir):
    log_prefix = f"{log_dir}/extract-%A_%a"
    srv_log = f"{log_dir}/srv-${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.log"
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=gotw-extract",
        f"#SBATCH --output={log_prefix}.out",
        f"#SBATCH --error={log_prefix}.err",
        "#SBATCH --clusters=gpu",
        f"#SBATCH --partition={partition}",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=96G",
        "#SBATCH --requeue",
        f"#SBATCH --time={wall}",
        f"#SBATCH --array=0-{nshards - 1}",
        "",
        "set -eo pipefail",
        "T=$SLURM_ARRAY_TASK_ID",
        f"PORT=$(( {port_base} + T ))",
        'echo "shard ${T} on $(hostname) port ${PORT}"',
        f"source {_CONDA_SH}",
        f"conda activate {_VLLM_ENV}",
        "module load cuda/12.8.0",
        f"export HF_HOME={_HF_HOME} XDG_CACHE_HOME=/tmp/vllmcache_${{T}} VLLM_ATTENTION_BACKEND=FLASH_ATTN",
        f"cp {db} /tmp/gotw_${{T}}.sqlite",
        f"vllm serve {hf_model} --host 127.0.0.1 --port ${{PORT}} --max-model-len 16384 \\",
        f"    --gpu-memory-utilization 0.92 --served-model-name {served} > {srv_log} 2>&1 &",
        "SRV=$!",
        f'echo "waiting for vLLM..."; for i in $(seq 1 150); do grep -qa "Application startup complete" {srv_log} && break; sleep 5; done',
        f'grep -qa "Application startup complete" {srv_log} || {{ echo "server failed"; tail -20 {srv_log}; exit 1; }}',
        f"export QWEN_BASE_URL=http://127.0.0.1:${{PORT}}/v1",
        f"cd {repo}",
        "python -u process/extract.py \\",
        "    --db /tmp/gotw_${T}.sqlite \\",
        f"    --provider vllm --model {served} \\",
        f"    --nshards {nshards} --shard ${{T}} --concurrency {conc} \\",
        f"    --out-jsonl {out_dir}/llama.${{T}}.jsonl",
        "kill $SRV 2>/dev/null || true",
        'echo "shard ${T} done"',
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="entries DB on /vast (copied to node /tmp per shard)")
    ap.add_argument("--repo", default="/vast/ishi/gotw", help="GOTW checkout on /vast (has process/, data/aat_shortlist.json)")
    ap.add_argument("--out-dir", default="/vast/ishi/gotw/llama_jsonl")
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--partition", default="h200")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--hf-model", default="casperhansen/llama-3.3-70b-instruct-awq")
    ap.add_argument("--served", default="llama-3.3-70b")
    ap.add_argument("--port-base", type=int, default=18940)
    ap.add_argument("--time", default="03:00:00")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_dir = Path(args.repo) / "logs"
    out_dir = Path(args.out_dir)
    for d in (log_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    script = build_sbatch(db=args.db, repo=args.repo, out_dir=args.out_dir, nshards=args.nshards,
                          partition=args.partition, conc=args.concurrency, hf_model=args.hf_model,
                          served=args.served, port_base=args.port_base, wall=args.time, log_dir=str(log_dir))
    sb = log_dir / "gotw-extract.sbatch"
    sb.write_text(script)
    print(f"{args.nshards} shards on '{args.partition}' (1 GPU each), concurrency {args.concurrency}")
    print(f"per-shard JSONL -> {out_dir}/llama.<shard>.jsonl ;  sbatch -> {sb}")
    if args.dry_run:
        print("\n--- DRY RUN ---\n" + script + "--- END ---")
    else:
        out = subprocess.run(["sbatch", str(sb)], capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())
    print(f"\nwhen the array drains, merge locally with:\n"
          f"  scp 'crc0:{out_dir}/llama.*.jsonl' /tmp/  &&  python3 process/extract.py --ingest '/tmp/llama.*.jsonl'")


if __name__ == "__main__":
    main()
