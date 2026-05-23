"""Submit the VLM heading-validation pass as a GPU SLURM array.

Each array task serves Qwen2.5-VL on a fresh port and runs vlm_validate.py over its image-index
range -> a per-shard JSONL (resumable). After the array drains, ingest into vlm_qa:
    python3 process/vlm_validate.py --db <db> --ingest '<out_dir>/<vol>.*.jsonl'

    python3 process/submit_vlm_slurm.py --vol v3 --dry-run
    python3 process/submit_vlm_slurm.py --vol v3
"""
import argparse, math, os, subprocess, sys
from pathlib import Path

_CONDA_SH = os.environ.get("CONDA_SH", "/ihome/ishi/stg135/miniconda3/etc/profile.d/conda.sh")
_VLLM_ENV = os.environ.get("VLLM_ENV", "/vast/ishi/envs/vllm")
_HF_HOME = os.environ.get("HF_HOME", "/vast/ishi/hf_cache")
_REPO = os.environ.get("GOTW_REPO", "/vast/ishi/gotw")
_VL = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"


def build(vol, img_dir, ocr, db, out_dir, n_imgs, shard, partition, wall):
    n_shards = math.ceil(n_imgs / shard)
    log = f"{_REPO}/logs"
    L = [
        "#!/bin/bash",
        f"#SBATCH --job-name=gotw-vlm-{vol}",
        f"#SBATCH --output={log}/vlm-{vol}-%A_%a.out",
        f"#SBATCH --error={log}/vlm-{vol}-%A_%a.err",
        "#SBATCH -M gpu", f"#SBATCH -p {partition}", "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8", "#SBATCH --mem=80G", "#SBATCH --requeue",
        f"#SBATCH --time={wall}", f"#SBATCH --array=0-{n_shards - 1}",
        "", "set -eo pipefail", "T=$SLURM_ARRAY_TASK_ID",
        f"source {_CONDA_SH}", f"conda activate {_VLLM_ENV}", "module load cuda/12.8.0",
        # NB: no VLLM_ATTENTION_BACKEND=FLASH_ATTN — Qwen2.5-VL vision tower needs SDPA.
        f"export HF_HOME={_HF_HOME} XDG_CACHE_HOME=/tmp/vlmcache_${{SLURM_JOB_ID}}_${{T}}",
        "PORT=$(( 18900 + (SLURM_JOB_ID % 700) + T ))",
        "SRVLOG=/tmp/vlsrv_${SLURM_JOB_ID}_${T}.log",
        f"vllm serve {_VL} --host 127.0.0.1 --port $PORT --max-model-len 16384 \\",
        f"    --gpu-memory-utilization 0.92 --served-model-name {_VL} > $SRVLOG 2>&1 &",
        "SRV=$!",
        'for i in $(seq 1 240); do grep -qa "Application startup complete" $SRVLOG && break; sleep 5; done',
        'grep -qa "Application startup complete" $SRVLOG || { echo "SERVER FAILED"; tail -20 $SRVLOG; kill $SRV; exit 1; }',
        "export TABLE_VL_BASE=http://127.0.0.1:$PORT/v1",
        f"SHARD={shard}", "LO=$(( T * SHARD ))", "HI=$(( LO + SHARD - 1 ))",
        f"cd {_REPO}",
        f"python3 process/vlm_validate.py --db {db} --vol {vol} --img-dir {img_dir} --ocr {ocr} \\",
        f'    --start "$LO" --end "$HI" --out-jsonl {out_dir}/{vol}.${{T}}.jsonl',
        "kill $SRV 2>/dev/null",
    ]
    return "\n".join(L) + "\n", n_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True)
    ap.add_argument("--img-dir", help="default /vast/ishi/gotw/img/<vol>")
    ap.add_argument("--ocr", help="default /vast/ishi/gotw/txt/gotw-<vol>-ocr.txt")
    ap.add_argument("--db", default=f"{_REPO}/data/gotw_seg.sqlite")
    ap.add_argument("--out-dir", help="default /vast/ishi/gotw/vlm_jsonl")
    ap.add_argument("--shard", type=int, default=240, help="image indices per array task")
    ap.add_argument("--partition", default="h200")
    ap.add_argument("--time", default="04:00:00")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    img_dir = args.img_dir or f"{_REPO}/img/{args.vol}"
    ocr = args.ocr or f"{_REPO}/txt/gotw-{args.vol}-ocr.txt"
    out_dir = Path(args.out_dir or f"{_REPO}/vlm_jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_imgs = len([p for p in Path(img_dir).iterdir() if p.suffix.lower() in
                  (".jpg", ".jpeg", ".png", ".tif", ".tiff")]) if Path(img_dir).exists() else 0
    if not n_imgs:
        sys.exit(f"no images in {img_dir}")
    script, n_shards = build(args.vol, img_dir, ocr, args.db, out_dir, n_imgs, args.shard, args.partition, args.time)
    sb = Path(_REPO) / "logs" / f"gotw-vlm-{args.vol}.sbatch"
    sb.parent.mkdir(parents=True, exist_ok=True)
    sb.write_text(script)
    print(f"{args.vol}: {n_imgs} images / shard {args.shard} = {n_shards} GPU array tasks on '{args.partition}'")
    print(f"per-shard JSONL -> {out_dir}/{args.vol}.<task>.jsonl ; sbatch -> {sb}")
    print(f"after it drains: python3 process/vlm_validate.py --db {args.db} --ingest '{out_dir}/{args.vol}.*.jsonl'")
    if args.dry_run:
        print("\n--- DRY RUN ---\n" + script + "--- END ---")
    else:
        out = subprocess.run(["sbatch", str(sb)], capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())


if __name__ == "__main__":
    main()
