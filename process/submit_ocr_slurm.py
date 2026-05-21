#!/usr/bin/env python3
"""Submit a Surya OCR run as a SLURM array on the Pitt CRC GPU cluster.

Shards a volume's pages across array tasks; each task runs process/ocr_pages.py over its
page range, writing one resumable file per page into --out-dir on /vast/ishi. A failed or
pre-empted task just re-runs and skips the pages already written. After the array drains,
stitch the per-page files with `ocr_pages.py --merge` (printed below).

Run on a CRC login node (ssh crc0) with the page images + repo on fast /vast/ishi and
surya installed in the `whg` conda env:

    python3 process/submit_ocr_slurm.py --img-dir /vast/ishi/gotw/img/v5 --vol v5 --dry-run
    python3 process/submit_ocr_slurm.py --img-dir /vast/ishi/gotw/img/v5 --vol v5
    python3 process/submit_ocr_slurm.py --pdf /vast/ishi/gotw/pdf/gotw-v5.pdf --vol v5   # PDF source

Defaults target the l40s partition (19 nodes); `--partition preempt` taps the large
pre-emptible pool (ocr_pages is resumable, so pre-emption is free). One GPU per task.
"""
from __future__ import annotations
import argparse, math, os, subprocess, sys
from pathlib import Path

_REPO = Path(os.environ.get("GOTW_REPO", Path(__file__).resolve().parent.parent))
_CONDA_ENV = os.environ.get("CONDA_ENV", "whg")
_CONDA_SH = os.environ.get("CONDA_SH", "/ihome/ishi/stg135/miniconda3/etc/profile.d/conda.sh")
_HF_HOME = os.environ.get("HF_HOME", "/vast/ishi/hf_cache")
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def count_pages(img_dir: str | None, pdf: str | None) -> int:
    if img_dir:
        return sum(1 for p in Path(img_dir).iterdir() if p.suffix.lower() in _IMG_EXTS)
    import fitz
    return fitz.open(pdf).page_count


def build_sbatch(*, source_arg: str, vol: str, out_dir: Path, n_pages: int, shard: int,
                 partition: str, dpi: int, wall: str) -> tuple[str, int]:
    n_shards = math.ceil(n_pages / shard)
    log_dir = _REPO / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = log_dir / f"gotw-ocr-{vol}-%A_%a"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=gotw-ocr-{vol}",
        f"#SBATCH --output={log_prefix}.out",
        f"#SBATCH --error={log_prefix}.err",
        "#SBATCH --clusters=gpu",
        f"#SBATCH --partition={partition}",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32G",
        "#SBATCH --requeue",
        f"#SBATCH --time={wall}",
        f"#SBATCH --array=0-{n_shards - 1}",
        "",
        "set -eo pipefail",
        f"export HF_HOME={_HF_HOME}",
        f"source {_CONDA_SH}",
        f"conda activate {_CONDA_ENV}",
        f"cd {_REPO}",
        "",
        f"SHARD={shard}",
        f"N={n_pages}",
        "LO=$(( SLURM_ARRAY_TASK_ID * SHARD ))",
        "HI=$(( LO + SHARD - 1 ))",
        "if [ $HI -ge $N ]; then HI=$(( N - 1 )); fi",
        'echo "task ${SLURM_ARRAY_TASK_ID}: pages ${LO}-${HI}"',
        "python -u process/ocr_pages.py \\",
        f"    {source_arg} \\",
        f"    --out-dir {out_dir} \\",
        f"    --dpi {dpi} \\",
        '    --start "$LO" --end "$HI"',
    ]
    return "\n".join(lines) + "\n", n_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", dest="img_dir", help="directory of page images (on /vast/ishi)")
    ap.add_argument("--pdf", help="volume PDF (alternative to --img-dir)")
    ap.add_argument("--vol", required=True, help="volume tag, e.g. v5")
    ap.add_argument("--out-dir", help="default /vast/ishi/gotw/ocr/<vol>")
    ap.add_argument("--shard", type=int, default=150, help="pages per array task")
    ap.add_argument("--partition", default="l40s", help="l40s | a100 | rtx6k | preempt")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--time", default="02:00:00")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.img_dir and not args.pdf:
        ap.error("one of --img-dir or --pdf is required")

    src = Path(args.img_dir or args.pdf)
    if not src.exists():
        sys.exit(f"source not found: {src} (stage it onto /vast/ishi first)")
    n_pages = count_pages(args.img_dir, args.pdf)
    if n_pages == 0:
        sys.exit(f"no pages found in {src}")
    source_arg = f"--img-dir {args.img_dir}" if args.img_dir else f"--pdf {args.pdf}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"/vast/ishi/gotw/ocr/{args.vol}")
    out_dir.mkdir(parents=True, exist_ok=True)

    script, n_shards = build_sbatch(
        source_arg=source_arg, vol=args.vol, out_dir=out_dir, n_pages=n_pages, shard=args.shard,
        partition=args.partition, dpi=args.dpi, wall=args.time)
    sbatch_path = _REPO / "logs" / f"gotw-ocr-{args.vol}.sbatch"
    sbatch_path.parent.mkdir(parents=True, exist_ok=True)
    sbatch_path.write_text(script)

    merge_cmd = (f"python3 process/ocr_pages.py {source_arg} "
                 f"--out-dir {out_dir} --merge --out data/txt/gotw-{args.vol}-ocr.txt")
    print(f"{args.vol}: {n_pages} pages / shard {args.shard} = {n_shards} array tasks "
          f"on partition '{args.partition}' (1 GPU each)")
    print(f"per-page output -> {out_dir}")
    print(f"sbatch script  -> {sbatch_path}")
    if args.dry_run:
        print("\n--- DRY RUN: sbatch script ---")
        print(script, end="")
        print("--- END ---")
    else:
        out = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())
    print(f"\nwhen the array finishes, stitch with:\n  {merge_cmd}")


if __name__ == "__main__":
    main()
