#!/usr/bin/env python3
"""Apply plate-orientation corrections reported from the explorer lightbox, then close the reports.

The lightbox "⚑ orientation" button files a GitHub issue titled "[plate orientation] <vol>/<file>.jpg —
rotate N° CW" (labelled `explorer-report`, body names the plate + the clockwise angle). Trigger this script
whenever such reports accumulate; it:

  1. fetches the OPEN plate-orientation reports (gh CLI),
  2. rotates each plate image N° clockwise in place (docs/plates/<vol>/<file>.jpg),
  3. records the cumulative correction in data/plate_orientation_overrides.json so a pipeline re-run
     (export_plates.py, which reads it) reproduces the fix instead of reverting to the auto-orientation,
  4. CLOSES each processed issue with a confirming comment,
  5. re-tars docs/plates and uploads plates.tar to the `site-assets` release (unless --no-publish).

    process/apply_plate_orientation.py              # apply + close + republish
    process/apply_plate_orientation.py --dry-run    # show what it would do, touch nothing
    process/apply_plate_orientation.py --no-publish  # apply + close, but don't re-upload plates.tar

Requires: gh (authenticated), Pillow. Run from the repo root. After republishing, trigger a Pages deploy
(push to docs/** or run the "Deploy Pages" workflow) for the corrected plates to go live.
"""
from __future__ import annotations
import argparse, json, re, subprocess
from pathlib import Path
from PIL import Image

REPO = "WorldHistoricalGazetteer/gazetteer-of-the-world"
PLATES = Path("docs/plates")
OVERRIDES = Path("data/plate_orientation_overrides.json")
TITLE_RE = re.compile(r"\[plate orientation\]", re.I)
PLATE_RE = re.compile(r"\*\*Plate:\*\*\s*`([^`]+)`")          # **Plate:** `plates/v1/p00548.jpg`
DEG_RE = re.compile(r"(\d+)\s*°", re.I)                       # first "N°" in title/body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    issues = json.loads(subprocess.check_output(
        ["gh", "issue", "list", "--repo", REPO, "--label", "explorer-report", "--state", "open",
         "--limit", "200", "--json", "number,title,body"], text=True))
    reports = [it for it in issues if TITLE_RE.search(it.get("title") or "")]
    if not reports:
        print("no open plate-orientation reports — nothing to do.")
        return 0
    print(f"{len(reports)} open plate-orientation report(s)")

    overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    done, rotated_any = [], False
    for it in reports:
        body, title, num = it.get("body") or "", it.get("title") or "", it["number"]
        pm = PLATE_RE.search(body)
        src = pm.group(1).strip() if pm else None
        if not src:                                          # fall back to the path in the title
            tm = re.search(r"(v\w+/p\w+\.jpg)", title)
            src = ("plates/" + tm.group(1)) if tm else None
        if not src:
            print(f"  #{num}: SKIP — no plate path found"); continue
        rel = src[len("plates/"):] if src.startswith("plates/") else src   # v1/p00548.jpg
        dm = DEG_RE.search(body) or DEG_RE.search(title)
        deg = int(dm.group(1)) % 360 if dm else 0
        img = PLATES / rel
        if not img.exists():
            print(f"  #{num}: SKIP — {img} not found"); continue
        print(f"  #{num}: {rel}  rotate {deg}° CW" + ("  (already correct — close only)" if deg == 0 else ""))
        if not args.dry_run and deg:
            Image.open(img).convert("RGB").rotate(-deg, expand=True).save(img, "JPEG", quality=90)
            overrides[rel] = (overrides.get(rel, 0) + deg) % 360   # cumulative, relative to the auto-orientation
            rotated_any = True
        done.append((num, rel, deg))

    if args.dry_run:
        print("(dry run — no images changed, no issues closed, nothing published)")
        return 0

    OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")
    print(f"recorded {len(overrides)} cumulative override(s) -> {OVERRIDES}")

    for num, rel, deg in done:                               # comment then close (no `close --comment`: older gh)
        comment = (f"✅ Applied **{deg}° clockwise** rotation to `{rel}` and recorded it in "
                   f"`data/plate_orientation_overrides.json` (so a pipeline re-run keeps it). "
                   f"Re-published plates; will appear on the next deploy." if deg
                   else f"Closing: `{rel}` was reported as already correctly oriented — no change needed.")
        subprocess.run(["gh", "issue", "comment", str(num), "--repo", REPO, "--body", comment], check=False)
        subprocess.run(["gh", "issue", "close", str(num), "--repo", REPO], check=False)
        print(f"  closed #{num}")

    if rotated_any and not args.no_publish:
        print("re-tarring docs/plates and uploading plates.tar to the site-assets release …")
        subprocess.run(["tar", "-cf", "/tmp/plates.tar", "-C", "docs", "plates"], check=True)
        subprocess.run(["gh", "release", "upload", "site-assets", "/tmp/plates.tar", "--repo", REPO, "--clobber"], check=True)
        print("uploaded plates.tar. Trigger a Pages deploy (push to docs/** or run 'Deploy Pages') to go live.")
    elif rotated_any:
        print("(--no-publish: plates.tar NOT uploaded; run process/publish_assets.sh when ready)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
