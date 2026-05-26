#!/bin/bash
# Stage the large generated site assets in the GitHub 'site-assets' release — GitHub-hosted, NOT in git
# history (no LFS, no third party). The Pages deploy workflow (.github/workflows/pages.yml) pulls them in
# at build time. Re-run this after regenerating the reader store / search DB. Needs the `gh` CLI, authed.
#
#   process/publish_assets.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "1/2  tar the reader store (+ plate images, if present)"
tar -cf /tmp/reader.tar -C docs reader            # extracts back to docs/reader
PLATES=()
[ -d docs/plates ] && { tar -cf /tmp/plates.tar -C docs plates; PLATES=(/tmp/plates.tar); }   # -> docs/plates
GEOM=()
[ -f docs/geometry.pmtiles ] && GEOM=(docs/geometry.pmtiles)   # WHG line/polygon tileset (~42MB) -> docs/geometry.pmtiles

echo "2/2  upload to the 'site-assets' release (creating it if absent)"
gh release view site-assets >/dev/null 2>&1 || \
  gh release create site-assets --title "Site assets" --notes "Large static assets for the Pages site, served same-origin but kept out of git history."
gh release upload site-assets \
  /tmp/reader.tar \
  "${PLATES[@]}" \
  "${GEOM[@]}" \
  docs/search/gotw-fts.sqlite.png \
  docs/search/symphonym.int8.onnx \
  docs/search/symphonym-embeddings.i8 \
  docs/search/symphonym-meta.json \
  docs/search/symphonym-manifest.json \
  --clobber

echo "done — reader + plates + geometry.pmtiles + FTS DB + Symphonym published to release 'site-assets'"
