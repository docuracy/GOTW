#!/bin/bash
# Build docs/places.pmtiles (MapLibre vector tiles) from the reconciled `place` rows — the scale-up
# path for the demo map once the full ~100k-place corpus is reconciled. A single FeatureCollection
# would force every visitor to download/parse the whole file; PMTiles is viewport-loaded, scales to the
# full corpus, and is served STATICALLY from GitHub Pages (HTTP range requests, no tile server).
#
# Requires tippecanoe >= 2.17 (native PMTiles output) on PATH:
#   https://github.com/felt/tippecanoe  (brew install tippecanoe / build from source)
# Then flip map.html's SRC to { mode:"pmtiles", url:"./places.pmtiles", layer:"places" }.
#
#   process/build_tiles.sh                       # data/gotw_seg.sqlite -> docs/places.pmtiles
#   DB=/path/to.sqlite OUT=docs/places.pmtiles process/build_tiles.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DB="${DB:-${1:-data/gotw_seg.sqlite}}"
GEOJSONL="${GEOJSONL:-/tmp/gotw_places.geojsonl}"
OUT="${OUT:-docs/places.pmtiles}"

command -v tippecanoe >/dev/null || { echo "ERROR: tippecanoe not on PATH — see header." >&2; exit 1; }

echo "1/2  export $DB -> $GEOJSONL"
python3 process/export_geojson.py --db "$DB" --out "$GEOJSONL"

echo "2/2  tippecanoe -> $OUT"
# -l places          : single named layer (map.html references source-layer "places")
# -zg                : choose max zoom automatically from feature density
# --drop-densest-as-needed + --extend-zooms-if-still-dropping : keep low zoom readable without
#                      dropping points permanently — denser areas simply resolve as you zoom in
# -r1                : no low-zoom point thinning beyond the size-limit logic (keep all points)
tippecanoe -o "$OUT" -f -l places \
  -zg --drop-densest-as-needed --extend-zooms-if-still-dropping -r1 \
  "$GEOJSONL"

echo "done -> $OUT"
echo "(set map.html SRC to { mode:\"pmtiles\", url:\"./places.pmtiles\", layer:\"places\" } to use it)"
