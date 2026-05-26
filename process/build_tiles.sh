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
# -l places             : single named layer (map.html references source-layer "places")
# -z9 --cluster-maxzoom=6 : cluster ONLY up to z6 (the low-zoom density HEATMAP, weighted by point_count);
#                         at z7+ clustering is OFF so EVERY place is an individual feature with its own id —
#                         so zooming in always yields a visible circle/marker per place (not a heatmap blob).
#                         z9 max gives a couple of unclustered levels to overzoom from.
# --cluster-distance=8  : within the clustered zooms, merge points within ~8px into a representative leader.
# -r1                   : drop-rate 1 = no rate-thinning; clustering is the sole reducer (exact point_count).
tippecanoe -o "$OUT" -f -l places \
  -z9 --cluster-distance=8 --cluster-maxzoom=6 -r1 \
  "$GEOJSONL"

echo "done -> $OUT"
echo "(set map.html SRC to { mode:\"pmtiles\", url:\"./places.pmtiles\", layer:\"places\" } to use it)"
