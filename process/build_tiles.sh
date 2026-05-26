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
GEOM_STORE="${GEOM_STORE:-/vast/ishi/geom}"          # consolidated WHG geom-store (CRC /vast)
GEOM_GEOJSONL="${GEOM_GEOJSONL:-/tmp/gotw_geometry.geojsonl}"
GEOM_OUT="${GEOM_OUT:-docs/geometry.pmtiles}"

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

# ── WHG line/polygon geometries (showcase) — only on the CRC, where the /vast geom-store lives ──────────
# Reconciled matches that resolve to OSM/OHM relations, Wikidata/TGN areas, etc. carry real boundaries.
# export_geometries.py reads them from the geom-store (NOT ES _source) and writes non-point GeoJSON;
# tippecanoe tiles them into a 'geometry' layer. Off-CRC the store is absent → empty export → skip.
echo
echo "geometry  export $DB + $GEOM_STORE -> $GEOM_GEOJSONL"
python3 process/export_geometries.py --db "$DB" --store "$GEOM_STORE" --out "$GEOM_GEOJSONL"
if [ -s "$GEOM_GEOJSONL" ]; then
  echo "geometry  tippecanoe -> $GEOM_OUT"
  # -Z3 -z10: large admin polygons visible from z3; map.html gates the FAINT layer above z5. lines = polygon
  # outlines + standalone coasts/straits. --no-tile-size-limit: keep all ~3k features (never drop to fit 500KB).
  tippecanoe -o "$GEOM_OUT" -f -l geometry \
    -Z3 -z10 --simplification=10 --no-tiny-polygon-reduction --no-tile-size-limit \
    "$GEOM_GEOJSONL"
  echo "done -> $GEOM_OUT"
else
  echo "geometry  no geom-store / no features (off-CRC) — skipping $GEOM_OUT"
fi
