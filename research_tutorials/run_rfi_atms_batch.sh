#!/usr/bin/env bash
# Run ATMS RFI for every .nc4 in SATELLITE_DIR. ECEF lookup must exist: {SAT}_ECEF_lookup_{stem}.csv
# Usage: ./run_rfi_atms_batch.sh SATELLITE_DIR   e.g. ./run_rfi_atms_batch.sh util/JPSS-1

set -e
SAT_DIR="${1:?Usage: run_rfi_atms_batch.sh SATELLITE_DIR}"
SAT_NAME=$(basename "$SAT_DIR")
SCRIPT_DIR=$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")
cd "$SCRIPT_DIR"

for nc4 in "$SAT_DIR"/*.nc4; do
  [ -f "$nc4" ] || continue
  stem=$(basename "$nc4" .nc4)
  ecef="$SAT_DIR/${SAT_NAME}_ECEF_lookup_${stem}.csv"
  if [ -f "$ecef" ]; then
    echo "--- $(basename "$nc4")"
    python ATMS_RFI_modeling.py --sat "$SAT_NAME" --nc4 "$nc4" --ecef "$ecef" --out_dir "$SAT_DIR"
  else
    echo "Skip $nc4: ECEF not found: $ecef"
  fi
done
echo "Done."
