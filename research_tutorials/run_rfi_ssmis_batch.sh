#!/usr/bin/env bash
# Run SSMI-S RFI (5G + Starlink gateway; default gateways CSV) for every .nc4 in SENSOR_DIR.
# Only SAID 285 / DMSP-F17. ECEF lookups from same dir.
# Usage: ./run_rfi_ssmis_batch.sh SENSOR_DIR   e.g. ./run_rfi_ssmis_batch.sh util/SSMI-S

set -e
SENSOR_DIR="${1:?Usage: run_rfi_ssmis_batch.sh SENSOR_DIR}"
SCRIPT_DIR=$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")
cd "$SCRIPT_DIR"

for nc4 in "$SENSOR_DIR"/*.nc4; do
  [ -f "$nc4" ] || continue
  echo "--- $(basename "$nc4")"
  python SSMI-S_RFI_modeling.py --sensor SSMI-S --nc4 "$nc4" --out_dir "$SENSOR_DIR"
done
echo "Done."
