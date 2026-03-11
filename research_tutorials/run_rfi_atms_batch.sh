#!/usr/bin/env bash
# Run ATMS RFI for every .nc4 in SENSOR_DIR. ECEF lookups loaded from same dir (*_ECEF_lookup_<stem>.csv).
# Usage: ./run_rfi_atms_batch.sh SENSOR_DIR   e.g. ./run_rfi_atms_batch.sh util/ATMS

set -e
SENSOR_DIR="${1:?Usage: run_rfi_atms_batch.sh SENSOR_DIR}"
SCRIPT_DIR=$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")
cd "$SCRIPT_DIR"

for nc4 in "$SENSOR_DIR"/*.nc4; do
  [ -f "$nc4" ] || continue
  echo "--- $(basename "$nc4")"
  python ATMS_RFI_modeling.py --sensor ATMS --nc4 "$nc4" --out_dir "$SENSOR_DIR"
done
echo "Done."
