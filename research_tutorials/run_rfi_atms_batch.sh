#!/usr/bin/env bash
# Run ATMS RFI (5G + Starlink gateway; default gateways CSV) for every input .nc4 in SENSOR_DIR.
# Skips *_RFI.nc4 (outputs from prior runs). ECEF lookups: same dir (*_ECEF_lookup_<stem>.csv).
# Usage: ./run_rfi_atms_batch.sh SENSOR_DIR   e.g. ./run_rfi_atms_batch.sh util/ATMS

set -e
SENSOR_DIR="${1:?Usage: run_rfi_atms_batch.sh SENSOR_DIR}"
SCRIPT_DIR=$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")
cd "$SCRIPT_DIR"

for nc4 in "$SENSOR_DIR"/*.nc4; do
  [ -f "$nc4" ] || continue
  case "$(basename "$nc4")" in
    *_RFI.nc4) echo "--- Skipping RFI output: $(basename "$nc4")"; continue ;;
  esac
  echo "--- $(basename "$nc4")"
  python ATMS_RFI_modeling.py --sensor ATMS --nc4 "$nc4" --out_dir "$SENSOR_DIR"
done
echo "Done."
