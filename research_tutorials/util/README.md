# `util/` — TLE and ECEF pre-processing for RFI modeling for NWP simulations

This directory holds **TLE (Two-Line Element) and ECEF pre-processing** tools used to prepare inputs for the weather satellite RFI modeling scripts for NWP simulations (`ATMS_RFI_modeling.py`, `AMSU-A_RFI_modeling.py`, `SSMI-S_RFI_modeling.py`) in `research_tutorials/`.

## Contents

- **TLE01–TLE03 pipeline**: Download TLEs, generate timestamp CSVs from nc4 files, and compute ECEF lookups. These lookups are required before running the ATMS/AMSU-A/SSMI-S RFI scripts.
  - **Full instructions:** [README_TLE01_TLE02_TLE03.md](README_TLE01_TLE02_TLE03.md)

- **`data/`**: TLE text files (e.g. `JPSS-1_TLE.txt`, `NOAA-19_TLE.txt`) produced by TLE01. One file per satellite and date range.

- **Satellite subdirectories** (`SUOMI-NPP/`, `JPSS-1/`, `NOAA-15/`, `NOAA-18`, `NOAA-19/`, `METOP-B/`, `METOP-C/`, and `DMSP-F17/`): Each contains sensor observation nc4 files and, after running TLE02 and TLE03, the corresponding `*_timestamp_*.csv` and `*_ECEF_lookup_*.csv` files used by the RFI modeling scripts for NWP simulations. These directories also serve as output directories for RFI modeling results.

Run TLE02 and TLE03 from this directory (or adjust paths). For the full pipeline and directory layout, see **[README_TLE01_TLE02_TLE03.md](README_TLE01_TLE02_TLE03.md)**.
