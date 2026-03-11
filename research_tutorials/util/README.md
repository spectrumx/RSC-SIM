# `util/` — TLE and ECEF pre-processing for RFI modeling for NWP simulations

This directory holds **TLE (Two-Line Element) and ECEF pre-processing** tools used to prepare inputs for the weather satellite RFI modeling scripts for NWP simulations (`ATMS_RFI_modeling.py`, `AMSU-A_RFI_modeling.py`, `SSMI-S_RFI_modeling.py`) in `research_tutorials/`.

## Contents

- **TLE01–TLE03 pipeline**: Download TLEs, generate per-satellite timestamp CSVs from nc4 files (split by SAID), and compute ECEF lookups. These lookups are required before running the ATMS/AMSU-A/SSMI-S RFI scripts.
  - **Full instructions:** [README_TLE01_TLE02_TLE03.md](README_TLE01_TLE02_TLE03.md)

- **`data/`**: TLE text files (e.g. `JPSS-1_TLE.txt`, `SUOMI-NPP_TLE.txt`, `NOAA-19_TLE.txt`) produced by TLE01. One file per satellite and date range.

- **Sensor-based directories** (**ATMS/**, **AMSU-A/**, **SSMI-S/**): Each contains:
  - nc4 sensor observation files (one file can include data from multiple satellites, distinguished by the SAID variable).
  - After TLE02 and TLE03: per-satellite `*_timestamp_*.csv` and `*_ECEF_lookup_*.csv` files (e.g. `SUOMI-NPP_timestamp_atms_2023080112.csv`, `JPSS-1_ECEF_lookup_atms_2023080112.csv`).
  - RFI modeling scripts use the same sensor directory to find the nc4 and the matching ECEF lookups per satellite.

Run TLE02 and TLE03 from this directory (or adjust paths). For the full pipeline and directory layout, see **[README_TLE01_TLE02_TLE03.md](README_TLE01_TLE02_TLE03.md)**.
