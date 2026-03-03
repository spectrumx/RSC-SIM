# TLE01–TLE03: Satellite position pre-processing

Brief instructions for the three-step pipeline that produces ECEF lookup CSVs used by RFI modeling.

---

## Prerequisite: directory layout and satellite names

**1. Create one directory per satellite** (names must match exactly). Allowed names (from `TLE01_download_tle_archive.py`, `NORAD_ID_TO_NAME`):

- SUOMI-NPP, JPSS-1, NOAA-15, NOAA-18, NOAA-19, METOP-B, METOP-C, DMSP-F17

**2. Place nc4 files** in the corresponding satellite directory.

**Example structure under `research_tutorials/util/`:**

```
research_tutorials/util/
  data/                          # TLE01 writes TLE files here
    JPSS-1_TLE.txt
    SUOMI-NPP_TLE.txt
    ...
  JPSS-1/                        # one dir per satellite; nc4 files here
    atms_2023080112.nc4
    atms_2023080113.nc4
    JPSS-1_timestamp_atms_2023080112.csv    # created by TLE02
    JPSS-1_timestamp_atms_2023080113.csv
    JPSS-1_ECEF_lookup_atms_2023080112.csv  # created by TLE03
    JPSS-1_ECEF_lookup_atms_2023080113.csv
    ...
  SUOMI-NPP/
    ...
  DMSP-F17/
    ...
  TLE01_download_tle_archive.py
  TLE02_generate_timestamps_csv_from_nc4.py
  TLE03_calculate_multiple_ecef.py
```

Run all commands from `research_tutorials/util/` (or adjust paths).

---

## Step 1: TLE01 — download TLE archive

### Note: this is a general instruction for TLE01. For NWP simulation from 2023-08-01 -end 2023-11-01, corresponding TLE files for those weather satellites were already prepared, thus this Step 1 can be skipped.

Downloads two-line elements from Space-Track.org. Requires an account; set `USERNAME` and `PASSWORD` in the script.

```bash
python TLE01_download_tle_archive.py -sat JPSS-1 -start 2023-08-01 -end 2023-11-01
```

- **Output:** `data/JPSS-1_TLE.txt` (created under the script’s directory).
- Repeat for each satellite and date range you need.

---

## Step 2: TLE02 — timestamps from nc4

Builds one timestamp CSV per nc4 file in the given directory. Writes into that same directory.

```bash
python TLE02_generate_timestamps_csv_from_nc4.py --input-dir JPSS-1
```

- **Input:** directory containing nc4 files (e.g. `JPSS-1/`). Directory name = satellite name.
- **Output:** `JPSS-1/JPSS-1_timestamp_<nc4_stem>.csv` for each nc4 file.

---

## Step 3: TLE03 — ECEF lookup from TLE + timestamps

Computes ECEF at each timestamp for every timestamp CSV in the directory. Writes one ECEF lookup CSV per timestamp file, in the same directory.

```bash
python TLE03_calculate_multiple_ecef.py -tle data/JPSS-1_TLE.txt -input-dir JPSS-1
```

- **Input:** TLE file (from TLE01) and directory containing `*_timestamp_*.csv` (from TLE02).
- **Output:** `JPSS-1/JPSS-1_ECEF_lookup_<stem>.csv` for each timestamp CSV.

---

## Order and per-satellite workflow

1. Create satellite directory; put nc4 files in it.
2. Run TLE01 for that satellite (and date range) → `data/<SAT>_TLE.txt`.
  - Skip this for the date range from 2023-08-01 to 2023-11-01 for the listed satellites above
3. Run TLE02 with `--input-dir <SAT>` → `<SAT>/*_timestamp_*.csv`.
4. Run TLE03 with `-tle data/<SAT>_TLE.txt -input-dir <SAT>` → `<SAT>/*_ECEF_lookup_*.csv`.

RFI modeling can then use one nc4 file and the matching ECEF lookup (same `<stem>`) from the same directory.
