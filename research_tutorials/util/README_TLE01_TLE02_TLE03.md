# TLE01–TLE03: Satellite position pre-processing

Brief instructions for the three-step pipeline that produces ECEF lookup CSVs used by RFI modeling. Each **nc4 file contains data from multiple satellites** (distinguished by the **SAID** variable). The directory layout is **sensor-based**: ATMS/, AMSU-A/, SSMI-S/.

---

## Prerequisite: directory layout and SAID mapping

**1. Sensor-based directories** (names must match exactly):

- **ATMS/** — nc4 files for ATMS sensor (SUOMI-NPP, JPSS-1)
- **AMSU-A/** — nc4 files for AMSU-A sensor (NOAA-15, NOAA-18, NOAA-19, METOP-B, METOP-C)
- **SSMI-S/** — nc4 files for SSMI-S sensor (DMSP-F17)

**2. Place nc4 files** in the corresponding sensor directory (e.g. `atms_2023080112.nc4` in `ATMS/`).

**3. SAID (Satellite ID)** in the nc4 file identifies which satellite each observation comes from:

| Sensor  | SAID | Satellite   |
|---------|------|-------------|
| ATMS    | 224  | SUOMI-NPP   |
| ATMS    | 225  | JPSS-1      |
| AMSU-A  | 206  | NOAA-15     |
| AMSU-A  | 209  | NOAA-18     |
| AMSU-A  | 223  | NOAA-19     |
| AMSU-A  | 3    | METOP-B     |
| AMSU-A  | 5    | METOP-C     |
| SSMI-S  | 285  | DMSP-F17    |

**Example structure under `research_tutorials/util/`:**

```
research_tutorials/util/
  data/                          # TLE01 writes TLE files here
    JPSS-1_TLE.txt
    SUOMI-NPP_TLE.txt
    NOAA-15_TLE.txt
    ...
  ATMS/                          # sensor dir; nc4 files here
    atms_2023080112.nc4
    SUOMI-NPP_timestamp_atms_2023080112.csv    # from TLE02 (split by SAID)
    JPSS-1_timestamp_atms_2023080112.csv
    SUOMI-NPP_ECEF_lookup_atms_2023080112.csv  # from TLE03
    JPSS-1_ECEF_lookup_atms_2023080112.csv
    ...
  AMSU-A/
    amsua_2023080112.nc4
    NOAA-15_timestamp_amsua_2023080112.csv
    ...
  SSMI-S/
    ssmis_2023080112.nc4
    DMSP-F17_timestamp_ssmis_2023080112.csv
    ...
  TLE01_download_tle_archive.py
  TLE02_generate_timestamps_csv_from_nc4.py
  TLE03_calculate_multiple_ecef.py
```

Run all commands from `research_tutorials/util/` (or adjust paths).

---

## Step 1: TLE01 — download TLE archive

**No change.** TLE01 still downloads TLEs per satellite from Space-Track.org.

### Note

For NWP simulation from 2023-08-01 to 2023-11-01, TLE files for the satellites above were already prepared; Step 1 can be skipped.

```bash
python TLE01_download_tle_archive.py -sat JPSS-1 -start 2023-08-01 -end 2023-11-01
```

- **Output:** `data/JPSS-1_TLE.txt`. Repeat for each satellite (SUOMI-NPP, JPSS-1, NOAA-15, NOAA-18, NOAA-19, METOP-B, METOP-C, DMSP-F17 as needed).

---

## Step 2: TLE02 — timestamps from nc4 (split by SAID)

Reads each nc4 file in the **sensor** directory, extracts YEAR, MONTH, DAYS, HOUR, MINUTE, SECOND and **SAID**, builds timestamps, and **splits by SAID** into one CSV per satellite per nc4 file.

```bash
python TLE02_generate_timestamps_csv_from_nc4.py --input-dir ATMS
python TLE02_generate_timestamps_csv_from_nc4.py --input-dir AMSU-A
python TLE02_generate_timestamps_csv_from_nc4.py --input-dir SSMI-S
```

- **Input:** sensor directory (ATMS, AMSU-A, or SSMI-S) containing .nc4 files. Directory name must be one of these sensor names.
- **Output:** For each nc4 file, one `{satellite}_timestamp_{nc4_stem}.csv` **per satellite present** in that file (e.g. for `atms_2023080112.nc4` → `SUOMI-NPP_timestamp_atms_2023080112.csv` and `JPSS-1_timestamp_atms_2023080112.csv`).

---

## Step 3: TLE03 — ECEF lookup from TLE + timestamps

Finds all `*_timestamp_*.csv` files in the **sensor** directory, infers the satellite name from each filename, loads the matching TLE from the TLE directory, and writes one ECEF lookup CSV per timestamp file.

```bash
python TLE03_calculate_multiple_ecef.py -input-dir ATMS -tle-dir data
python TLE03_calculate_multiple_ecef.py -input-dir AMSU-A -tle-dir data
python TLE03_calculate_multiple_ecef.py -input-dir SSMI-S -tle-dir data
```

- **Input:** `-input-dir` = sensor directory (ATMS, AMSU-A, or SSMI-S) containing `*_timestamp_*.csv` from TLE02. `-tle-dir` = directory containing `{satellite}_TLE.txt` files (default: `data` under the script directory).
- **Output:** `{satellite}_ECEF_lookup_{stem}.csv` in the same sensor directory for each timestamp CSV.

---

## Order and per-sensor workflow

1. Create sensor directories (ATMS/, AMSU-A/, SSMI-S/); put nc4 files in the appropriate one.
2. Run TLE01 for each satellite you need → `data/<SAT>_TLE.txt` (e.g. SUOMI-NPP, JPSS-1, NOAA-15, …).
3. Run TLE02 with `--input-dir ATMS` (then AMSU-A, then SSMI-S) → per-satellite `*_timestamp_*.csv` in each sensor dir.
4. Run TLE03 with `-input-dir ATMS -tle-dir data` (then AMSU-A, SSMI-S) → per-satellite `*_ECEF_lookup_*.csv` in each sensor dir.

RFI modeling then uses the **sensor** directory: one nc4 file plus the matching ECEF lookups (same `{satellite}` and `{stem}`) for each satellite in that nc4.
