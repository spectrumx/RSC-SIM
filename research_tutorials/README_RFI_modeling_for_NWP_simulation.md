# RFI estimation: ATMS, AMSU-A, and SSMI-S

Brief procedure to run 5G ground-emitter RFI estimation for weather satellite sensors. Each script processes **one nc4 file** at a time; use the batch scripts below to run all nc4 files in a satellite directory.

- Note that this will require large disk space due to the file sizes of .nc4 files and generated output CSVs.

---

## 1. Download GHS-POP data (`GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif`)

European Union's Global Human Settlement Layer (GHSL) Population data (GHG-POP) needs to be downloaded from EU [Global Human Settlement Layer (GHSL)](https://human-settlement.emergency.copernicus.eu/download.php?ds=pop) for Epoch: 2025, Resolution: 30 arcsec (~1 km²), and Coordinate system: WGS84. It is zip compressed but only `GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif` file is required. Please place it at `research_tutorial/data/` directory. It is used for 5G ground emitter density in ATMS/AMSU-A/SSMI-S RFI scripts.

---

## 2. Data preparation (ECEF lookups)

Before running any RFI script, you must have ECEF lookup CSVs for each nc4 file. That is done by the TLE01–TLE03 pipeline:

1. Create one directory per satellite; put nc4 files in it.
2. Skip TLE01 (download TLE) as TLEs are already created for the following:
    - Satellites: SUOMI-NPP, JPSS-1, NOAA-15, NOAA-18, NOAA-19, METOP-B, METOP-C, DMSP-F17
    - Date range: 2023-08-01 to 2023-11-01
3. Run TLE02 (timestamps from nc4), TLE03 (ECEF from TLE + timestamps).

**Full instructions:** [util/README_TLE01_TLE02_TLE03.md](util/README_TLE01_TLE02_TLE03.md).

After that, each satellite directory contains its nc4 files plus `{SAT}_timestamp_{stem}.csv` and `{SAT}_ECEF_lookup_{stem}.csv` per nc4 (same `stem` as the nc4 filename without file extension, .nc4).

**Directory structure (after data preparation):**

```
research_tutorials/
  util/
    data/
      JPSS-1_TLE.txt
      NOAA-15_TLE.txt
      DMSP-F17_TLE.txt
      ...
    JPSS-1/
      atms_2023080112.nc4
      JPSS-1_timestamp_atms_2023080112.csv
      JPSS-1_ECEF_lookup_atms_2023080112.csv
      ...
    NOAA-15/
      amsua_2023080112.nc4
      NOAA-15_timestamp_amsua_2023080112.csv
      NOAA-15_ECEF_lookup_amsua_2023080112.csv
      ...
    DMSP-F17/
      ssmis_2023080112.nc4
      DMSP-F17_timestamp_ssmis_2023080112.csv
      DMSP-F17_ECEF_lookup_ssmis_2023080112.csv
      ...
  ATMS_RFI_modeling.py
  AMSU-A_RFI_modeling.py
  SSMI-S_RFI_modeling.py
  run_rfi_atms_batch.bat
  run_rfi_amsua_batch.bat
  run_rfi_ssmis_batch.bat
  run_rfi_atms_batch.sh
  run_rfi_amsua_batch.sh
  run_rfi_ssmis_batch.sh
  README_RFI_estimation.md
  Input_parameters_ATMS_AMSU_A_SSMI-S_RFI_modeling.md
```

After running the RFI batch scripts (section 3), each satellite directory will also contain `{SAT}_{nc4_stem}_5G_RFI_chN.csv` files.

---

## 3. Running RFI scripts (single nc4)

Run from **`research_tutorials/`**. All three scripts use the same argument pattern: `--sat`, `--nc4`, `--ecef`, `--out_dir` (all required).

### ATMS (e.g. JPSS-1, SUOMI-NPP)

```bash
python rfi_atms_jpss1_modeling.py --sat JPSS-1 --nc4 util/JPSS-1/atms_2023080112.nc4 --ecef util/JPSS-1/JPSS-1_ECEF_lookup_atms_2023080112.csv --out_dir util/JPSS-1
```

Output: `--out_dir`/`<sat>_<nc4_stem>_5G_RFI_chN.csv` (channels 3–9).

### AMSU-A (e.g. NOAA-15, NOAA-18, NOAA-19, METOP-B, METOP-C)

```bash
python AMSU-A_RFI_modeling.py --sat NOAA-15 --nc4 util/NOAA-15/amsua_2023080112.nc4 --ecef util/NOAA-15/NOAA-15_ECEF_lookup_amsua_2023080112.csv --out_dir util/NOAA-15
```

Output: `--out_dir`/`<sat>_<nc4_stem>_5G_RFI_chN.csv` (channels 3–8).

### SSMI-S (DMSP-F17)

```bash
python SSMI-S_RFI_modeling.py --sat DMSP-F17 --nc4 util/DMSP-F17/ssmis_2023080112.nc4 --ecef util/DMSP-F17/DMSP-F17_ECEF_lookup_ssmis_2023080112.csv --out_dir util/DMSP-F17
```

Output: `--out_dir`/`<sat>_<nc4_stem>_5G_RFI_chN.csv` (channels 1–5).

---

## 4. Batch runs (all nc4 files in a directory)

Each RFI script handles a single nc4 file. To process every nc4 in a satellite directory, use the batch scripts below. They assume:

- You run from **`research_tutorials/`**.
- Satellite directory contains `*.nc4` and matching `{SAT}_ECEF_lookup_{stem}.csv` (from TLE03).
- Outputs are written into the same satellite directory.

**Usage:**

| Platform   | ATMS | AMSU-A | SSMI-S |
|-----------|------|--------|--------|
| Windows   | `run_rfi_atms_batch.bat util\JPSS-1` | `run_rfi_amsua_batch.bat util\NOAA-15` | `run_rfi_ssmis_batch.bat util\DMSP-F17` |
| macOS/Linux | `./run_rfi_atms_batch.sh util/JPSS-1` | `./run_rfi_amsua_batch.sh util/NOAA-15` | `./run_rfi_ssmis_batch.sh util/DMSP-F17` |

The single argument is the path to the satellite directory (relative to `research_tutorials/`). Satellite name is taken from the directory name (e.g. `JPSS-1` from `util/JPSS-1`).

**Windows:** Run in Command Prompt or PowerShell with current directory `research_tutorials\` (e.g. `cd research_tutorials` then run the .bat).

**macOS/Linux:** Make scripts executable once from `research_tutorials/`: `chmod +x run_rfi_*_batch.sh`, then run as above.

Batch scripts are located in `research_tutorials/` and must be run with the working directory set to `research_tutorials/` (the scripts change into that directory automatically on macOS/Linux).
