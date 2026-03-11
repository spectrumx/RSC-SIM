# RFI estimation: ATMS, AMSU-A, and SSMI-S

Brief procedure to run 5G ground-emitter RFI estimation for weather satellite sensors. Each nc4 file can contain data from **multiple satellites** (identified by SAID); each script processes one nc4 and uses ECEF lookups from the same sensor directory. Run one nc4 at a time, or use the batch scripts to process all nc4 files in a sensor directory.

- Note that this will require large disk space due to the file sizes of .nc4 files and generated output CSVs.

---

## 1. Download GHS-POP data (`GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif`)

European Union's Global Human Settlement Layer (GHSL) Population data (GHG-POP) needs to be downloaded from EU [Global Human Settlement Layer (GHSL)](https://human-settlement.emergency.copernicus.eu/download.php?ds=pop) for Epoch: 2025, Resolution: 30 arcsec (~1 km²), and Coordinate system: WGS84. It is zip compressed but only `GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif` file is required. Please place it at `research_tutorial/data/` directory. It is used for 5G ground emitter density in ATMS/AMSU-A/SSMI-S RFI scripts.

---

## 2. Data preparation (ECEF lookups)

Before running any RFI script, you must have ECEF lookup CSVs for each nc4 file. That is done by the TLE01–TLE03 pipeline:

1. Create **sensor-based** directories (e.g. `util/ATMS/`, `util/AMSU-A/`, `util/SSMI-S/`); put nc4 files in the corresponding sensor directory (already made for ATMS, AMSU-A, and SSMI-S).
2. Skip TLE01 (download TLE) as TLEs are already created for the satellites listed in [util/README_TLE01_TLE02_TLE03.md](util/README_TLE01_TLE02_TLE03.md).
3. Run TLE02 with `--input-dir` set to the sensor name (e.g. ATMS); then TLE03 with the same. TLE02 splits timestamps by SAID and writes `{SAT}_timestamp_{stem}.csv` per satellite; TLE03 writes `{SAT}_ECEF_lookup_{stem}.csv` per satellite.

**Full instructions:** [util/README_TLE01_TLE02_TLE03.md](util/README_TLE01_TLE02_TLE03.md).

After that, each sensor directory contains nc4 files plus per-satellite `{SAT}_timestamp_{stem}.csv` and `{SAT}_ECEF_lookup_{stem}.csv`. The RFI scripts load all `*_ECEF_lookup_{stem}.csv` for the given nc4 from the same directory.

**Directory structure (after data preparation):**

```
research_tutorials/
  util/
    data/
      JPSS-1_TLE.txt
      SUOMI-NPP_TLE.txt
      NOAA-15_TLE.txt
      DMSP-F17_TLE.txt
      ...
    ATMS/
      atms_2023080112.nc4
      SUOMI-NPP_timestamp_atms_2023080112.csv
      JPSS-1_timestamp_atms_2023080112.csv
      SUOMI-NPP_ECEF_lookup_atms_2023080112.csv
      JPSS-1_ECEF_lookup_atms_2023080112.csv
      ...
    AMSU-A/
      amsua_2023080112.nc4
      NOAA-15_ECEF_lookup_amsua_2023080112.csv
      ...
    SSMI-S/
      ssmis_2023080112.nc4
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
  run_rfi_atms_batch.py
  run_rfi_amsua_batch.py
  run_rfi_ssmis_batch.py
  README_RFI_modeling_for_NWP_simulation.md
  Input_parameters_ATMS_AMSU_A_SSMI-S_RFI_modeling.md
```

After running the RFI batch scripts (section 3), each sensor directory will also contain `{nc4_stem}_5G_RFI_chN.csv` (per channel, with columns timestamp, satellite, lat, lon, ...) and `{nc4_stem}_5G_RFI_combined.csv` (timestamp + channel Tb columns only).

---

## 3. Running RFI scripts (single nc4)

Run from **`research_tutorials/`**. Arguments: `--sensor` (required), `--nc4` (required), `--out_dir` (optional; default is the directory of the nc4 file). ECEF lookups are loaded automatically from the same directory as the nc4 (`*_ECEF_lookup_{stem}.csv`).

### ATMS (SUOMI-NPP, JPSS-1; SAID 224, 225)

```bash
python ATMS_RFI_modeling.py --sensor ATMS --nc4 util/ATMS/atms_2023080112.nc4 --out_dir util/ATMS
```

Output: `--out_dir`/`{nc4_stem}_5G_RFI_chN.csv` (channels 3–9), each with columns timestamp, satellite, lat, lon, saza, rfi_power_dBW, rfi_brightness_temperature_K.

### AMSU-A (NOAA-15/18/19, METOP-B/C)

```bash
python AMSU-A_RFI_modeling.py --sensor AMSU-A --nc4 util/AMSU-A/amsua_2023080112.nc4 --out_dir util/AMSU-A
```

Output: `--out_dir`/`{nc4_stem}_5G_RFI_chN.csv` (channels 3–8), same column layout.

### SSMI-S (only SAID 285 = DMSP-F17; SAID 286 and others get RFI = 0)

```bash
python SSMI-S_RFI_modeling.py --sensor SSMI-S --nc4 util/SSMI-S/ssmis_2023080112.nc4 --out_dir util/SSMI-S
```

Output: `--out_dir`/`{nc4_stem}_5G_RFI_chN.csv` (channels 1–5; no SAZA column). Combined CSV has timestamp and channel Tb columns only (no satellite column).

---

## 4. Batch runs (all nc4 files in a directory)

Each RFI script handles a single nc4 file. To process every nc4 in a **sensor directory**, use the batch scripts below. They assume:

- You run from **`research_tutorials/`**.
- Sensor directory (e.g. `util/ATMS/`) contains `*.nc4` and per-satellite `{SAT}_ECEF_lookup_{stem}.csv` (from TLE03).
- Outputs are written into the same sensor directory.

**Usage:**

| Platform   | ATMS | AMSU-A | SSMI-S |
|-----------|------|--------|--------|
| Windows   | `run_rfi_atms_batch.bat util/ATMS` | `run_rfi_amsua_batch.bat util/AMSU-A` | `run_rfi_ssmis_batch.bat util/SSMI-S` |
| macOS/Linux | `./run_rfi_atms_batch.sh util/ATMS` | `./run_rfi_amsua_batch.sh util/AMSU-A` | `./run_rfi_ssmis_batch.sh util/SSMI-S` |

The single argument is the path to the **sensor directory** (e.g. `util/ATMS`, `util/AMSU-A`, `util/SSMI-S`).

**Windows:** Run in Command Prompt or PowerShell with current directory `research_tutorials\` (e.g. `cd research_tutorials` then run the .bat).

**macOS/Linux:** Make scripts executable once from `research_tutorials/`: `chmod +x run_rfi_*_batch.sh`, then run as above.

Batch scripts are located in `research_tutorials/` and must be run with the working directory set to `research_tutorials/` (the scripts change into that directory automatically on macOS/Linux).

---

## 5. Parallel processing batch runs using multi-core processor

For faster batch runs on multi-core computers, use the Python batch drivers (if available) that run the same RFI logic as in section 4 but with **parallel workers**. This is **cross-platform** (Windows, macOS, Linux).

**Assumptions:** Same as section 4 (run from `research_tutorials/`, sensor directory contains `*.nc4` and per-satellite `{SAT}_ECEF_lookup_{stem}.csv`, outputs go into the same sensor directory).

**Usage:** One required argument — the path to the **sensor directory** (e.g. `util/ATMS`, `util/AMSU-A`, `util/SSMI-S`). Optional **`--workers`** (or **`-w`**) sets the number of parallel workers; if omitted, the default is CPU count minus 2. The minimum number of workers is 2.

| Sensor | Command (run from `research_tutorials/`) |
|--------|----------------------------------------|
| ATMS   | `python run_rfi_atms_batch.py util/ATMS` |
| AMSU-A | `python run_rfi_amsua_batch.py util/AMSU-A` |
| SSMI-S | `python run_rfi_ssmis_batch.py util/SSMI-S` |

**Examples with `--workers`:**
- `python run_rfi_atms_batch.py util/ATMS --workers 4` — use 4 parallel workers.
- `python run_rfi_amsua_batch.py util/AMSU-A -w 1` — run single-threaded (one nc4 at a time).

Progress is printed as each nc4 file completes, e.g. `[2/5] atms_2023080113.nc4 ... SUCCESS`.

Outputs are the same as in section 3: `{nc4_stem}_5G_RFI_chN.csv` and `{nc4_stem}_5G_RFI_combined.csv` in the given sensor directory.

---

## Note

- If you are interested in both RFI power in dBW and brightness temperature in Kelvin, then please set the argument `remove_channel_files` as `False` (`remove_channel_files=False`) at the `combine_channel_csvs()` function call, which is located around the end of each sensor's RFI modeling script (`ATMS_RFI_modeling.py`, `AMSU-A_RFI_modeling.py`, and `SSMI-S_RFI_modeling.py`). This will keep the output CSVs, RFI per channel. Currently, it is set to be `True` to save disk space.

```
...
  combined_path = combine_channel_csvs(out_dir, out_base_nc4, remove_channel_files=False)
...
```

This will remove intermidiately generated CSV files for each channel, which will save disk space.
