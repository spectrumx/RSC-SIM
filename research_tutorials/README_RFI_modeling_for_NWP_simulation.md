# RFI estimation: ATMS, AMSU-A, and SSMI-S

Procedure to run **5G ground-emitter** and **Starlink ground gateway** RFI for weather satellite sensors, then write **`{stem}_RFI.nc4`** with a new **`TMBR_RFI`** variable (native **`TMBR`** unchanged) holding summed RFI brightness temperature adjusted by **cloud/rain slant attenuation** (ITU P.840 / P.838). Each nc4 can hold **multiple satellites** (SAID); each script processes one nc4 and loads ECEF lookups from the same sensor directory. Run one nc4 at a time, or use batch / parallel batch drivers.

- Note that this will require large disk space due to the file sizes of .nc4 files and generated output CSVs.

Some input nc4 products encode **missing** float fields near **`1e11`** (nominal **`10e10`**, often slightly lower after float32 storage, e.g. **`~9.9999997952e10`**). **`src/weather_sat_nwp.py`** treats finite values **`>= NC4_MISSING_FLOAT_MIN` (`1e10`)** as that large sentinel for **lat/lon/time/SAID/etc.** (`NC4_MISSING_FLOAT`, `replace_missing_with_nan`, `obs_valid_cross_track` / `obs_valid_ssmis_conical`). **HMSL** (ATMS/AMSU-A only) uses **`NC4_MISSING_HMSL_VALUE` (`-9999`)** via **`scalar_altitude_m_from_hmsl`** (mean of valid heights or **`DEFAULT_LEO_ALTITUDE_M` (850 km)** if all missing). Loaders in **`ATMS_RFI_modeling.py`**, **`AMSU-A_RFI_modeling.py`**, and **`SSMI-S_RFI_modeling.py`** use those helpers. Invalid rows keep **row order** but get **no RFI** (default -300 dBW / 0 K); CSV lat/lon/saza may be **NaN**; cloud/rain attenuation uses **0 dB** for rows with non-finite lat/lon when building `{stem}_RFI.nc4`.

---

## 1. Download GHS-POP data (`GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif`)

European Union's Global Human Settlement Layer (GHSL) Population data (GHG-POP) needs to be downloaded from EU [Global Human Settlement Layer (GHSL)](https://human-settlement.emergency.copernicus.eu/download.php?ds=pop) for Epoch: 2025, Resolution: 30 arcsec (~1 km²), and Coordinate system: WGS84. It is zip compressed but only `GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif` file is required. Please place it at `research_tutorials/data/`. It is used for 5G ground emitter density in ATMS/AMSU-A/SSMI-S RFI scripts. Emitter density is assigned per km² from GHSL population and country allowlist (`country_5G_sensor_channel.csv`): **population > 10000 → 15 emitters/km²**; all other population tiers (> 5000, > 1500, > 300, and rural/open) → **0**. See `_population_to_density()` in `src/weather_sat_nwp.py` and `Input_parameters_ATMS_AMSU_A_SSMI-S_RFI_modeling.md` §5.1.

## 1-1. Download monthly ITU grids (`itu_iclw_rain_info_MM.nc`)
ITU cloud/rain grid file contains monthly mean/std of the integrated cloud liquid water content and rain fields for the consideration of cloud/rain attenuation. Place `itu_iclw_rain_info_MM.nc` (e.g. `…_08.nc` for August) in `research_tutorials/data/`. Month **MM** is inferred from the date token in the nc4/CSV stem: either ``sensor_yyyymmddhh...`` or ``sensor.yyyymmddhh...`` before the first ``_5G_`` segment (same rule as ``itu_iclw_rain_info_nc_path`` in ``attenuation_mdl``). If the file is missing, the RFI scripts raise **FileNotFoundError** (no silent 0 dB fallback). Note that since the file size is large, they are stored in a cloud storage.

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
      atms.2023080112.nc4
      SUOMI-NPP_timestamp_atms.2023080112.csv
      JPSS-1_timestamp_atms.2023080112.csv
      SUOMI-NPP_ECEF_lookup_atms.2023080112.csv
      JPSS-1_ECEF_lookup_atms.2023080112.csv
      ...
    AMSU-A/
      amsua.2023080112.nc4
      NOAA-15_ECEF_lookup_amsua.2023080112.csv
      ...
    SSMI-S/
      ssmis.2023080112.nc4
      DMSP-F17_ECEF_lookup_ssmis.2023080112.csv
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

After a successful run, the sensor directory typically gains (names depend on channel set): `{stem}_5G_RFI_ch*.csv`, `{stem}_Starlink_Gateway_RFI_ch*.csv`, combined `*_5G_RFI_combined.csv` and `*_Starlink_Gateway_RFI_combined.csv`, summed `*_5G_Starlink_Gateway_RFI_combined.csv`, `*_5G_Starlink_Gateway_top5.txt`, `*_5G_Starlink_Gateway_Attenuation_top5.txt` (effective Tb after path loss), and **`{stem}_RFI.nc4`** (copy of input nc4: native **`TMBR`** unchanged; new **`TMBR_RFI`** = native **`TMBR`** plus cloud/rain-scaled summed RFI on modeled channels; plus **`CELL_RFI`**, **`GATE_RFI`**, **`CLOUD_RAIN_ATT`** on the compact channel axis). Where pre-existing **`TMBR`** is missing (masked, non-finite, or large fill with value **`>= 1e10`**, plus other **`_FillValue`** / **`missing_value`** matches), those cells on **`TMBR_RFI`** for modeled channels are written as **`1e10`** (`NC4_MISSING_TMBR_OUT_RFI_NC4`; not the product fill **`10e10`**) and no RFI increment is applied there; **`CELL_RFI` / `GATE_RFI`** still hold **pre-attenuation** RFI Tb from the combined CSVs on those cells (diagnostic), except where the netCDF **mask** on **`TMBR`** suppresses writes.

**Starlink gateway OOBE:** After the direct link-budget RFI at the gateway carrier, each channel applies an out-of-band emission mask **`A(f)`** in dB vs. the **sensor channel center** `f` (assigned gateway band 51.4–52.4 GHz, `B_N` = 1 GHz; ITU-R SM.1541 / SM.329-style piecewise law in `starlink_gateway_mdl.starlink_gateway_uplink_oobe_attenuation_db`). Per-channel CSVs and merged gateway Tb use **post-OOBE** power (`dBW − A`) and Tb (`× 10^(−A/10)`). 5G is unchanged. The unified `*_5G_Starlink_Gateway_top5.txt` prints **`A(f)`** per Starlink channel above each top-5 list.

---

## 3. Running RFI scripts (single nc4)

Run from **`research_tutorials/`**. Arguments: `--sensor` (required), `--nc4` (required), `--out_dir` (optional; default is the nc4 directory), **`--gateways_csv`** (optional; default `data/starlink_gateways_geolocations.csv`). ECEF lookups load from the same folder as the nc4 (`*_ECEF_lookup_{stem}.csv`).
An example command is provided at each RFI modeling script (e.g., `ATMS_RFI_modeling.py` etc.)

### ATMS (SUOMI-NPP, JPSS-1; SAID 224, 225)

```bash
python ATMS_RFI_modeling.py --sensor ATMS --nc4 util/ATMS/atms.2023080112.nc4 --out_dir util/ATMS
```

Outputs: per-channel **5G** and **Starlink_Gateway** CSVs; combined and **summed** CSVs; top-5 text files; **`{stem}_RFI.nc4`**. Per-channel CSVs (ch 3–9) include timestamp, satellite, lat, lon, saza, `rfi_power_dBW`, `rfi_brightness_temperature_K` (Starlink columns are **after** uplink OOBE; see previous section).

### AMSU-A (NOAA-15/18/19, METOP-B/C)

```bash
python AMSU-A_RFI_modeling.py --sensor AMSU-A --nc4 util/AMSU-A/amsua.2023080112.nc4 --out_dir util/AMSU-A
```

Outputs: same pattern as ATMS for channels **3–8**.

### SSMI-S (only SAID 285 = DMSP-F17; SAID 286 and others get RFI = 0)

```bash
python SSMI-S_RFI_modeling.py --sensor SSMI-S --nc4 util/SSMI-S/ssmis.2023080112.nc4 --out_dir util/SSMI-S
```

Outputs: same dual-source pattern for channels **1–5**. Per-channel CSVs omit **SAZA**; combined Tb CSVs are timestamp + channel columns only (no satellite column).

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

Outputs match section 3 (5G + gateway CSVs, combined/summed files, top-5 files, `{stem}_RFI.nc4`).

---

## Note

- If you are interested in both RFI power in dBW and brightness temperature in Kelvin, then please set the argument `remove_channel_files` as `False` (`remove_channel_files=False`) at the `combine_channel_csvs()` function call, which is located around the end of each sensor's RFI modeling script (`ATMS_RFI_modeling.py`, `AMSU-A_RFI_modeling.py`, and `SSMI-S_RFI_modeling.py`). This will keep the output CSVs, RFI per channel. Currently, it is set to be `True` to save disk space.

```
...
  combined_path = combine_channel_csvs(out_dir, out_base_nc4, remove_channel_files=False)
...
```

This will remove intermidiately generated CSV files for each channel, which will save disk space.
