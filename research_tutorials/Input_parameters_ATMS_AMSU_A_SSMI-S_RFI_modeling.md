# RFI Modeling Input Parameters: Reference for Expert Tuning

Reference for tuning **5G** and **Starlink ground gateway** RFI (dBW, brightness temperature K (Tb)), and for understanding **cloud/rain slant attenuation** applied to the **sum** of those Tb values before they are added to `TMBR` in `{stem}_RFI.nc4`.

The three RFI modeling scripts are:

- **ATMS:** `ATMS_RFI_modeling.py` (cross-track, nadir geometry)
- **AMSU-A:** `AMSU-A_RFI_modeling.py` (cross-track, nadir geometry)
- **SSMI-S:** `SSMI-S_RFI_modeling.py` (conical scan, fixed elevation)

Shared 5G / density logic: `src/weather_sat_nwp.py`, `src/weather_sat_mdl.py`. Gateway geometry and antennas: `src/starlink_gateway_mdl.py`. Cloud/rain grids and attenuation: `src/attenuation_mdl.py` (P.840 / P.838); nc4 merge: `copy_nc4_with_tmbr_plus_rfi` in `weather_sat_nwp.py`.

---

## 1. Transmitter power and EIRP (high impact on RFI level)

These set the effective isotropic radiated power per emitter. Linear change in EIRP (dBW) translates directly to the same dB change in received RFI power and strongly affects brightness temperature.

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `TRANSMIT_POWER_DBW` | Each sensor script (ATMS, AMSU-A, SSMI-S) | dBW | Regulated transmit power per 5G emitter. Default -33 dBW (International/FCC). |
| `GROUND_EMITTER_GAIN_MAX` | Each sensor script | dBi | Peak gain of the 5G sector antenna at boresight. Default 24.5 dBi (ITU-R M.2101 / 3GPP 8x8 phased array). |
| `EIRP_PER_EMITTER_DBW` | Each sensor script | dBW | EIRP per emitter at boresight. Computed as `TRANSMIT_POWER_DBW + GROUND_EMITTER_GAIN_MAX` (default -8.5 dBW). Passed to the RFI model as `eirp_per_emitter_dbw`. |

**Note:** EIRP per emitter is the main lever for overall RFI level. Adjust `TRANSMIT_POWER_DBW` and/or `GROUND_EMITTER_GAIN_MAX` (or set `EIRP_PER_EMITTER_DBW` directly if the script allows) to match deployment assumptions (e.g., different regulatory limits or antenna types).

---

## 2. 5G ground emitter antenna pattern (high impact on angular dependence)

The emitter antenna pattern scales the link budget via relative gain toward the satellite. It affects how much RFI is received as a function of elevation and azimuth.

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `GROUND_EMITTER_HORIZ_BW` | Each sensor script | degrees | Horizontal (azimuth) half-power beamwidth of the 5G sector antenna. Default 65.0. |
| `GROUND_EMITTER_VERT_BW` | Each sensor script | degrees | Vertical (elevation) half-power beamwidth. Default 10.0. |
| `GROUND_EMITTER_ETA_RAD` | Each sensor script | dimensionless | Radiation efficiency (0–1). Default 0.8. Used when building the pattern in `create_5g_sector_antenna_pattern`. |

These are passed into `create_5g_sector_antenna_pattern()` in `weather_sat_mdl`. The resulting pattern is used to compute relative gain toward the satellite (emitter_dec, emitter_caz), which multiplies the link budget.

**Note:** Vertical beamwidth and pattern shape strongly affect RFI at low elevation (e.g., SSMI-S at 36.9°). Tune beamwidths and, if available, replace with a measured or more detailed pattern.

---

## 3. Channel frequency and bandwidth (high impact on RFI power and Tb)

Observation frequency and bandwidth determine which harmonic falls in band and how received power is converted to brightness temperature. Bandwidth does not change RFI power in dBW but scales brightness temperature (Tb = P / (k_B * B)).

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `ATMS_CHANNEL_CONFIGS` | `ATMS_RFI_modeling.py` | (ch, Hz, Hz) | List of (channel_number, center_freq_Hz, bandwidth_Hz) for ATMS Ch 3–9. |
| `AMSUA_CHANNEL_CONFIGS` | `AMSU-A_RFI_modeling.py` | (ch, Hz, Hz) | Same for AMSU-A Ch 3–8. |
| `SSMIS_CHANNEL_CONFIGS` | `SSMI-S_RFI_modeling.py` | (ch, Hz, Hz) | Same for SSMI-S Ch 1–5. |
| `emitter_fundamental_hz_list` | Each sensor script | Hz | Fundamental frequency of the 5G emitter per channel. The second harmonic (2 * fundamental) must fall within the channel band to get non-zero RFI. Change each channel's value accordingly. |

The model uses `freq_hz` (channel center), `bandwidth_hz`, and `emitter_fundamental_freq` in the core functions `model_rfi_nwp_5g_single_time` (ATMS/AMSU-A) and `model_rfi_nwp_5g_single_time_ssmis` (SSMI-S). If the second harmonic falls outside the channel band, RFI is set to zero.

**Note:** Ensure emitter fundamental and channel center/bandwidth are consistent with the actual 5G band plan (e.g., n258/n261) and sensor channelization. Bandwidth directly scales Tb (K); center frequency affects harmonic-in-band check and atmospheric loss.

---

## 4. Second harmonic factor (high impact on RFI level)

The model assumes interference enters the sensor via the second harmonic of the 5G fundamental. The harmonic level relative to the fundamental is a fixed scaling in the link budget.

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `second_harmonic_factor` | `src/weather_sat_nwp.py` (inside `model_rfi_nwp_5g_single_time()` and `model_rfi_nwp_5g_single_time_ssmis()` functions) | linear | Fraction of fundamental power at the second harmonic. Default **1e-6** (i.e. **-60 dBc**). Multiplies the fundamental link budget before conversion to received power. |

**Note:** This is hardcoded in the core module. Changing it requires editing `weather_sat_nwp.py`. Other dBc levels: `second_harmonic_factor = 10**(dBc/10)` with dBc negative (e.g. -40 dBc → `1e-4`).

---

## 5. Emitter density and FOV area (high impact on effective EIRP and RFI)

Effective EIRP per FOV is `eirp_per_emitter_dbw + 10*log10(n_emitters)`. So density and FOV area together set the number of emitters and thus the aggregate EIRP seen by the satellite.

### 5.1 Density (per km²)

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `country_5G_sensor_channel.csv` | `research_tutorials/data/` | CSV | Rows list countries; columns `country_name`, `ISO`, `ATMS`, `AMSU_A`, `SSMI_S`. A country is included for a given sensor channel when that sensor’s column equals the channel number (integer). |
| `load_country_5g_sensor_channel_csv`, `supported_5g_countries_for_channel` | `src/weather_sat_nwp.py` | — | Load the CSV once; build `dict[str, str]` (ISO → name) per sensor and channel for `get_emitter_density_vectorized(..., supported_5g_countries=...)`. |
| Population thresholds and density values | `src/weather_sat_nwp.py` (inside `_population_to_density()` function) | population per km²; density per km² | Logic: population > 10000 (Ultra-dense urban) → 30.0; > 5000 (Dense urban) → 15.0; > 1500 (Urban) → 5.0; > 300 (Suburban) → 3; else (Open/Rural) → 1. Units: emitter density per km². |

Density is computed by `get_emitter_density_vectorized(lat, lon, supported_5g_countries=...)` using the European Union (EU) GHSL population raster and reverse geocoder for country. Only FOVs whose ISO code appears in the per-channel allowlist get non-zero density.

**Note:** Adjust the population tiers and density values in `_population_to_density` to match deployment density (e.g., urban vs suburban vs rural), which will directly affect RFI. Edit `country_5G_sensor_channel.csv` to change which countries contribute emitters for each sensor channel.

### 5.2 FOV dimensions and n_emitters

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `SENSOR_BEAMWIDTH_DEG` | `src/weather_sat_nwp.py` | degrees | Sensor name to beamwidth: ATMS 2.2, AMSU-A 3.3. Used to compute FOV ellipse axes (d_max, d_min) from SAZA and altitude for cross-track scanners. |
| `SSMIS_VBAND_D_MAX_KM`, `SSMIS_VBAND_D_MIN_KM` | `src/weather_sat_nwp.py` | km | SSMI-S V-band FOV ellipse axes (27 km, 18 km). Fixed for conical scan. |
| `SSMIS_VBAND_FOV_AREA_KM2` | `src/weather_sat_nwp.py` | km² | pi * (D_MAX/2) * (D_MIN/2). Used in get_ssmis_n_emitters_vectorized. |

For ATMS/AMSU-A, FOV area is derived from beamwidth and geometry (SAZA, altitude); n_emitters = ceil(area_km2 * density). For SSMI-S, n_emitters = ceil(fixed `SSMIS_VBAND_FOV_AREA_KM2` * density).

**Note:** Beamwidth and SSMI-S ellipse dimensions directly scale FOV area and thus n_emitters. Use sensor-specific values if they differ from the defaults.

---

## 6. Weather satellite (receiver) antenna pattern (high impact on angular distribution of RFI)

The V-band antenna pattern of the weather satellite scales the link budget as a function of direction to the emitter (off-nadir angle). It is loaded from a CSV; if the file is missing, an ITU-style fallback is used.

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `v_band_csv` (path) | Each sensor script | path string | Path to the V-band antenna pattern CSV (e.g. `data/V-Band 50.3 GHz absolute antenna pattern.csv`). Columns typically include elevation angle and gain (dB). |
| `eta_rad` (in load_weather_sat_antenna_from_csv) | Each sensor script | dimensionless | Radiation efficiency passed when loading the antenna. Default 0.99 in the scripts. |
| `valid_freqs` | Each sensor script | (Hz, Hz) | Frequency range for which the pattern is valid, e.g. (40e9, 60e9). |

**Note:** The receiver pattern has strong impact on RFI versus scan angle (e.g., nadir vs edge for ATMS/AMSU-A, and conical angle for SSMI-S). Replace the CSV with a sensor-specific pattern if available.

---

## 7. Atmospheric loss (ITU-R P.676) (moderate impact)

Atmospheric absorption is computed from elevation (and optionally distance) and affects the link budget. Same parameters are used for ATMS, AMSU-A, and SSMI-S.

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `TEMPERATURE_K` | Each sensor script | K | Atmospheric temperature for ITU-R P.676. Default 288.15 (15 °C). |
| `PRESSURE_PA` | Each sensor script | Pa | Surface pressure. Default 101325. |
| `HUMIDITY_PCT` | Each sensor script | % | Relative humidity. Default 50.0. |

These are passed as `temperature`, `pressure`, and `humidity` into the RFI model, which calls `calculate_comprehensive_atmospheric_loss_vectorized` in `weather_sat_mdl`.

**Note:** For climatological or scenario studies, adjust to representative values. Humidity and pressure have a noticeable effect on absorption, especially in the 50–55 GHz band.

---

## 8. Starlink ground gateway (high impact when gateways fall in beam)

Direct in-band RFI from fixed gateway sites. Gateway list: CLI **`--gateways_csv`** (default `research_tutorials/data/starlink_gateways_geolocations.csv`). Per-sensor script constants (names aligned across ATMS / AMSU-A / SSMI-S):

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `EIRP_PER_GATEWAY_DBW` | Each sensor script | dBW | Aggregate EIRP reference per gateway (default 70.5 dBW). |
| `N_ANTENNAS_PER_GATEWAY` | Each sensor script | count | Antennas per site; scales gateway contribution in the model. Default 40 |
| `gateway_center_freq_hz_list` | Each sensor script | Hz | Direct RFI carrier per modeled channel (typically near channel center). |
| `GATEWAY_GAIN_MAX`, `GATEWAY_HORIZ_BW`, `GATEWAY_VERT_BW`, `GATEWAY_ETA_RAD` | Each sensor script | dBi, deg, deg, — | Gateway sector pattern (same family as 5G sector builder). |
| `GATEWAY_BORESIGHT_POINTING` | Each sensor script | bool | Boresight / random-boresight behavior (see script + `starlink_gateway_mdl`). |

**Note:** Changing EIRP, N antenna of gateway, or pattern strongly changes `GATE_RFI` and the summed Tb. `CELL_RFI` / `GATE_RFI` in nc4 are **pre–cloud/rain** Tb; only the increment added to `TMBR` uses the attenuation factor.

---

## 9. Cloud and rain path attenuation (moderate impact on effective Tb in `TMBR`)

After 5G + gateway Tb are summed per channel, slant attenuation **A** (dB, ≥ 0) is computed from ITU monthly NetCDF **`itu_iclw_rain_info_MM.nc`** in `research_tutorials/data/` (month from filename stem). Increment added to `TMBR` is **(5G Tb + gateway Tb) × 10^(-A/10)**. Variable **`CLOUD_RAIN_ATT`** stores **A** on the same compact channel axis as `CELL_RFI` / `GATE_RFI`.

| Item | Location | Description |
|------|----------|-------------|
| `CLOUD_RAIN_ICLW_ABS_THRESHOLD` | Each sensor script | \|ICLW\| threshold (kg m⁻²) below which cloud term is gated off in P.840 path (default **0.05** in scripts; not a hidden default inside `attenuation_mdl`). |
| ITU NetCDF | `research_tutorials/data/` | ICLW mean/std + `rain_prob` / `rain_rate`; used by `attenuation_mdl.load_*` / `map_fovs_*` / `compute_cloud_rain_atten_db_for_fovs`. |
| RNG | Scripts | `np.random.default_rng(None)` for ICLW/rain draws unless you change the code. |

**Note:** Missing monthly file → **FileNotFoundError** (no fallback). Tune `CLOUD_RAIN_ICLW_ABS_THRESHOLD` only if your cloud climatology assumptions change.

---

## 10. SSMI-S-specific geometry (moderate impact for SSMI-S only)

SSMI-S uses a fixed slant range and elevation instead of per-FOV distance and SAZA.

| Variable name | Location | Unit | Description |
|---------------|----------|------|-------------|
| `SSMIS_SLANT_RANGE_KM` | `SSMI-S_RFI_modeling.py` | km | Slant range from satellite to FOV center. Default 1020 (DMSP-F17). |
| `SSMIS_ELEVATION_DEG` | `SSMI-S_RFI_modeling.py` | degrees | Elevation angle from ground to satellite. Default 36.9. |

Passed as `slant_range_km` and `elevation_deg` into `model_rfi_nwp_5g_single_time_ssmis`. Used for free-space path length and atmospheric path (elevation).

**Note:** Ensure values match the satellite and orbit (e.g., different DMSP or mission). Slant range and elevation drive path loss and atmospheric loss for SSMI-S.

---

## 11. Summary: parameters with the strongest effect on RFI (dBW and K)

1. **EIRP per emitter** (`EIRP_PER_EMITTER_DBW` or `TRANSMIT_POWER_DBW` + `GROUND_EMITTER_GAIN_MAX`): linear in dB for 5G RFI power.
2. **Emitter density and FOV area** (population tiers in `_population_to_density`, `SENSOR_BEAMWIDTH_DEG`, SSMI-S FOV constants): set n_emitters and thus effective EIRP (10*log10(n_emitters)).
3. **Second harmonic factor** (`second_harmonic_factor` in `weather_sat_nwp.py`): linear scaling of 5G received power; change requires code edit.
4. **Emitter fundamental frequency** (`emitter_fundamental_hz_list`): must place second harmonic in channel band; also affects free-space and atmospheric loss via frequency.
5. **5G antenna pattern** (`GROUND_EMITTER_*`): angular dependence of emitter gain.
6. **Weather satellite antenna pattern** (V-band CSV and loading parameters): angular dependence of receiver gain.
7. **Channel bandwidth**: scales brightness temperature Tb = P / (k_B * B); does not change RFI power in dBW.
8. **Atmospheric parameters** (`TEMPERATURE_K`, `PRESSURE_PA`, `HUMIDITY_PCT`): P.676 path loss (5G and gateway chains).
9. **Starlink gateway** (`EIRP_PER_GATEWAY_DBW`, `N_ANTENNAS_PER_GATEWAY`, `gateway_center_freq_hz_list`, gateway antenna constants): sets gateway Tb and summed RFI.
10. **Cloud/rain attenuation** (monthly `itu_iclw_rain_info_MM.nc`, `CLOUD_RAIN_ICLW_ABS_THRESHOLD`): scales the **summed** Tb increment into `TMBR`; does not rescale `CELL_RFI` / `GATE_RFI`.
11. **SSMI-S geometry** (`SSMIS_SLANT_RANGE_KM`, `SSMIS_ELEVATION_DEG`): path length and elevation for SSMI-S only.

---
