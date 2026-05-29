### Directory Structure

The Python implementation is organized as follows:
- **`tuto_radiomdl.ipynb`**: Jupyter notebook for python which produces the same results with the Julia notebook (`Julia/test/tuto_mdl_obs_modified.ipynb`)

- **`tuto_radiomdl.py`**: Python script to run in a command line interface (CLI) that is equivalent to the above notebook

- **`tuto_radiomdl_250401.ipynb and .py`**: Testing Jupyter notebook and python script with newly generated Arrow files from the data creation scripts for different time span (see **Data Creation Scripts** section below)

- **`tuto_radiomdl_doppler.py`**: Enhanced Python script that extends `tuto_radiomdl.py` with Doppler effect analysis and compensation. It includes automatic risk assessment for satellite interference, radial velocity calculations, and physics-based Doppler correction in the frequency domain for more accurate satellite interference predictions

- **`tuto_radiomdl_transmitter.py`**: Advanced radio astronomy observation modeling with enhanced physics including Doppler effect correction and realistic transmitter characteristics modeling, extending `tuto_radiomdl_doppler.py`

- **`tuto_radiomdl_environment.py`**: Comprehensive environmental effects modeling for radio astronomy observations including terrain masking with DEM data, atmospheric refraction correction, water vapor effects, and limb refraction for space-to-space interactions, extending `tuto_radiomdl_transmitter.py`

  - Note that the DEM data (GeoTIFF) is taken from USGS 1 meter DEMS: https://data.usgs.gov/datacatalog/data/USGS:77ae0551-c61e-4979-aedd-d797abdcde0e

- **`tuto_radiomdl_direct.py`**: Ability to analyze a single satellite interference (direct effects) versus ensemble satellite interference (aggregate effects) through command-line arguments, enabling focused analysis of specific satellites or comprehensive multi-satellite impact assessment, and extending `tuto_radiomdl_environment.py`. Example commands are:
  ```
  - for aggregate effects (no argument required)
  python tuto_radiomdl_direct.py

  - for direct effects with the STARLINK-5322 satellite
  python tuto_radiomdl_direct.py --direct STARLINK-5322
  ```

- **Data directory**: `research_tutorials/data/` — Contains input data files. See **Data Files** below.

- **Data creation directory**: `research_tutorials/data_creation/` - Contains scripts to generate input data files, .arrow, which are for getting trajectories of a star and a satellite with user-specified date and time. Currently, Cas A (Cassiopeia A) and Starlink satellite are used

- **TLE utilities directory**: `research_tutorials/util/` - Generate timestamp and ECEF (Earth Center, Earth Fixed) lookup CSVs from netCDF-4 sensor observation files (.nc4) and [Space-Track](https://www.space-track.org/) TLE data.


### Weather satellite RFI modeling: two streams

- **Phase 1–3 tutorials** (`tuto_radiomdl_weather_phase1.py`, `_phase2.py`, `_phase3.py`): Target a **single field of view (FOV) / footprint** or a **certain area of interest**. They demonstrate Starlink backlobe/sidelobe interference, 5G ground emitters, atmospheric effects (ITU-R P.676), and terrain masking in a "looking-down" scenario. Used for understanding link budget and physics over one FOV or a local region.

  - **`tuto_radiomdl_weather_phase1.py`**: Phase 1 weather satellite RFI modeling tutorial for simulating interference from Starlink satellite backlobe and sidelobe emissions to weather satellites (e.g., Suomi-NPP) in a "looking-down" observation scenario. This tutorial demonstrates:
    - Coordinate transformations to weather satellite body frame (nadir, along-track, cross-track)
    - Suomi-NPP Weather satellite antenna pattern loading from CSV files (K-Band 23.8 GHz and V-Band 50.3 GHz)
    - Starlink backlobe/sidelobe interference modeling using ITU antenna patterns
    - Link budget calculations for space-to-space interference paths
    - Harmonic effects from Starlink fundamental frequency affecting observation bands
    - Earth brightness temperature and sky background modeling
    - See `tuto_radiomdl_weather_phase1_input_parameters.md` for detailed parameter documentation

  - **`tuto_radiomdl_weather_phase2.py`**: Phase 2 weather satellite RFI modeling tutorial that extends Phase 1 with ground-based 5G emitter interference modeling. This comprehensive tutorial includes:
    - All Phase 1 capabilities (Starlink backlobe interference, weather satellite modeling)
    - 5G ground emitter distribution modeling with configurable density (emitters/km²)
    - Two deployment scenarios:
      - **Suburban (Mid-Band)**: 3.5 GHz (n78 band) with 7th/14th harmonics affecting K/V-bands
      - **Urban (High-Band/mmWave)**: 25.15 GHz (n258 band) with 2nd harmonic affecting V-band
    - 5G sector antenna pattern modeling (configurable gain, beamwidth)
    - Ground emitter link budget with atmospheric absorption losses
    - Terrain masking and horizon visibility checks using DEM data
    - Out-of-band emission (OOBE) modeling for frequencies near observation bands
    - Harmonic interference analysis for both Starlink and ground emitters
    - Polarization mismatch loss modeling (Starlink circular vs. Suomi-NPP linear)
    - Comprehensive visualization: antenna patterns, emitter distribution, satellite positions, RFI power components
    - See `tuto_radiomdl_weather_phase2_input_parameters.md` for detailed parameter documentation

  - **`tuto_radiomdl_weather_phase3.py`**: Phase 3 weather satellite RFI modeling tutorial that extends Phase 2 with enhanced atmospheric effects modeling. This advanced tutorial includes:
    - All Phase 1 and Phase 2 capabilities (Starlink backlobe, ground emitter interference)
    - Comprehensive atmospheric absorption modeling using full ITU-R P.676 standard:
      - Separate oxygen and water vapor absorption components
      - Temperature, pressure, and humidity-dependent calculations
      - Frequency-dependent attenuation (especially important for V-band at 50.3 GHz)
    - Atmospheric refraction effects (optional, configurable)
    - Ground reflection modeling with configurable surface properties
    - Enhanced link budget calculations with path-integrated atmospheric losses
    - Detailed atmospheric attenuation breakdown visualization
    - Cached atmospheric calculator for improved computational performance
    - See `tuto_radiomdl_weather_phase3_input_parameters.md` for detailed parameter documentation

- **RFI modeling for NWP simulation** (`ATMS_RFI_modeling.py`, `AMSU-A_RFI_modeling.py`, `SSMI-S_RFI_modeling.py`): Designed for **numerical weather prediction (NWP) simulation** over full sensor scans. They read netCDF-4 observation files (many FOVs per file), use ECEF lookups, `src/weather_sat_nwp.py`, and `src/starlink_gateway_mdl.py` to compute **5G** (second harmonic in band) and **Starlink ground gateway** RFI (dBW and brightness temperature) per FOV, sum sources per channel, then apply **cloud/rain slant attenuation** (ITU-R P.840 / P.838 via `src/attenuation_mdl.py`) as a linear factor on the summed brightness temperature in Kelvin before writing **`TMBR_RFI`** in **`{stem}_RFI.nc4`** (native **`TMBR`** unchanged). Monthly ITU grids containing ICLW mean/std and rain fields, `data/itu_iclw_rain_info_MM.nc` at `research_tutorials/data`, are used for the calculation of cloud/rain attenuations. Batch scripts run all nc4 files in a satellite directory. See `README_RFI_modeling_for_NWP_simulation.md` for the procedure; TLE utilities in `util/` are documented in `util/README_TLE01_TLE02_TLE03.md`.

  - **`ATMS_RFI_modeling.py`**: RFI modeling for ATMS sensor on the SUOMI-NPP and JPSS-1 (NOAA-20) weather satellites

  - **`AMSU-A_RFI_modeling.py`**: RFI modeling for AMSU-A sensor on the NOAA-15, NOAA-18, NOAA-19, METOP-B, and METOP-C weather satellites

  - **`SSMI-S_RFI_modeling.py`**: RFI modeling for SSMI-S sensor on the DMSP-F17 weather satellite


### Data Files

The `research_tutorials/data/` directory contains input data files of simulations including:
- One **.cut** file: Gain pattern of the MIT Westford antenna generated from TRICA software
- **.arrow** files: Trajectory files for astronomical objects (e.g., Cas A) and satellites (e.g., Starlink)
  - **Phase 1 Weather Satellite RFI Modeling**:
    - `Starlink_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow`: Starlink constellation trajectory for weather satellite RFI analysis
    - `jpss_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow`: JPSS (Suomi-NPP) weather satellite trajectory
- **CSV files** (antenna patterns of weather sensors and Starlink gateway, and gateway geolocation data):
  - `K-Band 23.8 GHz absolute antenna pattern.csv`: ATMS K-band gain vs. elevation (dB).
  - `V-Band 50.3 GHz absolute antenna pattern.csv`: ATMS V-band gain vs. elevation (dB).
  - `AMSU-A V-Band 50.3 GHz absolute antenna pattern.csv`: AMSU-A V-band gain vs. elevation (dB).
  - `SSMI-S V-Band absolute antenna pattern.csv`: SSMI-S V-band receiver pattern for RFI scripts.
  - `starlink_gateway_antenna_pattern.csv`: Tabulated Starlilnk gateway antenna pattern used with gateway RFI modeling.
  - `starlink_gateways_geolocations.csv`: Geolocations (lat/lon) of Starlink ground gateway site for ATMS / AMSU-A / SSMI-S NWP RFI.
- **NetCDF (ITU cloud/rain for NWP RFI):** `itu_iclw_rain_info_MM.nc` (e.g. `…_08.nc` for August) — monthly integrated cloud liquid water content (ICLW) mean/std and rain fields for P.840/P.838 slant attenuation; month **MM** must match your observation stem. If absent, scripts raise **FileNotFoundError**. Note that since the file size is large, they are stored in a cloud storage.
- **GeoTIFF files**:
  - **DEM**: `USGS_OPR_MA_CentralEastern_2021_B21_be_19TBH294720.tif` — Digital Elevation Model for terrain analysis and environmental effects (area around MIT Westford antenna; see [USGS 1 meter DEMs](https://data.usgs.gov/datacatalog/data/USGS:77ae0551-c61e-4979-aedd-d797abdcde0e)).
  - **GHSL population data (GHS-POP)** for RFI modeling for NWP simulations: Please download `GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif` from EU [Global Human Settlement Layer (GHSL)](https://human-settlement.emergency.copernicus.eu/download.php?ds=pop) for Epoch: 2025, Resolution: 30 arcsec (~1 km²), and Coordinate system: WGS84. Used for 5G ground emitter density in ATMS/AMSU-A/SSMI-S RFI scripts (population > 10000 per km² → 15 emitters/km²; all other tiers → 0; see `Input_parameters_ATMS_AMSU_A_SSMI-S_RFI_modeling.md` §5.1).
- **Markdown documentation**:
  - `tuto_radiomdl_weather_phase1_input_parameters.md`: Comprehensive documentation of input parameters for Phase 1 weather satellite RFI modeling, including trajectory file generation instructions
  - `tuto_radiomdl_weather_phase2_input_parameters.md`: Comprehensive documentation of input parameters for Phase 2 weather satellite RFI modeling, including ground emitter configuration
  - `tuto_radiomdl_weather_phase3_input_parameters.md`: Comprehensive documentation of input parameters for Phase 3 weather satellite RFI modeling, including enhanced atmospheric modeling (ITU-R P.676) and ground reflection effects
  - `README_RFI_modeling_for_NWP_simulation.md`: Procedure for ATMS / AMSU-A / SSMI-S RFI for NWP: GHSL + ITU cloud/rain grids, ECEF lookups (TLE01–TLE03), 5G + Starlink gateway runs, combined outputs and `{stem}_RFI.nc4`.
  - `Input_parameters_ATMS_AMSU_A_SSMI-S_RFI_modeling.md`: Expert tuning reference: 5G emitters, Starlink gateways, channel geometry, atmosphere (P.676), and cloud/rain path attenuation (ICLW threshold, gateway EIRP/antenna).

The `research_tutorials/util/data/` directory contains TLE data of weather satellites such as SUOMI-NPP, JPSS-1, NOAA-15, NOAA-18, NOAA-19, METOP-B, METOP-C, and DMSP-F17, which were obtained through [Space-Track](https://www.space-track.org/) for a date range from 2023-08-01 to 2023-11-01 for RFI modeling for NWP simulations. These TLE data are used to identify weather satellite's ECEF coordinate at each timestamp in the sensor observation data files (.nc4).


### Data Creation Scripts

The `research_tutorials/data_creation/` directory contains Python scripts that generate Arrow input data files (trajectory files):
- **Stars**: Currently supports Cas A trajectory calculations
- **Satellites**: Currently supports Starlink trajectory calculations

The `research_tutorials/data_creation/traj_files` directory contains two input files for python scripts to generate Arrow files and it also stores generated arrow files:
- **de421.bsp**: positions for planets and their moons for time spans, e.g., https://rhodesmill.org/skyfield/planets.html
- **hipparcos.dat**: Hipparcos catalogue


### Usage

The Python implementation can be used through:
- Direct Python scripts in CLI (e.g., `tuto_radiomdl.py`)
- Jupyter notebooks (e.g., `tuto_radiomdl.ipynb`)

**Phase 1 Weather Satellite RFI Modeling:**
- Run the tutorial script: `python tuto_radiomdl_weather_phase1.py`
- Refer to `tuto_radiomdl_weather_phase1_input_parameters.md` for detailed parameter descriptions and configuration options
- Generate custom trajectory files using scripts in `data_creation/` directory (see parameter documentation for details)

**Phase 2 Weather Satellite RFI Modeling (with Ground Emitters):**
- Run the tutorial script: `python tuto_radiomdl_weather_phase2.py`
- Extends Phase 1 with 5G ground emitter interference modeling
- Configure deployment scenario by uncommenting/commenting Scenario A (Suburban) or Scenario B (Urban) in the script
- Outputs include: antenna pattern plots, ground emitter distribution map, satellite positions, and RFI power breakdown by component
- Refer to `tuto_radiomdl_weather_phase2_input_parameters.md` for detailed parameter descriptions

**Phase 3 Weather Satellite RFI Modeling (with Enhanced Atmospheric Effects):**
- Run the tutorial script: `python tuto_radiomdl_weather_phase3.py`
- Extends Phase 2 with comprehensive atmospheric modeling using ITU-R P.676 standard
- Configure atmospheric conditions (temperature, pressure, humidity) for location-specific analysis
- Features include: oxygen/water vapor absorption, atmospheric refraction, ground reflection modeling
- Outputs include: all Phase 2 outputs plus atmospheric attenuation breakdown plots
- Refer to `tuto_radiomdl_weather_phase3_input_parameters.md` for detailed parameter descriptions

**RFI modeling for NWP simulations (ATMS, AMSU-A, and SSMI-S sensors):**
- **Scripts** (process one nc4 per run; ECEF lookups from `util/` TLE02–TLE03; optional `--gateways_csv` for Starlink gateway lat/lon list):
  - `ATMS_RFI_modeling.py`: 5G + Starlink gateway RFI for ATMS (SUOMI-NPP, JPSS-1). Outputs include per-source and summed combined CSVs, top-5 summaries, `{stem}_RFI.nc4` with native `TMBR` unchanged, new `TMBR_RFI`, `CELL_RFI`, `GATE_RFI`, and `CLOUD_RAIN_ATT` (dB). Run from `research_tutorials/`; example nc4 under `util/ATMS/`.
  - `AMSU-A_RFI_modeling.py`: Same pattern for AMSU-A (NOAA/METOP). Example under `util/AMSU-A/`.
  - `SSMI-S_RFI_modeling.py`: Same pattern for SSMI-S (DMSP-F17 when SAID supported). Example under `util/SSMI-S/`.
- **Batch runs** (run all nc4 files in the corresponding satellite directory):
  - **Windows:** `run_rfi_atms_batch.bat`, `run_rfi_amsua_batch.bat`, `run_rfi_ssmis_batch.bat` for ATMS, AMSU-A, and SSMI-S sensors, respectively.
  - **macOS/Linux:** `run_rfi_atms_batch.sh`, `run_rfi_amsua_batch.sh`, `run_rfi_ssmis_batch.sh` for ATMS, AMSU-A, and SSMI-S sensors, respectively.
- **Parallel processing batch runs** (run all nc4 files in the corresponding satellite directory in parallel):
  - `run_rfi_atms_batch.py`, `run_rfi_amsua_batch.py`, `run_rfi_ssmis_batch.py` for ATMS, AMSU-A, and SSMI-S sensors, respectively. This is cross-platform (Windows, macOS, Linux).
- See [README_RFI_modeling_for_NWP_simulation.md](README_RFI_modeling_for_NWP_simulation.md) for data preparation, paths, and the combine step; [Input_parameters_ATMS_AMSU_A_SSMI-S_RFI_modeling.md](Input_parameters_ATMS_AMSU_A_SSMI-S_RFI_modeling.md) for parameter tuning.
