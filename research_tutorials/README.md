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

- **Data directory**: `research_tutorials/data/` - Contains input data files

- **Data creation directory**: `research_tutorials/data_creation/` - Contains scripts to generate input data files, .arrow, which are for getting trajectories of a star and a satellite with user-specified date and time. Currently, Cas A (Cassiopeia A) and Starlink satellite are used


### Data Files

The `research_tutorials/data/` directory contains input data files of simulations including:
- One **.cut** file: Gain pattern of the MIT Westford antenna generated from TRICA software
- **.arrow** files: Trajectory files for astronomical objects (e.g., Cas A) and satellites (e.g., Starlink)
  - **Phase 1 Weather Satellite RFI Modeling**:
    - `Starlink_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow`: Starlink constellation trajectory for weather satellite RFI analysis
    - `jpss_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow`: JPSS (Suomi-NPP) weather satellite trajectory
- **CSV files** (Phase 1 Weather Satellite RFI Modeling):
  - `K-Band 23.8 GHz absolute antenna pattern.csv`: Suomi-NPP ATMS K-Band antenna gain pattern (elevation angle vs. power in dB)
  - `V-Band 50.3 GHz absolute antenna pattern.csv`: Suomi-NPP ATMS V-Band antenna gain pattern (elevation angle vs. power in dB)
- One **.tif** file: DEM (Digital Elevation Model) GeoTIFF file for terrain analysis and environmental effects modeling: area around MIT Westford antenna (USGS_OPR_MA_CentralEastern_2021_B21_be_19TBH294720.tif)
- **Markdown documentation**:
  - `tuto_radiomdl_weather_phase1_input_parameters.md`: Comprehensive documentation of input parameters for Phase 1 weather satellite RFI modeling, including trajectory file generation instructions


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
