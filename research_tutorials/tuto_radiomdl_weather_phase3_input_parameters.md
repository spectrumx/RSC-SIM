# Weather Satellite RFI Modeling - Phase 3 Input Parameters

This document describes all configurable input parameters in `tuto_radiomdl_weather_phase3.py`. Phase 3 extends Phase 2 with enhanced atmospheric modeling (full ITU-R P.676) and ground reflection effects. Modify these parameters to customize the simulation for different scenarios.

## Table of Contents

### Phase 1 Parameters (Inherited)
0. [Trajectory File Generation](#trajectory-file-generation)
1. [Time Configuration](#time-configuration)
2. [Location Configuration](#location-configuration)
3. [Frequency Configuration](#frequency-configuration)
4. [Weather Satellite Antenna Parameters](#weather-satellite-antenna-parameters)
5. [Weather Satellite Instrument Parameters](#weather-satellite-instrument-parameters)
6. [Starlink Antenna Parameters](#starlink-antenna-parameters)
7. [Starlink Transmitter Parameters](#starlink-transmitter-parameters)

### Phase 2 Parameters (Inherited)
8. [Ground Emitter Configuration](#ground-emitter-configuration)
9. [Ground Emitter Deployment Scenarios](#ground-emitter-deployment-scenarios)
10. [Ground Emitter Antenna Parameters](#ground-emitter-antenna-parameters)
11. [Ground Emitter Harmonic Configuration](#ground-emitter-harmonic-configuration)
12. [Out-of-Band Emission (OOBE) Configuration](#out-of-band-emission-oobe-configuration)
13. [Terrain Masking](#terrain-masking)

### Phase 3 Parameters (New)
14. [Atmospheric Conditions](#atmospheric-conditions)
15. [Enhanced Atmospheric Model (Full ITU-R P.676)](#enhanced-atmospheric-model-full-itu-r-p676)
16. [Ground Reflection Modeling](#ground-reflection-modeling)
17. [Atmospheric Refraction](#atmospheric-refraction)

### Common Parameters
18. [Observation Model Parameters](#observation-model-parameters)
19. [Data File Paths](#data-file-paths)
20. [Visualization Parameters](#visualization-parameters)

---

## Trajectory File Generation

Before running `tuto_radiomdl_weather_phase3.py`, you need to generate trajectory files (`.arrow` format) for both Starlink satellites and weather satellites (e.g., Suomi-NPP). These trajectory files are created using `research_tutorials/data_creation/compute_satellites_overflights_full_traj.py`.

**Important**: You must run this script **twice** - once to generate Starlink trajectories and once to generate weather satellite trajectories, as they are saved to separate files.

### Observer Position (Telescope Location)

The script uses Westford telescope coordinates by default. To change the observer location:

#### Option 1: Modify ECEF Coordinates (Lines 234-239)
- **`WESTFORD_X`**: X coordinate in ECEF frame (meters)
  - **Default**: `1492206.5970`
  - **Location in script**: Line 234
- **`WESTFORD_Y`**: Y coordinate in ECEF frame (meters)
  - **Default**: `-4458130.5170`
  - **Location in script**: Line 235
- **`WESTFORD_Z`**: Z coordinate in ECEF frame (meters)
  - **Default**: `4296015.5320`
  - **Location in script**: Line 236
- **`WESTFORD_Z_OFFSET`**: Altitude offset correction (meters)
  - **Default**: `0.1582435`
  - **Location in script**: Line 239

The script automatically converts ECEF coordinates to latitude/longitude/altitude using `pyproj.Transformer`.

#### Option 2: Direct Latitude/Longitude/Altitude (Modify after conversion)
After the ECEF-to-geodetic conversion (lines 245-252), you can directly modify:
- **`WESTFORD_LAT`**: Observer latitude (degrees)
- **`WESTFORD_LON`**: Observer longitude (degrees)
- **`WESTFORD_ALT`**: Observer altitude (meters above sea level)

### JPSS Satellite Selection

To filter which JPSS satellites are included in the trajectory file, modify the `jpss_names` list in the `load_satellites_from_url()` function:

- **`jpss_names`**: List of JPSS satellite names to include
  - **Default**: `['SUOMI NPP', 'NOAA 20 (JPSS-1)', 'NOAA 21 (JPSS-2)']`
  - **For Suomi-NPP only**: `['SUOMI NPP']`
  - **Location in script**: Line 205

**Example**: To generate trajectories for only Suomi-NPP:
```python
jpss_names = ['SUOMI NPP']
```

### Satellite Configuration

Control which satellite types are processed using the `satellite_configs` dictionary:

- **`satellite_configs`**: Dictionary defining satellite types to load
  - **Location in script**: Lines 263-274
  - **Structure**: Each entry has:
    - `'url'`: Celestrak URL for satellite TLE data
    - `'enabled'`: Boolean flag to enable/disable this satellite type
    - `'description'`: Human-readable description

#### For Starlink Trajectory Generation:
```python
satellite_configs = {
    'Starlink': {
        'url': "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=csv",
        'enabled': True,  # Enable Starlink
        'description': 'Starlink constellation'
    },
    'jpss': {
        'url': "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=csv",
        'enabled': False,  # Disable JPSS
        'description': 'JPSS weather satellites'
    }
}
```

#### For Suomi-NPP Trajectory Generation:
```python
satellite_configs = {
    'Starlink': {
        'url': "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=csv",
        'enabled': False,  # Disable Starlink
        'description': 'Starlink constellation'
    },
    'jpss': {
        'url': "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=csv",
        'enabled': True,  # Enable JPSS
        'description': 'JPSS weather satellites (SUOMI NPP, NOAA 20, NOAA 21 only)'
    }
}
```

### Observation Time Window of Interest

Set the time range for trajectory computation:

- **`t0`**: Start time (Skyfield time object)
  - **Default**: `ts.utc(2025, 4, 1, 12, 30, 00)`
  - **Format**: `ts.utc(year, month, day, hour, minute, second)`
  - **Location in script**: Line 314
  - **Example**: `t0 = ts.utc(2025, 11, 1, 7, 45, 0)`

- **`t1`**: End time (Skyfield time object)
  - **Default**: `ts.utc(2025, 4, 1, 13, 30, 00)`
  - **Format**: `ts.utc(year, month, day, hour, minute, second)`
  - **Location in script**: Line 315
  - **Example**: `t1 = ts.utc(2025, 11, 1, 8, 45, 0)`

**Note**: The time window should be wider than the observation window used in `tuto_radiomdl_weather_phase3.py` to ensure all necessary trajectory data is available.

### Time Resolution

Control the temporal resolution of trajectory points:

- **`time_step`**: Time step for trajectory computation
  - **Type**: `timedelta`
  - **Default**: `timedelta(milliseconds=1000)` (1 second)
  - **Location in script**: Line 320
  - **Example**: `time_step = timedelta(milliseconds=500)` for 0.5 second resolution

- **`time_round`**: Time rounding frequency (must match `time_step`)
  - **Type**: `str`
  - **Default**: `'1000ms'`
  - **Location in script**: Line 321
  - **Example**: `time_round = '500ms'` for 0.5 second resolution

### Minimum Elevation Angle

- **`altitude_degrees`**: Minimum elevation angle for satellite visibility
  - **Type**: `float`
  - **Default**: `5.0` (degrees)
  - **Location in script**: Line 98 (in `compute_satellite_trajectories()` function)
  - **Description**: Satellites below this elevation are not included in trajectories
  - **Example**: `altitude_degrees=10.0` to only include satellites above 10° elevation

### Output Directory

- **`output_dir`**: Directory for saving downloaded CSV files and output trajectory files
  - **Type**: `str`
  - **Default**: `"traj_files"`
  - **Location in script**: Line 165 (function parameter), Line 354 (filename construction)
  - **Description**: The script creates this directory if it doesn't exist. Trajectory files are saved here.

### Output File Naming

The output filename is automatically generated based on:
- Enabled satellite types (e.g., `"Starlink"` or `"jpss"`)
- Observer location (currently hardcoded as `"Westford"`)
- Start and end times

**Format**: `{sat_types}_trajectory_{observer}_{start_time}_{end_time}.arrow`

**Examples**:
- Starlink: `Starlink_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow`
- JPSS: `jpss_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow`

### Workflow Summary

1. **Generate Starlink trajectory file**:
   - Set `satellite_configs['Starlink']['enabled'] = True`
   - Set `satellite_configs['jpss']['enabled'] = False`
   - Set `t0` and `t1` to desired time window
   - Run script → Output: `Starlink_trajectory_Westford_{times}.arrow`

2. **Generate weather satellite trajectory file**:
   - Set `satellite_configs['Starlink']['enabled'] = False`
   - Set `satellite_configs['jpss']['enabled'] = True`
   - Set `jpss_names = ['SUOMI NPP']` (if only Suomi-NPP needed)
   - Set `t0` and `t1` to same time window (or wider)
   - Run script → Output: `jpss_trajectory_Westford_{times}.arrow` (or similar)

3. **Copy trajectory files**:
   - Move both `.arrow` files to `research_tutorials/data/` directory
   - Ensure filenames match what `tuto_radiomdl_weather_phase3.py` expects (see [Data File Paths](#data-file-paths))

### Additional Notes

- **Internet Connection Required**: The script downloads satellite TLE data from Celestrak. Ensure internet connectivity.
- **Computation Time**: Processing thousands of Starlink satellites can take significant time (minutes to hours depending on time window and resolution).
- **File Size**: Trajectory files can be large (hundreds of MB to GB) depending on the number of satellites and time resolution.
- **Time Window**: The trajectory time window should encompass the observation window used in the main simulation script, with some buffer.

---

## Time Configuration

### `start_window`
- **Type**: `str`
- **Default**: `"2025-11-01T07:45:00.000"`
- **Description**: Start time for trajectory data window (ISO 8601 format). Must match the time range of available trajectory files.
- **Location in script**: Line 57

### `stop_window`
- **Type**: `str`
- **Default**: `"2025-11-01T08:45:00.000"`
- **Description**: Stop time for trajectory data window (ISO 8601 format). Must match the time range of available trajectory files.
- **Location in script**: Line 58

### `start_obs`
- **Type**: `datetime`
- **Default**: `datetime.strptime("2025-11-01T08:15:00.000", "%Y-%m-%dT%H:%M:%S.%f")`
- **Description**: Start time of the observation window (when Suomi-NPP is visible). Must be within `start_window` and `stop_window`.
- **Location in script**: Line 61

### `stop_obs`
- **Type**: `datetime`
- **Default**: `datetime.strptime("2025-11-01T08:21:00.000", "%Y-%m-%dT%H:%M:%S.%f")`
- **Description**: Stop time of the observation window (when Suomi-NPP is visible). Must be within `start_window` and `stop_window`.
- **Location in script**: Line 62

### `time_step`
- **Type**: `timedelta`
- **Default**: `timedelta(seconds=1)`
- **Description**: Time resolution for observation time grid. Smaller values increase computation time but provide higher temporal resolution.
- **Location in script**: Line 537

---

## Location Configuration

### `observer_lat`
- **Type**: `float`
- **Default**: `42.6129479883915`
- **Description**: Observer latitude in degrees (decimal degrees, WGS84). Default is Westford telescope location.
- **Location in script**: Line 69

### `observer_lon`
- **Type**: `float`
- **Default**: `-71.49379366344017`
- **Description**: Observer longitude in degrees (decimal degrees, WGS84). Default is Westford telescope location.
- **Location in script**: Line 70

### `observer_alt`
- **Type**: `float`
- **Default**: `86.7689687917009`
- **Description**: Observer altitude in meters above sea level. Default is Westford telescope altitude.
- **Location in script**: Line 71

### `target_lat`
- **Type**: `float`
- **Default**: `42.6129479883915`
- **Description**: Target latitude in degrees (center of resolution element). Currently set to observer location.
- **Location in script**: Line 74

### `target_lon`
- **Type**: `float`
- **Default**: `-71.49379366344017`
- **Description**: Target longitude in degrees (center of resolution element). Currently set to observer location.
- **Location in script**: Line 75

### `target_alt`
- **Type**: `float`
- **Default**: `86.7689687917009`
- **Description**: Target altitude in meters above sea level (center of resolution element). Currently set to observer altitude.
- **Location in script**: Line 76

---

## Frequency Configuration

### `freq_channels`
- **Type**: `np.ndarray`
- **Default**: `np.array([23.8e9, 50.3e9])`
- **Description**: Array of observation frequencies in Hz. Default includes K-Band (23.8 GHz) and V-Band (50.3 GHz) for Suomi-NPP ATMS.
- **Location in script**: Line 79

---

## Weather Satellite Antenna Parameters

### `k_band_csv`
- **Type**: `str`
- **Default**: `"K-Band 23.8 GHz absolute antenna pattern.csv"`
- **Description**: Filename for K-Band antenna pattern CSV file. If not found, ITU model is used as fallback.
- **Location in script**: Line 124

### `v_band_csv`
- **Type**: `str`
- **Default**: `"V-Band 50.3 GHz absolute antenna pattern.csv"`
- **Description**: Filename for V-Band antenna pattern CSV file. If not found, ITU model is used as fallback.
- **Location in script**: Line 144

### `eta_rad` (Weather Satellite)
- **Type**: `float`
- **Default**: `0.99`
- **Description**: Radiation efficiency for weather satellite antennas (dimensionless, 0-1). Suomi-NPP ATMS has >99% efficiency.
- **Location in script**: Lines 138, 159

### `valid_freqs` (K-Band)
- **Type**: `tuple`
- **Default**: `(20e9, 30e9)`
- **Description**: Valid frequency range for K-Band antenna in Hz (min, max).
- **Location in script**: Line 139

### `valid_freqs` (V-Band)
- **Type**: `tuple`
- **Default**: `(40e9, 60e9)`
- **Description**: Valid frequency range for V-Band antenna in Hz (min, max).
- **Location in script**: Line 160

### ITU Model Fallback Parameters (K-Band)
- **`gain_max`**: `50.0` (dB) - Maximum antenna gain
- **`half_beamwidth`**: `1.0` (degrees) - Half-power beamwidth
- **`alphas`**: `np.arange(0, 181, 1)` - Elevation angle grid (degrees)
- **`betas`**: `np.arange(0, 360, 1)` - Azimuth angle grid (degrees)
- **Location in script**: Lines 129-133

### ITU Model Fallback Parameters (V-Band)
- **`gain_max`**: `50.0` (dB) - Maximum antenna gain
- **`half_beamwidth`**: `0.5` (degrees) - Half-power beamwidth (narrower at higher frequency)
- **`alphas`**: `np.arange(0, 181, 1)` - Elevation angle grid (degrees)
- **`betas`**: `np.arange(0, 360, 1)` - Azimuth angle grid (degrees)
- **Location in script**: Lines 149-153

---

## Weather Satellite Instrument Parameters

### `T_phy`
- **Type**: `float`
- **Default**: `280.0`
- **Description**: Physical temperature in Kelvin. Typical for space environment.
- **Location in script**: Line 171

### `get_atms_bandwidth(freq)`
- **Type**: `function`
- **Default**: Returns `270e6` for K-Band (23.8 GHz), `180e6` for V-Band (50.3 GHz)
- **Description**: Function that returns ATMS channel bandwidth in Hz for a given frequency. ATMS bandwidths vary by channel.
- **Location in script**: Lines 179-187

### `T_RX(tim, freq)`
- **Type**: `function`
- **Default**: Returns `300.0` for K-Band (< 30 GHz), `400.0` for V-Band (≥ 30 GHz)
- **Description**: Receiver noise temperature function in Kelvin. Higher frequencies typically have higher receiver noise.
- **Location in script**: Lines 191-196

---

## Starlink Antenna Parameters

### `starlink_eta_rad`
- **Type**: `float`
- **Default**: `0.5`
- **Description**: Starlink antenna radiation efficiency (dimensionless, 0-1).
- **Location in script**: Line 261

### `starlink_gain_max`
- **Type**: `float`
- **Default**: `39.3`
- **Description**: Starlink maximum antenna gain in dBi.
- **Location in script**: Line 262

### `starlink_half_beamwidth`
- **Type**: `float`
- **Default**: `3.0`
- **Description**: Starlink half-power beamwidth in degrees.
- **Location in script**: Line 263

### `starlink_alphas`
- **Type**: `np.ndarray`
- **Default**: `np.arange(0, 181)`
- **Description**: Elevation angle grid for Starlink ITU antenna model (degrees).
- **Location in script**: Line 264

### `starlink_betas`
- **Type**: `np.ndarray`
- **Default**: `np.arange(0, 351, 10)`
- **Description**: Azimuth angle grid for Starlink ITU antenna model (degrees, 10° resolution).
- **Location in script**: Line 265

### Starlink Antenna Frequency Range
- **Type**: `tuple`
- **Default**: `(10.7e9, 12.7e9)`
- **Description**: Valid frequency range for Starlink downlink antenna in Hz (10.7-12.7 GHz).
- **Location in script**: Line 268

---

## Starlink Transmitter Parameters

### `starlink_T_phy`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Starlink transmitter physical temperature in Kelvin.
- **Location in script**: Line 271

### `starlink_freq`
- **Type**: `float`
- **Default**: `11.9e9` (Hz)
- **Description**: Starlink fundamental transmit frequency in Hz. Can be adjusted to test harmonic effects:
  - `11.9e9` Hz: 2nd harmonic (23.8 GHz) falls in K-band
  - `12.575e9` Hz: 4th harmonic (50.3 GHz) falls in V-band
  - `11.325e9` Hz: Standard Starlink frequency
- **Location in script**: Line 274

### `starlink_bw`
- **Type**: `float`
- **Default**: `250e6` (Hz)
- **Description**: Starlink transmit bandwidth in Hz (250 MHz).
- **Location in script**: Line 275

### `starlink_transmit_pow`
- **Type**: `float`
- **Default**: `-15 + 10 * np.log10(300)` (dBW)
- **Description**: Starlink transmit power in dBW. Default corresponds to EIRP calculation.
- **Location in script**: Line 276

### `starlink_harmonics`
- **Type**: `list` of `tuple`
- **Default**: `[(2.0, 0.01), (3.0, 0.003), (4.0, 0.001)]`
- **Description**: List of harmonic configurations as `(frequency_multiplier, power_reduction_factor)` tuples:
  - `(2.0, 0.01)`: 2nd harmonic at -20 dBc (1% of fundamental)
  - `(3.0, 0.003)`: 3rd harmonic at -25 dBc (0.3% of fundamental)
  - `(4.0, 0.001)`: 4th harmonic at -30 dBc (0.1% of fundamental)
- **Location in script**: Lines 290-294

---

## Ground Emitter Configuration

**Phase 2 Feature**: Ground emitter (5G) modeling adds terrestrial interference sources to the simulation.

### `resolution_km`
- **Type**: `float`
- **Default**: `32.0`
- **Description**: Diameter of the resolution element in kilometers. Typical for ATMS radiometer.
- **Location in script**: Line 343

### `emitter_density_per_km2`
- **Type**: `float`
- **Default**: `1.0` (Urban) or `0.15` (Suburban)
- **Description**: Number of ground emitters per square kilometer within the resolution element.
  - **Suburban**: 0.1-0.5 emitters/km²
  - **Urban**: 0.5-2.0 emitters/km²
- **Location in script**: Line 375 (Suburban) or Line 446 (Urban)

### `seed`
- **Type**: `int`
- **Default**: `42`
- **Description**: Random seed for reproducible emitter distribution. Change to generate different random placements.
- **Location in script**: Line 511

---

## Ground Emitter Deployment Scenarios

The script provides two pre-configured deployment scenarios. Switch between them by commenting/uncommenting the appropriate sections.

### Scenario A: Suburban (Mid-Band 5G)
- **Location in script**: Lines 345-408 (commented out by default in current configuration)
- **Characteristics**:
  - Lower emitter density (0.1-0.5 emitters/km²)
  - Mid-band frequencies (1-6 GHz)
  - Primary frequency: 3.5 GHz (n78 band)

**Frequency options for Suburban scenario**:

| Frequency | Band | 7th Harmonic | 14th Harmonic | Notes |
|-----------|------|--------------|---------------|-------|
| 3.5 GHz | n78 | 24.5 GHz (near K-band) | 49.0 GHz (near V-band) | **Recommended for parametric studies** |
| 3.7 GHz | n77 | 25.9 GHz | 48.1 GHz | US C-band |
| 2.5 GHz | n41 | 25.0 GHz (10th) | 50.0 GHz (20th) | TDD band |
| 4.9 GHz | n79 | 24.5 GHz (5th) | 49.0 GHz (10th) | Higher mid-band |

### Scenario B: Urban (High-Band/mmWave 5G)
- **Location in script**: Lines 410-490 (active by default)
- **Characteristics**:
  - Higher emitter density (0.5-2.0 emitters/km²)
  - High-band/mmWave frequencies (24-28 GHz)
  - Primary frequency: 25.15 GHz (n258 band)

**Frequency options for Urban scenario**:

| Frequency | Band | 2nd Harmonic | Direct Interference Risk |
|-----------|------|--------------|-------------------------|
| 25.15 GHz | n258 | 50.3 GHz (**exactly V-band!**) | No (above K-band) |
| 24.25 GHz | n258 edge | 48.5 GHz | **Yes - OOBE to K-band!** |
| 28.0 GHz | n257 | 56.0 GHz | No |
| 38.5 GHz | n260 | 77.0 GHz | No |

**⚠️ Critical Note**: The n258 band (24.25-27.5 GHz) can directly interfere with K-band (23.8 GHz) through out-of-band emissions!

### Switching Scenarios

To use **Suburban (Mid-Band)** scenario:
1. Uncomment lines 374-408 (Scenario A section)
2. Comment out lines 446-490 (Scenario B section)

To use **Urban (High-Band/mmWave)** scenario (default):
1. Comment out lines 374-408 (Scenario A section)
2. Uncomment lines 446-490 (Scenario B section)

---

## Ground Emitter Antenna Parameters

### `ground_emitter_gain_max`
- **Type**: `float`
- **Default**: `18.0`
- **Description**: Maximum antenna gain in dBi. Typical for 5G sector antennas.
- **Location in script**: Line 378 (Suburban) or Line 449 (Urban)

### `ground_emitter_horiz_bw`
- **Type**: `float`
- **Default**: `65.0`
- **Description**: Horizontal (azimuth) beamwidth in degrees. Typical for sector antennas.
- **Location in script**: Line 379 (Suburban) or Line 450 (Urban)

### `ground_emitter_vert_bw`
- **Type**: `float`
- **Default**: `10.0`
- **Description**: Vertical (elevation) beamwidth in degrees. Narrow for cellular coverage.
- **Location in script**: Line 380 (Suburban) or Line 451 (Urban)

### `ground_emitter_eta_rad`
- **Type**: `float`
- **Default**: `0.8`
- **Description**: Radiation efficiency (dimensionless, 0-1).
- **Location in script**: Line 381 (Suburban) or Line 452 (Urban)

### `ground_emitter_eirp_dbw`
- **Type**: `float`
- **Default**: `30.0`
- **Description**: Effective Isotropically Radiated Power in dBW. Typical range: 20-40 dBW.
  - **Note**: EIRP = Transmit Power + Antenna Gain
  - For 18 dBi peak gain, actual transmit power ≈ 12 dBW (30 - 18)
- **Location in script**: Line 386 (Suburban) or Line 457 (Urban)

### `ground_emitter_fundamental_freq`
- **Type**: `float`
- **Default**: `25.15e9` (Urban) or `3.5e9` (Suburban)
- **Description**: Fundamental transmit frequency in Hz.
  - **Urban (High-Band)**: 25.15 GHz → 2nd harmonic at 50.3 GHz (V-band)
  - **Suburban (Mid-Band)**: 3.5 GHz → 7th harmonic at 24.5 GHz (near K-band)
- **Location in script**: Line 389 (Suburban) or Line 462 (Urban)

---

## Ground Emitter Harmonic Configuration

### `ground_emitter_harmonics`
- **Type**: `list` of `tuple`
- **Description**: List of harmonic configurations as `(frequency_multiplier, power_reduction_factor)` tuples.

**Suburban (Mid-Band) Configuration** (Lines 395-401):
```python
ground_emitter_harmonics = [
    (2.0, 0.01),     # 2nd harmonic: -20 dBc (1% of fundamental)
    (3.0, 0.003),    # 3rd harmonic: -25 dBc (0.3% of fundamental)
    (4.0, 0.001),    # 4th harmonic: -30 dBc (0.1% of fundamental)
    (7.0, 0.0001),   # 7th harmonic: -40 dBc (0.01% of fundamental) - near K-band
    (14.0, 0.00001)  # 14th harmonic: -50 dBc (0.001% of fundamental) - near V-band
]
```

**Urban (High-Band/mmWave) Configuration** (Lines 486-490):
```python
ground_emitter_harmonics = [
    (2.0, 0.01),   # 2nd harmonic: -20 dBc (1% of fundamental) - near V-band
    (3.0, 0.003),  # 3rd harmonic: -25 dBc (0.3% of fundamental)
    (4.0, 0.001),  # 4th harmonic: -30 dBc (0.1% of fundamental)
]
```

**Harmonic Suppression Reference Table**:

| Harmonic | Power Reduction Factor | dBc | Typical Use |
|----------|------------------------|-----|-------------|
| 2nd | 0.01 | -20 dBc | All scenarios |
| 3rd | 0.003 | -25 dBc | All scenarios |
| 4th | 0.001 | -30 dBc | All scenarios |
| 7th | 0.0001 | -40 dBc | Mid-band (K-band interference) |
| 14th | 0.00001 | -50 dBc | Mid-band (V-band interference) |

---

## Out-of-Band Emission (OOBE) Configuration

OOBE occurs when the fundamental frequency is close to but outside the observation bandwidth. This is particularly relevant for the n258 band (24.25 GHz) which is near K-band (23.8 GHz).

### `ground_emitter_oobe_suppression_db`
- **Type**: `float` or `None`
- **Default**: `None` (OOBE disabled)
- **Description**: OOBE suppression in dB relative to in-band power.
  - Set to `None` to disable OOBE modeling
  - Typical values: -40 to -60 dB
- **Location in script**: Line 407 (Suburban) or Line 470 (Urban)
- **Example for enabling OOBE**: `-50.0` (50 dB suppression)

### `ground_emitter_oobe_freq_offset_max`
- **Type**: `float` or `None`
- **Default**: `None` (OOBE disabled)
- **Description**: Maximum frequency offset in Hz from the observation band edge.
  - Set to `None` to disable OOBE modeling
  - Typical values: 200-500 MHz
- **Location in script**: Line 408 (Suburban) or Line 471 (Urban)
- **Example**: `500e6` (500 MHz offset range)

### OOBE Configuration for K-Band Interference Study

To study OOBE from 24.25 GHz affecting K-band (23.8 GHz), use (Lines 476-483):
```python
ground_emitter_fundamental_freq = 24.25e9  # Hz (n258 band edge)
ground_emitter_oobe_suppression_db = -50.0  # dB relative to in-band power
ground_emitter_oobe_freq_offset_max = 500e6  # Hz (500 MHz)
```

---

## Terrain Masking

**Phase 2 Feature**: Terrain masking checks line-of-sight visibility using DEM data.

### `enable_terrain_masking`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable/disable terrain-based line-of-sight checking using DEM data.
  - Uses two-stage checking: geometric horizon (fast) + DEM ray tracing (accurate)
  - Automatically skips DEM ray tracing for high-elevation emitters (>30°)
- **Location in script**: Line 803

### `dem_file`
- **Type**: `str`
- **Default**: `"USGS_OPR_MA_CentralEastern_2021_B21_be_19TBH294720.tif"`
- **Description**: Path to Digital Elevation Model (DEM) file in GeoTIFF format.
  - Used for terrain masking calculations
  - Should cover the resolution element area
- **Location in script**: Line 806
- **DEM file source**: 1 meter DEMs by USGS National Map
  - https://data.usgs.gov/datacatalog/data/USGS:77ae0551-c61e-4979-aedd-d797abdcde0e
  - https://www.usgs.gov/the-national-map-data-delivery
  - https://apps.nationalmap.gov/downloader/

---

## Atmospheric Conditions

**Phase 3 New Feature**: Full ITU-R P.676-13 atmospheric model with configurable conditions.

### `temperature`
- **Type**: `float`
- **Default**: `288.15` (K)
- **Description**: Atmospheric temperature in Kelvin.
  - Default is standard atmosphere (15°C = 288.15 K)
  - Affects oxygen and water vapor absorption line widths and intensities
- **Location in script**: Line 601

**Temperature Reference Values**:

| Condition | Temperature (K) | Temperature (°C) |
|-----------|-----------------|------------------|
| Standard Atmosphere (ISA) | 288.15 | 15.0 |
| ITU-R P.676 Reference | 288.15 | 15.0 |
| Summer Typical | 293.15 | 20.0 |
| Winter Typical | 273.15 | 0.0 |
| Tropical | 303.15 | 30.0 |

### `pressure`
- **Type**: `float`
- **Default**: `101325.0` (Pa)
- **Description**: Atmospheric pressure in Pascals.
  - Default is standard atmosphere (1 atm = 101325 Pa = 1013.25 hPa)
  - Affects pressure broadening of absorption lines
- **Location in script**: Line 602

**Pressure Reference Values**:

| Condition | Pressure (Pa) | Pressure (hPa) | Pressure (atm) |
|-----------|---------------|----------------|----------------|
| Standard Atmosphere | 101325 | 1013.25 | 1.00 |
| High Altitude (2000m) | 79498 | 794.98 | 0.78 |
| Low Pressure System | 98000 | 980.00 | 0.97 |
| High Pressure System | 103000 | 1030.00 | 1.02 |

### `humidity`
- **Type**: `float`
- **Default**: `50.0` (%)
- **Description**: Relative humidity in percent.
  - Affects water vapor absorption
  - Converted internally to water vapor density (g/m³)
- **Location in script**: Line 603

**Humidity Reference Values**:

| Condition | Relative Humidity (%) | Typical Water Vapor Density (g/m³) |
|-----------|----------------------|-----------------------------------|
| Very Dry | 20 | ~3.5 |
| Standard | 50 | ~7.5 |
| Humid | 70 | ~10.5 |
| Very Humid | 90 | ~13.5 |

**Water Vapor Density Calculation**:
The script internally converts relative humidity to water vapor density using:
```python
T_celsius = temperature - 273.15
e_sat = 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))  # Saturation vapor pressure (hPa)
e_actual = (humidity / 100.0) * e_sat  # Actual vapor pressure
water_vapor_density = (e_actual * 216.7) / temperature  # g/m³
```

---

## Enhanced Atmospheric Model (Full ITU-R P.676)

**Phase 3 New Feature**: Full ITU-R P.676-13 line-by-line atmospheric absorption model.

### `include_atmospheric_loss`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable/disable atmospheric absorption in ground-to-satellite paths.
  - When `True`: Applies atmospheric loss to ground emitter and ground reflection paths
  - When `False`: No atmospheric loss applied
- **Location in script**: Line 804

### `use_enhanced_atmospheric`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Use full ITU-R P.676-13 model instead of simplified model.
  - **`True`**: Full line-by-line calculation with 44 oxygen lines and 35 water vapor lines
  - **`False`**: Simplified frequency-dependent model (Phase 2 style)
- **Location in script**: Line 805

### ITU-R P.676-13 Model Details

When `use_enhanced_atmospheric=True`, the model uses:

**Oxygen Absorption**:
- 44 spectral lines from ITU-R P.676 Table 1
- Line center frequencies: 50.474 to 834.145 GHz
- Temperature-dependent intensity coefficients (a₁, a₂)
- Pressure-dependent line widths (a₃, a₄)
- Dry air continuum term (Nd) for non-resonant absorption

**Water Vapor Absorption**:
- 35 spectral lines from ITU-R P.676 Table 2
- Line center frequencies: 22.235 to 1919.359 GHz
- Temperature-dependent intensity coefficients (b₁, b₂)
- Pressure-dependent line widths (b₃, b₄, b₅, b₆)

**Equivalent Heights**:
- Oxygen (h₀): ~5-6 km (interpolated from ITU coefficients)
- Water vapor (h_w): ~1.6-2.1 km (ITU-R P.676 Annex 2 formula)

**Slant Path Attenuation**:
```
A(θ) = (γ₀·h₀ + γ_w·h_w) / sin(θ)
```
Where:
- γ₀: oxygen specific attenuation (dB/km)
- γ_w: water vapor specific attenuation (dB/km)
- h₀: equivalent height for oxygen (km)
- h_w: equivalent height for water vapor (km)
- θ: elevation angle (degrees)

### Typical Attenuation Values (Standard Atmosphere)

| Frequency | Band | Oxygen (dB/km) | Water Vapor (dB/km) | Zenith Total (dB) | 30° Elev (dB) |
|-----------|------|----------------|---------------------|-------------------|---------------|
| 23.8 GHz | K-Band | ~0.01 | ~0.08 | ~0.22 | ~0.44 |
| 50.3 GHz | V-Band | ~0.35 | ~0.01 | ~1.87 | ~3.73 |
| 52.5 GHz | V-Band | ~1.45 | ~0.01 | ~7.85 | ~15.70 |

---

## Ground Reflection Modeling

**Phase 3 New Feature**: Starlink main lobe ground reflection to weather satellite.

### `include_ground_reflection`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable/disable ground reflection modeling.
  - Models Starlink main lobe signals reflecting off Earth's surface to weather satellite
  - Uses Golden Section Search algorithm to find specular reflection point
- **Location in script**: Line 817

### `surface_type`
- **Type**: `str`
- **Default**: `'land'`
- **Description**: Surface type for Fresnel reflection coefficient calculation.
  - **Options**: `'land'`, `'water'`, `'ice'`
  - Affects reflection coefficient based on surface dielectric properties
- **Location in script**: Line 818

**Surface Type Properties**:

| Surface Type | Typical Reflection Coefficient | Notes |
|--------------|-------------------------------|-------|
| `'land'` | Low (0.1-0.3) | Varies with soil moisture, vegetation |
| `'water'` | High (0.5-0.9) | Smooth water has higher reflection |
| `'ice'` | Medium (0.3-0.5) | Depends on ice type and roughness |

### Ground Reflection Algorithm

The specular reflection point is found using:
1. **Golden Section Search**: 1D optimization based on Fermat's principle
2. **Minimum Path Length**: Point where total path (Starlink → reflection → weather sat) is minimized
3. **Law of Reflection**: Incidence angle = reflection angle (automatic from Fermat's principle)

The algorithm returns `None` if:
- Satellites are on opposite sides of Earth
- Satellites are too close to Earth's surface
- No valid geometric solution exists

---

## Atmospheric Refraction

**Phase 3 Feature**: Optional atmospheric refraction modeling (currently disabled by default).

### `include_refraction`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable/disable atmospheric refraction effects.
  - Affects apparent elevation angles of ground emitters
  - Extends horizon visibility due to ray bending
  - **Note**: Currently disabled by default as most emitters have high elevation angles (>30°) where refraction effects are marginal
- **Location in script**: Line 816

**Refraction Effects by Elevation**:

| Elevation | Refraction Effect | Notes |
|-----------|-------------------|-------|
| > 30° | < 0.02° | Negligible |
| 10° | ~0.1° | Small |
| 5° | ~0.3° | Moderate |
| 0° (horizon) | ~0.6° | Significant |

---

## Observation Model Parameters

These parameters are passed to `model_weather_sat_observed_power_phase3()`:

### `earth_brightness_temp`
- **Type**: `float`
- **Default**: `280.0`
- **Description**: Earth brightness temperature in Kelvin. Typical value for Earth's surface emission.
- **Location in script**: Line 799

### `sky_brightness_temp`
- **Type**: `float`
- **Default**: `2.73`
- **Description**: Sky background brightness temperature in Kelvin. Cosmic microwave background temperature.
- **Location in script**: Line 800

### `system_temp`
- **Type**: `float`
- **Default**: `300.0`
- **Description**: System noise temperature in Kelvin. Includes receiver and other system noise contributions.
- **Location in script**: Line 801

### `starlink_eirp_dbw`
- **Type**: `float`
- **Default**: `starlink_transmit_pow` (from line 802)
- **Description**: Starlink Effective Isotropically Radiated Power (EIRP) in dBW. Passed to the observation model.
- **Location in script**: Line 802

---

## Data File Paths

### `data_dir`
- **Type**: `str`
- **Default**: `os.path.join(script_dir, "data")`
- **Description**: Directory containing trajectory, antenna pattern, and DEM data files.
- **Location in script**: Line 54

### Trajectory File Names
The script constructs filenames based on `start_window` and `stop_window`:
- **Weather satellite trajectory**: `jpss_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow`
- **Starlink trajectory**: `Starlink_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow`
- **Location in script**: Lines 91-93, 224-226

### DEM File
- **`dem_file`**: Digital Elevation Model file for terrain masking
  - **Default**: `"USGS_OPR_MA_CentralEastern_2021_B21_be_19TBH294720.tif"`
  - **Location in script**: Line 806

---

## Visualization Parameters

### Plot Output Files
- **`antenna_pattern_file`**: `'weather_sat_antenna_patterns.png'` - Weather satellite antenna patterns
- **`starlink_antenna_file`**: `'starlink_antenna_pattern.png'` - Starlink antenna pattern
- **`ground_emitter_antenna_file`**: `'ground_emitter_5g_antenna_pattern.png'` - 5G antenna pattern
- **`emitter_distribution_file`**: `'ground_emitter_distribution.png'` - Emitter distribution map
- **`satellite_positions_file`**: `'satellite_positions.png'` - Satellite position plot
- **`output_file2`**: `'weather_sat_rfi_by_frequency.png'` - RFI power vs time plot
- **`output_file3`**: `'weather_sat_rfi_atmospheric_comparison.png'` - Atmospheric effects comparison (Phase 3)
- **`output_file4`**: `'weather_sat_rfi_sources_comparison.png'` - RFI sources comparison (Phase 3)

### Plot Configuration
- **Figure size (antenna patterns)**: `(16, 8)` inches
- **Figure size (Starlink antenna)**: `(8, 8)` inches
- **Figure size (5G antenna)**: `(8, 8)` inches
- **Figure size (emitter distribution)**: `(10, 10)` inches
- **Figure size (satellite positions)**: `(12, 12)` inches
- **Figure size (RFI plot)**: `(12, 6 * len(freq_channels))` inches
- **Figure size (atmospheric comparison)**: `(14, 6 * len(freq_channels))` inches
- **Figure size (sources comparison)**: `(14, 6 * len(freq_channels))` inches
- **DPI**: `300` (high resolution for publications)

---

## Notes

1. **Polarization Mismatch**: The model assumes a -3 dB polarization loss factor (0.5 linear) due to mismatch between Starlink's circular polarization (RHCP) and Suomi-NPP's linear polarization. This is handled internally in the observation model.

2. **Harmonic Effects**: The script is configured to test harmonic interference:
   - **Starlink**: Adjust `starlink_freq` to align harmonics with observation bands
   - **Ground Emitters**: Choose deployment scenario (Suburban/Urban) for different harmonic patterns

3. **Trajectory Data**: Ensure trajectory files exist in the `data_dir` directory with the correct naming convention based on `start_window` and `stop_window`.

4. **Antenna Pattern Fallback**: If CSV antenna pattern files are not found, the script automatically uses ITU model approximations with reasonable default parameters.

5. **Time Resolution**: The default `time_step` of 1 second provides good temporal resolution. Reducing it will increase computation time proportionally.

6. **Ground Emitter Distribution**: Emitters are placed randomly within the resolution element using uniform distribution. The `seed` parameter ensures reproducibility.

7. **Terrain Masking**: Requires a DEM file covering the resolution element. If DEM file is not found, terrain masking will be skipped with a warning.

8. **Atmospheric Model Selection**: Phase 3 uses full ITU-R P.676-13 by default (`use_enhanced_atmospheric=True`). Set to `False` to use Phase 2 simplified model for comparison.

9. **Ground Reflection**: The specular reflection point algorithm may return `None` for certain satellite geometries. This is handled gracefully in the model.

10. **Comparison Runs**: The script automatically runs both enhanced (Phase 3) and simplified (Phase 2) atmospheric models for comparison.

---

## Example Modifications

### Change Observation Window
```python
start_obs = datetime.strptime("2025-11-01T08:20:00.000", "%Y-%m-%dT%H:%M:%S.%f")
stop_obs = datetime.strptime("2025-11-01T08:25:00.000", "%Y-%m-%dT%H:%M:%S.%f")
```

### Change Location
```python
observer_lat = 40.0  # New latitude
observer_lon = -75.0  # New longitude
observer_alt = 100.0  # New altitude in meters
target_lat = observer_lat
target_lon = observer_lon
target_alt = observer_alt
```

### Add More Frequency Channels
```python
freq_channels = np.array([23.8e9, 31.4e9, 50.3e9])  # Add 31.4 GHz channel
```

### Change Atmospheric Conditions (Phase 3)
```python
# Tropical conditions
temperature = 303.15  # K (30°C)
pressure = 101325.0  # Pa (1 atm)
humidity = 85.0  # % (high humidity)

# Winter conditions
temperature = 273.15  # K (0°C)
pressure = 101325.0  # Pa (1 atm)
humidity = 60.0  # %

# High altitude conditions
temperature = 278.15  # K (5°C)
pressure = 79498.0  # Pa (~2000m altitude)
humidity = 40.0  # %
```

### Use Simplified Atmospheric Model (Phase 2 style)
```python
use_enhanced_atmospheric = False  # Use simplified model instead of ITU-R P.676
```

### Disable Ground Reflection
```python
include_ground_reflection = False
```

### Change Surface Type for Ground Reflection
```python
surface_type = 'water'  # For coastal or oceanic observations
surface_type = 'ice'    # For polar observations
```

### Enable Atmospheric Refraction
```python
include_refraction = True  # Enable refraction for low-elevation observations
```

### Disable Terrain Masking (faster computation)
```python
enable_terrain_masking = False
```

### Change Random Seed for Different Emitter Distribution
```python
seed = 123  # Different random placement
```

---

## Phase Comparison Summary

| Feature | Phase 1 | Phase 2 | Phase 3 |
|---------|---------|---------|---------|
| Starlink Interference | ✓ | ✓ | ✓ |
| Ground Emitter (5G) | ✗ | ✓ | ✓ |
| Terrain Masking | ✗ | ✓ | ✓ |
| Simplified Atmospheric | ✗ | ✓ | ✓ (optional) |
| Full ITU-R P.676 | ✗ | ✗ | ✓ |
| Ground Reflection | ✗ | ✗ | ✓ |
| Atmospheric Refraction | ✗ | ✗ | ✓ (optional) |
| OOBE Modeling | ✗ | ✓ | ✓ |
| Comparison Plots | ✗ | ✗ | ✓ |
| Observation Model | `model_weather_sat_observed_power()` | `model_weather_sat_observed_power_phase2()` | `model_weather_sat_observed_power_phase3()` |

---

*Last updated: Based on `tuto_radiomdl_weather_phase3.py`*

