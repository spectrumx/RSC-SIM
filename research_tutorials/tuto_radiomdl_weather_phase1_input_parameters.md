# Weather Satellite RFI Modeling - Phase 1 Input Parameters

This document describes all configurable input parameters in `tuto_radiomdl_weather_phase1.py`. Modify these parameters to customize the simulation for different scenarios.

## Table of Contents

0. [Trajectory File Generation](#trajectory-file-generation)
1. [Time Configuration](#time-configuration)
2. [Location Configuration](#location-configuration)
3. [Frequency Configuration](#frequency-configuration)
4. [Weather Satellite Antenna Parameters](#weather-satellite-antenna-parameters)
5. [Weather Satellite Instrument Parameters](#weather-satellite-instrument-parameters)
6. [Starlink Antenna Parameters](#starlink-antenna-parameters)
7. [Starlink Transmitter Parameters](#starlink-transmitter-parameters)
8. [Observation Model Parameters](#observation-model-parameters)
9. [Data File Paths](#data-file-paths)
10. [Visualization Parameters](#visualization-parameters)

---

## Trajectory File Generation

Before running `tuto_radiomdl_weather_phase1.py`, you need to generate trajectory files (`.arrow` format) for both Starlink satellites and weather satellites (e.g., Suomi-NPP). These trajectory files are created using `research_tutorials/data_creation/compute_satellites_overflights_full_traj.py`.

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

**Note**: The time window should be wider than the observation window used in `tuto_radiomdl_weather_phase1.py` to ensure all necessary trajectory data is available.

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
   - Ensure filenames match what `tuto_radiomdl_weather_phase1.py` expects (see [Data File Paths](#data-file-paths))

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
- **Location in script**: Line 44

### `stop_window`
- **Type**: `str`
- **Default**: `"2025-11-01T08:45:00.000"`
- **Description**: Stop time for trajectory data window (ISO 8601 format). Must match the time range of available trajectory files.
- **Location in script**: Line 45

### `start_obs`
- **Type**: `datetime`
- **Default**: `datetime.strptime("2025-11-01T08:15:00.000", "%Y-%m-%dT%H:%M:%S.%f")`
- **Description**: Start time of the observation window (when Suomi-NPP is visible). Must be within `start_window` and `stop_window`.
- **Location in script**: Line 48

### `stop_obs`
- **Type**: `datetime`
- **Default**: `datetime.strptime("2025-11-01T08:21:00.000", "%Y-%m-%dT%H:%M:%S.%f")`
- **Description**: Stop time of the observation window (when Suomi-NPP is visible). Must be within `start_window` and `stop_window`.
- **Location in script**: Line 49

### `time_step`
- **Type**: `timedelta`
- **Default**: `timedelta(seconds=1)`
- **Description**: Time resolution for observation time grid. Smaller values increase computation time but provide higher temporal resolution.
- **Location in script**: Line 324

---

## Location Configuration

### `observer_lat`
- **Type**: `float`
- **Default**: `42.6129479883915`
- **Description**: Observer latitude in degrees (decimal degrees, WGS84). Default is Westford telescope location.
- **Location in script**: Line 56

### `observer_lon`
- **Type**: `float`
- **Default**: `-71.49379366344017`
- **Description**: Observer longitude in degrees (decimal degrees, WGS84). Default is Westford telescope location.
- **Location in script**: Line 57

### `observer_alt`
- **Type**: `float`
- **Default**: `86.7689687917009`
- **Description**: Observer altitude in meters above sea level. Default is Westford telescope altitude.
- **Location in script**: Line 58

### `target_lat`
- **Type**: `float`
- **Default**: `42.6129479883915`
- **Description**: Target latitude in degrees (center of resolution element). Currently set to observer location.
- **Location in script**: Line 61

### `target_lon`
- **Type**: `float`
- **Default**: `-71.49379366344017`
- **Description**: Target longitude in degrees (center of resolution element). Currently set to observer location.
- **Location in script**: Line 62

### `target_alt`
- **Type**: `float`
- **Default**: `86.7689687917009`
- **Description**: Target altitude in meters above sea level (center of resolution element). Currently set to observer altitude.
- **Location in script**: Line 63

---

## Frequency Configuration

### `freq_channels`
- **Type**: `np.ndarray`
- **Default**: `np.array([23.8e9, 50.3e9])`
- **Description**: Array of observation frequencies in Hz. Default includes K-Band (23.8 GHz) and V-Band (50.3 GHz) for Suomi-NPP ATMS.
- **Location in script**: Line 66

---

## Weather Satellite Antenna Parameters

### `k_band_csv`
- **Type**: `str`
- **Default**: `"K-Band 23.8 GHz absolute antenna pattern.csv"`
- **Description**: Filename for K-Band antenna pattern CSV file. If not found, ITU model is used as fallback.
- **Location in script**: Line 111

### `v_band_csv`
- **Type**: `str`
- **Default**: `"V-Band 50.3 GHz absolute antenna pattern.csv"`
- **Description**: Filename for V-Band antenna pattern CSV file. If not found, ITU model is used as fallback.
- **Location in script**: Line 131

### `eta_rad` (Weather Satellite)
- **Type**: `float`
- **Default**: `0.99`
- **Description**: Radiation efficiency for weather satellite antennas (dimensionless, 0-1). Suomi-NPP ATMS has >99% efficiency.
- **Location in script**: Lines 125, 145

### `valid_freqs` (K-Band)
- **Type**: `tuple`
- **Default**: `(20e9, 30e9)`
- **Description**: Valid frequency range for K-Band antenna in Hz (min, max).
- **Location in script**: Line 126

### `valid_freqs` (V-Band)
- **Type**: `tuple`
- **Default**: `(40e9, 60e9)`
- **Description**: Valid frequency range for V-Band antenna in Hz (min, max).
- **Location in script**: Line 146

### ITU Model Fallback Parameters (K-Band)
- **`gain_max`**: `50.0` (dB) - Maximum antenna gain
- **`half_beamwidth`**: `1.0` (degrees) - Half-power beamwidth
- **`alphas`**: `np.arange(0, 181, 1)` - Elevation angle grid (degrees)
- **`betas`**: `np.arange(0, 360, 1)` - Azimuth angle grid (degrees)
- **Location in script**: Lines 116-120

### ITU Model Fallback Parameters (V-Band)
- **`gain_max`**: `50.0` (dB) - Maximum antenna gain
- **`half_beamwidth`**: `0.5` (degrees) - Half-power beamwidth (narrower at higher frequency)
- **`alphas`**: `np.arange(0, 181, 1)` - Elevation angle grid (degrees)
- **`betas`**: `np.arange(0, 360, 1)` - Azimuth angle grid (degrees)
- **Location in script**: Lines 136-140

---

## Weather Satellite Instrument Parameters

### `T_phy`
- **Type**: `float`
- **Default**: `280.0`
- **Description**: Physical temperature in Kelvin. Typical for space environment.
- **Location in script**: Line 158

### `get_atms_bandwidth(freq)`
- **Type**: `function`
- **Default**: Returns `270e6` for K-Band (23.8 GHz), `180e6` for V-Band (50.3 GHz)
- **Description**: Function that returns ATMS channel bandwidth in Hz for a given frequency. ATMS bandwidths vary by channel.
- **Location in script**: Lines 166-174

### `T_RX(tim, freq)`
- **Type**: `function`
- **Default**: Returns `300.0` for K-Band (< 30 GHz), `400.0` for V-Band (≥ 30 GHz)
- **Description**: Receiver noise temperature function in Kelvin. Higher frequencies typically have higher receiver noise.
- **Location in script**: Lines 178-183

---

## Starlink Antenna Parameters

### `starlink_eta_rad`
- **Type**: `float`
- **Default**: `0.5`
- **Description**: Starlink antenna radiation efficiency (dimensionless, 0-1).
- **Location in script**: Line 248

### `starlink_gain_max`
- **Type**: `float`
- **Default**: `39.3`
- **Description**: Starlink maximum antenna gain in dBi.
- **Location in script**: Line 249

### `starlink_half_beamwidth`
- **Type**: `float`
- **Default**: `3.0`
- **Description**: Starlink half-power beamwidth in degrees.
- **Location in script**: Line 250

### `starlink_alphas`
- **Type**: `np.ndarray`
- **Default**: `np.arange(0, 181)`
- **Description**: Elevation angle grid for Starlink ITU antenna model (degrees).
- **Location in script**: Line 251

### `starlink_betas`
- **Type**: `np.ndarray`
- **Default**: `np.arange(0, 351, 10)`
- **Description**: Azimuth angle grid for Starlink ITU antenna model (degrees, 10° resolution).
- **Location in script**: Line 252

### Starlink Antenna Frequency Range
- **Type**: `tuple`
- **Default**: `(10.7e9, 12.7e9)`
- **Description**: Valid frequency range for Starlink downlink antenna in Hz (10.7-12.7 GHz).
- **Location in script**: Line 255

---

## Starlink Transmitter Parameters

### `starlink_T_phy`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Starlink transmitter physical temperature in Kelvin.
- **Location in script**: Line 258

### `starlink_freq`
- **Type**: `float`
- **Default**: `11.9e9` (Hz)
- **Description**: Starlink fundamental transmit frequency in Hz. Can be adjusted to test harmonic effects:
  - `11.9e9` Hz: 2nd harmonic (23.8 GHz) falls in K-band
  - `12.575e9` Hz: 4th harmonic (50.3 GHz) falls in V-band
  - `11.325e9` Hz: Standard Starlink frequency
- **Location in script**: Line 261

### `starlink_bw`
- **Type**: `float`
- **Default**: `250e6` (Hz)
- **Description**: Starlink transmit bandwidth in Hz (250 MHz).
- **Location in script**: Line 263

### `starlink_transmit_pow`
- **Type**: `float`
- **Default**: `-15 + 10 * np.log10(300)` (dBW)
- **Description**: Starlink transmit power in dBW. Default corresponds to EIRP calculation.
- **Location in script**: Line 264

### `starlink_harmonics`
- **Type**: `list` of `tuple`
- **Default**: `[(2.0, 0.01), (3.0, 0.003), (4.0, 0.001)]`
- **Description**: List of harmonic configurations as `(frequency_multiplier, power_reduction_factor)` tuples:
  - `(2.0, 0.01)`: 2nd harmonic at -20 dBc (1% of fundamental)
  - `(3.0, 0.003)`: 3rd harmonic at -25 dBc (0.3% of fundamental)
  - `(4.0, 0.001)`: 4th harmonic at -30 dBc (0.1% of fundamental)
- **Location in script**: Lines 278-282

---

## Observation Model Parameters

These parameters are passed to `model_weather_sat_observed_power()`:

### `earth_brightness_temp`
- **Type**: `float`
- **Default**: `280.0`
- **Description**: Earth brightness temperature in Kelvin. Typical value for Earth's surface emission.
- **Location in script**: Line 381

### `sky_brightness_temp`
- **Type**: `float`
- **Default**: `2.73`
- **Description**: Sky background brightness temperature in Kelvin. Cosmic microwave background temperature.
- **Location in script**: Line 382

### `system_temp`
- **Type**: `float`
- **Default**: `300.0`
- **Description**: System noise temperature in Kelvin. Includes receiver and other system noise contributions.
- **Location in script**: Line 383

### `starlink_eirp_dbw`
- **Type**: `float`
- **Default**: `starlink_transmit_pow` (from line 384)
- **Description**: Starlink Effective Isotropically Radiated Power (EIRP) in dBW. Passed to the observation model.
- **Location in script**: Line 384

### `starlink_fundamental_freq`
- **Type**: `float`
- **Default**: `starlink_freq` (from line 385)
- **Description**: Starlink fundamental frequency in Hz. Used for harmonic calculations.
- **Location in script**: Line 385

### `harmonics`
- **Type**: `list`
- **Default**: `starlink_harmonics` (from line 386)
- **Description**: Harmonic configuration list. Same as `starlink_harmonics` defined above.
- **Location in script**: Line 386

---

## Data File Paths

### `data_dir`
- **Type**: `str`
- **Default**: `os.path.join(script_dir, "data")`
- **Description**: Directory containing trajectory and antenna pattern data files.
- **Location in script**: Line 41

### Trajectory File Names
The script constructs filenames based on `start_window` and `stop_window`:
- **Weather satellite trajectory**: `jpss_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow`
- **Starlink trajectory**: `Starlink_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow`
- **Location in script**: Lines 78-80, 211-213

---

## Visualization Parameters

### Plot Output Files
- **`antenna_pattern_file`**: `'weather_sat_antenna_patterns.png'` - Weather satellite antenna patterns
- **`starlink_antenna_file`**: `'starlink_antenna_pattern.png'` - Starlink antenna pattern
- **`satellite_positions_file`**: `'satellite_positions.png'` - Satellite position plot
- **`output_file2`**: `'weather_sat_rfi_by_frequency.png'` - RFI power vs time plot

### Plot Configuration
- **Figure size (antenna patterns)**: `(16, 8)` inches
- **Figure size (Starlink antenna)**: `(8, 8)` inches
- **Figure size (satellite positions)**: `(12, 12)` inches
- **Figure size (RFI plot)**: `(12, 6 * len(freq_channels))` inches
- **DPI**: `300` (high resolution for publications)
- **Location in script**: Lines 414, 494, 517, 562

### Starlink Position Sampling
- **`sample_rate`**: `max(1, len(starlink_obs_data) // 1000)`
- **Description**: Samples up to 1000 Starlink points for position plot to avoid overcrowding.
- **Location in script**: Line 528

---

## Notes

1. **Polarization Mismatch**: The model assumes a -3 dB polarization loss factor (0.5 linear) due to mismatch between Starlink's circular polarization (RHCP) and Suomi-NPP's linear polarization. This is handled internally in the observation model.

2. **Harmonic Effects**: The script is configured to test harmonic interference. Adjust `starlink_freq` to align harmonics with observation bands:
   - For K-band (23.8 GHz): Use `starlink_freq = 11.9e9` (2nd harmonic)
   - For V-band (50.3 GHz): Use `starlink_freq = 12.575e9` (4th harmonic)

3. **Trajectory Data**: Ensure trajectory files exist in the `data_dir` directory with the correct naming convention based on `start_window` and `stop_window`.

4. **Antenna Pattern Fallback**: If CSV antenna pattern files are not found, the script automatically uses ITU model approximations with reasonable default parameters.

5. **Time Resolution**: The default `time_step` of 1 second provides good temporal resolution. Reducing it will increase computation time proportionally.

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
```

### Add More Frequency Channels
```python
freq_channels = np.array([23.8e9, 31.4e9, 50.3e9])  # Add 31.4 GHz channel
```

### Adjust Starlink EIRP
```python
starlink_transmit_pow = 25.0  # dBW (higher power)
```

### Modify Harmonic Suppression
```python
starlink_harmonics = [
    (2.0, 0.005),   # Stronger 2nd harmonic: -23 dBc (0.5% of fundamental)
    (3.0, 0.001),   # Stronger 3rd harmonic: -30 dBc (0.1% of fundamental)
    (4.0, 0.0005)   # Stronger 4th harmonic: -33 dBc (0.05% of fundamental)
]
```

---

*Last updated: Based on `tuto_radiomdl_weather_phase1.py`*

