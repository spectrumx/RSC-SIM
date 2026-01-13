"""
Weather Satellite RFI Modeling Tutorial - "Looking Down" Case (Phase 3)

This script models RFI from the perspective of a weather satellite (Suomi-NPP)
looking down at Earth, including:
- Starlink backlobe interference (Phase 1)
- Ground emitter (5G) interference (Phase 2)
- Enhanced atmospheric effects (Phase 3)
- Earth brightness temperature
- Sky background
- System noise

Phase 3 extends Phase 2 with:
- Comprehensive atmospheric absorption (separate oxygen and water vapor)
- Atmospheric refraction effects (optional)
- Temperature, pressure, and humidity-dependent modeling
- Ground reflection modeling

Author: Weather Satellite RFI Modeling Team
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter
import warnings
import pyarrow as pa
import time
warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from radio_types import Antenna, Instrument, Trajectory, Constellation, Observation  # noqa: E402
from astro_mdl import antenna_mdl_ITU, power_to_temperature  # noqa: E402
from weather_sat_mdl import (  # noqa: E402
    load_weather_sat_antenna_from_csv,
    model_weather_sat_observed_power_phase3,
    create_5g_sector_antenna_pattern,
    generate_ground_emitter_distribution
)
from attenuation_mdl import get_cached_calculator  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# Time window for trajectories
start_window = "2025-11-01T07:45:00.000"
stop_window = "2025-11-01T08:45:00.000"

# Observation window (Suomi-NPP visibility at Westford)
start_obs = datetime.strptime("2025-11-01T08:15:00.000", "%Y-%m-%dT%H:%M:%S.%f")
stop_obs = datetime.strptime("2025-11-01T08:21:00.000", "%Y-%m-%dT%H:%M:%S.%f")

# Replace colon with underscore for filenames
start_window_str = start_window.replace(":", "_")
stop_window_str = stop_window.replace(":", "_")

# Observer location (Westford telescope)
observer_lat = 42.6129479883915
observer_lon = -71.49379366344017
observer_alt = 86.7689687917009

# Target location (center of resolution element - using Westford for now)
target_lat = 42.6129479883915
target_lon = -71.49379366344017
target_alt = 86.7689687917009

# Frequency channels (Suomi-NPP ATMS)
freq_channels = np.array([23.8e9, 50.3e9])  # K-Band and V-Band in Hz

# =============================================================================
# Load Weather Satellite (Suomi-NPP) Trajectory
# =============================================================================

print("="*70)
print("WEATHER SATELLITE RFI MODELING - LOOKING DOWN CASE")
print("="*70)
print()

print("Step 1: Loading Suomi-NPP weather satellite trajectory...")
file_weather_sat_path = os.path.join(
    data_dir,
    f"jpss_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
)

if not os.path.exists(file_weather_sat_path):
    print(f"ERROR: Weather satellite trajectory file not found: {file_weather_sat_path}")
    sys.exit(1)

# Load trajectory of Suomi-NPP satellite
weather_sat_traj = Trajectory.from_file(
    file_weather_sat_path,
    time_tag='timestamp',
    elevation_tag='elevations',
    azimuth_tag='azimuths',
    distance_tag='ranges_westford'
)

print(f"  ✓ Loaded {len(weather_sat_traj.get_traj())} trajectory points")
print(f"  Time range: {weather_sat_traj.get_time_bounds()[0]} to {weather_sat_traj.get_time_bounds()[1]}")

# Filter trajectory to observation window
obs_traj_df = weather_sat_traj.get_traj_between(start_obs, stop_obs)
print(f"  Observation window: {len(obs_traj_df)} points")
print()

# =============================================================================
# Load Weather Satellite Antenna Patterns
# =============================================================================

print("Step 2: Loading Suomi-NPP antenna patterns...")

# K-Band antenna pattern (23.8 GHz)
k_band_csv = os.path.join(data_dir, "K-Band 23.8 GHz absolute antenna pattern.csv")
if not os.path.exists(k_band_csv):
    print(f"WARNING: K-Band antenna pattern not found: {k_band_csv}")
    print("  Using ITU model as fallback")
    # Create ITU model as fallback
    alphas = np.arange(0, 181, 1)
    betas = np.arange(0, 360, 1)
    gain_max = 50.0  # dB (typical for weather satellite)
    half_beamwidth = 1.0  # degrees
    k_band_ant_df = antenna_mdl_ITU(gain_max, half_beamwidth, alphas, betas)
    k_band_ant = Antenna.from_dataframe(k_band_ant_df, 0.9, (20e9, 30e9))
else:
    k_band_ant = load_weather_sat_antenna_from_csv(
        k_band_csv,
        eta_rad=0.99,  # Suomi-NPP ATMS radiation efficiency is >99%
        valid_freqs=(20e9, 30e9)
    )
    print(f"  ✓ Loaded K-Band antenna pattern from {os.path.basename(k_band_csv)}")

# V-Band antenna pattern (50.3 GHz)
v_band_csv = os.path.join(data_dir, "V-Band 50.3 GHz absolute antenna pattern.csv")
if not os.path.exists(v_band_csv):
    print(f"WARNING: V-Band antenna pattern not found: {v_band_csv}")
    print("  Using ITU model as fallback")
    # Create ITU model as fallback
    alphas = np.arange(0, 181, 1)
    betas = np.arange(0, 360, 1)
    gain_max = 50.0  # dB (typical for weather satellite)
    half_beamwidth = 0.5  # degrees (narrower beam at higher frequency)
    v_band_ant_df = antenna_mdl_ITU(gain_max, half_beamwidth, alphas, betas)
    v_band_ant = Antenna.from_dataframe(v_band_ant_df, 0.9, (40e9, 60e9))
else:
    v_band_ant = load_weather_sat_antenna_from_csv(
        v_band_csv,
        eta_rad=0.99,  # Suomi-NPP ATMS radiation efficiency is >99%
        valid_freqs=(40e9, 60e9)
    )
    print(f"  ✓ Loaded V-Band antenna pattern from {os.path.basename(v_band_csv)}")
print()

# =============================================================================
# Create Weather Satellite Instruments
# =============================================================================

print("Step 3: Creating weather satellite instruments...")

# Physical temperature
T_phy = 280.0  # K (typical for space environment)

# Bandwidth function for ATMS channels
# ATMS bandwidths vary by channel:
# - 23.8 GHz (K-Band): 270 MHz
# - 50.3 GHz (V-Band): 180 MHz


def get_atms_bandwidth(freq):
    """Get ATMS channel bandwidth for given frequency (Hz)"""
    if abs(freq - 23.8e9) < 0.1e9:  # K-Band (23.8 GHz)
        return 270e6  # 270 MHz
    elif abs(freq - 50.3e9) < 0.1e9:  # V-Band (50.3 GHz)
        return 180e6  # 180 MHz
    else:
        # Default fallback (should not be used for these channels)
        return 270e6


# Receiver temperature function
def T_RX(tim, freq):
    """Receiver noise temperature (K)"""
    if freq < 30e9:
        return 300.0  # K-Band receiver noise
    else:
        return 400.0  # V-Band receiver noise (higher at higher frequencies)


# Create instruments for each frequency channel
weather_sat_instruments = []
for freq in freq_channels:
    if freq < 30e9:
        ant = k_band_ant
    else:
        ant = v_band_ant

    # Get channel-specific bandwidth
    bw = get_atms_bandwidth(freq)

    instrument = Instrument(
        ant, T_phy, freq, bw, T_RX, freq_chan=1, coords=[]
    )
    weather_sat_instruments.append(instrument)
    print(f"    - {freq/1e9:.1f} GHz: bandwidth = {bw/1e6:.0f} MHz")

print(f"  ✓ Created {len(weather_sat_instruments)} instrument(s)")
print()

# =============================================================================
# Load Starlink Constellation
# =============================================================================

print("Step 4: Loading Starlink constellation...")
file_starlink_path = os.path.join(
    data_dir,
    f"Starlink_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
)

if not os.path.exists(file_starlink_path):
    print(f"ERROR: Starlink trajectory file not found: {file_starlink_path}")
    sys.exit(1)

# Load all Starlink data
with pa.memory_map(file_starlink_path, 'r') as source:
    table = pa.ipc.open_file(source).read_all()
starlink_data = table.to_pandas()

# Rename columns (matching format from compute_satellites_overflights_full_traj.py)
starlink_data = starlink_data.rename(columns={
    'timestamp': 'times',
    'sat': 'sat',
    'azimuths': 'azimuths',
    'elevations': 'elevations',
    'ranges_westford': 'distances'
})

starlink_data['times'] = pd.to_datetime(starlink_data['times'])

print(f"  ✓ Loaded {len(starlink_data)} Starlink trajectory points")
print(f"  Total Starlink satellites: {starlink_data['sat'].nunique()}")

# Filter to observation window
starlink_obs_data = starlink_data[
    (starlink_data['times'] >= start_obs) & (starlink_data['times'] <= stop_obs)
]
print(f"  Starlink points in observation window: {len(starlink_obs_data)}")
print(f"  Unique Starlink satellites in observation window: {starlink_obs_data['sat'].nunique()}")

# Create Starlink antenna (backlobe pattern using ITU model)
# Using same parameters as tuto_radiomdl_environment.py
starlink_eta_rad = 0.5  # Radiation efficiency (same as tuto_radiomdl_environment.py)
starlink_gain_max = 39.3  # dBi (same as tuto_radiomdl_environment.py)
starlink_half_beamwidth = 3.0  # degrees (same as tuto_radiomdl_environment.py)
starlink_alphas = np.arange(0, 181)
starlink_betas = np.arange(0, 351, 10)  # Same resolution as tuto_radiomdl_environment.py
starlink_ant_df = antenna_mdl_ITU(starlink_gain_max, starlink_half_beamwidth, starlink_alphas, starlink_betas)
# Frequency range of Starlink downlink is 10.7 - 12.7 GHz
starlink_ant = Antenna.from_dataframe(starlink_ant_df, starlink_eta_rad, (10.7e9, 12.7e9))

# Create Starlink transmitter instrument (same parameters as tuto_radiomdl_environment.py)
starlink_T_phy = 0.0  # Physical temperature (same as tuto_radiomdl_environment.py)

# =============================================================================
# Option 1: testing 2nd harmonic to K-band of Suomi-NPP (23.8 GHz): use 11.9 GHz
# =============================================================================
starlink_freq = 11.9e9  # Hz (2nd harmonic to K-band)

# # =============================================================================
# # Option 2: testing 4th harmonic to V-band of Suomi-NPP (50.3 GHz): use 12.575 GHz
# # =============================================================================
# starlink_freq = 12.575e9  # Hz (4th harmonic to V-band)

starlink_bw = 250e6  # 250 MHz bandwidth (same as tuto_radiomdl_environment.py)
starlink_transmit_pow = -15 + 10 * np.log10(300)  # dBW (same as tuto_radiomdl_environment.py)


def starlink_transmit_temp(tim, freq):
    """Transmitter temperature function (same as tuto_radiomdl_environment.py)"""
    return power_to_temperature(10**(starlink_transmit_pow/10), 1.0)  # in K


starlink_transmitter = Instrument(
    starlink_ant, starlink_T_phy, starlink_freq, starlink_bw,
    starlink_transmit_temp, freq_chan=1, coords=[]
)

# Starlink harmonic configuration
starlink_harmonics = [
    (2.0, 0.01),   # 2nd harmonic: -20 dBc (1% of fundamental)
    (3.0, 0.003),  # 3rd harmonic: -25 dBc (0.3% of fundamental)
    (4.0, 0.001)   # 4th harmonic: -30 dBc (0.1% of fundamental)
]

# Create a dummy observation for Constellation (it just needs time bounds)
# We'll create a minimal Observation-like object
# Create a dummy trajectory for the observation
dummy_traj_data = pd.DataFrame({
    'times': [start_obs, stop_obs],
    'azimuths': [0.0, 0.0],
    'elevations': [90.0, 90.0],
    'distances': [1e6, 1e6]
})
dummy_traj = Trajectory(dummy_traj_data)

# Create a dummy instrument
dummy_instrument = Instrument(
    starlink_ant, 300.0, 11.7e9, 1e6,
    lambda t, f: 0.0, freq_chan=1, coords=[]
)

# Create observation
dummy_obs = Observation.from_dates(start_obs, stop_obs, dummy_traj, dummy_instrument)

# Create Constellation
# Use default link budget model (will be overridden in observation model)
starlink_constellation = Constellation.from_observation(
    starlink_obs_data,
    dummy_obs,
    starlink_transmitter,
    lnk_bdgt_mdl=None,  # Will use default
    filt_funcs=()
)

print(f"  ✓ Created Starlink constellation with {len(starlink_constellation.get_sats_name())} satellites")
print()

# =============================================================================
# Create Ground Emitter Distribution (Phase 2)
# =============================================================================

print("Step 5: Creating ground emitter (5G) distribution...")

# =============================================================================
# GROUND EMITTER CONFIGURATION
# =============================================================================
# Two deployment scenarios: Suburban (mid-band) and Urban (high-band/mmWave)
# Switch between scenarios by commenting/uncommenting the appropriate section
# =============================================================================

# Resolution element size (same for both scenarios)
resolution_km = 32.0  # km (typical for ATMS)

# # =============================================================================
# # SCENARIO A: SUBURBAN (Mid-Band 5G) - COMMENTED OUT
# # =============================================================================
# # Frequencies
# #   Primary (most representative):
# #   * 3.5 GHz (n78) - Most common globally, DEFAULT for parametric studies
# #     → 7th harmonic = 24.5 GHz (near K-band 23.8 GHz)
# #     → 14th harmonic = 49.0 GHz (near V-band 50.3 GHz)
# #   Secondary options:
# #   * 3.7 GHz (n77) - US C-band, similar to 3.5 GHz
# #     → 7th harmonic = 25.9 GHz (above K-band)
# #     → 13th harmonic = 48.1 GHz (near V-band)
# #   * 2.5 GHz (n41) - TDD band, lower mid-band
# #     → 10th harmonic = 25.0 GHz (near K-band)
# #     → 20th harmonic = 50.0 GHz (near V-band)
# #   * 4.9 GHz (n79) - Higher mid-band, less common
# #     → 5th harmonic = 24.5 GHz (near K-band)
# #     → 10th harmonic = 49.0 GHz (near V-band)
# #   Other options (less representative):
# #   * 1.8 GHz, 2.1 GHz - LTE refarming bands
# #   * 5.0 GHz - Unlicensed/WiFi bands
# #
# # Characteristics:
# #   - Lower emitter density (0.1-0.5 emitters/km²)
# #   - Mid-band frequencies (1-6 GHz, typically 3.5 GHz n78)
# #   - Moderate EIRP (20-40 dBW)
# #   - Harmonics: 7th and 14th harmonics near K/V-band
# # =============================================================================

# # Emitter distribution parameters
# emitter_density_per_km2 = 0.15  # emitters/km² (suburban: 0.1-0.5)

# # Antenna parameters (same for both scenarios)
# ground_emitter_gain_max = 18.0  # dBi
# ground_emitter_horiz_bw = 65.0  # degrees
# ground_emitter_vert_bw = 10.0  # degrees
# ground_emitter_eta_rad = 0.8  # Radiation efficiency

# # EIRP (Effective Isotropic Radiated Power)
# # Note: EIRP = Transmit Power + Antenna Gain (already includes peak gain)
# # For 18 dBi peak gain, actual transmit power ≈ 12 dBW (30 - 18)
# ground_emitter_eirp_dbw = 30.0  # dBW (typical: 20-40 dBW)

# # Fundamental frequency (Mid-Band)
# ground_emitter_fundamental_freq = 3.5e9  # Hz (3.5 GHz, n78 band)
# # Harmonic analysis for 3.5 GHz:
# #   - 7th harmonic: 24.5 GHz (near K-band 23.8 GHz)
# #   - 14th harmonic: 49.0 GHz (near V-band 50.3 GHz)

# # Harmonic configuration (Mid-Band)
# ground_emitter_harmonics = [
#     (2.0, 0.01),   # 2nd harmonic: -20 dBc (1% of fundamental)
#     (3.0, 0.003),  # 3rd harmonic: -25 dBc (0.3% of fundamental)
#     (4.0, 0.001),  # 4th harmonic: -30 dBc (0.1% of fundamental)
#     (7.0, 0.0001),  # 7th harmonic: -40 dBc (0.01% of fundamental) - near K-band
#     (14.0, 0.00001)  # 14th harmonic: -50 dBc (0.001% of fundamental) - near V-band
# ]

# # Out-of-band emission (OOBE) configuration
# # OOBE occurs when fundamental frequency is close to but outside observation bandwidth
# # For mid-band (3.5 GHz), OOBE is not applicable to K/V-band (too far)
# # Set to None to disable OOBE modeling
# ground_emitter_oobe_suppression_db = None  # dB relative to in-band power (e.g., -40 dB)
# ground_emitter_oobe_freq_offset_max = None  # Hz (e.g., 500e6 for 500 MHz)

# =============================================================================
# SCENARIO B: URBAN (High-Band/mmWave 5G) - DEFAULT, ACTIVE
# =============================================================================
# Frequencies
#  Standardized 5G mmWave bands (3GPP):
#   * n258 (26 GHz): 24.25-27.5 GHz
#     → DIRECT INTERFERENCE RISK with K-band (23.665-23.935 GHz)!
#     → 2nd harmonic: 48.5-55.0 GHz (near V-band 50.21-50.39 GHz)
#     → Center frequency: ~25.875 GHz (RECOMMENDED for parametric studies)
#   * n257 (28 GHz): 26.5-29.5 GHz
#     → Above K-band, but close (potential out-of-band emissions)
#     → 2nd harmonic: 53.0-59.0 GHz (above V-band)
#     → Center frequency: ~28.0 GHz
#   * n260 (39 GHz): 37.0-40.0 GHz
#     → Below V-band, but 2nd harmonic: 74.0-80.0 GHz (above V-band)
#     → Center frequency: ~38.5 GHz
#   * n261 (28 GHz): 27.5-28.35 GHz (subset of n257)
#     → Center frequency: ~27.925 GHz
#  Regional allocations:
#   - US: 27.5-28.35 GHz, 37-40 GHz
#   - Europe: 24.25-27.5 GHz
#   - China: 24.25-27.5 GHz, 37-43.5 GHz
#   - Japan: 27.5-28.28 GHz
#   - South Korea: 26.5-29.5 GHz
#  CRITICAL: n258 (24.25-27.5 GHz) can DIRECTLY interfere with K-band!
#   For parametric studies: Use 25.875 GHz (n258 center) to assess direct K-band interference
#
# Characteristics:
#   - Higher emitter density (0.5-2.0 emitters/km²)
#   - High-band/mmWave frequencies (24-28 GHz, typically 25.15 GHz n258)
#   - Similar EIRP (20-40 dBW)
#   - Harmonics: 2nd harmonic near V-band (50.3 GHz)
# =============================================================================
# Uncomment the following section and comment out Scenario A to use Urban configuration:
#
# Emitter distribution parameters
emitter_density_per_km2 = 1.0  # emitters/km² (urban: 0.5-2.0)

# Antenna parameters (same for both scenarios)
ground_emitter_gain_max = 18.0  # dBi
ground_emitter_horiz_bw = 65.0  # degrees
ground_emitter_vert_bw = 10.0  # degrees
ground_emitter_eta_rad = 0.8  # Radiation efficiency

# EIRP (Effective Isotropic Radiated Power)
# Note: EIRP = Transmit Power + Antenna Gain (already includes peak gain)
# For 18 dBi peak gain, actual transmit power ≈ 12 dBW (30 - 18)
ground_emitter_eirp_dbw = 30.0  # dBW (typical: 20-40 dBW)

# ---------------------------------------------------------------------------------------
# Option 1: Fundamental frequency (High-Band/mmWave) for checking 2nd harmonic to V-Band
# ---------------------------------------------------------------------------------------
ground_emitter_fundamental_freq = 25.15e9  # Hz (25.15 GHz, n258 band)
# Harmonic analysis for 25.15 GHz:
#   - 2nd harmonic: 50.3 GHz (exactly V-band center! in the range of [50.21, 50.39] GHz)
#   - Note: No direct overlap with K-band (n258 starts at 24.25 GHz)

# Out-of-band emission (OOBE) configuration (High-Band/mmWave)
# For 25.15 GHz, OOBE is less likely to affect K-band (too far)
# Set to None to disable, or configure for testing
ground_emitter_oobe_suppression_db = None  # dB relative to in-band power (e.g., -50 dB)
ground_emitter_oobe_freq_offset_max = None  # Hz (e.g., 500e6 for 500 MHz)

# # -------------------------------------------------------------------------------------
# # Option 2: Fundamental frequency (High-Band/mmWave) for checking OOBE to K-Band
# # -------------------------------------------------------------------------------------
# ground_emitter_fundamental_freq = 24.25e9  # Hz (24.25 GHz, n258 band)

# # Out-of-band emission (OOBE) configuration (High-Band/mmWave)
# # For 24.25 GHz fundamental, OOBE can extend into K-band (23.8 GHz)
# # Typical OOBE suppression: -40 to -60 dB relative to in-band power
# # Typical frequency offset: 200-500 MHz from band edge
# ground_emitter_oobe_suppression_db = -50.0  # dB relative to in-band power
# ground_emitter_oobe_freq_offset_max = 500e6  # Hz (500 MHz)

# # -------------------------------------------------------------------------------------
# # Option 3: Direct V-band interference (mmWave 5G at 50.3 GHz)
# # -------------------------------------------------------------------------------------
# # This models direct interference to Suomi-NPP V-band (Channel 3, 50.3 GHz) from
# # future/proposed mmWave 5G networks operating near V-band frequencies.
# # Note: While not yet widely deployed, mmWave 5G bands above 40 GHz are being
# # considered/proposed in some regions (e.g., 45-50 GHz range).
# #
# ground_emitter_fundamental_freq = 50.3e9  # Hz (50.3 GHz, direct V-band)
# # Interference analysis for 50.3 GHz:
# #   - DIRECT interference to V-band (50.21-50.39 GHz) - fundamental is in-band!
# #   - No harmonic contribution needed (fundamental is already in observation band)
# #   - K-band (23.8 GHz) is NOT affected by this fundamental
# #
# # # Out-of-band emission (OOBE) configuration
# # # For 50.3 GHz fundamental directly in V-band, OOBE is not applicable
# ground_emitter_oobe_suppression_db = None
# ground_emitter_oobe_freq_offset_max = None

# Harmonic configuration (High-Band/mmWave)
ground_emitter_harmonics = [
    (2.0, 0.01),   # 2nd harmonic: -20 dBc (1% of fundamental) - near V-band
    (3.0, 0.003),  # 3rd harmonic: -25 dBc (0.3% of fundamental)
    (4.0, 0.001),  # 4th harmonic: -30 dBc (0.1% of fundamental)
]

# =============================================================================
# Create antenna pattern and generate emitter distribution
# =============================================================================

# Create 5G sector antenna pattern
ground_emitter_ant = create_5g_sector_antenna_pattern(
    gain_max=ground_emitter_gain_max,
    horiz_beamwidth=ground_emitter_horiz_bw,
    vert_beamwidth=ground_emitter_vert_bw,
    eta_rad=ground_emitter_eta_rad,
    valid_freqs=(1e9, 100e9)  # Broad frequency range for 5G
)

# Generate ground emitter distribution
ground_emitters = generate_ground_emitter_distribution(
    center_lat=target_lat,
    center_lon=target_lon,
    resolution_km=resolution_km,
    emitter_density_per_km2=emitter_density_per_km2,
    seed=42  # For reproducibility
)

print(f"  ✓ Generated {len(ground_emitters)} ground emitters")
print(f"  Resolution element: {resolution_km} km diameter")
print(f"  Emitter density: {emitter_density_per_km2:.2f} emitters/km²")

print("  ✓ Created 5G sector antenna pattern")
print(f"    Max gain: {ground_emitter_gain_max} dBi")
print(f"    Horizontal beamwidth: {ground_emitter_horiz_bw}°")
print(f"    Vertical beamwidth: {ground_emitter_vert_bw}°")
print(f"    EIRP: {ground_emitter_eirp_dbw} dBW")
print(f"    Fundamental frequency: {ground_emitter_fundamental_freq/1e9:.1f} GHz")
if ground_emitter_fundamental_freq < 10e9:
    print("    Scenario: SUBURBAN (Mid-Band)")
else:
    print("    Scenario: URBAN (High-Band/mmWave)")
print()

# =============================================================================
# Generate Observation Times
# =============================================================================

print("Step 6: Generating observation time grid...")

# Create time grid (1 second resolution)
time_step = timedelta(seconds=1)
obs_times = pd.date_range(start_obs, stop_obs, freq=time_step)
obs_times_array = obs_times.to_pydatetime()

print(f"  ✓ Generated {len(obs_times_array)} observation time points")
print(f"  Time step: {time_step.total_seconds()} seconds")
print()

# =============================================================================
# Model Observations
# =============================================================================

print("Step 7: Modeling weather satellite observations (Phase 3)...")
print("  This may take a few minutes...")
print()
print("  Polarization mismatch loss:")
print("    - Starlink: Circular polarization (RHCP)")
print("    - Suomi-NPP: Linear polarization")
print("    - Polarization loss: -3 dB (50% power reduction, factor = 0.5)")
print()
print("  Harmonic effects:")
print(f"    - Starlink fundamental frequency: {starlink_freq/1e9:.3f} GHz")
print(f"    - 2nd harmonic: {2 * starlink_freq/1e9:.3f} GHz (K-band: 23.665-23.935 GHz)")
print(f"    - 4th harmonic: {4 * starlink_freq/1e9:.3f} GHz (V-band: 50.21-50.39 GHz)")
print("    - Harmonic suppression: -20 dBc (2nd), -25 dBc (3rd), -30 dBc (4th)")
print()
print("  Ground emitter (5G) frequency and harmonics:")
print(f"    - Fundamental frequency: {ground_emitter_fundamental_freq/1e9:.1f} GHz")
if ground_emitter_fundamental_freq < 10e9:
    # Mid-band harmonics
    print(f"    - 7th harmonic: {7 * ground_emitter_fundamental_freq/1e9:.1f} GHz (near K-band: 23.665-23.935 GHz)")
    print(f"    - 14th harmonic: {14 * ground_emitter_fundamental_freq/1e9:.1f} GHz (near V-band: 50.21-50.39 GHz)")
else:
    # High-band harmonics
    print(f"    - 2nd harmonic: {2 * ground_emitter_fundamental_freq/1e9:.1f} GHz (near V-band: 50.21-50.39 GHz)")
    print("    - Note: No direct overlap with K-band (fundamental above 24.25 GHz)")
print("    - Harmonic suppression: -20 dBc (2nd), -25 dBc (3rd), -30 dBc (4th), -40 dBc (7th), -50 dBc (14th)")
if ground_emitter_oobe_suppression_db is not None:
    print("  Out-of-band emissions (OOBE):")
    print(f"    - OOBE suppression: {ground_emitter_oobe_suppression_db:.1f} dB relative to in-band power")
    print(f"    - OOBE frequency offset range: ±{ground_emitter_oobe_freq_offset_max/1e6:.0f} MHz from band edge")
    # Check if OOBE applies to K-band
    freq_k_band = 23.8e9
    bw_k_band = 270e6
    freq_min_k = freq_k_band - bw_k_band / 2
    freq_max_k = freq_k_band + bw_k_band / 2
    if (ground_emitter_fundamental_freq < freq_min_k and
       (freq_min_k - ground_emitter_fundamental_freq) <= ground_emitter_oobe_freq_offset_max):
        print(f"    - OOBE from {ground_emitter_fundamental_freq/1e9:.2f} GHz may affect K-band (23.8 GHz)")
    elif (ground_emitter_fundamental_freq > freq_max_k and
          (ground_emitter_fundamental_freq - freq_max_k) <= ground_emitter_oobe_freq_offset_max):
        print(f"    - OOBE from {ground_emitter_fundamental_freq/1e9:.2f} GHz may affect K-band (23.8 GHz)")
    else:
        print("    - OOBE does not affect K-band or V-band observation frequencies")
    print("    - OOBE modeling: ENABLED")
else:
    print("  Out-of-band emissions (OOBE): NOT CONSIDERED")
print()

# =============================================================================
# Atmospheric Conditions (Phase 3)
# =============================================================================
print("  Phase 3: Full ITU-R P.676 Atmospheric Effects")
print("    Atmospheric conditions:")
temperature = 288.15  # K (15°C, standard atmosphere)
pressure = 101325.0  # Pa (1 atm, standard atmosphere)
humidity = 50.0  # % (relative humidity)
print(f"      Temperature: {temperature:.1f} K ({temperature - 273.15:.1f}°C)")
print(f"      Pressure: {pressure:.0f} Pa ({pressure/101325.0:.2f} atm)")
print(f"      Humidity: {humidity:.0f}%")
print("    Full ITU-R P.676-13 atmospheric model:")
print("      - Line-by-line oxygen absorption (44 spectral lines)")
print("      - Line-by-line water vapor absorption (35 spectral lines)")
print("      - Frequency-dependent equivalent heights")
print("      - Temperature, pressure, and humidity-dependent")
print("      - Used for BOTH K-band (23.8 GHz) and V-band (50.3 GHz)")
print()

# =============================================================================
# Diagnostic: Calculate and display absorption coefficients using full ITU-R P.676
# =============================================================================
print("  Diagnostic: Atmospheric Absorption Coefficients (Full ITU-R P.676)")
print("    Calculating for observation frequencies and ground emitter fundamental...")
print()

# Get the cached ITU-R P.676 calculator
itu_calc = get_cached_calculator()

# Convert atmospheric parameters
pressure_hpa = pressure / 100.0  # Pa to hPa

# Convert relative humidity to water vapor density (g/m³)
T_celsius = temperature - 273.15
e_sat = 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))
e_actual = (humidity / 100.0) * e_sat
water_vapor_density = (e_actual * 216.7) / temperature

print("    Atmospheric parameters for ITU-R P.676:")
print(f"      Pressure: {pressure_hpa:.2f} hPa")
print(f"      Temperature: {temperature:.2f} K ({T_celsius:.1f}°C)")
print(f"      Water vapor density: {water_vapor_density:.2f} g/m³")
print()

# Calculate for each frequency channel
for freq in freq_channels:
    freq_ghz = freq / 1e9
    print(f"    Frequency: {freq_ghz:.1f} GHz")

    # Full ITU-R P.676 model
    result_full = itu_calc.total_slant_attenuation_detailed(
        freq_ghz=freq_ghz,
        elevation_deg=90.0,  # Zenith
        pressure_hpa=pressure_hpa,
        temperature_k=temperature,
        water_vapor_density=water_vapor_density,
        include_water_vapor=True
    )

    print("      Full ITU-R P.676 Model:")
    print(f"        Oxygen specific attenuation: {result_full['gamma_o']:.4f} dB/km")
    print(f"        Water vapor specific attenuation: {result_full['gamma_w']:.4f} dB/km")
    print(f"        Oxygen equivalent height: {result_full['h0']:.2f} km")
    print(f"        Water vapor equivalent height: {result_full['h_w']:.2f} km")
    print(f"        Oxygen zenith attenuation: {result_full['A_o_zenith']:.4f} dB")
    print(f"        Water vapor zenith attenuation: {result_full['A_w_zenith']:.4f} dB")
    print(f"        Total zenith attenuation: {result_full['A_o_zenith'] + result_full['A_w_zenith']:.4f} dB")

    # Also show at 30° elevation for comparison
    result_30deg = itu_calc.total_slant_attenuation_detailed(
        freq_ghz=freq_ghz,
        elevation_deg=30.0,
        pressure_hpa=pressure_hpa,
        temperature_k=temperature,
        water_vapor_density=water_vapor_density,
        include_water_vapor=True
    )

    print(f"        Oxygen slant attenuation (30° elev): {result_30deg['A_o_slant']:.4f} dB")
    print(f"        Water vapor slant attenuation (30° elev): {result_30deg['A_w_slant']:.4f} dB")
    print(f"        Total slant attenuation (30° elev): {result_30deg['A_total_slant']:.4f} dB")
    print(f"        Air mass factor (30° elev): {result_30deg['air_mass_factor']:.4f}")

    # Simplified model (Phase 2) for comparison
    if freq_ghz < 20:
        absorption_db_per_km_simplified = 0.01
    elif freq_ghz < 40:
        absorption_db_per_km_simplified = 0.2 + 0.3 * ((freq_ghz - 20) / 20.0)
    elif freq_ghz < 60:
        oxygen_band_center = 60.0
        distance_from_peak = abs(freq_ghz - oxygen_band_center)
        if distance_from_peak < 10:
            absorption_db_per_km_simplified = 10.0 - 8.0 * (distance_from_peak / 10.0)
        else:
            absorption_db_per_km_simplified = 2.0 + 1.0 * ((freq_ghz - 40) / 20.0)
    else:
        absorption_db_per_km_simplified = 15.0

    atmospheric_path_km = 25.0
    loss_total_db_simplified = absorption_db_per_km_simplified * atmospheric_path_km

    print("      Simplified Model (Phase 2, for comparison):")
    print(f"        Total absorption: {absorption_db_per_km_simplified:.4f} dB/km")
    print(f"        Total loss (25 km path): {loss_total_db_simplified:.2f} dB")

    # Difference
    diff_db = (result_full['A_o_zenith'] + result_full['A_w_zenith']) - loss_total_db_simplified
    print("      Difference (Full ITU-R P.676 - Simplified) at zenith:")
    print(f"        Loss difference: {diff_db:.2f} dB")
    print()

# If ground emitter fundamental frequency is set, also calculate for that
if ground_emitter_fundamental_freq is not None:
    freq_fund_ghz = ground_emitter_fundamental_freq / 1e9
    print(f"    Ground Emitter Fundamental: {freq_fund_ghz:.2f} GHz")

    # Full ITU-R P.676 model
    result_full_fund = itu_calc.total_slant_attenuation_detailed(
        freq_ghz=freq_fund_ghz,
        elevation_deg=90.0,  # Zenith
        pressure_hpa=pressure_hpa,
        temperature_k=temperature,
        water_vapor_density=water_vapor_density,
        include_water_vapor=True
    )

    print("      Full ITU-R P.676 Model:")
    print(f"        Oxygen specific attenuation: {result_full_fund['gamma_o']:.4f} dB/km")
    print(f"        Water vapor specific attenuation: {result_full_fund['gamma_w']:.4f} dB/km")
    print(f"        Oxygen equivalent height: {result_full_fund['h0']:.2f} km")
    print(f"        Water vapor equivalent height: {result_full_fund['h_w']:.2f} km")
    print(f"        Oxygen zenith attenuation: {result_full_fund['A_o_zenith']:.4f} dB")
    print(f"        Water vapor zenith attenuation: {result_full_fund['A_w_zenith']:.4f} dB")
    print(f"        Total zenith attenuation: {result_full_fund['A_o_zenith'] + result_full_fund['A_w_zenith']:.4f} dB")

    # Simplified model
    if freq_fund_ghz < 20:
        absorption_db_per_km_fund_simplified = 0.01
    elif freq_fund_ghz < 40:
        absorption_db_per_km_fund_simplified = (0.2 + 0.3 *
                                                ((freq_fund_ghz - 20) / 20.0))
    elif freq_fund_ghz < 60:
        oxygen_band_center = 60.0
        distance_from_peak = abs(freq_fund_ghz - oxygen_band_center)
        if distance_from_peak < 10:
            absorption_db_per_km_fund_simplified = (10.0 - 8.0 *
                                                    (distance_from_peak / 10.0))
        else:
            absorption_db_per_km_fund_simplified = (2.0 + 1.0 *
                                                    ((freq_fund_ghz - 40) / 20.0))
    else:
        absorption_db_per_km_fund_simplified = 15.0

    loss_total_db_fund_simplified = (absorption_db_per_km_fund_simplified *
                                     atmospheric_path_km)

    print("      Simplified Model (Phase 2, for comparison):")
    print(f"        Total absorption: {absorption_db_per_km_fund_simplified:.4f} dB/km")
    print(f"        Total loss (25 km path): {loss_total_db_fund_simplified:.2f} dB")

    # Difference
    total_full_fund = result_full_fund['A_o_zenith'] + result_full_fund['A_w_zenith']
    diff_db_fund = total_full_fund - loss_total_db_fund_simplified
    print("      Difference (Full ITU-R P.676 - Simplified) at zenith:")
    print(f"        Loss difference: {diff_db_fund:.2f} dB")
    print()

print()

# Start timing for Step 7
step7_start_time = time.time()

# Model for each frequency channel with enhanced atmospheric effects
results_enhanced = {}
for f_idx, freq in enumerate(freq_channels):
    print(f"  Processing frequency channel {f_idx + 1}/{len(freq_channels)}: "
          f"{freq/1e9:.1f} GHz (with enhanced atmospheric)")

    # Time each frequency channel
    freq_start_time = time.time()

    # Use appropriate instrument for this frequency
    if freq < 30e9:
        instrument = weather_sat_instruments[0]
    else:
        instrument = weather_sat_instruments[1]

    # Model observed power using Phase 3 function (enhanced atmospheric)
    result_dict = model_weather_sat_observed_power_phase3(
        weather_sat_trajectory=weather_sat_traj,
        weather_sat_instrument=instrument,
        starlink_constellation=starlink_constellation,
        observation_times=obs_times_array,
        observer_lat=observer_lat,
        observer_lon=observer_lon,
        observer_alt=observer_alt,
        target_lat=target_lat,
        target_lon=target_lon,
        target_alt=target_alt,
        freq_channels=np.array([freq]),
        ground_emitters=ground_emitters,
        ground_emitter_antenna=ground_emitter_ant,
        ground_emitter_eirp_dbw=ground_emitter_eirp_dbw,
        earth_brightness_temp=280.0,
        sky_brightness_temp=2.73,
        system_temp=300.0,
        starlink_eirp_dbw=starlink_transmit_pow,
        enable_terrain_masking=True,
        include_atmospheric_loss=True,
        use_enhanced_atmospheric=True,
        dem_file=os.path.join(data_dir, "USGS_OPR_MA_CentralEastern_2021_B21_be_19TBH294720.tif"),
        starlink_fundamental_freq=starlink_freq,
        harmonics=starlink_harmonics,
        ground_emitter_fundamental_freq=ground_emitter_fundamental_freq,
        ground_emitter_harmonics=ground_emitter_harmonics,
        ground_emitter_oobe_suppression_db=ground_emitter_oobe_suppression_db,
        ground_emitter_oobe_freq_offset_max=ground_emitter_oobe_freq_offset_max,
        temperature=temperature,
        pressure=pressure,
        humidity=humidity,
        include_refraction=False,
        include_ground_reflection=True,
        surface_type='land'
    )

    results_enhanced[freq] = result_dict

    # Report timing for this frequency channel
    freq_elapsed = time.time() - freq_start_time
    print(f"    ✓ Completed in {freq_elapsed:.1f} seconds")
    print()

# Report total timing for Step 7
step7_elapsed = time.time() - step7_start_time
print(f"  Total computation time for Step 7: {step7_elapsed:.1f} seconds ({step7_elapsed/60:.2f} minutes)")
print()

# Use enhanced results as primary results for plotting
results = results_enhanced

# =============================================================================
# Visualization
# =============================================================================

print("Step 8: Creating visualizations...")

# =============================================================================
# Plot Antenna Patterns
# =============================================================================

print("  Plotting antenna patterns...")

# Plot Suomi-NPP antenna patterns (side-by-side for K-Band and V-Band)
fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})


def plot_antenna_pattern_polar(ax, antenna, title, max_gain_db):
    """
    Plot antenna pattern in polar view with proper formatting.

    Args:
        ax: Matplotlib polar axes
        antenna: Antenna object
        title: Plot title
        max_gain_db: Maximum gain in dB
    """
    # Get single beta slice (beta=0) since pattern is azimuthally symmetric
    alphas, gains = antenna.get_slice_gain(0.0)

    # Convert gains to dB
    gains_db = 10 * np.log10(gains)

    # Sort by alpha to ensure smooth plotting
    sort_idx = np.argsort(alphas)
    alphas_sorted = alphas[sort_idx]
    gains_db_sorted = gains_db[sort_idx]

    # Map elevation (alpha) to polar angle
    # For antenna patterns: 0° elevation (nadir) should be at 0° (right side)
    # Elevation: 0° = nadir, 90° = horizon, 180° = opposite nadir
    # For polar plot: map elevation to 0-360 degrees
    # If elevation is negative, map to 180-360 range
    elevation_mapped = np.where(alphas_sorted < 0,
                                alphas_sorted + 360,
                                alphas_sorted)
    theta_rad = np.deg2rad(elevation_mapped)

    # For polar plot radius, we need positive values
    # Use offset similar to plot_suomi_antenna_patterns.py
    power_min = gains_db_sorted.min()
    power_offset = gains_db_sorted - power_min  # Shift so minimum is at 0
    power_offset = power_offset + 1  # Add small offset to avoid zero radius

    # Plot the pattern
    ax.plot(theta_rad, power_offset, 'b-', linewidth=2, label='Gain Pattern')

    # Set theta zero location to East (right side)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)  # Counter-clockwise (standard)

    # Set radial axis labels (convert back to dB)
    r_max = power_offset.max()
    r_min = power_offset.min()
    n_ticks = 6
    r_ticks = np.linspace(r_min, r_max, n_ticks)
    r_labels = [f'{p + power_min - 1:.0f}' for p in r_ticks]
    ax.set_rticks(r_ticks)
    ax.set_yticklabels(r_labels)

    # Set title and formatting
    ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Max Gain: {max_gain_db:.1f} dB',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_5g_antenna_pattern_polar(ax, antenna, title, max_gain_db):
    """
    Plot 5G sector antenna pattern in polar view with horizontal (azimuth) slice.
    Uses 0° at North (top) for better visualization of sector pattern.

    Args:
        ax: Matplotlib polar axes
        antenna: Antenna object (5G sector antenna)
        title: Plot title
        max_gain_db: Maximum gain in dB
    """
    # Get horizontal (azimuth) slice at alpha=90° (horizon, where peak is for sector antenna)
    gain_pat = antenna.get_gain_pattern()
    # Find alpha closest to 90° (horizon)
    alphas_unique = np.unique(gain_pat['alphas'].values)
    alpha_horizon_idx = np.argmin(np.abs(alphas_unique - 90.0))
    alpha_horizon = alphas_unique[alpha_horizon_idx]

    # Get horizontal (azimuth) slice at horizon
    horiz_slice = gain_pat[gain_pat['alphas'] == alpha_horizon]
    betas = horiz_slice['betas'].values
    gains = horiz_slice['gains'].values

    # Sort by beta (azimuth)
    sort_idx = np.argsort(betas)
    betas_sorted = betas[sort_idx]
    gains_sorted = gains[sort_idx]

    # Convert gains to dB
    gains_db = 10 * np.log10(gains_sorted)

    # Ensure pattern is closed
    if len(betas_sorted) > 0:
        if abs(betas_sorted[0] - betas_sorted[-1]) > 1e-6:
            betas_sorted = np.concatenate([betas_sorted, [betas_sorted[0]]])
            gains_db = np.concatenate([gains_db, [gains_db[0]]])

    # Map beta (azimuth) to polar angle theta
    # Beta = 0° is main direction, map to theta = 0° (East/right) like Starlink pattern
    # Use beta directly (no shift) so 0° azimuth is at 0° (East)
    theta_rad = np.deg2rad(betas_sorted)

    # For polar plot radius, we need positive values
    power_min = gains_db.min()
    power_offset = gains_db - power_min  # Shift so minimum is at 0
    power_offset = power_offset + 1  # Add small offset to avoid zero radius

    # Plot the pattern
    ax.plot(theta_rad, power_offset, 'b-', linewidth=2, label='Gain Pattern')

    # Set theta zero location to East (right side) - like Starlink pattern
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)  # Counter-clockwise (standard)

    # Set radial axis labels (convert back to dB)
    r_max = power_offset.max()
    r_min = power_offset.min()
    n_ticks = 6
    r_ticks = np.linspace(r_min, r_max, n_ticks)
    r_labels = [f'{p + power_min - 1:.0f}' for p in r_ticks]
    ax.set_rticks(r_ticks)
    ax.set_yticklabels(r_labels)

    # Set title and formatting
    ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Max Gain: {max_gain_db:.1f} dB',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# K-Band antenna pattern
ax1 = axes[0]
max_gain_k_db = 10 * np.log10(k_band_ant.get_boresight_gain())
plot_antenna_pattern_polar(ax1, k_band_ant, 'Suomi-NPP K-Band (23.8 GHz)\nAntenna Gain Pattern', max_gain_k_db)

# V-Band antenna pattern
ax2 = axes[1]
max_gain_v_db = 10 * np.log10(v_band_ant.get_boresight_gain())
plot_antenna_pattern_polar(ax2, v_band_ant, 'Suomi-NPP V-Band (50.3 GHz)\nAntenna Gain Pattern', max_gain_v_db)

plt.tight_layout()
antenna_pattern_file = os.path.join(script_dir, 'weather_sat_antenna_patterns.png')
plt.savefig(antenna_pattern_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved antenna patterns: {os.path.basename(antenna_pattern_file)}")

# Plot Starlink antenna pattern
fig_starlink, ax_starlink = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
max_gain_s_db = 10 * np.log10(starlink_ant.get_boresight_gain())
plot_antenna_pattern_polar(ax_starlink, starlink_ant,
                           'Starlink Antenna Gain Pattern\n(ITU Model, Backlobe)',
                           max_gain_s_db)
plt.tight_layout()
starlink_antenna_file = os.path.join(script_dir, 'starlink_antenna_pattern.png')
plt.savefig(starlink_antenna_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved Starlink antenna pattern: {os.path.basename(starlink_antenna_file)}")

# =============================================================================
# Plot 5G Ground Emitter Antenna Pattern
# =============================================================================

print("  Plotting 5G ground emitter antenna pattern...")

fig_5g, ax_5g = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
max_gain_5g_db = 10 * np.log10(ground_emitter_ant.get_boresight_gain())
plot_5g_antenna_pattern_polar(ax_5g, ground_emitter_ant,
                              '5G Ground Emitter Sector Antenna Pattern',
                              max_gain_5g_db)
plt.tight_layout()
ground_emitter_antenna_file = os.path.join(script_dir, 'ground_emitter_5g_antenna_pattern.png')
plt.savefig(ground_emitter_antenna_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved 5G ground emitter antenna pattern: {os.path.basename(ground_emitter_antenna_file)}")

# =============================================================================
# Plot Ground Emitter Distribution
# =============================================================================

print("  Plotting ground emitter distribution...")

# Create a map plot showing emitter distribution
fig_emitters, ax_emitters = plt.subplots(figsize=(10, 10))

# Calculate resolution element radius in degrees
# Account for longitude compression at this latitude (same as emitter generation)
center_lat_rad = np.deg2rad(target_lat)
r_max_meters = (resolution_km / 2) * 1000  # radius in meters
# Convert radius to degrees (matching emitter generation code)
lat_radius_deg = r_max_meters / 111320.0  # latitude: constant ~111.32 km/degree
lon_radius_deg = r_max_meters / (111320.0 * np.cos(center_lat_rad))  # longitude: varies with latitude

# Plot resolution element as an ellipse to match the actual geographical shape
# Since we use equal aspect, we need to plot a circle with radius matching the smaller dimension
# But actually, for visualization, we'll create an ellipse that represents the true shape
ellipse = Ellipse((target_lon, target_lat),
                  2 * lon_radius_deg, 2 * lat_radius_deg,
                  fill=False, edgecolor='black', linewidth=2, linestyle='--',
                  label=f'Resolution Element ({resolution_km} km diameter)')
ax_emitters.add_patch(ellipse)

# Plot emitters
ax_emitters.scatter(ground_emitters['lon'], ground_emitters['lat'],
                    c='red', s=50, alpha=0.7, edgecolors='darkred', linewidths=0.5,
                    label=f'Ground Emitters (n={len(ground_emitters)})', zorder=5)

# Plot center point
ax_emitters.scatter(target_lon, target_lat, c='blue', s=200, marker='*',
                    edgecolors='darkblue', linewidths=1, zorder=6,
                    label='Center (Target Location)')

# Set equal aspect ratio and labels
ax_emitters.set_aspect('equal', adjustable='box')
ax_emitters.set_xlabel('Longitude (degrees)', fontsize=12)
ax_emitters.set_ylabel('Latitude (degrees)', fontsize=12)
ax_emitters.set_title(f'Ground Emitter Distribution in Resolution Element\n'
                      f'Density: {emitter_density_per_km2:.2f} emitters/km², '
                      f'Total: {len(ground_emitters)} emitters',
                      fontsize=13, fontweight='bold')
ax_emitters.grid(True, alpha=0.3)
ax_emitters.legend(loc='best', fontsize=10)

# Add text box with statistics
stats_text = (f'Resolution: {resolution_km} km diameter\n'
              f'Area: {np.pi * (resolution_km/2)**2:.1f} km²\n'
              f'Density: {emitter_density_per_km2:.2f} emitters/km²\n'
              f'Total Emitters: {len(ground_emitters)}\n'
              f'Height Range: 20-50 m')
ax_emitters.text(0.02, 0.98, stats_text,
                 transform=ax_emitters.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=9)

plt.tight_layout()
emitter_distribution_file = os.path.join(script_dir, 'ground_emitter_distribution.png')
plt.savefig(emitter_distribution_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved ground emitter distribution: {os.path.basename(emitter_distribution_file)}")

# =============================================================================
# Plot Satellite Positions
# =============================================================================

print("  Plotting satellite positions...")

# Get observation window trajectory data (if not already filtered)
obs_traj_df = weather_sat_traj.get_traj_between(start_obs, stop_obs)
starlink_obs_data = starlink_data[
    (starlink_data['times'] >= start_obs) & (starlink_data['times'] <= stop_obs)
]

# Create polar plot for satellite positions
fig_pos, ax_pos = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})

# Plot Suomi-NPP positions (from observer's perspective)
ws_azimuths = obs_traj_df['azimuths'].values
ws_elevations = obs_traj_df['elevations'].values
ax_pos.scatter(np.radians(ws_azimuths), 90 - ws_elevations,
               c='red', s=50, alpha=0.6, label='Suomi-NPP', marker='s', edgecolors='darkred', linewidths=1)

# Plot Starlink satellite positions
if len(starlink_obs_data) > 0:
    # Sample Starlink positions (plot every Nth point to avoid overcrowding)
    sample_rate = max(1, len(starlink_obs_data) // 1000)  # Sample up to 1000 points
    starlink_sample = starlink_obs_data.iloc[::sample_rate]
    starlink_azimuths = starlink_sample['azimuths'].values
    starlink_elevations = starlink_sample['elevations'].values
    ax_pos.scatter(np.radians(starlink_azimuths), 90 - starlink_elevations,
                   c='blue', s=20, alpha=0.4, label='Starlink Satellites',
                   marker='o', edgecolors='darkblue', linewidths=0.5)

ax_pos.set_theta_zero_location("N")
ax_pos.set_theta_direction(-1)  # Clockwise
ax_pos.set_ylim(0, 90)
ax_pos.set_yticks(range(0, 91, 10))
ax_pos.set_yticklabels([str(x) for x in range(90, -1, -10)])
ax_pos.set_xlabel('Azimuth (degrees)', labelpad=20)
ax_pos.set_ylabel('Elevation (degrees)', labelpad=30)
ax_pos.set_title('Satellite Positions (Observer Frame)\nObservation Window', pad=20, fontsize=14, fontweight='bold')
ax_pos.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax_pos.grid(True, alpha=0.3)

plt.tight_layout()
satellite_positions_file = os.path.join(script_dir, 'satellite_positions.png')
plt.savefig(satellite_positions_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved satellite positions: {os.path.basename(satellite_positions_file)}")

print()

# =============================================================================
# Plot RFI Power vs Time
# =============================================================================

# Create time axis for plotting (relative to start of observation)
time_axis = [(t - start_obs).total_seconds() / 60.0 for t in obs_times_array]  # minutes

# Plot 2: Breakdown of RFI components by frequency (Phase 3: includes ground emitters)
fig, axes = plt.subplots(len(freq_channels), 1, figsize=(12, 6 * len(freq_channels)))
if len(freq_channels) == 1:
    axes = [axes]

for f_idx, freq in enumerate(freq_channels):
    ax = axes[f_idx]
    result_dict = results[freq]

    # Plot individual components
    ax.plot(time_axis, result_dict['starlink'][:, 0],
            label='Starlink Backlobe', linewidth=2, color='blue', linestyle='--')
    # Check if ground reflection is available
    if 'starlink_reflection' in result_dict:
        ax.plot(time_axis, result_dict['starlink_reflection'][:, 0],
                label='Starlink Ground Reflection', linewidth=2, color='cyan', linestyle=':')
    ax.plot(time_axis, result_dict['ground_emitter'][:, 0],
            label='Ground Emitter (5G)', linewidth=2, color='purple', linestyle='-.')
    ax.plot(time_axis, result_dict['earth'][:, 0],
            label='Earth Brightness', linewidth=2, color='green', linestyle='-.')
    ax.plot(time_axis, result_dict['sky'][:, 0],
            label='Sky Background', linewidth=2, color='orange', linestyle=':')
    ax.plot(time_axis, result_dict['system'][:, 0],
            label='System Noise', linewidth=2, color='red', linestyle='--')
    # Plot total
    ax.plot(time_axis, result_dict['total'][:, 0],
            label='Total Received Power', linewidth=2.5, color='black')

    ax.set_xlabel('Time from observation start (minutes)', fontsize=12)
    ax.set_ylabel('Received Power (dBW)', fontsize=12)
    ax.set_title(f'Weather Satellite: Received Power Components at {freq/1e9:.1f} GHz',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

plt.tight_layout()
output_file2 = os.path.join(script_dir, 'weather_sat_rfi_by_frequency.png')
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved plot: {os.path.basename(output_file2)}")

# =============================================================================
# Plot Comparison: RFI Sources (Starlink Backlobe, Ground Reflection, 5G)
# =============================================================================

print("  Plotting RFI sources comparison...")

# Plot 4: Comparison of all RFI sources
fig, axes = plt.subplots(len(freq_channels), 1, figsize=(14, 6 * len(freq_channels)))
if len(freq_channels) == 1:
    axes = [axes]

for f_idx, freq in enumerate(freq_channels):
    ax = axes[f_idx]
    result_dict = results_enhanced[freq]

    # Plot RFI sources only (excluding Earth brightness, sky, system noise)
    ax.plot(time_axis, result_dict['starlink'][:, 0],
            label='Starlink Backlobe', linewidth=2.5, color='blue', linestyle='--')

    # Check if ground reflection is available
    if 'starlink_reflection' in result_dict:
        ax.plot(time_axis, result_dict['starlink_reflection'][:, 0],
                label='Starlink Ground Reflection', linewidth=2.5, color='cyan', linestyle='-')

    ax.plot(time_axis, result_dict['ground_emitter'][:, 0],
            label='Ground Emitter (5G)', linewidth=2.5, color='purple', linestyle='-.')

    ax.set_xlabel('Time from observation start (minutes)', fontsize=12)
    ax.set_ylabel('RFI Power (dBW)', fontsize=12)
    ax.set_title(f'Phase 3: RFI Sources Comparison at {freq/1e9:.1f} GHz',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    # Format y-axis
    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style='plain', axis='y', useOffset=False, useMathText=False)

    # Add text box with statistics for each RFI source
    starlink_mean = np.mean(result_dict['starlink'][:, 0])
    starlink_max = np.max(result_dict['starlink'][:, 0])

    stats_text = (f'RFI Sources Statistics:\n'
                  f'Starlink Backlobe:\n'
                  f'  Mean: {starlink_mean:.2f} dBW\n'
                  f'  Max: {starlink_max:.2f} dBW\n')

    if 'starlink_reflection' in result_dict:
        reflection_mean = np.mean(result_dict['starlink_reflection'][:, 0])
        reflection_max = np.max(result_dict['starlink_reflection'][:, 0])
        stats_text += (f'Starlink Ground Reflection:\n'
                       f'  Mean: {reflection_mean:.2f} dBW\n'
                       f'  Max: {reflection_max:.2f} dBW\n')

    ground_emitter_mean = np.mean(result_dict['ground_emitter'][:, 0])
    ground_emitter_max = np.max(result_dict['ground_emitter'][:, 0])
    stats_text += (f'Ground Emitter (5G):\n'
                   f'  Mean: {ground_emitter_mean:.2f} dBW\n'
                   f'  Max: {ground_emitter_max:.2f} dBW')

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)

plt.tight_layout()
output_file4 = os.path.join(script_dir, 'weather_sat_rfi_sources_comparison.png')
plt.savefig(output_file4, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved plot: {os.path.basename(output_file4)}")

# DKDK
plt.show()

# =============================================================================
# Summary Statistics
# =============================================================================

print()
print("="*70)
print("SUMMARY STATISTICS")
print("="*70)

for freq, result_dict in results.items():
    total_power = result_dict['total'][:, 0]
    starlink_power = result_dict['starlink'][:, 0]
    earth_power = result_dict['earth'][:, 0]
    sky_power = result_dict['sky'][:, 0]
    system_power = result_dict['system'][:, 0]
    ground_emitter_power = result_dict['ground_emitter'][:, 0]

    # Check if ground reflection is available
    if 'starlink_reflection' in result_dict:
        starlink_reflection_power = result_dict['starlink_reflection'][:, 0]
    else:
        starlink_reflection_power = None

    print(f"\nFrequency: {freq/1e9:.1f} GHz")
    print("  Total Received Power:")
    print(f"    Min: {np.min(total_power):.2f} dBW")
    print(f"    Max: {np.max(total_power):.2f} dBW")
    print(f"    Mean: {np.mean(total_power):.2f} dBW")
    print(f"    Std: {np.std(total_power):.2f} dBW")
    print("  Components (mean power):")
    print(f"    Starlink Backlobe: {np.mean(starlink_power):.2f} dBW")
    if starlink_reflection_power is not None:
        print(f"    Starlink Ground Reflection: {np.mean(starlink_reflection_power):.2f} dBW")
    print(f"    Ground Emitter (5G): {np.mean(ground_emitter_power):.2f} dBW")
    print(f"    Earth: {np.mean(earth_power):.2f} dBW")
    print(f"    Sky: {np.mean(sky_power):.2f} dBW")
    print(f"    System: {np.mean(system_power):.2f} dBW")
    print("  Maximum RFI from Starlink Backlobe:")
    print(f"    Max: {np.max(starlink_power):.2f} dBW")
    if starlink_reflection_power is not None:
        print("  Maximum RFI from Starlink Ground Reflection:")
        print(f"    Max: {np.max(starlink_reflection_power):.2f} dBW")
    print("  Maximum RFI from Ground Emitters (5G):")
    print(f"    Max: {np.max(ground_emitter_power):.2f} dBW")

print()
print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print()
print(f"Plots saved to: {script_dir}")
print()

# Show plots (optional - comment out if running in headless mode)
# plt.show()
