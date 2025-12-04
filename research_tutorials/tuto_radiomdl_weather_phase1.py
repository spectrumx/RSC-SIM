"""
Weather Satellite RFI Modeling Tutorial - "Looking Down" Case

This script models RFI from the perspective of a weather satellite (Suomi-NPP)
looking down at Earth, including:
- Starlink backlobe interference
- Earth brightness temperature
- Sky background
- System noise

Author: Weather Satellite RFI Modeling Team
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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
    model_weather_sat_observed_power
)

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

# Create Starlink transmitter instrument
starlink_T_phy = 0.0  # Physical temperature (same as tuto_radiomdl_environment.py)
# TODO Please change starlink_freq value for testing harmonic effects
# starlink_freq = 11.325e9  # Hz (same as tuto_radiomdl_environment.py)
starlink_freq = 11.9e9  # Hz (for testing 2nd harmonic to K-band)
# starlink_freq = 12.575e9  # Hz (for testing 4th harmonic to V-band)
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
# Generate Observation Times
# =============================================================================

print("Step 5: Generating observation time grid...")

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

print("Step 6: Modeling weather satellite observations...")
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

# Start timing for Step 6
step6_start_time = time.time()

# Model for each frequency channel
results = {}
for f_idx, freq in enumerate(freq_channels):
    print(f"  Processing frequency channel {f_idx + 1}/{len(freq_channels)}: {freq/1e9:.1f} GHz")

    # Time each frequency channel
    freq_start_time = time.time()

    # Use appropriate instrument for this frequency
    if freq < 30e9:
        instrument = weather_sat_instruments[0]
    else:
        instrument = weather_sat_instruments[1]

    # Model observed power (now returns dictionary with components)
    result_dict = model_weather_sat_observed_power(
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
        earth_brightness_temp=280.0,
        sky_brightness_temp=2.73,
        system_temp=300.0,
        starlink_eirp_dbw=starlink_transmit_pow,
        starlink_fundamental_freq=starlink_freq,
        harmonics=starlink_harmonics
    )

    results[freq] = result_dict

    # Report timing for this frequency channel
    freq_elapsed = time.time() - freq_start_time
    print(f"    ✓ Completed in {freq_elapsed:.1f} seconds")
    print()

# Report total timing for Step 6
step6_elapsed = time.time() - step6_start_time
print(f"  Total computation time for Step 6: {step6_elapsed:.1f} seconds ({step6_elapsed/60:.2f} minutes)")
print()

# =============================================================================
# Visualization
# =============================================================================

print("Step 7: Creating visualizations...")

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

# Plot 2: Breakdown of RFI components by frequency
fig, axes = plt.subplots(len(freq_channels), 1, figsize=(12, 6 * len(freq_channels)))
if len(freq_channels) == 1:
    axes = [axes]

for f_idx, freq in enumerate(freq_channels):
    ax = axes[f_idx]
    result_dict = results[freq]

    # Plot individual components
    ax.plot(time_axis, result_dict['starlink'][:, 0],
            label='Starlink Backlobe', linewidth=2, color='blue', linestyle='--')
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

# show plots
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

    print(f"\nFrequency: {freq/1e9:.1f} GHz")
    print("  Total Received Power:")
    print(f"    Min: {np.min(total_power):.2f} dBW")
    print(f"    Max: {np.max(total_power):.2f} dBW")
    print(f"    Mean: {np.mean(total_power):.2f} dBW")
    print(f"    Std: {np.std(total_power):.2f} dBW")
    print("  Components (mean power):")
    print(f"    Starlink: {np.mean(starlink_power):.2f} dBW")
    print(f"    Earth: {np.mean(earth_power):.2f} dBW")
    print(f"    Sky: {np.mean(sky_power):.2f} dBW")
    print(f"    System: {np.mean(system_power):.2f} dBW")
    print("  Maximum RFI from Starlink Backlobe:")
    print(f"    Max: {np.max(starlink_power):.2f} dBW")

print()
print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print()
print(f"Plots saved to: {script_dir}")
print()

# Show plots (optional - comment out if running in headless mode)
# plt.show()
