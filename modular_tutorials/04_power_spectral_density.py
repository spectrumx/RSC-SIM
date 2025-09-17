"""
04_power_spectral_density.py - Frequency Analysis

This script demonstrates power spectral density (PSD) analysis for radio astronomy
observations, showing how satellite interference affects the frequency domain
characteristics of the received signal.

The script models a wider bandwidth observation with multiple frequency channels
to visualize the spectral characteristics of both astronomical sources and
satellite interference.
"""

import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import RSC-SIM modules
from radio_types import Observation, Constellation, Trajectory, Instrument  # noqa: E402
from astro_mdl import power_to_temperature, temperature_to_power  # noqa: E402
from sat_mdl import sat_link_budget_vectorized  # noqa: E402
from obs_mdl import model_observed_temp  # noqa: E402

# Import shared utilities
from shared.config import STANDARD_FIGURE_SIZE, CENTER_FREQUENCY  # noqa: E402
from shared.instrument_setup import (  # noqa: E402
    setup_psd_instrument,
    setup_frequency_dependent_satellite
)
from shared.sky_models import create_sky_model  # noqa: E402
from shared.plotting_utils import plot_psd_spectrogram, setup_plotting  # noqa: E402

# Set up plotting
setup_plotting()


def create_frequency_dependent_transmission_profile(num_channels):
    """
    Create a frequency-dependent transmission profile for satellites.

    This simulates realistic satellite transmission patterns with:
    - Reduced power at band edges
    - Notches for interference avoidance
    - Peak transmission at center frequency

    Args:
        num_channels: Number of frequency channels

    Returns:
        array: Transmission profile (0-1 scale)
    """
    tmt_profile = np.ones(num_channels)

    # Reduce power at band edges (first and last 10% of channels)
    edge_reduction = num_channels // 10
    tmt_profile[:edge_reduction] = 0.0
    tmt_profile[-edge_reduction:] = 0.0

    # Create interference avoidance notch in center
    notch_width = num_channels // 10
    center_start = num_channels // 2 - notch_width // 2
    center_end = num_channels // 2 + notch_width // 2
    tmt_profile[center_start:center_end] = 0.0

    # Ensure center frequency has full power
    tmt_profile[num_channels // 2] = 1.0

    return tmt_profile


def main():
    """Main function to run the PSD analysis."""
    print("=== Power Spectral Density Analysis ===")
    print("Modeling frequency-domain characteristics of radio observations")
    print()

    # Get path to data files (located in tutorial/data/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "tutorial", "data")

    # Time window for analysis
    start_window = "2025-02-18T15:00:00.000"
    stop_window = "2025-02-18T15:45:00.000"

    # Replace colons with underscores for filename
    start_window_str = start_window.replace(":", "_")
    stop_window_str = stop_window.replace(":", "_")

    # Load source trajectory
    file_traj_obj_path = os.path.join(
        data_dir,
        f"casA_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
    )

    print(f"Loading source trajectory from: {file_traj_obj_path}")
    traj_src = Trajectory.from_file(
        file_traj_obj_path,
        time_tag='time_stamps',
        elevation_tag='altitudes',
        azimuth_tag='azimuths',
        distance_tag='distances'
    )

    # Define observation parameters
    dateformat = "%Y-%m-%dT%H:%M:%S.%f"
    start_obs = datetime.strptime("2025-02-18T15:30:00.000", dateformat)
    stop_obs = datetime.strptime("2025-02-18T15:40:00.000", dateformat)

    # Create pointing trajectory with offset
    offset_angles = (-40, 0.)  # (az,el) in degrees
    time_off_src = start_obs
    time_on_src = time_off_src + timedelta(minutes=5)

    # Copy and modify trajectory
    traj_obj = Trajectory(traj_src.traj.copy())
    mask = (traj_obj.traj['times'] >= time_off_src) & (traj_obj.traj['times'] <= time_on_src)
    traj_obj.traj.loc[mask, 'azimuths'] += offset_angles[0]
    traj_obj.traj.loc[mask, 'elevations'] += offset_angles[1]

    # Filter points below 5deg elevation
    filt_el = ('elevations', lambda e: e > 5.)

    # Create high-resolution instrument for PSD analysis
    print("Creating high-resolution instrument for PSD analysis...")

    # Instrument parameters for PSD analysis
    new_bw = 30e6  # 30 MHz bandwidth
    new_freq_chan = 164  # 164 frequency channels

    # Create telescope instrument for PSD analysis
    westford_freqs = setup_psd_instrument()

    # Create observation
    observ_freqs = Observation.from_dates(
        start_obs, stop_obs, traj_obj, westford_freqs,
        filt_funcs=(filt_el,)
    )

    # Create sky model
    sky_mdl = create_sky_model(observ_freqs)

    # Create satellite constellation with frequency-dependent transmission
    print("Setting up satellite constellation with frequency-dependent transmission...")

    # Create frequency-dependent transmission profile
    tmt_profile = create_frequency_dependent_transmission_profile(new_freq_chan)
    freq_bins = westford_freqs.get_center_freq_chans()

    # Plot transmission profile
    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)
    ax.plot(freq_bins / 1e9, tmt_profile)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Normalized Transmission Power")
    ax.set_title("Satellite Transmission Profile")
    ax.grid(True)
    plt.savefig("04_satellite_transmission_profile.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Create satellite instrument with frequency-dependent transmission
    def transmit_temp_freqs(tim, freq):
        ind_freq = np.argmin(np.abs(freq_bins - freq))
        transmit_pow = -15 + 10 * np.log10(300)  # dBW
        return tmt_profile[ind_freq] * power_to_temperature(10**(transmit_pow/10), 1.0)

    # Create custom satellite instrument with our frequency-dependent transmission
    sat_ant = setup_frequency_dependent_satellite().get_antenna()
    sat_transmit_freqs = Instrument(
        sat_ant,
        0.0,  # physical temperature
        CENTER_FREQUENCY,
        new_bw,
        transmit_temp_freqs,
        new_freq_chan,
        []
    )

    # Load satellite trajectories
    file_traj_sats_path = os.path.join(
        data_dir,
        f"Starlink_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
    )

    # Filter satellites
    filt_name = ('sat', lambda s: ~s.str.contains('DTC'))
    filt_el = ('elevations', lambda e: e > 20)

    # Create constellation
    starlink_constellation_freqs = Constellation.from_file(
        file_traj_sats_path, observ_freqs, sat_transmit_freqs,
        sat_link_budget_vectorized,
        name_tag='sat',
        time_tag='timestamp',
        elevation_tag='elevations',
        azimuth_tag='azimuths',
        distance_tag='ranges_westford',
        filt_funcs=(filt_name, filt_el)
    )

    # Compute PSD results
    print("Computing power spectral density...")
    print("Note: This may take several minutes due to high frequency resolution...")

    start_time = time.time()
    result_freqs = model_observed_temp(
        observ_freqs, sky_mdl, starlink_constellation_freqs
    )
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"PSD computation completed in {execution_time:.2f} seconds")

    # Get time and frequency data
    time_samples = observ_freqs.get_time_stamps()
    freq_bins = westford_freqs.get_center_freq_chans()

    # Convert temperature to power (use same bandwidth as original: 1 kHz)
    # The original code uses bw/freq_chan = 1e3/1 = 1000 Hz for PSD calculation
    plot_psd = temperature_to_power(result_freqs, 1e3)

    # Note: Single-channel computation removed since it's no longer needed

    # Create PSD spectrogram plot
    print("Creating PSD spectrogram...")
    fig, axes, psd_imshow = plot_psd_spectrogram(
        time_samples, freq_bins, plot_psd,
        title="Power Spectral Density with Satellite Interference",
        colormap="plasma"  # Use plasma colormap to match original plot
    )

    # Note: No explicit color limits set to match original auto-scaling behavior

    plt.savefig("04_power_spectral_density.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Note: Four-scenario comparison plot moved to 02_satellite_interference.py
    # for better organization and faster computation (single-channel vs multi-channel)

    # Frequency slice analysis
    print("Creating frequency slice analysis...")

    # Select a time slice for frequency analysis
    time_idx = len(time_samples) // 2  # Middle of observation
    selected_time = time_samples.iloc[time_idx]

    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)

    # Plot power vs frequency at selected time
    freq_power = plot_psd[time_idx, 0, :]
    ax.plot(freq_bins / 1e9, 10 * np.log10(freq_power), linewidth=2)

    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Power [dBW]")
    ax.set_title(f"Frequency Slice at {selected_time.strftime('%H:%M:%S')} UTC")
    ax.grid(True)

    # Add transmission profile for reference
    ax_twin = ax.twinx()
    ax_twin.plot(freq_bins / 1e9, tmt_profile, 'r--', alpha=0.5, label="Satellite TX Profile")
    ax_twin.set_ylabel("Normalized Transmission Power", color='r')
    ax_twin.tick_params(axis='y', labelcolor='r')

    plt.savefig("04_frequency_slice.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Summary statistics
    print("\n=== PSD Analysis Summary ===")
    print(f"Observation duration: {stop_obs - start_obs}")
    print(f"Frequency range: {freq_bins[0]/1e9:.3f} - {freq_bins[-1]/1e9:.3f} GHz")
    print(f"Frequency resolution: {(freq_bins[1] - freq_bins[0])/1e6:.1f} MHz")
    print(f"Time resolution: {(time_samples.iloc[1] - time_samples.iloc[0]).total_seconds():.1f} seconds")
    print(f"Total frequency channels: {new_freq_chan}")
    print(f"Total time samples: {len(time_samples)}")

    # Power statistics
    max_power = np.max(10 * np.log10(plot_psd))
    min_power = np.min(10 * np.log10(plot_psd))
    mean_power = np.mean(10 * np.log10(plot_psd))

    print(f"Power range: {min_power:.1f} to {max_power:.1f} dBW")
    print(f"Mean power: {mean_power:.1f} dBW")

    print("\nPSD analysis completed successfully!")
    print("Generated files:")
    print("- 04_satellite_transmission_profile.png")
    print("- 04_power_spectral_density.png")
    print("- 04_frequency_slice.png")
    print("\nNote: Four-scenario comparison plot is now available in 02_satellite_interference.py")


if __name__ == "__main__":
    main()
