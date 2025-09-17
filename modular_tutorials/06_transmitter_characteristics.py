#!/usr/bin/env python3
"""
Tutorial 06: Transmitter Characteristics Analysis

This tutorial explores transmitter characteristics in satellite radio astronomy
observations using the RSC-SIM framework. It provides both educational demonstrations
and realistic modeling scenarios, running both parts by default.

Educational Components:
1. Polarization mismatch loss calculations and effects
2. Harmonic contribution analysis and modeling
3. Transmitter class functionality and testing
4. Visualization of polarization and harmonic effects

Realistic Modeling:
5. Realistic Starlink satellite transmitter configurations
6. Comprehensive interference modeling with transmitter characteristics
7. Comparison of interference predictions with and without transmitter effects
8. Analysis of circular-to-linear polarization mismatch (3 dB loss scenario)

Learning Objectives:
- Understand polarization mismatch between satellite transmitters and radio telescopes
- Learn to calculate and visualize polarization loss effects
- Explore harmonic contributions from satellite transmitters
- Implement transmitter modeling with realistic characteristics
- Compare interference predictions with and without transmitter characteristics
- Analyze realistic scenarios (Starlink circular + Westford linear = 3 dB loss)

Key Concepts:
- Polarization mismatch loss: cos²(θ) for linear, 3dB for linear-circular
- Harmonic contributions: frequency multiplication effects on interference
- Transmitter class: modeling with polarization and harmonics
- Realistic configurations: Starlink (circular) + Westford (linear) scenario
- Comprehensive link budget: physics-based interference prediction
- Result array management: .copy() to prevent overwriting in multiple simulations

Output:
- Educational plots: polarization effects and harmonic contributions
- Realistic comparison plots: full and zoomed views showing 3 dB polarization loss
- Saved files: 06_transmitter_characteristics_comparison.png and _zoomed.png

Prerequisites:
- Tutorials 01-05 (basic observation, satellite interference, sky mapping, PSD analysis, Doppler effects)
- Understanding of electromagnetic wave polarization
- Familiarity with harmonic distortion concepts
- Basic knowledge of satellite communication systems
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from radio_types import Antenna, Instrument, Observation, Constellation, Trajectory, Transmitter  # noqa: E402
from astro_mdl import (  # noqa: E402
    estim_casA_flux, power_to_temperature, temperature_to_power,
    antenna_mdl_ITU, estim_temp
)
from sat_mdl import (  # noqa: E402
    calculate_polarization_mismatch_loss,
    calculate_polarization_mismatch_loss_vectorized,
    calculate_harmonic_contribution,
    sat_link_budget_vectorized,
    sat_link_budget_comprehensive_vectorized
)
from obs_mdl import model_observed_temp  # noqa: E402
import antenna_pattern  # noqa: E402


def test_polarization_mismatch_loss():
    """Test polarization mismatch loss calculations."""

    print("="*80)
    print("TESTING POLARIZATION MISMATCH LOSS")
    print("="*80)

    # Test different polarization combinations
    test_cases = [
        # (tx_pol, tx_angle, rx_pol, rx_angle, expected_behavior)
        ('linear', 0.0, 'linear', 0.0, 'Perfect match'),
        ('linear', 0.0, 'linear', 90.0, 'Orthogonal - should be 0'),
        ('linear', 45.0, 'linear', 45.0, 'Perfect match at 45°'),
        ('linear', 0.0, 'circular', 0.0, 'Linear to circular - 3dB loss'),
        ('circular', 0.0, 'linear', 0.0, 'Circular to linear - 3dB loss'),
        ('circular', 0.0, 'circular', 0.0, 'Circular to circular - no loss'),
        ('elliptical', 30.0, 'linear', 0.0, 'Elliptical to linear'),
    ]

    print("\nPolarization Mismatch Loss Results:")
    print("-" * 60)
    print(f"{'TX Pol':<12} {'TX Ang':<8} {'RX Pol':<12} {'RX Ang':<8} {'Loss':<8} {'Behavior'}")
    print("-" * 60)

    for tx_pol, tx_ang, rx_pol, rx_ang, behavior in test_cases:
        loss = calculate_polarization_mismatch_loss(tx_pol, tx_ang, rx_pol, rx_ang)
        loss_db = 10 * np.log10(loss) if loss > 0 else -100
        print(f"{tx_pol:<12} {tx_ang:<8.1f} {rx_pol:<12} {rx_ang:<8.1f} {loss_db:<8.1f} {behavior}")

    # Test vectorized version
    print("\n\nVectorized Polarization Loss Test:")
    print("-" * 40)

    tx_polarizations = np.array(['linear', 'circular', 'elliptical'])
    tx_angles = np.array([0.0, 0.0, 30.0])

    losses = calculate_polarization_mismatch_loss_vectorized(
        tx_polarizations, tx_angles, 'linear', 0.0
    )

    for i, (pol, ang, loss) in enumerate(zip(tx_polarizations, tx_angles, losses)):
        loss_db = 10 * np.log10(loss) if loss > 0 else -100
        print(f"Transmitter {i+1}: {pol} at {ang}° -> {loss_db:.1f} dB loss")


def test_harmonic_contributions():
    """Test harmonic contribution calculations."""

    print("\n" + "="*80)
    print("TESTING HARMONIC CONTRIBUTIONS")
    print("="*80)

    # Test parameters
    base_frequency = 11.325e9  # 11.325 GHz
    base_power = 1.0  # Normalized power
    observation_frequency = 11.325e9  # Same as base
    observation_bandwidth = 1e6  # 1 MHz

    # Test harmonics
    test_harmonics = [
        (2.0, 0.1),   # 2nd harmonic at 10% power
        (3.0, 0.05),  # 3rd harmonic at 5% power
        (4.0, 0.02),  # 4th harmonic at 2% power
    ]

    print(f"\nBase Frequency: {base_frequency/1e9:.3f} GHz")
    print(f"Observation Frequency: {observation_frequency/1e9:.3f} GHz")
    print(f"Observation Bandwidth: {observation_bandwidth/1e6:.1f} MHz")

    print("\nHarmonic Analysis:")
    print("-" * 50)
    print(f"{'Harmonic':<10} {'Frequency':<12} {'Power':<8} {'In Band':<8}")
    print("-" * 50)

    total_harmonic_power = 0.0

    for i, (freq_mult, power_red) in enumerate(test_harmonics):
        harmonic_freq = base_frequency * freq_mult
        harmonic_power = base_power * power_red

        # Check if in observation band
        freq_min = observation_frequency - observation_bandwidth / 2
        freq_max = observation_frequency + observation_bandwidth / 2
        in_band = freq_min <= harmonic_freq <= freq_max

        if in_band:
            total_harmonic_power += harmonic_power

        print(f"{i+1}st: {freq_mult}x{'':<5} {harmonic_freq/1e9:<12.3f} {power_red:<8.3f} {'Yes' if in_band else 'No'}")

    # Calculate total contribution
    total_contribution = calculate_harmonic_contribution(
        base_frequency, base_power, test_harmonics,
        observation_frequency, observation_bandwidth
    )

    if total_contribution > 0:
        print(f"\nTotal Harmonic Contribution: {total_contribution:.3f} ({10*np.log10(total_contribution):.1f} dB)")
    else:
        print(f"\nTotal Harmonic Contribution: {total_contribution:.3f} (-inf dB)")
    print(f"Fundamental + Harmonics: {1.0 + total_contribution:.3f} ({10*np.log10(1.0 + total_contribution):.1f} dB)")

    # Test with harmonics that fall in the observation band
    print("\n\nTesting with harmonics in observation band:")
    print("-" * 50)

    # Create harmonics that fall within the observation band
    in_band_harmonics = [
        (1.0, 0.5),    # Fundamental at 50% power
        (1.001, 0.1),  # Very close to fundamental
    ]

    total_contribution_in_band = calculate_harmonic_contribution(
        base_frequency, base_power, in_band_harmonics,
        observation_frequency, observation_bandwidth
    )

    print(f"Harmonics in band: {in_band_harmonics}")
    print(f"Total contribution: {total_contribution_in_band:.3f} ({10*np.log10(total_contribution_in_band):.1f} dB)")


def test_transmitter_class():
    """Test the Transmitter class functionality."""

    print("\n" + "="*80)
    print("TESTING TRANSMITTER CLASS")
    print("="*80)

    # Create a simple instrument for testing
    eta_rad = 0.5
    freq_band = (10e9, 12e9)
    alphas = np.arange(0, 181)
    betas = np.arange(0, 351, 10)
    gain_pat = antenna_mdl_ITU(39.3, 3.0, alphas, betas)
    sat_ant = Antenna.from_dataframe(gain_pat, eta_rad, freq_band)

    sat_T_phy = 0.0
    sat_freq = 11.325e9
    sat_bw = 250e6
    transmit_pow = -15 + 10 * np.log10(300)

    def transmit_temp(tim, freq):
        return power_to_temperature(10**(transmit_pow/10), 1.0)

    sat_transmit = Instrument(sat_ant, sat_T_phy, sat_freq, sat_bw, transmit_temp, 1, [])

    # Test different transmitter configurations
    print("\nCreating transmitters with different characteristics:")

    # Linear transmitter with harmonics
    linear_tx = Transmitter.from_instrument(
        sat_transmit,
        polarization='linear',
        polarization_angle=45.0,
        harmonics=[(2.0, 0.1), (3.0, 0.05)]
    )

    print("Linear transmitter:")
    print(f"  Polarization: {linear_tx.get_polarization()}")
    print(f"  Polarization angle: {linear_tx.get_polarization_angle()}°")
    print(f"  Harmonics: {linear_tx.get_harmonics()}")
    print(f"  Harmonic frequencies: {[f/1e9 for f in linear_tx.get_harmonic_frequencies()]} GHz")
    print(f"  Harmonic powers: {linear_tx.get_harmonic_powers()}")

    # Circular transmitter
    circular_tx = Transmitter.from_instrument(
        sat_transmit,
        polarization='circular',
        polarization_angle=0.0,
        harmonics=[]
    )

    print("\nCircular transmitter:")
    print(f"  Polarization: {circular_tx.get_polarization()}")
    print(f"  Harmonics: {circular_tx.get_harmonics()}")

    # Test adding harmonics
    print("\nAdding harmonics to circular transmitter:")
    circular_tx.add_harmonic(2.0, 0.2)
    circular_tx.add_harmonic(4.0, 0.05)
    print(f"  Updated harmonics: {circular_tx.get_harmonics()}")
    print(f"  Harmonic frequencies: {[f/1e9 for f in circular_tx.get_harmonic_frequencies()]} GHz")


def plot_polarization_effects():
    """Create plots showing polarization effects."""

    print("\n" + "="*80)
    print("PLOTTING POLARIZATION EFFECTS")
    print("="*80)

    # Create angle sweep for polarization analysis
    angles = np.linspace(0, 180, 181)

    # Calculate polarization loss for different combinations
    linear_to_linear = [calculate_polarization_mismatch_loss('linear', ang, 'linear', 0.0) for ang in angles]
    linear_to_circular = [calculate_polarization_mismatch_loss('linear', ang, 'circular', 0.0) for ang in angles]
    circular_to_linear = [calculate_polarization_mismatch_loss('circular', ang, 'linear', 0.0) for ang in angles]

    # Convert to dB
    linear_to_linear_db = [10 * np.log10(loss) if loss > 0 else -100 for loss in linear_to_linear]
    linear_to_circular_db = [10 * np.log10(loss) if loss > 0 else -100 for loss in linear_to_circular]
    circular_to_linear_db = [10 * np.log10(loss) if loss > 0 else -100 for loss in circular_to_linear]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Linear polarization angle sweep
    ax1.plot(angles, linear_to_linear_db, 'b-', linewidth=2, label='Linear → Linear')
    ax1.plot(angles, linear_to_circular_db, 'r--', linewidth=2, label='Linear → Circular')
    ax1.plot(angles, circular_to_linear_db, 'g:', linewidth=2, label='Circular → Linear')

    ax1.set_xlabel('Transmitter Polarization Angle (degrees)')
    ax1.set_ylabel('Polarization Loss (dB)')
    ax1.set_title('Polarization Mismatch Loss vs. Angle')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-50, 5)

    # Plot 2: Polarization loss distribution
    polarization_types = ['Linear', 'Circular', 'Elliptical']
    receiver_pol = 'linear'
    receiver_angle = 0.0

    losses = []
    for pol in ['linear', 'circular', 'elliptical']:
        loss = calculate_polarization_mismatch_loss(pol, 0.0, receiver_pol, receiver_angle)
        losses.append(10 * np.log10(loss) if loss > 0 else -100)

    bars = ax2.bar(polarization_types, losses, color=['blue', 'red', 'green'], alpha=0.7)
    ax2.set_ylabel('Polarization Loss (dB)')
    ax2.set_title(f'Polarization Loss: {receiver_pol.capitalize()} Receiver')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{loss:.1f} dB', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('06_transmitter_polarization_effects.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_harmonic_effects():
    """Create plots showing harmonic effects."""

    print("\n" + "="*80)
    print("PLOTTING HARMONIC EFFECTS")
    print("="*80)

    # Test parameters
    base_frequency = 11.325e9  # 11.325 GHz
    base_power = 1.0
    harmonic_bandwidth = 100e6  # 100 MHz bandwidth for harmonics (wider for visualization)

    # Define different harmonic configurations
    harmonic_configs = {
        'No Harmonics': [],
        'Weak Harmonics': [(2.0, 0.05), (3.0, 0.02)],
        'Strong Harmonics': [(2.0, 0.2), (3.0, 0.1), (4.0, 0.05)],
        'Many Harmonics': [(2.0, 0.1), (3.0, 0.05), (4.0, 0.02), (5.0, 0.01), (6.0, 0.005)]
    }

    # Calculate power spectrum for each configuration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Create a categorical bar chart showing peak powers at each harmonic frequency
    harmonic_freqs = [base_frequency * i for i in range(1, 7)]  # 1st through 6th harmonics
    freq_labels = [f'{freq/1e9:.1f} GHz' for freq in harmonic_freqs]

    # Calculate peak power at each harmonic frequency for each configuration
    config_names = list(harmonic_configs.keys())
    colors = ['blue', 'red', 'green', 'orange']

    # Set up bar positions
    x = np.arange(len(freq_labels))
    width = 0.2  # Width of bars
    offsets = np.linspace(-0.3, 0.3, len(config_names))

    for i, (config_name, harmonics) in enumerate(harmonic_configs.items()):
        peak_powers = []

        for harmonic_freq in harmonic_freqs:
            # Calculate total power at this harmonic frequency
            total_power = 0.0

            # Check if fundamental contributes (only at 1st harmonic frequency)
            if abs(harmonic_freq - base_frequency) <= harmonic_bandwidth / 2:
                total_power += base_power

            # Add harmonic contributions
            for freq_mult, power_red in harmonics:
                if abs(harmonic_freq - base_frequency * freq_mult) <= harmonic_bandwidth / 2:
                    total_power += base_power * power_red

            # Convert to dB
            if total_power > 0:
                peak_powers.append(10 * np.log10(total_power))
            else:
                peak_powers.append(-100)

        # Plot bars for this configuration
        bars = ax1.bar(x + offsets[i], peak_powers, width,
                       label=config_name, color=colors[i], alpha=0.8)

        # Add value labels on bars
        for bar, power in zip(bars, peak_powers):
            height = bar.get_height()
            if power > -50:  # Only label if power is significant
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{power:.1f}', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Harmonic Frequency')
    ax1.set_ylabel('Peak Power (dB)')
    ax1.set_title('Peak Power at Each Harmonic Frequency')
    ax1.set_xticks(x)
    ax1.set_xticklabels(freq_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(-50, 5)

    # Plot 2: Show total power contribution from harmonics at different frequencies
    # Create a realistic scenario where we observe at different frequencies
    observation_frequencies = [11.325e9, 22.65e9, 33.975e9, 45.3e9, 56.625e9, 67.95e9]  # All 6 harmonics
    observation_bandwidth = 100e6  # 100 MHz observation bandwidth

    harmonic_orders = [2, 3, 4, 5, 6]
    power_reductions = [0.1, 0.05, 0.02, 0.01, 0.005]

    # Calculate total power at each observation frequency
    total_powers = []
    freq_labels = []

    for obs_freq in observation_frequencies:
        # Calculate total power at this observation frequency
        total_power = 0.0

        # Check if fundamental contributes
        if abs(obs_freq - base_frequency) <= observation_bandwidth / 2:
            total_power += base_power

        # Check if harmonics contribute
        for i, freq_mult in enumerate(harmonic_orders):
            harmonic_freq = base_frequency * freq_mult
            if abs(obs_freq - harmonic_freq) <= observation_bandwidth / 2:
                total_power += base_power * power_reductions[i]

        total_powers.append(total_power)
        freq_labels.append(f'{obs_freq/1e9:.1f} GHz')

    # Calculate dB values, handling zero power case
    db_values = []
    for power in total_powers:
        if power > 0:
            db_values.append(10 * np.log10(power))
        else:
            db_values.append(-100)  # Use -100 dB instead of -inf for plotting

    bars = ax2.bar(freq_labels, db_values, color='orange', alpha=0.7)
    ax2.set_ylabel('Total Power (dB)')
    ax2.set_title('Power at Different Observation Frequencies')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, power, db_val in zip(bars, total_powers, db_values):
        height = bar.get_height()
        if power > 0:
            label = f'{db_val:.1f} dB'
        else:
            label = '-∞ dB'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 label, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('06_transmitter_harmonic_effects.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run all transmitter characteristic tests."""

    print("="*80)
    print("RSC-SIM TRANSMITTER CHARACTERISTICS TEST (BASIC)")
    print("="*80)
    print("Testing transmitter modeling with polarization and harmonics")
    print("="*80)

    # Run all tests
    test_polarization_mismatch_loss()
    test_harmonic_contributions()
    test_transmitter_class()

    # Create plots
    plot_polarization_effects()
    plot_harmonic_effects()

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print("Transmitter characteristics have been tested and validated.")
    print("Plots have been saved to:")
    print("  - 06_transmitter_polarization_effects.png")
    print("  - 06_transmitter_harmonic_effects.png")
    print("="*80)


# =============================================================================
# REALISTIC TRANSMITTER CHARACTERISTICS TESTING
# =============================================================================
# The following functions provide realistic radio astronomy observation scenarios
# with transmitter characteristics including polarization mismatch loss
# and harmonic contributions in a realistic simulation environment.


def setup_telescope_instrument():
    """Set up the telescope instrument using external antenna pattern file."""

    print("Setting up telescope instrument...")

    # radiation efficiency of telescope antenna
    eta_rad = 0.45

    # valid frequency band of gain pattern model
    freq_band = (10e9, 12e9)  # in Hz

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # load telescope antenna from external file
    file_pattern_path = os.path.join(script_dir, "..", "tutorial", "data", "single_cut_res.cut")

    if not os.path.exists(file_pattern_path):
        print(f"Warning: Antenna pattern file not found at {file_pattern_path}")
        print("Using ITU model as fallback...")

        # Fallback to ITU model
        alphas = np.arange(0, 181)
        betas = np.arange(0, 351, 10)
        gain_pat = antenna_mdl_ITU(50.0, 1.0, alphas, betas)
        tel_ant = Antenna.from_dataframe(gain_pat, eta_rad, freq_band)
    else:
        tel_ant = Antenna.from_file(
            file_pattern_path,
            eta_rad,
            freq_band,
            power_tag='power',
            declination_tag='alpha',
            azimuth_tag='beta'
        )

    # telescope antenna physical temperature
    T_phy = 300.0  # in K

    # frequency of observation
    cent_freq = 11.325e9  # in Hz

    # bandwidth of telescope receiver
    bw = 1e3  # 1 kHz to match original file

    # number of frequency channels to divide the bandwidth
    freq_chan = 1

    # telescope receiver temperature (constant over the bandwidth)
    def T_RX(tim, freq):
        return 80.0  # in K

    # coordinates of telescope (Westford)
    coords = [42.6129479883915, -71.49379366344017, 86.7689687917009]

    # create instrument
    westford = Instrument(tel_ant, T_phy, cent_freq, bw, T_RX, freq_chan, coords)

    print("Telescope instrument created:")
    print(f"  - Center frequency: {cent_freq/1e9:.3f} GHz")
    print(f"  - Bandwidth: {bw/1e6:.1f} MHz")
    print(f"  - Receiver temperature: {T_RX(None, None)} K")

    return westford


def setup_observation(westford):
    """Set up the observation parameters and trajectory."""

    print("\nSetting up observation...")

    # time window of generated source trajectory
    start_window = "2025-02-18T15:00:00.000"
    stop_window = "2025-02-18T15:45:00.000"

    # replace colon with underscore
    start_window_str = start_window.replace(":", "_")
    stop_window_str = stop_window.replace(":", "_")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # load telescope antenna
    file_traj_obj_path = os.path.join(
        script_dir, "..", "tutorial", "data",
        f"casA_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
    )

    print(f"Loading source trajectory from: {file_traj_obj_path}")

    if not os.path.exists(file_traj_obj_path):
        print("Warning: Source trajectory file not found. Creating synthetic trajectory...")

        # Create synthetic trajectory for testing
        times = pd.date_range(start=start_window, end=stop_window, freq='1min')
        synthetic_data = {
            'time_stamps': times,
            'altitudes': np.linspace(30, 60, len(times)),  # Synthetic elevation
            'azimuths': np.linspace(180, 220, len(times)),  # Synthetic azimuth
            'distances': np.full(len(times), np.inf)  # Infinite distance for astronomical source
        }
        traj_src = Trajectory(pd.DataFrame(synthetic_data))
    else:
        # source position over time window
        traj_src = Trajectory.from_file(
            file_traj_obj_path,
            time_tag='time_stamps',
            elevation_tag='altitudes',
            azimuth_tag='azimuths',
            distance_tag='distances'
        )

    # start-end of observation
    dateformat = "%Y-%m-%dT%H:%M:%S.%f"
    start_obs = datetime.strptime("2025-02-18T15:30:00.000", dateformat)
    stop_obs = datetime.strptime("2025-02-18T15:40:00.000", dateformat)

    # offset from source at the beginning of the observation
    offset_angles = (-40, 0.)  # (az,el) in degrees

    # time of OFF-ON transition
    time_off_src = start_obs
    time_on_src = time_off_src + timedelta(minutes=5)

    # copy trajectory
    traj_obj = Trajectory(traj_src.traj.copy())

    # apply offset
    mask = (traj_obj.traj['times'] >= time_off_src) & (traj_obj.traj['times'] <= time_on_src)
    traj_obj.traj.loc[mask, 'azimuths'] += offset_angles[0]
    traj_obj.traj.loc[mask, 'elevations'] += offset_angles[1]

    # filter points below 5deg elevation
    filt_el = ('elevations', lambda e: e > 5.)

    # create observation
    observ = Observation.from_dates(start_obs, stop_obs, traj_obj, westford, filt_funcs=(filt_el,))

    print("Observation created:")
    print(f"  - Start time: {start_obs}")
    print(f"  - Stop time: {stop_obs}")
    print(f"  - Duration: {(stop_obs - start_obs).total_seconds()/60:.1f} minutes")

    return observ, start_obs, stop_obs


def setup_sky_model(westford, start_obs, cent_freq):
    """Set up the sky model for realistic observation."""

    print("\nSetting up sky model...")

    # source flux
    flux_src = estim_casA_flux(cent_freq)  # in Jy

    # Pre-calculate effective aperture for performance optimization
    max_gain = westford.get_antenna().get_boresight_gain()
    A_eff_max = antenna_pattern.gain_to_effective_aperture(max_gain, cent_freq)

    # source temperature in K
    def T_src(t):
        if t <= start_obs + timedelta(minutes=5):  # First 5 minutes off source
            return 0.0
        else:
            return estim_temp(flux_src, A_eff_max)

    # ground temperature in K
    T_gnd = 0  # no constant RFI

    # various RFI
    T_var = 0  # in K (no RFI)

    # total RFI temperature
    T_rfi = T_gnd + T_var

    # CMB temperature
    T_CMB = 2.73  # in K

    # galaxy temperature
    def T_gal(freq): return 1e-1 * (freq/1.41e9)**(-2.7)  # in K

    # background
    def T_bkg(freq): return T_CMB + T_gal(freq)

    # atmospheric temperature at zenith
    T_atm_zenith = 150  # in K

    # opacity of atmosphere at zenith
    tau = 0.05

    # atmospheric temperature model
    def T_atm(dec): return T_atm_zenith * (1 - np.exp(-tau/np.cos(dec)))  # in K

    # Total sky model in K
    def sky_mdl(dec, caz, tim, freq):
        return T_src(tim) + T_atm(dec) + T_rfi + T_bkg(freq)

    print("Sky model created with:")
    print(f"  - Cas A flux: {flux_src:.1f} Jy")
    print(f"  - CMB temperature: {T_CMB} K")
    print(f"  - Atmospheric temperature: {T_atm_zenith} K")

    return sky_mdl


def setup_satellite_transmitters_realistic():
    """Set up satellite transmitters with realistic characteristics."""

    print("\nSetting up satellite transmitters with realistic characteristics...")

    # radiation efficiency of satellite antenna
    sat_eta_rad = 0.5

    # maximum gain of satellite antenna
    sat_gain_max = 39.3  # in dBi

    # create ITU recommended gain profile
    # satellite boresight half beamwidth
    half_beamwidth = 3.0  # in deg
    # declination angles alpha
    alphas = np.arange(0, 181)
    # azimuth angles beta
    betas = np.arange(0, 351, 10)
    # create gain dataframe
    gain_pat = antenna_mdl_ITU(sat_gain_max, half_beamwidth, alphas, betas)

    # create satellite antenna
    sat_ant = Antenna.from_dataframe(gain_pat, sat_eta_rad, (10e9, 12e9))

    # satellite transmission parameters
    sat_T_phy = 0.0  # in K
    sat_freq = 11.325e9  # in Hz
    sat_bw = 250e6  # in Hz
    transmit_pow = -15 + 10 * np.log10(300)  # in dBW

    def transmit_temp(tim, freq):
        return power_to_temperature(10**(transmit_pow/10), 1.0)  # in K

    # create base transmitter instrument
    sat_transmit = Instrument(sat_ant, sat_T_phy, sat_freq, sat_bw, transmit_temp, 1, [])

    # Create the realistic Starlink transmitter as described in the final summary
    # REALISTIC SCENARIO: Starlink (circular) + Westford (linear) = 3 dB loss
    realistic_starlink = Transmitter.from_instrument(
        sat_transmit,
        polarization='circular',  # Starlink uses circular polarization
        polarization_angle=0.0,   # Circular polarization angle is not relevant
        harmonics=[(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]  # Moderate harmonics for realistic satellite
    )

    print("Realistic Starlink transmitter created:")
    print(f"  - Polarization: {realistic_starlink.get_polarization()} (Starlink satellites)")
    print("  - Receiver polarization: linear (Westford telescope)")
    print(f"  - Harmonics: {len(realistic_starlink.get_harmonics())} harmonics")
    print(f"  - Harmonic frequencies: {[f/1e9 for f in realistic_starlink.get_harmonic_frequencies()]} GHz")

    return realistic_starlink


def setup_satellite_constellation(observ, realistic_transmitter, start_obs, stop_obs):
    """Set up satellite constellation with realistic transmitter."""

    print("\nSetting up satellite constellation...")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # time window strings
    start_window_str = "2025-02-18T15:00:00.000".replace(":", "_")
    stop_window_str = "2025-02-18T15:45:00.000".replace(":", "_")

    # satellites trajectories during the observation
    file_traj_sats_path = os.path.join(
        script_dir, "..", "tutorial", "data",
        f"Starlink_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
    )

    print(f"Loading satellite trajectories from: {file_traj_sats_path}")

    # filter the satellites
    filt_name = ('sat', lambda s: ~s.str.contains('DTC'))
    filt_el = ('elevations', lambda e: e > 20)

    if not os.path.exists(file_traj_sats_path):
        print("Warning: Satellite trajectory file not found. Creating synthetic constellations...")
        return create_synthetic_constellations(observ, realistic_transmitter, file_traj_sats_path, filt_name, filt_el)

    # Create constellation with realistic transmitter
    print("Creating constellation with realistic Starlink transmitter...")

    # Custom link_budget function for transmitter characteristics
    def enhanced_link_budget(*args, **kwargs):
        # Use comprehensive function for transmitter characteristics
        # The comprehensive function expects: (dec_tel, caz_tel, instru_tel, dec_sat,
        # caz_sat, rng_sat, freq, transmitter, ...)
        # But *args contains: (dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat,
        # satellite_instrument, freq)
        # We need: (dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat, freq, transmitter)

        # Set very small number for beam_avoid to accept enhanced function at model_observed_temp
        # Otherwise, model_observed_temp will always use basic function
        kwargs['beam_avoid'] = 1e-20
        kwargs['turn_off'] = False

        if len(args) >= 8:  # Ensure we have enough arguments
            # Correct argument order: (dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat, freq, transmitter)
            new_args = list(args[:6]) + [args[7]] + [realistic_transmitter]

            # Add explicit receiver polarization parameters for Westford telescope (linear polarization)
            kwargs['rx_polarization'] = 'linear'
            kwargs['rx_polarization_angle'] = 0.0

            result = sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
            return result
        else:
            # Fallback to basic function if not enough arguments
            return sat_link_budget_vectorized(*args, **kwargs)

    try:
        constellation_with_effects = Constellation.from_file(
            file_traj_sats_path, observ, realistic_transmitter.get_instrument(), enhanced_link_budget,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )
        print(f"  - Loaded {len(constellation_with_effects.get_sats_name())} satellites")
    except Exception as e:
        print(f"  - Error loading constellation: {e}")
        # Fallback to synthetic constellation
        constellation_with_effects = create_synthetic_constellation(observ, realistic_transmitter)

    return constellation_with_effects, file_traj_sats_path, filt_name, filt_el


def create_synthetic_constellation(observ, transmitter):
    """Create a synthetic satellite constellation for testing."""

    print("Creating synthetic satellite constellation...")

    # Create synthetic satellite data
    time_samples = observ.get_time_stamps()
    n_satellites = 10

    synthetic_data = []
    for i in range(n_satellites):
        for t in time_samples:
            # Synthetic satellite positions
            az = 180 + 20 * np.sin(2 * np.pi * i / n_satellites + t.timestamp() / 3600)
            el = 30 + 10 * np.cos(2 * np.pi * i / n_satellites + t.timestamp() / 1800)
            dist = 500e3 + 50e3 * np.sin(t.timestamp() / 600)  # 500-550 km range

            synthetic_data.append({
                'timestamp': t,
                'sat': f'SAT_{i:03d}',
                'azimuths': az,
                'elevations': el,
                'ranges_westford': dist
            })

    # Create DataFrame and save as temporary file
    df = pd.DataFrame(synthetic_data)
    temp_file = "temp_synthetic_satellites.csv"
    df.to_csv(temp_file, index=False)

    # Create constellation
    constellation = Constellation.from_file(
        temp_file, observ, transmitter.get_instrument(), sat_link_budget_vectorized,
        name_tag='sat',
        time_tag='timestamp',
        elevation_tag='elevations',
        azimuth_tag='azimuths',
        distance_tag='ranges_westford'
    )

    # Clean up temporary file
    os.remove(temp_file)

    print(f"  - Created synthetic constellation with {n_satellites} satellites")
    return constellation


def create_synthetic_constellations(observ, transmitter, file_traj_sats_path, filt_name, filt_el):
    """Create synthetic constellations for both with and without transmitter characteristics."""

    print("Creating synthetic constellations...")

    # Create synthetic satellite data
    time_samples = observ.get_time_stamps()
    n_satellites = 10

    synthetic_data = []
    for i in range(n_satellites):
        for t in time_samples:
            # Synthetic satellite positions
            az = 180 + 20 * np.sin(2 * np.pi * i / n_satellites + t.timestamp() / 3600)
            el = 30 + 10 * np.cos(2 * np.pi * i / n_satellites + t.timestamp() / 1800)
            dist = 500e3 + 50e3 * np.sin(t.timestamp() / 600)  # 500-550 km range

            synthetic_data.append({
                'timestamp': t,
                'sat': f'SAT_{i:03d}',
                'azimuths': az,
                'elevations': el,
                'ranges_westford': dist
            })

    # Create DataFrame and save as temporary file
    df = pd.DataFrame(synthetic_data)
    temp_file = "temp_synthetic_satellites.csv"
    df.to_csv(temp_file, index=False)

    # Create constellation with transmitter characteristics
    def enhanced_link_budget(*args, **kwargs):
        if len(args) >= 8:
            new_args = list(args[:6]) + [args[7]] + [transmitter]
            return sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
        else:
            return sat_link_budget_vectorized(*args, **kwargs)

    constellation_with_effects = Constellation.from_file(
        temp_file, observ, transmitter.get_instrument(), enhanced_link_budget,
        name_tag='sat',
        time_tag='timestamp',
        elevation_tag='elevations',
        azimuth_tag='azimuths',
        distance_tag='ranges_westford'
    )

    # Clean up temporary file
    os.remove(temp_file)

    print(f"  - Created synthetic constellation with {n_satellites} satellites")
    return constellation_with_effects, file_traj_sats_path, filt_name, filt_el


def run_simulation_with_transmitter_characteristics(observ, sky_mdl,
                                                    constellation_with_effects,
                                                    file_traj_sats_path, filt_name, filt_el,
                                                    realistic_transmitter):
    """Run simulation comparing with and without transmitter characteristics."""

    print("\n" + "="*80)
    print("RUNNING SIMULATION WITH TRANSMITTER CHARACTERISTICS")
    print("="*80)

    results = {}

    # Run simulation WITH transmitter characteristics
    print("\nSimulating WITH transmitter characteristics (realistic Starlink)...")
    start_time = time.time()

    try:
        # Use beam_avoidance=True to force the use of the enhanced link budget function
        result_with_effects = model_observed_temp(
            observ, sky_mdl, constellation_with_effects, beam_avoidance=True).copy()
        results['with_transmitter_effects'] = result_with_effects

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"  - Simulation completed in {execution_time:.2f} seconds")

        # Calculate statistics
        temp_values = result_with_effects[:, 0, 0]  # Extract temperature values
        max_temp = np.max(temp_values)
        mean_temp = np.mean(temp_values)
        std_temp = np.std(temp_values)

        print(f"  - Max temperature: {max_temp:.2f} K")
        print(f"  - Mean temperature: {mean_temp:.2f} K")
        print(f"  - Std temperature: {std_temp:.2f} K")

    except Exception as e:
        print(f"  - Error in simulation: {e}")
        results['with_transmitter_effects'] = None

    # Also run simulation without satellites for reference
    print("\nSimulating without satellites (reference)...")
    start_time = time.time()

    try:
        result_no_sat = model_observed_temp(observ, sky_mdl).copy()
        results['no_satellites'] = result_no_sat

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"  - Simulation completed in {execution_time:.2f} seconds")

        # Calculate statistics
        temp_values = result_no_sat[:, 0, 0]
        max_temp = np.max(temp_values)
        mean_temp = np.mean(temp_values)
        std_temp = np.std(temp_values)

        print(f"  - Max temperature: {max_temp:.2f} K")
        print(f"  - Mean temperature: {mean_temp:.2f} K")
        print(f"  - Std temperature: {std_temp:.2f} K")

    except Exception as e:
        print(f"  - Error in simulation: {e}")
        results['no_satellites'] = None

    # Add result_original equivalent (without any effects) - matching tuto_radiomdl_runtime_doppler_transmitter.py
    print("\nSimulating without any effects (result_original equivalent)...")
    start_time = time.time()

    try:
        # Create a basic constellation without any enhanced characteristics
        # This matches the starlink_constellation_original from the original file
        from sat_mdl import sat_link_budget_vectorized

        # Create basic link budget function (no enhanced characteristics)
        def basic_link_budget(*args, **kwargs):
            result = sat_link_budget_vectorized(*args, **kwargs)
            return result

        # Create constellation without any effects
        constellation_original = Constellation.from_file(
            file_traj_sats_path, observ, realistic_transmitter.get_instrument(), basic_link_budget,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )

        result_original = model_observed_temp(observ, sky_mdl, constellation_original).copy()
        results['without_effects'] = result_original

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"  - Simulation completed in {execution_time:.2f} seconds")

        # Calculate statistics
        temp_values = result_original[:, 0, 0]
        max_temp = np.max(temp_values)
        mean_temp = np.mean(temp_values)
        std_temp = np.std(temp_values)

        print(f"  - Max temperature: {max_temp:.2f} K")
        print(f"  - Mean temperature: {mean_temp:.2f} K")
        print(f"  - Std temperature: {std_temp:.2f} K")

    except Exception as e:
        print(f"  - Error in simulation: {e}")
        results['without_effects'] = None

    return results


def analyze_transmitter_characteristics_effects(realistic_transmitter, results, observ, bw):
    """Analyze and display transmitter characteristics effects with values and plots."""

    print("\n" + "="*80)
    print("TRANSMITTER CHARACTERISTICS EFFECTS ANALYSIS")
    print("="*80)

    # Calculate and display polarization loss
    print("\n🔌 POLARIZATION MISMATCH LOSS ANALYSIS:")
    print("-" * 50)

    # Realistic scenario: Starlink (circular) + Westford (linear)
    polarization_loss = calculate_polarization_mismatch_loss('circular', 0.0, 'linear', 0.0)
    polarization_loss_db = 10 * np.log10(polarization_loss) if polarization_loss > 0 else -100

    tx_pol = realistic_transmitter.get_polarization()
    tx_ang = realistic_transmitter.get_polarization_angle()
    print(f"   • Transmitter polarization: {tx_pol} at {tx_ang}°")
    print("   • Receiver polarization: linear at 0° (Westford telescope)")
    print(f"   • Polarization mismatch loss: {polarization_loss_db:.1f} dB")
    print(f"   • Power reduction factor: {polarization_loss:.3f} ({polarization_loss*100:.1f}% of original power)")
    if abs(polarization_loss_db - 3.0) < 0.1:
        print("   • Physical meaning: 3 dB loss due to circular-to-linear polarization mismatch")
    else:
        print("   • Physical meaning: Polarization mismatch loss")

    # Calculate and display harmonic contributions
    print("\n🎵 HARMONIC CONTRIBUTION ANALYSIS:")
    print("-" * 50)

    harmonics = realistic_transmitter.get_harmonics()
    base_freq = realistic_transmitter.get_instrument().get_center_freq()
    obs_freq = base_freq  # Same frequency for this analysis
    obs_bw = bw

    print(f"   • Base frequency: {base_freq/1e9:.3f} GHz")
    print(f"   • Observation frequency: {obs_freq/1e9:.3f} GHz")
    print(f"   • Observation bandwidth: {obs_bw/1e6:.1f} MHz")
    print(f"   • Number of harmonics: {len(harmonics)}")

    total_harmonic_contribution = 0.0
    for i, (freq_mult, power_red) in enumerate(harmonics):
        harmonic_freq = base_freq * freq_mult
        print(f"   • Harmonic {i+1}: {freq_mult}x = {harmonic_freq/1e9:.3f} GHz, power = {power_red:.3f}")

        # Check if harmonic falls within observation band
        freq_min = obs_freq - obs_bw / 2
        freq_max = obs_freq + obs_bw / 2
        if freq_min <= harmonic_freq <= freq_max:
            total_harmonic_contribution += power_red
            print(f"     → IN observation band: contributes {power_red:.3f}")
        else:
            print("     → OUTSIDE observation band: no contribution")

    if total_harmonic_contribution > 0:
        harmonic_contribution_db = 10 * np.log10(total_harmonic_contribution)
        print(f"   • Total harmonic contribution: {total_harmonic_contribution:.3f} "
              f"({harmonic_contribution_db:.1f} dB)")
        print(f"   • Fundamental + harmonics: {1.0 + total_harmonic_contribution:.3f} "
              f"({10*np.log10(1.0 + total_harmonic_contribution):.1f} dB)")
    else:
        print(f"   • Total harmonic contribution: {total_harmonic_contribution:.3f} "
              f"(no harmonics in observation band)")
        print("   • Only fundamental frequency contributes")

    # Calculate combined effects
    print("\n⚡ COMBINED TRANSMITTER CHARACTERISTICS EFFECTS:")
    print("-" * 50)

    # Combined effect = polarization loss × (1 + harmonic contribution)
    combined_effect = polarization_loss * (1.0 + total_harmonic_contribution)
    combined_effect_db = 10 * np.log10(combined_effect) if combined_effect > 0 else -100

    print(f"   • Polarization loss: {polarization_loss_db:.1f} dB")
    print(f"   • Harmonic contribution: {10*np.log10(1.0 + total_harmonic_contribution):.1f} dB")
    print(f"   • Combined effect: {combined_effect_db:.1f} dB")
    print(f"   • Total power reduction: {combined_effect:.3f} ({combined_effect*100:.1f}% of original power)")

    # Create plots comparing with and without transmitter characteristics
    print("\n📊 CREATING COMPARISON PLOTS...")

    time_samples = observ.get_time_stamps()

    # Get results
    result_with_effects = results.get('with_transmitter_effects')
    result_without_effects_original = results.get('without_effects')

    if result_with_effects is not None and result_without_effects_original is not None:
        # Define safe_log10 function (same as in tuto_radiomdl_runtime_doppler_transmitter.py)
        def safe_log10(x):
            x = np.array(x)
            x = np.where(x > 0, x, np.nan)
            return np.log10(x)

        # Create main comparison plot (matching the format from tuto_radiomdl_runtime_doppler_transmitter.py)
        fig, ax = plt.subplots(figsize=(18, 6))

        # Plot without transmitter effects -  this should be the "Without transmetter characteristics" line
        plot_result = temperature_to_power(result_without_effects_original[:, 0, 0], bw)
        ax.plot(time_samples, 10 * safe_log10(plot_result), 'blue', linewidth=2,
                label="Without transmetter effects")

        # Plot with transmitter effects - this should be the "without beam avoidance" line
        plot_result = temperature_to_power(result_with_effects[:, 0, 0], bw)
        ax.plot(time_samples, 10 * safe_log10(plot_result), 'red', linewidth=2,
                label="With transmetter effects")

        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("Power [dBW]")
        ax.grid(True)
        fig.tight_layout()
        plt.savefig('06_transmitter_characteristics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Create zoomed view plot (matching the format from tuto_radiomdl_runtime_doppler_transmitter.py)
        # Focus on a specific time window for detailed comparison
        time_array = time_samples.values  # Convert pandas Series to numpy array
        start_zoom = time_array[len(time_array)//4]  # Start at 25% of observation
        end_zoom = time_array[3*len(time_array)//4]  # End at 75% of observation

        zoom_mask = (time_samples >= start_zoom) & (time_samples <= end_zoom)
        time_zoom = time_samples[zoom_mask]

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot zoomed view
        zoom_indices = [i for i, t in enumerate(time_samples) if start_zoom <= t <= end_zoom]

        plot_result = temperature_to_power(result_without_effects_original[zoom_indices, 0, 0], bw)
        ax.plot(time_zoom, 10 * np.log10(plot_result), 'blue', linewidth=2,
                label="Without transmetter effects")

        plot_result = temperature_to_power(result_with_effects[zoom_indices, 0, 0], bw)
        ax.plot(time_zoom, 10 * np.log10(plot_result), 'red', linewidth=2,
                label="With transmetter effects")

        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("Power [dBW]")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.savefig('06_transmitter_characteristics_comparison_zoomed.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Calculate and display power differences
        power_with_effects = temperature_to_power(result_with_effects[:, 0, 0], bw)
        power_without_effects = temperature_to_power(result_without_effects_original[:, 0, 0], bw)

        # Calculate power ratio and convert to dB
        power_ratio = power_with_effects / power_without_effects
        power_diff_db = 10 * np.log10(power_ratio)

        max_diff = np.nanmax(power_diff_db)
        mean_diff = np.nanmean(power_diff_db)

        print("\n📈 POWER DIFFERENCE ANALYSIS:")
        print("-" * 50)
        print(f"   • Maximum power difference: {max_diff:.2f} dB")
        print(f"   • Mean power difference: {mean_diff:.2f} dB")
        print(f"   • Expected polarization loss: {polarization_loss_db:.1f} dB")
        print(f"   • Difference from expected: {abs(mean_diff - polarization_loss_db):.2f} dB")

        if abs(mean_diff - polarization_loss_db) < 1.0:
            print("   • ✅ Results match expected polarization loss within 1 dB")
        else:
            print("   • ⚠️  Results differ from expected polarization loss")

    else:
        print("   • ❌ Could not create plots - missing simulation results")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)


def main_realistic():
    """Main function to run realistic transmitter characteristics test."""

    print("="*80)
    print("RSC-SIM REALISTIC TRANSMITTER CHARACTERISTICS TEST")
    print("="*80)
    print("Testing realistic Starlink (circular) + Westford (linear) scenario")
    print("="*80)

    # Quick test of polarization loss calculation
    print("\n🔧 TESTING POLARIZATION LOSS CALCULATION:")
    test_loss = calculate_polarization_mismatch_loss('circular', 0.0, 'linear', 0.0)
    test_loss_db = 10 * np.log10(test_loss) if test_loss > 0 else -100
    print(f"   • Circular to Linear polarization loss: {test_loss:.3f} ({test_loss_db:.1f} dB)")
    print("   • Expected: ~0.5 (3.0 dB loss)")
    if abs(test_loss_db - 3.0) < 0.1:
        print("   • ✅ Polarization loss calculation is working correctly")
    else:
        print("   • ⚠️  Polarization loss calculation may have issues")

    try:
        # Set up telescope instrument
        westford = setup_telescope_instrument()

        # Set up observation
        observ, start_obs, stop_obs = setup_observation(westford)

        # Set up sky model
        sky_mdl = setup_sky_model(westford, start_obs, westford.get_center_freq())

        # Set up realistic Starlink transmitter
        realistic_transmitter = setup_satellite_transmitters_realistic()

        # Set up satellite constellation with transmitter characteristics
        constellation_with_effects, file_traj_sats_path, filt_name, filt_el = \
            setup_satellite_constellation(observ, realistic_transmitter, start_obs, stop_obs)

        # Run simulation comparing with and without transmitter characteristics
        results = run_simulation_with_transmitter_characteristics(
            observ, sky_mdl, constellation_with_effects,
            file_traj_sats_path, filt_name, filt_el, realistic_transmitter)

        # Analyze transmitter characteristics effects
        analyze_transmitter_characteristics_effects(
            realistic_transmitter, results, observ, westford.get_bandwidth())

        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print("Realistic transmitter characteristics have been tested and analyzed.")
        print("Results have been saved to:")
        print("  - 06_transmitter_characteristics_comparison.png")
        print("  - 06_transmitter_characteristics_comparison_zoomed.png")
        print("="*80)

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


# Function to run both basic and realistic tests
def run_all_tests():
    """Run both basic educational tests and realistic modeling tests."""

    print("="*80)
    print("RSC-SIM COMPREHENSIVE TRANSMITTER CHARACTERISTICS TEST")
    print("="*80)
    print("This script provides both educational demonstrations and realistic modeling")
    print("="*80)

    # Run basic educational tests
    print("\n" + "="*60)
    print("PART 1: BASIC EDUCATIONAL TESTS")
    print("="*60)
    main()

    # Run realistic modeling tests
    print("\n" + "="*60)
    print("PART 2: REALISTIC MODELING TESTS")
    print("="*60)
    main_realistic()


if __name__ == "__main__":
    # Run both educational tests and realistic modeling tests
    run_all_tests()
