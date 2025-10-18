#!/usr/bin/env python3
"""
Tutorial 08: Direct vs Aggregate Satellite Effects Analysis

This tutorial demonstrates the calculation of direct effects (single satellite) vs
aggregate effects (multiple satellites) interacting with a receiver, representing
the core distinction between "one-to-one" and "many-to-one" calculations in the
RSC-SIM framework.

It covers:

1. Direct effects: Single satellite interference analysis
2. Aggregate effects: Multiple satellite interference analysis
3. Comparison between direct and aggregate effects
4. Visualization of direct vs aggregate interference effects

Learning Objectives:
- Understand the difference between direct (one-to-one) and aggregate (many-to-one) satellite effects
- Learn to implement single satellite direct effects analysis
- Learn to implement many-to-one aggregate interference calculations
- Visualize direct vs aggregate interference patterns

Prerequisites:
- Completion of Tutorial 01: Basic Radio Astronomy Observation
- Completion of Tutorial 02: Satellite Interference Analysis
- Understanding of satellite constellation modeling

Output Files:
- 08_direct_vs_aggregate_comparison.png: Comparison of direct vs aggregate effects
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from radio_types import Trajectory, Observation, Constellation  # noqa: E402
from obs_mdl import model_observed_temp_with_atmospheric_refraction_vectorized  # noqa: E402
from astro_mdl import temperature_to_power  # noqa: E402
from sat_mdl import sat_link_budget_vectorized  # noqa: E402

# Import shared utilities
from shared import (  # noqa: E402
    setup_westford_telescope,
    setup_satellite_transmitter,
    create_sky_model,
    setup_plotting,
    OBSERVATION_START,
    OBSERVATION_END,
    OFFSET_ANGLES,
    TIME_ON_SOURCE,
    MIN_ELEVATION,
    STARLINK_TRAJECTORY_FILE
)


def lnk_bdgt(*args, **kwargs):
    """
    Link budget wrapper function for satellite interference calculations.
    """
    # Set beam avoidance parameters to accept custom link budget model_observed_temp_with_atmospheric_refraction_vectorized  # noqa: E501
    kwargs['beam_avoid'] = 1e-20
    kwargs['turn_off'] = False

    return sat_link_budget_vectorized(*args, **kwargs)


def ensure_full_time_coverage(observation, time_samples, target_length):
    """
    Ensure time samples cover the full observation period.

    Args:
        observation: Observation object
        time_samples: Current time samples
        target_length: Target length for time samples

    Returns:
        Time samples covering the full observation period
    """
    if len(time_samples) < target_length:
        print(f"Padding time samples from {len(time_samples)} to {target_length}")
        # Create time samples for the full observation period
        start_time, end_time = observation.get_time_bounds()
        full_time_samples = pd.date_range(start=start_time, end=end_time,
                                          periods=target_length, freq='S')
        return full_time_samples
    elif len(time_samples) > target_length:
        return time_samples[:target_length]
    else:
        return time_samples


def create_individual_satellite_constellations(
    observation: Observation,
    satellite_transmitter,
    satellite_data: pd.DataFrame,
    max_satellites: int = 1
) -> List[Constellation]:
    """
    Create individual constellation object for a single satellite (direct effects).

    Args:
        observation: Observation object
        satellite_transmitter: Satellite transmitter instrument
        satellite_data: DataFrame containing satellite trajectory data
        max_satellites: Maximum number of satellites to analyze individually (should be 1 for direct effects)

    Returns:
        List containing single Constellation object for direct effects analysis
    """
    individual_constellations = []

    # Get observation time bounds
    start_time, end_time = observation.get_time_bounds()

    # Find satellites that are visible throughout the ENTIRE observation period
    fully_visible_satellites = []
    unique_satellites = satellite_data['sat'].unique()

    print(f"Checking full visibility of {len(unique_satellites)} satellites during entire observation period...")
    print("Looking for a single satellite for direct effects analysis...")

    for sat_name in unique_satellites:
        # Filter data for this specific satellite
        sat_data = satellite_data[satellite_data['sat'] == sat_name].copy()

        # Check if satellite has data throughout the ENTIRE observation period
        sat_times = pd.to_datetime(sat_data['times'])
        visible_times = sat_times[(sat_times >= start_time) & (sat_times <= end_time)]

        # Check if satellite is visible for a significant portion of the observation period
        if len(visible_times) > 0:
            # Check if the satellite covers a significant time range
            time_coverage = (visible_times.max() - visible_times.min()).total_seconds()
            observation_duration = (end_time - start_time).total_seconds()

            # Satellite must be visible for at least 50% of the observation period
            # and have data points throughout the period (not just at the beginning or end)
            if time_coverage >= 0.5 * observation_duration:
                # Check if satellite has data distributed throughout the observation period
                time_span = visible_times.max() - visible_times.min()
                data_density = len(visible_times) / time_span.total_seconds() if time_span.total_seconds() > 0 else 0

                # Require reasonable data density (at least 1 point per 10 seconds)
                if data_density >= 0.1:
                    fully_visible_satellites.append(sat_name)
                    print(f"  ✓ {sat_name}: visible for {time_coverage:.1f}s out of "
                          f"{observation_duration:.1f}s (density: {data_density:.2f} pts/s)")
                    print("  Selected for direct effects analysis")
                    break  # Only need one satellite for direct effects

    visible_satellites = fully_visible_satellites

    if len(visible_satellites) == 0:
        print("No satellites found with 50% visibility. Trying with 30% visibility...")
        # Fallback: try with lower visibility threshold
        for sat_name in unique_satellites:
            sat_data = satellite_data[satellite_data['sat'] == sat_name].copy()
            sat_times = pd.to_datetime(sat_data['times'])
            visible_times = sat_times[(sat_times >= start_time) & (sat_times <= end_time)]

            if len(visible_times) > 0:
                time_coverage = (visible_times.max() - visible_times.min()).total_seconds()
                observation_duration = (end_time - start_time).total_seconds()

                # Lower threshold: 30% visibility
                if time_coverage >= 0.3 * observation_duration:
                    time_span = visible_times.max() - visible_times.min()
                    data_density = (len(visible_times) / time_span.total_seconds()
                                    if time_span.total_seconds() > 0 else 0)

                    if data_density >= 0.05:  # Lower density requirement
                        visible_satellites.append(sat_name)
                        print(f"  ✓ {sat_name}: visible for {time_coverage:.1f}s out of "
                              f"{observation_duration:.1f}s (density: {data_density:.2f} pts/s)")
                        print("  Selected for direct effects analysis")
                        break  # Only need one satellite for direct effects

        if len(visible_satellites) == 0:
            raise ValueError("No satellites are visible for a significant portion of "
                             "the observation period (15:30 to 15:40)!")

    print(f"Found {len(visible_satellites)} visible satellite(s). Creating individual "
          "constellation for direct effects...")

    # Debug: Show all visible satellites
    print(f"📡 Available satellites: {visible_satellites[:10]}")  # Show first 10 satellites
    if len(visible_satellites) > 10:
        print(f"   ... and {len(visible_satellites) - 10} more satellites")

    # Only process STARLINK-5322 for direct effects (same as main script)
    target_satellite = "STARLINK-5322"
    if target_satellite in visible_satellites:
        sat_name = target_satellite
        print(f"🎯 08_ SCRIPT: Analyzing target satellite '{sat_name}' for direct effects")
    else:
        # Fallback to first visible satellite if target not found
        sat_name = visible_satellites[0]
        print(f"🎯 08_ SCRIPT: Target satellite '{target_satellite}' not found, using '{sat_name}' for direct effects")
        print(f"   Available satellites include: {[s for s in visible_satellites if 'STARLINK-5322' in s or '5322' in s]}")  # noqa: E501
    # Filter data for this specific satellite
    sat_data = satellite_data[satellite_data['sat'] == sat_name].copy()

    # Create a temporary observation with the same time bounds
    temp_obs = Observation.from_dates(
        start_time,
        end_time,
        Trajectory(sat_data),
        observation.get_instrument()
    )

    # Create constellation for this single satellite
    constellation = Constellation.from_observation(
        sat_data, temp_obs, satellite_transmitter, lnk_bdgt
    )

    individual_constellations.append(constellation)
    print(f"  ✓ Created constellation for satellite {sat_name} (direct effects)")

    return individual_constellations


def calculate_individual_interference(
    observation: Observation,
    sky_model,
    individual_constellations: List[Constellation]
) -> Dict[str, np.ndarray]:
    """
    Calculate interference from a single satellite (direct effects).

    Args:
        observation: Observation object
        sky_model: Sky temperature model function
        individual_constellations: List containing single satellite constellation

    Returns:
        Dictionary mapping satellite name to interference results
    """
    individual_results = {}

    print("Calculating direct effects from single satellite...")

    # Process the single satellite constellation
    constellation = individual_constellations[0]
    sat_name = constellation.get_sats_name()[0]  # Get the single satellite name

    # Calculate interference from this single satellite (direct effects)
    # Use same advanced function as main script with atmospheric refraction and beam avoidance
    start_time = time.time()

    # Note that model_observed_temp function cannot handle a single satellite
    # thus model_observed_temp_with_atmospheric_refraction_vectorized is used instead
    # with no atmospheric refraction correction

    # Configure atmospheric refraction parameters (same as main script)
    atmospheric_refraction_config = {
        'temperature': 288.15,  # K
        'pressure': 101325,     # Pa
        'humidity': 50.0,       # %
        'apply_refraction_correction': False,
        'refraction_model': 'standard'
    }

    result, refraction_summary = model_observed_temp_with_atmospheric_refraction_vectorized(
        observation, sky_model, constellation=constellation, beam_avoidance=True,
        atmospheric_refraction=atmospheric_refraction_config
    )
    end_time = time.time()

    individual_results[sat_name] = result
    print(f"  ✓ {sat_name} (direct effects): {end_time - start_time:.3f} seconds")

    return individual_results


def calculate_aggregate_interference(
    observation: Observation,
    sky_model,
    full_constellation: Constellation
) -> np.ndarray:
    """
    Calculate aggregate interference from all satellites simultaneously.

    Args:
        observation: Observation object
        sky_model: Sky temperature model function
        full_constellation: Constellation containing all satellites

    Returns:
        Aggregate interference results
    """
    print("Calculating aggregate satellite interference...")

    start_time = time.time()

    # Configure atmospheric refraction parameters (same as main script)
    atmospheric_refraction_config = {
        'temperature': 288.15,  # K
        'pressure': 101325,     # Pa
        'humidity': 50.0,       # %
        'apply_refraction_correction': False,
        'refraction_model': 'standard'
    }

    result, refraction_summary = model_observed_temp_with_atmospheric_refraction_vectorized(
        observation, sky_model, constellation=full_constellation, beam_avoidance=True,
        atmospheric_refraction=atmospheric_refraction_config
    )
    end_time = time.time()

    print(f"  ✓ Aggregate calculation: {end_time - start_time:.3f} seconds")
    return result


def plot_individual_vs_aggregate_comparison(
    individual_results: Dict[str, np.ndarray],
    aggregate_result: np.ndarray,
    observation: Observation,
    save_filename: str = "08_direct_vs_aggregate_comparison.png"
):
    """
    Create comparison plot between direct and aggregate interference.

    Args:
        individual_results: Dictionary containing single satellite result (direct effects)
        aggregate_result: Aggregate interference result
        observation: Observation object
        save_filename: Filename to save the plot
    """
    print("Creating direct vs aggregate comparison plot...")

    # Convert to power
    time_samples = observation.get_time_stamps()
    # Use natural time sampling (no forced resampling to match main script)
    print(f"Using natural time sampling: {len(time_samples)} samples")
    # Note: Removed forced 600-sample resampling to match main script's natural sampling
    bandwidth = observation.get_instrument().get_bandwidth()

    # Convert individual results to power (single satellite for direct effects)
    individual_powers = {}
    for sat_name, result in individual_results.items():
        individual_powers[sat_name] = temperature_to_power(result[:, 0, 0], bandwidth)

    # Convert aggregate result to power
    aggregate_power = temperature_to_power(aggregate_result[:, 0, 0], bandwidth)

    # Get direct power (single satellite)
    direct_power = list(individual_powers.values())[0]

    # Ensure aggregate power has the same shape as direct power
    if aggregate_power.shape != direct_power.shape:
        print(f"Warning: Aggregate power shape {aggregate_power.shape} != direct power shape {direct_power.shape}")
        # Truncate to the smaller size to ensure compatibility
        if len(aggregate_power.shape) == 1:
            min_length = min(len(aggregate_power), len(direct_power))
            aggregate_power = aggregate_power[:min_length]
            direct_power = direct_power[:min_length]
        else:
            min_shape = (min(aggregate_power.shape[0], direct_power.shape[0]),
                         min(aggregate_power.shape[1], direct_power.shape[1]))
            aggregate_power = aggregate_power[:min_shape[0], :min_shape[1]]
            direct_power = direct_power[:min_shape[0], :min_shape[1]]
        print(f"Adjusted both arrays to shape: {aggregate_power.shape}")

        # Ensure time samples match the final array length
        if len(time_samples) != len(aggregate_power):
            time_samples = time_samples[:len(aggregate_power)]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    # Top plot: Direct effects (single satellite)
    sat_name = list(individual_powers.keys())[0]
    ax1.plot(time_samples, 10 * np.log10(direct_power + 1e-20),
             label=f"Direct Effects ({sat_name})", color='red', linewidth=3)

    ax1.set_ylabel("Power [dBW]")
    ax1.set_title("Direct Effects: Single Satellite Interference")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Direct vs Aggregate
    ax2.plot(time_samples, 10 * np.log10(aggregate_power + 1e-20),
             label="Aggregate Effects (Many-to-One)", color='blue', linewidth=2)
    ax2.plot(time_samples, 10 * np.log10(direct_power + 1e-20),
             label="Direct Effects (One-to-One)", color='red', linewidth=2)

    ax2.set_xlabel("Time [UTC]")
    ax2.set_ylabel("Power [dBW]")
    ax2.set_title("Direct vs Aggregate Interference Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    try:
        plt.tight_layout()
    except Exception:
        print("Warning: Could not apply tight layout, using default layout")
    try:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved as '{save_filename}'")
    except MemoryError:
        print("Warning: Memory error saving high-resolution plot, trying lower resolution...")
        plt.savefig(save_filename, dpi=100, bbox_inches='tight')
        print(f"✓ Plot saved as '{save_filename}' (lower resolution)")
    plt.show()
    print(f"✓ Comparison plot saved as '{save_filename}'")


def main():
    """
    Main tutorial function demonstrating direct vs aggregate satellite effects analysis.
    """
    print("=" * 80)
    print("TUTORIAL 08: Direct vs Aggregate Satellite Effects Analysis")
    print("=" * 80)
    print("This tutorial demonstrates the difference between 'one-to-one' (direct)")
    print("and 'many-to-one' (aggregate) calculations in satellite interference modeling.")
    print("=" * 80)

    # Set up plotting
    setup_plotting()

    # =============================================================================
    # STEP 1: SET UP THE TELESCOPE INSTRUMENT
    # =============================================================================
    print("\nStep 1: Setting up the Westford telescope instrument...")

    telescope = setup_westford_telescope()
    print("✓ Telescope instrument created")
    print(f"  - Center frequency: {telescope.get_center_freq()/1e9:.3f} GHz")
    print(f"  - Bandwidth: {telescope.get_bandwidth()/1e3:.1f} kHz")

    # =============================================================================
    # STEP 2: LOAD SOURCE TRAJECTORY AND CREATE OBSERVATION
    # =============================================================================
    print("\nStep 2: Loading source trajectory and creating observation...")

    # Load source trajectory
    source_trajectory = Trajectory.from_file(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "research_tutorials", "data",
                     "casA_trajectory_Westford_2025-02-18T15_00_00.000_2025-02-18T15_45_00.000.arrow"),
        time_tag='time_stamps',
        elevation_tag='altitudes',
        azimuth_tag='azimuths',
        distance_tag='distances'
    )

    # Create pointing trajectory with offset
    pointing_trajectory = Trajectory(source_trajectory.traj.copy())
    mask = (pointing_trajectory.traj['times'] >= OBSERVATION_START) & \
           (pointing_trajectory.traj['times'] <= TIME_ON_SOURCE)
    pointing_trajectory.traj.loc[mask, 'azimuths'] += OFFSET_ANGLES[0]
    pointing_trajectory.traj.loc[mask, 'elevations'] += OFFSET_ANGLES[1]

    # Create observation
    elevation_filter = ('elevations', lambda e: e > MIN_ELEVATION)
    observation = Observation.from_dates(
        OBSERVATION_START,
        OBSERVATION_END,
        pointing_trajectory,
        telescope,
        filt_funcs=(elevation_filter,)
    )
    print("✓ Observation created")

    # =============================================================================
    # STEP 3: SET UP SATELLITE TRANSMITTERS AND LOAD CONSTELLATION DATA
    # =============================================================================
    print("\nStep 3: Setting up satellite transmitters and loading constellation data...")

    # Create satellite transmitter
    satellite_transmitter = setup_satellite_transmitter()
    print("✓ Satellite transmitter created")

    # Load satellite data using same method as main script (manual pyarrow + pandas)
    import pyarrow as pa
    with pa.memory_map(STARLINK_TRAJECTORY_FILE, 'r') as source:
        table = pa.ipc.open_file(source).read_all()
    satellite_data = table.to_pandas()

    # Rename columns and filter (same as main script)
    satellite_data = satellite_data.rename(columns={
        'timestamp': 'times',
        'sat': 'sat',
        'azimuths': 'azimuths',
        'elevations': 'elevations',
        'ranges_westford': 'distances'
    })
    satellite_data['times'] = pd.to_datetime(satellite_data['times'])

    # Filter satellites (remove DTC satellites and below 20° elevation)
    satellite_data = satellite_data[~satellite_data['sat'].str.contains('DTC')]
    satellite_data = satellite_data[satellite_data['elevations'] > 20.0]

    print(f"✓ Satellite data loaded: {len(satellite_data['sat'].unique())} unique satellites")

    # =============================================================================
    # STEP 4: CREATE INDIVIDUAL SATELLITE CONSTELLATIONS
    # =============================================================================
    print("\nStep 4: Creating single satellite constellation for direct effects...")

    # Create single satellite constellation for direct effects analysis
    individual_constellations = create_individual_satellite_constellations(
        observation, satellite_transmitter, satellite_data, max_satellites=1
    )

    # =============================================================================
    # STEP 5: CREATE FULL CONSTELLATION FOR AGGREGATE ANALYSIS
    # =============================================================================
    print("\nStep 5: Creating full constellation for aggregate analysis...")

    # Create full constellation with all satellites
    full_constellation = Constellation.from_file(
        STARLINK_TRAJECTORY_FILE,
        observation,
        satellite_transmitter,
        lnk_bdgt,
        name_tag='sat',
        time_tag='timestamp',
        elevation_tag='elevations',
        azimuth_tag='azimuths',
        distance_tag='ranges_westford',
        filt_funcs=(('sat', lambda s: ~s.str.contains('DTC')), ('elevations', lambda e: e > 20.0))
    )
    print(f"✓ Full constellation created: {len(full_constellation.get_sats_name())} satellites")

    # =============================================================================
    # STEP 6: CREATE SKY MODEL
    # =============================================================================
    print("\nStep 6: Creating sky temperature model...")

    sky_model = create_sky_model(observation)
    print("✓ Sky model created")

    # =============================================================================
    # STEP 7: CALCULATE INDIVIDUAL SATELLITE INTERFERENCE
    # =============================================================================
    print("\nStep 7: Calculating direct effects from single satellite...")

    individual_results = calculate_individual_interference(
        observation, sky_model, individual_constellations
    )

    # =============================================================================
    # STEP 8: CALCULATE AGGREGATE SATELLITE INTERFERENCE
    # =============================================================================
    print("\nStep 8: Calculating aggregate effects from all satellites...")

    aggregate_result = calculate_aggregate_interference(
        observation, sky_model, full_constellation
    )

    # =============================================================================
    # STEP 9: CREATE VISUALIZATION PLOT
    # =============================================================================
    print("\nStep 9: Creating direct vs aggregate comparison plot...")

    # Create only the direct vs aggregate comparison plot
    plot_individual_vs_aggregate_comparison(
        individual_results, aggregate_result, observation
    )

    # =============================================================================
    # SUMMARY
    # =============================================================================
    print("\n" + "=" * 80)
    print("TUTORIAL SUMMARY")
    print("=" * 80)
    print("✓ Successfully demonstrated direct vs aggregate satellite effects")
    print("✓ Implemented single satellite direct effects analysis")
    print("✓ Implemented many-to-one aggregate interference calculations")
    print("✓ Created direct vs aggregate comparison visualization")
    print("\nKey Concepts Learned:")
    print("- Direct effects analysis provides single satellite interference baseline")
    print("- Aggregate analysis captures combined effects from multiple satellites")
    print("- Many-to-one calculations are more computationally efficient")
    print("- Visual comparison reveals interference enhancement patterns")
    print("\nArchitecture Shift:")
    print("- FROM: One-to-one satellite interference calculations (direct effects)")
    print("- TO: Many-to-one aggregate interference calculations")
    print("- BENEFIT: Improved computational efficiency and realistic modeling")
    print("=" * 80)


if __name__ == "__main__":
    main()
