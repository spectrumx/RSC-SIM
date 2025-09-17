#!/usr/bin/env python3
"""
Tutorial 01: Basic Radio Astronomy Observation

This tutorial introduces the fundamental concepts of radio astronomy observation
using the RSC-SIM framework. It covers:

1. Setting up a radio telescope instrument
2. Creating observation plans and trajectories
3. Modeling sky temperature components
4. Running basic observations
5. Visualizing results

Learning Objectives:
- Understand the components of a radio telescope system
- Learn how to set up observations with ON/OFF source tracking
- Explore sky temperature models and their components
- See how radio astronomy observations are simulated

Prerequisites:
- Basic understanding of Python
- Familiarity with radio astronomy concepts (helpful but not required)
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from radio_types import Trajectory, Observation  # noqa: E402
from obs_mdl import model_observed_temp  # noqa: E402
from astro_mdl import temperature_to_power  # noqa: E402

# Import shared utilities
from shared import (  # noqa: E402
    setup_westford_telescope,
    create_sky_model,
    setup_plotting,
    plot_antenna_pattern,
    plot_trajectory_comparison,
    plot_power_time_series,
    # safe_log10,
    OBSERVATION_START,
    OBSERVATION_END,
    OFFSET_ANGLES,
    TIME_ON_SOURCE,
    MIN_ELEVATION,
    BANDWIDTH
)


def main():
    """
    Main tutorial function demonstrating basic radio astronomy observation.
    """
    print("=" * 60)
    print("TUTORIAL 01: Basic Radio Astronomy Observation")
    print("=" * 60)

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
    print(f"  - Physical temperature: {telescope.get_phy_temp():.1f} K")

    # Plot antenna pattern
    fig, ax = plot_antenna_pattern(telescope.get_antenna(), "Westford Telescope Antenna Pattern")
    plt.savefig('01_antenna_pattern.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Antenna pattern saved as '01_antenna_pattern.png'")

    # =============================================================================
    # STEP 2: LOAD SOURCE TRAJECTORY
    # =============================================================================
    print("\nStep 2: Loading Cas A source trajectory...")

    # Load source trajectory from file
    source_trajectory = Trajectory.from_file(
        os.path.join(os.path.dirname(__file__), "..", "tutorial", "data",
                     "casA_trajectory_Westford_2025-02-18T15_00_00.000_2025-02-18T15_45_00.000.arrow"),
        time_tag='time_stamps',
        elevation_tag='altitudes',
        azimuth_tag='azimuths',
        distance_tag='distances'
    )
    print("✓ Source trajectory loaded")
    print(f"  - Time range: {source_trajectory.get_traj()['times'].min()} to "
          f"{source_trajectory.get_traj()['times'].max()}")

    # =============================================================================
    # STEP 3: CREATE OBSERVATION PLAN
    # =============================================================================
    print("\nStep 3: Creating observation plan with ON/OFF source tracking...")

    # Create pointing trajectory with offset
    pointing_trajectory = Trajectory(source_trajectory.traj.copy())

    # Apply offset for OFF source observation
    mask = (pointing_trajectory.traj['times'] >= OBSERVATION_START) & \
           (pointing_trajectory.traj['times'] <= TIME_ON_SOURCE)
    pointing_trajectory.traj.loc[mask, 'azimuths'] += OFFSET_ANGLES[0]
    pointing_trajectory.traj.loc[mask, 'elevations'] += OFFSET_ANGLES[1]

    # Filter low elevation points
    elevation_filter = ('elevations', lambda e: e > MIN_ELEVATION)

    # Create observation
    observation = Observation.from_dates(
        OBSERVATION_START,
        OBSERVATION_END,
        pointing_trajectory,
        telescope,
        filt_funcs=(elevation_filter,)
    )
    print("✓ Observation created")
    print(f"  - Duration: {OBSERVATION_END - OBSERVATION_START}")
    print(f"  - OFF source: {OBSERVATION_START} to {TIME_ON_SOURCE}")
    print(f"  - ON source: {TIME_ON_SOURCE} to {OBSERVATION_END}")

    # Plot trajectory comparison
    fig, ax = plot_trajectory_comparison(
        source_trajectory.get_traj(),
        observation.get_traj(),
        "Cas A Source vs Telescope Pointing"
    )
    plt.savefig('01_trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Trajectory comparison saved as '01_trajectory_comparison.png'")

    # =============================================================================
    # STEP 4: CREATE SKY MODEL
    # =============================================================================
    print("\nStep 4: Creating sky temperature model...")

    sky_model = create_sky_model(observation)
    print("✓ Sky model created with components:")
    print("  - Astronomical source (Cas A)")
    print("  - Atmospheric emission")
    print("  - Cosmic Microwave Background (2.73 K)")
    print("  - Galactic background")
    print("  - Radio Frequency Interference")

    # =============================================================================
    # STEP 5: RUN OBSERVATION SIMULATION
    # =============================================================================
    print("\nStep 5: Running observation simulation...")

    start_time = time.time()
    result = model_observed_temp(observation, sky_model)
    end_time = time.time()

    print(f"✓ Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"  - Result shape: {result.shape}")
    print(f"  - Time samples: {result.shape[0]}")
    print(f"  - Pointings: {result.shape[1]}")
    print(f"  - Frequency channels: {result.shape[2]}")

    # =============================================================================
    # STEP 6: ANALYZE AND VISUALIZE RESULTS
    # =============================================================================
    print("\nStep 6: Analyzing and visualizing results...")

    # Convert temperature to power
    time_samples = observation.get_time_stamps()
    power_data = temperature_to_power(result[:, 0, 0], BANDWIDTH)

    # Create time series plot
    fig, ax = plot_power_time_series(
        time_samples,
        [power_data],
        ["Total Power"],
        "Radio Telescope Power vs Time",
        "Power [dBW]"
    )

    # Add vertical line for ON/OFF transition
    ax.axvline(x=TIME_ON_SOURCE, color='red', linestyle='--', alpha=0.7, label='ON/OFF transition')
    ax.legend()
    plt.savefig('01_power_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Power time series saved as '01_power_time_series.png'")

    # =============================================================================
    # STEP 7: UNDERSTAND THE RESULTS
    # =============================================================================
    print("\nStep 7: Understanding the results...")

    # Calculate statistics
    off_source_mask = time_samples <= TIME_ON_SOURCE
    on_source_mask = time_samples > TIME_ON_SOURCE

    off_power = power_data[off_source_mask]
    on_power = power_data[on_source_mask]

    print("OFF source statistics:")
    print(f"  - Mean power: {np.mean(off_power):.2e} W")
    print(f"  - Std power: {np.std(off_power):.2e} W")

    print("ON source statistics:")
    print(f"  - Mean power: {np.mean(on_power):.2e} W")
    print(f"  - Std power: {np.std(on_power):.2e} W")

    # Calculate signal-to-noise ratio
    signal = np.mean(on_power) - np.mean(off_power)
    noise = np.std(off_power)
    snr = signal / noise if noise > 0 else 0

    print(f"Signal-to-Noise Ratio: {snr:.2f}")

    # =============================================================================
    # SUMMARY
    # =============================================================================
    print("\n" + "=" * 60)
    print("TUTORIAL SUMMARY")
    print("=" * 60)
    print("✓ Successfully set up a radio telescope instrument")
    print("✓ Created an observation plan with ON/OFF source tracking")
    print("✓ Modeled sky temperature components")
    print("✓ Simulated a complete radio astronomy observation")
    print("✓ Analyzed the results and calculated signal-to-noise ratio")
    print("\nKey Concepts Learned:")
    print("- Radio telescopes measure power from astronomical sources")
    print("- ON/OFF observations help separate source signal from background")
    print("- Sky temperature models include multiple components")
    print("- Signal-to-noise ratio quantifies observation quality")
    print("\nNext Tutorial: Satellite Interference Analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
