#!/usr/bin/env python3
"""
Tutorial 07: Environmental Effects Analysis

This tutorial explores environmental effects in satellite radio astronomy
observations using the RSC-SIM framework. It provides both educational demonstrations
and realistic modeling scenarios, running both parts by default.

Educational Components:
1. DEM loading and processing with rasterio for terrain analysis
2. Elevation masking for space-to-ground interactions
3. Advanced atmospheric refraction models (Bennett's formula + enhanced models)
4. Water vapor modeling for high-frequency simulations (weather radiometry)
5. Limb refraction effects for space-to-space interactions
6. Terrestrial antenna pointing limitations and mechanical constraints
7. Integrated line-of-sight calculations with DEM ray tracing and atmospheric effects

Realistic Modeling:
8. Real Westford antenna data with environmental effects integration
9. Comprehensive terrain masking using DEM data for the Westford site
10. Atmospheric refraction and water vapor effects in link budget calculations
11. Detailed blocking analysis and visibility statistics
12. Comparison of interference predictions with and without environmental effects
13. Full RadioMdlPy workflow integration with environmental modeling

Learning Objectives:
- Understand terrain masking and line-of-sight obstruction effects
- Learn atmospheric refraction modeling and correction techniques
- Explore water vapor absorption and emission in radio astronomy
- Implement comprehensive environmental effects in link budget calculations
- Analyze realistic scenarios with terrain and atmospheric effects
- Compare interference predictions with and without environmental effects

Key Concepts:
- Terrain masking: DEM-based ray tracing for line-of-sight calculations
- Atmospheric refraction: signal bending and telescope pointing corrections
- Water vapor effects: frequency-dependent absorption and emission
- Limb refraction: space-to-space signal path atmospheric effects
- Environmental factors: integrated terrain and atmospheric modeling
- Comprehensive link budget: physics-based interference prediction with environmental effects

Output:
- Educational plots: atmospheric effects analysis and terrain masking
- Realistic comparison plots: environmental effects impact on interference
- Saved files: 07_environmental_effects.png
- Detailed blocking statistics and visibility analysis

Prerequisites:
- Tutorials 01-06 (basic observation, satellite interference, sky mapping, PSD analysis,
  Doppler effects, transmitter characteristics)
- Understanding of atmospheric propagation effects
- Familiarity with terrain analysis and DEM data
- Basic knowledge of radio astronomy observation constraints

Data Requirements:
- DEM file: USGS_OPR_MA_CentralEastern_2021_B21_be_19TBH294720.tif (Westford site)
- Antenna pattern: single_cut_res.cut (Westford antenna)
- Trajectory files: casA_trajectory_Westford_*.arrow and Starlink_trajectory_Westford_*.arrow
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from radio_types import Antenna, Instrument, Constellation, Trajectory, Observation  # noqa: E402
from astro_mdl import (  # noqa: E402
    estim_casA_flux, power_to_temperature, temperature_to_power, antenna_mdl_ITU, estim_temp
)
from sat_mdl import (  # noqa: E402
    sat_link_budget_vectorized,
    # Environmental effects functions
    calculate_comprehensive_environmental_effects_vectorized
)
from obs_mdl import model_observed_temp_with_atmospheric_refraction_vectorized  # noqa: E402
from env_mdl import AdvancedEnvironmentalEffects  # noqa: E402
import antenna_pattern  # noqa: E402

warnings.filterwarnings('ignore')


def create_demo_antenna():
    """Create a demo antenna for the Westford site"""
    # Simple antenna pattern (constant gain for demo)
    alphas = np.linspace(0, 180, 19)
    betas = np.linspace(0, 359, 37)  # Avoid 360 to prevent wrap-around issues
    gains = np.ones((len(alphas), len(betas))) * 0.5  # 50% efficiency

    # Create DataFrame for antenna pattern
    alpha_grid, beta_grid = np.meshgrid(alphas, betas, indexing='ij')
    gain_data = {
        'alphas': alpha_grid.flatten(),
        'betas': beta_grid.flatten(),
        'gains': gains.flatten()
    }
    gain_df = pd.DataFrame(gain_data)

    # Create antenna using from_dataframe method
    antenna = Antenna.from_dataframe(gain_df, rad_eff=0.5, valid_freqs=(10e9, 12e9))
    return antenna


def create_demo_satellite_trajectory():
    """Create a demo satellite trajectory"""
    # Create a simple satellite trajectory
    times = np.arange(0, 3600, 60)  # 1 hour, 1 minute intervals

    # Simple circular orbit simulation
    orbit_altitude = 550000  # meters (typical LEO)
    earth_radius = 6371000
    orbit_radius = earth_radius + orbit_altitude

    # Satellite passes from south to north
    # Use elevation angles (0° to 90°) instead of declinations
    elevation_angles = np.linspace(0, 90, len(times))  # degrees - horizon to zenith
    azimuths = np.linspace(180, 0, len(times))  # degrees
    ranges = np.full(len(times), orbit_radius)

    # Convert times to datetime objects
    start_time = datetime.now()
    datetime_times = [start_time + timedelta(seconds=int(t)) for t in times]

    traj_df = pd.DataFrame({
        'times': datetime_times,
        'azimuths': azimuths,
        'elevations': elevation_angles,
        'distances': ranges
    })

    trajectory = Trajectory(traj=traj_df)
    return trajectory


def demonstrate_advanced_environmental_effects():
    """
    Demonstrate advanced propagation and environmental effects using demo data.

    This function uses:
    - Demo antenna patterns (constant gain)
    - Demo satellite trajectories (2025-09-10 time range)
    - Demo atmospheric conditions
    - Demo environmental effects parameters

    This is for educational purposes to show how environmental effects work
    without requiring real data files.
    """
    print("=" * 80)
    print("ADVANCED PROPAGATION & ENVIRONMENTAL EFFECTS DEMONSTRATION")
    print("=" * 80)

    # Westford antenna coordinates
    westford_lat = 42.6129479883915  # degrees
    westford_lon = -71.49379366344017  # degrees
    westford_elevation = 86.7689687917009  # meters above sea level

    # Atmospheric conditions (typical for Westford, MA)
    temperature = 288.15  # K (15°C)
    pressure = 101325  # Pa (1 atm)
    humidity = 60.0  # % (moderate humidity)

    print("\nAntenna Location: Westford, MA")
    print(f"  Latitude: {westford_lat}°")
    print(f"  Longitude: {westford_lon}°")
    print(f"  Elevation: {westford_elevation} m")
    print(f"  Temperature: {temperature:.1f} K ({temperature-273.15:.1f}°C)")
    print(f"  Pressure: {pressure/1000:.1f} kPa")
    print(f"  Humidity: {humidity:.1f}%")

    # Initialize environmental effects modeling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dem_file = os.path.join(
        script_dir, "..", "tutorial", "data",
        "USGS_OPR_MA_CentralEastern_2021_B21_be_19TBH294720.tif"
    )
    environment = AdvancedEnvironmentalEffects(
        dem_file, westford_lat, westford_lon, westford_elevation,
        temperature, pressure, humidity)

    # Test specific satellite positions with atmospheric effects
    # Fix: Use proper satellite range (altitude + Earth radius)
    earth_radius = 6371000  # meters
    satellite_altitude = 550000  # meters (typical LEO altitude)
    satellite_range = earth_radius + satellite_altitude  # Total range from antenna to satellite

    test_positions = [
        (30, 0, satellite_range),    # High elevation, due north
        (10, 0, satellite_range),    # Medium elevation, due north
        (5, 0, satellite_range),     # Low elevation, due north
        (30, 90, satellite_range),   # High elevation, due east
        (30, 180, satellite_range),  # High elevation, due south
        (30, 270, satellite_range),  # High elevation, due west
        (2, 0, satellite_range),     # Very low elevation (should be blocked)
        (45, 45, satellite_range),   # High elevation, northeast
    ]

    print("\nAdvanced environmental effects results for test positions:")
    print("Alt(°)  Az(°)   Range(m)  Antenna  Elevation  LOS     Final  Apparent  WV Abs  Atm Loss")
    print("                                 Access   Masking  Visible  Elev(°)   (dB)    (dB)")
    print("-" * 85)

    for alt, az, rng in test_positions:
        antenna_access = environment.check_antenna_limitations(alt)
        elevation_ok = environment.check_elevation_masking(alt)
        is_visible, blocking_elev, atm_effects = environment.check_line_of_sight(alt, az, rng)
        final_masking, final_atm_effects = environment.apply_terrain_masking(alt, az, rng)

        apparent_elev = final_atm_effects.get('apparent_elevation', alt)
        wv_abs = final_atm_effects.get('water_vapor_absorption', 0.0)
        atm_loss = final_atm_effects.get('total_atmospheric_loss', 0.0)

        print(f"{alt:6.1f}  {az:6.1f}  {rng:8.0f}  {antenna_access!s:7s}  {elevation_ok!s:9s} "
              f"{is_visible!s:7s}  {final_masking:5.2f}  {apparent_elev:8.2f}  {wv_abs:6.3f}  {atm_loss:7.3f}")

    # Demonstrate with satellite trajectory
    print("\n" + "=" * 80)
    print("SATELLITE TRAJECTORY ANALYSIS")
    print("=" * 80)

    # Create demo satellite trajectory for analysis
    print("Creating demo satellite trajectory...")
    trajectory = create_demo_satellite_trajectory()

    # Analyze trajectory with comprehensive terrain masking
    print(f"\nAnalyzing {len(trajectory.traj)} trajectory points...")

    times = trajectory.traj['times']
    elevation_angles = trajectory.traj['elevations']
    azimuths = trajectory.traj['azimuths']
    ranges = trajectory.traj['distances']

    # Calculate comprehensive terrain masking factors
    masking_factors = []
    terrain_blocking_details = []
    visible_count = 0

    for i in range(len(times)):
        alt = elevation_angles.iloc[i]
        az = azimuths.iloc[i]
        rng = ranges.iloc[i]

        # Apply comprehensive environmental effects
        masking_factor, atm_effects = environment.apply_terrain_masking(alt, az, rng)
        masking_factors.append(masking_factor)

        # Get detailed blocking information
        antenna_ok = environment.check_antenna_limitations(alt)
        elevation_ok = environment.check_elevation_masking(alt)
        los_visible, blocking_elev, los_atm_effects = environment.check_line_of_sight(alt, az, rng)

        terrain_blocking_details.append({
            'antenna_ok': antenna_ok,
            'elevation_ok': elevation_ok,
            'los_visible': los_visible,
            'blocking_elev': blocking_elev
        })

        if masking_factor > 0.5:
            visible_count += 1

    masking_factors = np.array(masking_factors)
    visibility_percentage = (visible_count / len(times)) * 100

    # Summary statistics
    print("\n" + "=" * 80)
    print("TERRAIN MASKING ANALYSIS SUMMARY")
    print("=" * 80)

    # Calculate comprehensive blocking statistics
    antenna_blocked = sum(1 for detail in terrain_blocking_details if not detail['antenna_ok'])
    elevation_blocked = sum(1 for detail in terrain_blocking_details if not detail['elevation_ok'])
    terrain_blocked = sum(1 for detail in terrain_blocking_details if not detail['los_visible'])

    print(f"Total observation points: {len(times)}")
    print(f"Visible points: {visible_count}")
    print(f"Visibility percentage: {visibility_percentage:.1f}%")
    print(f"Blocked points: {len(times) - visible_count}")

    print("\nBlocking breakdown:")
    print(f"  Antenna limitations: {antenna_blocked} points ({antenna_blocked/len(times)*100:.1f}%)")
    print(f"  Elevation masking: {elevation_blocked} points ({elevation_blocked/len(times)*100:.1f}%)")
    print(f"  Terrain blocking: {terrain_blocked} points ({terrain_blocked/len(times)*100:.1f}%)")

    # Show trajectory details
    print("\nTrajectory details:")
    print(f"  Time range: {times.iloc[0]} to {times.iloc[-1]}")
    print(f"  Elevation range: {np.min(elevation_angles):.1f}° to {np.max(elevation_angles):.1f}°")
    print(f"  Azimuth range: {np.min(azimuths):.1f}° to {np.max(azimuths):.1f}°")
    print(f"  Range: {np.min(ranges)/1000:.1f} to {np.max(ranges)/1000:.1f} km")

    # Show detailed example points
    print("\nDetailed trajectory analysis (every 10th point):")
    print("Index  Alt(°)  Az(°)   Range(m)  Ant  Elev  LOS   Final")
    print("-" * 55)
    for i in range(0, len(times), max(1, len(times)//10)):
        alt = elevation_angles.iloc[i]
        az = azimuths.iloc[i]
        rng = ranges.iloc[i]
        mf = masking_factors[i]
        detail = terrain_blocking_details[i]

        ant_str = "✓" if detail['antenna_ok'] else "✗"
        elev_str = "✓" if detail['elevation_ok'] else "✗"
        los_str = "✓" if detail['los_visible'] else "✗"

        print(f"{i:5d}  {alt:6.1f}  {az:6.1f}  {rng:8.0f}  {ant_str:3s}  {elev_str:4s}  {los_str:3s}  {mf:6.2f}")

    # Environmental effects parameters
    print("\nEnvironmental effects parameters:")
    print(f"  Minimum elevation angle: {environment.min_elevation_angle}°")
    print(f"  Refraction coefficient: {environment.refraction_coefficient}")
    print(f"  Antenna mechanical limit: {environment.antenna_mechanical_limit}°")
    print(f"  Antenna elevation: {environment.antenna_elevation} m")
    print(f"  Surface temperature: {environment.temperature:.1f} K")
    print(f"  Surface pressure: {environment.pressure/1000:.1f} kPa")
    print(f"  Relative humidity: {environment.humidity:.1f}%")
    print(f"  Water vapor scale height: {environment.water_vapor_scale_height:.0f} m")

    # Demonstrate advanced atmospheric effects
    print("\n" + "=" * 80)
    print("ADVANCED ATMOSPHERIC EFFECTS ANALYSIS")
    print("=" * 80)

    elevation_angles = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90]
    frequencies = [1e9, 11e9, 22e9, 94e9]  # Different frequency bands

    print("Atmospheric Refraction Analysis:")
    print("True Elev  Apparent Elev  Refraction  Atm Delay  WV Abs    WV Emiss  Total Loss")
    print("                              Correction    (m)      (dB)      (K)      (dB)")
    print("-" * 80)

    for true_elev in elevation_angles:
        (apparent_elev, atm_delay, wv_abs, wv_emiss, total_loss) = \
            environment.calculate_integrated_atmospheric_effects(true_elev, 11e9)
        correction = apparent_elev - true_elev
        print(f"{true_elev:10.1f}°  {apparent_elev:12.3f}°  {correction:10.3f}°  "
              f"{atm_delay:8.1f}  {wv_abs:8.3f}  {wv_emiss:8.1f}  {total_loss:8.3f}")

    print("\nWater Vapor Effects at Different Frequencies:")
    print("Frequency  True Elev  WV Absorption  WV Emission  Total Atm Loss")
    print("   (GHz)      (°)        (dB)          (K)          (dB)")
    print("-" * 60)

    for freq in frequencies:
        for true_elev in [5, 15, 30]:
            (apparent_elev, atm_delay, wv_abs, wv_emiss, total_loss) = \
                environment.calculate_integrated_atmospheric_effects(true_elev, freq)
            print(f"{freq/1e9:8.1f}  {true_elev:10.1f}°  {wv_abs:12.3f}  "
                  f"{wv_emiss:10.1f}  {total_loss:12.3f}")

    print("\nAtmospheric Profile Analysis:")
    print("Height (m)  Temperature (K)  Pressure (kPa)  Water Vapor (g/m³)")
    print("-" * 60)

    heights = [0, 1000, 2000, 5000, 10000, 15000, 20000]
    for height in heights:
        T, P, WV = environment.calculate_atmospheric_profile(height)
        print(f"{height:10.0f}  {T:14.1f}  {P/1000:12.1f}  {WV*1000:14.2f}")

    # Demonstrate antenna pointing limitations
    print("\n" + "=" * 80)
    print("ANTENNA POINTING LIMITATIONS")
    print("=" * 80)

    print("Common antenna pointing limitations:")
    print("  - Mechanical limits: typically 5-10° minimum elevation")
    print("  - Atmospheric effects: increased noise at low elevations")
    print("  - Terrain masking: local topography blocks line of sight")
    print("  - RFI sources: ground-based interference at low elevations")

    # Show impact of different minimum elevation angles
    min_elevations = [1, 3, 5, 10, 15]
    print("\nImpact of different minimum elevation angles:")
    print("Min Elev  Visible Points  Visibility %")
    print("-" * 35)

    test_elevations = np.linspace(0, 90, 91)
    for min_elev in min_elevations:
        environment.min_elevation_angle = min_elev
        visible_count = 0
        for elev_angle in test_elevations:
            if environment.check_elevation_masking(elev_angle):
                visible_count += 1
        visibility_pct = (visible_count / len(test_elevations)) * 100
        print(f"{min_elev:8.1f}°  {visible_count:12d}  {visibility_pct:11.1f}%")

    # Demonstrate limb refraction effects (space-to-space)
    print("\n" + "=" * 80)
    print("LIMB REFRACTION EFFECTS (SPACE-TO-SPACE)")
    print("=" * 80)

    print("Limb refraction effects for space-to-space interactions:")
    print("  - Occurs when satellite signals pass through Earth's atmosphere")
    print("  - Most significant for grazing incidence angles")
    print("  - Can cause signal bending and attenuation")
    print("  - Important for satellite-to-satellite communications")

    # Simple limb refraction model
    grazing_angles = [0.1, 0.5, 1.0, 2.0, 5.0]  # degrees
    print("\nLimb refraction effects at different grazing angles:")
    print("Grazing Angle  Refraction Effect  Signal Bending")
    print("-" * 45)

    for angle in grazing_angles:
        # Apply limb refraction model
        limb_refraction, signal_bending = environment.apply_limb_refraction(angle)
        print(f"{angle:11.1f}°  {limb_refraction:14.2f}°  {signal_bending:12.3f}°")

    print("\nAnalysis complete!")
    print("\nKey findings:")
    print(f"  - {visibility_percentage:.1f}% of satellite trajectory is visible")
    print("  - Elevation masking blocks observations below minimum threshold")
    print("  - Advanced atmospheric refraction significantly affects low elevation observations")
    print("  - Water vapor absorption increases with frequency and humidity")
    print("  - Atmospheric-terrain interactions influence local atmospheric conditions")
    print("  - Integrated atmospheric effects provide comprehensive signal path modeling")
    print("  - Antenna pointing limitations further reduce observable sky area")
    print("  - Limb refraction affects space-to-space communications")
    print("  - Environmental effects modeling enables realistic radio astronomy simulations")


def practical_environmental_effects_with_real_data():
    """
    Practical application of environmental effects with real Westford data, Cas A star,
    and Starlink satellite trajectories. This function demonstrates the complete integration
    of environmental effects with the existing RadioMdlPy framework.

    IMPORTANT: This function uses real .arrow files for the 2025-02-18 time window:
    - casA_trajectory_Westford_2025-02-18T15_00_00.000_2025-02-18T15_45_00.000.arrow
    - Starlink_trajectory_Westford_2025-02-18T15_00_00.000_2025-02-18T15_45_00.000.arrow

    The observation window is set to 2025-02-18T15:30:00.000 to 2025-02-18T15:40:00.000
    """
    print("=" * 80)
    print("PRACTICAL ENVIRONMENTAL EFFECTS WITH REAL DATA")
    print("=" * 80)

    # Westford antenna coordinates and parameters
    westford_lat = 42.6129479883915  # degrees
    westford_lon = -71.49379366344017  # degrees
    westford_elevation = 86.7689687917009  # meters above sea level

    # Atmospheric conditions (typical for Westford, MA)
    temperature = 288.15  # K (15°C)
    pressure = 101325  # Pa (1 atm)
    humidity = 60.0  # % (moderate humidity)

    print("\nData Configuration:")
    print("  Antenna Location: Westford, MA")
    print(f"  Latitude: {westford_lat}°")
    print(f"  Longitude: {westford_lon}°")
    print(f"  Elevation: {westford_elevation} m")
    print(f"  Temperature: {temperature:.1f} K ({temperature-273.15:.1f}°C)")
    print(f"  Pressure: {pressure/1000:.1f} kPa")
    print(f"  Humidity: {humidity:.1f}%")

    # Environmental effects configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    environmental_config = {
        'dem_file': os.path.join(
            script_dir, "..", "tutorial", "data",
            "USGS_OPR_MA_CentralEastern_2021_B21_be_19TBH294720.tif"
        ),
        'antenna_lat': westford_lat,
        'antenna_lon': westford_lon,
        'antenna_elevation': westford_elevation,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'min_elevation_angle': 5.0,  # degrees
        'apply_terrain_masking': True,
        'apply_atmospheric_effects': True,
        'apply_limb_refraction': True  # Enable limb refraction for space-to-space interactions
    }

    print("  Environmental effects configuration:")
    print(f"    Min elevation angle: {environmental_config['min_elevation_angle']}°")
    print(f"    Apply terrain masking: {environmental_config['apply_terrain_masking']}")
    print(f"    Apply atmospheric effects: {environmental_config['apply_atmospheric_effects']}")

    # Initialize environmental effects
    print("\nInitializing environmental effects...")
    environment = AdvancedEnvironmentalEffects(
        environmental_config['dem_file'], westford_lat, westford_lon, westford_elevation,
        temperature, pressure, humidity
    )

    # Load real antenna pattern
    print("\nLoading real Westford antenna pattern...")
    file_pattern_path = os.path.join(script_dir, "..", "tutorial", "data", "single_cut_res.cut")

    # Antenna parameters
    eta_rad = 0.45  # radiation efficiency
    freq_band = (10e9, 12e9)  # valid frequency band

    try:
        tel_ant = Antenna.from_file(
            file_pattern_path,
            eta_rad,
            freq_band,
            power_tag='power',
            declination_tag='alpha',
            azimuth_tag='beta'
        )
        print("  Antenna loaded successfully")
        print(f"  Radiation efficiency: {eta_rad}")
        print(f"  Frequency band: {freq_band[0]/1e9:.1f} - {freq_band[1]/1e9:.1f} GHz")
    except Exception as e:
        print(f"  Warning: Could not load antenna pattern: {e}")
        print("  Using demo antenna pattern instead")
        tel_ant = create_demo_antenna()

    # time window of generated source trajectory (matching tuto_radiomdl_runtime_environment.py)
    start_window = "2025-02-18T15:00:00.000"
    stop_window = "2025-02-18T15:45:00.000"

    # replace colon with underscore
    start_window_str = start_window.replace(":", "_")
    stop_window_str = stop_window.replace(":", "_")

    # Load Cas A trajectory
    print("\nLoading Cas A trajectory...")
    cas_a_file = os.path.join(
        script_dir, "..", "tutorial", "data",
        f"casA_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
    )
    print(f"  Looking for Cas A file: {cas_a_file}")

    try:
        # Load trajectory with correct column names (matching reference file)
        traj_src = Trajectory.from_file(
            cas_a_file,
            time_tag='time_stamps',
            elevation_tag='altitudes',
            azimuth_tag='azimuths',
            distance_tag='distances'
        )
        print(f"  Cas A trajectory loaded: {len(traj_src.traj)} points")
        print(f"  Time range: {traj_src.traj['times'].min()} to {traj_src.traj['times'].max()}")
    except Exception as e:
        print(f"  ERROR: Could not load Cas A trajectory: {e}")
        print("  This is likely because the .arrow file doesn't exist or is corrupted")
        print("  Please ensure the file exists and is accessible")
        raise e

    # Define observation parameters
    cent_freq = 11.325e9  # Hz

    # start-end of observation (matching tuto_radiomdl_runtime_environment.py)
    dateformat = "%Y-%m-%dT%H:%M:%S.%f"
    start_obs = datetime.strptime("2025-02-18T15:30:00.000", dateformat)
    stop_obs = datetime.strptime("2025-02-18T15:40:00.000", dateformat)

    print("  Observation time window:")
    print(f"    Start: {start_obs}")
    print(f"    Stop: {stop_obs}")
    print(f"    Duration: {stop_obs - start_obs}")

    print("\nObservation Parameters:")
    print(f"  Start time: {start_obs}")
    print(f"  Stop time: {stop_obs}")
    print(f"  Duration: {stop_obs - start_obs}")
    print(f"  Center frequency: {cent_freq/1e9:.3f} GHz")

    # telescope receiver temperature (constant over the bandwidth): T_RX
    def T_RX(t, f):
        return 80.0  # 80 K noise temperature

    # Create instrument
    westford = Instrument(
        antenna=tel_ant,
        phy_temp=300.0,  # Physical temperature in Kelvin
        cent_freq=cent_freq,
        bw=1e3,  # 1 kHz bandwidth (matching tuto_radiomdl_runtime_environment.py)
        signal_func=T_RX,
        freq_chan=1,  # Single frequency channel
        coords=[westford_lat, westford_lon, westford_elevation]
    )

    # Create observation with environmental effects
    print("\nCreating observation with environmental effects...")

    # Apply offset for OFF-ON observation
    offset_angles = (-40, 0.)  # (az,el) in degrees
    time_off_src = start_obs
    time_on_src = time_off_src + timedelta(minutes=5)

    # Copy and modify trajectory
    traj_obj = Trajectory(traj_src.traj.copy())

    print("  Trajectory data analysis:")
    print(f"    Total points: {len(traj_obj.traj)}")
    print(f"    Time range: {traj_obj.traj['times'].min()} to {traj_obj.traj['times'].max()}")
    print(f"    Elevation range: {float(traj_obj.traj['elevations'].min()):.1f}° to "
          f"{float(traj_obj.traj['elevations'].max()):.1f}°")
    print(f"    Elevations > 5°: {(traj_obj.traj['elevations'] > 5.0).sum()}")

    # Verify trajectory data matches observation window
    traj_start = traj_obj.traj['times'].min()
    traj_end = traj_obj.traj['times'].max()

    print("  Time window verification:")
    print(f"    Observation window: {start_obs} to {stop_obs}")
    print(f"    Trajectory window: {traj_start} to {traj_end}")

    # Check if trajectory data matches observation window
    if traj_start > stop_obs or traj_end < start_obs:
        print("  ERROR: Trajectory data doesn't match observation window!")
        print("  This indicates the .arrow file is for a different time period")
        print("  Please ensure you're using the correct trajectory file")
        raise ValueError(
            f"Trajectory time range ({traj_start} to {traj_end}) "
            f"doesn't match observation window ({start_obs} to {stop_obs})"
        )

    # apply offset
    mask = (traj_obj.traj['times'] >= time_off_src) & (traj_obj.traj['times'] <= time_on_src)
    traj_obj.traj.loc[mask, 'azimuths'] += offset_angles[0]
    traj_obj.traj.loc[mask, 'elevations'] += offset_angles[1]

    print(f"    Points in observation window: {mask.sum()}")
    print(f"    Note: Filter will be applied to entire trajectory ({len(traj_obj.traj)} points)")

    # Filter points below minimum elevation (with environmental effects)
    def elevation_filter_with_env(elevation):
        """Enhanced elevation filter with environmental effects"""
        # Start with basic elevation filter (like the reference file)
        mask = elevation > 5.0

        print(f"  Basic elevation filter: {mask.sum()} out of {len(mask)} points pass")

        # Apply environmental effects check for each valid elevation
        # Only check environmental effects for elevations that pass the basic filter
        env_blocked = 0
        for i, elev in enumerate(elevation):
            if mask.iloc[i]:
                if not environment.check_elevation_masking(elev):
                    mask.iloc[i] = False
                    env_blocked += 1

        print(f"  Environmental effects blocked: {env_blocked} additional points")
        print(f"  Final filter result: {mask.sum()} out of {len(mask)} points pass")

        # If no points pass, fall back to basic elevation filter only
        if mask.sum() == 0:
            print("  Warning: No points pass environmental filter, using basic elevation filter only")
            return elevation > 5.0

        return mask

    # Create observation
    observ = Observation.from_dates(
        start_obs, stop_obs, traj_obj, westford,
        filt_funcs=(('elevations', elevation_filter_with_env),)
    )

    print(f"  Observation created: {len(observ.get_traj())} valid points")

    # Load Starlink satellite data
    print("\nLoading Starlink satellite trajectories...")
    # Use the same time window as the Cas A trajectory
    start_window_str = start_window.replace(":", "_")
    stop_window_str = stop_window.replace(":", "_")

    file_traj_sats_path = os.path.join(
        script_dir, "..", "tutorial", "data",
        f"Starlink_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
    )

    print(f"  Looking for Starlink data file: {file_traj_sats_path}")

    try:
        import pyarrow as pa
        with pa.memory_map(file_traj_sats_path, 'r') as source:
            table = pa.ipc.open_file(source).read_all()
        all_sat_data = table.to_pandas()

        # Apply column renaming and datetime conversion
        all_sat_data = all_sat_data.rename(columns={
            'timestamp': 'times',
            'sat': 'sat',
            'azimuths': 'azimuths',
            'elevations': 'elevations',
            'ranges_westford': 'distances'
        })
        all_sat_data['times'] = pd.to_datetime(all_sat_data['times'])

        print(f"  Starlink data loaded: {len(all_sat_data)} satellite positions")
        print(f"  Satellites: {all_sat_data['sat'].nunique()} unique satellites")
        print(f"  Time range: {all_sat_data['times'].min()} to {all_sat_data['times'].max()}")

        # Debug: Show satellite types
        sat_types = all_sat_data['sat'].str.contains('DTC').value_counts()
        print(f"  Satellite types: {sat_types.get(False, 0)} non-DTC, {sat_types.get(True, 0)} DTC")

        # Debug: Show elevation distribution
        print(f"  Elevation range: {float(all_sat_data['elevations'].min()):.1f}° to "
              f"{float(all_sat_data['elevations'].max()):.1f}°")
        print(f"  Satellites above 20°: {(all_sat_data['elevations'] > 20).sum()}")
        print(f"  Satellites above 5°: {(all_sat_data['elevations'] > 5).sum()}")

    except Exception as e:
        print(f"  WARNING: Could not load Starlink data: {e}")
        print("  Continuing without satellite interference data")
        all_sat_data = None

    # Create satellite constellation with environmental effects
    if all_sat_data is not None:
        print("\nCreating satellite constellation with environmental effects...")

        # Filter satellites for observation window and elevation (matching reference script)
        # Track statistics for environmental summary
        total_satellites_in_time_window = len(all_sat_data[
            (all_sat_data['times'] >= start_obs) &
            (all_sat_data['times'] <= stop_obs)
        ])

        elevation_blocked_count = len(all_sat_data[
            (all_sat_data['times'] >= start_obs) &
            (all_sat_data['times'] <= stop_obs) &
            (all_sat_data['elevations'] <= 20)  # Blocked by elevation filter
        ])

        obs_window_data = all_sat_data[
            (all_sat_data['times'] >= start_obs) &
            (all_sat_data['times'] <= stop_obs) &
            (all_sat_data['elevations'] > 20) &  # Satellite elevation filter (intentional design)
            (~all_sat_data['sat'].str.contains('DTC'))  # Exclude DTC satellites (matching reference script)
        ].copy()

        print(f"  Satellites in observation window: {len(obs_window_data)} positions")

        # Debug: Show filtering breakdown
        time_filtered = all_sat_data[
            (all_sat_data['times'] >= start_obs) &
            (all_sat_data['times'] <= stop_obs)
        ]
        print(f"    After time filter: {len(time_filtered)} positions")

        elevation_filtered = time_filtered[time_filtered['elevations'] > 20]
        print(f"    After elevation > 20°: {len(elevation_filtered)} positions")

        dtc_filtered = elevation_filtered[~elevation_filtered['sat'].str.contains('DTC')]
        print(f"    After DTC exclusion: {len(dtc_filtered)} positions")

        # Debug: Show elevation distribution of satellites
        if len(obs_window_data) > 0:
            print(f"    Satellite elevation range: {float(obs_window_data['elevations'].min()):.1f}° to "
                  f"{float(obs_window_data['elevations'].max()):.1f}°")
            print(f"    Satellites above 20°: {(obs_window_data['elevations'] > 20).sum()}")
        else:
            print("    No satellites found in observation window with elevation > 20°")

        # Apply environmental effects to satellite data
        # Initialize environmental statistics dictionary
        env_stats = {
            'terrain_blocked_count': 0,
            'antenna_limited_count': 0,
            'environmental_factor_mean': 1.0
        }

        if environmental_config.get('apply_terrain_masking', True):
            print("  Applying environmental effects to satellite data...")
            start_time = time.time()
            filtered_satellites = []
            masking_factors = []

            total_positions = len(obs_window_data)
            print(f"    Starting VECTORIZED processing of {total_positions} satellite positions...")

            # VECTORIZED APPROACH: Process all satellites at once
            # Extract all data as numpy arrays
            elevations = obs_window_data['elevations'].values
            azimuths = obs_window_data['azimuths'].values
            ranges = obs_window_data['distances'].values
            indices = obs_window_data.index.values

            print(f"    Processing {len(elevations)} positions with vectorized terrain masking...")

            # Apply vectorized terrain masking
            masking_factors = environment.apply_terrain_masking_vectorized(elevations, azimuths, ranges)

            # Calculate terrain blocking statistics
            env_stats['terrain_blocked_count'] = (masking_factors == 0.0).sum()
            env_stats['environmental_factor_mean'] = masking_factors.mean() if len(masking_factors) > 0 else 0.0

            # Vectorized filtering
            visible_mask = masking_factors > 0.5
            filtered_satellites = indices[visible_mask].tolist()

            processing_time = time.time() - start_time
            print(f"    Vectorized processing completed in {processing_time:.2f} seconds")
            print(f"    Environmental effects processing completed in {processing_time:.2f} seconds")
            print(f"    Processed {len(obs_window_data)} satellite positions")
            print(f"    Processing rate: {len(obs_window_data)/processing_time:.0f} positions/second")

            # Debug: Show masking factor statistics
            if len(masking_factors) > 0:
                print("    Masking factor statistics:")
                print(f"      Min: {float(masking_factors.min()):.3f}")
                print(f"      Max: {float(masking_factors.max()):.3f}")
                print(f"      Mean: {float(masking_factors.mean()):.3f}")
                print(f"      Satellites with factor > 0.5: {(masking_factors > 0.5).sum()}")
                print(f"      Satellites with factor > 0.1: {(masking_factors > 0.1).sum()}")
                print(f"      Satellites with factor > 0.0: {(masking_factors > 0.0).sum()}")
                print(f"      Terrain blocked (factor = 0.0): {env_stats['terrain_blocked_count']}")

            obs_window_data = obs_window_data.loc[filtered_satellites]
            print(f"  Satellites after environmental filtering: {len(obs_window_data)} positions")

            # If no satellites pass the 0.5 threshold, try a lower threshold
            if len(obs_window_data) == 0 and len(masking_factors) > 0:
                print("  No satellites passed 0.5 threshold, trying 0.1 threshold...")
                # Restore original data and try with lower threshold
                obs_window_data = all_sat_data[
                    (all_sat_data['times'] >= start_obs) &
                    (all_sat_data['times'] <= stop_obs) &
                    (all_sat_data['elevations'] > 20) &
                    (~all_sat_data['sat'].str.contains('DTC'))
                ].copy()

                # Time the fallback processing
                fallback_start_time = time.time()
                filtered_satellites = []
                for idx, sat_row in obs_window_data.iterrows():
                    alt = sat_row['elevations']
                    az = sat_row['azimuths']
                    rng = sat_row['distances']
                    masking_factor, atm_effects = environment.apply_terrain_masking(alt, az, rng)
                    if masking_factor > 0.1:  # Lower threshold
                        filtered_satellites.append(idx)

                fallback_end_time = time.time()
                fallback_processing_time = fallback_end_time - fallback_start_time

                obs_window_data = obs_window_data.loc[filtered_satellites]
                print(f"  Fallback processing completed in {fallback_processing_time:.2f} seconds")
                print(f"  Satellites after environmental filtering (0.1 threshold): {len(obs_window_data)} positions")

        # Create environmental summary right after environmental processing
        total_points = len(obs_window_data)
        total_satellites_processed = total_satellites_in_time_window
        visibility_percentage = ((total_points / total_satellites_processed * 100)
                                 if total_satellites_processed > 0 else 0.0)

        # Initialize variables that will be defined later (with default values)
        refraction_summary = {
            'min_refraction_correction': 0.0,
            'avg_refraction_correction': 0.0,
            'refraction_model_used': 'none'
        }  # Will be updated later

        environmental_summary = {
            'total_points': total_satellites_processed,  # Total satellites in time window
            'visible_points': total_points,  # Satellites that passed all filters
            'blocked_points': total_satellites_processed - total_points,  # Total blocked satellites
            'terrain_blocked_points': env_stats['terrain_blocked_count'],
            'elevation_blocked_points': elevation_blocked_count,
            'antenna_limited_points': env_stats['antenna_limited_count'],
            'visibility_percentage': visibility_percentage,
            'atmospheric_effects_applied': total_points,  # All visible points have atmospheric effects
            'environmental_factor': env_stats['environmental_factor_mean'],
            'atmospheric_refraction_applied': True,
            'refraction_summary': refraction_summary
        }

        # Using vectorized observation modeling functions from obs_mdl.py
        print("  Using vectorized observation modeling functions from obs_mdl.py...")

        # Create satellite antenna and transmitter
        sat_eta_rad = 0.5
        sat_gain_max = 39.3  # dBi
        half_beamwidth = 3.0  # degrees

        alphas = np.arange(0, 181)
        betas = np.arange(0, 351, 10)
        gain_pat = antenna_mdl_ITU(sat_gain_max, half_beamwidth, alphas, betas)
        sat_ant = Antenna.from_dataframe(gain_pat, sat_eta_rad, freq_band)

        # Satellite transmitter parameters
        sat_T_phy = 0.0  # K
        sat_freq = 11.325e9  # Hz
        sat_bw = 250e6  # Hz
        transmit_pow = -15 + 10 * np.log10(300)  # dBW

        def transmit_temp(tim, freq):
            return power_to_temperature(10**(transmit_pow/10), 1.0)

        sat_transmit = Instrument(sat_ant, sat_T_phy, sat_freq, sat_bw, transmit_temp, 1, [])

        # Create custom link budget function with environmental effects (following reference script workflow)
        print("\n🔧 CREATING CUSTOM LINK BUDGET FUNCTION WITH ENVIRONMENTAL EFFECTS")

        def lnk_bdgt_with_environmental_effects_vectorized(*args, **kwargs):
            """
            VECTORIZED version of the custom link budget function with environmental effects.

            This function optimizes the environmental effects calculation by:
            - Vectorizing environmental factor calculations
            - Minimizing function calls
            - Using numpy operations where possible
            """
            # Extract arguments
            dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat, instru_sat, freq = args[:8]

            # Set beam avoidance parameters to accept custom link budget at model_observed_temp...
            kwargs['beam_avoid'] = 1e-20
            kwargs['turn_off'] = False

            # VECTORIZED ENVIRONMENTAL EFFECTS CALCULATION
            if environmental_config is not None:
                # Convert to degrees for environmental calculations
                alt_deg = 90 - np.degrees(dec_sat)
                az_deg = -np.degrees(caz_sat)

                # Vectorized environmental factor calculation
                env_factors = calculate_comprehensive_environmental_effects_vectorized(
                    alt_deg, az_deg, rng_sat, freq, environment
                )

                # If satellite is completely blocked, return zero
                if env_factors['total_factor'] == 0.0:
                    return 0.0
            else:
                env_factors = {'total_factor': 1.0}

            # Use standard link budget calculation (no Doppler or Transmitter effects)
            result = sat_link_budget_vectorized(*args, **kwargs)

            # Apply environmental factor
            return result * env_factors['total_factor']

        # Create constellation with custom link budget function (includes environmental effects only)
        print("    Creating constellation with custom link budget function...")
        print("    Custom function includes: environmental effects only")
        constellation = Constellation.from_observation(
            sats=obs_window_data,
            observation=observ,
            sat_tmt=sat_transmit,
            lnk_bdgt_mdl=lnk_bdgt_with_environmental_effects_vectorized  # ← VECTORIZED function with env effects
        )

        print(f"  Constellation created with {len(obs_window_data)} satellite positions")
        print("  Custom link budget function includes:")
        print(f"    • Environmental effects: {'Yes' if environmental_config is not None else 'No'}")
        print("      - Terrain masking: Line-of-sight obstructions")
        print("      - Limb refraction: Signal bending at low elevations")
        print("      - Atmospheric absorption: Frequency-dependent attenuation")
        print("      - Water vapor effects: Additional absorption")
        print("    • Standard link budget calculation (no Doppler or transmitter effects)")
        print("    • Environmental effects integrated in link budget function")

    # Define sky model
    print("\nDefining sky model...")

    # Cas A flux
    flux_src = estim_casA_flux(cent_freq)
    print(f"  Cas A flux at {cent_freq/1e9:.3f} GHz: {flux_src:.1f} Jy")

    # Pre-calculate effective aperture
    max_gain = tel_ant.get_boresight_gain()
    A_eff_max = antenna_pattern.gain_to_effective_aperture(max_gain, cent_freq)

    # Source temperature function
    def T_src(t):
        if t <= time_on_src:
            return 0.0
        else:
            return estim_temp(flux_src, A_eff_max)

    # Atmospheric and background models
    T_atm_zenith = 150  # K
    tau = 0.05
    def T_atm(dec): return T_atm_zenith * (1 - np.exp(-tau/np.cos(dec)))

    T_CMB = 2.73  # K
    def T_gal(freq): return 1e-1 * (freq/1.41e9)**(-2.7)
    def T_bkg(freq): return T_CMB + T_gal(freq)

    # Total sky model
    def sky_mdl(dec, caz, tim, freq):
        return T_src(tim) + T_atm(dec) + T_bkg(freq)

    print("  Sky model defined with Cas A, atmosphere, and background")

    # Run observation modeling with environmental effects
    print("\n" + "=" * 80)
    print("RUNNING OBSERVATION MODELING WITH ENVIRONMENTAL EFFECTS")
    print("=" * 80)

    start_modeling_time = time.time()

    # Use standard observation modeling (environmental effects are integrated in the link budget function)
    print("    Running observation modeling with integrated environmental effects...")
    print("    Environmental effects are handled by the custom link budget function")

    # Use observation modeling with atmospheric refraction correction (Category 2 effects)
    print("    Using observation modeling with atmospheric refraction correction...")
    print("    This includes Category 2 effects: Telescope pointing correction for atmospheric refraction")

    # Define atmospheric refraction configuration
    atmospheric_refraction_config = {
        'temperature': 288.15,  # K
        'pressure': 101325,     # Pa
        'humidity': 50.0,       # %
        'apply_refraction_correction': True,
        'refraction_model': 'standard'  # 'standard' or 'advanced'
    }

    try:
        # Use atmospheric refraction correction function (Category 2 effects)
        # Set beam avoidance parameters to accept custom link budget model_observed_temp
        print("    Using VECTORIZED observation modeling...")
        result, refraction_summary = model_observed_temp_with_atmospheric_refraction_vectorized(
            observ, sky_mdl, constellation=constellation, beam_avoidance=True,
            atmospheric_refraction=atmospheric_refraction_config
        )

        # Update environmental summary with computed refraction summary
        environmental_summary['refraction_summary'] = refraction_summary

        modeling_time = time.time() - start_modeling_time

        print("    VECTORIZED observation modeling with atmospheric refraction completed successfully!")
        print(f"    VECTORIZED modeling completed in {modeling_time:.2f} seconds")
        print(f"    Observed temperature range: {float(result.min()):.3f} - {float(result.max()):.3f} K")
        print("    Atmospheric refraction corrections applied:")
        print(f"      • Refraction corrections applied: {refraction_summary['refraction_corrections_applied']}")
        print(f"      • Max refraction correction: {float(refraction_summary['max_refraction_correction']):.3f}°")
        print(f"      • Min refraction correction: {float(refraction_summary['min_refraction_correction']):.3f}°")
        print(f"      • Avg refraction correction: {float(refraction_summary['avg_refraction_correction']):.3f}°")
        print(f"      • Refraction model used: {refraction_summary['refraction_model_used']}")
        print("    All effects integrated:")
        print("      • Category 1 (Link Budget): Terrain masking, limb refraction, "
              "atmospheric absorption, water vapor")
        print("      • Category 2 (Observation): Atmospheric refraction correction for telescope pointing")

        # Handle case when no satellites are found
        if total_points == 0:
            print("    WARNING: No satellites found in observation window!")
            print("    This may be due to:")
            print("      • No satellites in the specified time range")
            print("      • All satellites below 20° elevation threshold")
            print("      • All satellites filtered out by environmental effects")
            print("    Proceeding with empty constellation...")

        # Calculate baseline result without environmental effects for comparison
        print("\n📊 CALCULATING BASELINE RESULT (NO ENVIRONMENTAL EFFECTS)")
        print("=" * 60)

        # Create a baseline constellation without environmental effects
        print("    Creating baseline constellation without environmental effects...")

        # Use the original link budget function (no environmental effects)
        def lnk_bdgt_baseline(*args, **kwargs):
            """Baseline link budget function without environmental effects"""
            # Remove any environmental parameters
            kwargs.pop('beam_avoid', None)
            kwargs.pop('turn_off', None)
            return sat_link_budget_vectorized(*args, **kwargs)

        # Define filter for baseline observation
        filt_el_observ_original = ('elevations', lambda e: e > 5)

        # Define observ_original to avoid overwriting the main observ object
        observ_original = Observation.from_dates(
            start_obs, stop_obs, traj_obj, westford,
            filt_funcs=(filt_el_observ_original,)
        )

        # Define filters for baseline constellation (same as main constellation)
        filt_name = ('sat', lambda s: ~s.str.contains('DTC'))
        filt_el_constellation_original = ('elevations', lambda e: e > 20)

        # Create baseline constellation
        baseline_constellation = Constellation.from_file(
            file_traj_sats_path, observ_original, sat_transmit, lnk_bdgt_baseline,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el_constellation_original)
        )

        # Calculate baseline result
        print("    Computing baseline result (no environmental effects)...")
        start_baseline_time = time.time()

        # Import model_observed_temp for baseline calculation
        from obs_mdl import model_observed_temp  # noqa: E402

        # CRITICAL: Use .copy() to avoid modifying the same result array
        # When we call model_observed_temp multiple times with the same "observ" object,
        # it modifies the same internal result array, causing the main result variable to be overwritten.
        # .copy() creates a copy of the result array to avoid this issue.
        result_original = model_observed_temp(observ_original, sky_mdl, baseline_constellation).copy()

        baseline_time = time.time() - start_baseline_time
        print(f"    Baseline computation completed in {baseline_time:.2f} seconds")
        print(f"    Baseline temperature range: {float(result_original.min()):.3f} - "
              f"{float(result_original.max()):.3f} K")

        # Calculate comparison statistics
        print("\n📊 ENVIRONMENTAL EFFECTS IMPACT ANALYSIS")
        print("=" * 50)

        def safe_log10(x):
            """Safe log10 function that handles zero and negative values"""
            x = np.array(x)
            x = np.where(x > 0, x, np.nan)
            return np.log10(x)

        # Calculate power differences
        power_with_effects = temperature_to_power(result[:, 0, 0], westford.bw)
        power_baseline = temperature_to_power(result_original[:, 0, 0], westford.bw)

        # Calculate statistics
        power_diff_db = 10 * (safe_log10(power_with_effects) - safe_log10(power_baseline))
        max_impact_db = float(np.nanmax(power_diff_db))
        min_impact_db = float(np.nanmin(power_diff_db))
        mean_impact_db = float(np.nanmean(power_diff_db))

        print("    Power impact analysis:")
        print(f"      • Maximum power reduction: {max_impact_db:.2f} dB")
        print(f"      • Minimum power reduction: {min_impact_db:.2f} dB")
        print(f"      • Average power reduction: {mean_impact_db:.2f} dB")

        # Calculate temperature differences
        temp_diff = result[:, 0, 0] - result_original[:, 0, 0]
        max_temp_reduction = float(np.max(temp_diff))
        min_temp_reduction = float(np.min(temp_diff))
        mean_temp_reduction = float(np.mean(temp_diff))

        print("    Temperature impact analysis:")
        print(f"      • Maximum temperature reduction: {max_temp_reduction:.3f} K")
        print(f"      • Minimum temperature reduction: {min_temp_reduction:.3f} K")
        print(f"      • Average temperature reduction: {mean_temp_reduction:.3f} K")

        # Calculate percentage of time with significant impact (>1 dB)
        significant_impact_points = np.sum(np.abs(power_diff_db) > 1.0)
        total_points = len(power_diff_db)
        significant_impact_percentage = (significant_impact_points / total_points) * 100

        print("    Impact significance:")
        print(f"      • Points with >1 dB impact: {significant_impact_points}/{total_points} "
              f"({significant_impact_percentage:.1f}%)")
        print(f"      • Environmental effects are {'significant' if significant_impact_percentage > 10 else 'minimal'}")

        # Create time vs power plot (following reference script)
        print("\n📊 CREATING TIME VS POWER COMPARISON PLOT")
        print("=" * 50)

        # Check if we have data to plot
        has_result = 'result' in locals() and result is not None
        has_result_original = 'result_original' in locals() and result_original is not None
        if total_points > 0 and has_result and has_result_original:
            # Create the comparison plot
            fig, ax = plt.subplots(figsize=(18, 6))
            time_samples = observ.get_time_stamps()

            # Convert temperature to power and plot both results
            plot_result = temperature_to_power(result[:, 0, 0], westford.bw)
            plot_result_original = temperature_to_power(result_original[:, 0, 0], westford.bw)

            ax.plot(time_samples, 10 * safe_log10(plot_result_original),
                    label="without effects (baseline)", linewidth=2)
            ax.plot(time_samples, 10 * safe_log10(plot_result),
                    label="with effects", linewidth=2)

            # Add plot formatting
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Power [dBW]")
            ax.grid(True)
            ax.set_title("Observed Power: Environmental Effects")
            fig.tight_layout()

            # Save the plot
            plot_filename = "07_environmental_effects.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"    Plot saved as: {plot_filename}")

            # Show the plot
            plt.show()
        else:
            print("    Skipping plot creation: No satellite data available")
            print("    This is expected when no satellites are found in the observation window")

    except Exception as e:
        print(f"    ERROR in vectorized observation modeling: {str(e)}")
        print("    The model_observed_temp_with_atmospheric_refraction_vectorized function failed.")
        print("    This indicates an issue with the vectorized implementation that needs to be debugged.")
        print("    No fallback will be attempted - the error will be re-raised.")

        # Re-raise the exception to stop execution
        raise e

    # Display results
    print("\n" + "=" * 80)
    print("ENVIRONMENTAL EFFECTS ANALYSIS RESULTS")
    print("=" * 80)

    print("\nObservation Summary:")
    print(f"  Total observation points: {environmental_summary['total_points']}")
    visible_points = (environmental_summary['total_points'] -
                      environmental_summary['terrain_blocked_points'] -
                      environmental_summary['elevation_blocked_points'] -
                      environmental_summary['antenna_limited_points'])
    print(f"  Visible points: {visible_points}")
    print(f"  Visibility percentage: {environmental_summary['visibility_percentage']:.1f}%")

    print("\nBlocking Analysis:")
    terrain_pct = (environmental_summary['terrain_blocked_points'] /
                   environmental_summary['total_points'] * 100)
    elevation_pct = (environmental_summary['elevation_blocked_points'] /
                     environmental_summary['total_points'] * 100)
    antenna_pct = (environmental_summary['antenna_limited_points'] /
                   environmental_summary['total_points'] * 100)

    print(f"  Terrain blocked: {environmental_summary['terrain_blocked_points']} points ({terrain_pct:.1f}%)")
    print(f"  Elevation blocked: {environmental_summary['elevation_blocked_points']} points ({elevation_pct:.1f}%)")
    print(f"  Antenna limited: {environmental_summary['antenna_limited_points']} points ({antenna_pct:.1f}%)")
    print(f"  Atmospheric effects applied: {environmental_summary['atmospheric_effects_applied']} points")

    # Temperature analysis
    print("\nTemperature Analysis:")
    print(f"  Mean system temperature: {np.mean(result):.1f} K")
    print(f"  Max system temperature: {np.max(result):.1f} K")
    print(f"  Min system temperature: {np.min(result):.1f} K")
    print(f"  Temperature range: {np.max(result) - np.min(result):.1f} K")

    print("\n" + "=" * 80)
    print("PRACTICAL APPLICATION COMPLETE")
    print("=" * 80)

    print("\nKey Achievements:")
    print("  ✓ Integrated environmental effects with real Westford antenna data")
    print("  ✓ Applied terrain masking using DEM data for the Westford site")
    print("  ✓ Incorporated atmospheric refraction and water vapor effects")
    print("  ✓ Demonstrated comprehensive link budget calculations")
    print("  ✓ Provided detailed blocking analysis and visibility statistics")
    print("  ✓ Enabled realistic radio astronomy observation modeling")

    return {
        'observation': observ,
        'result': result,
        'environmental_summary': environmental_summary,
        'environmental_config': environmental_config,
        'modeling_time': modeling_time
    }


def main():
    """Main function"""
    print("Starting advanced environmental effects demonstration...")
    try:
        # Part 1: General tutorial demonstration (uses demo data)
        print("\n" + "=" * 80)
        print("PART 1: GENERAL TUTORIAL DEMONSTRATION")
        print("(Uses demo data and trajectories)")
        print("=" * 80)

        demonstrate_advanced_environmental_effects()

        # Part 2: Practical application with real data (uses .arrow files)
        print("\n" + "=" * 80)
        print("PART 2: PRACTICAL APPLICATION WITH REAL DATA")
        print("(Uses real .arrow files for 2025-02-18 time window)")
        print("=" * 80)

        practical_environmental_effects_with_real_data()

        print("\n" + "=" * 80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
