"""
Multi-satellite trajectory computation script for radio astronomy interference studies.

This script can handle multiple satellite types (Starlink, JPSS/NOAA, etc.) and
generates trajectory data for radio astronomy interference modeling.

Note: change t0 and t1 for the time range of interest

"""

import pyproj
from skyfield.api import load, wgs84, EarthSatellite
from pathlib import Path
from urllib.request import urlopen
from sgp4 import omm
from sgp4.api import Satrec
from datetime import timedelta
import pandas as pd
import time
import numpy as np


def safe_time_str(dt):
    """Convert datetime to safe string format for filenames."""
    return dt.strftime('%Y-%m-%dT%H_%M_%S.%f')[:-3]  # up to milliseconds


def validate_trajectory_data(elevations, azimuths, distances):
    """
    Validate and clean trajectory data using pure NumPy operations.

    Args:
        elevations: Array of elevation angles
        azimuths: Array of azimuth angles
        distances: Array of distances

    Returns:
        tuple: (elevations_array, azimuths_array, distances_array)
    """
    # Convert to numpy arrays
    elev_array = np.asarray(elevations, dtype=np.float64)
    azim_array = np.asarray(azimuths, dtype=np.float64)
    dist_array = np.asarray(distances, dtype=np.float64)

    # Use NumPy's vectorized operations for bounds checking (much faster than loops)
    # Clamp elevation to valid range [-90, 90]
    elev_array = np.clip(elev_array, -90, 90)

    # Normalize azimuth to 0-360 range
    azim_array = np.mod(azim_array, 360)

    # Ensure distance is positive
    dist_array = np.maximum(dist_array, 0)

    return elev_array, azim_array, dist_array


def compute_satellite_trajectories(satellites, observer, t0, t1, time_step, time_round, fmt):
    """
    Compute trajectories using pure NumPy operations for maximum performance.

    This version uses:
    1. Pure NumPy arrays for all data processing
    2. Vectorized operations throughout (no loops)
    3. Minimal Python object creation during computation
    4. DataFrame creation only at the end

    Args:
        satellites (list): List of EarthSatellite objects
        observer: Skyfield observer object (e.g., westford)
        t0: Start time
        t1: End time
        time_step: Time step for trajectory computation
        time_round: Time rounding frequency
        fmt: Time format string

    Returns:
        pandas.DataFrame: DataFrame with columns ['timestamp', 'sat', 'elevations', 'azimuths', 'ranges_westford']
    """
    total_sats = len(satellites)

    # Pre-allocate lists to collect all data
    all_timestamps = []
    all_sat_names = []
    all_elevations = []
    all_azimuths = []
    all_distances = []

    for sat_idx, sat in enumerate(satellites):
        # Progress tracking
        if sat_idx % 1000 == 0 or sat_idx == total_sats - 1:
            print(f"  Processing satellite {sat_idx + 1}/{total_sats}: {sat.name}")

        # Distance from satellite to observer over time
        diff_obs = sat - observer

        # Find satellite visibility events (rise, culmination, set)
        t_obs, events_obs = sat.find_events(observer, t0, t1, altitude_degrees=5.0)

        # Process each visibility period
        for ind_t_obs in range(0, len(t_obs) - 2, 3):
            # Time of rise
            t_obs_r = t_obs[ind_t_obs]
            dt_obs_r = t_obs_r.utc_datetime()

            # Time of set
            t_obs_s = t_obs[ind_t_obs + 2]
            dt_obs_s = t_obs_s.utc_datetime()

            # Get the beginning of the sample in sync with rounded time_step
            dt_beg_sync = pd.Timestamp(dt_obs_r).round(freq=time_round).to_pydatetime()

            # Temporal grid of satellite
            rise_to_set_range = pd.date_range(
                dt_beg_sync, dt_obs_s, freq=time_step, tz='UTC'
            )

            if len(rise_to_set_range) == 0:
                continue

            # PURE NUMPY VECTORIZED COMPUTATION
            # Convert pandas datetime range to skyfield times
            skyfield_times = ts.from_datetimes(rise_to_set_range.to_pydatetime())

            # Compute positions for all time points at once
            diff_t_rs = diff_obs.at(skyfield_times)

            # Convert to angles and distance for all time points
            ang_t_rs = diff_t_rs.altaz()

            # Extract arrays of values and convert to numpy arrays immediately
            elev_array = np.asarray(ang_t_rs[0].degrees, dtype=np.float64)
            azim_array = np.asarray(ang_t_rs[1].degrees, dtype=np.float64)
            dist_array = np.asarray(ang_t_rs[2].m, dtype=np.float64)

            # Validate and clean the data using pure NumPy operations
            elev_array, azim_array, dist_array = validate_trajectory_data(
                elev_array, azim_array, dist_array
            )

            # Convert timestamps to strings efficiently using vectorized operations
            n_points = len(rise_to_set_range)
            timestamp_strings = [dt.strftime(fmt)[:-3] for dt in rise_to_set_range]

            # Extend lists with numpy arrays (very fast)
            all_timestamps.extend(timestamp_strings)
            all_sat_names.extend([sat.name] * n_points)
            all_elevations.extend(elev_array)
            all_azimuths.extend(azim_array)
            all_distances.extend(dist_array)

    # Create DataFrame only at the end - this is much faster
    print(f"  Creating DataFrame with {len(all_timestamps)} trajectory points...")
    df = pd.DataFrame({
        'timestamp': all_timestamps,
        'sat': all_sat_names,
        'elevations': all_elevations,
        'azimuths': all_azimuths,
        'ranges_westford': all_distances
    })

    return df


def load_satellites_from_url(url, satellite_type, output_dir="traj_files"):
    """
    Load satellites from a Celestrak URL.

    Args:
        url (str): Celestrak URL for satellite data
        satellite_type (str): Type of satellites (e.g., 'starlink', 'noaa', 'jpss')
        output_dir (str): Directory to save CSV files

    Returns:
        list: List of EarthSatellite objects
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    # Download satellite data
    filename = f"{satellite_type}_active.csv"
    filepath = Path(output_dir) / filename

    try:
        data = urlopen(url)
        filepath.write_bytes(data.read())
        print(f"Downloaded {satellite_type} data to {filepath}")
    except Exception as e:
        print(f"Error downloading {satellite_type} data: {e}")
        return []

    # Parse satellite data
    satellites = []
    try:
        with open(filepath) as f:
            for fields in omm.parse_csv(f):
                sat = Satrec()
                omm.initialize(sat, fields)
                e = EarthSatellite.from_satrec(sat, ts)
                e.name = fields.get('OBJECT_NAME')

                # Filter for JPSS satellites if satellite_type is 'jpss'
                if satellite_type == 'jpss':
                    # Only include the three specific JPSS satellites
                    jpss_names = ['SUOMI NPP', 'NOAA 20 (JPSS-1)', 'NOAA 21 (JPSS-2)']
                    if e.name not in jpss_names:
                        continue

                satellites.append(e)
    except Exception as e:
        print(f"Error parsing {satellite_type} data: {e}")
        return []

    print(f'Loaded {len(satellites)} {satellite_type} satellites')
    return satellites


def main():
    """Main function to compute satellite trajectories."""

    # Start overall timing
    start_time = time.time()
    print("="*60)
    print("SATELLITE TRAJECTORY COMPUTATION")
    print("="*60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # =============================================================================
    # TELESCOPE POSITION
    # =============================================================================

    # Westford coordinates
    WESTFORD_X = 1492206.5970
    WESTFORD_Y = -4458130.5170
    WESTFORD_Z = 4296015.5320
    # Observed offset from WGS84 ellipsoid location and Westford STK file
    # Not sure which is really right. Need to check with Chet / Diman
    WESTFORD_Z_OFFSET = 0.1582435

    transformer = pyproj.Transformer.from_crs(
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
    )
    lon1, lat1, alt1 = transformer.transform(
        WESTFORD_X, WESTFORD_Y, WESTFORD_Z, radians=False
    )
    print(f"Westford coordinates: {lat1}, {lon1}, {alt1}")

    WESTFORD_LAT = lat1
    WESTFORD_LON = lon1
    WESTFORD_ALT = alt1 + WESTFORD_Z_OFFSET

    # Observer location
    westford = wgs84.latlon(WESTFORD_LAT, WESTFORD_LON, WESTFORD_ALT)

    # =============================================================================
    # SATELLITE CONFIGURATION
    # =============================================================================

    # Define satellite types to load
    # You can modify this list to include/exclude different satellite types
    satellite_configs = {
        'Starlink': {
            'url': "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=csv",
            'enabled': True,  # Set to False to disable
            'description': 'Starlink constellation'
        },
        'jpss': {
            'url': "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=csv",
            'enabled': False,  # Set to False to disable
            'description': 'JPSS weather satellites (SUOMI NPP, NOAA 20, NOAA 21 only)'
        }
    }

    # =============================================================================
    # LOAD SATELLITES
    # =============================================================================

    # Time scale
    global ts
    ts = load.timescale()

    # Load all enabled satellite types
    all_satellites = []
    satellite_type_counts = {}

    for sat_type, config in satellite_configs.items():
        if config['enabled']:
            print(f"\nLoading {config['description']}...")
            # Note: 'jpss' type automatically filters to only the 3 JPSS satellites
            satellites = load_satellites_from_url(config['url'], sat_type)
            all_satellites.extend(satellites)
            satellite_type_counts[sat_type] = len(satellites)
        else:
            print(f"Skipping {sat_type} (disabled)")
            satellite_type_counts[sat_type] = 0

    print(f"\nTotal satellites loaded: {len(all_satellites)}")
    for sat_type, count in satellite_type_counts.items():
        if count > 0:
            print(f"  - {sat_type}: {count} satellites")

    if not all_satellites:
        print("No satellites loaded. Exiting.")
        return

    # =============================================================================
    # OBSERVATION PARAMETERS
    # =============================================================================

    # Time of observation
    fmt = '%Y-%m-%dT%H:%M:%S.%f'
    t0 = ts.utc(2025, 4, 1, 12, 30, 00)
    t1 = ts.utc(2025, 4, 1, 13, 30, 00)
    dt0 = t0.utc_datetime()
    dt1 = t1.utc_datetime()

    # Time resolution of trajectories
    time_step = timedelta(milliseconds=1000)
    time_round = '1000ms'

    print(f"\nObservation period: {dt0} to {dt1}")
    print(f"Time resolution: {time_step}")

    # =============================================================================
    # COMPUTE TRAJECTORIES
    # =============================================================================

    print("Computing satellite trajectories...")
    print(f"Processing {len(all_satellites)} satellites...")

    # Compute trajectories - now returns DataFrame directly
    trajectory_df = compute_satellite_trajectories(
        all_satellites, westford, t0, t1, time_step, time_round, fmt
    )
    print(f"Computed {len(trajectory_df)} trajectory points")

    # =============================================================================
    # SAVE RESULTS
    # =============================================================================

    # Sort by time (DataFrame is already created)
    trajectory_df = trajectory_df.sort_values('timestamp')

    # Generate filename
    start_str = safe_time_str(dt0)
    end_str = safe_time_str(dt1)

    # Create descriptive filename based on enabled satellite types
    enabled_types = [sat_type for sat_type, config in satellite_configs.items()
                     if config['enabled']]
    sat_types_str = "_".join(enabled_types)
    filename = (f"traj_files/{sat_types_str}_trajectory_Westford_"
                f"{start_str}_{end_str}.arrow")

    # Save to file
    trajectory_df.to_feather(filename)
    print(f"Trajectory data saved to: {filename}")

    # =============================================================================
    # SUMMARY STATISTICS
    # =============================================================================

    print("\n" + "="*60)
    print("TRAJECTORY COMPUTATION SUMMARY")
    print("="*60)
    print(f"Observation period: {dt0} to {dt1}")
    print(f"Total trajectory points: {len(trajectory_df)}")
    print(f"Unique satellites: {trajectory_df['sat'].nunique()}")

    # Statistics by satellite type
    print("\nSatellites by type:")
    for sat_type, count in satellite_type_counts.items():
        if count > 0:
            # Count satellites of this type that appear in trajectories
            sat_names = trajectory_df['sat'].unique()
            type_sats = [name for name in sat_names
                         if sat_type.upper() in name.upper()]
            print(f"  - {sat_type}: {len(type_sats)} satellites with trajectories")

    # Elevation statistics
    print(f"\nElevation range: {trajectory_df['elevations'].min():.1f}° to "
          f"{trajectory_df['elevations'].max():.1f}°")
    print(f"Mean elevation: {trajectory_df['elevations'].mean():.1f}°")

    # Distance statistics
    print(f"\nDistance range: {trajectory_df['ranges_westford'].min()/1000:.1f} km to "
          f"{trajectory_df['ranges_westford'].max()/1000:.1f} km")
    print(f"Mean distance: {trajectory_df['ranges_westford'].mean()/1000:.1f} km")

    # =============================================================================
    # EXECUTION TIME SUMMARY
    # =============================================================================

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("EXECUTION TIME SUMMARY")
    print("="*60)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTrajectory computation completed successfully!")


if __name__ == "__main__":
    main()
