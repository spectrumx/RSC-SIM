"""
Weather Satellite Modeling Functions for "Looking Down" RFI Analysis

This module provides functions for modeling RFI from the perspective of a weather
satellite (e.g., Suomi-NPP) looking down at Earth, including:
- Coordinate transformations to weather satellite frame
- Weather satellite antenna pattern loading from CSV
- Link budget calculations for Starlink backlobe interference
- Observation modeling for weather satellite scenarios
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Import existing modules without modifying them
from radio_types import Antenna, Instrument, Trajectory, Constellation
from astro_mdl import power_to_temperature, temperature_to_power
from sat_mdl import free_space_loss, simple_link_budget

# Constants
speed_c = 3e8  # m/s
k_boltz = 1.380649e-23  # J/K
R_earth = 6378137.0  # Earth radius in meters (WGS84)
rad = np.pi / 180.0  # degree to radian conversion


# =============================================================================
# Coordinate Transformation Functions
# =============================================================================

def ecef_to_weather_sat_frame(
    target_ecef: np.ndarray,
    weather_sat_ecef: np.ndarray,
    weather_sat_velocity_ecef: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform target position from ECEF to weather satellite body frame.

    Weather satellite frame:
    - X (nadir): Points toward Earth center
    - Y (along-track): Points in velocity direction
    - Z (cross-track): Completes right-handed system

    Args:
        target_ecef: Target position in ECEF [x, y, z] (meters) or (N, 3) array
        weather_sat_ecef: Weather satellite position in ECEF [x, y, z] (meters) or (N, 3) array
        weather_sat_velocity_ecef: Weather satellite velocity in ECEF [vx, vy, vz] (m/s) or (N, 3) array

    Returns:
        tuple: (dec, caz) in radians
            - dec: Declination angle from nadir (0 = nadir, π/2 = horizon)
            - caz: Counter-azimuth angle (0 = along-track, counter-clockwise)
    """
    # Ensure arrays are at least 2D and have correct dtype
    target_ecef = np.asarray(target_ecef, dtype=np.float64)
    weather_sat_ecef = np.asarray(weather_sat_ecef, dtype=np.float64)
    weather_sat_velocity_ecef = np.asarray(weather_sat_velocity_ecef, dtype=np.float64)

    # Reshape to ensure 2D: (N, 3) shape
    if target_ecef.ndim == 0:
        raise ValueError("target_ecef cannot be a scalar")
    if target_ecef.ndim == 1:
        target_ecef = target_ecef.reshape(1, -1)
    if weather_sat_ecef.ndim == 0:
        raise ValueError("weather_sat_ecef cannot be a scalar")
    if weather_sat_ecef.ndim == 1:
        weather_sat_ecef = weather_sat_ecef.reshape(1, -1)
    if weather_sat_velocity_ecef.ndim == 0:
        raise ValueError("weather_sat_velocity_ecef cannot be a scalar")
    if weather_sat_velocity_ecef.ndim == 1:
        weather_sat_velocity_ecef = weather_sat_velocity_ecef.reshape(1, -1)

    # Ensure all arrays have compatible shapes for broadcasting
    # Broadcast to same shape if needed (use tile instead of broadcast_to to get writable arrays)
    max_len = max(target_ecef.shape[0], weather_sat_ecef.shape[0], weather_sat_velocity_ecef.shape[0])
    if target_ecef.shape[0] == 1 and max_len > 1:
        target_ecef = np.tile(target_ecef, (max_len, 1))
    if weather_sat_ecef.shape[0] == 1 and max_len > 1:
        weather_sat_ecef = np.tile(weather_sat_ecef, (max_len, 1))
    if weather_sat_velocity_ecef.shape[0] == 1 and max_len > 1:
        weather_sat_velocity_ecef = np.tile(weather_sat_velocity_ecef, (max_len, 1))

    # Vector from weather satellite to target
    r_vec = target_ecef - weather_sat_ecef
    # Ensure r_vec is 2D and has correct shape
    if r_vec.ndim == 1:
        r_vec = r_vec.reshape(1, -1)
    elif r_vec.ndim == 0:
        raise ValueError("r_vec cannot be a scalar")

    # Ensure r_vec is contiguous and writable
    r_vec = np.ascontiguousarray(r_vec)
    r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r_vec_norm = r_vec / r

    # X-axis: nadir (toward Earth center)
    x_axis = -weather_sat_ecef / np.linalg.norm(weather_sat_ecef, axis=-1, keepdims=True)

    # Y-axis: along-track (velocity direction)
    v_norm = np.linalg.norm(weather_sat_velocity_ecef, axis=-1, keepdims=True)
    if np.any(v_norm < 1e-6):
        # Fallback: use cross product if velocity is too small
        z_temp = np.cross(weather_sat_ecef, np.array([0, 0, 1]))
        z_temp_norm = z_temp / (np.linalg.norm(z_temp, axis=-1, keepdims=True) + 1e-10)
        y_axis = np.cross(z_temp_norm, x_axis)
    else:
        y_axis = weather_sat_velocity_ecef / v_norm

    # Z-axis: cross-track (right-handed system)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis, axis=-1, keepdims=True) + 1e-10)

    # Re-normalize y-axis to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis, axis=-1, keepdims=True) + 1e-10)

    # Transform target vector to weather satellite frame
    # Project onto x, y, z axes
    x_proj = np.sum(r_vec_norm * x_axis, axis=-1, keepdims=True)
    y_proj = np.sum(r_vec_norm * y_axis, axis=-1, keepdims=True)
    z_proj = np.sum(r_vec_norm * z_axis, axis=-1, keepdims=True)

    # Convert to spherical coordinates in weather satellite frame
    # dec: angle from nadir (x-axis)
    dec = np.arccos(np.clip(x_proj.flatten(), -1.0, 1.0))

    # caz: counter-azimuth (angle from y-axis in x-y plane)
    # Handle edge cases
    rho = np.sqrt(y_proj.flatten()**2 + z_proj.flatten()**2)
    caz = np.where(
        rho > 1e-10,
        np.arctan2(z_proj.flatten(), y_proj.flatten()),
        0.0
    )
    # Normalize to [0, 2π)
    caz = np.mod(caz, 2 * np.pi)

    return dec, caz


def ecef_to_weather_sat_frame_vectorized(
    target_ecef: np.ndarray,
    weather_sat_ecef: np.ndarray,
    weather_sat_velocity_ecef: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version of ecef_to_weather_sat_frame.

    Args:
        target_ecef: Target positions in ECEF, shape (N, 3) or broadcastable
        weather_sat_ecef: Weather satellite position in ECEF, shape (..., 3)
        weather_sat_velocity_ecef: Weather satellite velocity in ECEF, shape (..., 3)

    Returns:
        tuple: (dec, caz) arrays in radians
    """
    # Ensure arrays are at least 2D
    target_ecef = np.asarray(target_ecef)
    weather_sat_ecef = np.asarray(weather_sat_ecef)
    weather_sat_velocity_ecef = np.asarray(weather_sat_velocity_ecef)

    # Handle broadcasting
    if target_ecef.ndim == 1:
        target_ecef = target_ecef[np.newaxis, :]
    if weather_sat_ecef.ndim == 1:
        weather_sat_ecef = weather_sat_ecef[np.newaxis, :]
    if weather_sat_velocity_ecef.ndim == 1:
        weather_sat_velocity_ecef = weather_sat_velocity_ecef[np.newaxis, :]

    # Broadcast to common shape
    n_targets = target_ecef.shape[0]
    n_times = weather_sat_ecef.shape[0]

    # Broadcast weather satellite position/velocity for each target
    if n_times == 1:
        ws_ecef = np.broadcast_to(weather_sat_ecef, (n_targets, 3))
        ws_vel = np.broadcast_to(weather_sat_velocity_ecef, (n_targets, 3))
    else:
        # For multiple times, need to handle broadcasting carefully
        ws_ecef = weather_sat_ecef
        ws_vel = weather_sat_velocity_ecef

    return ecef_to_weather_sat_frame(target_ecef, ws_ecef, ws_vel)


def latlonalt_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """
    Convert latitude, longitude, altitude to ECEF coordinates.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters

    Returns:
        np.ndarray: ECEF coordinates [x, y, z] in meters
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis
    e2 = 0.00669437999014  # first eccentricity squared

    # Calculate radius of curvature in prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)

    return np.array([x, y, z])


# =============================================================================
# Weather Satellite Antenna Pattern Loading
# =============================================================================

def load_weather_sat_antenna_from_csv(
    csv_file: str,
    eta_rad: float = 0.9,
    valid_freqs: Tuple[float, float] = (0.0, 0.0)
) -> Antenna:
    """
    Load weather satellite antenna pattern from CSV file.

    CSV format: elevation angle (degrees) vs power (dB)
    Elevation: 0° = nadir, 90° = horizon, 180° = opposite nadir
    Note: CSV contains absolute gain values (not relative to peak).
          If values already include efficiency, set eta_rad=1.0.

    Args:
        csv_file: Path to CSV file with columns: elevation, power (absolute gain in dB)
        eta_rad: Radiation efficiency to apply (default: 0.9)
                 Set to 1.0 if CSV values already include efficiency
        valid_freqs: Valid frequency range (min, max) in Hz

    Returns:
        Antenna: Antenna object with gain pattern
    """
    # Read CSV file
    df = pd.read_csv(csv_file, header=None, names=['elevation', 'power'])

    # Sort by elevation angle
    df = df.sort_values('elevation').reset_index(drop=True)

    # Remove duplicate elevations (keep first occurrence, or average if needed)
    # RegularGridInterpolator requires strictly ascending/descending values
    df_unique = df.drop_duplicates(subset=['elevation'], keep='first')

    # If still duplicates due to rounding, group by elevation and take mean
    if len(df_unique) < len(df):
        df_unique = df.groupby('elevation')['power'].mean().reset_index()
        df_unique = df_unique.sort_values('elevation').reset_index(drop=True)

    # Get unique elevation angles (now guaranteed unique and sorted)
    elevations = df_unique['elevation'].values
    power_db = df_unique['power'].values

    # Convert elevation to alpha (declination from nadir in antenna frame)
    # In antenna frame: alpha = 0° is nadir, alpha = 90° is horizon
    # Elevation = 0° means nadir, so alpha = 0°
    # Elevation = 90° means horizon, so alpha = 90°
    alphas = elevations.copy()

    # Ensure alphas are strictly ascending (remove any remaining duplicates)
    # This handles edge cases where floating point precision causes issues
    unique_alphas, unique_indices = np.unique(alphas, return_index=True)
    if len(unique_alphas) < len(alphas):
        # If there were duplicates, use only unique values
        alphas = unique_alphas
        power_db = power_db[unique_indices]

    # Create full 2D pattern by rotating around azimuth (beta)
    # For simplicity, assume azimuthal symmetry (typical for weather satellite antennas)
    n_alpha = len(alphas)
    n_beta = 360  # 1 degree resolution in azimuth

    betas = np.arange(0, 360, 1)

    # Ensure alphas are strictly ascending (final check)
    if not np.all(np.diff(alphas) > 0):
        # If not strictly ascending, sort and remove any remaining duplicates
        sort_idx = np.argsort(alphas)
        alphas_sorted = alphas[sort_idx]
        power_db_sorted = power_db[sort_idx]

        # Remove duplicates (keep first)
        unique_mask = np.concatenate(([True], np.diff(alphas_sorted) > 1e-10))
        alphas = alphas_sorted[unique_mask]
        power_db = power_db_sorted[unique_mask]
        n_alpha = len(alphas)

    # Create 2D pattern: same power for all azimuths at same elevation
    # Shape: (n_alpha, n_beta) where each row is one alpha (elevation) and columns are betas (azimuths)
    power_db_2d = np.tile(power_db[:, np.newaxis], (1, n_beta))

    # Convert power (dB) to linear gain
    # CSV contains absolute gain values (not relative to peak)
    gain_linear = 10**(power_db_2d / 10.0)

    # Apply radiation efficiency
    # Note: If CSV values already include efficiency, set eta_rad=1.0
    gain_linear = gain_linear * eta_rad

    # Create DataFrame in format expected by Antenna class
    # The order matters: we need to match what map_sphere expects (Fortran order)
    # For each beta, we repeat all alphas; for each alpha, we tile across all betas
    # This creates: (alpha0, beta0), (alpha1, beta0), ..., (alphaN, beta0), (alpha0, beta1), ...
    gain_pat = pd.DataFrame({
        'alphas': np.tile(alphas, n_beta),
        'betas': np.repeat(betas, n_alpha),
        'gains': gain_linear.flatten(order='F')  # Fortran order to match map_sphere expectations
    })

    # Create Antenna object
    return Antenna.from_dataframe(gain_pat, eta_rad, valid_freqs)


# =============================================================================
# Harmonic Calculation Functions
# =============================================================================

def calculate_starlink_harmonic_contribution(
    base_frequency: float,
    base_link_budget: float,
    harmonics: list,
    observation_frequency: float,
    observation_bandwidth: float,
    starlink_distance: float
) -> float:
    """
    Calculate harmonic contribution from Starlink transmitter at observation frequency.

    Args:
        base_frequency: Starlink fundamental frequency (Hz)
        base_link_budget: Link budget at fundamental frequency (dimensionless)
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples.
            Example: [(2.0, 0.01), (3.0, 0.003), (4.0, 0.001)]
            - frequency_multiplier: Harmonic order (2.0 = 2nd harmonic)
            - power_reduction_factor: Power relative to fundamental (linear, 0-1)
        observation_frequency: Observation frequency (Hz)
        observation_bandwidth: Observation bandwidth (Hz)
        starlink_distance: Distance from weather sat to Starlink (meters)

    Returns:
        float: Harmonic link budget contribution (dimensionless)
    """
    harmonic_contribution = 0.0

    # Calculate path loss at fundamental frequency
    L_fundamental = free_space_loss(starlink_distance, base_frequency)

    for freq_mult, power_red in harmonics:
        harmonic_frequency = base_frequency * freq_mult

        # Check if harmonic falls within observation band
        freq_min = observation_frequency - observation_bandwidth / 2
        freq_max = observation_frequency + observation_bandwidth / 2

        if freq_min <= harmonic_frequency <= freq_max:
            # Calculate path loss at harmonic frequency (frequency-dependent)
            L_harmonic = free_space_loss(starlink_distance, harmonic_frequency)

            # Harmonic link budget accounts for:
            # 1. Power reduction factor (harmonic suppression)
            # 2. Path loss ratio (harmonic frequency vs fundamental)
            path_loss_ratio = L_fundamental / L_harmonic
            harmonic_contribution += base_link_budget * power_red * path_loss_ratio

    return harmonic_contribution


# =============================================================================
# Link Budget Functions
# =============================================================================

def starlink_backlobe_to_weather_sat_link_budget(
    weather_sat_dec: float,
    weather_sat_caz: float,
    weather_sat_antenna: Antenna,
    starlink_dec: float,
    starlink_caz: float,
    starlink_distance: float,
    starlink_antenna: Antenna,
    freq: float,
    polarization_loss_factor: float = 0.5,
    starlink_fundamental_freq: float = None,
    harmonics: list = None,
    observation_bandwidth: float = None
) -> float:
    """
    Calculate link budget for Starlink backlobe interference to weather satellite.

    Args:
        weather_sat_dec: Weather satellite pointing declination (nadir = 0)
        weather_sat_caz: Weather satellite pointing counter-azimuth
        weather_sat_antenna: Weather satellite antenna object
        starlink_dec: Starlink position declination in weather sat frame
        starlink_caz: Starlink position counter-azimuth in weather sat frame
        starlink_distance: Distance from weather sat to Starlink (meters)
        starlink_antenna: Starlink antenna object (backlobe pattern)
        freq: Frequency in Hz
        polarization_loss_factor: Polarization mismatch loss factor (linear, 0-1).
            Default 0.5 = -3 dB for circular (Starlink) to linear (Suomi-NPP) mismatch.

    Returns:
        float: Link budget (dimensionless)
    """
    # Weather satellite receive gain (main lobe looking at Earth)
    # Weather satellite is pointing at nadir (dec=0, caz=0 typically)
    # But we need gain in direction of Starlink
    gain_weather_sat = weather_sat_antenna.get_gain_value(starlink_dec, starlink_caz)

    # Starlink backlobe gain (toward weather satellite)
    # For backlobe, we need to find the angle from Starlink's boresight
    # Since we don't know Starlink's pointing, we'll use the backlobe pattern
    # which is typically much lower gain than main lobe
    # For now, use the angle from Starlink's nadir (assuming it's pointing at Earth)
    # The backlobe is roughly opposite to the main lobe
    # Angle from Starlink's nadir to weather satellite
    starlink_to_weather_dec = np.pi - starlink_dec  # Opposite direction
    starlink_to_weather_caz = (starlink_caz + np.pi) % (2 * np.pi)

    gain_starlink = starlink_antenna.get_gain_value(
        starlink_to_weather_dec,
        starlink_to_weather_caz
    )

    # Check if we have fundamental frequency and bandwidth information
    if (starlink_fundamental_freq is not None and observation_bandwidth is not None):
        # Calculate observation bandwidth bounds
        freq_min = freq - observation_bandwidth / 2
        freq_max = freq + observation_bandwidth / 2

        # Calculate base link budget at fundamental frequency
        base_link_budget_fund = simple_link_budget(
            gain_weather_sat, gain_starlink, starlink_distance,
            starlink_fundamental_freq
        )
        base_link_budget_fund *= polarization_loss_factor

        # Check if fundamental falls within observation bandwidth
        if freq_min <= starlink_fundamental_freq <= freq_max:
            # Fundamental is within observation band - include it
            link_budget = base_link_budget_fund
        else:
            # Fundamental is outside observation band - no base contribution
            link_budget = 0.0

        # Add harmonic contributions if harmonics are provided
        if harmonics is not None:
            harmonic_contribution = calculate_starlink_harmonic_contribution(
                starlink_fundamental_freq,
                base_link_budget_fund,
                harmonics,
                freq,
                observation_bandwidth,
                starlink_distance
            )
            # Add harmonic contribution
            link_budget += harmonic_contribution
    else:
        # No fundamental frequency info: calculate base link budget at observation frequency
        # (legacy behavior - may not be physically correct if frequencies don't match)
        link_budget = simple_link_budget(
            gain_weather_sat, gain_starlink, starlink_distance, freq
        )
        link_budget *= polarization_loss_factor

    return link_budget


def starlink_backlobe_to_weather_sat_link_budget_vectorized(
    weather_sat_dec: np.ndarray,
    weather_sat_caz: np.ndarray,
    weather_sat_antenna: Antenna,
    starlink_dec: np.ndarray,
    starlink_caz: np.ndarray,
    starlink_distance: np.ndarray,
    starlink_antenna: Antenna,
    freq: float,
    polarization_loss_factor: float = 0.5,
    starlink_fundamental_freq: float = None,
    harmonics: list = None,
    observation_bandwidth: float = None
) -> np.ndarray:
    """
    Vectorized version of starlink_backlobe_to_weather_sat_link_budget.

    Args:
        weather_sat_dec: Weather satellite pointing declination array
        weather_sat_caz: Weather satellite pointing counter-azimuth array
        weather_sat_antenna: Weather satellite antenna object
        starlink_dec: Starlink positions declination in weather sat frame
        starlink_caz: Starlink positions counter-azimuth in weather sat frame
        starlink_distance: Distances from weather sat to Starlink (meters)
        starlink_antenna: Starlink antenna object
        freq: Frequency in Hz
        polarization_loss_factor: Polarization mismatch loss factor (linear, 0-1).
            Default 0.5 = -3 dB for circular (Starlink) to linear (Suomi-NPP) mismatch.

    Returns:
        np.ndarray: Link budget array
    """
    # Broadcast arrays
    starlink_dec = np.asarray(starlink_dec)
    starlink_caz = np.asarray(starlink_caz)
    starlink_distance = np.asarray(starlink_distance)

    # Weather satellite gain (vectorized)
    gain_weather_sat = weather_sat_antenna.get_gain_values(starlink_dec, starlink_caz)

    # Starlink backlobe gain
    starlink_to_weather_dec = np.pi - starlink_dec
    starlink_to_weather_caz = np.mod(starlink_caz + np.pi, 2 * np.pi)

    gain_starlink = starlink_antenna.get_gain_values(
        starlink_to_weather_dec,
        starlink_to_weather_caz
    )

    # Check if we have fundamental frequency and bandwidth information
    if (starlink_fundamental_freq is not None and observation_bandwidth is not None):
        # Calculate observation bandwidth bounds
        freq_min = freq - observation_bandwidth / 2
        freq_max = freq + observation_bandwidth / 2

        # Calculate base link budget at fundamental frequency
        L_fund = free_space_loss(starlink_distance, starlink_fundamental_freq)
        base_link_budget_fund = (gain_weather_sat * (1.0 / L_fund) * gain_starlink *
                                 polarization_loss_factor)

        # Check if fundamental falls within observation bandwidth
        fundamental_in_band = (freq_min <= starlink_fundamental_freq <= freq_max)
        if fundamental_in_band:
            # Fundamental is within observation band - include it
            link_budget = base_link_budget_fund
        else:
            # Fundamental is outside observation band - no base contribution
            link_budget = np.zeros_like(base_link_budget_fund)

        # Add harmonic contributions if harmonics are provided
        if harmonics is not None:
            # Calculate frequency bounds once
            L_fundamental = L_fund

            # Vectorized calculation for all harmonics
            harmonic_contributions = np.zeros_like(base_link_budget_fund)

            for freq_mult, power_red in harmonics:
                harmonic_frequency = starlink_fundamental_freq * freq_mult

                # Check if harmonic falls within observation band
                if freq_min <= harmonic_frequency <= freq_max:
                    # Calculate path loss at harmonic frequency (vectorized)
                    L_harmonic = free_space_loss(starlink_distance, harmonic_frequency)

                    # Path loss ratio (vectorized)
                    path_loss_ratio = L_fundamental / L_harmonic

                    # Apply power reduction and path loss ratio (vectorized)
                    harmonic_contributions += base_link_budget_fund * power_red * path_loss_ratio

            # Add harmonic contribution
            link_budget += harmonic_contributions
    else:
        # No fundamental frequency info: calculate base link budget at observation frequency
        # (legacy behavior - may not be physically correct if frequencies don't match)
        L = free_space_loss(starlink_distance, freq)
        link_budget = (gain_weather_sat * (1.0 / L) * gain_starlink *
                       polarization_loss_factor)

    return link_budget


# =============================================================================
# Frequency-Dependent Earth Brightness Temperature
# =============================================================================

def calculate_earth_brightness_temperature(freq: float, base_temp: float = 280.0) -> float:
    """
    Calculate frequency-dependent Earth brightness temperature accounting for
    atmospheric absorption and emission.

    At higher frequencies (especially near 50 GHz), atmospheric oxygen absorption
    becomes significant, and the atmosphere itself emits thermal radiation,
    increasing the effective brightness temperature seen by a satellite looking down.

    Args:
        freq: Frequency in Hz
        base_temp: Base Earth brightness temperature (K) at low frequencies

    Returns:
        float: Effective Earth brightness temperature (K)
    """
    freq_ghz = freq / 1e9

    # Atmospheric oxygen absorption peak is around 60 GHz
    # For frequencies near this peak, atmospheric emission increases brightness temp
    # Simplified model based on oxygen absorption band characteristics
    if freq_ghz < 20:
        # Low frequencies: minimal atmospheric effects
        return base_temp
    elif freq_ghz < 40:
        # K-Band region (20-40 GHz): moderate atmospheric effects
        # Water vapor absorption becomes significant
        # Increase of ~10-30 K due to atmospheric emission
        return base_temp + 15.0 * ((freq_ghz - 20) / 20.0)
    elif freq_ghz < 60:
        # V-Band region (40-60 GHz): strong atmospheric effects
        # Approaching oxygen absorption band (60 GHz)
        # Atmospheric emission significantly increases brightness temp
        # At 50.3 GHz, typical increase is 50-100 K
        oxygen_band_center = 60.0  # GHz
        distance_from_peak = abs(freq_ghz - oxygen_band_center)
        # Peak emission occurs near 60 GHz, but significant at 50 GHz
        if distance_from_peak < 10:
            # Within 10 GHz of oxygen line
            emission_factor = 1.0 - (distance_from_peak / 10.0)
            # Maximum increase of ~80 K near the line, ~50 K at 50.3 GHz
            temp_increase = 50.0 + 30.0 * emission_factor
            return base_temp + temp_increase
        else:
            # Far from oxygen line, but still in V-band
            return base_temp + 30.0
    else:
        # Above 60 GHz: very strong oxygen absorption
        # Brightness temperature can approach atmospheric temperature (~250-280 K)
        return min(base_temp + 100.0, 380.0)


# =============================================================================
# Observation Modeling
# =============================================================================

def compute_weather_sat_ecef_from_trajectory(
    trajectory: Trajectory,
    observer_lat: float,
    observer_lon: float,
    observer_alt: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute weather satellite ECEF positions and velocities from trajectory.

    Trajectory contains elevation, azimuth, distance from observer.
    We convert these to ECEF coordinates.

    Args:
        trajectory: Weather satellite trajectory
        observer_lat: Observer latitude (degrees)
        observer_lon: Observer longitude (degrees)
        observer_alt: Observer altitude (meters)

    Returns:
        tuple: (ecef_positions, ecef_velocities) DataFrames
    """
    traj_df = trajectory.get_traj().sort_values('times').reset_index(drop=True)

    # Observer position in ECEF
    observer_ecef = latlonalt_to_ecef(observer_lat, observer_lon, observer_alt)

    # Convert spherical coordinates to ECEF
    n_points = len(traj_df)
    ecef_positions = np.zeros((n_points, 3))
    ecef_velocities = np.zeros((n_points, 3))

    for i, row in traj_df.iterrows():
        elev_rad = np.deg2rad(row['elevations'])
        azim_rad = np.deg2rad(row['azimuths'])
        dist = row['distances']

        # Convert to local ENU (East-North-Up) coordinates
        # From observer to satellite
        e = dist * np.sin(elev_rad) * np.sin(azim_rad)  # East
        n = dist * np.sin(elev_rad) * np.cos(azim_rad)  # North
        u = dist * np.cos(elev_rad)  # Up

        # Convert ENU to ECEF (simplified - would need proper transformation)
        # For now, use approximate conversion
        lat_rad = np.deg2rad(observer_lat)
        lon_rad = np.deg2rad(observer_lon)

        # Rotation matrix from ENU to ECEF
        R = np.array([
            [-np.sin(lon_rad), -np.sin(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.cos(lon_rad)],
            [np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad) * np.sin(lon_rad)],
            [0, np.cos(lat_rad), np.sin(lat_rad)]
        ])

        enu_vec = np.array([e, n, u])
        ecef_vec = R @ enu_vec + observer_ecef
        ecef_positions[i] = ecef_vec

    # Compute velocities from positions (finite difference)
    time_diffs = (traj_df['times'].diff().dt.total_seconds()).values
    time_diffs[0] = time_diffs[1] if len(time_diffs) > 1 else 1.0

    for i in range(n_points):
        if i == 0:
            # Forward difference
            if n_points > 1:
                dt = time_diffs[1]
                ecef_velocities[i] = (ecef_positions[1] - ecef_positions[0]) / dt
            else:
                ecef_velocities[i] = np.zeros(3)
        elif i == n_points - 1:
            # Backward difference
            dt = time_diffs[i]
            ecef_velocities[i] = (ecef_positions[i] - ecef_positions[i-1]) / dt
        else:
            # Central difference
            dt = (time_diffs[i] + time_diffs[i+1]) / 2.0
            ecef_velocities[i] = (ecef_positions[i+1] - ecef_positions[i-1]) / (2.0 * dt)

    ecef_pos_df = pd.DataFrame(ecef_positions, columns=['x', 'y', 'z'])
    ecef_pos_df['times'] = traj_df['times'].values

    ecef_vel_df = pd.DataFrame(ecef_velocities, columns=['vx', 'vy', 'vz'])
    ecef_vel_df['times'] = traj_df['times'].values

    return ecef_pos_df, ecef_vel_df


def model_weather_sat_observed_power(
    weather_sat_trajectory: Trajectory,
    weather_sat_instrument: Instrument,
    starlink_constellation: Constellation,
    observation_times: np.ndarray,
    observer_lat: float,
    observer_lon: float,
    observer_alt: float,
    target_lat: float,
    target_lon: float,
    target_alt: float,
    freq_channels: np.ndarray,
    earth_brightness_temp: float = 280.0,
    sky_brightness_temp: float = 2.73,
    system_temp: float = 300.0,
    starlink_eirp_dbw: float = 40.0,
    polarization_loss_factor: float = 0.5,
    starlink_fundamental_freq: float = None,
    harmonics: list = None
) -> dict:
    """
    Model observed power at weather satellite from all RFI sources.

    Args:
        weather_sat_trajectory: Weather satellite trajectory
        weather_sat_instrument: Weather satellite instrument
        starlink_constellation: Starlink constellation
        observation_times: Array of observation timestamps
        observer_lat: Observer latitude (degrees) - location where trajectory was computed from
        observer_lon: Observer longitude (degrees)
        observer_alt: Observer altitude (meters)
        target_lat: Target latitude (degrees) - center of resolution element
        target_lon: Target longitude (degrees)
        target_alt: Target altitude (meters)
        freq_channels: Frequency channels in Hz
        earth_brightness_temp: Earth brightness temperature (K)
        sky_brightness_temp: Sky background temperature (K)
        system_temp: System temperature (K)
        starlink_eirp_dbw: Starlink EIRP in dBW
        polarization_loss_factor: Polarization mismatch loss factor (linear, 0-1).
            Default 0.5 = -3 dB for circular (Starlink) to linear (Suomi-NPP) mismatch.
        starlink_fundamental_freq: Starlink fundamental frequency (Hz). If None,
            harmonics are not calculated.
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples.
            Example: [(2.0, 0.01), (3.0, 0.003), (4.0, 0.001)]

    Returns:
        np.ndarray: Observed power in dBW, shape (n_times, n_freqs)
    """
    n_times = len(observation_times)
    n_freqs = len(freq_channels)

    # Get weather satellite antenna
    weather_sat_antenna = weather_sat_instrument.get_antenna()

    # Get Starlink antenna (backlobe pattern)
    starlink_antenna = starlink_constellation.get_antenna()

    # Get bandwidth
    bandwidth = weather_sat_instrument.get_bandwidth()

    # Compute weather satellite ECEF positions and velocities
    ws_ecef_pos, ws_ecef_vel = compute_weather_sat_ecef_from_trajectory(
        weather_sat_trajectory,
        observer_lat, observer_lon, observer_alt
    )

    # Convert target location to ECEF
    target_ecef = latlonalt_to_ecef(target_lat, target_lon, target_alt)

    # Get Starlink trajectory data
    starlink_traj_df = starlink_constellation.sats.copy()
    starlink_traj_df = starlink_traj_df.sort_values('times').reset_index(drop=True)

    # Compute Starlink ECEF positions (from trajectory)
    # For now, use simplified approach - would need proper computation
    # from Starlink trajectory data

    # Initialize result arrays for total and individual components
    result_power = np.zeros((n_times, n_freqs))  # Will store total power in W
    result_starlink = np.zeros((n_times, n_freqs))  # Starlink interference
    result_earth = np.zeros((n_times, n_freqs))  # Earth brightness
    result_sky = np.zeros((n_times, n_freqs))  # Sky background
    result_system = np.zeros((n_times, n_freqs))  # System noise

    print(f"  Processing {n_times} time steps and {n_freqs} frequency channels...")
    print(f"  Total Starlink satellites in constellation: {starlink_traj_df['sat'].nunique()}")

    # Process each time step
    for t_idx, obs_time in enumerate(observation_times):
        if (t_idx + 1) % max(1, n_times // 10) == 0 or (t_idx + 1) == n_times:
            print(f"    Progress: {t_idx + 1}/{n_times} ({100*(t_idx+1)/n_times:.1f}%)")

        # Find weather satellite position at this time
        ws_mask = ws_ecef_pos['times'] == obs_time
        if not ws_mask.any():
            # Interpolate or use nearest
            time_diffs = np.abs((ws_ecef_pos['times'] - obs_time).dt.total_seconds())
            nearest_idx = time_diffs.idxmin()
            ws_ecef = ws_ecef_pos.iloc[nearest_idx][['x', 'y', 'z']].values
            ws_vel = ws_ecef_vel.iloc[nearest_idx][['vx', 'vy', 'vz']].values
        else:
            ws_ecef = ws_ecef_pos[ws_mask].iloc[0][['x', 'y', 'z']].values
            ws_vel = ws_ecef_vel[ws_mask].iloc[0][['vx', 'vy', 'vz']].values

        # Transform target to weather satellite frame
        target_dec, target_caz = ecef_to_weather_sat_frame(
            target_ecef[np.newaxis, :],
            ws_ecef[np.newaxis, :],
            ws_vel[np.newaxis, :]
        )
        target_dec = target_dec[0]
        target_caz = target_caz[0]

        # Find Starlink satellites at this time
        starlink_sats = starlink_traj_df[
            starlink_traj_df['times'] == obs_time
        ]

        n_visible_starlinks = len(starlink_sats)
        if t_idx == 0:
            print(f"    Visible Starlink satellites at first time step: {n_visible_starlinks}")

        # Process each frequency channel
        for f_idx, freq in enumerate(freq_channels):
            starlink_interference_temp = 0.0

            # Process each Starlink satellite
            if n_visible_starlinks > 0:
                for _, sat_row in starlink_sats.iterrows():
                    # Get Starlink position from trajectory
                    # For now, use spherical coordinates relative to observer
                    # In practice, would compute Starlink ECEF from trajectory
                    sat_dist = sat_row['distances']
                    sat_elev = np.deg2rad(sat_row['elevations'])
                    sat_azim = np.deg2rad(sat_row['azimuths'])

                    # Convert Starlink position to ECEF (from observer)
                    observer_ecef = latlonalt_to_ecef(observer_lat, observer_lon, observer_alt)

                    # Convert spherical coordinates (elev, azim, dist) to local ENU
                    e = sat_dist * np.sin(sat_elev) * np.sin(sat_azim)  # East
                    n = sat_dist * np.sin(sat_elev) * np.cos(sat_azim)  # North
                    u = sat_dist * np.cos(sat_elev)  # Up

                    # Convert ENU to ECEF
                    obs_lat_rad = np.deg2rad(observer_lat)
                    obs_lon_rad = np.deg2rad(observer_lon)

                    # Rotation matrix from ENU to ECEF
                    R_enu_to_ecef = np.array([
                        [-np.sin(obs_lon_rad), -np.sin(obs_lat_rad) * np.cos(obs_lon_rad),
                         np.cos(obs_lat_rad) * np.cos(obs_lon_rad)],
                        [np.cos(obs_lon_rad), -np.sin(obs_lat_rad) * np.sin(obs_lon_rad),
                         np.cos(obs_lat_rad) * np.sin(obs_lon_rad)],
                        [0, np.cos(obs_lat_rad), np.sin(obs_lat_rad)]
                    ])

                    enu_vec = np.array([e, n, u])
                    sat_ecef = R_enu_to_ecef @ enu_vec + observer_ecef

                    # Transform Starlink to weather satellite frame
                    sat_dec, sat_caz = ecef_to_weather_sat_frame(
                        sat_ecef[np.newaxis, :],
                        ws_ecef[np.newaxis, :],
                        ws_vel[np.newaxis, :]
                    )
                    sat_dec = sat_dec[0]
                    sat_caz = sat_caz[0]

                    # Calculate actual distance from weather satellite to Starlink
                    sat_to_ws_vec = sat_ecef - ws_ecef
                    sat_to_ws_dist = np.linalg.norm(sat_to_ws_vec)

                    # Calculate link budget (includes polarization mismatch loss and harmonics)
                    link_budget = starlink_backlobe_to_weather_sat_link_budget(
                        0.0, 0.0,  # Weather sat pointing at nadir
                        weather_sat_antenna,
                        sat_dec, sat_caz,
                        sat_to_ws_dist,
                        starlink_antenna,
                        freq,
                        polarization_loss_factor=polarization_loss_factor,
                        starlink_fundamental_freq=starlink_fundamental_freq,
                        harmonics=harmonics,
                        observation_bandwidth=bandwidth
                    )

                    # Starlink transmit power (convert to temperature)
                    starlink_power_w = 10**(starlink_eirp_dbw / 10.0)
                    starlink_temp = power_to_temperature(starlink_power_w, bandwidth)

                    # Interference temperature
                    interference_temp = link_budget * starlink_temp
                    starlink_interference_temp += interference_temp

            # Convert interference to power
            starlink_power = temperature_to_power(starlink_interference_temp, bandwidth)

            # Earth brightness (through main lobe pointing at target)
            # Use frequency-dependent brightness temperature to account for atmospheric effects
            earth_temp_freq = calculate_earth_brightness_temperature(freq, earth_brightness_temp)
            earth_gain = weather_sat_antenna.get_gain_value(target_dec, target_caz)
            earth_power = temperature_to_power(earth_temp_freq, bandwidth) * earth_gain

            # Sky background (through sidelobes - use average gain)
            sky_power = temperature_to_power(sky_brightness_temp, bandwidth) * 0.1  # Approximate sidelobe gain

            # System noise
            system_power = temperature_to_power(system_temp, bandwidth)

            # Store individual components
            result_starlink[t_idx, f_idx] = starlink_power
            result_earth[t_idx, f_idx] = earth_power
            result_sky[t_idx, f_idx] = sky_power
            result_system[t_idx, f_idx] = system_power

            # Total power
            result_power[t_idx, f_idx] = (
                starlink_power + earth_power + sky_power + system_power
            )

    # Convert to dBW
    result_power_dbw = 10 * np.log10(result_power + 1e-100)  # Add small value to avoid log(0)
    result_starlink_dbw = 10 * np.log10(result_starlink + 1e-100)
    result_earth_dbw = 10 * np.log10(result_earth + 1e-100)
    result_sky_dbw = 10 * np.log10(result_sky + 1e-100)
    result_system_dbw = 10 * np.log10(result_system + 1e-100)

    # Return dictionary with total and individual components
    return {
        'total': result_power_dbw,
        'starlink': result_starlink_dbw,
        'earth': result_earth_dbw,
        'sky': result_sky_dbw,
        'system': result_system_dbw
    }
