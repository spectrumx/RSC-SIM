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
from typing import Tuple, Optional

# Import existing modules without modifying them
from radio_types import Antenna, Instrument, Trajectory, Constellation
from astro_mdl import power_to_temperature, temperature_to_power
from sat_mdl import free_space_loss, simple_link_budget

# Import ITU-R P.676 calculator for full atmospheric modeling
try:
    from attenuation_mdl import get_cached_calculator, ITUP676Calculator
    ITU_P676_AVAILABLE = True
except ImportError:
    ITU_P676_AVAILABLE = False
    print("Warning: attenuation_mdl not available. Using simplified atmospheric model.")

# Try to import rasterio for DEM support
try:
    import rasterio
    from rasterio.transform import rowcol
    from pyproj import Proj, transform as pyproj_transform
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. DEM-based terrain masking will be disabled.")
    print("Install with: pip install rasterio")

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

    Weather satellite body frame (origin at satellite):
    - X (nadir): Points toward Earth center
    - Y (along-track): Points in velocity direction
    - Z (cross-track): Completes right-handed system (Z = X × Y)

    Direction to a target (e.g. 5G ground emitter) from the satellite is given by
    spherical angles (dec, caz) in this frame:
    - dec: Declination from nadir (angle from X); 0 = nadir, π/2 = horizon
    - caz: Counter-azimuth in the Y-Z plane; 0 = along-track (+Y), increases
      counter-clockwise when viewed from +X (i.e. from satellite toward Earth)

    Diagram (schematic; X = nadir into page at origin, Y = along-track, Z = cross-track):

                Y (along-track)
                |
                |    * target (e.g. 5G ground emitter)
                |   /
                |  /  dec (angle from nadir X)
                | /
                |/
                +---------------- Y
               /|
              / |
             Z  X (nadir, toward Earth; into page)
         (cross-track)
             caz = angle in Y-Z plane from +Y toward +Z (counter-clockwise about X)

    So the 5G ground emitter has no separate local frame in this model: its
    position is in ECEF, and the direction satellite → emitter is expressed
    in the weather satellite body frame as (dec, caz).

    Args:
        target_ecef: Target position in ECEF [x, y, z] (meters) or (N, 3) array
        weather_sat_ecef: Weather satellite position in ECEF [x, y, z] (meters) or (N, 3) array
        weather_sat_velocity_ecef: Weather satellite velocity in ECEF [vx, vy, vz] (m/s) or (N, 3) array

    Returns:
        tuple: (dec, caz) in radians
            - dec: Declination angle from nadir (0 = nadir, π/2 = horizon)
            - caz: Counter-azimuth angle (0 = along-track, counter-clockwise in Y-Z plane)
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

    # caz: counter-azimuth (angle from y-axis in Y-Z plane)
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


def latlonalt_to_ecef_vectorized(lats: np.ndarray, lons: np.ndarray, alts: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of latitude, longitude, altitude to ECEF coordinates.

    Args:
        lats: Array of latitudes in degrees (shape: [n])
        lons: Array of longitudes in degrees (shape: [n])
        alts: Array of altitudes in meters (shape: [n])

    Returns:
        np.ndarray: ECEF coordinates [n, 3] in meters
    """
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)

    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis
    e2 = 0.00669437999014  # first eccentricity squared

    # Calculate radius of curvature in prime vertical (vectorized)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # ECEF coordinates (vectorized)
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    x = (N + alts) * cos_lat * cos_lon
    y = (N + alts) * cos_lat * sin_lon
    z = (N * (1 - e2) + alts) * sin_lat

    return np.column_stack([x, y, z])


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
            # Extract gain components from base_link_budget
            # base_link_budget = gain_ws * (1/L_fundamental) * gain_starlink
            # For harmonic: gain_ws * (1/L_harmonic) * gain_starlink * power_red
            # Ratio: (L_fundamental / L_harmonic) * power_red
            path_loss_ratio = L_fundamental / L_harmonic
            harmonic_contribution += base_link_budget * power_red * path_loss_ratio

    return harmonic_contribution


def calculate_starlink_harmonic_contribution_vectorized(
    base_frequency: float,
    base_link_budgets: np.ndarray,
    harmonics: list,
    observation_frequency: float,
    observation_bandwidth: float,
    starlink_distances: np.ndarray
) -> np.ndarray:
    """
    Vectorized version of calculate_starlink_harmonic_contribution.

    Args:
        base_frequency: Starlink fundamental frequency (Hz)
        base_link_budgets: Array of link budgets at fundamental frequency
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples
        observation_frequency: Observation frequency (Hz)
        observation_bandwidth: Observation bandwidth (Hz)
        starlink_distances: Array of distances from weather sat to Starlink (meters)

    Returns:
        np.ndarray: Harmonic link budget contributions (dimensionless)
    """
    base_link_budgets = np.asarray(base_link_budgets)
    starlink_distances = np.asarray(starlink_distances)
    harmonic_contributions = np.zeros_like(base_link_budgets)

    # Calculate frequency bounds once
    freq_min = observation_frequency - observation_bandwidth / 2
    freq_max = observation_frequency + observation_bandwidth / 2

    # Calculate path loss at fundamental frequency (vectorized)
    L_fundamental = free_space_loss(starlink_distances, base_frequency)

    # Vectorized calculation for all harmonics
    for freq_mult, power_red in harmonics:
        harmonic_frequency = base_frequency * freq_mult

        # Check if harmonic falls within observation band
        if freq_min <= harmonic_frequency <= freq_max:
            # Calculate path loss at harmonic frequency (vectorized)
            L_harmonic = free_space_loss(starlink_distances, harmonic_frequency)

            # Path loss ratio (vectorized)
            path_loss_ratio = L_fundamental / L_harmonic

            # Apply power reduction and path loss ratio (vectorized)
            harmonic_contributions += base_link_budgets * power_red * path_loss_ratio

    return harmonic_contributions


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
        starlink_fundamental_freq: Starlink fundamental frequency (Hz). If None,
            harmonics are not calculated.
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples.
            Example: [(2.0, 0.01), (3.0, 0.003), (4.0, 0.001)]
        observation_bandwidth: Observation bandwidth (Hz). Required if harmonics
            are provided.

    Returns:
        float: Total link budget including harmonics (dimensionless)
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
        starlink_fundamental_freq: Starlink fundamental frequency (Hz). If None,
            harmonics are not calculated.
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples.
        observation_bandwidth: Observation bandwidth (Hz). Required if harmonics
            are provided.

    Returns:
        np.ndarray: Total link budget array including harmonics (dimensionless)
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
            harmonic_contribution = calculate_starlink_harmonic_contribution_vectorized(
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

    Ground (observer) frame: East-North-Up (ENU) at observer position.
    - East (E), North (N), Up (U = local zenith).
    - Trajectory columns: 'elevations', 'azimuths', 'distances'.

    Angle convention used in this conversion:
    - Trajectory 'elevations': treated as altitude above horizon (0° = horizon,
      90° = zenith), e.g. from Skyfield altaz(). Converted internally to polar
      from zenith (0° = zenith, 90° = horizon) for ENU: polar = 90° - altitude.
    - Trajectory 'azimuths': from North clockwise, 0° = North, 90° = East
      (standard geodetic / Skyfield convention).

    N-W-Z (North-West-Zenith) vs ENU: This routine uses ENU. In N-W-Z, polar from
    Zenith is the same; counter-azimuth (from North toward West) relates to
    azimuth (from North toward East) by azimuth = 360° - co_azimuth (or
    co_azimuth = 360° - azimuth), depending on sign convention.

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
        # Trajectory typically stores altitude above horizon (e.g. Skyfield altaz);
        # ENU formula uses polar angle from zenith (0=zenith, 90=horizon).
        alt_above_horizon_deg = row['elevations']
        polar_from_zenith_deg = 90.0 - alt_above_horizon_deg
        elev_rad = np.deg2rad(polar_from_zenith_deg)
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
        dict: Dictionary with total and individual components of observed power in dBW.
              Keys: 'total', 'starlink', 'earth', 'sky', 'system'.
              Values: np.ndarray of shape (n_times, n_freqs)
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
                    sat_dist = sat_row['distances']
                    # Trajectory 'elevations' = altitude above horizon; ENU uses polar from zenith
                    sat_polar_zenith_deg = 90.0 - sat_row['elevations']
                    sat_elev = np.deg2rad(sat_polar_zenith_deg)
                    sat_azim = np.deg2rad(sat_row['azimuths'])

                    # Convert Starlink position to ECEF (from observer)
                    observer_ecef = latlonalt_to_ecef(observer_lat, observer_lon, observer_alt)

                    # Convert spherical coordinates (polar from zenith, azim, dist) to local ENU
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


# =============================================================================
# Phase 2: Ground Emitter Modeling (5G Cellular Networks)
# =============================================================================

def create_5g_sector_antenna_pattern(
    gain_max: float = 18.0,
    horiz_beamwidth: float = 65.0,
    vert_beamwidth: float = 10.0,
    eta_rad: float = 0.8,
    valid_freqs: Tuple[float, float] = (0.0, 0.0)
) -> Antenna:
    """
    Create 5G sector antenna pattern using ITU/empirical model.

    Typical 5G base station antennas are sector antennas with:
    - Horizontal beamwidth: 65° (typical for 3-sector sites)
    - Vertical beamwidth: 10° (typical for downtilt)
    - Maximum gain: ~18 dBi (typical for sector antennas)

    Args:
        gain_max: Maximum gain in dBi (typical: 18-20 dBi for 5G sector antennas)
        horiz_beamwidth: Horizontal beamwidth in degrees (typical: 65°)
        vert_beamwidth: Vertical beamwidth in degrees (typical: 10°)
        eta_rad: Radiation efficiency
        valid_freqs: Valid frequency range (min, max) in Hz

    Returns:
        Antenna: Antenna object with 5G sector pattern
    """
    # Create angular grid
    alphas = np.arange(0, 181, 1)  # Elevation angles (0-180°)
    betas = np.arange(0, 360, 1)  # Azimuth angles (0-360°)

    # Create 2D gain pattern (will be computed below)

    # Vectorized calculation for better performance
    alphas_2d = np.tile(alphas[:, np.newaxis], (1, len(betas)))
    betas_2d = np.tile(betas[np.newaxis, :], (len(alphas), 1))

    # Horizontal gain (azimuth-dependent) - normalized relative pattern
    # Sector typically points in one direction (beta=0)
    # For simplicity, model single sector pointing at beta=0
    beta_rel = np.abs(betas_2d)
    beta_rel = np.minimum(beta_rel, 360 - beta_rel)

    # Horizontal pattern: Gaussian approximation (relative, normalized to 1.0 at peak)
    # Use relative pattern so multiplication doesn't double the gain
    horiz_gain_db_rel = -3.0 * (beta_rel / (horiz_beamwidth / 2)) ** 2
    # Limit sidelobe levels
    mask_high_angle = beta_rel > horiz_beamwidth
    horiz_gain_db_rel[mask_high_angle] = (
        -20.0 - 25.0 * np.log10(np.maximum(beta_rel[mask_high_angle] / horiz_beamwidth, 1.0))
    )

    # Vertical pattern (elevation-dependent) - normalized relative pattern
    # For ground-based sector antenna, peak is typically at alpha = 90° (horizon)
    # Sector antennas are typically downtilted, so peak is near horizon, not straight up
    # Model peak at alpha = 90° (horizon) - typical for sector antennas
    alpha_rel = np.abs(alphas_2d - 90.0)

    # Vertical pattern: Gaussian approximation (relative, normalized to 1.0 at peak)
    vert_gain_db_rel = -3.0 * (alpha_rel / (vert_beamwidth / 2)) ** 2
    # Limit backlobe levels
    mask_high_elev = alpha_rel > vert_beamwidth
    vert_gain_db_rel[mask_high_elev] = (
        -20.0 - 25.0 * np.log10(np.maximum(alpha_rel[mask_high_elev] / vert_beamwidth, 1.0))
    )

    # Combined gain (use product of relative patterns, then scale by gain_max)
    # For sector antennas, both patterns contribute multiplicatively
    # Convert relative patterns to linear, multiply, then scale by gain_max
    horiz_gain_linear_rel = 10 ** (horiz_gain_db_rel / 10.0)
    vert_gain_linear_rel = 10 ** (vert_gain_db_rel / 10.0)
    combined_gain_linear_rel = horiz_gain_linear_rel * vert_gain_linear_rel

    # Scale by gain_max to get absolute gain
    gain_max_linear = 10 ** (gain_max / 10.0)
    combined_gain_linear = combined_gain_linear_rel * gain_max_linear

    # Apply radiation efficiency to get absolute gain
    # Peak gain should be gain_max * eta_rad (in linear scale)
    combined_gain_linear = combined_gain_linear * eta_rad

    # Limit minimum gain to reasonable backlobe level (in linear scale)
    gain_min_linear = 10 ** ((gain_max - 30.0) / 10.0) * eta_rad
    combined_gain_linear = np.maximum(combined_gain_linear, gain_min_linear)

    # Use the linear gain values directly (already have absolute values with efficiency)
    gain_linear = combined_gain_linear

    # Note: For link budget calculations, EIRP already includes the antenna gain.
    # However, for plotting purposes, we want absolute gain values.
    # The link budget will use this pattern, but EIRP accounts for the peak gain.
    # So we need to scale the pattern appropriately in the link budget.

    # Create DataFrame
    gain_pat = pd.DataFrame({
        'alphas': np.tile(alphas, len(betas)),
        'betas': np.repeat(betas, len(alphas)),
        'gains': gain_linear.flatten(order='F')
    })

    return Antenna.from_dataframe(gain_pat, eta_rad, valid_freqs)


def generate_ground_emitter_distribution(
    center_lat: float,
    center_lon: float,
    resolution_km: float = 32.0,
    emitter_density_per_km2: float = 0.1,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate statistical distribution of 5G ground emitters within resolution element.

    Args:
        center_lat: Center latitude of resolution element (degrees)
        center_lon: Center longitude of resolution element (degrees)
        resolution_km: Resolution element diameter in km (default: 32 km)
        emitter_density_per_km2: Number of emitters per km² (default: 0.1)
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame with columns: 'lat', 'lon', 'alt' (meters)
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate number of emitters
    area_km2 = np.pi * (resolution_km / 2) ** 2
    n_emitters = int(np.ceil(area_km2 * emitter_density_per_km2))

    # Generate random positions within circular resolution element
    # Using uniform distribution in polar coordinates
    r_max = resolution_km / 2 * 1000  # Convert to meters
    theta = np.random.uniform(0, 2 * np.pi, n_emitters)
    r = r_max * np.sqrt(np.random.uniform(0, 1, n_emitters))  # Uniform in area

    # Convert to lat/lon offsets (approximate, for small areas)
    center_lat_rad = np.deg2rad(center_lat)

    # Approximate conversion (meters to degrees)
    lat_offset = r * np.cos(theta) / 111320.0  # meters to degrees (latitude)
    lon_offset = r * np.sin(theta) / (111320.0 * np.cos(center_lat_rad))  # meters to degrees (longitude)

    emitter_lats = center_lat + lat_offset
    emitter_lons = center_lon + lon_offset

    # Typical antenna height: 20-50 meters above ground
    # For simplicity, use average of 30 meters
    emitter_alts = np.random.uniform(20.0, 50.0, n_emitters)

    # Create DataFrame
    emitters_df = pd.DataFrame({
        'lat': emitter_lats,
        'lon': emitter_lons,
        'alt': emitter_alts
    })

    return emitters_df


# =============================================================================
# DEM-Based Terrain Masking (Phase 2)
# =============================================================================

class DEMTerrainMasker:
    """
    DEM-based terrain masking for weather satellite perspective (looking down).

    This class handles terrain elevation lookups and ray tracing for line-of-sight
    checks from weather satellite to ground emitters.
    """

    def __init__(self, dem_file: Optional[str] = None):
        """
        Initialize DEM terrain masker.

        Args:
            dem_file: Path to DEM GeoTIFF file (optional)
        """
        self.dem_file = dem_file
        self.dem_data = None
        self.dem_transform = None
        self.dem_crs = None
        self.dem_bounds = None
        # Cache Proj objects for performance (created once, reused many times)
        self.wgs84_proj = None
        self.dem_proj = None

        if dem_file is not None and RASTERIO_AVAILABLE:
            self.load_dem()
        elif dem_file is not None and not RASTERIO_AVAILABLE:
            print("Warning: DEM file provided but rasterio not available.")
            print("  Falling back to geometric horizon check only.")

    def load_dem(self):
        """Load DEM data from GeoTIFF file"""
        if not RASTERIO_AVAILABLE:
            return

        try:
            with rasterio.open(self.dem_file) as src:
                self.dem_data = src.read(1)
                self.dem_transform = src.transform
                self.dem_crs = src.crs
                self.dem_bounds = src.bounds

                # Create and cache Proj objects for performance
                # These are expensive to create, so we do it once and reuse
                if RASTERIO_AVAILABLE:
                    try:
                        self.wgs84_proj = Proj(init='epsg:4326')
                        self.dem_proj = Proj(self.dem_crs)
                    except Exception:
                        # Fallback if Proj creation fails
                        self.wgs84_proj = None
                        self.dem_proj = None

                print("DEM loaded successfully:")
                print(f"  Shape: {self.dem_data.shape}")
                print(f"  CRS: {self.dem_crs}")
                print(f"  Bounds: {self.dem_bounds}")
                print(f"  Elevation range: {np.nanmin(self.dem_data):.1f} to "
                      f"{np.nanmax(self.dem_data):.1f} m")

        except Exception as e:
            print(f"Error loading DEM: {e}")
            self.dem_data = None

    def latlon_to_dem_coords(self, lat: float, lon: float) -> Tuple[Optional[int], Optional[int]]:
        """
        Convert latitude/longitude to DEM pixel coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            row, col: DEM pixel coordinates, or (None, None) if outside bounds
        """
        if self.dem_data is None or not RASTERIO_AVAILABLE:
            return None, None

        try:
            # Use cached Proj objects if available (much faster than creating new ones)
            if self.wgs84_proj is None or self.dem_proj is None:
                # Fallback: create Proj objects if not cached
                wgs84 = Proj(init='epsg:4326')
                dem_proj = Proj(self.dem_crs)
            else:
                wgs84 = self.wgs84_proj
                dem_proj = self.dem_proj

            # Transform coordinates
            x, y = pyproj_transform(wgs84, dem_proj, lon, lat)

            # Convert to pixel coordinates
            col, row = rowcol(self.dem_transform, x, y)

            # Check bounds
            if 0 <= row < self.dem_data.shape[0] and 0 <= col < self.dem_data.shape[1]:
                return row, col
            else:
                return None, None

        except Exception:
            return None, None

    def get_terrain_elevation(self, lat: float, lon: float) -> Optional[float]:
        """
        Get terrain elevation at given latitude/longitude.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            elevation: Terrain elevation in meters, or None if outside DEM bounds
        """
        row, col = self.latlon_to_dem_coords(lat, lon)
        if row is not None and col is not None:
            return float(self.dem_data[row, col])
        return None

    def ecef_to_latlon(self, ecef: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert ECEF coordinates to latitude, longitude, altitude.

        Args:
            ecef: ECEF position [x, y, z] in meters

        Returns:
            lat, lon, alt: Latitude (degrees), longitude (degrees), altitude (meters)
        """
        x, y, z = ecef[0], ecef[1], ecef[2]

        # Calculate longitude
        lon = np.arctan2(y, x)

        # Calculate latitude using iterative method
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p)

        # Iterative refinement for latitude
        for _ in range(5):
            sin_lat = np.sin(lat)
            N = R_earth / np.sqrt(1 - 0.00669437999014 * sin_lat**2)  # WGS84 flattening
            alt = p / np.cos(lat) - N
            lat_new = np.arctan2(z, p * (1 - 0.00669437999014 * N / (N + alt)))
            if abs(lat_new - lat) < 1e-10:
                break
            lat = lat_new

        # Calculate altitude
        sin_lat = np.sin(lat)
        N = R_earth / np.sqrt(1 - 0.00669437999014 * sin_lat**2)
        alt = p / np.cos(lat) - N

        return np.degrees(lat), np.degrees(lon), alt

    def ecef_to_latlon_vectorized(
        self, ecef_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized conversion of ECEF coordinates to latitude, longitude, altitude.

        Args:
            ecef_array: ECEF positions [n, 3] in meters

        Returns:
            lats, lons, alts: Arrays of latitude (degrees), longitude (degrees), altitude (meters)
        """
        # Ensure input is 2D [n, 3]
        ecef_array = np.asarray(ecef_array, dtype=np.float64)
        if ecef_array.ndim == 1:
            ecef_array = ecef_array.reshape(1, 3)

        x = ecef_array[:, 0]
        y = ecef_array[:, 1]
        z = ecef_array[:, 2]

        # Calculate longitude (vectorized)
        lons = np.arctan2(y, x)

        # Calculate latitude using iterative method (vectorized)
        p = np.sqrt(x**2 + y**2)
        lats = np.arctan2(z, p)

        # Iterative refinement for latitude (vectorized)
        for _ in range(5):
            sin_lat = np.sin(lats)
            N = R_earth / np.sqrt(1 - 0.00669437999014 * sin_lat**2)
            alts = p / np.cos(lats) - N
            lats_new = np.arctan2(z, p * (1 - 0.00669437999014 * N / (N + alts)))
            # Check convergence (vectorized)
            diff = np.abs(lats_new - lats)
            if np.all(diff < 1e-10):
                break
            lats = lats_new

        # Calculate altitude (vectorized)
        sin_lat = np.sin(lats)
        N = R_earth / np.sqrt(1 - 0.00669437999014 * sin_lat**2)
        alts = p / np.cos(lats) - N

        return np.degrees(lats), np.degrees(lons), alts

    def check_line_of_sight_dem(
        self,
        emitter_ecef: np.ndarray,
        weather_sat_ecef: np.ndarray,
        emitter_alt_above_ground: float = 0.0,
        num_points: int = 10
    ) -> bool:
        """
        Check if ground emitter is visible from weather satellite using DEM ray tracing.

        This performs optimized ray tracing from weather satellite to ground emitter, checking
        if terrain blocks the line-of-sight. Uses adaptive sampling (dense near ground).

        Args:
            emitter_ecef: Ground emitter position in ECEF [x, y, z] (meters)
            weather_sat_ecef: Weather satellite position in ECEF [x, y, z] (meters)
            emitter_alt_above_ground: Emitter antenna height above ground (meters)
            num_points: Number of points to sample along the ray (default: 10, optimized for speed)

        Returns:
            bool: True if emitter is visible (not blocked by terrain), False otherwise
        """
        # If no DEM available, fall back to geometric horizon check
        if self.dem_data is None:
            return self.check_horizon_visibility_geometric(emitter_ecef, weather_sat_ecef)

        # Convert ECEF to lat/lon for both positions
        ws_lat, ws_lon, ws_alt = self.ecef_to_latlon(weather_sat_ecef)
        emitter_lat, emitter_lon, emitter_alt_ecef = self.ecef_to_latlon(emitter_ecef)

        # Get terrain elevation at emitter location
        emitter_terrain_elev = self.get_terrain_elevation(emitter_lat, emitter_lon)
        if emitter_terrain_elev is None:
            # Outside DEM bounds, use geometric check
            return self.check_horizon_visibility_geometric(emitter_ecef, weather_sat_ecef)

        # Calculate distance from weather satellite to emitter
        sat_to_emitter_vec = emitter_ecef - weather_sat_ecef
        sat_to_emitter_dist = np.linalg.norm(sat_to_emitter_vec)

        # Adaptive sampling: sample more densely near the ground (emitter end)
        # Use cubic spacing to focus on ground region where blocking occurs
        # Most blocking happens in the lower 10% of the path
        t_linear = np.linspace(0.0, 1.0, num_points)
        # Cubic spacing: even more points near t=1 (ground end) for better accuracy with fewer points
        t_adaptive = t_linear ** 3
        distances = t_adaptive * sat_to_emitter_dist

        # For weather satellite looking down, we need to check if terrain
        # between satellite and emitter blocks the line-of-sight
        # OPTIMIZATION: Vectorize sample point calculations for better performance
        # Skip endpoints (first and last points)
        valid_indices = np.arange(1, len(distances) - 1)
        if len(valid_indices) == 0:
            return True  # Not enough points to check

        dist_ratios = distances[valid_indices] / sat_to_emitter_dist

        # Vectorized: Calculate all sample positions along ray at once
        sample_ecef_array = weather_sat_ecef + dist_ratios[:, np.newaxis] * sat_to_emitter_vec

        # Vectorized: Convert all sample points to lat/lon at once
        sample_lats, sample_lons, sample_alts_ecef = self.ecef_to_latlon_vectorized(sample_ecef_array)

        # Check terrain elevations for all sample points
        # (Still need to iterate for terrain lookups, but coordinate conversion is vectorized)
        for i in range(len(valid_indices)):
            # Get terrain elevation at this point
            sample_terrain_elev = self.get_terrain_elevation(sample_lats[i], sample_lons[i])
            if sample_terrain_elev is None:
                continue  # Outside DEM bounds, skip this point

            # Calculate altitude of ray at this point
            sample_alt = sample_alts_ecef[i] - R_earth  # Height above sea level

            # Check if terrain height exceeds ray height
            # If terrain is above the ray, it blocks the line-of-sight
            # Early termination: exit immediately if blocked
            if sample_terrain_elev > sample_alt:
                return False  # Terrain blocks the view (early termination)

        return True  # No terrain blocking found

    def check_horizon_visibility_geometric(
        self,
        emitter_ecef: np.ndarray,
        weather_sat_ecef: np.ndarray,
        earth_radius: float = R_earth
    ) -> bool:
        """
        Check if ground emitter is visible from weather satellite (geometric horizon only).

        This is a fallback method when DEM is not available.

        Args:
            emitter_ecef: Emitter position in ECEF [x, y, z] (meters)
            weather_sat_ecef: Weather satellite position in ECEF [x, y, z] (meters)
            earth_radius: Earth radius in meters (default: WGS84)

        Returns:
            bool: True if emitter is above geometric horizon (visible), False otherwise
        """
        # Vector from Earth center to weather satellite
        ws_vec = weather_sat_ecef
        ws_distance = np.linalg.norm(ws_vec)

        # Vector from Earth center to emitter
        emitter_vec = emitter_ecef

        # Vector from weather satellite to emitter
        sat_to_emitter = emitter_vec - ws_vec
        sat_to_emitter_dist = np.linalg.norm(sat_to_emitter)

        # Calculate angle between satellite-to-Earth-center and satellite-to-emitter
        # Use cosine law
        cos_angle = np.dot(-ws_vec, sat_to_emitter) / (ws_distance * sat_to_emitter_dist)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Calculate horizon angle (angle from nadir to horizon)
        # From satellite, horizon is at angle where line-of-sight is tangent to Earth
        sin_horizon_angle = earth_radius / ws_distance
        horizon_angle = np.arcsin(np.clip(sin_horizon_angle, 0.0, 1.0))

        # Emitter is visible if angle from nadir is less than horizon angle
        # (i.e., emitter is above horizon)
        return angle < horizon_angle


def check_horizon_visibility(
    emitter_ecef: np.ndarray,
    weather_sat_ecef: np.ndarray,
    dem_masker: Optional[DEMTerrainMasker] = None,
    emitter_alt_above_ground: float = 0.0,
    earth_radius: float = R_earth
) -> bool:
    """
    Check if ground emitter is visible from weather satellite.

    Uses two-stage checking: fast geometric check first, then DEM-based terrain
    masking if available. This provides significant speedup by filtering out
    obviously blocked emitters before expensive DEM ray tracing.

    Args:
        emitter_ecef: Emitter position in ECEF [x, y, z] (meters)
        weather_sat_ecef: Weather satellite position in ECEF [x, y, z] (meters)
        dem_masker: DEMTerrainMasker object (optional, for DEM-based checking)
        emitter_alt_above_ground: Emitter antenna height above ground (meters)
        earth_radius: Earth radius in meters (default: WGS84)

    Returns:
        bool: True if emitter is visible, False otherwise
    """
    # Stage 1: Quick geometric horizon check (fast pre-filter)
    # This filters out obviously blocked emitters before expensive DEM ray tracing
    dem_masker_temp = DEMTerrainMasker()
    if not dem_masker_temp.check_horizon_visibility_geometric(
        emitter_ecef,
        weather_sat_ecef,
        earth_radius
    ):
        return False  # Failed geometric check, no need for DEM

    # Stage 2: DEM-based terrain masking (only if DEM available and passed geometric check)
    if dem_masker is not None and dem_masker.dem_data is not None:
        return dem_masker.check_line_of_sight_dem(
            emitter_ecef,
            weather_sat_ecef,
            emitter_alt_above_ground
        )
    else:
        # No DEM available, geometric check passed
        return True


def calculate_atmospheric_loss(
    distance: float,
    freq: float,
    elevation_angle: float = None
) -> float:
    """
    Calculate atmospheric absorption loss for ground-to-space path.

    Simplified model based on ITU-R P.676:
    - Oxygen absorption: significant at 50-60 GHz
    - Water vapor absorption: significant at 22 GHz and above

    Args:
        distance: Path length in meters
        freq: Frequency in Hz
        elevation_angle: Elevation angle from ground (degrees, optional)

    Returns:
        float: Atmospheric loss factor (linear, > 1.0)
    """
    freq_ghz = freq / 1e9

    # Simplified atmospheric absorption model
    # Based on ITU-R P.676 (approximate)
    if freq_ghz < 20:
        # Low frequencies: minimal absorption
        absorption_db_per_km = 0.01
    elif freq_ghz < 40:
        # K-Band (20-40 GHz): water vapor absorption
        # Typical: 0.1-0.5 dB/km depending on conditions
        absorption_db_per_km = 0.2 + 0.3 * ((freq_ghz - 20) / 20.0)
    elif freq_ghz < 60:
        # V-Band (40-60 GHz): strong oxygen absorption
        # Approaching 60 GHz oxygen line, absorption increases dramatically
        oxygen_band_center = 60.0
        distance_from_peak = abs(freq_ghz - oxygen_band_center)
        if distance_from_peak < 10:
            # Near oxygen line: very high absorption
            absorption_db_per_km = 10.0 - 8.0 * (distance_from_peak / 10.0)
        else:
            # Far from oxygen line, but still significant
            absorption_db_per_km = 2.0 + 1.0 * ((freq_ghz - 40) / 20.0)
    else:
        # Above 60 GHz: very strong oxygen absorption
        absorption_db_per_km = 15.0

    # Path length through atmosphere depends on elevation angle
    # For space-to-ground links, most of path is in VACUUM
    # Only the lower ~20-30 km is in the atmosphere
    # Use atmospheric path length, not total distance
    atmospheric_path_km = 25.0  # Approximate atmospheric path length (km)
    if elevation_angle is not None:
        # Longer path at low elevation angles
        elev_rad = np.deg2rad(elevation_angle)
        effective_path_multiplier = 1.0 / max(np.sin(elev_rad), 0.1)
        # Limit multiplier for reasonable values
        effective_path_multiplier = min(effective_path_multiplier, 3.0)
    else:
        effective_path_multiplier = 1.0

    # Total atmospheric loss (only over atmospheric portion of path)
    total_loss_db = absorption_db_per_km * atmospheric_path_km * effective_path_multiplier

    # Convert to linear loss factor
    loss_factor = 10 ** (total_loss_db / 10.0)

    return loss_factor


def calculate_ground_emitter_harmonic_contribution(
    base_frequency: float,
    base_link_budget: float,
    harmonics: list,
    observation_frequency: float,
    observation_bandwidth: float,
    emitter_distance: float
) -> float:
    """
    Calculate harmonic contribution from ground emitter (5G) at observation frequency.

    Args:
        base_frequency: Ground emitter fundamental frequency (Hz)
        base_link_budget: Link budget at fundamental frequency (dimensionless)
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples.
            Example: [(2.0, 0.01), (3.0, 0.003), (4.0, 0.001)]
            - frequency_multiplier: Harmonic order (2.0 = 2nd harmonic)
            - power_reduction_factor: Power relative to fundamental (linear, 0-1)
        observation_frequency: Observation frequency (Hz)
        observation_bandwidth: Observation bandwidth (Hz)
        emitter_distance: Distance from weather sat to emitter (meters)

    Returns:
        float: Harmonic link budget contribution (dimensionless)
    """
    harmonic_contribution = 0.0

    # Calculate path loss at fundamental frequency
    L_fundamental = free_space_loss(emitter_distance, base_frequency)

    for freq_mult, power_red in harmonics:
        harmonic_frequency = base_frequency * freq_mult

        # Check if harmonic falls within observation band
        freq_min = observation_frequency - observation_bandwidth / 2
        freq_max = observation_frequency + observation_bandwidth / 2

        if freq_min <= harmonic_frequency <= freq_max:
            # Calculate path loss at harmonic frequency (frequency-dependent)
            L_harmonic = free_space_loss(emitter_distance, harmonic_frequency)

            # Harmonic link budget accounts for:
            # 1. Power reduction factor (harmonic suppression)
            # 2. Path loss ratio (harmonic frequency vs fundamental)
            path_loss_ratio = L_fundamental / L_harmonic
            harmonic_contribution += base_link_budget * power_red * path_loss_ratio

    return harmonic_contribution


def calculate_ground_emitter_oobe_contribution(
    base_frequency: float,
    base_link_budget: float,
    observation_frequency: float,
    observation_bandwidth: float,
    emitter_distance: float,
    oobe_suppression_db: float,
    oobe_freq_offset_max: float,
    include_atmospheric_loss: bool = True,
    elevation_angle: float = None
) -> float:
    """
    Calculate out-of-band emission (OOBE) contribution from ground emitter.

    OOBE occurs when the fundamental frequency is close to but outside the
    observation bandwidth. Emissions leak into adjacent bands due to imperfect
    filtering.

    Args:
        base_frequency: Ground emitter fundamental frequency (Hz)
        base_link_budget: Link budget at fundamental frequency (dimensionless)
        observation_frequency: Observation frequency (Hz)
        observation_bandwidth: Observation bandwidth (Hz)
        emitter_distance: Distance from weather sat to emitter (meters)
        oobe_suppression_db: OOBE suppression level in dB relative to in-band power
            (e.g., -40 dB means OOBE is 40 dB below in-band power)
        oobe_freq_offset_max: Maximum frequency offset for OOBE consideration (Hz)
            (e.g., 500e6 for 500 MHz)
        include_atmospheric_loss: Whether to include atmospheric loss
        elevation_angle: Elevation angle from ground (degrees)

    Returns:
        float: OOBE link budget contribution (dimensionless), 0.0 if not applicable
    """
    # Calculate observation bandwidth bounds
    freq_min = observation_frequency - observation_bandwidth / 2
    freq_max = observation_frequency + observation_bandwidth / 2

    # Check if fundamental is outside observation band but within OOBE range
    if freq_min <= base_frequency <= freq_max:
        # Fundamental is in-band, OOBE not applicable
        return 0.0

    # Calculate frequency offset from observation band
    if base_frequency < freq_min:
        freq_offset = freq_min - base_frequency
    else:  # base_frequency > freq_max
        freq_offset = base_frequency - freq_max

    # Check if within OOBE range
    if freq_offset > oobe_freq_offset_max:
        # Too far from observation band, OOBE negligible
        return 0.0

    # Calculate path loss at observation frequency (not fundamental)
    L_fs_obs = free_space_loss(emitter_distance, observation_frequency)
    L_fs_fund = free_space_loss(emitter_distance, base_frequency)

    # Atmospheric loss at observation frequency
    if include_atmospheric_loss:
        L_atm_obs = calculate_atmospheric_loss(
            emitter_distance, observation_frequency, elevation_angle
        )
        L_atm_fund = calculate_atmospheric_loss(
            emitter_distance, base_frequency, elevation_angle
        )
    else:
        L_atm_obs = 1.0
        L_atm_fund = 1.0

    # OOBE link budget accounts for:
    # 1. OOBE suppression factor (relative to in-band power)
    # 2. Path loss ratio (observation frequency vs fundamental)
    # 3. Atmospheric loss ratio (observation frequency vs fundamental)
    path_loss_ratio = L_fs_fund / L_fs_obs
    atm_loss_ratio = L_atm_fund / L_atm_obs
    # oobe_suppression_db is already negative (e.g., -50.0), so don't negate it
    oobe_suppression_linear = 10 ** (oobe_suppression_db / 10.0)

    oobe_contribution = (base_link_budget *
                         oobe_suppression_linear *
                         path_loss_ratio *
                         atm_loss_ratio)

    return oobe_contribution


def ground_emitter_to_weather_sat_link_budget(
    emitter_ecef: np.ndarray,
    weather_sat_ecef: np.ndarray,
    weather_sat_antenna: Antenna,
    emitter_antenna: Antenna,
    freq: float,
    include_atmospheric_loss: bool = True,
    emitter_fundamental_freq: float = None,
    harmonics: list = None,
    observation_bandwidth: float = None,
    oobe_suppression_db: float = None,
    oobe_freq_offset_max: float = None
) -> float:
    """
    Calculate link budget for ground emitter (5G) to weather satellite.

    Args:
        emitter_ecef: Ground emitter position in ECEF [x, y, z] (meters)
        weather_sat_ecef: Weather satellite position in ECEF [x, y, z] (meters)
        weather_sat_antenna: Weather satellite antenna object
        emitter_antenna: Ground emitter antenna object
        freq: Observation frequency in Hz
        include_atmospheric_loss: Whether to include atmospheric absorption loss
        emitter_fundamental_freq: Ground emitter fundamental frequency (Hz). If None,
            interference is calculated at observation frequency (legacy behavior).
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples.
            Example: [(2.0, 0.01), (3.0, 0.003), (4.0, 0.001)]
        observation_bandwidth: Observation bandwidth (Hz). Required if harmonics
            or OOBE are provided.
        oobe_suppression_db: OOBE suppression level in dB relative to in-band power
            (e.g., -40 dB). If None, OOBE is not considered.
        oobe_freq_offset_max: Maximum frequency offset for OOBE consideration (Hz).
            If None, OOBE is not considered.

    Returns:
        float: Link budget (dimensionless)
    """
    # Transform emitter to weather satellite frame
    # Note: velocity is needed for coordinate transformation, but for ground emitters
    # we can use a zero velocity placeholder since the transformation primarily
    # depends on position
    weather_sat_velocity_ecef = np.array([0.0, 0.0, 0.0])
    emitter_dec, emitter_caz = ecef_to_weather_sat_frame(
        emitter_ecef[np.newaxis, :],
        weather_sat_ecef[np.newaxis, :],
        weather_sat_velocity_ecef[np.newaxis, :]
    )
    emitter_dec = emitter_dec[0]
    emitter_caz = emitter_caz[0]

    # Weather satellite receive gain (toward emitter)
    gain_weather_sat = weather_sat_antenna.get_gain_value(emitter_dec, emitter_caz)

    # Ground emitter transmit gain (toward weather satellite)
    # For emitter, we need angle from emitter's boresight to weather satellite
    # Calculate distance
    sat_to_emitter_vec = weather_sat_ecef - emitter_ecef
    sat_to_emitter_dist = np.linalg.norm(sat_to_emitter_vec)

    # For emitter antenna pattern, we need elevation and azimuth relative to emitter
    # The antenna pattern uses:
    # - alpha: angle from z-axis (0 = up, 90 = horizontal, 180 = down)
    # - beta: azimuth angle (0-360 degrees)
    #
    # From weather satellite frame:
    # - emitter_dec: angle from nadir (0 = nadir/down, π/2 = horizon) in radians
    # - emitter_caz: counter-azimuth in radians
    #
    # For ground emitter looking up at weather satellite:
    # - If satellite is at nadir from weather sat perspective (dec=0),
    #   emitter sees it overhead (alpha=0, pointing up)
    # - If satellite is at horizon from weather sat perspective (dec=π/2),
    #   emitter sees it at horizon (alpha=90, pointing horizontally)
    #
    # So: alpha_emitter = emitter_dec (direct mapping in radians)
    # Note: get_gain_value expects angles in RADIANS (matching the interpolator)
    # The interpolator was created with radians from map_sphere conversion

    # Get valid angle ranges from antenna pattern (these are in degrees)
    valid_alphas, valid_betas = emitter_antenna.get_def_angles()
    alpha_min_deg, alpha_max_deg = valid_alphas.min(), valid_alphas.max()
    beta_min_deg, beta_max_deg = valid_betas.min(), valid_betas.max()

    # Convert to radians for clamping
    alpha_min_rad = np.deg2rad(alpha_min_deg)
    alpha_max_rad = np.deg2rad(alpha_max_deg)
    beta_min_rad = np.deg2rad(beta_min_deg)
    beta_max_rad = np.deg2rad(beta_max_deg)

    # Clamp emitter_dec to valid alpha range (in radians)
    emitter_alpha_rad = np.clip(emitter_dec, alpha_min_rad, alpha_max_rad)

    # Beta: wrap to valid range and clamp
    emitter_beta_rad = emitter_caz % (2 * np.pi)
    if emitter_beta_rad < 0:
        emitter_beta_rad += 2 * np.pi
    # Wrap beta to [0, 2π) range, but the interpolator might expect [0, 2π]
    # Check if beta_max includes 360 degrees (2π)
    if beta_max_deg >= 360.0:
        # Pattern includes full 360, so wrap is fine
        emitter_beta_rad = np.clip(emitter_beta_rad, beta_min_rad, beta_max_rad)
    else:
        # Pattern might not include full 360, need to handle wrapping
        # For now, clamp to valid range
        emitter_beta_rad = np.clip(emitter_beta_rad, beta_min_rad, beta_max_rad)

    # Get emitter antenna gain (angles in radians)
    # Pattern stores absolute gain values, but EIRP already includes peak gain
    # So we need to normalize to relative pattern for link budget
    gain_emitter_absolute = emitter_antenna.get_gain_value(
        emitter_alpha_rad,
        emitter_beta_rad
    )
    # Normalize to relative pattern (since EIRP already includes peak gain)
    peak_gain_emitter = emitter_antenna.get_boresight_gain()
    gain_emitter = gain_emitter_absolute / peak_gain_emitter

    # Check if we have fundamental frequency and bandwidth information
    if (emitter_fundamental_freq is not None and observation_bandwidth is not None):
        # Calculate observation bandwidth bounds
        freq_min = freq - observation_bandwidth / 2
        freq_max = freq + observation_bandwidth / 2

        # Calculate base link budget at fundamental frequency
        L_fs_fund = free_space_loss(sat_to_emitter_dist, emitter_fundamental_freq)

        # Atmospheric loss at fundamental frequency
        if include_atmospheric_loss:
            elevation_angle = np.rad2deg(np.pi / 2 - emitter_alpha_rad)
            L_atm_fund = calculate_atmospheric_loss(
                sat_to_emitter_dist, emitter_fundamental_freq, elevation_angle
            )
        else:
            L_atm_fund = 1.0

        base_link_budget_fund = (gain_weather_sat * (1.0 / L_fs_fund) *
                                 (1.0 / L_atm_fund) * gain_emitter)

        # Check if fundamental falls within observation bandwidth
        if freq_min <= emitter_fundamental_freq <= freq_max:
            # Fundamental is within observation band - include it
            link_budget = base_link_budget_fund
        else:
            # Fundamental is outside observation band - no base contribution
            link_budget = 0.0

        # Add harmonic contributions if harmonics are provided
        if harmonics is not None:
            harmonic_contribution = calculate_ground_emitter_harmonic_contribution(
                emitter_fundamental_freq,
                base_link_budget_fund,
                harmonics,
                freq,
                observation_bandwidth,
                sat_to_emitter_dist
            )
            # Add harmonic contribution
            link_budget += harmonic_contribution

        # Add OOBE contribution if OOBE parameters are provided
        if (oobe_suppression_db is not None and
                oobe_freq_offset_max is not None):
            elevation_angle = None
            if include_atmospheric_loss:
                elevation_angle = np.rad2deg(np.pi / 2 - emitter_alpha_rad)
            oobe_contribution = calculate_ground_emitter_oobe_contribution(
                emitter_fundamental_freq,
                base_link_budget_fund,
                freq,
                observation_bandwidth,
                sat_to_emitter_dist,
                oobe_suppression_db,
                oobe_freq_offset_max,
                include_atmospheric_loss,
                elevation_angle
            )
            # Add OOBE contribution
            link_budget += oobe_contribution
    else:
        # No fundamental frequency info: calculate base link budget at observation frequency
        # (legacy behavior - may not be physically correct if frequencies don't match)
        L_fs = free_space_loss(sat_to_emitter_dist, freq)

        # Atmospheric loss
        if include_atmospheric_loss:
            elevation_angle = np.rad2deg(np.pi / 2 - emitter_alpha_rad)
            L_atm = calculate_atmospheric_loss(sat_to_emitter_dist, freq, elevation_angle)
        else:
            L_atm = 1.0

        # Total link budget
        link_budget = gain_weather_sat * (1.0 / L_fs) * (1.0 / L_atm) * gain_emitter

    return link_budget


def model_weather_sat_observed_power_phase2(
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
    ground_emitters: pd.DataFrame = None,
    ground_emitter_antenna: Antenna = None,
    ground_emitter_eirp_dbw: float = 30.0,
    earth_brightness_temp: float = 280.0,
    sky_brightness_temp: float = 2.73,
    system_temp: float = 300.0,
    starlink_eirp_dbw: float = 40.0,
    enable_terrain_masking: bool = True,
    include_atmospheric_loss: bool = True,
    dem_file: Optional[str] = None,
    polarization_loss_factor: float = 0.5,
    starlink_fundamental_freq: float = None,
    harmonics: list = None,
    ground_emitter_fundamental_freq: float = None,
    ground_emitter_harmonics: list = None,
    ground_emitter_oobe_suppression_db: float = None,
    ground_emitter_oobe_freq_offset_max: float = None
) -> dict:
    """
    Model observed power at weather satellite from all RFI sources (Phase 2: includes ground emitters).

    This extends Phase 1 to include ground emitter (5G) interference with DEM-based terrain masking.

    Args:
        weather_sat_trajectory: Weather satellite trajectory
        weather_sat_instrument: Weather satellite instrument
        starlink_constellation: Starlink constellation
        observation_times: Array of observation timestamps
        observer_lat: Observer latitude (degrees)
        observer_lon: Observer longitude (degrees)
        observer_alt: Observer altitude (meters)
        target_lat: Target latitude (degrees) - center of resolution element
        target_lon: Target longitude (degrees)
        target_alt: Target altitude (meters)
        freq_channels: Frequency channels in Hz
        ground_emitters: DataFrame with ground emitter positions (columns: 'lat', 'lon', 'alt')
        ground_emitter_antenna: Ground emitter antenna object
        ground_emitter_eirp_dbw: Ground emitter EIRP in dBW
        earth_brightness_temp: Earth brightness temperature (K)
        sky_brightness_temp: Sky background temperature (K)
        system_temp: System temperature (K)
        starlink_eirp_dbw: Starlink EIRP in dBW
        enable_terrain_masking: Whether to check horizon visibility for ground emitters
        include_atmospheric_loss: Whether to include atmospheric absorption loss
        dem_file: Path to DEM GeoTIFF file for terrain masking (optional)
        polarization_loss_factor: Polarization mismatch loss factor (linear, 0-1).
            Default 0.5 = -3 dB for circular (Starlink) to linear (Suomi-NPP) mismatch.
        starlink_fundamental_freq: Starlink fundamental frequency (Hz). If None,
            harmonics are not calculated.
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples.
            Example: [(2.0, 0.01), (3.0, 0.003), (4.0, 0.001)]
        ground_emitter_fundamental_freq: Ground emitter fundamental frequency (Hz). If None,
            interference is calculated at observation frequency (legacy behavior).
        ground_emitter_harmonics: List of (frequency_multiplier, power_reduction_factor) tuples
            for ground emitter harmonics. Example: [(2.0, 0.01), (7.0, 0.0001), (14.0, 0.00001)]

    Returns:
        dict: Dictionary with total and individual components of observed power in dBW.
              Keys: 'total', 'starlink', 'ground_emitter', 'earth', 'sky', 'system'.
              Values: np.ndarray of shape (n_times, n_freqs)
    """
    n_times = len(observation_times)
    n_freqs = len(freq_channels)

    # Initialize DEM terrain masker if DEM file is provided
    dem_masker = None
    if enable_terrain_masking and dem_file is not None:
        dem_masker = DEMTerrainMasker(dem_file)
        if dem_masker.dem_data is None:
            print("  Warning: DEM loading failed, using geometric horizon check only")
    elif enable_terrain_masking:
        print("  Note: DEM file not provided, using geometric horizon check only")

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

    # Initialize result arrays for total and individual components
    result_power = np.zeros((n_times, n_freqs))  # Will store total power in W
    result_starlink = np.zeros((n_times, n_freqs))  # Starlink interference
    result_ground_emitter = np.zeros((n_times, n_freqs))  # Ground emitter interference
    result_earth = np.zeros((n_times, n_freqs))  # Earth brightness
    result_sky = np.zeros((n_times, n_freqs))  # Sky background
    result_system = np.zeros((n_times, n_freqs))  # System noise

    print(f"  Processing {n_times} time steps and {n_freqs} frequency channels...")
    print(f"  Total Starlink satellites in constellation: {starlink_traj_df['sat'].nunique()}")
    if ground_emitters is not None:
        print(f"  Total ground emitters: {len(ground_emitters)}")

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
            ground_emitter_interference_temp = 0.0

            # Process each Starlink satellite
            if n_visible_starlinks > 0:
                for _, sat_row in starlink_sats.iterrows():
                    # Get Starlink position from trajectory
                    sat_dist = sat_row['distances']
                    # Trajectory 'elevations' = altitude above horizon; ENU uses polar from zenith
                    sat_polar_zenith_deg = 90.0 - sat_row['elevations']
                    sat_elev = np.deg2rad(sat_polar_zenith_deg)
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

            # Process ground emitters (Phase 2) - OPTIMIZED VECTORIZED VERSION
            if ground_emitters is not None and ground_emitter_antenna is not None:
                # Pre-cache terrain elevations for all emitters (once per emitter, reused across time steps)
                # This is done outside the time loop for efficiency
                if t_idx == 0:
                    # Cache terrain elevations for all emitters (only compute once)
                    n_emitters = len(ground_emitters)
                    emitter_terrain_elevations = np.zeros(n_emitters)
                    if dem_masker is not None and dem_masker.dem_data is not None:
                        for i, (_, emitter_row) in enumerate(ground_emitters.iterrows()):
                            terrain_elev = dem_masker.get_terrain_elevation(
                                emitter_row['lat'],
                                emitter_row['lon']
                            )
                            if terrain_elev is not None:
                                emitter_terrain_elevations[i] = terrain_elev
                    # Store in ground_emitters DataFrame for reuse
                    ground_emitters['terrain_elev'] = emitter_terrain_elevations
                else:
                    # Retrieve cached terrain elevations
                    emitter_terrain_elevations = ground_emitters['terrain_elev'].values

                # Vectorized processing: convert all emitter positions to ECEF at once
                emitter_lats = ground_emitters['lat'].values
                emitter_lons = ground_emitters['lon'].values
                emitter_alts_above_ground = ground_emitters['alt'].values
                emitter_total_alts = emitter_terrain_elevations + emitter_alts_above_ground
                emitter_ecef_all = latlonalt_to_ecef_vectorized(
                    emitter_lats, emitter_lons, emitter_total_alts
                )
                # Ensure emitter_ecef_all has shape [n_emitters, 3]
                emitter_ecef_all = np.asarray(emitter_ecef_all, dtype=np.float64)
                if emitter_ecef_all.ndim == 1:
                    emitter_ecef_all = emitter_ecef_all.reshape(1, 3)
                elif emitter_ecef_all.ndim == 2 and emitter_ecef_all.shape[1] != 3:
                    emitter_ecef_all = emitter_ecef_all.reshape(-1, 3)

                # Vectorized visibility check (geometric horizon check first, then DEM for visible ones)
                n_emitters = len(ground_emitters)
                visibility_mask = np.ones(n_emitters, dtype=bool)

                if enable_terrain_masking:
                    # Stage 1: Vectorized geometric horizon check for all emitters
                    # Ensure proper array shapes for broadcasting
                    ws_vec = np.asarray(ws_ecef, dtype=np.float64).flatten()
                    if len(ws_vec) != 3:
                        raise ValueError(f"ws_ecef must have 3 elements, got {len(ws_vec)}")
                    ws_vec = ws_vec.reshape(1, 3)  # [1, 3] for broadcasting
                    ws_distance = float(np.linalg.norm(ws_vec))  # Ensure scalar

                    # Ensure emitter_ecef_all has shape [n_emitters, 3]
                    emitter_vecs = np.asarray(emitter_ecef_all, dtype=np.float64)
                    if emitter_vecs.ndim == 1:
                        # Single emitter case: [3] -> [1, 3]
                        if len(emitter_vecs) == 3:
                            emitter_vecs = emitter_vecs.reshape(1, 3)
                        else:
                            emitter_vecs = emitter_vecs.reshape(-1, 3)
                    elif emitter_vecs.ndim == 2:
                        if emitter_vecs.shape[1] != 3:
                            emitter_vecs = emitter_vecs.reshape(-1, 3)
                    else:
                        raise ValueError(f"Unexpected emitter_vecs shape: {emitter_vecs.shape}")

                    # Vectors from weather satellite to all emitters
                    # Broadcasting: [n_emitters, 3] - [1, 3] = [n_emitters, 3]
                    sat_to_emitter_vecs = emitter_vecs - ws_vec
                    # Ensure result is 2D with shape [n_emitters, 3]
                    if sat_to_emitter_vecs.ndim == 1:
                        sat_to_emitter_vecs = sat_to_emitter_vecs.reshape(1, 3)
                    elif sat_to_emitter_vecs.ndim == 2 and sat_to_emitter_vecs.shape[1] != 3:
                        sat_to_emitter_vecs = sat_to_emitter_vecs.reshape(-1, 3)
                    # Ensure it's contiguous for better performance
                    sat_to_emitter_vecs = np.ascontiguousarray(sat_to_emitter_vecs)
                    sat_to_emitter_dists = np.linalg.norm(sat_to_emitter_vecs, axis=1)

                    # Calculate angles between satellite-to-Earth-center and satellite-to-emitter (vectorized)
                    # Use dot product: cos(angle) = (-ws_vec · sat_to_emitter) / (|ws_vec| * |sat_to_emitter|)
                    # ws_vec is [1, 3], sat_to_emitter_vecs is [n_emitters, 3]
                    # Broadcasting: -ws_vec * sat_to_emitter_vecs gives [n_emitters, 3]
                    dot_products = np.sum(-ws_vec * sat_to_emitter_vecs, axis=1)
                    cos_angles = dot_products / (ws_distance * sat_to_emitter_dists)
                    cos_angles = np.clip(cos_angles, -1.0, 1.0)
                    angles = np.arccos(cos_angles)

                    # Calculate horizon angle (angle from nadir to horizon)
                    sin_horizon_angle = R_earth / ws_distance
                    horizon_angle = np.arcsin(np.clip(sin_horizon_angle, 0.0, 1.0))

                    # Emitter is visible if angle from nadir is less than horizon angle
                    visibility_mask = angles < horizon_angle
                    n_geometric_visible = np.sum(visibility_mask)

                    # Stage 2: DEM ray tracing only for emitters that passed geometric check
                    # OPTIMIZATION: Skip DEM ray tracing for high-elevation emitters
                    # Emitters at high elevation angles (>30°) are very unlikely to be blocked
                    # by terrain, so we can skip expensive DEM ray tracing for them
                    n_dem_checked = 0
                    n_dem_blocked = 0
                    n_high_elevation = 0

                    if dem_masker is not None and dem_masker.dem_data is not None:
                        visible_indices = np.where(visibility_mask)[0]
                        if len(visible_indices) > 0:
                            # Calculate elevation angles for visible emitters (vectorized)
                            # Elevation angle = 90° - declination angle (from nadir)
                            visible_angles_deg = np.rad2deg(angles[visible_indices])
                            elevation_angles_deg = 90.0 - visible_angles_deg

                            # High elevation threshold: skip DEM for emitters above 30°
                            high_elevation_threshold_deg = 30.0
                            high_elevation_mask = elevation_angles_deg > high_elevation_threshold_deg
                            n_high_elevation = np.sum(high_elevation_mask)

                            # For high-elevation emitters, assume visible (skip DEM ray tracing)
                            # For low-elevation emitters, perform DEM ray tracing
                            low_elevation_indices = visible_indices[~high_elevation_mask]
                            n_dem_checked = len(low_elevation_indices)

                            # Only perform DEM ray tracing for low-elevation emitters
                            for i in low_elevation_indices:
                                if not dem_masker.check_line_of_sight_dem(
                                    emitter_ecef_all[i],
                                    ws_ecef,
                                    emitter_alts_above_ground[i],
                                    num_points=10  # Reduced from 15 to 10 for speed
                                ):
                                    visibility_mask[i] = False
                                    n_dem_blocked += 1
                else:
                    # No terrain masking enabled
                    n_geometric_visible = n_emitters
                    n_dem_checked = 0
                    n_dem_blocked = 0
                    n_high_elevation = 0

                # Filter to visible emitters only
                visible_indices = np.where(visibility_mask)[0]
                n_visible_emitters = len(visible_indices)

                if n_visible_emitters > 0:
                    # Vectorized coordinate transformation for visible emitters
                    emitter_ecef_visible = emitter_ecef_all[visible_indices]
                    n_visible = len(visible_indices)

                    # Ensure emitter_ecef_visible has shape [n_visible, 3]
                    if emitter_ecef_visible.ndim == 1:
                        # If it's 1D, reshape to [n_visible, 3]
                        emitter_ecef_visible = emitter_ecef_visible.reshape(n_visible, 3)
                    elif emitter_ecef_visible.ndim == 2 and emitter_ecef_visible.shape[1] != 3:
                        # If shape is wrong, fix it
                        emitter_ecef_visible = emitter_ecef_visible.reshape(n_visible, 3)
                    # Ensure it's float64 and contiguous
                    emitter_ecef_visible = np.ascontiguousarray(
                        emitter_ecef_visible.astype(np.float64)
                    )

                    # Vectorized link budget calculation
                    # Transform emitters to weather satellite frame (vectorized)
                    weather_sat_velocity_ecef = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    ws_ecef_array = np.asarray(ws_ecef, dtype=np.float64).reshape(1, 3)

                    emitter_decs, emitter_cazs = ecef_to_weather_sat_frame(
                        emitter_ecef_visible,
                        np.tile(ws_ecef_array, (n_visible, 1)),
                        np.tile(weather_sat_velocity_ecef.reshape(1, 3), (n_visible, 1))
                    )

                    # Calculate distances (vectorized)
                    # Broadcasting: [n_visible, 3] - [1, 3] = [n_visible, 3]
                    sat_to_emitter_vecs = emitter_ecef_visible - ws_ecef_array
                    sat_to_emitter_dists = np.linalg.norm(sat_to_emitter_vecs, axis=1)

                    # Vectorized antenna gain calculations using get_gain_values()
                    gain_weather_sat_visible = weather_sat_antenna.get_gain_values(
                        emitter_decs, emitter_cazs
                    )

                    # Get valid angle ranges for emitter antenna (once)
                    valid_alphas, valid_betas = ground_emitter_antenna.get_def_angles()
                    alpha_min_deg, alpha_max_deg = valid_alphas.min(), valid_alphas.max()
                    beta_min_deg, beta_max_deg = valid_betas.min(), valid_betas.max()
                    alpha_min_rad = np.deg2rad(alpha_min_deg)
                    alpha_max_rad = np.deg2rad(alpha_max_deg)
                    beta_min_rad = np.deg2rad(beta_min_deg)
                    beta_max_rad = np.deg2rad(beta_max_deg)

                    # Clamp angles for emitter antenna (vectorized)
                    emitter_alphas_rad = np.clip(emitter_decs, alpha_min_rad, alpha_max_rad)
                    emitter_betas_rad = emitter_cazs % (2 * np.pi)
                    emitter_betas_rad = np.where(
                        emitter_betas_rad < 0,
                        emitter_betas_rad + 2 * np.pi,
                        emitter_betas_rad
                    )
                    if beta_max_deg >= 360.0:
                        emitter_betas_rad = np.clip(emitter_betas_rad, beta_min_rad, beta_max_rad)
                    else:
                        emitter_betas_rad = np.clip(emitter_betas_rad, beta_min_rad, beta_max_rad)

                    # Get emitter antenna gains (vectorized using get_gain_values())
                    # Pattern stores absolute gain values, but EIRP already includes peak gain
                    # So we need to normalize to relative pattern for link budget
                    gain_emitter_visible_absolute = ground_emitter_antenna.get_gain_values(
                        emitter_alphas_rad, emitter_betas_rad
                    )
                    # Normalize to relative pattern (since EIRP already includes peak gain)
                    peak_gain_emitter = ground_emitter_antenna.get_boresight_gain()
                    gain_emitter_visible = gain_emitter_visible_absolute / peak_gain_emitter

                    # Check if we have fundamental frequency and bandwidth information
                    if (ground_emitter_fundamental_freq is not None and bandwidth is not None):
                        # Calculate observation bandwidth bounds
                        freq_min = freq - bandwidth / 2
                        freq_max = freq + bandwidth / 2

                        # Calculate base link budget at fundamental frequency
                        speed_c = 3e8
                        wavelength_fund = speed_c / ground_emitter_fundamental_freq
                        L_fs_fund_visible = (4 * np.pi * sat_to_emitter_dists / wavelength_fund) ** 2

                        # Atmospheric loss at fundamental frequency
                        if include_atmospheric_loss:
                            elevation_angles_deg = np.rad2deg(np.pi / 2 - emitter_alphas_rad)
                            freq_ghz_fund = ground_emitter_fundamental_freq / 1e9

                            # Simplified atmospheric absorption model (vectorized)
                            if freq_ghz_fund < 20:
                                absorption_db_per_km = 0.01
                            elif freq_ghz_fund < 40:
                                absorption_db_per_km = 0.2 + 0.3 * ((freq_ghz_fund - 20) / 20.0)
                            elif freq_ghz_fund < 60:
                                oxygen_band_center = 60.0
                                distance_from_peak = abs(freq_ghz_fund - oxygen_band_center)
                                if distance_from_peak < 10:
                                    absorption_db_per_km = 10.0 - 8.0 * (distance_from_peak / 10.0)
                                else:
                                    absorption_db_per_km = 2.0 + 1.0 * ((freq_ghz_fund - 40) / 20.0)
                            else:
                                absorption_db_per_km = 15.0

                            # Path length multiplier based on elevation (vectorized)
                            elev_rad = np.deg2rad(elevation_angles_deg)
                            effective_path_multiplier = 1.0 / np.maximum(np.sin(elev_rad), 0.1)
                            effective_path_multiplier = np.minimum(effective_path_multiplier, 3.0)

                            # Total atmospheric loss (vectorized)
                            # For space-to-ground, only ~25 km is in atmosphere
                            atmospheric_path_km = 25.0
                            total_loss_db = absorption_db_per_km * atmospheric_path_km * effective_path_multiplier
                            L_atm_fund_visible = 10 ** (total_loss_db / 10.0)
                        else:
                            L_atm_fund_visible = np.ones(n_visible)

                        base_link_budgets = (
                            gain_weather_sat_visible *
                            (1.0 / L_fs_fund_visible) *
                            (1.0 / L_atm_fund_visible) *
                            gain_emitter_visible
                        )

                        # Check if fundamental falls within observation bandwidth
                        fundamental_in_band = (freq_min <= ground_emitter_fundamental_freq <= freq_max)
                        if fundamental_in_band:
                            # Fundamental is within observation band - include it
                            link_budgets = base_link_budgets
                        else:
                            # Fundamental is outside observation band - no base contribution
                            link_budgets = np.zeros(n_visible)

                        # Add harmonic contributions if harmonics are provided
                        if ground_emitter_harmonics is not None:
                            # Calculate path loss at fundamental frequency (for harmonic scaling)
                            L_fundamental = L_fs_fund_visible

                            # Initialize harmonic contributions
                            harmonic_contributions = np.zeros(n_visible)

                            for freq_mult, power_red in ground_emitter_harmonics:
                                harmonic_frequency = ground_emitter_fundamental_freq * freq_mult

                                # Check if harmonic falls within observation band
                                if freq_min <= harmonic_frequency <= freq_max:
                                    # Calculate path loss at harmonic frequency
                                    wavelength_harm = speed_c / harmonic_frequency
                                    L_harmonic = (4 * np.pi * sat_to_emitter_dists / wavelength_harm) ** 2

                                    # Atmospheric loss at harmonic frequency
                                    if include_atmospheric_loss:
                                        freq_ghz_harm = harmonic_frequency / 1e9
                                        if freq_ghz_harm < 20:
                                            absorption_db_per_km = 0.01
                                        elif freq_ghz_harm < 40:
                                            absorption_db_per_km = 0.2 + 0.3 * ((freq_ghz_harm - 20) / 20.0)
                                        elif freq_ghz_harm < 60:
                                            oxygen_band_center = 60.0
                                            distance_from_peak = abs(freq_ghz_harm - oxygen_band_center)
                                            if distance_from_peak < 10:
                                                absorption_db_per_km = 10.0 - 8.0 * (distance_from_peak / 10.0)
                                            else:
                                                absorption_db_per_km = 2.0 + 1.0 * ((freq_ghz_harm - 40) / 20.0)
                                        else:
                                            absorption_db_per_km = 15.0

                                        elev_rad = np.deg2rad(elevation_angles_deg)
                                        effective_path_multiplier = 1.0 / np.maximum(np.sin(elev_rad), 0.1)
                                        effective_path_multiplier = np.minimum(
                                            effective_path_multiplier, 3.0)
                                        # For space-to-ground, only ~25 km is in atmosphere
                                        atmospheric_path_km = 25.0
                                        total_loss_db = (absorption_db_per_km *
                                                         atmospheric_path_km *
                                                         effective_path_multiplier)
                                        L_atm_harm = 10 ** (total_loss_db / 10.0)
                                    else:
                                        L_atm_harm = np.ones(n_visible)

                                    # Path loss ratio (fundamental vs harmonic)
                                    path_loss_ratio = L_fundamental / L_harmonic

                                    # Harmonic link budget contribution
                                    harmonic_link_budgets = (
                                        base_link_budgets *
                                        power_red *
                                        path_loss_ratio *
                                        (L_atm_fund_visible / L_atm_harm)  # Atmospheric loss ratio
                                    )
                                    harmonic_contributions += harmonic_link_budgets

                            # Add harmonic contribution
                            link_budgets += harmonic_contributions

                        # Add OOBE contribution if OOBE parameters are provided
                        if (ground_emitter_oobe_suppression_db is not None and
                                ground_emitter_oobe_freq_offset_max is not None):
                            # Diagnostic: print OOBE check (first time step, first frequency only)
                            if t_idx == 0 and f_idx == 0:
                                print(f"\n      ===== OOBE DIAGNOSTIC (t={t_idx}, f={f_idx}) =====")
                                print(f"      OOBE check (first time step, {freq/1e9:.1f} GHz):")
                                print(f"        Fundamental: {ground_emitter_fundamental_freq/1e9:.2f} GHz")
                                print(f"        Observation band: {freq_min/1e9:.3f} - {freq_max/1e9:.3f} GHz")
                                print(f"        OOBE suppression: {ground_emitter_oobe_suppression_db:.1f} dB")
                                print(f"        OOBE offset max: {ground_emitter_oobe_freq_offset_max/1e6:.0f} MHz")
                                supp_check = ground_emitter_oobe_suppression_db is not None
                                offset_check = ground_emitter_oobe_freq_offset_max is not None
                                print(f"        OOBE params check: suppression={supp_check}, offset={offset_check}")

                            # Check if fundamental is outside observation band but within OOBE range
                            fundamental_in_band = (freq_min <= ground_emitter_fundamental_freq <= freq_max)

                            if not fundamental_in_band:
                                # Calculate frequency offset from observation band (scalar)
                                if ground_emitter_fundamental_freq < freq_min:
                                    freq_offset = freq_min - ground_emitter_fundamental_freq
                                else:  # ground_emitter_fundamental_freq > freq_max
                                    freq_offset = ground_emitter_fundamental_freq - freq_max

                                # Check if within OOBE range
                                if freq_offset <= ground_emitter_oobe_freq_offset_max:
                                    oobe_applicable = True
                                    if t_idx == 0 and f_idx == 0:
                                        print(f"        Fundamental is OUTSIDE band, offset: {freq_offset/1e6:.1f} MHz")
                                        offset_max_mhz = ground_emitter_oobe_freq_offset_max / 1e6
                                        print(f"        OOBE APPLICABLE (offset <= {offset_max_mhz:.0f} MHz)")
                                else:
                                    oobe_applicable = False
                                    if t_idx == 0 and f_idx == 0:
                                        print(f"        Fundamental is OUTSIDE band, offset: {freq_offset/1e6:.1f} MHz")
                                        offset_max_mhz = ground_emitter_oobe_freq_offset_max / 1e6
                                        print(f"        OOBE NOT APPLICABLE (offset > {offset_max_mhz:.0f} MHz)")
                            else:
                                oobe_applicable = False
                                if t_idx == 0 and f_idx == 0:
                                    print("        Fundamental is INSIDE band - OOBE not applicable")

                            if oobe_applicable:
                                # Calculate path loss at observation frequency
                                wavelength_obs = speed_c / freq
                                L_fs_obs = (4 * np.pi * sat_to_emitter_dists / wavelength_obs) ** 2

                                # Atmospheric loss at observation frequency
                                if include_atmospheric_loss:
                                    freq_ghz_obs = freq / 1e9
                                    if freq_ghz_obs < 20:
                                        absorption_db_per_km = 0.01
                                    elif freq_ghz_obs < 40:
                                        absorption_db_per_km = 0.2 + 0.3 * ((freq_ghz_obs - 20) / 20.0)
                                    elif freq_ghz_obs < 60:
                                        oxygen_band_center = 60.0
                                        distance_from_peak = abs(freq_ghz_obs - oxygen_band_center)
                                        if distance_from_peak < 10:
                                            absorption_db_per_km = 10.0 - 8.0 * (distance_from_peak / 10.0)
                                        else:
                                            absorption_db_per_km = 2.0 + 1.0 * ((freq_ghz_obs - 40) / 20.0)
                                    else:
                                        absorption_db_per_km = 15.0

                                    elev_rad = np.deg2rad(elevation_angles_deg)
                                    effective_path_multiplier = 1.0 / np.maximum(np.sin(elev_rad), 0.1)
                                    effective_path_multiplier = np.minimum(
                                        effective_path_multiplier, 3.0)
                                    atmospheric_path_km = 25.0
                                    total_loss_db = (absorption_db_per_km *
                                                     atmospheric_path_km *
                                                     effective_path_multiplier)
                                    L_atm_obs = 10 ** (total_loss_db / 10.0)
                                else:
                                    L_atm_obs = np.ones(n_visible)

                                # OOBE link budget accounts for:
                                # 1. OOBE suppression factor
                                # 2. Path loss ratio (observation vs fundamental)
                                # 3. Atmospheric loss ratio (observation vs fundamental)
                                path_loss_ratio = L_fs_fund_visible / L_fs_obs
                                atm_loss_ratio = L_atm_fund_visible / L_atm_obs
                                # ground_emitter_oobe_suppression_db is already negative (e.g., -50.0)
                                oobe_suppression_linear = 10 ** (ground_emitter_oobe_suppression_db / 10.0)

                                # Calculate OOBE contribution (applies to all visible emitters)
                                oobe_link_budgets = (
                                    base_link_budgets *
                                    oobe_suppression_linear *
                                    path_loss_ratio *
                                    atm_loss_ratio
                                )

                                # Add OOBE contribution
                                link_budgets += oobe_link_budgets

                                # Diagnostic output for OOBE (first time step, first frequency only)
                                if (t_idx == 0 and f_idx == 0 and
                                        ground_emitter_oobe_suppression_db is not None):
                                    if len(oobe_link_budgets) > 0:
                                        max_oobe = np.max(oobe_link_budgets)
                                        mean_oobe = np.mean(oobe_link_budgets)
                                        # Calculate sample power for diagnostics
                                        sample_emitter_temp = power_to_temperature(
                                            10**(ground_emitter_eirp_dbw / 10.0), bandwidth)
                                        k_boltz = 1.38e-23
                                        sample_power_w = max_oobe * sample_emitter_temp * bandwidth * k_boltz
                                        sample_power_dbw = 10 * np.log10(sample_power_w + 1e-100)
                                        total_oobe_temp = np.sum(oobe_link_budgets) * sample_emitter_temp
                                        total_oobe_power_w = total_oobe_temp * bandwidth * k_boltz
                                        total_oobe_power_dbw = 10 * np.log10(total_oobe_power_w + 1e-100)
                                    else:
                                        max_oobe = 0.0
                                        mean_oobe = 0.0
                                        sample_power_dbw = -1000.0
                                        total_oobe_power_dbw = -1000.0
                                    print(f"      OOBE diagnostic (first time step, {freq/1e9:.1f} GHz):")
                                    print(f"        Fundamental: {ground_emitter_fundamental_freq/1e9:.2f} GHz")
                                    print(f"        Observation: {freq/1e9:.2f} GHz")
                                    print(f"        OOBE suppression: {ground_emitter_oobe_suppression_db:.1f} dB")
                                    print(f"        Max OOBE link budget: {max_oobe:.2e}")
                                    print(f"        Mean OOBE link budget: {mean_oobe:.2e}")
                                    print(f"        Number of emitters with OOBE: {len(oobe_link_budgets)}")
                                    print(f"        Sample emitter power (max): {sample_power_dbw:.2f} dBW")
                                    print(f"        Total OOBE power (all emitters): {total_oobe_power_dbw:.2f} dBW")
                                    print(f"        Path loss ratio (fund/obs): {np.mean(path_loss_ratio):.4f}")
                                    print(f"        Atm loss ratio (fund/obs): {np.mean(atm_loss_ratio):.4f}")
                                    # Calculate what in-band power would be (for comparison)
                                    if len(base_link_budgets) > 0:
                                        total_base_temp = np.sum(base_link_budgets) * sample_emitter_temp
                                        total_base_power_w = total_base_temp * bandwidth * k_boltz
                                        total_base_power_dbw = 10 * np.log10(total_base_power_w + 1e-100)
                                        oobe_vs_inband_diff = total_oobe_power_dbw - total_base_power_dbw
                                        print(f"        In-band power (if fundamental in-band): "
                                              f"{total_base_power_dbw:.2f} dBW")
                                        print(f"        OOBE vs in-band difference: "
                                              f"{oobe_vs_inband_diff:.2f} dB")
                                    # Show actual values for verification
                                    print(f"        Max base link budget: {np.max(base_link_budgets):.2e}")
                                    print(f"        Mean base link budget: {np.mean(base_link_budgets):.2e}")
                                    print(f"        OOBE suppression (linear): {oobe_suppression_linear:.2e}")
                                    print(f"        Mean path loss ratio: {np.mean(path_loss_ratio):.6f}")
                                    print(f"        Mean atm loss ratio: {np.mean(atm_loss_ratio):.6f}")
                    else:
                        # No fundamental frequency info: calculate base link budget at observation frequency
                        # (legacy behavior - may not be physically correct if frequencies don't match)
                        # Vectorized free-space path loss (fully vectorized)
                        # L = (4 * π * rng / (c / freq))^2
                        speed_c = 3e8
                        wavelength = speed_c / freq
                        L_fs_visible = (4 * np.pi * sat_to_emitter_dists / wavelength) ** 2

                        # Vectorized atmospheric loss (if enabled)
                        if include_atmospheric_loss:
                            elevation_angles_deg = np.rad2deg(np.pi / 2 - emitter_alphas_rad)
                            # Vectorize atmospheric loss calculation
                            freq_ghz = freq / 1e9

                            # Simplified atmospheric absorption model (vectorized)
                            if freq_ghz < 20:
                                absorption_db_per_km = 0.01
                            elif freq_ghz < 40:
                                absorption_db_per_km = 0.2 + 0.3 * ((freq_ghz - 20) / 20.0)
                            elif freq_ghz < 60:
                                oxygen_band_center = 60.0
                                distance_from_peak = abs(freq_ghz - oxygen_band_center)
                                if distance_from_peak < 10:
                                    absorption_db_per_km = 10.0 - 8.0 * (distance_from_peak / 10.0)
                                else:
                                    absorption_db_per_km = 2.0 + 1.0 * ((freq_ghz - 40) / 20.0)
                            else:
                                absorption_db_per_km = 15.0

                            # Path length multiplier based on elevation (vectorized)
                            elev_rad = np.deg2rad(elevation_angles_deg)
                            effective_path_multiplier = 1.0 / np.maximum(np.sin(elev_rad), 0.1)
                            effective_path_multiplier = np.minimum(effective_path_multiplier, 3.0)

                            # Total atmospheric loss (vectorized)
                            # For space-to-ground, only ~25 km is in atmosphere
                            atmospheric_path_km = 25.0
                            total_loss_db = absorption_db_per_km * atmospheric_path_km * effective_path_multiplier
                            L_atm_visible = 10 ** (total_loss_db / 10.0)
                        else:
                            L_atm_visible = np.ones(n_visible)

                        # Vectorized link budget calculation
                        link_budgets = (
                            gain_weather_sat_visible *
                            (1.0 / L_fs_visible) *
                            (1.0 / L_atm_visible) *
                            gain_emitter_visible
                        )

                    # Ground emitter transmit power (convert to temperature)
                    emitter_power_w = 10**(ground_emitter_eirp_dbw / 10.0)
                    emitter_temp = power_to_temperature(emitter_power_w, bandwidth)

                    # Vectorized interference temperature
                    interference_temps = link_budgets * emitter_temp
                    ground_emitter_interference_temp = np.sum(interference_temps)
                else:
                    # No visible emitters
                    ground_emitter_interference_temp = 0.0

                # Print detailed visibility statistics at first time step
                if t_idx == 0:
                    print("    Ground emitter visibility statistics (first time step):")
                    print(f"      Total emitters: {len(ground_emitters)}")
                    if enable_terrain_masking:
                        print(f"      Geometric horizon check: {n_geometric_visible}/{len(ground_emitters)} passed")
                        if dem_masker is not None and dem_masker.dem_data is not None:
                            print(f"      High elevation (>30°): {n_high_elevation} (skipped DEM ray tracing)")
                            print(f"      DEM ray tracing checked: {n_dem_checked}")
                            print(f"      Terrain-masked (blocked): {n_dem_blocked}")
                        else:
                            print("      DEM not available - using geometric check only")
                    else:
                        print("      Terrain masking disabled - all emitters assumed visible")
                    print(f"      Final visible: {n_visible_emitters}/{len(ground_emitters)}")

            # Convert interference to power
            starlink_power = temperature_to_power(starlink_interference_temp, bandwidth)
            ground_emitter_power = temperature_to_power(ground_emitter_interference_temp, bandwidth)

            # Earth brightness (through main lobe pointing at target)
            earth_temp_freq = calculate_earth_brightness_temperature(freq, earth_brightness_temp)
            earth_gain = weather_sat_antenna.get_gain_value(target_dec, target_caz)
            earth_power = temperature_to_power(earth_temp_freq, bandwidth) * earth_gain

            # Sky background (through sidelobes - use average gain)
            sky_power = temperature_to_power(sky_brightness_temp, bandwidth) * 0.1  # Approximate sidelobe gain

            # System noise
            system_power = temperature_to_power(system_temp, bandwidth)

            # Store individual components
            result_starlink[t_idx, f_idx] = starlink_power
            result_ground_emitter[t_idx, f_idx] = ground_emitter_power
            result_earth[t_idx, f_idx] = earth_power
            result_sky[t_idx, f_idx] = sky_power
            result_system[t_idx, f_idx] = system_power

            # Total power
            result_power[t_idx, f_idx] = (
                starlink_power + ground_emitter_power + earth_power + sky_power + system_power
            )

    # Convert to dBW
    result_power_dbw = 10 * np.log10(result_power + 1e-100)
    result_starlink_dbw = 10 * np.log10(result_starlink + 1e-100)
    result_ground_emitter_dbw = 10 * np.log10(result_ground_emitter + 1e-100)
    result_earth_dbw = 10 * np.log10(result_earth + 1e-100)
    result_sky_dbw = 10 * np.log10(result_sky + 1e-100)
    result_system_dbw = 10 * np.log10(result_system + 1e-100)

    # Return dictionary with total and individual components
    return {
        'total': result_power_dbw,
        'starlink': result_starlink_dbw,
        'ground_emitter': result_ground_emitter_dbw,
        'earth': result_earth_dbw,
        'sky': result_sky_dbw,
        'system': result_system_dbw
    }


# =============================================================================
# Phase 3: Enhanced Atmospheric Effects
# =============================================================================

def calculate_oxygen_absorption(
    freq: float,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    elevation_angle: float = None
) -> float:
    """
    Calculate oxygen absorption coefficient based on ITU-R P.676.

    Oxygen absorption is significant at 50-60 GHz (oxygen resonance line).

    Args:
        freq: Frequency in Hz
        temperature: Temperature in Kelvin (default: 288.15 K = 15°C)
        pressure: Pressure in Pa (default: 101325 Pa = 1 atm)
        elevation_angle: Elevation angle from ground (degrees, optional)

    Returns:
        float: Oxygen absorption coefficient in dB/km
    """
    freq_ghz = freq / 1e9

    # ITU-R P.676 simplified model for oxygen absorption
    # Strong absorption near 60 GHz oxygen resonance line
    oxygen_band_center = 60.0  # GHz
    distance_from_peak = abs(freq_ghz - oxygen_band_center)

    if freq_ghz < 50:
        # Below 50 GHz: moderate oxygen absorption
        # Increases as we approach 60 GHz
        if freq_ghz < 40:
            # Below 40 GHz: minimal oxygen absorption
            alpha_o2 = 0.01 * (freq_ghz / 40.0) ** 2
        else:
            # 40-50 GHz: increasing oxygen absorption
            alpha_o2 = 0.1 + 1.9 * ((freq_ghz - 40) / 10.0)
    elif freq_ghz < 70:
        # Near oxygen line (50-70 GHz): very strong absorption
        if distance_from_peak < 5:
            # Very close to 60 GHz: extremely high absorption
            alpha_o2 = 15.0 - 12.0 * (distance_from_peak / 5.0)
        elif distance_from_peak < 10:
            # Close to 60 GHz: high absorption
            alpha_o2 = 3.0 - 2.0 * ((distance_from_peak - 5) / 5.0)
        else:
            # Far from peak but still in band: moderate absorption
            alpha_o2 = 1.0 + 0.5 * ((freq_ghz - 50) / 20.0)
    else:
        # Above 70 GHz: decreasing but still significant
        alpha_o2 = 1.5 - 0.5 * ((freq_ghz - 70) / 30.0)
        alpha_o2 = max(0.5, alpha_o2)

    # Pressure and temperature scaling (simplified)
    # Higher pressure -> more absorption, higher temperature -> less absorption
    pressure_factor = pressure / 101325.0
    temperature_factor = (288.15 / temperature) ** 0.5
    alpha_o2 *= pressure_factor * temperature_factor

    return max(0.001, alpha_o2)  # Minimum 0.001 dB/km


def calculate_water_vapor_absorption(
    freq: float,
    humidity: float = 50.0,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    elevation_angle: float = None
) -> float:
    """
    Calculate water vapor absorption coefficient based on ITU-R P.676.

    Water vapor absorption is significant at 22.235 GHz (water vapor line)
    and increases with frequency.

    Args:
        freq: Frequency in Hz
        humidity: Relative humidity in % (default: 50%)
        temperature: Temperature in Kelvin (default: 288.15 K = 15°C)
        pressure: Pressure in Pa (default: 101325 Pa = 1 atm)
        elevation_angle: Elevation angle from ground (degrees, optional)

    Returns:
        float: Water vapor absorption coefficient in dB/km
    """
    freq_ghz = freq / 1e9

    # ITU-R P.676 simplified model for water vapor absorption
    # Strong absorption near 22.235 GHz water vapor line
    water_vapor_line = 22.235  # GHz
    distance_from_line = abs(freq_ghz - water_vapor_line)

    if freq_ghz < 15:
        # Below 15 GHz: minimal water vapor absorption
        alpha_wv = 0.01 * (freq_ghz / 15.0) ** 2
    elif freq_ghz < 30:
        # 15-30 GHz: water vapor absorption region
        if distance_from_line < 2:
            # Very close to 22.235 GHz: high absorption
            alpha_wv = 0.5 - 0.4 * (distance_from_line / 2.0)
        elif distance_from_line < 5:
            # Close to line: moderate absorption
            alpha_wv = 0.1 + 0.1 * ((distance_from_line - 2) / 3.0)
        else:
            # Far from line: low absorption
            alpha_wv = 0.05 + 0.05 * ((freq_ghz - 15) / 15.0)
    else:
        # Above 30 GHz: increasing with frequency
        # Water vapor absorption increases quadratically with frequency
        alpha_wv = 0.1 + 0.3 * ((freq_ghz - 30) / 30.0) ** 2

    # Humidity scaling (linear with relative humidity)
    humidity_factor = humidity / 50.0

    # Temperature and pressure scaling (simplified)
    # Higher temperature -> more water vapor capacity -> more absorption
    # Higher pressure -> more absorption
    temperature_factor = (temperature / 288.15) ** 1.5
    pressure_factor = pressure / 101325.0
    alpha_wv *= humidity_factor * temperature_factor * pressure_factor

    return max(0.001, alpha_wv)  # Minimum 0.001 dB/km


def calculate_comprehensive_atmospheric_loss(
    distance: float,
    freq: float,
    elevation_angle: float = None,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    humidity: float = 50.0,
    separate_components: bool = False,
    use_full_itu_p676: bool = True
):
    """
    Calculate comprehensive atmospheric absorption loss (Phase 3).

    Uses full ITU-R P.676 model with line-by-line calculations for oxygen
    and water vapor absorption. Falls back to simplified model if the
    ITU-R P.676 calculator is not available.

    Args:
        distance: Path length in meters (not used, kept for compatibility)
        freq: Frequency in Hz
        elevation_angle: Elevation angle from ground (degrees, optional)
        temperature: Temperature in Kelvin (default: 288.15 K = 15°C)
        pressure: Pressure in Pa (default: 101325 Pa = 1 atm)
        humidity: Relative humidity in % (default: 50%)
        separate_components: If True, returns dict with separate components
        use_full_itu_p676: If True, use full ITU-R P.676 model (default: True)

    Returns:
        float or dict: Atmospheric loss factor (linear, > 1.0) or dict with
            components if separate_components=True
    """
    freq_ghz = freq / 1e9

    # Convert pressure from Pa to hPa
    pressure_hpa = pressure / 100.0

    # Convert relative humidity to water vapor density (g/m³)
    # Using simplified formula: rho_wv ≈ 7.5 * (humidity/50) at 15°C
    # More accurate: use saturation vapor pressure
    T_celsius = temperature - 273.15
    # Saturation vapor pressure (Magnus formula, hPa)
    e_sat = 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))
    # Actual vapor pressure
    e_actual = (humidity / 100.0) * e_sat
    # Water vapor density (g/m³)
    water_vapor_density = (e_actual * 216.7) / temperature

    # Default elevation angle for zenith if not specified
    if elevation_angle is None:
        elevation_deg = 90.0
    else:
        elevation_deg = max(elevation_angle, 1.0)  # Minimum 1 degree

    # Use full ITU-R P.676 model if available and requested
    if use_full_itu_p676 and ITU_P676_AVAILABLE:
        try:
            calc = get_cached_calculator()

            # Get detailed attenuation breakdown
            result = calc.total_slant_attenuation_detailed(
                freq_ghz=freq_ghz,
                elevation_deg=elevation_deg,
                pressure_hpa=pressure_hpa,
                temperature_k=temperature,
                water_vapor_density=water_vapor_density,
                include_water_vapor=True
            )

            loss_total_db = result['A_total_slant']
            loss_o2_db = result['A_o_slant']
            loss_wv_db = result['A_w_slant']

            # Convert to linear loss factors
            loss_o2 = 10 ** (loss_o2_db / 10.0)
            loss_wv = 10 ** (loss_wv_db / 10.0)
            loss_total = 10 ** (loss_total_db / 10.0)

            if separate_components:
                return {
                    'total': loss_total,
                    'oxygen': loss_o2,
                    'water_vapor': loss_wv,
                    'total_db': loss_total_db,
                    'oxygen_db': loss_o2_db,
                    'water_vapor_db': loss_wv_db,
                    'gamma_o': result['gamma_o'],
                    'gamma_w': result['gamma_w'],
                    'h0': result['h0'],
                    'h_w': result['h_w']
                }
            else:
                return loss_total

        except Exception as e:
            # Fall back to simplified model on error
            print(f"Warning: ITU-R P.676 calculation failed ({e}), using simplified model")

    # Simplified model fallback
    alpha_o2 = calculate_oxygen_absorption(freq, temperature, pressure, elevation_angle)
    alpha_wv = calculate_water_vapor_absorption(
        freq, humidity, temperature, pressure, elevation_angle
    )

    # Total absorption coefficient
    alpha_total = alpha_o2 + alpha_wv  # dB/km

    # Path length through atmosphere depends on elevation angle
    # For space-to-ground links, only ~20-30 km is in the atmosphere
    atmospheric_path_km = 25.0  # Approximate atmospheric path length (km)
    if elevation_angle is not None:
        # Longer path at low elevation angles
        elev_rad = np.deg2rad(elevation_angle)
        effective_path_multiplier = 1.0 / max(np.sin(elev_rad), 0.1)
        # Limit multiplier for reasonable values
        effective_path_multiplier = min(effective_path_multiplier, 3.0)
    else:
        effective_path_multiplier = 1.0

    # Calculate losses for each component
    loss_o2_db = alpha_o2 * atmospheric_path_km * effective_path_multiplier
    loss_wv_db = alpha_wv * atmospheric_path_km * effective_path_multiplier
    loss_total_db = alpha_total * atmospheric_path_km * effective_path_multiplier

    # Convert to linear loss factors
    loss_o2 = 10 ** (loss_o2_db / 10.0)
    loss_wv = 10 ** (loss_wv_db / 10.0)
    loss_total = 10 ** (loss_total_db / 10.0)

    if separate_components:
        return {
            'total': loss_total,
            'oxygen': loss_o2,
            'water_vapor': loss_wv,
            'total_db': loss_total_db,
            'oxygen_db': loss_o2_db,
            'water_vapor_db': loss_wv_db
        }
    else:
        return loss_total


def calculate_atmospheric_refraction(
    elevation_angle: float,
    freq: float = None,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    humidity: float = 50.0
) -> Tuple[float, float]:
    """
    Calculate atmospheric refraction effects for space-to-ground paths.

    Refraction bends the ray path, making the apparent elevation higher
    than the true elevation, especially at low elevation angles.

    Args:
        elevation_angle: True elevation angle in degrees
        freq: Frequency in Hz (optional, for frequency-dependent effects)
        temperature: Temperature in Kelvin (default: 288.15 K = 15°C)
        pressure: Pressure in Pa (default: 101325 Pa = 1 atm)
        humidity: Relative humidity in % (default: 50%)

    Returns:
        Tuple[float, float]: (apparent_elevation, refraction_angle)
            - apparent_elevation: Apparent elevation after refraction (degrees)
            - refraction_angle: Refraction correction in degrees
    """
    if elevation_angle <= 0:
        return elevation_angle, 0.0

    # Simplified refraction model (Bennett's formula with enhancements)
    # Refraction coefficient depends on atmospheric conditions
    # Standard value: ~0.13 degrees / tan(elevation) for standard atmosphere

    # Atmospheric condition factors
    # Higher pressure -> more refraction
    # Higher temperature -> less refraction
    # Higher humidity -> more refraction (water vapor has higher refractive index)
    pressure_factor = pressure / 101325.0
    temperature_factor = 288.15 / temperature
    humidity_factor = 1.0 + 0.01 * (humidity / 50.0)

    # Base refraction coefficient (degrees)
    base_refraction_coeff = 0.13

    # Adjusted refraction coefficient
    refraction_coeff = (base_refraction_coeff * pressure_factor *
                        temperature_factor * humidity_factor)

    # Refraction correction (increases at lower elevation angles)
    elev_rad = np.deg2rad(elevation_angle)
    refraction_correction = refraction_coeff / np.tan(elev_rad)

    # Limit refraction for very low elevation angles
    if elevation_angle < 5.0:
        # Enhanced refraction at very low angles
        refraction_correction *= 1.5
    if elevation_angle < 1.0:
        # Extreme refraction near horizon
        refraction_correction *= 2.0

    # Apparent elevation (refraction makes objects appear higher)
    apparent_elevation = elevation_angle + refraction_correction

    return apparent_elevation, refraction_correction


def calculate_comprehensive_atmospheric_loss_vectorized(
    distances: np.ndarray,
    freqs: np.ndarray,
    elevation_angles: np.ndarray = None,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    humidity: float = 50.0,
    use_full_itu_p676: bool = True
) -> np.ndarray:
    """
    Vectorized version of calculate_comprehensive_atmospheric_loss.

    Uses full ITU-R P.676 model with line-by-line calculations for oxygen
    and water vapor absorption. Falls back to simplified model if the
    ITU-R P.676 calculator is not available.

    Args:
        distances: Array of path lengths in meters (not used, kept for compatibility)
        freqs: Array of frequencies in Hz
        elevation_angles: Array of elevation angles from ground (degrees, optional)
        temperature: Temperature in Kelvin (default: 288.15 K = 15°C)
        pressure: Pressure in Pa (default: 101325 Pa = 1 atm)
        humidity: Relative humidity in % (default: 50%)
        use_full_itu_p676: If True, use full ITU-R P.676 model (default: True)

    Returns:
        np.ndarray: Array of atmospheric loss factors (linear, > 1.0)
    """
    freqs = np.asarray(freqs)
    n_points = len(freqs)

    if elevation_angles is not None:
        elevation_angles = np.asarray(elevation_angles)
    else:
        elevation_angles = np.full(n_points, 90.0)  # Default to zenith

    # Convert pressure from Pa to hPa
    pressure_hpa = pressure / 100.0

    # Convert relative humidity to water vapor density (g/m³)
    T_celsius = temperature - 273.15
    e_sat = 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))
    e_actual = (humidity / 100.0) * e_sat
    water_vapor_density = (e_actual * 216.7) / temperature

    # Use full ITU-R P.676 model if available and requested
    if use_full_itu_p676 and ITU_P676_AVAILABLE:
        try:
            calc = get_cached_calculator()

            # Calculate attenuation for each point
            loss_total = np.zeros(n_points)
            for i in range(n_points):
                freq_ghz = freqs[i] / 1e9
                elev_deg = max(float(elevation_angles[i]), 1.0)

                atten_db = calc.total_slant_attenuation(
                    freq_ghz=freq_ghz,
                    elevation_deg=elev_deg,
                    pressure_hpa=pressure_hpa,
                    temperature_k=temperature,
                    water_vapor_density=water_vapor_density,
                    include_water_vapor=True
                )
                loss_total[i] = 10 ** (atten_db / 10.0)

            return loss_total

        except Exception as e:
            # Fall back to simplified model on error
            print(f"Warning: ITU-R P.676 calculation failed ({e}), using simplified model")

    # Simplified model fallback (vectorized)
    freq_ghz = freqs / 1e9

    # Oxygen absorption (vectorized)
    oxygen_band_center = 60.0
    distance_from_peak = np.abs(freq_ghz - oxygen_band_center)

    # Initialize alpha_o2 array
    alpha_o2 = np.zeros_like(freq_ghz)

    # Below 50 GHz
    mask_below_50 = freq_ghz < 50
    mask_below_40 = freq_ghz < 40
    alpha_o2[mask_below_40] = 0.01 * (freq_ghz[mask_below_40] / 40.0) ** 2
    mask_40_50 = mask_below_50 & ~mask_below_40
    alpha_o2[mask_40_50] = 0.1 + 1.9 * ((freq_ghz[mask_40_50] - 40) / 10.0)

    # 50-70 GHz (near oxygen line)
    mask_50_70 = (freq_ghz >= 50) & (freq_ghz < 70)
    mask_close = mask_50_70 & (distance_from_peak < 5)
    mask_moderate = mask_50_70 & (distance_from_peak >= 5) & (distance_from_peak < 10)
    mask_far = mask_50_70 & (distance_from_peak >= 10)

    alpha_o2[mask_close] = 15.0 - 12.0 * (distance_from_peak[mask_close] / 5.0)
    alpha_o2[mask_moderate] = 3.0 - 2.0 * ((distance_from_peak[mask_moderate] - 5) / 5.0)
    alpha_o2[mask_far] = 1.0 + 0.5 * ((freq_ghz[mask_far] - 50) / 20.0)

    # Above 70 GHz
    mask_above_70 = freq_ghz >= 70
    alpha_o2[mask_above_70] = np.maximum(0.5, 1.5 - 0.5 * ((freq_ghz[mask_above_70] - 70) / 30.0))

    # Apply pressure and temperature scaling
    pressure_factor = pressure / 101325.0
    temperature_factor = (288.15 / temperature) ** 0.5
    alpha_o2 *= pressure_factor * temperature_factor
    alpha_o2 = np.maximum(0.001, alpha_o2)

    # Water vapor absorption (vectorized)
    water_vapor_line = 22.235
    distance_from_line = np.abs(freq_ghz - water_vapor_line)

    # Initialize alpha_wv array
    alpha_wv = np.zeros_like(freq_ghz)

    # Below 15 GHz
    mask_below_15 = freq_ghz < 15
    alpha_wv[mask_below_15] = 0.01 * (freq_ghz[mask_below_15] / 15.0) ** 2

    # 15-30 GHz
    mask_15_30 = (freq_ghz >= 15) & (freq_ghz < 30)
    mask_very_close = mask_15_30 & (distance_from_line < 2)
    mask_close_wv = mask_15_30 & (distance_from_line >= 2) & (distance_from_line < 5)
    mask_far_wv = mask_15_30 & (distance_from_line >= 5)

    alpha_wv[mask_very_close] = 0.5 - 0.4 * (distance_from_line[mask_very_close] / 2.0)
    alpha_wv[mask_close_wv] = 0.1 + 0.1 * ((distance_from_line[mask_close_wv] - 2) / 3.0)
    alpha_wv[mask_far_wv] = 0.05 + 0.05 * ((freq_ghz[mask_far_wv] - 15) / 15.0)

    # Above 30 GHz
    mask_above_30 = freq_ghz >= 30
    alpha_wv[mask_above_30] = 0.1 + 0.3 * ((freq_ghz[mask_above_30] - 30) / 30.0) ** 2

    # Apply humidity, temperature, and pressure scaling
    humidity_factor = humidity / 50.0
    temperature_factor_wv = (temperature / 288.15) ** 1.5
    pressure_factor_wv = pressure / 101325.0
    alpha_wv *= humidity_factor * temperature_factor_wv * pressure_factor_wv
    alpha_wv = np.maximum(0.001, alpha_wv)

    # Total absorption coefficient
    alpha_total = alpha_o2 + alpha_wv  # dB/km

    # Path length through atmosphere
    atmospheric_path_km = 25.0
    if elevation_angles is not None:
        elev_rad = np.deg2rad(elevation_angles)
        effective_path_multiplier = 1.0 / np.maximum(np.sin(elev_rad), 0.1)
        effective_path_multiplier = np.minimum(effective_path_multiplier, 3.0)
    else:
        effective_path_multiplier = np.ones_like(freqs)

    # Total atmospheric loss
    loss_total_db = alpha_total * atmospheric_path_km * effective_path_multiplier
    loss_total = 10 ** (loss_total_db / 10.0)

    return loss_total


# =============================================================================
# Ground Reflection Functions
# =============================================================================

def calculate_specular_reflection_point(
    starlink_ecef: np.ndarray,
    weather_sat_ecef: np.ndarray,
    earth_radius: float = R_earth,
    max_iterations: int = 50,
    tolerance: float = 1e-6
) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Calculate the specular reflection point on Earth's surface for a Starlink
    satellite signal reflecting to a weather satellite.

    Uses geometric optics: the reflection point is where the angle of incidence
    equals the angle of reflection (law of reflection).

    Implements an iterative method to find the true specular reflection point
    that satisfies the law of reflection, which minimizes path length while
    maintaining equal incident and reflected angles.

    Args:
        starlink_ecef: Starlink satellite position in ECEF [x, y, z] (meters)
        weather_sat_ecef: Weather satellite position in ECEF [x, y, z] (meters)
        earth_radius: Earth radius in meters (default: WGS84)
        max_iterations: Maximum number of iterations for convergence (default: 50)
        tolerance: Convergence tolerance for angle difference in radians (default: 1e-6)

    Returns:
        Tuple[Optional[np.ndarray], float, float]:
            - reflection_point_ecef: Reflection point on Earth surface in ECEF [x, y, z],
              or None if no valid reflection point exists
            - path_length_starlink: Distance from Starlink to reflection point (meters)
            - path_length_weather_sat: Distance from reflection point to weather sat (meters)
    """
    starlink_ecef = np.asarray(starlink_ecef, dtype=np.float64).flatten()
    weather_sat_ecef = np.asarray(weather_sat_ecef, dtype=np.float64).flatten()

    if len(starlink_ecef) != 3 or len(weather_sat_ecef) != 3:
        return None, 0.0, 0.0

    # Vector from Earth center to Starlink
    r_starlink = np.linalg.norm(starlink_ecef)
    r_weather_sat = np.linalg.norm(weather_sat_ecef)

    # Check if both satellites are above Earth
    if r_starlink < earth_radius or r_weather_sat < earth_radius:
        return None, 0.0, 0.0

    # Unit vectors from Earth center
    u_starlink = starlink_ecef / r_starlink
    u_weather_sat = weather_sat_ecef / r_weather_sat

    # Check if satellites are on opposite sides (no valid reflection)
    dot_product = np.dot(u_starlink, u_weather_sat)
    if dot_product < -0.99:  # Nearly opposite (cos(180°) = -1)
        return None, 0.0, 0.0

    # Use Fermat's principle: the specular reflection point minimizes
    # the total path length (Starlink → reflection point → weather sat).
    # This automatically satisfies the law of reflection.

    # The reflection point lies on the great circle arc between the two satellites.
    # We'll use golden section search to find the point that minimizes path length.

    def total_path_length(alpha):
        """
        Calculate total path length for a point on the great circle.

        Args:
            alpha: Interpolation parameter (0 = Starlink direction, 1 = weather sat direction)

        Returns:
            Total path length in meters
        """
        # Interpolate between u_starlink and u_weather_sat along great circle
        u_point = (1 - alpha) * u_starlink + alpha * u_weather_sat
        u_point_norm = np.linalg.norm(u_point)
        if u_point_norm < 1e-12:
            return np.inf  # Invalid point
        u_point = u_point / u_point_norm
        point = u_point * earth_radius

        # Calculate distances
        dist1 = np.linalg.norm(point - starlink_ecef)
        dist2 = np.linalg.norm(weather_sat_ecef - point)

        # Check validity
        if dist1 < 1e3 or dist2 < 1e3:
            return np.inf

        return dist1 + dist2

    # Golden section search for minimum path length
    # This is a robust 1D optimization method
    # Use better initial bounds based on weighted average
    weight_starlink = 1.0 / (r_starlink - earth_radius)
    weight_weather_sat = 1.0 / (r_weather_sat - earth_radius)
    total_weight = weight_starlink + weight_weather_sat
    alpha_initial = weight_weather_sat / total_weight  # Closer to closer satellite

    # Narrow initial search range around the initial guess
    search_range = 0.3  # Search ±30% around initial guess
    a = max(0.0, alpha_initial - search_range)
    b = min(1.0, alpha_initial + search_range)

    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    tolerance_alpha = 1e-6  # Convergence tolerance (relaxed from 1e-8)

    # Reduced iterations: golden section converges quickly
    for iteration in range(25):  # Reduced from 50 to 25 (usually converges in ~15)
        if abs(b - a) < tolerance_alpha:
            break

        c = b - (b - a) / phi
        d = a + (b - a) / phi

        fc = total_path_length(c)
        fd = total_path_length(d)

        if fc < fd:
            b = d
        else:
            a = c

    # Final reflection point
    alpha_opt = (a + b) / 2.0
    u_reflection = (1 - alpha_opt) * u_starlink + alpha_opt * u_weather_sat
    u_reflection = u_reflection / np.linalg.norm(u_reflection)
    reflection_point = u_reflection * earth_radius

    # Verify the result satisfies law of reflection (optional check)
    # This is automatically satisfied by Fermat's principle, but we can verify
    vec_to_starlink = starlink_ecef - reflection_point
    vec_to_weather_sat = weather_sat_ecef - reflection_point
    dist_to_starlink = np.linalg.norm(vec_to_starlink)
    dist_to_weather_sat = np.linalg.norm(vec_to_weather_sat)

    if dist_to_starlink < 1e3 or dist_to_weather_sat < 1e3:
        return None, 0.0, 0.0

    # Optional: verify law of reflection (for diagnostics)
    # This should be satisfied automatically, but we can check
    surface_normal = u_reflection
    incident_dir = -vec_to_starlink / dist_to_starlink
    reflected_dir = vec_to_weather_sat / dist_to_weather_sat
    cos_incidence = np.clip(np.dot(incident_dir, surface_normal), -1.0, 1.0)
    cos_reflection = np.clip(np.dot(reflected_dir, surface_normal), -1.0, 1.0)
    angle_incidence = np.arccos(cos_incidence)
    angle_reflection = np.arccos(cos_reflection)
    angle_error = abs(angle_incidence - angle_reflection)

    # If angle error is too large, something went wrong
    if angle_error > 0.01:  # ~0.57 degrees
        # Fallback: try iterative refinement
        for refine_iter in range(20):
            if angle_error < tolerance:
                break

            # Move toward the satellite with larger angle
            if angle_incidence > angle_reflection:
                correction_dir = u_starlink - np.dot(u_starlink, u_reflection) * u_reflection
            else:
                correction_dir = u_weather_sat - np.dot(u_weather_sat, u_reflection) * u_reflection

            correction_norm = np.linalg.norm(correction_dir)
            if correction_norm < 1e-12:
                break

            correction_dir = correction_dir / correction_norm
            step_size = min(angle_error * 0.1, 0.05)
            u_reflection = u_reflection + step_size * correction_dir
            u_reflection = u_reflection / np.linalg.norm(u_reflection)
            reflection_point = u_reflection * earth_radius

            # Recalculate
            vec_to_starlink = starlink_ecef - reflection_point
            vec_to_weather_sat = weather_sat_ecef - reflection_point
            dist_to_starlink = np.linalg.norm(vec_to_starlink)
            dist_to_weather_sat = np.linalg.norm(vec_to_weather_sat)
            surface_normal = u_reflection
            incident_dir = -vec_to_starlink / dist_to_starlink
            reflected_dir = vec_to_weather_sat / dist_to_weather_sat
            cos_incidence = np.clip(np.dot(incident_dir, surface_normal), -1.0, 1.0)
            cos_reflection = np.clip(np.dot(reflected_dir, surface_normal), -1.0, 1.0)
            angle_incidence = np.arccos(cos_incidence)
            angle_reflection = np.arccos(cos_reflection)
            angle_error = abs(angle_incidence - angle_reflection)

    # Final path lengths
    path_starlink = np.linalg.norm(reflection_point - starlink_ecef)
    path_weather_sat = np.linalg.norm(weather_sat_ecef - reflection_point)

    # Validate result
    if path_starlink < 1e3 or path_weather_sat < 1e3:
        return None, 0.0, 0.0

    # Check visibility: both satellites should see the reflection point
    # (reflection point should be below horizon for both)
    # This is satisfied if the reflection point is on the Earth's surface
    # and both satellites are above it

    return reflection_point, path_starlink, path_weather_sat


def calculate_fresnel_reflection_coefficient(
    incidence_angle: float,
    surface_type: str = 'land',
    freq: float = 23.8e9,
    polarization: str = 'mixed'
) -> float:
    """
    Calculate Fresnel reflection coefficient for ground reflection.

    Args:
        incidence_angle: Incidence angle in degrees (0 = normal, 90 = grazing)
        surface_type: Surface type ('water', 'land', 'ice', 'mixed')
            - 'water': Sea water (high permittivity, high loss, strong reflection)
            - 'land': Typical land surface (soil, vegetation, moderate reflection)
            - 'ice': Ice/snow (low permittivity, very low loss, weak reflection)
            - 'mixed': Same as 'land' - represents typical/average terrestrial surface
              (good default when surface type is unknown or heterogeneous)
        freq: Frequency in Hz (for frequency-dependent permittivity)
        polarization: Polarization ('horizontal', 'vertical', 'mixed')
            'mixed' uses average of both polarizations

    Returns:
        float: Reflection coefficient (0-1, linear power)
    """
    if incidence_angle < 0 or incidence_angle >= 90:
        return 0.0

    # Convert to radians
    theta_i = np.deg2rad(incidence_angle)

    # Frequency-dependent permittivity (approximate)
    freq_ghz = freq / 1e9

    # Complex permittivity (epsilon = epsilon' - j*epsilon'')
    # Values at microwave frequencies
    if surface_type == 'water':
        # Sea water: high permittivity, frequency-dependent
        if freq_ghz < 30:
            epsilon_real = 70.0 - 0.3 * freq_ghz  # Real part
            epsilon_imag = 30.0 - 0.2 * freq_ghz   # Imaginary part (loss)
        else:
            epsilon_real = 60.0
            epsilon_imag = 25.0
    elif surface_type == 'ice':
        epsilon_real = 3.15
        epsilon_imag = 0.001  # Very low loss
    else:  # 'land' or 'mixed'
        # Typical land: soil, vegetation
        # Note: 'mixed' uses the same properties as 'land' - represents
        # a typical/average terrestrial surface (good default for heterogeneous areas)
        epsilon_real = 5.0 - 0.05 * freq_ghz
        epsilon_imag = 0.5

    # Complex permittivity
    epsilon = epsilon_real - 1j * epsilon_imag

    # Calculate reflection coefficients for both polarizations
    # Horizontal polarization (perpendicular to plane of incidence)
    # sin_theta_t = sin(theta_i) / sqrt(epsilon)
    sqrt_epsilon = np.sqrt(epsilon)  # Complex square root
    sin_theta_t = np.sin(theta_i) / sqrt_epsilon

    # Check for total internal reflection (shouldn't happen for air-to-ground)
    if np.abs(sin_theta_t) >= 1.0:
        # Total internal reflection
        R_h = 1.0
        R_v = 1.0
    else:
        cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)  # Complex sqrt

        # Horizontal polarization reflection coefficient
        numerator_h = np.cos(theta_i) - sqrt_epsilon * cos_theta_t
        denominator_h = np.cos(theta_i) + sqrt_epsilon * cos_theta_t
        R_h = np.abs(numerator_h / denominator_h)**2

        # Vertical polarization reflection coefficient
        numerator_v = sqrt_epsilon * np.cos(theta_i) - cos_theta_t
        denominator_v = sqrt_epsilon * np.cos(theta_i) + cos_theta_t
        R_v = np.abs(numerator_v / denominator_v)**2

    # Select polarization
    if polarization == 'horizontal':
        return R_h
    elif polarization == 'vertical':
        return R_v
    else:  # 'mixed' - average
        return 0.5 * (R_h + R_v)


def starlink_ground_reflection_to_weather_sat_link_budget(
    starlink_ecef: np.ndarray,
    weather_sat_ecef: np.ndarray,
    weather_sat_antenna: Antenna,
    starlink_antenna: Antenna,
    freq: float,
    starlink_fundamental_freq: float = None,
    harmonics: list = None,
    observation_bandwidth: float = None,
    surface_type: str = 'mixed',
    polarization_loss_factor: float = 0.5,
    include_atmospheric_loss: bool = True,
    use_enhanced_atmospheric: bool = True,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    humidity: float = 50.0,
    return_diagnostics: bool = False
) -> Tuple[float, Optional[dict]]:
    """
    Calculate link budget for Starlink ground reflection interference.

    Path: Starlink (main lobe) → Earth surface (reflection) → Weather satellite

    Args:
        starlink_ecef: Starlink position in ECEF [x, y, z] (meters)
        weather_sat_ecef: Weather satellite position in ECEF [x, y, z] (meters)
        weather_sat_antenna: Weather satellite antenna object
        starlink_antenna: Starlink antenna object
        freq: Observation frequency in Hz
        starlink_fundamental_freq: Starlink fundamental frequency (Hz)
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples
        observation_bandwidth: Observation bandwidth (Hz)
        surface_type: Surface type ('water', 'land', 'ice', 'mixed')
        polarization_loss_factor: Polarization mismatch loss factor (linear, 0-1)
        include_atmospheric_loss: Whether to include atmospheric loss
        use_enhanced_atmospheric: Whether to use enhanced atmospheric model
        temperature: Temperature in Kelvin
        pressure: Pressure in Pa
        humidity: Relative humidity in %
        return_diagnostics: If True, return diagnostic information (default: False)

    Returns:
        float or Tuple[float, Optional[dict]]:
            - If return_diagnostics=False: link budget (dimensionless)
            - If return_diagnostics=True: (link_budget, diagnostics_dict)
              where diagnostics_dict contains:
                - 'valid': bool - whether reflection point is valid
                - 'starlink_angle_from_nadir': float - angle in degrees
                - 'incidence_angle': float - incidence angle in degrees
                - 'path_starlink': float - path length in meters
                - 'path_weather_sat': float - path length in meters
                - 'reflection_coeff': float - Fresnel reflection coefficient
                - 'gain_starlink': float - Starlink gain (linear)
                - 'gain_weather_sat': float - Weather sat gain (linear)
    """
    # Calculate reflection point
    reflection_point, path_starlink, path_weather_sat = (
        calculate_specular_reflection_point(
            starlink_ecef, weather_sat_ecef
        )
    )

    # Initialize diagnostics
    diagnostics = None
    if return_diagnostics:
        diagnostics = {
            'valid': reflection_point is not None,
            'starlink_angle_from_nadir': 0.0,
            'incidence_angle': 0.0,
            'path_starlink': path_starlink,
            'path_weather_sat': path_weather_sat,
            'reflection_coeff': 0.0,
            'gain_starlink': 0.0,
            'gain_weather_sat': 0.0
        }

    if reflection_point is None:
        if return_diagnostics:
            return 0.0, diagnostics
        return 0.0

    # Transform reflection point to weather satellite frame
    weather_sat_velocity_ecef = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    reflection_dec, reflection_caz = ecef_to_weather_sat_frame(
        reflection_point[np.newaxis, :],
        weather_sat_ecef[np.newaxis, :],
        weather_sat_velocity_ecef[np.newaxis, :]
    )
    reflection_dec = reflection_dec[0]
    reflection_caz = reflection_caz[0]

    # Weather satellite receive gain (main lobe looking at Earth, at reflection point)
    gain_weather_sat = weather_sat_antenna.get_gain_value(reflection_dec, reflection_caz)

    # Calculate incidence angle at reflection point
    # Angle between Starlink-to-reflection vector and surface normal
    surface_normal = reflection_point / np.linalg.norm(reflection_point)
    vec_starlink_to_reflection = reflection_point - starlink_ecef
    vec_starlink_to_reflection_unit = (
        vec_starlink_to_reflection / np.linalg.norm(vec_starlink_to_reflection)
    )

    # Incidence angle (angle from surface normal)
    cos_incidence = np.clip(np.dot(-vec_starlink_to_reflection_unit, surface_normal), -1.0, 1.0)
    incidence_angle_deg = np.rad2deg(np.arccos(cos_incidence))

    # Starlink main lobe gain (toward reflection point)
    # Transform reflection point to Starlink frame (simplified: assume Starlink pointing at nadir)
    starlink_to_reflection_vec = reflection_point - starlink_ecef
    starlink_to_reflection_dist = np.linalg.norm(starlink_to_reflection_vec)

    # For Starlink, assume it's pointing at Earth (nadir)
    # The reflection point is at some angle from nadir
    starlink_nadir = -starlink_ecef / np.linalg.norm(starlink_ecef)
    cos_starlink_angle = np.clip(
        np.dot(starlink_to_reflection_vec / starlink_to_reflection_dist, starlink_nadir),
        -1.0, 1.0
    )
    starlink_angle_from_nadir = np.rad2deg(np.arccos(cos_starlink_angle))

    # Starlink gain toward reflection point (main lobe, not backlobe)
    # Use the angle from nadir to get gain
    starlink_dec = np.deg2rad(starlink_angle_from_nadir)
    starlink_caz = 0.0  # Azimuth doesn't matter for symmetric pattern
    gain_starlink = starlink_antenna.get_gain_value(starlink_dec, starlink_caz)

    # Fresnel reflection coefficient
    reflection_coeff = calculate_fresnel_reflection_coefficient(
        incidence_angle_deg, surface_type, freq
    )

    # Update diagnostics if requested
    if return_diagnostics:
        diagnostics['starlink_angle_from_nadir'] = starlink_angle_from_nadir
        diagnostics['incidence_angle'] = incidence_angle_deg
        diagnostics['reflection_coeff'] = reflection_coeff
        diagnostics['gain_starlink'] = gain_starlink
        diagnostics['gain_weather_sat'] = gain_weather_sat

    # Free-space path loss for both legs
    if starlink_fundamental_freq is not None:
        freq_for_path_loss = starlink_fundamental_freq
    else:
        freq_for_path_loss = freq

    L_fs_starlink = free_space_loss(path_starlink, freq_for_path_loss)
    L_fs_weather_sat = free_space_loss(path_weather_sat, freq)

    # Atmospheric loss (only on ground-to-space leg, not space-to-space)
    if include_atmospheric_loss:
        # Calculate elevation angle from ground (reflection point) to weather sat
        # Elevation = 90° - angle from nadir
        elevation_from_ground = 90.0 - np.rad2deg(reflection_dec)

        if use_enhanced_atmospheric:
            # Use enhanced atmospheric model
            L_atm = calculate_comprehensive_atmospheric_loss(
                path_weather_sat, freq, elevation_from_ground,
                temperature, pressure, humidity
            )
        else:
            # Use simplified atmospheric model
            L_atm = calculate_atmospheric_loss(path_weather_sat, freq, elevation_from_ground)
    else:
        L_atm = 1.0

    # Base link budget at fundamental frequency
    base_link_budget = (
        gain_weather_sat *
        gain_starlink *
        reflection_coeff *
        (1.0 / L_fs_starlink) *
        (1.0 / L_fs_weather_sat) *
        (1.0 / L_atm) *
        polarization_loss_factor
    )

    # Check if fundamental falls within observation bandwidth
    if (starlink_fundamental_freq is not None and observation_bandwidth is not None):
        freq_min = freq - observation_bandwidth / 2
        freq_max = freq + observation_bandwidth / 2

        if freq_min <= starlink_fundamental_freq <= freq_max:
            link_budget = base_link_budget
        else:
            link_budget = 0.0

        # Add harmonic contributions
        if harmonics is not None:
            for harmonic_mult, harmonic_factor in harmonics:
                harmonic_freq = starlink_fundamental_freq * harmonic_mult

                if freq_min <= harmonic_freq <= freq_max:
                    # Recalculate path losses at harmonic frequency
                    L_fs_starlink_harm = free_space_loss(path_starlink, harmonic_freq)
                    L_fs_weather_sat_harm = free_space_loss(path_weather_sat, harmonic_freq)

                    if include_atmospheric_loss:
                        if use_enhanced_atmospheric:
                            L_atm_harm = calculate_comprehensive_atmospheric_loss(
                                path_weather_sat, harmonic_freq, elevation_from_ground,
                                temperature, pressure, humidity
                            )
                        else:
                            L_atm_harm = calculate_atmospheric_loss(
                                path_weather_sat, harmonic_freq, elevation_from_ground
                            )
                    else:
                        L_atm_harm = 1.0

                    # Reflection coefficient at harmonic frequency
                    reflection_coeff_harm = calculate_fresnel_reflection_coefficient(
                        incidence_angle_deg, surface_type, harmonic_freq
                    )

                    harmonic_link_budget = (
                        gain_weather_sat *
                        gain_starlink *
                        reflection_coeff_harm *
                        (1.0 / L_fs_starlink_harm) *
                        (1.0 / L_fs_weather_sat_harm) *
                        (1.0 / L_atm_harm) *
                        polarization_loss_factor *
                        harmonic_factor
                    )

                    link_budget += harmonic_link_budget
    else:
        link_budget = base_link_budget

    if return_diagnostics:
        return link_budget, diagnostics
    else:
        return link_budget


def model_weather_sat_observed_power_phase3(
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
    ground_emitters: pd.DataFrame = None,
    ground_emitter_antenna: Antenna = None,
    ground_emitter_eirp_dbw: float = 30.0,
    earth_brightness_temp: float = 280.0,
    sky_brightness_temp: float = 2.73,
    system_temp: float = 300.0,
    starlink_eirp_dbw: float = 40.0,
    enable_terrain_masking: bool = True,
    include_atmospheric_loss: bool = True,
    use_enhanced_atmospheric: bool = True,
    dem_file: Optional[str] = None,
    polarization_loss_factor: float = 0.5,
    starlink_fundamental_freq: float = None,
    harmonics: list = None,
    ground_emitter_fundamental_freq: float = None,
    ground_emitter_harmonics: list = None,
    ground_emitter_oobe_suppression_db: float = None,
    ground_emitter_oobe_freq_offset_max: float = None,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    humidity: float = 50.0,
    include_refraction: bool = False,
    include_ground_reflection: bool = True,
    surface_type: str = 'mixed'
) -> dict:
    """
    Model observed power at weather satellite from all RFI sources (Phase 3: enhanced atmospheric effects).

    This extends Phase 2 with enhanced atmospheric modeling:
    - Comprehensive atmospheric absorption (separate oxygen and water vapor)
    - Atmospheric refraction effects (optional)
    - Temperature, pressure, and humidity-dependent modeling

    Args:
        weather_sat_trajectory: Weather satellite trajectory
        weather_sat_instrument: Weather satellite instrument
        starlink_constellation: Starlink constellation
        observation_times: Array of observation timestamps
        observer_lat: Observer latitude (degrees)
        observer_lon: Observer longitude (degrees)
        observer_alt: Observer altitude (meters)
        target_lat: Target latitude (degrees) - center of resolution element
        target_lon: Target longitude (degrees)
        target_alt: Target altitude (meters)
        freq_channels: Frequency channels in Hz
        ground_emitters: DataFrame with ground emitter positions (columns: 'lat', 'lon', 'alt')
        ground_emitter_antenna: Ground emitter antenna object
        ground_emitter_eirp_dbw: Ground emitter EIRP in dBW
        earth_brightness_temp: Earth brightness temperature (K)
        sky_brightness_temp: Sky background temperature (K)
        system_temp: System temperature (K)
        starlink_eirp_dbw: Starlink EIRP in dBW
        enable_terrain_masking: Whether to check horizon visibility for ground emitters
        include_atmospheric_loss: Whether to include atmospheric absorption loss
        use_enhanced_atmospheric: If True, use Phase 3 enhanced atmospheric model.
            If False, use Phase 2 simplified model (for comparison).
        dem_file: Path to DEM GeoTIFF file for terrain masking (optional)
        polarization_loss_factor: Polarization mismatch loss factor (linear, 0-1).
            Default 0.5 = -3 dB for circular (Starlink) to linear (Suomi-NPP) mismatch.
        starlink_fundamental_freq: Starlink fundamental frequency (Hz). If None,
            harmonics are not calculated.
        harmonics: List of (frequency_multiplier, power_reduction_factor) tuples.
            Example: [(2.0, 0.01), (3.0, 0.003), (4.0, 0.001)]
        ground_emitter_fundamental_freq: Ground emitter fundamental frequency (Hz). If None,
            interference is calculated at observation frequency (legacy behavior).
        ground_emitter_harmonics: List of (frequency_multiplier, power_reduction_factor) tuples
            for ground emitter harmonics.
        ground_emitter_oobe_suppression_db: OOBE suppression level in dB relative to in-band power
        ground_emitter_oobe_freq_offset_max: Maximum frequency offset for OOBE consideration (Hz)
        temperature: Temperature in Kelvin (default: 288.15 K = 15°C)
        pressure: Pressure in Pa (default: 101325 Pa = 1 atm)
        humidity: Relative humidity in % (default: 50%)
        include_refraction: Whether to include atmospheric refraction effects (default: False)
        include_ground_reflection: Whether to include Starlink ground reflection effects (default: True)
        surface_type: Surface type for reflection ('water', 'land', 'ice', 'mixed') (default: 'mixed')
            - 'water': Sea water (strong reflection, especially at grazing angles)
            - 'land': Typical land surface (soil, vegetation, moderate reflection)
            - 'ice': Ice/snow (weak reflection)
            - 'mixed': Same as 'land' - typical/average terrestrial surface (good default)

    Returns:
        dict: Dictionary with total and individual components of observed power in dBW.
              Keys: 'total', 'starlink', 'ground_emitter', 'earth', 'sky', 'system'.
              Values: np.ndarray of shape (n_times, n_freqs)
    """
    # Phase 3 extends Phase 2, so we call phase2 but replace atmospheric loss calculations
    # For now, we'll modify the phase2 function call to use enhanced atmospheric model
    # by passing a flag, but since phase2 doesn't have this, we'll need to modify
    # the atmospheric loss calculation within the phase2 function call

    # Call phase2 function first to get base results
    # We'll need to modify the atmospheric loss calculation after calling phase2
    # Actually, it's better to copy phase2 and modify it to use enhanced atmospheric model

    # For Phase 3, we'll use the same structure as phase2 but replace
    # calculate_atmospheric_loss() calls with calculate_comprehensive_atmospheric_loss()

    # Since we can't easily modify phase2 without changing it, we'll create
    # a modified version that uses enhanced atmospheric effects
    # The key difference is in the ground emitter link budget calculation

    n_times = len(observation_times)
    n_freqs = len(freq_channels)

    # Initialize DEM terrain masker if DEM file is provided
    dem_masker = None
    if enable_terrain_masking and dem_file is not None:
        dem_masker = DEMTerrainMasker(dem_file)
        if dem_masker.dem_data is None:
            print("  Warning: DEM loading failed, using geometric horizon check only")
    elif enable_terrain_masking:
        print("  Note: DEM file not provided, using geometric horizon check only")

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

    # Initialize result arrays for total and individual components
    result_power = np.zeros((n_times, n_freqs))  # Will store total power in W
    result_starlink = np.zeros((n_times, n_freqs))  # Starlink interference (backlobe)
    result_starlink_reflection = np.zeros((n_times, n_freqs))  # Starlink ground reflection
    result_ground_emitter = np.zeros((n_times, n_freqs))  # Ground emitter interference
    result_earth = np.zeros((n_times, n_freqs))  # Earth brightness
    result_sky = np.zeros((n_times, n_freqs))  # Sky background
    result_system = np.zeros((n_times, n_freqs))  # System noise

    print(f"  Processing {n_times} time steps and {n_freqs} frequency channels...")
    print(f"  Total Starlink satellites in constellation: {starlink_traj_df['sat'].nunique()}")
    if ground_emitters is not None:
        print(f"  Total ground emitters: {len(ground_emitters)}")
    if use_enhanced_atmospheric:
        print("  Using Phase 3 enhanced atmospheric model (comprehensive O2/H2O absorption)")
    else:
        print("  Using Phase 2 simplified atmospheric model (for comparison)")
    if include_ground_reflection:
        print(f"  Ground reflection enabled (surface type: {surface_type})")

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
            starlink_reflection_interference_temp = 0.0
            ground_emitter_interference_temp = 0.0

            # Process each Starlink satellite (same as phase2)
            if n_visible_starlinks > 0:
                for _, sat_row in starlink_sats.iterrows():
                    # Get Starlink position from trajectory
                    sat_dist = sat_row['distances']
                    # Trajectory 'elevations' = altitude above horizon; ENU uses polar from zenith
                    sat_polar_zenith_deg = 90.0 - sat_row['elevations']
                    sat_elev = np.deg2rad(sat_polar_zenith_deg)
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

                    # Ground reflection (if enabled)
                    if include_ground_reflection:
                        # Calculate direct path distance for diagnostic
                        sat_to_ws_dist_direct = np.linalg.norm(sat_ecef - ws_ecef)

                        reflection_link_budget, reflection_diagnostics = (
                            starlink_ground_reflection_to_weather_sat_link_budget(
                                sat_ecef,
                                ws_ecef,
                                weather_sat_antenna,
                                starlink_antenna,
                                freq,
                                starlink_fundamental_freq=starlink_fundamental_freq,
                                harmonics=harmonics,
                                observation_bandwidth=bandwidth,
                                surface_type=surface_type,
                                polarization_loss_factor=polarization_loss_factor,
                                include_atmospheric_loss=include_atmospheric_loss,
                                use_enhanced_atmospheric=use_enhanced_atmospheric,
                                temperature=temperature,
                                pressure=pressure,
                                humidity=humidity,
                                return_diagnostics=True
                            )
                        )

                        # Diagnostic output (first time step, first frequency, first satellite)
                        first_sat_check = (t_idx == 0 and f_idx == 0 and len(starlink_sats) > 0 and
                                           sat_row.name == starlink_sats.index[0])
                        if first_sat_check:
                            if reflection_diagnostics is not None:
                                print("      Ground reflection diagnostics (first Starlink, first time step):")
                                print(f"        Reflection point valid: {reflection_diagnostics['valid']}")
                                if reflection_diagnostics['valid']:
                                    print(f"        Starlink angle from nadir to reflection: "
                                          f"{reflection_diagnostics['starlink_angle_from_nadir']:.2f}°")
                                    print(f"        Incidence angle at reflection point: "
                                          f"{reflection_diagnostics['incidence_angle']:.2f}°")
                                    print(f"        Path length (Starlink → reflection): "
                                          f"{reflection_diagnostics['path_starlink']/1e3:.2f} km")
                                    print(f"        Path length (reflection → weather sat): "
                                          f"{reflection_diagnostics['path_weather_sat']/1e3:.2f} km")
                                    total_reflected_path = (
                                        reflection_diagnostics['path_starlink'] +
                                        reflection_diagnostics['path_weather_sat']
                                    )
                                    print(f"        Total reflected path: "
                                          f"{total_reflected_path/1e3:.2f} km")
                                    print(f"        Direct path (Starlink → weather sat): "
                                          f"{sat_to_ws_dist_direct/1e3:.2f} km")
                                    total_reflected = (reflection_diagnostics['path_starlink'] +
                                                       reflection_diagnostics['path_weather_sat'])
                                    path_ratio = total_reflected / sat_to_ws_dist_direct
                                    print(f"        Path length ratio (reflected/direct): {path_ratio:.3f}")
                                    refl_coeff = reflection_diagnostics['reflection_coeff']
                                    refl_coeff_db = (10*np.log10(refl_coeff)
                                                     if refl_coeff > 0 else -np.inf)
                                    print(f"        Reflection coefficient: "
                                          f"{reflection_diagnostics['reflection_coeff']:.4f} "
                                          f"({refl_coeff_db:.2f} dB)")
                                    gain_starlink_db = (10*np.log10(reflection_diagnostics['gain_starlink'])
                                                        if reflection_diagnostics['gain_starlink'] > 0 else -np.inf)
                                    gain_ws_db = (10*np.log10(reflection_diagnostics['gain_weather_sat'])
                                                  if reflection_diagnostics['gain_weather_sat'] > 0 else -np.inf)
                                    print(f"        Starlink gain toward reflection: "
                                          f"{gain_starlink_db:.2f} dBi")
                                    print(f"        Weather sat gain at reflection point: "
                                          f"{gain_ws_db:.2f} dBi")
                                else:
                                    print("        No valid reflection point found")

                        reflection_interference_temp = reflection_link_budget * starlink_temp
                        starlink_reflection_interference_temp += reflection_interference_temp

            # Process ground emitters (Phase 3: enhanced atmospheric model)
            if ground_emitters is not None and ground_emitter_antenna is not None:
                # Pre-cache terrain elevations for all emitters (once per emitter, reused across time steps)
                if t_idx == 0:
                    # Cache terrain elevations for all emitters (only compute once)
                    n_emitters = len(ground_emitters)
                    emitter_terrain_elevations = np.zeros(n_emitters)
                    if dem_masker is not None and dem_masker.dem_data is not None:
                        for i, (_, emitter_row) in enumerate(ground_emitters.iterrows()):
                            terrain_elev = dem_masker.get_terrain_elevation(
                                emitter_row['lat'],
                                emitter_row['lon']
                            )
                            if terrain_elev is not None:
                                emitter_terrain_elevations[i] = terrain_elev
                    # Store in ground_emitters DataFrame for reuse
                    ground_emitters['terrain_elev'] = emitter_terrain_elevations
                else:
                    # Retrieve cached terrain elevations
                    emitter_terrain_elevations = ground_emitters['terrain_elev'].values

                # Vectorized processing: convert all emitter positions to ECEF at once
                emitter_lats = ground_emitters['lat'].values
                emitter_lons = ground_emitters['lon'].values
                emitter_alts_above_ground = ground_emitters['alt'].values
                emitter_total_alts = emitter_terrain_elevations + emitter_alts_above_ground
                emitter_ecef_all = latlonalt_to_ecef_vectorized(
                    emitter_lats, emitter_lons, emitter_total_alts
                )
                # Ensure emitter_ecef_all has shape [n_emitters, 3]
                emitter_ecef_all = np.asarray(emitter_ecef_all, dtype=np.float64)
                if emitter_ecef_all.ndim == 1:
                    emitter_ecef_all = emitter_ecef_all.reshape(1, 3)
                elif emitter_ecef_all.ndim == 2 and emitter_ecef_all.shape[1] != 3:
                    emitter_ecef_all = emitter_ecef_all.reshape(-1, 3)

                # Vectorized visibility check (same as phase2)
                n_emitters = len(ground_emitters)
                visibility_mask = np.ones(n_emitters, dtype=bool)

                if enable_terrain_masking:
                    # Stage 1: Vectorized geometric horizon check for all emitters
                    ws_vec = np.asarray(ws_ecef, dtype=np.float64).flatten()
                    if len(ws_vec) != 3:
                        raise ValueError(f"ws_ecef must have 3 elements, got {len(ws_vec)}")
                    ws_vec = ws_vec.reshape(1, 3)
                    ws_distance = float(np.linalg.norm(ws_vec))

                    emitter_vecs = np.asarray(emitter_ecef_all, dtype=np.float64)
                    if emitter_vecs.ndim == 1:
                        if len(emitter_vecs) == 3:
                            emitter_vecs = emitter_vecs.reshape(1, 3)
                        else:
                            emitter_vecs = emitter_vecs.reshape(-1, 3)
                    elif emitter_vecs.ndim == 2:
                        if emitter_vecs.shape[1] != 3:
                            emitter_vecs = emitter_vecs.reshape(-1, 3)

                    sat_to_emitter_vecs = emitter_vecs - ws_vec
                    if sat_to_emitter_vecs.ndim == 1:
                        sat_to_emitter_vecs = sat_to_emitter_vecs.reshape(1, 3)
                    elif sat_to_emitter_vecs.ndim == 2 and sat_to_emitter_vecs.shape[1] != 3:
                        sat_to_emitter_vecs = sat_to_emitter_vecs.reshape(-1, 3)
                    sat_to_emitter_vecs = np.ascontiguousarray(sat_to_emitter_vecs)
                    sat_to_emitter_dists = np.linalg.norm(sat_to_emitter_vecs, axis=1)

                    dot_products = np.sum(-ws_vec * sat_to_emitter_vecs, axis=1)
                    cos_angles = dot_products / (ws_distance * sat_to_emitter_dists)
                    cos_angles = np.clip(cos_angles, -1.0, 1.0)
                    angles = np.arccos(cos_angles)

                    sin_horizon_angle = R_earth / ws_distance
                    horizon_angle = np.arcsin(np.clip(sin_horizon_angle, 0.0, 1.0))

                    visibility_mask = angles < horizon_angle
                    n_geometric_visible = np.sum(visibility_mask)

                    # Stage 2: DEM ray tracing only for emitters that passed geometric check
                    n_dem_checked = 0
                    n_dem_blocked = 0
                    n_high_elevation = 0

                    if dem_masker is not None and dem_masker.dem_data is not None:
                        visible_indices = np.where(visibility_mask)[0]
                        if len(visible_indices) > 0:
                            visible_angles_deg = np.rad2deg(angles[visible_indices])
                            elevation_angles_deg = 90.0 - visible_angles_deg

                            high_elevation_threshold_deg = 30.0
                            high_elevation_mask = elevation_angles_deg > high_elevation_threshold_deg
                            n_high_elevation = np.sum(high_elevation_mask)

                            low_elevation_indices = visible_indices[~high_elevation_mask]
                            n_dem_checked = len(low_elevation_indices)

                            for i in low_elevation_indices:
                                if not dem_masker.check_line_of_sight_dem(
                                    emitter_ecef_all[i],
                                    ws_ecef,
                                    emitter_alts_above_ground[i],
                                    num_points=10
                                ):
                                    visibility_mask[i] = False
                                    n_dem_blocked += 1
                else:
                    n_geometric_visible = n_emitters
                    n_dem_checked = 0
                    n_dem_blocked = 0
                    n_high_elevation = 0

                # Print detailed visibility statistics at first time step
                if t_idx == 0:
                    print("    Ground emitter visibility statistics (first time step):")
                    print(f"      Total emitters: {len(ground_emitters)}")
                    if enable_terrain_masking:
                        print(f"      Geometric horizon check: {n_geometric_visible}/{len(ground_emitters)} passed")
                        if dem_masker is not None and dem_masker.dem_data is not None:
                            print(f"      High elevation (>30°): {n_high_elevation} (skipped DEM ray tracing)")
                            print(f"      DEM ray tracing checked: {n_dem_checked}")
                            print(f"      Terrain-masked (blocked): {n_dem_blocked}")
                        else:
                            print("      DEM not available - using geometric check only")
                    else:
                        print("      Terrain masking disabled - all emitters assumed visible")

                # Filter to visible emitters only
                visible_indices = np.where(visibility_mask)[0]
                n_visible_emitters = len(visible_indices)

                if t_idx == 0:
                    print(f"      Final visible: {n_visible_emitters}/{len(ground_emitters)}")

                if n_visible_emitters > 0:
                    # Calculate elevation angles for diagnostic output (first time step, first frequency only)
                    if t_idx == 0 and f_idx == 0:
                        # Calculate elevation angles for all visible emitters
                        emitter_ecef_visible_diag = emitter_ecef_all[visible_indices]
                        n_visible_diag = len(visible_indices)
                        if emitter_ecef_visible_diag.ndim == 1:
                            emitter_ecef_visible_diag = emitter_ecef_visible_diag.reshape(1, 3)
                        elif emitter_ecef_visible_diag.ndim == 2 and emitter_ecef_visible_diag.shape[1] != 3:
                            emitter_ecef_visible_diag = emitter_ecef_visible_diag.reshape(n_visible_diag, 3)
                        emitter_ecef_visible_diag = np.ascontiguousarray(
                            emitter_ecef_visible_diag.astype(np.float64)
                        )
                        weather_sat_velocity_ecef_diag = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                        ws_ecef_array_diag = np.asarray(ws_ecef, dtype=np.float64).reshape(1, 3)
                        emitter_decs_diag, _ = ecef_to_weather_sat_frame(
                            emitter_ecef_visible_diag,
                            np.tile(ws_ecef_array_diag, (n_visible_diag, 1)),
                            np.tile(weather_sat_velocity_ecef_diag.reshape(1, 3), (n_visible_diag, 1))
                        )
                        # Elevation from ground emitter perspective: elevation = π/2 - dec
                        # (dec is angle from nadir in weather sat frame: 0 = nadir, π/2 = horizon)
                        elevation_angles_diag_deg = np.rad2deg(np.pi / 2 - emitter_decs_diag)
                        print("      Ground emitter elevation angles (from ground, first time step):")
                        print(f"        Min: {np.min(elevation_angles_diag_deg):.2f}°")
                        print(f"        Max: {np.max(elevation_angles_diag_deg):.2f}°")
                        print(f"        Mean: {np.mean(elevation_angles_diag_deg):.2f}°")
                        print(f"        Median: {np.median(elevation_angles_diag_deg):.2f}°")
                        print(f"        Std: {np.std(elevation_angles_diag_deg):.2f}°")
                        # Count emitters in different elevation ranges
                        n_low_elev = np.sum(elevation_angles_diag_deg < 10.0)
                        n_med_elev = np.sum(
                            (elevation_angles_diag_deg >= 10.0) &
                            (elevation_angles_diag_deg < 30.0)
                        )
                        n_high_elev = np.sum(elevation_angles_diag_deg >= 30.0)
                        print("        Distribution:")
                        print(f"          < 10° (low): {n_low_elev} emitters "
                              f"({100*n_low_elev/n_visible_emitters:.1f}%)")
                        print(f"          10-30° (medium): {n_med_elev} emitters "
                              f"({100*n_med_elev/n_visible_emitters:.1f}%)")
                        print(f"          ≥ 30° (high): {n_high_elev} emitters "
                              f"({100*n_high_elev/n_visible_emitters:.1f}%)")

                    # Vectorized coordinate transformation for visible emitters
                    emitter_ecef_visible = emitter_ecef_all[visible_indices]
                    n_visible = len(visible_indices)

                    if emitter_ecef_visible.ndim == 1:
                        emitter_ecef_visible = emitter_ecef_visible.reshape(n_visible, 3)
                    elif emitter_ecef_visible.ndim == 2 and emitter_ecef_visible.shape[1] != 3:
                        emitter_ecef_visible = emitter_ecef_visible.reshape(n_visible, 3)
                    emitter_ecef_visible = np.ascontiguousarray(
                        emitter_ecef_visible.astype(np.float64)
                    )

                    # Vectorized link budget calculation
                    weather_sat_velocity_ecef = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    ws_ecef_array = np.asarray(ws_ecef, dtype=np.float64).reshape(1, 3)

                    emitter_decs, emitter_cazs = ecef_to_weather_sat_frame(
                        emitter_ecef_visible,
                        np.tile(ws_ecef_array, (n_visible, 1)),
                        np.tile(weather_sat_velocity_ecef.reshape(1, 3), (n_visible, 1))
                    )

                    # Calculate distances (vectorized)
                    sat_to_emitter_vecs = emitter_ecef_visible - ws_ecef_array
                    sat_to_emitter_dists = np.linalg.norm(sat_to_emitter_vecs, axis=1)

                    # Vectorized antenna gain calculations
                    gain_weather_sat_visible = weather_sat_antenna.get_gain_values(
                        emitter_decs, emitter_cazs
                    )

                    # Get valid angle ranges for emitter antenna
                    valid_alphas, valid_betas = ground_emitter_antenna.get_def_angles()
                    alpha_min_deg, alpha_max_deg = valid_alphas.min(), valid_alphas.max()
                    beta_min_deg, beta_max_deg = valid_betas.min(), valid_betas.max()
                    alpha_min_rad = np.deg2rad(alpha_min_deg)
                    alpha_max_rad = np.deg2rad(alpha_max_deg)
                    beta_min_rad = np.deg2rad(beta_min_deg)
                    beta_max_rad = np.deg2rad(beta_max_deg)

                    # Clamp angles for emitter antenna (vectorized)
                    emitter_alphas_rad = np.clip(emitter_decs, alpha_min_rad, alpha_max_rad)
                    emitter_betas_rad = emitter_cazs % (2 * np.pi)
                    emitter_betas_rad = np.where(
                        emitter_betas_rad < 0,
                        emitter_betas_rad + 2 * np.pi,
                        emitter_betas_rad
                    )
                    if beta_max_deg >= 360.0:
                        emitter_betas_rad = np.clip(emitter_betas_rad, beta_min_rad, beta_max_rad)
                    else:
                        emitter_betas_rad = np.clip(emitter_betas_rad, beta_min_rad, beta_max_rad)

                    # Get emitter antenna gains (vectorized)
                    gain_emitter_visible_absolute = ground_emitter_antenna.get_gain_values(
                        emitter_alphas_rad, emitter_betas_rad
                    )
                    peak_gain_emitter = ground_emitter_antenna.get_boresight_gain()
                    gain_emitter_visible = gain_emitter_visible_absolute / peak_gain_emitter

                    # Check if we have fundamental frequency and bandwidth information
                    if (ground_emitter_fundamental_freq is not None and bandwidth is not None):
                        # Calculate observation bandwidth bounds
                        freq_min = freq - bandwidth / 2
                        freq_max = freq + bandwidth / 2

                        # Calculate base link budget at fundamental frequency
                        speed_c = 3e8
                        wavelength_fund = speed_c / ground_emitter_fundamental_freq
                        L_fs_fund_visible = (4 * np.pi * sat_to_emitter_dists / wavelength_fund) ** 2

                        # Atmospheric loss at fundamental frequency (PHASE 3: enhanced model)
                        if include_atmospheric_loss:
                            # Calculate elevation angles from ground emitter perspective
                            # emitter_alphas_rad: angle from nadir in weather sat frame (0 = nadir, π/2 = horizon)
                            # For ground emitter looking up: elevation = π/2 - alpha
                            elevation_angles_deg = np.rad2deg(np.pi / 2 - emitter_alphas_rad)

                            if use_enhanced_atmospheric:
                                # Phase 3: Use comprehensive atmospheric loss (vectorized)
                                L_atm_fund_visible = calculate_comprehensive_atmospheric_loss_vectorized(
                                    sat_to_emitter_dists,
                                    np.full(n_visible, ground_emitter_fundamental_freq),
                                    elevation_angles_deg,
                                    temperature,
                                    pressure,
                                    humidity
                                )
                            else:
                                # Phase 2: Use simplified atmospheric loss (vectorized)
                                freq_ghz_fund = ground_emitter_fundamental_freq / 1e9
                                if freq_ghz_fund < 20:
                                    absorption_db_per_km = 0.01
                                elif freq_ghz_fund < 40:
                                    absorption_db_per_km = 0.2 + 0.3 * ((freq_ghz_fund - 20) / 20.0)
                                elif freq_ghz_fund < 60:
                                    oxygen_band_center = 60.0
                                    distance_from_peak = abs(freq_ghz_fund - oxygen_band_center)
                                    if distance_from_peak < 10:
                                        absorption_db_per_km = 10.0 - 8.0 * (distance_from_peak / 10.0)
                                    else:
                                        absorption_db_per_km = 2.0 + 1.0 * ((freq_ghz_fund - 40) / 20.0)
                                else:
                                    absorption_db_per_km = 15.0

                                elev_rad = np.deg2rad(elevation_angles_deg)
                                effective_path_multiplier = 1.0 / np.maximum(np.sin(elev_rad), 0.1)
                                effective_path_multiplier = np.minimum(effective_path_multiplier, 3.0)
                                atmospheric_path_km = 25.0
                                total_loss_db = absorption_db_per_km * atmospheric_path_km * effective_path_multiplier
                                L_atm_fund_visible = 10 ** (total_loss_db / 10.0)
                        else:
                            L_atm_fund_visible = np.ones(n_visible)

                        base_link_budgets = (
                            gain_weather_sat_visible *
                            (1.0 / L_fs_fund_visible) *
                            (1.0 / L_atm_fund_visible) *
                            gain_emitter_visible
                        )

                        # Check if fundamental falls within observation bandwidth
                        fundamental_in_band = (freq_min <= ground_emitter_fundamental_freq <= freq_max)
                        if fundamental_in_band:
                            link_budgets = base_link_budgets
                        else:
                            link_budgets = np.zeros(n_visible)

                        # Add harmonic contributions if harmonics are provided
                        if ground_emitter_harmonics is not None:
                            L_fundamental = L_fs_fund_visible
                            harmonic_contributions = np.zeros(n_visible)

                            for freq_mult, power_red in ground_emitter_harmonics:
                                harmonic_frequency = ground_emitter_fundamental_freq * freq_mult

                                # Check if harmonic falls within observation band
                                if freq_min <= harmonic_frequency <= freq_max:
                                    # Calculate path loss at harmonic frequency
                                    wavelength_harm = speed_c / harmonic_frequency
                                    L_harmonic = (4 * np.pi * sat_to_emitter_dists / wavelength_harm) ** 2

                                    # Atmospheric loss at harmonic frequency (PHASE 3: enhanced model)
                                    if include_atmospheric_loss:
                                        if use_enhanced_atmospheric:
                                            # Phase 3: Use comprehensive atmospheric loss (vectorized)
                                            L_atm_harm = calculate_comprehensive_atmospheric_loss_vectorized(
                                                sat_to_emitter_dists,
                                                np.full(n_visible, harmonic_frequency),
                                                elevation_angles_deg,
                                                temperature,
                                                pressure,
                                                humidity
                                            )
                                        else:
                                            # Phase 2: Use simplified atmospheric loss (vectorized)
                                            freq_ghz_harm = harmonic_frequency / 1e9
                                            if freq_ghz_harm < 20:
                                                absorption_db_per_km = 0.01
                                            elif freq_ghz_harm < 40:
                                                absorption_db_per_km = 0.2 + 0.3 * ((freq_ghz_harm - 20) / 20.0)
                                            elif freq_ghz_harm < 60:
                                                oxygen_band_center = 60.0
                                                distance_from_peak = abs(freq_ghz_harm - oxygen_band_center)
                                                if distance_from_peak < 10:
                                                    absorption_db_per_km = 10.0 - 8.0 * (distance_from_peak / 10.0)
                                                else:
                                                    absorption_db_per_km = 2.0 + 1.0 * ((freq_ghz_harm - 40) / 20.0)
                                            else:
                                                absorption_db_per_km = 15.0

                                            elev_rad = np.deg2rad(elevation_angles_deg)
                                            effective_path_multiplier = 1.0 / np.maximum(np.sin(elev_rad), 0.1)
                                            effective_path_multiplier = np.minimum(
                                                effective_path_multiplier, 3.0)
                                            atmospheric_path_km = 25.0
                                            total_loss_db = (absorption_db_per_km *
                                                             atmospheric_path_km *
                                                             effective_path_multiplier)
                                            L_atm_harm = 10 ** (total_loss_db / 10.0)
                                    else:
                                        L_atm_harm = np.ones(n_visible)

                                    # Path loss ratio (fundamental vs harmonic)
                                    path_loss_ratio = L_fundamental / L_harmonic

                                    # Harmonic link budget contribution
                                    harmonic_link_budgets = (
                                        base_link_budgets *
                                        power_red *
                                        path_loss_ratio *
                                        (L_atm_fund_visible / L_atm_harm)  # Atmospheric loss ratio
                                    )
                                    harmonic_contributions += harmonic_link_budgets

                            # Add harmonic contribution
                            link_budgets += harmonic_contributions

                        # Add OOBE contribution if OOBE parameters are provided
                        if (ground_emitter_oobe_suppression_db is not None and
                                ground_emitter_oobe_freq_offset_max is not None):
                            fundamental_in_band = (freq_min <= ground_emitter_fundamental_freq <= freq_max)

                            if not fundamental_in_band:
                                # Calculate frequency offset from observation band
                                if ground_emitter_fundamental_freq < freq_min:
                                    freq_offset = freq_min - ground_emitter_fundamental_freq
                                else:
                                    freq_offset = ground_emitter_fundamental_freq - freq_max

                                # Check if within OOBE range
                                if freq_offset <= ground_emitter_oobe_freq_offset_max:
                                    # Calculate path loss at observation frequency
                                    wavelength_obs = speed_c / freq
                                    L_fs_obs = (4 * np.pi * sat_to_emitter_dists / wavelength_obs) ** 2

                                    # Atmospheric loss at observation frequency (PHASE 3: enhanced model)
                                    if include_atmospheric_loss:
                                        if use_enhanced_atmospheric:
                                            # Phase 3: Use comprehensive atmospheric loss (vectorized)
                                            L_atm_obs = calculate_comprehensive_atmospheric_loss_vectorized(
                                                sat_to_emitter_dists,
                                                np.full(n_visible, freq),
                                                elevation_angles_deg,
                                                temperature,
                                                pressure,
                                                humidity
                                            )
                                        else:
                                            # Phase 2: Use simplified atmospheric loss (vectorized)
                                            freq_ghz_obs = freq / 1e9
                                            if freq_ghz_obs < 20:
                                                absorption_db_per_km = 0.01
                                            elif freq_ghz_obs < 40:
                                                absorption_db_per_km = 0.2 + 0.3 * ((freq_ghz_obs - 20) / 20.0)
                                            elif freq_ghz_obs < 60:
                                                oxygen_band_center = 60.0
                                                distance_from_peak = abs(freq_ghz_obs - oxygen_band_center)
                                                if distance_from_peak < 10:
                                                    absorption_db_per_km = 10.0 - 8.0 * (distance_from_peak / 10.0)
                                                else:
                                                    absorption_db_per_km = 2.0 + 1.0 * ((freq_ghz_obs - 40) / 20.0)
                                            else:
                                                absorption_db_per_km = 15.0

                                            elev_rad = np.deg2rad(elevation_angles_deg)
                                            effective_path_multiplier = 1.0 / np.maximum(np.sin(elev_rad), 0.1)
                                            effective_path_multiplier = np.minimum(
                                                effective_path_multiplier, 3.0)
                                            atmospheric_path_km = 25.0
                                            total_loss_db = (absorption_db_per_km *
                                                             atmospheric_path_km *
                                                             effective_path_multiplier)
                                            L_atm_obs = 10 ** (total_loss_db / 10.0)
                                    else:
                                        L_atm_obs = np.ones(n_visible)

                                    # OOBE link budget accounts for:
                                    # 1. OOBE suppression factor
                                    # 2. Path loss ratio (observation vs fundamental)
                                    # 3. Atmospheric loss ratio (observation vs fundamental)
                                    path_loss_ratio = L_fs_fund_visible / L_fs_obs
                                    atm_loss_ratio = L_atm_fund_visible / L_atm_obs
                                    oobe_suppression_linear = 10 ** (ground_emitter_oobe_suppression_db / 10.0)

                                    # Calculate OOBE contribution
                                    oobe_link_budgets = (
                                        base_link_budgets *
                                        oobe_suppression_linear *
                                        path_loss_ratio *
                                        atm_loss_ratio
                                    )

                                    # Add OOBE contribution
                                    link_budgets += oobe_link_budgets
                    else:
                        # No fundamental frequency info: calculate base link budget at observation frequency
                        # Vectorized free-space path loss
                        speed_c = 3e8
                        wavelength = speed_c / freq
                        L_fs_visible = (4 * np.pi * sat_to_emitter_dists / wavelength) ** 2

                        # Vectorized atmospheric loss (PHASE 3: enhanced model)
                        if include_atmospheric_loss:
                            elevation_angles_deg = np.rad2deg(np.pi / 2 - emitter_alphas_rad)

                            if use_enhanced_atmospheric:
                                # Phase 3: Use comprehensive atmospheric loss (vectorized)
                                L_atm_visible = calculate_comprehensive_atmospheric_loss_vectorized(
                                    sat_to_emitter_dists,
                                    np.full(n_visible, freq),
                                    elevation_angles_deg,
                                    temperature,
                                    pressure,
                                    humidity
                                )
                            else:
                                # Phase 2: Use simplified atmospheric loss (vectorized)
                                freq_ghz = freq / 1e9
                                if freq_ghz < 20:
                                    absorption_db_per_km = 0.01
                                elif freq_ghz < 40:
                                    absorption_db_per_km = 0.2 + 0.3 * ((freq_ghz - 20) / 20.0)
                                elif freq_ghz < 60:
                                    oxygen_band_center = 60.0
                                    distance_from_peak = abs(freq_ghz - oxygen_band_center)
                                    if distance_from_peak < 10:
                                        absorption_db_per_km = 10.0 - 8.0 * (distance_from_peak / 10.0)
                                    else:
                                        absorption_db_per_km = 2.0 + 1.0 * ((freq_ghz - 40) / 20.0)
                                else:
                                    absorption_db_per_km = 15.0

                                elev_rad = np.deg2rad(elevation_angles_deg)
                                effective_path_multiplier = 1.0 / np.maximum(np.sin(elev_rad), 0.1)
                                effective_path_multiplier = np.minimum(effective_path_multiplier, 3.0)
                                atmospheric_path_km = 25.0
                                total_loss_db = absorption_db_per_km * atmospheric_path_km * effective_path_multiplier
                                L_atm_visible = 10 ** (total_loss_db / 10.0)
                        else:
                            L_atm_visible = np.ones(n_visible)

                        # Vectorized link budget calculation
                        link_budgets = (
                            gain_weather_sat_visible *
                            (1.0 / L_fs_visible) *
                            (1.0 / L_atm_visible) *
                            gain_emitter_visible
                        )

                    # Ground emitter transmit power (convert to temperature)
                    emitter_power_w = 10**(ground_emitter_eirp_dbw / 10.0)
                    emitter_temp = power_to_temperature(emitter_power_w, bandwidth)

                    # Vectorized interference temperature
                    interference_temps = link_budgets * emitter_temp
                    ground_emitter_interference_temp = np.sum(interference_temps)
                else:
                    # No visible emitters
                    ground_emitter_interference_temp = 0.0

            # Convert interference to power
            starlink_power = temperature_to_power(starlink_interference_temp, bandwidth)
            starlink_reflection_power = temperature_to_power(
                starlink_reflection_interference_temp, bandwidth
            )
            ground_emitter_power = temperature_to_power(ground_emitter_interference_temp, bandwidth)

            # Earth brightness (through main lobe pointing at target)
            earth_temp_freq = calculate_earth_brightness_temperature(freq, earth_brightness_temp)
            earth_gain = weather_sat_antenna.get_gain_value(target_dec, target_caz)
            earth_power = temperature_to_power(earth_temp_freq, bandwidth) * earth_gain

            # Sky background (through sidelobes - use average gain)
            sky_power = temperature_to_power(sky_brightness_temp, bandwidth) * 0.1

            # System noise
            system_power = temperature_to_power(system_temp, bandwidth)

            # Store individual components
            result_starlink[t_idx, f_idx] = starlink_power
            result_starlink_reflection[t_idx, f_idx] = starlink_reflection_power
            result_ground_emitter[t_idx, f_idx] = ground_emitter_power
            result_earth[t_idx, f_idx] = earth_power
            result_sky[t_idx, f_idx] = sky_power
            result_system[t_idx, f_idx] = system_power

            # Total power
            result_power[t_idx, f_idx] = (
                starlink_power + starlink_reflection_power + ground_emitter_power +
                earth_power + sky_power + system_power
            )

    # Convert to dBW
    result_power_dbw = 10 * np.log10(result_power + 1e-100)
    result_starlink_dbw = 10 * np.log10(result_starlink + 1e-100)
    result_starlink_reflection_dbw = 10 * np.log10(result_starlink_reflection + 1e-100)
    result_ground_emitter_dbw = 10 * np.log10(result_ground_emitter + 1e-100)
    result_earth_dbw = 10 * np.log10(result_earth + 1e-100)
    result_sky_dbw = 10 * np.log10(result_sky + 1e-100)
    result_system_dbw = 10 * np.log10(result_system + 1e-100)

    # Return dictionary with total and individual components
    return {
        'total': result_power_dbw,
        'starlink': result_starlink_dbw,
        'starlink_reflection': result_starlink_reflection_dbw,
        'ground_emitter': result_ground_emitter_dbw,
        'earth': result_earth_dbw,
        'sky': result_sky_dbw,
        'system': result_system_dbw
    }
