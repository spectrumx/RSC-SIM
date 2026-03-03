"""
Weather Satellite Modeling Functions for "Looking Down" RFI Analysis
for numerical Weather Prediction (NWP) simulation

This module provides additional functions for numerical Weather Prediction (NWP) simulation, including:
- Number of emitters per square km based on [lat, lon] coordinates
- Elliptical FOV ground emitter distribution for 5G mmWave small cells
- Vectorized RFI observation model for cross-track V-band sounders (ATMS, AMSU-A) with 5G ground emitters only.
- SSMI-S (conical scanning): FOV bearing, fixed V-band area, and emitter count helpers.
"""

import os
import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import reverse_geocoder as rg

# Map reverse_geocoder's ISO 3166-1 alpha-2 codes to your 5G list
SUPPORTED_5G_COUNTRIES = {
    'IT': 'Italy',
    'AE': 'UAE',
    'PR': 'Puerto Rico',
    'US': 'USA',
    'AU': 'Australia',
    'JP': 'Japan',
    'IN': 'India'
}

# Sensor name -> antenna beamwidth (deg) for FOV dimension calculations (cross-track scanners).
# ATMS V-band Ch 3–9: 2.2°; AMSU-A V-band Ch 3–8: 3.3°.
SENSOR_BEAMWIDTH_DEG = {
    "ATMS": 2.2,
    "AMSU-A": 3.3,
}


def _beamwidth_for_sensor(sensor_name):
    """Resolve beamwidth (deg) from sensor_name. Raises ValueError if sensor_name is missing or unknown."""
    if sensor_name is None:
        raise ValueError(
            "sensor_name is required for FOV beamwidth (no default). "
            f"Use one of: {list(SENSOR_BEAMWIDTH_DEG.keys())}"
        )
    key = str(sensor_name).strip()
    if key not in SENSOR_BEAMWIDTH_DEG:
        raise ValueError(
            f"Unknown sensor_name={sensor_name!r}. "
            f"Known sensors: {list(SENSOR_BEAMWIDTH_DEG.keys())}"
        )
    return SENSOR_BEAMWIDTH_DEG[key]


# SSMI-S (conical scanning): V-band FOV fixed dimensions and scan layout
# V-band channels 1-5, beamwidth 5°; elliptical footprint d_max x d_min (km)
SSMIS_VBAND_D_MAX_KM = 27.0
SSMIS_VBAND_D_MIN_KM = 18.0
SSMIS_VBAND_FOV_AREA_KM2 = np.pi * (SSMIS_VBAND_D_MAX_KM / 2.0) * (SSMIS_VBAND_D_MIN_KM / 2.0)
SSMIS_FOVS_PER_SCAN = 60


# GHSL population GeoTIFF (EU global health population data); resolved from project root.
# Resolve to absolute path so it works in multiprocessing workers (spawn on Windows).
_GHSL_TIF_NAME = "GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"
_GHSL_TIF_PATH = Path(__file__).resolve().parent.parent / "research_tutorials" / "data" / _GHSL_TIF_NAME
_GHSL_TIF_PATH_ABS = _GHSL_TIF_PATH.resolve()

# Optional override from environment (e.g. for multiprocessing when workers have different cwd)
_GHSL_TIF_PATH_RESOLVED = Path(os.environ.get("GHSL_TIF_PATH", str(_GHSL_TIF_PATH_ABS)))

# Cached raster data (loaded once per process on first use; avoids spawn issues from importing rasterio at module load)
_GHSL_RASTER = None
_GHSL_TRANSFORM = None
_GHSL_HEIGHT = None
_GHSL_WIDTH = None


def _load_ghsl_raster() -> tuple[np.ndarray, object, int, int]:
    """Load GHSL band and transform into memory once per process; return (raster_2d, transform, height, width)."""
    global _GHSL_RASTER, _GHSL_TRANSFORM, _GHSL_HEIGHT, _GHSL_WIDTH
    if _GHSL_RASTER is not None:
        return _GHSL_RASTER, _GHSL_TRANSFORM, _GHSL_HEIGHT, _GHSL_WIDTH
    if not _GHSL_TIF_PATH_RESOLVED.is_file():
        raise FileNotFoundError(
            f"GHSL GeoTIFF not found: {_GHSL_TIF_PATH_RESOLVED}. "
            "Ensure the file exists or set GHSL_TIF_PATH to an absolute path."
        )
    # Lazy import so rasterio/GDAL are not loaded at module import (avoids multiprocessing spawn failures on Windows)
    import rasterio
    with rasterio.open(str(_GHSL_TIF_PATH_RESOLVED)) as dataset:
        _GHSL_RASTER = np.asarray(dataset.read(1), dtype=np.float64)
        _GHSL_TRANSFORM = dataset.transform
        _GHSL_HEIGHT, _GHSL_WIDTH = _GHSL_RASTER.shape
    return _GHSL_RASTER, _GHSL_TRANSFORM, _GHSL_HEIGHT, _GHSL_WIDTH


def _rowcol(transform, lon: float, lat: float) -> tuple[int, int]:
    """Get (row, col) from (lon, lat); uses lazy import to avoid loading rasterio at module import."""
    from rasterio.transform import rowcol
    r, c = rowcol(transform, lon, lat)
    return int(r), int(c)


def get_emitter_density(lat_lon: list) -> float:
    """
    Determines the number of 5G emitters per square km based on a
    [lat, lon] coordinate, terrain type, and GHSL population data.

    Uses the GHSL population GeoTIFF for both population and water/land:
    population <= 0 (including -200 NoData for water) -> no emitters;
    population > 0 -> country check and population-based density. No global_land_mask used.

    Args:
        lat_lon (list): A list containing [latitude, longitude].

    Returns:
        float: The emitter density per square km.
    """
    lat, lon = lat_lon

    # ---------------------------------------------------------
    # STEP 1: Read population from cached GHSL raster
    # ---------------------------------------------------------
    try:
        raster, transform, height, width = _load_ghsl_raster()
        row, col = _rowcol(transform, lon, lat)
        if 0 <= row < height and 0 <= col < width:
            population = float(raster[row, col])
        else:
            population = 0.0
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading GeoTIFF at {_GHSL_TIF_PATH_RESOLVED}: {e}") from e

    # ---------------------------------------------------------
    # STEP 2: population <= 0 -> water (NoData -200) or uninhabited land (0); no emitters
    # ---------------------------------------------------------
    if population <= 0:
        terrain_type = 'water or uninhabited land'
        emitter_density_per_km2 = 0.0
        print(f"[{lat:7.4f}, {lon:8.4f}] -> Population: {population:7.0f}, "
              f"Terrain: '{terrain_type}', Density: {emitter_density_per_km2}")
        return emitter_density_per_km2

    # ---------------------------------------------------------
    # STEP 3: population > 0 -> check country and 5G list
    # ---------------------------------------------------------
    rg_result = rg.search((lat, lon), verbose=False)
    country_code = rg_result[0].get('cc', '')

    if country_code not in SUPPORTED_5G_COUNTRIES:
        country_name = country_code
        terrain_type = 'terrain'
        emitter_density_per_km2 = 0.0
        print(f"[{lat:7.4f}, {lon:8.4f}] -> Country: '{country_name}', "
              f"Population: {population:7.0f}, Terrain: '{terrain_type}', Density: {emitter_density_per_km2}")
        return emitter_density_per_km2

    # ---------------------------------------------------------
    # STEP 4: Supported 5G country; set terrain and density from population
    # ---------------------------------------------------------
    country_name = SUPPORTED_5G_COUNTRIES[country_code]
    if population > 10000:
        terrain_type = 'Ultra-dense urban'
        emitter_density_per_km2 = 40.0
    elif population > 5000:
        terrain_type = 'Dense urban'
        emitter_density_per_km2 = 25.0
    elif population > 1500:
        terrain_type = 'Urban'
        emitter_density_per_km2 = 12.0
    elif population > 300:
        terrain_type = 'Suburban'
        emitter_density_per_km2 = 3.5
    else:
        terrain_type = 'Open/Rural'
        emitter_density_per_km2 = 0.3

    print(f"[{lat:7.4f}, {lon:8.4f}] -> Country: '{country_name}', "
          f"Population: {population:7.0f}, Terrain: '{terrain_type}', Density: {emitter_density_per_km2}")
    return emitter_density_per_km2


# calculate d_max_km and d_min_min based on SAZA (satellite zenith angle, degrees) and HMSL (satellite altitude, meters)
def calculate_fov_dimensions(saza_deg, altitude_m, beamwidth_deg=None, sensor_name=None):
    """
    Calculates the major (d_max) and minor (d_min) axes of a satellite's footprint.

    Inputs:
    - saza_deg: Satellite Zenith Angle in degrees (SAZA from ATMS/AMSU-A data)
    - altitude_m: Satellite altitude in meters (HMSL from data)
    - beamwidth_deg: Antenna beamwidth in degrees. If None, sensor_name is required (see SENSOR_BEAMWIDTH_DEG).
    - sensor_name: Sensor name ('ATMS', 'AMSU-A') to look up beamwidth; required when beamwidth_deg is None.

    Returns:
    - d_max_km, d_min_km: The dimensions of the footprint ellipse in kilometers
    """
    beamwidth_deg = beamwidth_deg if beamwidth_deg is not None else _beamwidth_for_sensor(sensor_name)
    # 1. Constants
    R_e = 6371.0              # Earth's mean radius in km
    H = altitude_m / 1000.0   # Convert altitude from meters to km

    # 2. Convert degrees to radians for trigonometric functions
    theta_z = math.radians(saza_deg)
    delta_theta = math.radians(beamwidth_deg)

    # 3. Calculate Mechanical Scan Angle (theta_s)
    sin_theta_s = (R_e / (R_e + H)) * math.sin(theta_z)
    theta_s = math.asin(sin_theta_s)

    # 4. Calculate the Slant Range (L) in km
    L = (R_e + H) * math.cos(theta_s) - R_e * math.cos(theta_z)

    # 5. Calculate Minor Axis (d_min) - Along-track resolution
    d_min = L * delta_theta

    # 6. Calculate Major Axis (d_max) - Cross-track resolution
    # We divide by cos(theta_z) to account for the stretching on the curved Earth
    # Note: adding a tiny safeguard to prevent division by absolute zero at the exact horizon
    d_max = d_min / max(math.cos(theta_z), 0.0001)

    return d_max, d_min


# compute d_max_km and d_min_km based on SAZA (satellite zenith angle, degrees) and HMSL (satellite altitude, meters)
def generate_elliptical_ground_emitter_distribution(
    center_lat: float,
    center_lon: float,
    saza_deg: float,
    altitude_m: float,
    ellipse_azimuth_deg: float,
    beamwidth_deg: float = None,
    sensor_name=None,
    emitter_density_per_km2: float = 0.1,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate a uniform random distribution of 5G mmWave ground emitters inside an ellipse.

    Emitters represent 5G mmWave small cells; antenna heights are uniformly sampled
    from 5–25 m (typical for lampposts, building facades, low rooftops).

    The ellipse represents a satellite FOV (field of view): d_max is the major axis
    length, d_min the minor axis length (km), and ellipse_azimuth_deg is the angle
    of the major axis from North, clockwise in degrees.

    Args:
        center_lat: Center latitude of the ellipse (degrees).
        center_lon: Center longitude of the ellipse (degrees).
        saza_deg: Satellite Zenith Angle in degrees (SAZA from ATMS/AMSU-A data).
        altitude_m: Satellite altitude in meters (HMSL from data).
        ellipse_azimuth_deg: Angle of the major axis from North, clockwise (degrees).
        beamwidth_deg: Antenna beamwidth in degrees. If None, sensor_name is required.
        sensor_name: Sensor name ('ATMS', 'AMSU-A') to look up beamwidth; required when beamwidth_deg is None.
        emitter_density_per_km2: Number of emitters per km² (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Columns 'lat', 'lon', 'alt' (alt in meters, 5–25 m).
    """
    if seed is not None:
        np.random.seed(seed)

    # calculate d_max_km and d_min_min based on SAZA (satellite zenith angle, degrees) and HMSL (satellite altitude, meters)  # noqa: E501
    d_max_km, d_min_km = calculate_fov_dimensions(saza_deg, altitude_m, beamwidth_deg=beamwidth_deg, sensor_name=sensor_name)  # noqa: E501

    # Ensure d_max >= d_min (major vs minor)
    a_km = max(d_max_km, d_min_km) / 2.0
    b_km = min(d_max_km, d_min_km) / 2.0

    area_km2 = np.pi * a_km * b_km
    n_emitters = int(np.ceil(area_km2 * emitter_density_per_km2))

    # Uniform points in unit circle (r = sqrt(U) for uniform area)
    u = np.random.uniform(0, 1, n_emitters)
    theta = np.random.uniform(0, 2 * np.pi, n_emitters)
    r = np.sqrt(u)
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)

    # Scale to ellipse (x = semi-major, y = semi-minor) in meters
    a_m = a_km * 1000.0
    b_m = b_km * 1000.0
    x_local_m = a_m * x_circle
    y_local_m = b_m * y_circle

    # Rotate so major axis is at ellipse_azimuth from North (clockwise)
    # In (E, N) frame: major unit = (sin(az), cos(az)), minor = (cos(az), -sin(az))
    az_rad = np.deg2rad(ellipse_azimuth_deg)
    east_m = x_local_m * np.sin(az_rad) + y_local_m * np.cos(az_rad)
    north_m = x_local_m * np.cos(az_rad) - y_local_m * np.sin(az_rad)

    # Convert meters to degrees at center_lat
    center_lat_rad = np.deg2rad(center_lat)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(center_lat_rad)
    lat_offset = north_m / m_per_deg_lat
    lon_offset = east_m / m_per_deg_lon

    emitter_lats = center_lat + lat_offset
    emitter_lons = center_lon + lon_offset
    # 5G mmWave small-cell antenna heights (lampposts, facades, low rooftops)
    emitter_alts = np.random.uniform(5.0, 25.0, n_emitters)

    return pd.DataFrame({
        'lat': emitter_lats,
        'lon': emitter_lons,
        'alt': emitter_alts,
    }), d_max_km, d_min_km


def _population_to_density(population: float, country_code: str) -> float:
    """Map population and country to emitter density per km^2 (no print)."""
    if population <= 0 or country_code not in SUPPORTED_5G_COUNTRIES:
        return 0.0
    if population > 10000:
        return 50.0
    if population > 5000:
        return 30.0
    if population > 1500:
        return 15.0
    if population > 300:
        return 5.0
    return 1.0


def get_emitter_density_vectorized(
    lat: np.ndarray,
    lon: np.ndarray,
    chunk_size: int = 50000,
    ghsl_tile_size: int = 256,
) -> np.ndarray:
    """
    Vectorized emitter density per km^2 from GHSL and country (5G support).

    Uses same logic as get_emitter_density but over arrays; no printing.
    Tries to load the full GHSL raster once for fast vectorized indexing; on
    memory/read errors falls back to tile-based windowed reads. For indices with
    population > 0, reverse_geocoder is called in batches.

    Args:
        lat: Latitude array (degrees), shape (n,).
        lon: Longitude array (degrees), shape (n,).
        chunk_size: Max points per batch for reverse_geocoder (default 50000).
        ghsl_tile_size: GHSL tile size for fallback windowed reads (default 256).

    Returns:
        np.ndarray: Emitter density per km^2, shape (n,).
    """
    import rasterio
    from rasterio.transform import rowcol
    try:
        from rasterio.errors import RasterioIOError
        _ghsl_load_errors = (OSError, MemoryError, RasterioIOError)
    except ImportError:
        _ghsl_load_errors = (OSError, MemoryError)

    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    if lat.shape != lon.shape:
        raise ValueError("lat and lon must have the same shape")
    n = lat.size
    lat = lat.ravel()
    lon = lon.ravel()

    if not _GHSL_TIF_PATH_RESOLVED.is_file():
        raise FileNotFoundError(
            f"GHSL GeoTIFF not found: {_GHSL_TIF_PATH_RESOLVED}. "
            "Ensure the file exists or set GHSL_TIF_PATH to an absolute path."
        )

    rows = np.zeros(n, dtype=np.int64)
    cols = np.zeros(n, dtype=np.int64)

    population = np.zeros(n)
    try:
        raster, transform, height, width = _load_ghsl_raster()
        for i in range(n):
            r, c = rowcol(transform, float(lon[i]), float(lat[i]))
            rows[i], cols[i] = int(r), int(c)
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        population[valid] = raster[rows[valid], cols[valid]]
    except _ghsl_load_errors:
        from rasterio.windows import Window

        with rasterio.open(str(_GHSL_TIF_PATH_RESOLVED)) as dataset:
            height, width = dataset.height, dataset.width
            transform = dataset.transform
            for i in range(n):
                r, c = rowcol(transform, float(lon[i]), float(lat[i]))
                rows[i], cols[i] = int(r), int(c)
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            if not np.any(valid):
                return np.zeros(n)
            r_valid = rows[valid]
            c_valid = cols[valid]
            tile_r = r_valid // ghsl_tile_size
            tile_c = c_valid // ghsl_tile_size
            valid_inds = np.where(valid)[0]
            for (tr, tc) in set(zip(tile_r.tolist(), tile_c.tolist())):
                in_tile = (tile_r == tr) & (tile_c == tc)
                inds = valid_inds[in_tile]
                r_min = int(rows[inds].min())
                r_max = int(rows[inds].max())
                c_min = int(cols[inds].min())
                c_max = int(cols[inds].max())
                win = Window(c_min, r_min, c_max - c_min + 1, r_max - r_min + 1)
                block = np.asarray(dataset.read(1, window=win), dtype=np.float64)
                for i in inds:
                    population[i] = block[rows[i] - r_min, cols[i] - c_min]

    density = np.zeros(n)
    density[population <= 0] = 0.0
    idx_pos = np.where(population > 0)[0]
    if idx_pos.size == 0:
        return density

    for start in range(0, idx_pos.size, chunk_size):
        end = min(start + chunk_size, idx_pos.size)
        inds = idx_pos[start:end]
        coords = [(float(lat[i]), float(lon[i])) for i in inds]
        results = rg.search(coords, verbose=False)
        for j, res in enumerate(results):
            cc = res.get("cc", "")
            i = inds[j]
            density[i] = _population_to_density(float(population[i]), cc)
    return density


def calculate_fov_dimensions_vectorized(
    saza_deg: np.ndarray,
    altitude_m: np.ndarray,
    beamwidth_deg: float = None,
    sensor_name=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized FOV ellipse axes (km) from SAZA and altitude.

    Args:
        saza_deg: Satellite zenith angle (degrees), shape (n,).
        altitude_m: Satellite altitude (meters), shape (n,) or scalar.
        beamwidth_deg: Beamwidth (degrees). If None, sensor_name is required (see SENSOR_BEAMWIDTH_DEG).
        sensor_name: Sensor name ('ATMS', 'AMSU-A') to look up beamwidth; required when beamwidth_deg is None.

    Returns:
        (d_max_km, d_min_km): Major and minor axis lengths in km.
    """
    beamwidth_deg = beamwidth_deg if beamwidth_deg is not None else _beamwidth_for_sensor(sensor_name)
    saza_deg = np.atleast_1d(np.asarray(saza_deg, dtype=np.float64))
    altitude_m = np.broadcast_to(
        np.atleast_1d(np.asarray(altitude_m, dtype=np.float64)), saza_deg.shape
    )
    r_e = 6371.0
    h_km = altitude_m / 1000.0
    theta_z = np.deg2rad(saza_deg)
    delta_theta = np.deg2rad(beamwidth_deg)
    sin_theta_s = (r_e / (r_e + h_km)) * np.sin(theta_z)
    theta_s = np.arcsin(np.clip(sin_theta_s, -1.0, 1.0))
    l_km = (r_e + h_km) * np.cos(theta_s) - r_e * np.cos(theta_z)
    d_min = l_km * delta_theta
    cos_theta_z = np.maximum(np.cos(theta_z), 1e-10)
    d_max = d_min / cos_theta_z
    return d_max, d_min


# -----------------------------------------------------------------------------
# SSMI-S (conical scanning): bearing, FOV area, and emitter count
# -----------------------------------------------------------------------------


def calculate_ssmis_fov_bearing_vectorized(
    lat: np.ndarray,
    lon: np.ndarray,
    fovs_per_scan: int = SSMIS_FOVS_PER_SCAN,
) -> np.ndarray:
    """
    Bearing (azimuth) of the major axis of each SSMI-S elliptical FOV, in degrees from North (0–360).

    SSMI-S is conical scanning; the minor axis follows the scan arc and the major axis is
    perpendicular. For FOV 1..59, forward azimuth from FOV i to FOV i+1 gives the minor-axis
    direction; major = minor + 90°. For FOV 60 (last in scan), backward azimuth from FOV 59
    to FOV 60 is used. Uses pyproj.Geod (WGS84) for forward/back azimuth.

    Args:
        lat: Latitude (degrees), shape (n,). Must be in scan order: scan0_fov0..fov59, scan1_fov0...
        lon: Longitude (degrees), shape (n,). Same order as lat.
        fovs_per_scan: Number of FOVs per scan (default 60 for SSMI-S).

    Returns:
        bearing_major_deg: Bearing of the 27 km major axis (degrees from North, 0–360), shape (n,).
    """
    from pyproj import Geod
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64)).ravel()
    n = lat.size
    if n % fovs_per_scan != 0:
        raise ValueError(
            f"lat/lon size {n} must be divisible by fovs_per_scan={fovs_per_scan}"
        )
    n_scans = n // fovs_per_scan
    lat_2d = lat.reshape(n_scans, fovs_per_scan)
    lon_2d = lon.reshape(n_scans, fovs_per_scan)
    geod = Geod(ellps="WGS84")

    # Forward azimuth from FOV i to FOV i+1 for i = 0..58 (minor axis along arc)
    lon0 = lon_2d[:, 0:-1]   # (n_scans, 59)
    lat0 = lat_2d[:, 0:-1]
    lon1 = lon_2d[:, 1:]     # (n_scans, 59)
    lat1 = lat_2d[:, 1:]
    az12, az21, _ = geod.inv(lon0, lat0, lon1, lat1)  # az12 forward, (n_scans, 59)
    major_first59 = (az12 + 90.0) % 360.0   # major axis bearing for FOV 0..58

    # FOV 59 (last in scan): backward azimuth from 58 to 59
    az12_end, az21_end, _ = geod.inv(
        lon_2d[:, -2], lat_2d[:, -2],
        lon_2d[:, -1], lat_2d[:, -1],
    )  # az21_end = backward at 59, shape (n_scans,)
    major_last = (az21_end + 90.0) % 360.0

    out = np.empty((n_scans, fovs_per_scan), dtype=np.float64)
    out[:, 0:-1] = major_first59
    out[:, -1] = major_last
    return out.ravel()


def get_ssmis_vband_fov_area_km2_vectorized(n: int) -> np.ndarray:
    """
    Return FOV area (km²) for each of n SSMI-S V-band FOVs (all same: 27 km × 18 km ellipse).

    Args:
        n: Number of FOVs.

    Returns:
        area_km2: Shape (n,) with each element = SSMIS_VBAND_FOV_AREA_KM2.
    """
    return np.full(int(n), SSMIS_VBAND_FOV_AREA_KM2, dtype=np.float64)


def get_ssmis_n_emitters_vectorized(
    density_per_km2: np.ndarray,
) -> np.ndarray:
    """
    Number of emitters per SSMI-S V-band FOV from density and fixed FOV area (27 km × 18 km).

    Uses same density logic as get_emitter_density_vectorized (GHSL + country); call that
    to get density_per_km2 at each FOV center (lat, lon), then pass here.

    Args:
        density_per_km2: Emitter density per km², shape (n,) (e.g. from get_emitter_density_vectorized).

    Returns:
        n_emitters: Shape (n,), dtype int64; at least 1 where density > 0.
    """
    density_per_km2 = np.atleast_1d(np.asarray(density_per_km2, dtype=np.float64)).ravel()
    area_km2 = SSMIS_VBAND_FOV_AREA_KM2
    n_emitters = np.maximum(1, np.ceil(area_km2 * density_per_km2).astype(np.int64))
    n_emitters = np.where(density_per_km2 <= 0, 0, n_emitters)
    return n_emitters


def timestamp_from_nc4_vars(
    year: np.ndarray,
    month: np.ndarray,
    day: np.ndarray,
    hour: np.ndarray,
    minute: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """
    Build timestamp strings from nc4 time variables for ECEF lookup.

    Handles both scalar and array inputs; second can be float (e.g. 37.352).
    Output format: 'YYYY-MM-DD HH:MM:SS.fff' to match jpss1_ecef_lookup.csv.

    Returns:
        np.ndarray of strings, shape (n,).
    """
    y = np.atleast_1d(np.asarray(year).ravel())
    mo = np.atleast_1d(np.asarray(month).ravel())
    d = np.atleast_1d(np.asarray(day).ravel())
    h = np.atleast_1d(np.asarray(hour).ravel())
    mi = np.atleast_1d(np.asarray(minute).ravel())
    sec = np.atleast_1d(np.asarray(second, dtype=np.float64).ravel())
    n = int(sec.size)
    y = np.resize(y, n)
    mo = np.resize(mo, n)
    d = np.resize(d, n)
    h = np.resize(h, n)
    mi = np.resize(mi, n)
    sec = np.resize(sec, n)
    out = []
    for i in range(n):
        sec_i = float(sec.flat[i])
        sec_int = int(sec_i)
        frac = sec_i - sec_int
        ms = min(999, int(round(frac * 1000)))
        out.append(
            f"{int(y.flat[i]):04d}-{int(mo.flat[i]):02d}-{int(d.flat[i]):02d} "
            f"{int(h.flat[i]):02d}:{int(mi.flat[i]):02d}:{sec_int:02d}.{ms:03d}"
        )
    return np.array(out)


def model_rfi_nwp_5g_single_time(
    sat_ecef_km: np.ndarray,
    fov_lat: np.ndarray,
    fov_lon: np.ndarray,
    fov_saza_deg: np.ndarray,
    fov_altitude_m: float,
    density: np.ndarray,
    weather_sat_antenna,
    emitter_antenna,
    freq_hz: float = 50.3e9,
    bandwidth_hz: float = 180e6,
    eirp_per_emitter_dbw: float = 30.0,
    emitter_fundamental_freq: float = 26e9,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    humidity: float = 50.0,
    sensor_name=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized RFI from 5G ground emitters only (one equivalent emitter per FOV).

    Applies to cross-track V-band sounders (e.g. ATMS, AMSU-A). Uses full ITU-R P.676
    atmospheric absorption. 5G at emitter_fundamental_freq (e.g. 26 GHz); only 2nd
    harmonic is in V-band 50–55.5 GHz. No Starlink, polarization loss, terrain, or OOBE.
    FOV size (and thus n_emitters) is set by sensor_name via SENSOR_BEAMWIDTH_DEG
    (e.g. ATMS 2.2°, AMSU-A 3.3°).

    Args:
        sat_ecef_km: Satellite ECEF position (3,) in km.
        fov_lat: FOV center latitude (degrees), shape (n,).
        fov_lon: FOV center longitude (degrees), shape (n,).
        fov_saza_deg: Satellite zenith angle (degrees), shape (n,).
        fov_altitude_m: Satellite altitude (m), scalar.
        density: Emitter density per km^2 from get_emitter_density_vectorized, (n,).
        weather_sat_antenna: Antenna instance (V-band) with get_gain_values(alphas, betas).
        emitter_antenna: 5G sector antenna with get_gain_values, get_boresight_gain.
        freq_hz: Observation frequency (Hz), e.g. 50.3e9.
        bandwidth_hz: Channel bandwidth (Hz), e.g. 180e6.
        eirp_per_emitter_dbw: EIRP per emitter (dBW).
        emitter_fundamental_freq: 5G fundamental (Hz), e.g. 26e9.
        temperature: Temperature (K) for atmospheric loss.
        pressure: Pressure (Pa) for atmospheric loss.
        humidity: Relative humidity (%) for atmospheric loss.
        sensor_name: Sensor name ('ATMS', 'AMSU-A') for FOV beamwidth; required (no default).

    Returns:
        rfi_power_dBW: RFI power at satellite (dBW), shape (n,).
        rfi_tb_K: RFI brightness temperature (K), shape (n,).
    """
    if sensor_name is None:
        raise ValueError(
            "sensor_name is required for FOV beamwidth (no default). "
            f"Use one of: {list(SENSOR_BEAMWIDTH_DEG.keys())}"
        )
    from astro_mdl import power_to_temperature
    from sat_mdl import free_space_loss
    from weather_sat_mdl import (
        ecef_to_weather_sat_frame,
        latlonalt_to_ecef_vectorized,
        calculate_comprehensive_atmospheric_loss_vectorized,
    )

    n = int(np.size(fov_lat))
    fov_lat = np.atleast_1d(fov_lat).ravel()[:n]
    fov_lon = np.atleast_1d(fov_lon).ravel()[:n]
    fov_saza_deg = np.atleast_1d(fov_saza_deg).ravel()[:n]
    density = np.atleast_1d(density).ravel()[:n]

    sat_ecef_m = np.asarray(sat_ecef_km, dtype=np.float64).ravel() * 1000.0
    if sat_ecef_m.size != 3:
        raise ValueError("sat_ecef_km must be length 3")
    sat_ecef_m = sat_ecef_m.reshape(3)

    emitter_ecef = latlonalt_to_ecef_vectorized(
        fov_lat, fov_lon, np.full(n, 15.0)
    )
    ws_ecef = np.broadcast_to(sat_ecef_m[np.newaxis, :], (n, 3))
    ws_vel = np.zeros((n, 3))
    emitter_dec, emitter_caz = ecef_to_weather_sat_frame(
        emitter_ecef, ws_ecef, ws_vel
    )
    distance = np.linalg.norm(emitter_ecef - ws_ecef, axis=1)
    elevation_deg = 90.0 - np.rad2deg(emitter_dec)

    d_max, d_min = calculate_fov_dimensions_vectorized(
        fov_saza_deg, np.full(n, fov_altitude_m), sensor_name=sensor_name
    )
    area_km2 = np.pi * (d_max / 2.0) * (d_min / 2.0)
    n_emitters = np.maximum(1, np.ceil(area_km2 * density).astype(np.int64))
    effective_eirp_dbw = eirp_per_emitter_dbw + 10.0 * np.log10(
        n_emitters.astype(np.float64)
    )
    effective_eirp_dbw = np.where(density <= 0, -np.inf, effective_eirp_dbw)

    L_fs_fund = free_space_loss(distance, emitter_fundamental_freq)
    L_atm_fund = calculate_comprehensive_atmospheric_loss_vectorized(
        distance,
        np.full(n, emitter_fundamental_freq),
        elevation_angles=elevation_deg,
        temperature=temperature,
        pressure=pressure,
        humidity=humidity,
        use_full_itu_p676=True,
    )
    gain_ws = weather_sat_antenna.get_gain_values(emitter_dec, emitter_caz)
    gain_emitter_abs = emitter_antenna.get_gain_values(emitter_dec, emitter_caz)
    peak_emitter = emitter_antenna.get_boresight_gain()
    gain_emitter_rel = gain_emitter_abs / peak_emitter

    base_lb_fund = (
        gain_ws * (1.0 / L_fs_fund) * (1.0 / L_atm_fund) * gain_emitter_rel
    )
    harmonic_freq = 2.0 * emitter_fundamental_freq
    freq_min = freq_hz - bandwidth_hz / 2.0
    freq_max = freq_hz + bandwidth_hz / 2.0
    harmonic_in_band = (freq_min <= harmonic_freq <= freq_max)
    L_fs_harm = free_space_loss(distance, harmonic_freq)
    path_loss_ratio = np.where(
        L_fs_harm > 0, L_fs_fund / L_fs_harm, 0.0
    )
    second_harmonic_factor = 0.01  # 2nd harmonic: -20 dBc (1% of fundamental)
    link_budget = (
        base_lb_fund * second_harmonic_factor * path_loss_ratio
        * (1.0 if harmonic_in_band else 0.0)
    )

    effective_eirp_linear = np.where(
        density <= 0, 0.0, 10.0 ** (effective_eirp_dbw / 10.0)
    )
    p_w = effective_eirp_linear * link_budget
    p_w = np.where(density <= 0, 0.0, p_w)

    rfi_power_dBW = 10.0 * np.log10(np.maximum(p_w, 1e-30))
    rfi_tb_K = power_to_temperature(p_w, bandwidth_hz)
    return rfi_power_dBW, rfi_tb_K


def model_rfi_nwp_5g_single_time_ssmis(
    sat_ecef_km: np.ndarray,
    fov_lat: np.ndarray,
    fov_lon: np.ndarray,
    density: np.ndarray,
    weather_sat_antenna,
    emitter_antenna,
    freq_hz: float = 50.3e9,
    bandwidth_hz: float = 180e6,
    eirp_per_emitter_dbw: float = 30.0,
    emitter_fundamental_freq: float = 26e9,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    humidity: float = 50.0,
    slant_range_km: float = 1020.0,
    elevation_deg: float = 36.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized RFI from 5G ground emitters for SSMI-S (conical scanning).

    Same link budget as model_rfi_nwp_5g_single_time (ITU-R P.676, 2nd harmonic -20 dBc)
    but with constant slant range and elevation for all FOVs. n_emitters from fixed
    SSMI-S V-band FOV area (27 km x 18 km). Antenna gain direction (emitter_dec, emitter_caz)
    still varies by FOV for realistic pattern.

    Args:
        sat_ecef_km: Satellite ECEF position (3,) in km.
        fov_lat: FOV center latitude (degrees), shape (n,).
        fov_lon: FOV center longitude (degrees), shape (n,).
        density: Emitter density per km^2 from get_emitter_density_vectorized, (n,).
        weather_sat_antenna: V-band antenna with get_gain_values(alphas, betas).
        emitter_antenna: 5G sector antenna with get_gain_values, get_boresight_gain.
        freq_hz: Observation frequency (Hz).
        bandwidth_hz: Channel bandwidth (Hz).
        eirp_per_emitter_dbw: EIRP per emitter (dBW).
        emitter_fundamental_freq: 5G fundamental (Hz).
        temperature, pressure, humidity: For ITU-R P.676.
        slant_range_km: Slant distance satellite to FOV center (km), default 1020 (DMSP-F17).
        elevation_deg: Elevation angle from ground to satellite (deg), default 36.9.

    Returns:
        rfi_power_dBW: RFI power at satellite (dBW), shape (n,).
        rfi_tb_K: RFI brightness temperature (K), shape (n,).
    """
    from astro_mdl import power_to_temperature
    from sat_mdl import free_space_loss
    from weather_sat_mdl import (
        ecef_to_weather_sat_frame,
        latlonalt_to_ecef_vectorized,
        calculate_comprehensive_atmospheric_loss_vectorized,
    )

    n = int(np.size(fov_lat))
    fov_lat = np.atleast_1d(fov_lat).ravel()[:n]
    fov_lon = np.atleast_1d(fov_lon).ravel()[:n]
    density = np.atleast_1d(density).ravel()[:n]

    sat_ecef_m = np.asarray(sat_ecef_km, dtype=np.float64).ravel() * 1000.0
    if sat_ecef_m.size != 3:
        raise ValueError("sat_ecef_km must be length 3")
    sat_ecef_m = sat_ecef_m.reshape(3)

    emitter_ecef = latlonalt_to_ecef_vectorized(
        fov_lat, fov_lon, np.full(n, 15.0)
    )
    ws_ecef = np.broadcast_to(sat_ecef_m[np.newaxis, :], (n, 3))
    ws_vel = np.zeros((n, 3))
    emitter_dec, emitter_caz = ecef_to_weather_sat_frame(
        emitter_ecef, ws_ecef, ws_vel
    )
    # SSMI-S: constant slant range and elevation for all FOVs (conical scan)
    distance_m = np.full(n, slant_range_km * 1000.0)
    elevation_deg_arr = np.full(n, elevation_deg)

    n_emitters = get_ssmis_n_emitters_vectorized(density)
    effective_eirp_dbw = eirp_per_emitter_dbw + 10.0 * np.log10(
        np.where(n_emitters > 0, n_emitters.astype(np.float64), 1.0)
    )
    effective_eirp_dbw = np.where(density <= 0, -np.inf, effective_eirp_dbw)

    L_fs_fund = free_space_loss(distance_m, emitter_fundamental_freq)
    L_atm_fund = calculate_comprehensive_atmospheric_loss_vectorized(
        distance_m,
        np.full(n, emitter_fundamental_freq),
        elevation_angles=elevation_deg_arr,
        temperature=temperature,
        pressure=pressure,
        humidity=humidity,
        use_full_itu_p676=True,
    )
    gain_ws = weather_sat_antenna.get_gain_values(emitter_dec, emitter_caz)
    gain_emitter_abs = emitter_antenna.get_gain_values(emitter_dec, emitter_caz)
    peak_emitter = emitter_antenna.get_boresight_gain()
    gain_emitter_rel = gain_emitter_abs / peak_emitter

    base_lb_fund = (
        gain_ws * (1.0 / L_fs_fund) * (1.0 / L_atm_fund) * gain_emitter_rel
    )
    harmonic_freq = 2.0 * emitter_fundamental_freq
    freq_min = freq_hz - bandwidth_hz / 2.0
    freq_max = freq_hz + bandwidth_hz / 2.0
    harmonic_in_band = (freq_min <= harmonic_freq <= freq_max)
    L_fs_harm = free_space_loss(distance_m, harmonic_freq)
    path_loss_ratio = np.where(
        L_fs_harm > 0, L_fs_fund / L_fs_harm, 0.0
    )
    second_harmonic_factor = 0.01  # 2nd harmonic: -20 dBc (1% of fundamental)
    link_budget = (
        base_lb_fund * second_harmonic_factor * path_loss_ratio
        * (1.0 if harmonic_in_band else 0.0)
    )

    effective_eirp_linear = np.where(
        density <= 0, 0.0, 10.0 ** (effective_eirp_dbw / 10.0)
    )
    p_w = effective_eirp_linear * link_budget
    p_w = np.where(density <= 0, 0.0, p_w)

    rfi_power_dBW = 10.0 * np.log10(np.maximum(p_w, 1e-30))
    rfi_tb_K = power_to_temperature(p_w, bandwidth_hz)
    return rfi_power_dBW, rfi_tb_K


# Find ECEF from lookup table
def load_ecef_lookup(filepath):
    """
    Loads the CSV into a dictionary for instant O(1) lookups.
    Key: Timestamp string (e.g., '2023-08-01 00:00:26.667')
    Value: Tuple of floats (X, Y, Z)
    """
    print(f"Loading ECEF data from {filepath} into memory...")
    ecef_dict = {}

    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            # row[0] is the timestamp string
            # We convert the coordinate strings back into fast floats
            ecef_dict[row[0]] = (float(row[1]), float(row[2]), float(row[3]))

    print(f"Successfully loaded {len(ecef_dict):,} coordinates.")
    return ecef_dict


def combine_channel_csvs(out_dir, satellite_name, nc4_stem, remove_channel_files=True):
    """
    Combine per-channel RFI CSVs into one CSV (timestamp + channelN_rfi_brightness_temperature_K).
    Optionally remove the per-channel CSV files after writing the combined file.

    Args:
        out_dir: Directory containing channel CSVs (str or Path).
        satellite_name: Satellite name used in filenames (e.g. JPSS-1).
        nc4_stem: nc4 filename without extension (e.g. atms_2023080112).
        remove_channel_files: If True, delete per-channel CSVs after combining (default True).

    Returns:
        Path to the combined CSV file, or None if no channel files were found.
    """
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_dir():
        return None

    pattern = re.compile(rf"^{re.escape(satellite_name)}_{re.escape(nc4_stem)}_5G_RFI_ch(\d+)\.csv$")
    channel_files = []
    for f in out_dir.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            channel_files.append((int(m.group(1)), f))

    if not channel_files:
        return None

    channel_files.sort(key=lambda x: x[0])

    _, path0 = channel_files[0]
    df0 = pd.read_csv(path0)
    if "timestamp" not in df0.columns or "rfi_brightness_temperature_K" not in df0.columns:
        return None

    data = {"timestamp": df0["timestamp"].values}
    data[f"channel{channel_files[0][0]}_rfi_brightness_temperature_K"] = df0["rfi_brightness_temperature_K"].values
    n_rows = len(df0)

    for ch_num, path in channel_files[1:]:
        df = pd.read_csv(path)
        if "rfi_brightness_temperature_K" not in df.columns or len(df) != n_rows:
            continue
        data[f"channel{ch_num}_rfi_brightness_temperature_K"] = df["rfi_brightness_temperature_K"].values

    combined = pd.DataFrame(data)
    out_name = f"{satellite_name}_{nc4_stem}_5G_RFI_combined.csv"
    out_path = out_dir / out_name
    combined.to_csv(out_path, index=False)

    if remove_channel_files:
        for _ch_num, path in channel_files:
            try:
                path.unlink()
            except OSError:
                pass

    return out_path
