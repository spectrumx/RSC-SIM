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
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import reverse_geocoder as rg

# Per-sensor/channel 5G country allowlist from CSV (see load_country_5g_sensor_channel_csv).
_COUNTRY_5G_CHANNEL_CSV = (
    Path(__file__).resolve().parent.parent
    / "research_tutorials"
    / "data"
    / "country_5G_sensor_channel.csv"
)


def load_country_5g_sensor_channel_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load ``country_5G_sensor_channel.csv`` with columns
    ``country_name``, ``ISO``, ``ATMS``, ``AMSU_A``, ``SSMI_S``.
    """
    p = path if path is not None else _COUNTRY_5G_CHANNEL_CSV
    p = Path(p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Country/sensor/channel CSV not found: {p}")
    df = pd.read_csv(p)
    required = ("country_name", "ISO", "ATMS", "AMSU_A", "SSMI_S")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {p} missing columns {missing}; found {list(df.columns)}"
        )
    return df


def supported_5g_countries_for_channel(
    df: pd.DataFrame,
    sensor: str,
    channel_number: int,
) -> dict[str, str]:
    """
    ISO code -> country name for rows where the sensor's channel column matches ``channel_number``.

    CSV column ``AMSU_A`` is keyed by script sensor name ``AMSU-A``.
    If the same ISO appears more than once, the last row wins.
    """
    col_map = {"ATMS": "ATMS", "AMSU-A": "AMSU_A", "SSMI-S": "SSMI_S"}
    key = str(sensor).strip()
    if key not in col_map:
        raise ValueError(
            f"Unknown sensor {sensor!r}; use one of {list(col_map.keys())}"
        )
    col = col_map[key]
    ch = int(channel_number)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        v = row[col]
        if pd.isna(v):
            continue
        try:
            if int(v) != ch:
                continue
        except (TypeError, ValueError):
            continue
        iso = str(row["ISO"]).strip()
        name = str(row["country_name"]).strip()
        if iso:
            out[iso] = name
    return out


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


def clear_ghsl_raster_cache() -> None:
    """Clear the cached GHSL raster (frees RAM; next use reloads from disk)."""
    global _GHSL_RASTER, _GHSL_TRANSFORM, _GHSL_HEIGHT, _GHSL_WIDTH
    _GHSL_RASTER = None
    _GHSL_TRANSFORM = None
    _GHSL_HEIGHT = None
    _GHSL_WIDTH = None


def _rowcol(transform, lon: float, lat: float) -> tuple[int, int]:
    """Get (row, col) from (lon, lat); uses lazy import to avoid loading rasterio at module import."""
    from rasterio.transform import rowcol
    r, c = rowcol(transform, lon, lat)
    return int(r), int(c)


def get_emitter_density(lat_lon: list, *, supported_5g_countries: dict[str, str]) -> float:
    """
    Determines the number of 5G emitters per square km based on a
    [lat, lon] coordinate, terrain type, and GHSL population data.

    Uses the GHSL population GeoTIFF for both population and water/land:
    population <= 0 (including -200 NoData for water) -> no emitters;
    population > 0 -> country check and population-based density. No global_land_mask used.

    Args:
        lat_lon (list): A list containing [latitude, longitude].
        supported_5g_countries: Map ISO alpha-2 -> country name for allowed 5G (caller-defined).

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

    if country_code not in supported_5g_countries:
        country_name = country_code
        terrain_type = 'terrain'
        emitter_density_per_km2 = 0.0
        print(f"[{lat:7.4f}, {lon:8.4f}] -> Country: '{country_name}', "
              f"Population: {population:7.0f}, Terrain: '{terrain_type}', Density: {emitter_density_per_km2}")
        return emitter_density_per_km2

    # ---------------------------------------------------------
    # STEP 4: Supported 5G country; set terrain and density from population
    # ---------------------------------------------------------
    country_name = supported_5g_countries[country_code]
    if population > 10000:
        terrain_type = 'Ultra-dense urban'
        emitter_density_per_km2 = 30.0
    elif population > 5000:
        terrain_type = 'Dense urban'
        emitter_density_per_km2 = 15.0
    elif population > 1500:
        terrain_type = 'Urban'
        emitter_density_per_km2 = 5.0
    elif population > 300:
        terrain_type = 'Suburban'
        emitter_density_per_km2 = 3.0
    else:
        terrain_type = 'Open/Rural'
        emitter_density_per_km2 = 1.0

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


def _population_to_density(
    population: float,
    country_code: str,
    supported_5g_countries: dict[str, str],
) -> float:
    """Map population and country to emitter density per km^2 (no print)."""
    if population <= 0 or country_code not in supported_5g_countries:
        return 0.0
    if population > 10000:
        return 30.0
    if population > 5000:
        return 15.0
    if population > 1500:
        return 5.0
    if population > 300:
        return 3.0
    return 1.0


def get_emitter_density_vectorized(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    supported_5g_countries: dict[str, str],
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
        supported_5g_countries: ISO alpha-2 -> country name; density is 0 outside this map.
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
    finite_geo = np.isfinite(lat) & np.isfinite(lon)

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
            if not finite_geo[i]:
                rows[i], cols[i] = -1, -1
                continue
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
                if not finite_geo[i]:
                    rows[i], cols[i] = -1, -1
                    continue
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
            density[i] = _population_to_density(
                float(population[i]), cc, supported_5g_countries
            )
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
    y = np.atleast_1d(np.asarray(year, dtype=np.float64).ravel())
    mo = np.atleast_1d(np.asarray(month, dtype=np.float64).ravel())
    d = np.atleast_1d(np.asarray(day, dtype=np.float64).ravel())
    h = np.atleast_1d(np.asarray(hour, dtype=np.float64).ravel())
    mi = np.atleast_1d(np.asarray(minute, dtype=np.float64).ravel())
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
        if not (
            np.isfinite(y.flat[i])
            and np.isfinite(mo.flat[i])
            and np.isfinite(d.flat[i])
            and np.isfinite(h.flat[i])
            and np.isfinite(mi.flat[i])
            and np.isfinite(sec_i)
        ):
            out.append("")
            continue
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
    # Emitter: eirp_per_emitter_dbw is EIRP at boresight (already P_tx + G_tx_max).
    # gain_emitter_rel = pattern in direction of satellite (0..1), not gain again—no double-count.
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
    second_harmonic_factor = 1e-6  # 2nd harmonic: -60 dBc relative to fundamental
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

    Same link budget as model_rfi_nwp_5g_single_time (ITU-R P.676, 2nd harmonic -60 dBc)
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
    # Emitter: eirp_per_emitter_dbw is EIRP at boresight (already P_tx + G_tx_max).
    # gain_emitter_rel = pattern in direction of satellite (0..1), not gain again—no double-count.
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
    second_harmonic_factor = 1e-6  # 2nd harmonic: -60 dBc relative to fundamental
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


# -----------------------------------------------------------------------------
# NC4 missing-value sentinel (product fill; colleague-confirmed).
# Most float fields: nominal missing ~1e11; float32 rounding yields values slightly below
# ``10e10``; detection uses ``>= NC4_MISSING_FLOAT_MIN``, not exact equality to
# ``NC4_MISSING_FLOAT``. **HMSL** (meters) uses ``NC4_MISSING_HMSL_VALUE`` only, not the
# large-sentinel band.
# -----------------------------------------------------------------------------
NC4_MISSING_FLOAT = 10e10  # nominal 1e11 (attrs / documentation)
NC4_MISSING_FLOAT_MIN = 1e10  # treat finite values >= this as the large missing sentinel band
NC4_MISSING_HMSL_VALUE = -9999.0  # HMSL missing in nc4 (not ~1e11)
# Placeholder written to augmented ``TMBR`` when pre-existing Tb is invalid (not ``10e10``).
NC4_MISSING_TMBR_OUT_RFI_NC4 = 1e10
DEFAULT_LEO_ALTITUDE_M = 850_000.0


def _large_missing_float_mask(values: np.ndarray) -> np.ndarray:
    """True where ``values`` are finite and ``>= NC4_MISSING_FLOAT_MIN`` (near-1e11 fill)."""
    v = np.asarray(values, dtype=np.float64)
    return np.isfinite(v) & (v >= NC4_MISSING_FLOAT_MIN)


def replace_missing_with_nan(
    arr: np.ndarray,
    fill: Optional[float] = None,
) -> np.ndarray:
    """
    Return float64 copy with product missing floats replaced by NaN.

    When ``fill`` is ``None`` or any value ``>= NC4_MISSING_FLOAT_MIN`` (including the
    default ``NC4_MISSING_FLOAT``), all finite elements ``>= NC4_MISSING_FLOAT_MIN`` are
    set to NaN (covers float32-rounded ``1e11``). For a smaller explicit ``fill``, uses
    exact equality ``==`` only.
    """
    if fill is None:
        fill = NC4_MISSING_FLOAT
    a = np.asarray(arr, dtype=np.float64)
    out = np.array(a, copy=True, dtype=np.float64)
    fill_f = float(fill)
    if fill_f >= NC4_MISSING_FLOAT_MIN:
        out[_large_missing_float_mask(out)] = np.nan
    else:
        out[out == fill_f] = np.nan
    return out


def scalar_altitude_m_from_hmsl(
    alt_1d: np.ndarray,
    fill: Optional[float] = None,
    default_m: Optional[float] = None,
) -> float:
    """
    Mean HMSL (m) ignoring missing cells; ``default_m`` if no valid samples.

    By default, HMSL missing is ``NC4_MISSING_HMSL_VALUE`` (``-9999``), not the large
    ``~1e11`` sentinel. Pass ``fill`` explicitly to override (e.g. ``NC4_MISSING_FLOAT``
    if a product encodes HMSL that way). ``replace_missing_with_nan`` then ``nanmean``.
    """
    if fill is None:
        fill = NC4_MISSING_HMSL_VALUE
    if default_m is None:
        default_m = DEFAULT_LEO_ALTITUDE_M
    a = replace_missing_with_nan(np.asarray(alt_1d, dtype=np.float64), fill=fill)
    m = np.nanmean(a)
    if np.isfinite(m):
        return float(m)
    return float(default_m)


def _netcdf_float_missing_values_from_var(var) -> list[float]:
    """
    Collect finite float missing/fill values from a netCDF variable's attributes.

    Includes ``_FillValue``, ``missing_value`` when present, and always
    ``NC4_MISSING_FLOAT`` (deduplicated). Near-``1e11`` payloads are also detected via
    ``NC4_MISSING_FLOAT_MIN`` in ``replace_missing_with_nan`` / ``_tmbr_preexisting_valid_mask``.
    """
    raw_list: list[float] = []
    for name in ("_FillValue", "missing_value"):
        if name not in var.ncattrs():
            continue
        raw = var.getncattr(name)
        arr = np.atleast_1d(np.asarray(raw, dtype=np.float64)).ravel()
        for x in arr:
            xf = float(x)
            if np.isfinite(xf):
                raw_list.append(xf)
    raw_list.append(float(NC4_MISSING_FLOAT))
    uniq: list[float] = []
    for xf in raw_list:
        tol = 1.0 if max(abs(xf), 1.0) >= 1e6 else 1e-6
        if any(abs(xf - u) <= tol for u in uniq):
            continue
        uniq.append(xf)
    return uniq


def _tmbr_preexisting_valid_mask(
    base2: np.ndarray,
    mask2: np.ndarray,
    fill_values: Sequence[float],
) -> np.ndarray:
    """
    Per-element validity for pre-existing brightness temperature (K).

    False where not finite, masked, finite values ``>= NC4_MISSING_FLOAT_MIN`` (near-``1e11``
    missing, including float32-rounded), or equal (within tolerance) to any other
    ``fill_values`` entry (e.g. small ``_FillValue`` / ``missing_value``).
    """
    bad_fill = np.zeros(base2.shape, dtype=bool)
    bad_fill |= _large_missing_float_mask(base2)
    for fv in fill_values:
        fv = float(fv)
        if not np.isfinite(fv):
            continue
        if fv >= NC4_MISSING_FLOAT_MIN:
            continue
        if abs(fv) >= 1e6:
            bad_fill |= np.isclose(base2, fv, rtol=0.0, atol=1.0, equal_nan=False)
        else:
            bad_fill |= np.isclose(base2, fv, rtol=0.0, atol=1e-4, equal_nan=False)
    m2 = np.asarray(mask2, dtype=bool)
    return np.isfinite(base2) & ~m2 & ~bad_fill


def _time_components_plausible_mask(
    year: np.ndarray,
    month: np.ndarray,
    day: np.ndarray,
    hour: np.ndarray,
    minute: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Finite time fields with loose calendar bounds (nc4 integer-like components)."""
    y = np.asarray(year, dtype=np.float64)
    mo = np.asarray(month, dtype=np.float64)
    d = np.asarray(day, dtype=np.float64)
    h = np.asarray(hour, dtype=np.float64)
    mi = np.asarray(minute, dtype=np.float64)
    s = np.asarray(second, dtype=np.float64)
    return (
        np.isfinite(y)
        & np.isfinite(mo)
        & np.isfinite(d)
        & np.isfinite(h)
        & np.isfinite(mi)
        & np.isfinite(s)
        & (y >= 1980.0)
        & (y <= 2060.0)
        & (mo >= 1.0)
        & (mo <= 12.0)
        & (d >= 1.0)
        & (d <= 31.0)
        & (h >= 0.0)
        & (h <= 23.0)
        & (mi >= 0.0)
        & (mi <= 59.0)
        & (s >= 0.0)
        & (s < 61.0)
    )


def _satellite_name_nonempty_mask(satellite: np.ndarray) -> np.ndarray:
    """True where mapped satellite name is non-empty (RFI / ECEF grouping applies)."""
    s = np.asarray(satellite, dtype=str)
    s = np.char.strip(s)
    return np.char.str_len(s) > 0


def obs_valid_cross_track(
    lat: np.ndarray,
    lon: np.ndarray,
    saza_deg: np.ndarray,
    bearaz_deg: np.ndarray,
    year: np.ndarray,
    month: np.ndarray,
    day: np.ndarray,
    hour: np.ndarray,
    minute: np.ndarray,
    second: np.ndarray,
    satellite: np.ndarray,
) -> np.ndarray:
    """
    Per-row validity for cross-track RFI (ATMS, AMSU-A): geo + angles + time + satellite name.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    saza_deg = np.asarray(saza_deg, dtype=np.float64)
    bearaz_deg = np.asarray(bearaz_deg, dtype=np.float64)
    geo = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(saza_deg)
        & np.isfinite(bearaz_deg)
        & (np.abs(lat) <= 90.0)
        & (np.abs(lon) <= 180.0)
    )
    tm = _time_components_plausible_mask(year, month, day, hour, minute, second)
    sat_ok = _satellite_name_nonempty_mask(satellite)
    return geo & tm & sat_ok


def obs_valid_ssmis_conical(
    lat: np.ndarray,
    lon: np.ndarray,
    year: np.ndarray,
    month: np.ndarray,
    day: np.ndarray,
    hour: np.ndarray,
    minute: np.ndarray,
    second: np.ndarray,
    satellite: np.ndarray,
) -> np.ndarray:
    """Per-row validity for SSMI-S RFI: lat/lon + time + satellite name (no HMSL in nc4)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    geo = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (np.abs(lat) <= 90.0)
        & (np.abs(lon) <= 180.0)
    )
    tm = _time_components_plausible_mask(year, month, day, hour, minute, second)
    sat_ok = _satellite_name_nonempty_mask(satellite)
    return geo & tm & sat_ok


# SAID (Satellite ID) in nc4: map to satellite name per sensor. Used by RFI scripts to select ECEF per observation.
SENSOR_SAID_TO_SATELLITE = {
    "ATMS": {224: "SUOMI-NPP", 225: "JPSS-1"},
    "AMSU-A": {206: "NOAA-15", 209: "NOAA-18", 223: "NOAA-19", 3: "METOP-B", 5: "METOP-C"},
    "SSMI-S": {285: "DMSP-F17", 286: "DMSP-F18"},
}
# SAIDs for which we compute RFI; others get RFI = 0. SSMI-S: only DMSP-F17 (285).
SENSOR_ALLOWED_SAIDS = {
    "ATMS": (224, 225),
    "AMSU-A": (206, 209, 223, 3, 5),
    "SSMI-S": (285,),
}


def said_to_satellite_array(said_arr, sensor_name: str) -> np.ndarray:
    """Map SAID (int) per observation to satellite name (str). Unknown SAID -> empty string. Vectorized."""
    said_flat = np.asarray(said_arr).ravel().astype(np.int64)
    mapping = SENSOR_SAID_TO_SATELLITE.get(sensor_name, {})
    if not mapping:
        return np.full(said_flat.shape, "", dtype=object)
    out = np.full(said_flat.shape, "", dtype=object)
    for said_val, name in mapping.items():
        out[said_flat == said_val] = name
    return out


def iter_valid_ts_sat_indices(timestamps, satellite, allowed, ecef_by_satellite):
    """
    Single-pass grouping: yield (ts, sat, indices, coords) for each valid (ts, sat) that has ECEF.
    Avoids building a set of n_obs tuples, repeated full-array mask builds, and repeated dict lookups.
    indices: 1d int array of row indices. coords: (X, Y, Z) tuple from ECEF lookup.
    """
    from collections import defaultdict
    key_to_data = defaultdict(lambda: {"indices": [], "coords": None})
    timestamps = np.asarray(timestamps).ravel()
    satellite = np.asarray(satellite).ravel()
    n_obs = len(timestamps)
    for i in range(n_obs):
        ts, sat = timestamps.flat[i], satellite.flat[i]
        if not sat or sat not in allowed or sat not in ecef_by_satellite:
            continue
        coords = ecef_by_satellite[sat].get(ts)
        if coords is None:
            continue
        key = (ts, sat)
        key_to_data[key]["indices"].append(i)
        key_to_data[key]["coords"] = coords
    for (ts, sat), data in key_to_data.items():
        yield ts, sat, np.asarray(data["indices"], dtype=np.intp), data["coords"]


def load_ecef_lookups_for_nc4(nc4_path: str) -> dict:
    """
    Load all ECEF lookup CSVs for the given nc4 file from the same directory.
    nc4_path e.g. util/ATMS/atms.2023080112.nc4 → finds *_ECEF_lookup_atms.2023080112.csv,
    parses satellite name from each filename, returns dict[satellite_name] -> ecef_lookup (timestamp -> (X,Y,Z)).
    """
    path = Path(nc4_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"nc4 file not found: {nc4_path}")
    directory = path.parent
    stem = path.stem
    pattern = f"*_ECEF_lookup_{stem}.csv"
    ecef_files = sorted(directory.glob(pattern))
    result = {}
    for fp in ecef_files:
        # Parse: {satellite}_ECEF_lookup_{stem}.csv
        name = fp.stem
        prefix = f"_ECEF_lookup_{stem}"
        if not name.endswith(prefix):
            continue
        satellite_name = name[: -len(prefix)]
        if not satellite_name:
            continue
        result[satellite_name] = load_ecef_lookup(str(fp))
    return result


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


def combine_channel_csvs(out_dir, nc4_stem, remove_channel_files=True, rfi_prefix="5G"):
    """
    Combine per-channel RFI CSVs into one CSV.

    Columns: ``timestamp``, then ``lat`` and ``lon`` if present in the first channel file,
    then ``saza`` (satellite zenith angle, degrees) if present, then
    ``channelN_rfi_brightness_temperature_K`` for each channel. Lat/lon/saza are taken from
    the first channel CSV only (same for all rows across channels in typical NWP RFI outputs).
    No satellite column in combined output. Finds files matching
    ``{nc4_stem}_{rfi_prefix}_RFI_chN.csv``. Optionally remove per-channel CSVs after writing.

    Args:
        out_dir: Directory containing channel CSVs (str or Path).
        nc4_stem: nc4 filename without extension (e.g. atms.2023080112).
        remove_channel_files: If True, delete per-channel CSVs after combining (default True).
        rfi_prefix: Prefix for RFI filenames (default "5G"); e.g. "Starlink_Gateway" for gateway RFI.

    Returns:
        Path to the combined CSV file, or None if no channel files were found.
    """
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_dir():
        return None

    safe_prefix = re.escape(rfi_prefix)
    pattern = re.compile(rf"^{re.escape(nc4_stem)}_{safe_prefix}_RFI_ch(\d+)\.csv$")
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
    if "lat" in df0.columns and "lon" in df0.columns:
        data["lat"] = df0["lat"].values
        data["lon"] = df0["lon"].values
    if "saza" in df0.columns:
        data["saza"] = df0["saza"].values
    data[f"channel{channel_files[0][0]}_rfi_brightness_temperature_K"] = df0["rfi_brightness_temperature_K"].values
    n_rows = len(df0)

    for ch_num, path in channel_files[1:]:
        df = pd.read_csv(path)
        if "rfi_brightness_temperature_K" not in df.columns or len(df) != n_rows:
            continue
        data[f"channel{ch_num}_rfi_brightness_temperature_K"] = df["rfi_brightness_temperature_K"].values

    combined = pd.DataFrame(data)
    out_name = f"{nc4_stem}_{rfi_prefix}_RFI_combined.csv"
    out_path = out_dir / out_name
    combined.to_csv(out_path, index=False)

    if remove_channel_files:
        for _ch_num, path in channel_files:
            try:
                path.unlink()
            except OSError:
                pass

    return out_path


def sum_two_rfi_combined_csvs_by_channel(
    out_dir,
    nc4_stem,
    rfi_prefix_a: str = "5G",
    rfi_prefix_b: str = "Starlink_Gateway",
    output_rfi_prefix: str = "5G_Starlink_Gateway",
):
    """
    Sum per-channel RFI brightness temperature from two combined CSVs into one file.

    Reads ``{nc4_stem}_{rfi_prefix_a}_RFI_combined.csv`` and
    ``{nc4_stem}_{rfi_prefix_b}_RFI_combined.csv``. For each column named
    ``channelN_rfi_brightness_temperature_K`` present in both files, the output column
    is the element-wise sum (linear Tb). ``timestamp`` and, if present, ``lat``,
    ``lon``, and ``saza`` are taken from the first file (``saza`` is checked against
    the second file when both have it). Row counts must match.

    Args:
        out_dir: Directory containing the two combined CSVs.
        nc4_stem: nc4 filename stem (e.g. atms.2023080112).
        rfi_prefix_a: Filename prefix for the first combined CSV (default ``5G``).
        rfi_prefix_b: Filename prefix for the second combined CSV (default ``Starlink_Gateway``).
        output_rfi_prefix: Prefix for the output filename (default ``5G_Starlink_Gateway``).

    Returns:
        Path to ``{nc4_stem}_{output_rfi_prefix}_RFI_combined.csv``, or None if inputs
        are missing or incompatible.
    """
    out_dir = Path(out_dir).resolve()
    path_a = out_dir / f"{nc4_stem}_{rfi_prefix_a}_RFI_combined.csv"
    path_b = out_dir / f"{nc4_stem}_{rfi_prefix_b}_RFI_combined.csv"
    if not path_a.is_file() or not path_b.is_file():
        return None

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    if len(df_a) != len(df_b) or "timestamp" not in df_a.columns:
        return None

    ch_pat = re.compile(r"^channel(\d+)_rfi_brightness_temperature_K$")
    ch_cols_a = [c for c in df_a.columns if ch_pat.match(c)]
    ch_cols_b = set(c for c in df_b.columns if ch_pat.match(c))
    ch_cols = sorted(
        (c for c in ch_cols_a if c in ch_cols_b),
        key=lambda c: int(ch_pat.match(c).group(1)),
    )
    if not ch_cols:
        return None

    out_data = {"timestamp": df_a["timestamp"].values}
    if "lat" in df_a.columns and "lon" in df_a.columns:
        out_data["lat"] = df_a["lat"].values
        out_data["lon"] = df_a["lon"].values
    if "saza" in df_a.columns:
        out_data["saza"] = df_a["saza"].values
        if "saza" in df_b.columns:
            sa_a = pd.to_numeric(df_a["saza"], errors="coerce").to_numpy(dtype=np.float64)
            sa_b = pd.to_numeric(df_b["saza"], errors="coerce").to_numpy(dtype=np.float64)
            if not np.allclose(sa_a, sa_b, rtol=0.0, atol=1e-4, equal_nan=True):
                print(
                    "WARNING: sum_two_rfi_combined_csvs_by_channel: saza differs between "
                    f"{path_a.name} and {path_b.name}; using values from the first file."
                )

    for col in ch_cols:
        sa = pd.to_numeric(df_a[col], errors="coerce").fillna(0.0)
        sb = pd.to_numeric(df_b[col], errors="coerce").fillna(0.0)
        out_data[col] = sa + sb

    out_path = out_dir / f"{nc4_stem}_{output_rfi_prefix}_RFI_combined.csv"
    pd.DataFrame(out_data).to_csv(out_path, index=False)
    return out_path


def _rfi_tb_table_from_combined_csv(
    df: pd.DataFrame,
    mask2: np.ndarray,
    tmbr_channel_numbers_with_rfi: Sequence[int],
    n_obs: int,
    n_tmbr_channels: int,
    accumulate: bool,
    target: Optional[np.ndarray],
    cloud_rain_atten_db_by_channel: Optional[Dict[int, np.ndarray]] = None,
    tmbr_valid2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build or update (n_obs, n_tmbr_channels) RFI brightness temperature (K) from a combined CSV.

    If ``accumulate`` is True, add into ``target`` (which must be the same shape); otherwise
    write into ``target`` (zeroed layer) or a new zeros array if ``target`` is None.

    If ``cloud_rain_atten_db_by_channel`` is set, keys are instrument channel numbers and values
    are (n_obs,) attenuation in dB; summed RFI added to the layer is multiplied by
    ``10**(-atten_dB/10)`` element-wise (non-finite attenuation treated as 0 dB).

    If ``tmbr_valid2`` is set (shape ``(n_obs, n_tmbr_channels)``), summed RFI is added only
    where that mask is True for the channel; if ``None``, use ``~mask2`` only (legacy).
    """
    if target is None:
        layer = np.zeros((n_obs, n_tmbr_channels), dtype=np.float64)
    else:
        layer = target
    for ch in tmbr_channel_numbers_with_rfi:
        col = f"channel{int(ch)}_rfi_brightness_temperature_K"
        if col not in df.columns:
            continue
        idx = int(ch) - 1
        if idx < 0 or idx >= n_tmbr_channels:
            continue
        rfi = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if rfi.shape[0] != n_obs:
            raise ValueError(f"Column {col} length {rfi.shape[0]} != {n_obs}.")
        if cloud_rain_atten_db_by_channel is not None and int(ch) in cloud_rain_atten_db_by_channel:
            att = np.asarray(cloud_rain_atten_db_by_channel[int(ch)], dtype=np.float64)
            fac = np.power(
                10.0,
                -np.where(np.isfinite(att), att, 0.0) / 10.0,
            )
            rfi = rfi * fac
        if tmbr_valid2 is not None:
            ok = tmbr_valid2[:, idx]
        else:
            ok = ~mask2[:, idx]
        if accumulate:
            layer[ok, idx] = layer[ok, idx] + rfi[ok]
        else:
            layer[ok, idx] = rfi[ok]
    return layer


def _rfi_tb_compact_from_combined_csv(
    df: pd.DataFrame,
    mask2: np.ndarray,
    tmbr_channel_numbers_with_rfi: Sequence[int],
    n_obs: int,
    n_tmbr_channels: int,
    tmbr_valid2: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[int]]:
    """
    RFI Tb (K) only for modeled channels: shape ``(n_obs, n_rfi)``, column order = sorted
    instrument channel numbers. Uses ``tmbr_valid2`` when given, else ``~mask2`` (no write
    where that TMBR channel element is masked / invalid).
    """
    ch_sorted = sorted(int(c) for c in tmbr_channel_numbers_with_rfi)
    n_rfi = len(ch_sorted)
    layer = np.zeros((n_obs, n_rfi), dtype=np.float64)
    for j, ch in enumerate(ch_sorted):
        col = f"channel{int(ch)}_rfi_brightness_temperature_K"
        if col not in df.columns:
            continue
        idx_tmbr = int(ch) - 1
        if idx_tmbr < 0 or idx_tmbr >= n_tmbr_channels:
            continue
        rfi = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if rfi.shape[0] != n_obs:
            raise ValueError(f"Column {col} length {rfi.shape[0]} != {n_obs}.")
        if tmbr_valid2 is not None:
            ok = tmbr_valid2[:, idx_tmbr]
        else:
            ok = ~mask2[:, idx_tmbr]
        layer[ok, j] = rfi[ok]
    return layer, ch_sorted


def write_attenuated_combined_rfi_top5_file(
    top5_path: Union[str, Path],
    df_sum: pd.DataFrame,
    df_5g: pd.DataFrame,
    df_sl: pd.DataFrame,
    channel_numbers: Sequence[int],
    atten_db_by_channel: Dict[int, np.ndarray],
) -> None:
    """
    Write top-5 report for (5G + gateway) RFI Tb (K) after cloud/rain path attenuation factor.

    Ranking uses absolute effective Tb. Columns in output: lat, lon, saza, RFI K (no dBW).
    """
    top5_path = Path(top5_path)
    lines_out: list[str] = []
    lines_out.append("=" * 72)
    lines_out.append(
        "Combined 5G + Starlink gateway RFI (K) after cloud/rain attenuation factor"
    )
    lines_out.append("=" * 72)

    n_obs = len(df_sum)
    for ch in sorted(int(c) for c in channel_numbers):
        col = f"channel{ch}_rfi_brightness_temperature_K"
        if col not in df_5g.columns or col not in df_sl.columns:
            continue
        sa = pd.to_numeric(df_5g[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        sb = pd.to_numeric(df_sl[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        raw = sa + sb
        att = np.asarray(
            atten_db_by_channel.get(int(ch), np.zeros(n_obs, dtype=np.float64)),
            dtype=np.float64,
        )
        fac = np.power(10.0, -np.where(np.isfinite(att), att, 0.0) / 10.0)
        tb_eff = raw * fac

        ch_header = f"[Combined attenuated] Channel {ch}"
        lines_out.append(f"\n{ch_header}\n")
        lines_out.append("  Top 5 by |RFI brightness temperature (K)| after attenuation:\n")

        abs_tb = np.abs(tb_eff)
        k = min(5, abs_tb.size)
        if k == 0:
            continue
        part = np.argpartition(abs_tb, -k)[-k:]
        idx_top5 = part[np.argsort(-abs_tb[part])]
        for rank, row_idx in enumerate(idx_top5, start=1):
            lat_v = df_sum["lat"].iloc[row_idx] if "lat" in df_sum.columns else ""
            lon_v = df_sum["lon"].iloc[row_idx] if "lon" in df_sum.columns else ""
            saza_v = (
                df_sum["saza"].iloc[row_idx] if "saza" in df_sum.columns else ""
            )
            tb_v = tb_eff[row_idx]
            line = (
                f"    [{rank}] lat: {lat_v}, lon: {lon_v}, saza: {saza_v}, "
                f"rfi_Tb_K_attenuated: {tb_v:.6g}"
            )
            lines_out.append(line + "\n")

    text = "".join(lines_out)
    top5_path.parent.mkdir(parents=True, exist_ok=True)
    top5_path.write_text(text, encoding="utf-8")
    print(f"\nAttenuated combined RFI top 5 written to {top5_path}")
    print(text)


def copy_nc4_with_tmbr_plus_rfi(
    src_nc4: Union[str, Path],
    dst_nc4: Union[str, Path],
    combined_rfi_csv: Union[str, Path],
    tmbr_channel_numbers_with_rfi: Sequence[int],
    n_tmbr_channels: int,
    tmbr_var_name: str = "TMBR",
    combined_rfi_csv_5g: Optional[Union[str, Path]] = None,
    combined_rfi_csv_starlink: Optional[Union[str, Path]] = None,
    cloud_rain_atten_db_by_channel: Optional[Dict[int, np.ndarray]] = None,
    combined_rfi_df: Optional[pd.DataFrame] = None,
    combined_rfi_df_5g: Optional[pd.DataFrame] = None,
    combined_rfi_df_starlink: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Copy a netCDF-4 file and add summed RFI brightness temperature (K) into ``TMBR``.

    The destination file is a byte copy of the source, then ``TMBR`` is updated in place.
    Rows in ``combined_rfi_csv`` must match the number of observations implied by ``TMBR``
    (product of all dimensions except the one equal to ``n_tmbr_channels``), in the same
    order as the RFI modeling scripts (C-order ravel of leading dimensions when the channel
    axis is last after transpose).

    For each instrument channel number ``N`` in ``tmbr_channel_numbers_with_rfi``, the column
    ``channelN_rfi_brightness_temperature_K`` (if present) is added to ``TMBR`` index ``N - 1``
    along the channel dimension (1-based channel numbering as in ATMS/AMSU-A/SSMI-S docs).

    Where pre-existing ``TMBR`` is invalid (masked, non-finite, ``>= NC4_MISSING_FLOAT_MIN``
    large-sentinel band, or other ``_FillValue`` / ``missing_value`` matches), no RFI is added
    on that channel and the output ``TMBR`` for that cell is set to ``NC4_MISSING_TMBR_OUT_RFI_NC4``
    (``1e10``, colleague convention for augmented files). ``CELL_RFI`` / ``GATE_RFI`` still record pre-attenuation
    RFI Tb (K) from the combined CSVs on those cells (diagnostic); only elements where the
    netCDF **mask** applies on ``TMBR`` for that channel are left unchanged (zeros in the
    compact layer). Other channels are unchanged by this rule.

    Sets ``TMBR`` attribute ``long_name`` to
    ``BRIGHTNESS TEMPERATURE with 5G and Starlink gateway``.

    If both ``combined_rfi_csv_5g`` and ``combined_rfi_csv_starlink`` are provided and exist,
    creates variables ``CELL_RFI`` (5G-only Tb RFI) and ``GATE_RFI`` (Starlink gateway-only
    Tb RFI) with the same **spatial** dimensions as ``TMBR`` but a **reduced** channel axis
    whose size equals ``len(tmbr_channel_numbers_with_rfi)`` (e.g. 7 for ATMS ch 3–9 only).
    Dimension order matches ``TMBR`` (channel axis in the same position). The channel
    dimension is named ``nchans_rfi`` so Panoply's default axis choice (lexicographic
    dimension names) yields **X = channel** and **Y = obsNumber``, like ``TMBR``'s
    ``nchans`` vs ``obsNumber``. The coordinate variable ``channel_index_rfi``
    lists the instrument channel number for each index along ``nchans_rfi``. Attributes
    ``long_name`` and ``units`` (``Kelvin``) are set on ``CELL_RFI`` and ``GATE_RFI``.

    If ``cloud_rain_atten_db_by_channel`` is provided (instrument channel -> (n_obs,) dB),
    writes ``CLOUD_RAIN_ATT`` (dB) on the same reduced channel grid as ``CELL_RFI`` /
    ``GATE_RFI``, and scales the **summed** RFI added into ``TMBR`` by ``10**(-dB/10)`` per
    FOV and channel. ``CELL_RFI`` / ``GATE_RFI`` are left as pre-attenuation Tb (K).

    Args:
        src_nc4: Original .nc4 path.
        dst_nc4: Output path (e.g. ``.../atms.2023080112_RFI.nc4``).
        combined_rfi_csv: Combined CSV with summed 5G+Starlink RFI Tb columns.
        tmbr_channel_numbers_with_rfi: Instrument channels to update (e.g. 3–9 for ATMS).
        n_tmbr_channels: Full channel count on the TMBR channel axis (22 / 15 / 24).
        tmbr_var_name: Variable name (default ``TMBR``); matched case-insensitively if missing.
        combined_rfi_csv_5g: Optional path to 5G-only combined RFI CSV for ``CELL_RFI``.
        combined_rfi_csv_starlink: Optional path to Starlink-only combined RFI CSV for ``GATE_RFI``.
        cloud_rain_atten_db_by_channel: Optional map channel -> slant attenuation (dB) per FOV.
        combined_rfi_df: If set, use this DataFrame instead of reading ``combined_rfi_csv``
            again (lowers peak RAM when the caller already loaded it).
        combined_rfi_df_5g: Same for the 5G combined CSV (must match row count of summed CSV).
        combined_rfi_df_starlink: Same for the Starlink combined CSV.

    Returns:
        Resolved path to ``dst_nc4``.

    Raises:
        FileNotFoundError: If source or primary combined CSV is missing.
        ValueError: If ``TMBR`` is missing, channel dimension is ambiguous, or row count mismatches.
    """
    try:
        from netCDF4 import Dataset
    except ImportError as e:
        raise ImportError(
            "netCDF4 is required for copy_nc4_with_tmbr_plus_rfi. Install with: pip install netCDF4"
        ) from e

    src_nc4 = Path(src_nc4).resolve()
    dst_nc4 = Path(dst_nc4).resolve()
    combined_rfi_csv = Path(combined_rfi_csv).resolve()

    if not src_nc4.is_file():
        raise FileNotFoundError(f"Source netCDF not found: {src_nc4}")
    if not combined_rfi_csv.is_file():
        raise FileNotFoundError(f"Combined RFI CSV not found: {combined_rfi_csv}")

    if combined_rfi_df is not None:
        df = combined_rfi_df
        n_rows_csv = len(df)
    else:
        df = pd.read_csv(combined_rfi_csv)
        n_rows_csv = len(df)

    ch_sorted = sorted(int(c) for c in tmbr_channel_numbers_with_rfi)
    ch_lo, ch_hi = ch_sorted[0], ch_sorted[-1]
    ch_span = f"(ch {ch_lo} - ch {ch_hi})"

    path_5g = Path(combined_rfi_csv_5g).resolve() if combined_rfi_csv_5g else None
    path_sl = Path(combined_rfi_csv_starlink).resolve() if combined_rfi_csv_starlink else None
    df_5g: Optional[pd.DataFrame] = None
    df_sl: Optional[pd.DataFrame] = None
    if combined_rfi_df_5g is not None:
        df_5g = combined_rfi_df_5g
        if len(df_5g) != n_rows_csv:
            raise ValueError(
                f"5G combined DataFrame rows ({len(df_5g)}) != summed CSV rows ({n_rows_csv})"
            )
    elif path_5g is not None and path_5g.is_file():
        df_5g = pd.read_csv(path_5g)
        if len(df_5g) != n_rows_csv:
            raise ValueError(
                f"5G combined CSV rows ({len(df_5g)}) != summed CSV rows ({n_rows_csv}): {path_5g}"
            )
    if combined_rfi_df_starlink is not None:
        df_sl = combined_rfi_df_starlink
        if len(df_sl) != n_rows_csv:
            raise ValueError(
                f"Starlink combined DataFrame rows ({len(df_sl)}) != summed CSV rows ({n_rows_csv})"
            )
    elif path_sl is not None and path_sl.is_file():
        df_sl = pd.read_csv(path_sl)
        if len(df_sl) != n_rows_csv:
            raise ValueError(
                f"Starlink combined CSV rows ({len(df_sl)}) != summed CSV rows ({n_rows_csv}): {path_sl}"
            )

    shutil.copy2(src_nc4, dst_nc4)

    with Dataset(dst_nc4, "r+") as ds:
        vname = None
        if tmbr_var_name in ds.variables:
            vname = tmbr_var_name
        else:
            lower = tmbr_var_name.lower()
            for k in ds.variables:
                if k.lower() == lower:
                    vname = k
                    break
        if vname is None:
            raise ValueError(
                f"Variable {tmbr_var_name!r} not found in {dst_nc4}. "
                f"Available: {list(ds.variables.keys())}"
            )

        v = ds.variables[vname]
        raw = v[...]
        is_ma = np.ma.isMaskedArray(raw)
        base = np.array(np.ma.getdata(raw), dtype=np.float64)
        if is_ma:
            mask = np.ma.getmaskarray(raw)
        else:
            mask = np.zeros(base.shape, dtype=bool)

        shape = base.shape
        ch_axes = [i for i, s in enumerate(shape) if int(s) == int(n_tmbr_channels)]
        if len(ch_axes) != 1:
            raise ValueError(
                f"{vname} shape {shape}: expected exactly one dimension of size "
                f"{n_tmbr_channels} (channel axis); found {len(ch_axes)} matching dim(s)."
            )
        ch_axis = ch_axes[0]
        n_obs = int(np.prod([shape[i] for i in range(base.ndim) if i != ch_axis]))
        if n_obs != n_rows_csv:
            raise ValueError(
                f"{vname} observation count ({n_obs}) does not match combined CSV rows ({n_rows_csv})."
            )

        order = [i for i in range(base.ndim) if i != ch_axis] + [ch_axis]
        base2 = np.transpose(base, order).reshape(n_obs, n_tmbr_channels)
        mask2 = np.transpose(mask, order).reshape(n_obs, n_tmbr_channels)

        fill_vals = _netcdf_float_missing_values_from_var(v)
        tmbr_valid2 = _tmbr_preexisting_valid_mask(base2, mask2, fill_vals)

        _rfi_tb_table_from_combined_csv(
            df,
            mask2,
            tmbr_channel_numbers_with_rfi,
            n_obs,
            n_tmbr_channels,
            accumulate=True,
            target=base2,
            cloud_rain_atten_db_by_channel=cloud_rain_atten_db_by_channel,
            tmbr_valid2=tmbr_valid2,
        )

        for ch in tmbr_channel_numbers_with_rfi:
            idx = int(ch) - 1
            if idx < 0 or idx >= n_tmbr_channels:
                continue
            bad = ~tmbr_valid2[:, idx]
            base2[bad, idx] = float(NC4_MISSING_TMBR_OUT_RFI_NC4)

        out_perm = base2.reshape([shape[i] for i in order])
        inv_order = np.argsort(order)
        out = np.transpose(out_perm, inv_order)

        tmbr_dtype = np.dtype(v.dtype)
        if tmbr_dtype == np.dtype(np.float64):
            v[...] = out
        else:
            v[...] = out.astype(tmbr_dtype, copy=False)

        try:
            v.setncattr(
                "long_name",
                "BRIGHTNESS TEMPERATURE with 5G and Starlink gateway",
            )
        except (AttributeError, TypeError):
            pass

        write_layers = df_5g is not None and df_sl is not None
        if not write_layers:
            has_any = (
                (path_5g is not None and path_5g.is_file())
                or (path_sl is not None and path_sl.is_file())
                or (combined_rfi_df_5g is not None)
                or (combined_rfi_df_starlink is not None)
            )
            if has_any:
                print(
                    "WARNING: copy_nc4_with_tmbr_plus_rfi: need both 5G and Starlink Gateway "
                    "combined CSVs to write CELL_RFI and GATE_RFI; skipping those variables."
                )
        else:
            # Diagnostic layers: follow CSV Tb even when native TMBR is invalid; only ma mask.
            cell2, ch_order = _rfi_tb_compact_from_combined_csv(
                df_5g,
                mask2,
                tmbr_channel_numbers_with_rfi,
                n_obs,
                n_tmbr_channels,
                tmbr_valid2=None,
            )
            gate2, _ = _rfi_tb_compact_from_combined_csv(
                df_sl,
                mask2,
                tmbr_channel_numbers_with_rfi,
                n_obs,
                n_tmbr_channels,
                tmbr_valid2=None,
            )
            n_rfi_ch = cell2.shape[1]
            spatial_sizes = [shape[order[i]] for i in range(len(order) - 1)]
            cell_perm = cell2.reshape(spatial_sizes + [n_rfi_ch])
            gate_perm = gate2.reshape(spatial_sizes + [n_rfi_ch])
            out_cell = np.transpose(cell_perm, inv_order)
            out_gate = np.transpose(gate_perm, inv_order)
            if tmbr_dtype != np.dtype(np.float64):
                out_cell = out_cell.astype(tmbr_dtype, copy=False)
                out_gate = out_gate.astype(tmbr_dtype, copy=False)

            # Name sorts before "obsNumber" (like "nchans") so Panoply defaults X=channel, Y=obs.
            rfi_dim = "nchans_rfi"
            if rfi_dim not in ds.dimensions:
                ds.createDimension(rfi_dim, n_rfi_ch)
            elif ds.dimensions[rfi_dim].size != n_rfi_ch:
                raise ValueError(
                    f"Dimension {rfi_dim!r} already exists with size "
                    f"{ds.dimensions[rfi_dim].size}, need {n_rfi_ch}."
                )

            orig_dims = v.dimensions
            new_dims = tuple(
                rfi_dim if ax == ch_axis else orig_dims[ax]
                for ax in range(len(orig_dims))
            )

            coord_name = "channel_index_rfi"
            if coord_name not in ds.variables:
                id_v = ds.createVariable(coord_name, "i4", (rfi_dim,))
                id_v[:] = np.array(ch_order, dtype=np.int32)
                id_v.setncattr(
                    "long_name",
                    "Instrument channel number per index along nchans_rfi",
                )
                id_v.setncattr(
                    "description",
                    "Maps CELL_RFI / GATE_RFI channel axis to TMBR channel index (value - 1).",
                )

            cell_v = ds.createVariable("CELL_RFI", tmbr_dtype, new_dims)
            gate_v = ds.createVariable("GATE_RFI", tmbr_dtype, new_dims)
            cell_v.setncattr("long_name", f"5G cellular network RFI {ch_span}")
            cell_v.setncattr("units", "Kelvin")
            cell_v.setncattr("coordinates", coord_name)
            gate_v.setncattr("long_name", f"Starlink ground gateway RFI {ch_span}")
            gate_v.setncattr("units", "Kelvin")
            gate_v.setncattr("coordinates", coord_name)
            cell_v[:] = out_cell
            gate_v[:] = out_gate

            if cloud_rain_atten_db_by_channel:
                atten_compact = np.zeros((n_obs, n_rfi_ch), dtype=np.float64)
                for j, chn in enumerate(ch_order):
                    if int(chn) in cloud_rain_atten_db_by_channel:
                        atten_compact[:, j] = cloud_rain_atten_db_by_channel[int(chn)]
                atten_perm = atten_compact.reshape(spatial_sizes + [n_rfi_ch])
                out_atten = np.transpose(atten_perm, inv_order)
                cloud_v = ds.createVariable("CLOUD_RAIN_ATT", "f8", new_dims)
                cloud_v.setncattr(
                    "long_name",
                    f"Cloud and rain slant attenuation {ch_span} (dB; P.840/P.838)",
                )
                cloud_v.setncattr("units", "dB")
                cloud_v.setncattr("coordinates", coord_name)
                cloud_v[:] = out_atten

        if cloud_rain_atten_db_by_channel is not None and not (
            df_5g is not None and df_sl is not None
        ):
            print(
                "WARNING: copy_nc4_with_tmbr_plus_rfi: CLOUD_RAIN_ATT not written; "
                "need both combined_rfi_csv_5g and combined_rfi_csv_starlink."
            )

    return dst_nc4
