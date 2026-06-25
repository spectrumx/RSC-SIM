"""
Starlink ground gateway utilities for NWP RFI modeling.

Provides:
- load_starlink_gateways: load gateway (lat, lon) from CSV.
- load_starlink_gateway_antenna_from_csv: gateway transmit pattern (elevation vs dBi) → ``Antenna``.
- gateways_in_fov_mask / any_gateway_in_fov: which FOVs contain at least one gateway
  (elliptical footprint; chunked for large nc4 files).
- gateways_in_fov_power_contrib: fused in-FOV mask + beam-relative receive gain + sum.
- model_rfi_nwp_starlink_gateway_single_time: link-budget RFI from ground gateways
  (power in dBW and brightness temperature in K per FOV), analogous to
  weather_sat_nwp.model_rfi_nwp_5g_single_time for 5G. Receive gain uses beam boresight
  (sat → FOV center at ``NWP_GROUND_TARGET_ALTITUDE_M``), not nadir-frame dec/caz.
- model_rfi_nwp_starlink_gateway_single_time_in_fov_first: same physics; computes
  in-FOV mask before ITU-R / link budget (only for gateways inside a footprint).

Default gateway transmit behavior uses **per-antenna boresight**
(``gateway_per_antenna_boresight=True``): ``N`` independent uniform random boresights
(azimuth ``[0,360)`` deg, elevation ``[25,90]`` deg) per site with **incoherent** linear
sum of relative tx gains; ``eirp_per_gateway_dbw`` is **per-antenna** EIRP over
``eirp_reference_bandwidth_hz`` (default 400 MHz). Set
``gateway_per_antenna_boresight=False`` for legacy co-pointed ``+10·log₁₀(N)``.
Received power is scaled by ``bandwidth_hz / eirp_reference_bandwidth_hz`` before Tb
conversion (OOBE applied in driver scripts afterward).

For realistic NWP RFI modeling you can:
1. Use any_gateway_in_fov to gate: only run RFI where a gateway is in FOV.
2. Use model_rfi_nwp_starlink_gateway_single_time (or ..._in_fov_first) to compute
   RFI power and Tb for each (timestamp, satellite) over FOVs, matching the 5G pipeline
   pattern. ``sensor_name`` may be ``'ATMS'``, ``'AMSU-A'``, or ``'SSMI-S'`` for FOV ellipse geometry
   (SSMI-S uses fixed 27×18 km axes; pass ``ellipse_azimuth_deg`` for conical-scan orientation).

**Out-of-band emissions (OOBE)** from the Starlink gateway uplink (assigned 51.4–52.4 GHz,
``B_N`` = 1 GHz) are applied in the RFI driver scripts *after* direct link-budget RFI, using
``starlink_gateway_uplink_oobe_attenuation_db`` (ITU-R SM.1541-style OOB segment and SM.329-style
60 dB spur cap vs. offset from band edge). Sensor channel center frequency sets ``A(f)``; 5G is
unchanged.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from radio_types import Antenna

# Default path to gateways CSV (relative to project root)
_DEFAULT_GATEWAYS_CSV = (
    Path(__file__).resolve().parent.parent
    / "research_tutorials"
    / "data"
    / "starlink_gateways_geolocations.csv"
)

# Default Starlink gateway transmit pattern: elevation vs absolute gain (dBi) CSV
_DEFAULT_GATEWAY_PATTERN_CSV = (
    Path(__file__).resolve().parent.parent
    / "research_tutorials"
    / "data"
    / "starlink_gateway_antenna_pattern.csv"
)

# Approx meters per degree at equator (WGS84)
_M_PER_DEG_LAT = 111320.0

# Random gateway antenna orientation (uniform): azimuth [0, 360) deg from North clockwise;
# elevation [25, 90] deg above local horizontal (see ``sample_gateway_boresight_az_el_uniform``).
GATEWAY_RANDOM_EL_MIN_DEG = 25.0
GATEWAY_RANDOM_EL_MAX_DEG = 90.0

# Starlink gateway uplink OOBE mask (GHz-equivalent constants stored in Hz for script API).
STARLINK_GATEWAY_UL_BAND_LOW_HZ = 51.4e9
STARLINK_GATEWAY_UL_BAND_HIGH_HZ = 52.4e9
STARLINK_GATEWAY_UL_BN_HZ = 1.0e9
STARLINK_GATEWAY_OOBE_SPURIOUS_DB = 60.0
STARLINK_GATEWAY_OOBE_F_MAX_PCT = 200.0

# EIRP reference bandwidth for gateway link budget (Comment 3); distinct from OOBE B_N.
DEFAULT_GATEWAY_EIRP_REFERENCE_BANDWIDTH_HZ = 400e6
# Effective radiating height under radome (m above WGS84); not NWP 15 m FOV convention.
DEFAULT_GATEWAY_ALTITUDE_M = 2.5


def starlink_gateway_uplink_oobe_attenuation_db(sensor_center_freq_hz: float) -> float:
    """
    Out-of-band emission attenuation ``A(f)`` (dB) vs. sensor channel center ``f``.

    Assigned gateway band is 51.4–52.4 GHz (inclusive); necessary bandwidth ``B_N`` = 1 GHz.
    If ``f`` is in-band, returns 0. Otherwise uses nearest band edge, offset percentage ``F``
    of ``B_N``, then SM.1541-style ``40*log10(F/50+1)`` dB for ``F <= 200`` else 60 dB
    (spurious domain). ``sensor_center_freq_hz`` is the receiver channel center in Hz.

    Example: ATMS channel 4 center ~51.76 GHz is in-band → 0 dB; channel 3 at 50.3 GHz is OOB
    with positive ``A``.
    """
    f_ghz = float(sensor_center_freq_hz) / 1.0e9
    lo = STARLINK_GATEWAY_UL_BAND_LOW_HZ / 1.0e9
    hi = STARLINK_GATEWAY_UL_BAND_HIGH_HZ / 1.0e9
    bn_ghz = STARLINK_GATEWAY_UL_BN_HZ / 1.0e9
    if lo <= f_ghz <= hi:
        return 0.0
    f_edge = lo if f_ghz < lo else hi
    f_pct = abs(f_ghz - f_edge) / bn_ghz * 100.0
    if f_pct > STARLINK_GATEWAY_OOBE_F_MAX_PCT:
        return float(STARLINK_GATEWAY_OOBE_SPURIOUS_DB)
    return float(40.0 * np.log10(f_pct / 50.0 + 1.0))


def apply_starlink_gateway_oobe_in_place(
    rfi_power_dbw: np.ndarray,
    rfi_tb_k: np.ndarray,
    sensor_center_freq_hz: float,
) -> float:
    """
    Apply OOBE mask in place: power dBW -= A, Tb K *= 10**(-A/10).

    Returns:
        ``A`` in dB (same as ``starlink_gateway_uplink_oobe_attenuation_db``).
    """
    a_db = starlink_gateway_uplink_oobe_attenuation_db(sensor_center_freq_hz)
    lin = 10.0 ** (-a_db / 10.0)
    rfi_power_dbw -= a_db
    rfi_tb_k *= lin
    return a_db


def sample_gateway_boresight_az_el_uniform(
    n: int,
    rng: np.random.Generator,
    el_min_deg: float = GATEWAY_RANDOM_EL_MIN_DEG,
    el_max_deg: float = GATEWAY_RANDOM_EL_MAX_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw independent uniform random boresight directions in local horizontal coordinates.

    Azimuth is uniform on ``[0, 360)`` degrees (from North, clockwise). Elevation is
    uniform on ``[el_min_deg, el_max_deg]`` above the local horizon.

    Returns:
        azimuth_deg, elevation_deg: each shape ``(n,)`` float64.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    az = rng.uniform(0.0, 360.0, size=n)
    el = rng.uniform(float(el_min_deg), float(el_max_deg), size=n)
    return az, el


def gateway_boresight_unit_vector_ecef(
    gateway_lat_deg: np.ndarray,
    gateway_lon_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
) -> np.ndarray:
    """
    Boresight unit vector in ECEF from local azimuth/elevation at each gateway.

    Azimuth: degrees from North, clockwise. Elevation: degrees above local horizontal
    (angle from horizontal plane up). Uses standard ENU at ``(lat, lon)`` then expresses
    the combined direction in ECEF.

    Args:
        gateway_lat_deg: Gateway latitude (degrees), shape ``(n,)``.
        gateway_lon_deg: Gateway longitude (degrees), shape ``(n,)``.
        azimuth_deg: Azimuth (degrees), shape ``(n,)``.
        elevation_deg: Elevation (degrees), shape ``(n,)``.

    Returns:
        Unit vectors shape ``(n, 3)`` in ECEF.
    """
    lat = np.deg2rad(np.asarray(gateway_lat_deg, dtype=np.float64).ravel())
    lon = np.deg2rad(np.asarray(gateway_lon_deg, dtype=np.float64).ravel())
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64).ravel())
    el = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64).ravel())
    n = lat.size
    ce = np.cos(el)
    se = np.sin(el)
    ca = np.cos(az)
    sa = np.sin(az)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros(n, dtype=np.float64)], axis=1)
    north = np.stack(
        [
            -np.sin(lat) * np.cos(lon),
            -np.sin(lat) * np.sin(lon),
            np.cos(lat),
        ],
        axis=1,
    )
    up = np.stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ],
        axis=1,
    )
    east_c = (sa * ce)[:, np.newaxis]
    north_c = (ca * ce)[:, np.newaxis]
    up_c = se[:, np.newaxis]
    u = east * east_c + north * north_c + up * up_c
    norm = np.linalg.norm(u, axis=1, keepdims=True)
    return u / (norm + 1e-15)


def load_starlink_gateway_antenna_from_csv(
    csv_file: Optional[str] = None,
    eta_rad: float = 1.0,
    valid_freqs: Tuple[float, float] = (1e9, 100e9),
) -> Antenna:
    """
    Load Starlink gateway transmit pattern from a 1-D cut CSV (elevation vs gain in dBi).

    CSV formats supported:
    - Header row: ``elevation_deg`` and a second column (e.g. ``relative_power_dBW``) with
      **absolute gain in dBi** (despite legacy column names).
    - Headerless two columns: elevation (deg), power (dBi), same as V-band weather CSV style.

    Elevation is interpreted as angle from **electrical boresight** (0° = peak gain); the
    file may span ``[-180, 180]``. Values are folded with ``abs(elevation)`` and averaged
    at duplicate angles to enforce rotational symmetry, then expanded to an azimuthally
    symmetric 2-D grid for ``Antenna`` (same approach as
    ``weather_sat_mdl.load_weather_sat_antenna_from_csv``).

    Instances are marked with ``_starlink_symmetric_elevation_cut`` so link code uses
    off-boresight angle (sat direction vs. bore) instead of the legacy 5G sector frame.

    Args:
        csv_file: Path to CSV. If None, uses ``research_tutorials/data/starlink_gateway_antenna_pattern.csv``.
        eta_rad: Radiation efficiency applied in linear domain (default 1.0 if CSV is absolute gain).
        valid_freqs: Declared frequency range (Hz) for the antenna model.

    Returns:
        ``Antenna`` with ``get_gain_values(alpha, beta)`` (radians); for this pattern,
        ``alpha`` is polar angle from boresight in radians, ``beta`` is ignored.
    """
    path = Path(csv_file) if csv_file is not None else _DEFAULT_GATEWAY_PATTERN_CSV
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Starlink gateway antenna CSV not found: {path}")

    df_try = pd.read_csv(path, nrows=2)
    if "elevation_deg" in df_try.columns:
        df = pd.read_csv(path)
        elev = np.asarray(df["elevation_deg"], dtype=np.float64).ravel()
        gain_cols = [c for c in df.columns if c != "elevation_deg"]
        if not gain_cols:
            raise ValueError(f"No gain column in {path}")
        power_db = np.asarray(df[gain_cols[0]], dtype=np.float64).ravel()
    else:
        df = pd.read_csv(path, header=None, names=["elevation", "power"])
        elev = np.asarray(df["elevation"], dtype=np.float64).ravel()
        power_db = np.asarray(df["power"], dtype=np.float64).ravel()

    theta_deg = np.abs(elev)
    agg = (
        pd.DataFrame({"theta_deg": theta_deg, "power_db": power_db})
        .groupby("theta_deg", as_index=False)
        .mean()
        .sort_values("theta_deg")
        .reset_index(drop=True)
    )
    alphas_deg = agg["theta_deg"].to_numpy(dtype=np.float64)
    power_db_u = agg["power_db"].to_numpy(dtype=np.float64)

    if alphas_deg.size < 2:
        raise ValueError(f"Need at least two theta samples in {path}")

    if not np.all(np.diff(alphas_deg) > 0):
        raise ValueError(f"theta_deg must be strictly increasing after aggregation: {path}")

    n_alpha = int(alphas_deg.size)
    n_beta = 360
    betas = np.arange(0, n_beta, dtype=np.float64)
    power_db_2d = np.tile(power_db_u[:, np.newaxis], (1, n_beta))
    gain_linear = 10.0 ** (power_db_2d / 10.0) * float(eta_rad)

    gain_pat = pd.DataFrame(
        {
            "alphas": np.tile(alphas_deg, n_beta),
            "betas": np.repeat(betas, n_alpha),
            "gains": gain_linear.flatten(order="F"),
        }
    )
    ant = Antenna.from_dataframe(gain_pat, eta_rad, valid_freqs)
    ant._starlink_symmetric_elevation_cut = True  # noqa: SLF001
    return ant


def _gateway_antenna_gain_relative_sector_boresight_ecef(
    gateway_ecef_m: np.ndarray,
    sat_ecef_m: np.ndarray,
    u_boresight_ecef: np.ndarray,
    gateway_antenna,
) -> np.ndarray:
    """
    Legacy 5G sector pattern: bore maps to +X; (alpha, beta) from orthonormal frame.
    Used only when ``gateway_antenna`` is not marked ``_starlink_symmetric_elevation_cut``.
    """
    gw = np.asarray(gateway_ecef_m, dtype=np.float64)
    sat = np.asarray(sat_ecef_m, dtype=np.float64)
    if sat.ndim == 1:
        sat = np.broadcast_to(sat.reshape(1, 3), (gw.shape[0], 3))
    u_bs = np.asarray(u_boresight_ecef, dtype=np.float64)
    u_sat = sat - gw
    u_sat = u_sat / (np.linalg.norm(u_sat, axis=1, keepdims=True) + 1e-15)

    u_up = gw / (np.linalg.norm(gw, axis=1, keepdims=True) + 1e-15)
    c0 = u_bs
    c1 = np.cross(u_up, c0)
    n1 = np.linalg.norm(c1, axis=1, keepdims=True)
    small = n1.squeeze(-1) < 1e-10
    if np.any(small):
        ez = np.zeros_like(c0)
        ez[:, 2] = 1.0
        c1_alt = np.cross(ez, c0)
        c1 = np.where(small[:, np.newaxis], c1_alt, c1)
        n1 = np.linalg.norm(c1, axis=1, keepdims=True)
    c1 = c1 / (n1 + 1e-15)
    c2 = np.cross(c0, c1)
    n2 = np.linalg.norm(c2, axis=1, keepdims=True)
    c2 = c2 / (n2 + 1e-15)

    px = np.sum(u_sat * c0, axis=1)
    py = np.sum(u_sat * c1, axis=1)
    pz = np.sum(u_sat * c2, axis=1)
    alpha = np.arccos(np.clip(pz, -1.0, 1.0))
    beta = np.mod(np.arctan2(py, px), 2.0 * np.pi)

    gain_abs = np.asarray(gateway_antenna.get_gain_values(alpha, beta), dtype=np.float64).ravel()
    peak = float(gateway_antenna.get_boresight_gain())
    if peak <= 0:
        return np.zeros(gw.shape[0], dtype=np.float64)
    return gain_abs / peak


def gateway_antenna_gain_relative_symmetric_off_boresight_ecef(
    gateway_ecef_m: np.ndarray,
    sat_ecef_m: np.ndarray,
    u_boresight_ecef: np.ndarray,
    gateway_antenna,
) -> np.ndarray:
    """
    Relative gain (linear 0–1) for a **rotationally symmetric** cut loaded via
    ``load_starlink_gateway_antenna_from_csv``.

    Off-boresight angle ``theta`` is the angle between unit vectors along boresight and
    toward the satellite; ``theta = arccos(u_bore · u_sat)``. Pattern lookup uses
    ``alpha = theta`` (radians), ``beta = 0``.
    """
    gw = np.asarray(gateway_ecef_m, dtype=np.float64)
    sat = np.asarray(sat_ecef_m, dtype=np.float64)
    if sat.ndim == 1:
        sat = np.broadcast_to(sat.reshape(1, 3), (gw.shape[0], 3))
    u_bs = np.asarray(u_boresight_ecef, dtype=np.float64)
    u_bs = u_bs / (np.linalg.norm(u_bs, axis=1, keepdims=True) + 1e-15)

    u_sat = sat - gw
    u_sat = u_sat / (np.linalg.norm(u_sat, axis=1, keepdims=True) + 1e-15)

    cos_t = np.sum(u_sat * u_bs, axis=1)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta_rad = np.arccos(cos_t)
    beta_0 = np.zeros_like(theta_rad, dtype=np.float64)

    gain_abs = np.asarray(
        gateway_antenna.get_gain_values(theta_rad, beta_0), dtype=np.float64
    ).ravel()
    peak = float(gateway_antenna.get_boresight_gain())
    if peak <= 0:
        return np.zeros(gw.shape[0], dtype=np.float64)
    return gain_abs / peak


def gateway_antenna_gain_relative_with_boresight_ecef(
    gateway_ecef_m: np.ndarray,
    sat_ecef_m: np.ndarray,
    u_boresight_ecef: np.ndarray,
    gateway_antenna,
) -> np.ndarray:
    """
    Relative gateway antenna gain toward the satellite for random (or fixed) boresight.

    If ``gateway_antenna`` was built with ``load_starlink_gateway_antenna_from_csv``,
    uses polar angle from boresight to the satellite. Otherwise uses the legacy 5G sector
    orthonormal frame (``create_5g_sector_antenna_pattern``).
    """
    if getattr(gateway_antenna, "_starlink_symmetric_elevation_cut", False):
        return gateway_antenna_gain_relative_symmetric_off_boresight_ecef(
            gateway_ecef_m, sat_ecef_m, u_boresight_ecef, gateway_antenna
        )
    return _gateway_antenna_gain_relative_sector_boresight_ecef(
        gateway_ecef_m, sat_ecef_m, u_boresight_ecef, gateway_antenna
    )


def load_starlink_gateways(csv_path=None):
    """
    Load Starlink gateway locations from CSV.

    CSV must have columns including 'lat' and 'lon'. Other columns (name, layer,
    gateway_type) are ignored for the in-FOV check.

    Args:
        csv_path: Path to CSV file. If None, uses research_tutorials/data/starlink_gateways.csv.

    Returns:
        tuple: (gateway_lat, gateway_lon) each shape (n_gateways,) float64 in degrees.
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_GATEWAYS_CSV
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Starlink gateways CSV not found: {path}")
    df = pd.read_csv(path)
    lat = np.asarray(df["lat"], dtype=np.float64).ravel()
    lon = np.asarray(df["lon"], dtype=np.float64).ravel()
    return lat, lon


def _fov_ellipse_semi_axes_km(
    n_fovs: int,
    sensor_name: str,
    fov_saza_deg,
    fov_altitude_m,
):
    """Return (d_max_km, d_min_km) vectors length ``n_fovs`` for footprint ellipse."""
    from weather_sat_nwp import (
        SSMIS_VBAND_D_MAX_KM,
        SSMIS_VBAND_D_MIN_KM,
        calculate_fov_dimensions_vectorized,
    )

    if sensor_name.upper() == "SSMI-S":
        d_max_km = np.full(n_fovs, SSMIS_VBAND_D_MAX_KM, dtype=np.float64)
        d_min_km = np.full(n_fovs, SSMIS_VBAND_D_MIN_KM, dtype=np.float64)
    else:
        if fov_saza_deg is None or fov_altitude_m is None:
            raise ValueError(
                "fov_saza_deg and fov_altitude_m are required for sensor_name ATMS/AMSU-A"
            )
        saza = np.broadcast_to(
            np.atleast_1d(np.asarray(fov_saza_deg, dtype=np.float64)).ravel(), (n_fovs,)
        )
        alt = np.broadcast_to(
            np.atleast_1d(np.asarray(fov_altitude_m, dtype=np.float64)).ravel(), (n_fovs,)
        )
        d_max_km, d_min_km = calculate_fov_dimensions_vectorized(
            saza, alt, sensor_name=sensor_name
        )
    return d_max_km, d_min_km


def _ellipse_inside_mask_core(
    fov_lat,
    fov_lon,
    gateway_lat,
    gateway_lon,
    d_max_km,
    d_min_km,
    ellipse_az_deg,
):
    """Elliptical footprint test: shape (n_fovs, n_gateways)."""
    a_km = (np.maximum(d_max_km, d_min_km) / 2.0).reshape(-1, 1)
    b_km = (np.minimum(d_max_km, d_min_km) / 2.0).reshape(-1, 1)
    delta_lat = gateway_lat[np.newaxis, :] - fov_lat.reshape(-1, 1)
    delta_lon = gateway_lon[np.newaxis, :] - fov_lon.reshape(-1, 1)
    fov_lat_rad = np.deg2rad(fov_lat)
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(fov_lat_rad)
    north_m = delta_lat * _M_PER_DEG_LAT
    east_m = delta_lon * m_per_deg_lon.reshape(-1, 1)
    az_rad = np.deg2rad(ellipse_az_deg).reshape(-1, 1)
    cos_az = np.cos(az_rad)
    sin_az = np.sin(az_rad)
    x_km = (north_m * cos_az + east_m * sin_az) / 1000.0
    y_km = (-north_m * sin_az + east_m * cos_az) / 1000.0
    return (x_km / a_km) ** 2 + (y_km / b_km) ** 2 <= 1.0


def gateways_in_fov_mask(
    fov_lat,
    fov_lon,
    gateway_lat,
    gateway_lon,
    sensor_name,
    fov_saza_deg=None,
    fov_altitude_m=None,
    ellipse_azimuth_deg=0.0,
    chunk_size=100000,
):
    """
    Vectorized (chunked): for each FOV, which gateways lie inside the elliptical footprint.

    FOV ellipse is defined by center (fov_lat, fov_lon), semi-axes d_max/2 and d_min/2 (km),
    and optional orientation ellipse_azimuth_deg (major axis from North, clockwise, degrees).
    For ATMS/AMSU-A, d_max and d_min come from SAZA and altitude; for SSMI-S they are fixed.
    Processes FOVs in chunks to avoid allocating (n_fovs x n_gateways) float64 arrays at once.

    Args:
        fov_lat: FOV center latitude (degrees), shape (n_fovs,).
        fov_lon: FOV center longitude (degrees), shape (n_fovs,).
        gateway_lat: Gateway latitude (degrees), shape (n_gateways,).
        gateway_lon: Gateway longitude (degrees), shape (n_gateways,).
        sensor_name: 'ATMS', 'AMSU-A', or 'SSMI-S' (drives FOV dimensions).
        fov_saza_deg: Satellite zenith angle (degrees), shape (n_fovs,) or scalar. Required for ATMS/AMSU-A.
        fov_altitude_m: Satellite altitude (m), shape (n_fovs,) or scalar. Required for ATMS/AMSU-A.
        ellipse_azimuth_deg: Major axis bearing from North, clockwise (degrees), shape (n_fovs,) or scalar. Default 0.
        chunk_size: Max FOVs per chunk to limit memory (default 100000). Use None to process all at once (small runs only).

    Returns:
        mask: boolean array shape (n_fovs, n_gateways). mask[i, j] True if gateway j is inside FOV i.
    """  # noqa: E501
    fov_lat = np.atleast_1d(np.asarray(fov_lat, dtype=np.float64)).ravel()
    fov_lon = np.atleast_1d(np.asarray(fov_lon, dtype=np.float64)).ravel()
    gateway_lat = np.atleast_1d(np.asarray(gateway_lat, dtype=np.float64)).ravel()
    gateway_lon = np.atleast_1d(np.asarray(gateway_lon, dtype=np.float64)).ravel()

    n_fovs = fov_lat.size
    n_gw = gateway_lat.size
    d_max_km, d_min_km = _fov_ellipse_semi_axes_km(
        n_fovs, sensor_name, fov_saza_deg, fov_altitude_m
    )
    ellipse_az = np.broadcast_to(
        np.atleast_1d(np.asarray(ellipse_azimuth_deg, dtype=np.float64)).ravel(),
        (n_fovs,),
    )

    use_chunks = chunk_size is not None and chunk_size > 0 and n_fovs > chunk_size
    if use_chunks:
        out = np.zeros((n_fovs, n_gw), dtype=np.bool_)
        for start in range(0, n_fovs, chunk_size):
            end = min(start + chunk_size, n_fovs)
            sl = slice(start, end)
            out[sl, :] = _ellipse_inside_mask_core(
                fov_lat[sl],
                fov_lon[sl],
                gateway_lat,
                gateway_lon,
                d_max_km[sl],
                d_min_km[sl],
                ellipse_az[sl],
            )
        return out

    return _ellipse_inside_mask_core(
        fov_lat,
        fov_lon,
        gateway_lat,
        gateway_lon,
        d_max_km,
        d_min_km,
        ellipse_az,
    )


def gateways_in_fov_power_contrib(
    sat_ecef_m: np.ndarray,
    fov_lat: np.ndarray,
    fov_lon: np.ndarray,
    gateway_lat: np.ndarray,
    gateway_lon: np.ndarray,
    gateway_ecef_m: np.ndarray,
    base_power_w: np.ndarray,
    weather_sat_antenna,
    sensor_name: str,
    fov_saza_deg=None,
    fov_altitude_m=None,
    ellipse_azimuth_deg=0.0,
    chunk_size=100000,
    ground_altitude_m: Optional[float] = None,
    in_fov_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fused beam-relative gateway RFI sum per FOV (W).

    For each FOV ``i`` and gateway ``j`` inside the footprint:
    ``P[i] += mask[i,j] * gain_rx[i,j] * base_power_w[j]`` where ``gain_rx`` uses
    off-boresight angle between sat→FOV center and sat→gateway (symmetric V-band pattern).

    ``base_power_w[j]`` must exclude weather-sat receive gain (FSPL, ITU P.676, tx only).
    """
    from weather_sat_mdl import latlonalt_to_ecef_vectorized
    from weather_sat_nwp import (
        NWP_GROUND_TARGET_ALTITUDE_M,
        off_boresight_angle_rad,
        unit_vectors_sat_to_targets_ecef,
        weather_sat_receive_gain_beam_relative,
    )

    fov_lat = np.atleast_1d(np.asarray(fov_lat, dtype=np.float64)).ravel()
    fov_lon = np.atleast_1d(np.asarray(fov_lon, dtype=np.float64)).ravel()
    gateway_lat = np.atleast_1d(np.asarray(gateway_lat, dtype=np.float64)).ravel()
    gateway_lon = np.atleast_1d(np.asarray(gateway_lon, dtype=np.float64)).ravel()
    base_power_w = np.asarray(base_power_w, dtype=np.float64).ravel()
    n_fovs = fov_lat.size
    n_gw = gateway_lat.size
    if n_gw == 0 or n_fovs == 0:
        return np.zeros(n_fovs, dtype=np.float64)
    if base_power_w.size != n_gw:
        raise ValueError("base_power_w length must match gateway_lat/lon")

    sat = np.asarray(sat_ecef_m, dtype=np.float64).reshape(3)
    gw_ecef = np.asarray(gateway_ecef_m, dtype=np.float64).reshape(n_gw, 3)
    u_gw = unit_vectors_sat_to_targets_ecef(sat, gw_ecef)

    if ground_altitude_m is None:
        ground_altitude_m = NWP_GROUND_TARGET_ALTITUDE_M

    if in_fov_mask is not None:
        inside_full = np.asarray(in_fov_mask, dtype=np.bool_)
        if inside_full.shape != (n_fovs, n_gw):
            raise ValueError(
                f"in_fov_mask shape {inside_full.shape} != ({n_fovs}, {n_gw})"
            )
        d_max_km = d_min_km = ellipse_az = None
    else:
        d_max_km, d_min_km = _fov_ellipse_semi_axes_km(
            n_fovs, sensor_name, fov_saza_deg, fov_altitude_m
        )
        ellipse_az = np.broadcast_to(
            np.atleast_1d(np.asarray(ellipse_azimuth_deg, dtype=np.float64)).ravel(),
            (n_fovs,),
        )

    use_chunks = chunk_size is not None and chunk_size > 0 and n_fovs > chunk_size
    rfi_power_w = np.zeros(n_fovs, dtype=np.float64)
    alt_chunk = np.full(max(1, chunk_size or n_fovs), ground_altitude_m, dtype=np.float64)

    if use_chunks:
        for start in range(0, n_fovs, chunk_size):
            end = min(start + chunk_size, n_fovs)
            sl = slice(start, end)
            n_chunk = end - start
            if in_fov_mask is not None:
                inside = inside_full[sl, :]
            else:
                inside = _ellipse_inside_mask_core(
                    fov_lat[sl],
                    fov_lon[sl],
                    gateway_lat,
                    gateway_lon,
                    d_max_km[sl],
                    d_min_km[sl],
                    ellipse_az[sl],
                )
            fov_ecef = latlonalt_to_ecef_vectorized(
                fov_lat[sl],
                fov_lon[sl],
                alt_chunk[:n_chunk],
            )
            u_fov = unit_vectors_sat_to_targets_ecef(sat, fov_ecef)
            theta = off_boresight_angle_rad(u_fov, u_gw)
            gain_rx = weather_sat_receive_gain_beam_relative(weather_sat_antenna, theta)
            rfi_power_w[sl] = np.sum(inside * gain_rx * base_power_w[np.newaxis, :], axis=1)
    else:
        if in_fov_mask is not None:
            inside = inside_full
        else:
            inside = _ellipse_inside_mask_core(
                fov_lat,
                fov_lon,
                gateway_lat,
                gateway_lon,
                d_max_km,
                d_min_km,
                ellipse_az,
            )
        fov_ecef = latlonalt_to_ecef_vectorized(
            fov_lat,
            fov_lon,
            np.full(n_fovs, ground_altitude_m, dtype=np.float64),
        )
        u_fov = unit_vectors_sat_to_targets_ecef(sat, fov_ecef)
        theta = off_boresight_angle_rad(u_fov, u_gw)
        gain_rx = weather_sat_receive_gain_beam_relative(weather_sat_antenna, theta)
        rfi_power_w = np.sum(inside * gain_rx * base_power_w[np.newaxis, :], axis=1)

    return rfi_power_w


def _gateway_tx_gain_relative_single_boresight(
    gateway_ecef_m: np.ndarray,
    ws_ecef_m: np.ndarray,
    gateway_lat: np.ndarray,
    gateway_lon: np.ndarray,
    gateway_antenna,
    dec_gw: np.ndarray,
    caz_gw: np.ndarray,
    gateway_random_boresight: bool,
    gateway_boresight_pointing: bool,
    gateway_boresight_random_seed: Optional[int],
    gateway_boresight_el_min_deg: float,
    gateway_boresight_el_max_deg: float,
) -> np.ndarray:
    """Relative gateway tx gain (linear) with one boresight per site."""
    n_gw = gateway_ecef_m.shape[0]
    if gateway_random_boresight:
        rng = np.random.default_rng(gateway_boresight_random_seed)
        az_r, el_r = sample_gateway_boresight_az_el_uniform(
            n_gw,
            rng,
            el_min_deg=gateway_boresight_el_min_deg,
            el_max_deg=gateway_boresight_el_max_deg,
        )
        u_bs = gateway_boresight_unit_vector_ecef(gateway_lat, gateway_lon, az_r, el_r)
        return gateway_antenna_gain_relative_with_boresight_ecef(
            gateway_ecef_m, ws_ecef_m, u_bs, gateway_antenna
        )
    if gateway_boresight_pointing:
        return np.ones(n_gw, dtype=np.float64)
    if getattr(gateway_antenna, "_starlink_symmetric_elevation_cut", False):
        u_ts = ws_ecef_m - gateway_ecef_m
        u_ts = u_ts / (np.linalg.norm(u_ts, axis=1, keepdims=True) + 1e-15)
        return gateway_antenna_gain_relative_with_boresight_ecef(
            gateway_ecef_m, ws_ecef_m, u_ts, gateway_antenna
        )
    dec_gw_to_sat = np.pi - dec_gw
    caz_gw_to_sat = np.mod(caz_gw + np.pi, 2.0 * np.pi)
    gain_gateway_abs = gateway_antenna.get_gain_values(dec_gw_to_sat, caz_gw_to_sat)
    peak_gw = gateway_antenna.get_boresight_gain()
    return np.where(peak_gw > 0, gain_gateway_abs / peak_gw, 0.0)


def _compute_gateway_tx_gain_relative_sum(
    gateway_ecef_m: np.ndarray,
    ws_ecef_m: np.ndarray,
    gateway_lat: np.ndarray,
    gateway_lon: np.ndarray,
    gateway_antenna,
    dec_gw: np.ndarray,
    caz_gw: np.ndarray,
    n_antennas_per_gateway: int,
    gateway_per_antenna_boresight: bool,
    gateway_random_boresight: bool,
    gateway_boresight_pointing: bool,
    gateway_boresight_random_seed: Optional[int],
    gateway_boresight_el_min_deg: float,
    gateway_boresight_el_max_deg: float,
) -> np.ndarray:
    """
    Per-gateway incoherent relative transmit gain (linear), shape ``(n_gw,)``.

    When ``gateway_per_antenna_boresight`` is True, sums ``N`` antenna gains without
    ``+10·log₁₀(N)``. Legacy co-pointed mode uses one boresight per site.
    """
    n_gw = gateway_ecef_m.shape[0]
    n_ant = max(1, int(n_antennas_per_gateway))

    if gateway_per_antenna_boresight and gateway_random_boresight:
        n_batch = n_gw * n_ant
        rng = np.random.default_rng(gateway_boresight_random_seed)
        az_r, el_r = sample_gateway_boresight_az_el_uniform(
            n_batch,
            rng,
            el_min_deg=gateway_boresight_el_min_deg,
            el_max_deg=gateway_boresight_el_max_deg,
        )
        lat_batch = np.repeat(gateway_lat, n_ant)
        lon_batch = np.repeat(gateway_lon, n_ant)
        gw_batch = np.repeat(gateway_ecef_m, n_ant, axis=0)
        ws_batch = np.repeat(ws_ecef_m, n_ant, axis=0)
        u_bs = gateway_boresight_unit_vector_ecef(lat_batch, lon_batch, az_r, el_r)
        gain_rel = gateway_antenna_gain_relative_with_boresight_ecef(
            gw_batch, ws_batch, u_bs, gateway_antenna
        )
        return gain_rel.reshape(n_gw, n_ant).sum(axis=1)

    if gateway_per_antenna_boresight:
        gain_single = _gateway_tx_gain_relative_single_boresight(
            gateway_ecef_m,
            ws_ecef_m,
            gateway_lat,
            gateway_lon,
            gateway_antenna,
            dec_gw,
            caz_gw,
            gateway_random_boresight=False,
            gateway_boresight_pointing=gateway_boresight_pointing,
            gateway_boresight_random_seed=gateway_boresight_random_seed,
            gateway_boresight_el_min_deg=gateway_boresight_el_min_deg,
            gateway_boresight_el_max_deg=gateway_boresight_el_max_deg,
        )
        return float(n_ant) * gain_single

    return _gateway_tx_gain_relative_single_boresight(
        gateway_ecef_m,
        ws_ecef_m,
        gateway_lat,
        gateway_lon,
        gateway_antenna,
        dec_gw,
        caz_gw,
        gateway_random_boresight,
        gateway_boresight_pointing,
        gateway_boresight_random_seed,
        gateway_boresight_el_min_deg,
        gateway_boresight_el_max_deg,
    )


def _compute_gateway_base_power_w(
    gateway_ecef_m: np.ndarray,
    ws_ecef_m: np.ndarray,
    gateway_lat: np.ndarray,
    gateway_lon: np.ndarray,
    gateway_antenna,
    freq_hz: float,
    eirp_per_gateway_dbw: float,
    n_antennas_per_gateway: int,
    gateway_per_antenna_boresight: bool,
    gateway_random_boresight: bool,
    gateway_boresight_pointing: bool,
    gateway_boresight_random_seed: Optional[int],
    gateway_boresight_el_min_deg: float,
    gateway_boresight_el_max_deg: float,
    temperature: float,
    pressure: float,
    humidity: float,
) -> np.ndarray:
    """Per-gateway received power (W) before weather-sat receive antenna gain."""
    from sat_mdl import free_space_loss
    from weather_sat_mdl import (
        ecef_to_weather_sat_frame,
        calculate_comprehensive_atmospheric_loss_vectorized,
    )

    n_gw = gateway_ecef_m.shape[0]
    ws_vel = np.zeros((n_gw, 3), dtype=np.float64)
    dec_gw, caz_gw = ecef_to_weather_sat_frame(gateway_ecef_m, ws_ecef_m, ws_vel)
    if np.isscalar(dec_gw):
        dec_gw = np.full(n_gw, dec_gw, dtype=np.float64)
    else:
        dec_gw = np.asarray(dec_gw, dtype=np.float64).ravel()[:n_gw]
    if np.isscalar(caz_gw):
        caz_gw = np.full(n_gw, caz_gw, dtype=np.float64)
    else:
        caz_gw = np.asarray(caz_gw, dtype=np.float64).ravel()[:n_gw]

    distance_m = np.linalg.norm(gateway_ecef_m - ws_ecef_m, axis=1)
    elevation_deg_gw = 90.0 - np.rad2deg(dec_gw)
    L_fs = free_space_loss(distance_m, freq_hz)
    L_atm = calculate_comprehensive_atmospheric_loss_vectorized(
        distance_m,
        np.full(n_gw, freq_hz),
        elevation_angles=elevation_deg_gw,
        temperature=temperature,
        pressure=pressure,
        humidity=humidity,
        use_full_itu_p676=True,
    )

    gain_tx_sum = _compute_gateway_tx_gain_relative_sum(
        gateway_ecef_m,
        ws_ecef_m,
        gateway_lat,
        gateway_lon,
        gateway_antenna,
        dec_gw,
        caz_gw,
        n_antennas_per_gateway,
        gateway_per_antenna_boresight,
        gateway_random_boresight,
        gateway_boresight_pointing,
        gateway_boresight_random_seed,
        gateway_boresight_el_min_deg,
        gateway_boresight_el_max_deg,
    )

    if gateway_per_antenna_boresight:
        eirp_linear = 10.0 ** (eirp_per_gateway_dbw / 10.0)
    else:
        effective_eirp_dbw = eirp_per_gateway_dbw + 10.0 * np.log10(
            max(1, n_antennas_per_gateway)
        )
        eirp_linear = 10.0 ** (effective_eirp_dbw / 10.0)

    link_budget = (
        (1.0 / np.where(L_fs > 0, L_fs, np.nan))
        * (1.0 / np.where(L_atm > 0, L_atm, np.nan))
        * gain_tx_sum
    )
    link_budget = np.nan_to_num(link_budget, nan=0.0, posinf=0.0, neginf=0.0)
    return eirp_linear * link_budget


def any_gateway_in_fov(
    fov_lat,
    fov_lon,
    gateway_lat,
    gateway_lon,
    sensor_name,
    fov_saza_deg=None,
    fov_altitude_m=None,
    ellipse_azimuth_deg=0.0,
    chunk_size=100000,
):
    """
    Vectorized (chunked): for each FOV, True if at least one gateway lies inside the footprint.

    Convenience wrapper around gateways_in_fov_mask; reduces over gateways with np.any(..., axis=1).

    Returns:
        mask: boolean array shape (n_fovs,). mask[i] True if any gateway is inside FOV i.
    """
    in_mask = gateways_in_fov_mask(
        fov_lat,
        fov_lon,
        gateway_lat,
        gateway_lon,
        sensor_name,
        fov_saza_deg=fov_saza_deg,
        fov_altitude_m=fov_altitude_m,
        ellipse_azimuth_deg=ellipse_azimuth_deg,
        chunk_size=chunk_size,
    )
    return np.any(in_mask, axis=1)


def model_rfi_nwp_starlink_gateway_single_time(
    sat_ecef_km: np.ndarray,
    fov_lat: np.ndarray,
    fov_lon: np.ndarray,
    fov_saza_deg: np.ndarray,
    fov_altitude_m: float,
    gateway_lat: np.ndarray,
    gateway_lon: np.ndarray,
    weather_sat_antenna,
    gateway_antenna,
    freq_hz: float = 50.3e9,
    bandwidth_hz: float = 180e6,
    eirp_per_gateway_dbw: float = 40.0,
    n_antennas_per_gateway: int = 40,
    gateway_per_antenna_boresight: bool = True,
    gateway_random_boresight: bool = True,
    gateway_boresight_pointing: bool = True,
    gateway_boresight_random_seed: Optional[int] = None,
    gateway_boresight_el_min_deg: float = GATEWAY_RANDOM_EL_MIN_DEG,
    gateway_boresight_el_max_deg: float = GATEWAY_RANDOM_EL_MAX_DEG,
    eirp_reference_bandwidth_hz: float = DEFAULT_GATEWAY_EIRP_REFERENCE_BANDWIDTH_HZ,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    humidity: float = 50.0,
    gateway_altitude_m: float = DEFAULT_GATEWAY_ALTITUDE_M,
    ellipse_azimuth_deg: float = 0.0,
    sensor_name: str = "ATMS",
    chunk_size: int = 100000,
    gateway_bbox_margin_deg: float = 1.0,
    min_elevation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RFI from Starlink ground gateways for NWP (one satellite epoch, all FOVs).

    Direct RFI at channel center frequency (no harmonic). For each FOV, sums
    received power from every gateway inside that FOV's elliptical footprint,
    using free-space loss, full ITU-R P.676 atmospheric loss (same as 5G RFI), weather-sat antenna gain
    toward each gateway, and gateway antenna gain toward the satellite. Returns
    RFI power (dBW) and brightness temperature (K) per FOV. FOVs with no gateway
    in footprint get zero power / 0 K.

    Uses gateways_in_fov_mask (chunked) to avoid large allocations; processes
    link budget per gateway once and sums over gateways per FOV in chunks.

    Args:
        sat_ecef_km: Satellite ECEF position (3,) in km.
        fov_lat: FOV center latitude (degrees), shape (n_fovs,).
        fov_lon: FOV center longitude (degrees), shape (n_fovs,).
        fov_saza_deg: Satellite zenith angle (degrees), shape (n_fovs,).
        fov_altitude_m: Satellite altitude (m), scalar.
        gateway_lat: Gateway latitude (degrees), shape (n_gateways,).
        gateway_lon: Gateway longitude (degrees), shape (n_gateways,).
        weather_sat_antenna: Weather sat antenna with get_gain_values (symmetric V-band CSV).
        gateway_antenna: Gateway antenna with get_gain_values(dec, caz) and get_boresight_gain().
        freq_hz: Observation frequency (Hz) (direct RFI at channel center).
        bandwidth_hz: Channel bandwidth (Hz).
        eirp_per_gateway_dbw: EIRP per single antenna at gateway (dBW), e.g. 70.5,
            referenced to ``eirp_reference_bandwidth_hz``. When
            ``gateway_per_antenna_boresight`` is True (default), ``N`` antennas are
            summed incoherently without ``+10·log₁₀(N)``.
        n_antennas_per_gateway: Number of antennas per gateway site.
        gateway_per_antenna_boresight: If True (default), independent boresight per
            antenna with incoherent tx gain sum. If False, legacy co-pointed site
            with ``+10·log₁₀(n_antennas_per_gateway)``.
        gateway_random_boresight: If True (default), draw uniform random azimuth
            ``[0,360)`` deg and elevation ``[gateway_boresight_el_min_deg, gateway_boresight_el_max_deg]``
            per antenna (or per site when ``gateway_per_antenna_boresight`` is False)
            and apply pattern off-boresight toward the satellite.
        gateway_boresight_pointing: Used only if ``gateway_random_boresight`` is False.
            If True, relative gateway gain = 1; if False, pattern toward satellite (legacy).
        gateway_boresight_random_seed: Optional RNG seed for reproducible random boresight.
        gateway_boresight_el_min_deg, gateway_boresight_el_max_deg: Elevation limits (deg)
            for uniform draw when ``gateway_random_boresight`` is True.
        eirp_reference_bandwidth_hz: Reference bandwidth (Hz) for EIRP; received power
            is scaled by ``bandwidth_hz / eirp_reference_bandwidth_hz`` before Tb.
        temperature, pressure, humidity: For ITU-R P.676 atmospheric loss.
        gateway_altitude_m: Gateway altitude above WGS84 (m); default 2.5 (radome height).
        ellipse_azimuth_deg: FOV ellipse orientation (degrees), scalar or (n_fovs,).
        sensor_name: 'ATMS', 'AMSU-A', or 'SSMI-S' (for FOV ellipse).
        chunk_size: FOV chunk size for in-FOV mask and power sum.
        gateway_bbox_margin_deg: If > 0, restrict in-FOV check to gateways within
            (fov_lat min/max +/- margin) and (fov_lon min/max +/- margin) to speed up.
            Use 0 or None to disable (process all gateways).
        min_elevation_deg: Exclude gateways with elevation below this (deg) from satellite.
            Default 0 excludes only below-horizon gateways; use e.g. 1.0 for a small margin.

    Returns:
        rfi_power_dBW: RFI power at satellite (dBW), shape (n_fovs,).
        rfi_tb_K: RFI brightness temperature (K), shape (n_fovs,).
    """
    from astro_mdl import power_to_temperature
    from weather_sat_mdl import (
        ecef_to_weather_sat_frame,
        latlonalt_to_ecef_vectorized,
    )

    fov_lat = np.atleast_1d(np.asarray(fov_lat, dtype=np.float64)).ravel()
    fov_lon = np.atleast_1d(np.asarray(fov_lon, dtype=np.float64)).ravel()
    fov_saza_deg = np.atleast_1d(np.asarray(fov_saza_deg, dtype=np.float64)).ravel()
    gateway_lat = np.atleast_1d(np.asarray(gateway_lat, dtype=np.float64)).ravel()
    gateway_lon = np.atleast_1d(np.asarray(gateway_lon, dtype=np.float64)).ravel()

    n_fovs = fov_lat.size
    n_gw = gateway_lat.size

    if n_gw == 0:
        return (
            np.full(n_fovs, -np.inf, dtype=np.float64),
            np.zeros(n_fovs, dtype=np.float64),
        )

    sat_ecef_m = np.asarray(sat_ecef_km, dtype=np.float64).ravel() * 1000.0
    if sat_ecef_m.size != 3:
        raise ValueError("sat_ecef_km must be length 3")
    sat_ecef_m = sat_ecef_m.reshape(3)

    gateway_alt = np.full(n_gw, gateway_altitude_m, dtype=np.float64)
    gateway_ecef = latlonalt_to_ecef_vectorized(gateway_lat, gateway_lon, gateway_alt)
    ws_ecef = np.broadcast_to(sat_ecef_m[np.newaxis, :], (n_gw, 3))
    ws_vel = np.zeros((n_gw, 3), dtype=np.float64)
    dec_gw, _caz_gw = ecef_to_weather_sat_frame(gateway_ecef, ws_ecef, ws_vel)
    if np.isscalar(dec_gw):
        dec_gw = np.full(n_gw, dec_gw, dtype=np.float64)
    else:
        dec_gw = np.asarray(dec_gw, dtype=np.float64).ravel()[:n_gw]
    elevation_deg_gw = 90.0 - np.rad2deg(dec_gw)
    above_horizon = elevation_deg_gw >= min_elevation_deg

    if gateway_bbox_margin_deg is not None and gateway_bbox_margin_deg > 0:
        lat_lo = fov_lat.min() - gateway_bbox_margin_deg
        lat_hi = fov_lat.max() + gateway_bbox_margin_deg
        lon_lo = fov_lon.min() - gateway_bbox_margin_deg
        lon_hi = fov_lon.max() + gateway_bbox_margin_deg
        in_bbox = (
            (gateway_lat >= lat_lo) & (gateway_lat <= lat_hi)
            & (gateway_lon >= lon_lo) & (gateway_lon <= lon_hi)
        )
        subset = np.where(in_bbox & above_horizon)[0]
    else:
        subset = np.where(above_horizon)[0]

    if len(subset) == 0:
        rfi_power_w = np.zeros(n_fovs, dtype=np.float64)
    else:
        gw_lat_sub = gateway_lat[subset]
        gw_lon_sub = gateway_lon[subset]
        gateway_ecef_sub = gateway_ecef[subset]
        ws_ecef_sub = ws_ecef[subset]
        base_power_sub = _compute_gateway_base_power_w(
            gateway_ecef_sub,
            ws_ecef_sub,
            gw_lat_sub,
            gw_lon_sub,
            gateway_antenna,
            freq_hz,
            eirp_per_gateway_dbw,
            n_antennas_per_gateway,
            gateway_per_antenna_boresight,
            gateway_random_boresight,
            gateway_boresight_pointing,
            gateway_boresight_random_seed,
            gateway_boresight_el_min_deg,
            gateway_boresight_el_max_deg,
            temperature,
            pressure,
            humidity,
        )
        rfi_power_w = gateways_in_fov_power_contrib(
            sat_ecef_m,
            fov_lat,
            fov_lon,
            gw_lat_sub,
            gw_lon_sub,
            gateway_ecef_sub,
            base_power_sub,
            weather_sat_antenna,
            sensor_name,
            fov_saza_deg=fov_saza_deg,
            fov_altitude_m=fov_altitude_m,
            ellipse_azimuth_deg=ellipse_azimuth_deg,
            chunk_size=chunk_size,
        )

    rfi_power_w *= bandwidth_hz / eirp_reference_bandwidth_hz
    rfi_power_dBW = 10.0 * np.log10(np.maximum(rfi_power_w, 1e-30))
    rfi_tb_K = power_to_temperature(rfi_power_w, bandwidth_hz)
    return rfi_power_dBW, rfi_tb_K


def model_rfi_nwp_starlink_gateway_single_time_in_fov_first(
    sat_ecef_km: np.ndarray,
    fov_lat: np.ndarray,
    fov_lon: np.ndarray,
    fov_saza_deg: np.ndarray,
    fov_altitude_m: float,
    gateway_lat: np.ndarray,
    gateway_lon: np.ndarray,
    weather_sat_antenna,
    gateway_antenna,
    freq_hz: float = 50.3e9,
    bandwidth_hz: float = 180e6,
    eirp_per_gateway_dbw: float = 40.0,
    n_antennas_per_gateway: int = 40,
    gateway_per_antenna_boresight: bool = True,
    gateway_random_boresight: bool = True,
    gateway_boresight_pointing: bool = True,
    gateway_boresight_random_seed: Optional[int] = None,
    gateway_boresight_el_min_deg: float = GATEWAY_RANDOM_EL_MIN_DEG,
    gateway_boresight_el_max_deg: float = GATEWAY_RANDOM_EL_MAX_DEG,
    eirp_reference_bandwidth_hz: float = DEFAULT_GATEWAY_EIRP_REFERENCE_BANDWIDTH_HZ,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    humidity: float = 50.0,
    gateway_altitude_m: float = DEFAULT_GATEWAY_ALTITUDE_M,
    ellipse_azimuth_deg: float = 0.0,
    sensor_name: str = "ATMS",
    chunk_size: int = 100000,
    gateway_bbox_margin_deg: float = 1.0,
    min_elevation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same physics and outputs as ``model_rfi_nwp_starlink_gateway_single_time``, but
    **computation order**: optional lat/lon bbox → horizon (ECEF only for survivors) →
    elliptical in-FOV mask → **then** free-space loss, full ITU-R P.676, and gains **only
    for gateways that lie inside at least one FOV** (plus the same pre-filters).

    Use this variant to avoid atmospheric / link-budget work for gateways that never
    fall in any footprint. Numerical results should match the attenuation-first function
    (within floating-point tolerance) when both are applicable.

    Args:
        Same as ``model_rfi_nwp_starlink_gateway_single_time``.

    Returns:
        rfi_power_dBW: RFI power at satellite (dBW), shape (n_fovs,).
        rfi_tb_K: RFI brightness temperature (K), shape (n_fovs,).
    """
    from astro_mdl import power_to_temperature
    from weather_sat_mdl import (
        ecef_to_weather_sat_frame,
        latlonalt_to_ecef_vectorized,
    )

    fov_lat = np.atleast_1d(np.asarray(fov_lat, dtype=np.float64)).ravel()
    fov_lon = np.atleast_1d(np.asarray(fov_lon, dtype=np.float64)).ravel()
    fov_saza_deg = np.atleast_1d(np.asarray(fov_saza_deg, dtype=np.float64)).ravel()
    gateway_lat = np.atleast_1d(np.asarray(gateway_lat, dtype=np.float64)).ravel()
    gateway_lon = np.atleast_1d(np.asarray(gateway_lon, dtype=np.float64)).ravel()

    n_fovs = fov_lat.size
    n_gw = gateway_lat.size

    if n_gw == 0:
        return (
            np.full(n_fovs, -np.inf, dtype=np.float64),
            np.zeros(n_fovs, dtype=np.float64),
        )

    sat_ecef_m = np.asarray(sat_ecef_km, dtype=np.float64).ravel() * 1000.0
    if sat_ecef_m.size != 3:
        raise ValueError("sat_ecef_km must be length 3")
    sat_ecef_m = sat_ecef_m.reshape(3)

    if gateway_bbox_margin_deg is not None and gateway_bbox_margin_deg > 0:
        lat_lo = fov_lat.min() - gateway_bbox_margin_deg
        lat_hi = fov_lat.max() + gateway_bbox_margin_deg
        lon_lo = fov_lon.min() - gateway_bbox_margin_deg
        lon_hi = fov_lon.max() + gateway_bbox_margin_deg
        in_bbox = (
            (gateway_lat >= lat_lo)
            & (gateway_lat <= lat_hi)
            & (gateway_lon >= lon_lo)
            & (gateway_lon <= lon_hi)
        )
        subset = np.flatnonzero(in_bbox)
    else:
        subset = np.arange(n_gw, dtype=np.intp)

    if subset.size == 0:
        rfi_power_w = np.zeros(n_fovs, dtype=np.float64)
        rfi_power_dBW = 10.0 * np.log10(np.maximum(rfi_power_w, 1e-30))
        rfi_tb_K = power_to_temperature(rfi_power_w, bandwidth_hz)
        return rfi_power_dBW, rfi_tb_K

    gw_lat_s = gateway_lat[subset]
    gw_lon_s = gateway_lon[subset]
    n_sub = gw_lat_s.size
    gateway_alt = np.full(n_sub, gateway_altitude_m, dtype=np.float64)
    gateway_ecef = latlonalt_to_ecef_vectorized(gw_lat_s, gw_lon_s, gateway_alt)
    ws_ecef = np.broadcast_to(sat_ecef_m[np.newaxis, :], (n_sub, 3))
    ws_vel = np.zeros((n_sub, 3), dtype=np.float64)
    dec_gw, _caz_gw = ecef_to_weather_sat_frame(gateway_ecef, ws_ecef, ws_vel)
    if np.isscalar(dec_gw):
        dec_gw = np.full(n_sub, dec_gw, dtype=np.float64)
    else:
        dec_gw = np.asarray(dec_gw, dtype=np.float64).ravel()[:n_sub]

    elevation_deg_gw = 90.0 - np.rad2deg(dec_gw)
    above_horizon = elevation_deg_gw >= min_elevation_deg
    gw_lat_s2 = gw_lat_s[above_horizon]
    gw_lon_s2 = gw_lon_s[above_horizon]
    gateway_ecef_s2 = gateway_ecef[above_horizon]

    if gw_lat_s2.size == 0:
        rfi_power_w = np.zeros(n_fovs, dtype=np.float64)
        rfi_power_dBW = 10.0 * np.log10(np.maximum(rfi_power_w, 1e-30))
        rfi_tb_K = power_to_temperature(rfi_power_w, bandwidth_hz)
        return rfi_power_dBW, rfi_tb_K

    in_fov = gateways_in_fov_mask(
        fov_lat,
        fov_lon,
        gw_lat_s2,
        gw_lon_s2,
        sensor_name,
        fov_saza_deg=fov_saza_deg,
        fov_altitude_m=fov_altitude_m,
        ellipse_azimuth_deg=ellipse_azimuth_deg,
        chunk_size=chunk_size,
    )
    active_local = np.flatnonzero(np.any(in_fov, axis=0))
    if active_local.size == 0:
        rfi_power_w = np.zeros(n_fovs, dtype=np.float64)
        rfi_power_dBW = 10.0 * np.log10(np.maximum(rfi_power_w, 1e-30))
        rfi_tb_K = power_to_temperature(rfi_power_w, bandwidth_hz)
        return rfi_power_dBW, rfi_tb_K

    gw_lat_a = gw_lat_s2[active_local]
    gw_lon_a = gw_lon_s2[active_local]
    gateway_ecef_a = gateway_ecef_s2[active_local]
    ws_ecef_a = np.broadcast_to(sat_ecef_m[np.newaxis, :], (gw_lat_a.size, 3))
    base_power_a = _compute_gateway_base_power_w(
        gateway_ecef_a,
        ws_ecef_a,
        gw_lat_a,
        gw_lon_a,
        gateway_antenna,
        freq_hz,
        eirp_per_gateway_dbw,
        n_antennas_per_gateway,
        gateway_per_antenna_boresight,
        gateway_random_boresight,
        gateway_boresight_pointing,
        gateway_boresight_random_seed,
        gateway_boresight_el_min_deg,
        gateway_boresight_el_max_deg,
        temperature,
        pressure,
        humidity,
    )
    rfi_power_w = gateways_in_fov_power_contrib(
        sat_ecef_m,
        fov_lat,
        fov_lon,
        gw_lat_a,
        gw_lon_a,
        gateway_ecef_a,
        base_power_a,
        weather_sat_antenna,
        sensor_name,
        fov_saza_deg=fov_saza_deg,
        fov_altitude_m=fov_altitude_m,
        ellipse_azimuth_deg=ellipse_azimuth_deg,
        chunk_size=chunk_size,
        in_fov_mask=in_fov[:, active_local],
    )

    rfi_power_w *= bandwidth_hz / eirp_reference_bandwidth_hz
    rfi_power_dBW = 10.0 * np.log10(np.maximum(rfi_power_w, 1e-30))
    rfi_tb_K = power_to_temperature(rfi_power_w, bandwidth_hz)
    return rfi_power_dBW, rfi_tb_K
