"""
RFI modeling for SSMI-S for NWP data biasing (5G + Starlink ground gateways).

Computes (1) 5G ground-emitter RFI via second harmonic in band and (2) Starlink ground-gateway RFI (direct link at gateway carrier, then uplink OOBE vs. sensor
center) for all FOVs in SSMI-S netCDF-4 files.
Each nc4 can contain multiple satellites (DMSP-F17, DMSP-F18); only SAID 285 (DMSP-F17)
is processed for RFI when ECEF lookups exist—same as the standalone scripts. Parametric
study over SSMI-S channels 1–5.

Execution order: all 5G channels first, then all Starlink gateway channels. Outputs
per-source CSVs, combined CSVs, unified top-5 (``*_5G_Starlink_Gateway_top5.txt``),
and summed Tb combined CSV (``*_5G_Starlink_Gateway_RFI_combined.csv``).

Starlink FOV footprints use fixed V-band ellipse (27 km × 18 km) from ``weather_sat_nwp``;
ellipse orientation uses ``calculate_ssmis_fov_bearing_vectorized`` when the FOV count per
run is a multiple of 60, else North-aligned (see ``_ssmis_ellipse_azimuth_deg``).

Full ITU-R P.676 atmospheric absorption; no polarization loss, terrain masking, or OOBE for 5G.
Starlink gateway CSVs apply uplink OOBE vs. sensor center (``starlink_gateway_mdl``) after
direct link-budget RFI. Cloud/rain slant attenuation scales summed RFI into ``TMBR_RFI`` and
is in ``CLOUD_RAIN_ATT`` (dB).

Usage:
  python SSMI-S_RFI_modeling.py --sensor SSMI-S --nc4 path [--out_dir dir] [--gateways_csv path]

  python SSMI-S_RFI_modeling.py --sensor SSMI-S --nc4 util/SSMI-S/ssmis.2023080112.nc4 --out_dir util/SSMI-S

  Default 5G density uses contiguous dense-urban metro GHSL (place shared
  ``GHS_POP_*_metro.tif`` in ``research_tutorials/data/``). Per-cell legacy:
  add ``--legacy-per-cell-5g``.

Also writes ``<stem>_RFI.nc4``: native ``TMBR`` unchanged; ``TMBR_RFI`` = ``TMBR`` + summed RFI Tb on ch 1–5 after cloud/rain factor;
``CELL_RFI`` / ``GATE_RFI`` hold 5G-only and gateway-only Tb (K) on a compact channel axis (ch 1–5 only);
``CLOUD_RAIN_ATT`` (dB) on the same axis; channel dim ``nchans_rfi`` (Panoply defaults like ``nchans``);
coordinate ``channel_index_rfi`` gives instrument channel per index. Top-5 attenuated Tb:
``*_5G_Starlink_Gateway_Attenuation_top5.txt``.
"""  # noqa: E501

import argparse
import gc
import os
import sys
import time as time_module
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add src to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(os.path.dirname(_script_dir), "src")
sys.path.insert(0, _src_dir)

from netCDF4 import Dataset  # noqa: E402

from attenuation_mdl import (  # noqa: E402
    atten_db_to_by_channel_dict,
    compute_cloud_rain_atten_db_for_fovs,
    itu_iclw_rain_info_nc_path,
)
from starlink_gateway_mdl import (  # noqa: E402
    DEFAULT_GATEWAY_ALTITUDE_M,
    DEFAULT_GATEWAY_EIRP_REFERENCE_BANDWIDTH_HZ,
    apply_starlink_gateway_oobe_in_place,
    load_starlink_gateways,
    load_starlink_gateway_antenna_from_csv,
    model_rfi_nwp_starlink_gateway_single_time,
    model_rfi_nwp_starlink_gateway_single_time_in_fov_first,
    starlink_gateway_uplink_oobe_attenuation_db,
)
from weather_sat_mdl import (  # noqa: E402
    create_5g_sector_antenna_pattern,
    load_weather_sat_antenna_from_csv,
)
from weather_sat_nwp import (  # noqa: E402
    calculate_ssmis_fov_bearing_vectorized,
    clear_ghsl_metro_raster_cache,
    clear_ghsl_raster_cache,
    combine_channel_csvs,
    copy_nc4_with_tmbr_plus_rfi,
    get_emitter_density_metro_vectorized,
    get_emitter_density_vectorized,
    iter_valid_ts_sat_indices,
    load_country_5g_sensor_channel_csv,
    load_ecef_lookups_for_nc4,
    model_rfi_nwp_5g_single_time_ssmis,
    obs_valid_ssmis_conical,
    replace_missing_with_nan,
    resolve_ghsl_metro_tif_path,
    said_to_satellite_array,
    SENSOR_ALLOWED_SAIDS,
    SENSOR_SAID_TO_SATELLITE,
    sum_two_rfi_combined_csvs_by_channel,
    supported_5g_countries_for_channel,
    timestamp_from_nc4_vars,
    write_attenuated_combined_rfi_top5_file,
)

# =============================================================================
# Configuration (override with CLI where applicable)
# =============================================================================

# V-band (50–55.5 GHz), SSMI-S Ch 1–5: parametric study over channels
# Each entry: (channel_number, center_freq_Hz, bandwidth_Hz). Use Hz, not GHz
SSMIS_CHANNEL_CONFIGS = [
    (1, 50.3e9, 400e6),
    (2, 52.8e9, 400e6),
    (3, 53.596e9, 400e6),
    (4, 54.4e9, 400e6),
    (5, 55.5e9, 400e6),
]
# Emitter fundamental (Hz) per channel; 2nd harmonic = 2 * fundamental at channel center.
# One element per SSMIS_CHANNEL_CONFIGS entry, same order. Use Hz, not GHz
emitter_fundamental_hz_list = [
    SSMIS_CHANNEL_CONFIGS[0][1] / 2.0,  # Ch 1
    SSMIS_CHANNEL_CONFIGS[1][1] / 2.0,  # Ch 2
    SSMIS_CHANNEL_CONFIGS[2][1] / 2.0,  # Ch 3
    SSMIS_CHANNEL_CONFIGS[3][1] / 2.0,  # Ch 4
    SSMIS_CHANNEL_CONFIGS[4][1] / 2.0,  # Ch 5
]

gateway_center_freq_hz_list = [
    SSMIS_CHANNEL_CONFIGS[0][1] / 1,  # Ch 1
    SSMIS_CHANNEL_CONFIGS[1][1] / 1,  # Ch 2
    SSMIS_CHANNEL_CONFIGS[2][1] / 1,  # Ch 3
    SSMIS_CHANNEL_CONFIGS[3][1] / 1,  # Ch 4
    SSMIS_CHANNEL_CONFIGS[4][1] / 1,  # Ch 5
]

# 5G sector antenna
# ITU-R M.2101 and 3GPP standards recommend 24 - 25 dBi for 5G sector antenna with standard 8x8 phased array
GROUND_EMITTER_GAIN_MAX = 24.5  # dBi
GROUND_EMITTER_HORIZ_BW = 65.0
GROUND_EMITTER_VERT_BW = 10.0
GROUND_EMITTER_ETA_RAD = 0.8

TRANSMIT_POWER_DBW = -33
EIRP_PER_EMITTER_DBW = TRANSMIT_POWER_DBW + GROUND_EMITTER_GAIN_MAX

# Starlink ground gateway
EIRP_PER_GATEWAY_DBW = 70.5
N_ANTENNAS_PER_GATEWAY = 40
GATEWAY_EIRP_REFERENCE_BANDWIDTH_HZ = DEFAULT_GATEWAY_EIRP_REFERENCE_BANDWIDTH_HZ
GATEWAY_ALTITUDE_M = DEFAULT_GATEWAY_ALTITUDE_M
GATEWAY_PER_ANTENNA_BORESIGHT = True
GATEWAY_BORESIGHT_POINTING = True
GATEWAY_GAIN_MAX = 24.5
GATEWAY_HORIZ_BW = 65.0
GATEWAY_VERT_BW = 10.0
GATEWAY_ETA_RAD = 0.8

# Atmopheric defaults for ITU-R P.676
TEMPERATURE_K = 288.15
PRESSURE_PA = 101325.0
HUMIDITY_PCT = 50.0

# netCDF-4 (.nc4) variable names
VAR_LAT = "LAT"
VAR_LON = "LON"
VAR_YEAR = "YEAR"
VAR_MONTH = "MONTH"
VAR_DAYS = "DAYS"
VAR_HOUR = "HOUR"
VAR_MINU = "MINUTE"
VAR_SECO = "SECOND"
VAR_SAID = "SAID"
SENSOR_NAME = "SSMI-S"

SSMIS_ALTITUDE_M = 850_000.0
SSMIS_SLANT_RANGE_KM = 1020.0
SSMIS_ELEVATION_DEG = 36.9
# Conical geometry in this repo: zenith from nadir = 90° − incidence elevation (see SSMIS_RFI_trace_one_scan.md).
SSMIS_SAZA_DEG = 90.0 - SSMIS_ELEVATION_DEG
SSMIS_FOVS_PER_SCAN = 60

RFI_PREFIX_5G = "5G"
RFI_PREFIX_STARLINK = "Starlink_Gateway"
TOP5_STEM = "5G_Starlink_Gateway"
COMBINED_SUM_PREFIX = "5G_Starlink_Gateway"

TMBR_N_CHANNELS = 24

CLOUD_RAIN_ICLW_ABS_THRESHOLD = 0.05

_SSMIS_BEARING_FALLBACK_WARNED = False


def _read_nc4_var(ds, name: str, fallback_name: str = None):
    """Read variable; optionally try fallback (e.g. HMSL if HSML missing)."""
    if name in ds.variables:
        return np.asarray(ds.variables[name][:])
    if fallback_name and fallback_name in ds.variables:
        return np.asarray(ds.variables[fallback_name][:])
    raise KeyError(f"Variable {name} (or {fallback_name}) not found in nc4.")


def _flatten_or_repeat_time(var, n_fov_per_scan: int):
    """If var is 1D (per scan), repeat to (n_scan * n_fov_per_scan,). Else flatten."""
    v = np.asarray(var)
    if v.ndim == 1:
        return np.repeat(v, n_fov_per_scan)
    return v.ravel()


def load_ssmis_nc4_and_build_arrays(nc4_path: str):
    """
    Load SSMI-S nc4 (LAT, LON, time; SAID if present). Return flat arrays: lat, lon,
    altitude_m (constant), timestamps, satellite (per obs), n_obs. Only SAID 285 (DMSP-F17)
    is used for RFI when allowed by SENSOR_ALLOWED_SAIDS.
    """
    with Dataset(nc4_path, "r") as ds:
        lat = _read_nc4_var(ds, VAR_LAT)
        lon = _read_nc4_var(ds, VAR_LON)
        year = _read_nc4_var(ds, VAR_YEAR)
        month = _read_nc4_var(ds, VAR_MONTH)
        days = _read_nc4_var(ds, VAR_DAYS)
        hour = _read_nc4_var(ds, VAR_HOUR)
        minu = _read_nc4_var(ds, VAR_MINU)
        seco = _read_nc4_var(ds, VAR_SECO)
        said = _read_nc4_var(ds, VAR_SAID) if VAR_SAID in ds.variables else None

    if lat.ndim == 2:
        n_scan, n_fov = lat.shape
        n_obs = n_scan * n_fov
        lat = lat.ravel()
        lon = lon.ravel()
        year = _flatten_or_repeat_time(year, n_fov)
        month = _flatten_or_repeat_time(month, n_fov)
        days = _flatten_or_repeat_time(days, n_fov)
        hour = _flatten_or_repeat_time(hour, n_fov)
        minu = _flatten_or_repeat_time(minu, n_fov)
        seco = _flatten_or_repeat_time(seco, n_fov)
        if said is not None:
            said = _flatten_or_repeat_time(said, n_fov)
    else:
        n_obs = int(lat.size)
        lat = np.atleast_1d(lat).ravel()
        lon = np.atleast_1d(lon).ravel()
        year = np.resize(np.atleast_1d(year).ravel(), n_obs)
        month = np.resize(np.atleast_1d(month).ravel(), n_obs)
        days = np.resize(np.atleast_1d(days).ravel(), n_obs)
        hour = np.resize(np.atleast_1d(hour).ravel(), n_obs)
        minu = np.resize(np.atleast_1d(minu).ravel(), n_obs)
        seco = np.resize(np.atleast_1d(seco).ravel(), n_obs)
        if said is not None:
            said = np.resize(np.atleast_1d(said).ravel(), n_obs)

    lat = replace_missing_with_nan(lat)
    lon = replace_missing_with_nan(lon)
    year = replace_missing_with_nan(year)
    month = replace_missing_with_nan(month)
    days = replace_missing_with_nan(days)
    hour = replace_missing_with_nan(hour)
    minu = replace_missing_with_nan(minu)
    seco = replace_missing_with_nan(seco)
    if said is not None:
        said_f = replace_missing_with_nan(np.asarray(said, dtype=np.float64))
        said_int = np.where(np.isfinite(said_f), said_f, -1.0).astype(np.int64)
        satellite = said_to_satellite_array(said_int, SENSOR_NAME)
    else:
        satellite = np.array([""] * n_obs, dtype=object)
    timestamps = timestamp_from_nc4_vars(year, month, days, hour, minu, seco)
    obs_valid = obs_valid_ssmis_conical(
        lat, lon, year, month, days, hour, minu, seco, satellite
    )
    return {
        "lat": lat,
        "lon": lon,
        "altitude_m": SSMIS_ALTITUDE_M,
        "year": year,
        "month": month,
        "days": days,
        "hour": hour,
        "minu": minu,
        "seco": seco,
        "timestamps": timestamps,
        "satellite": satellite,
        "n_obs": n_obs,
        "obs_valid": obs_valid,
    }


def _ssmis_ellipse_azimuth_deg(fov_lat: np.ndarray, fov_lon: np.ndarray) -> np.ndarray:
    global _SSMIS_BEARING_FALLBACK_WARNED
    fov_lat = np.asarray(fov_lat, dtype=np.float64).ravel()
    fov_lon = np.asarray(fov_lon, dtype=np.float64).ravel()
    n = fov_lat.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n % SSMIS_FOVS_PER_SCAN == 0:
        return calculate_ssmis_fov_bearing_vectorized(
            fov_lat, fov_lon, fovs_per_scan=SSMIS_FOVS_PER_SCAN
        )
    if not _SSMIS_BEARING_FALLBACK_WARNED:
        warnings.warn(
            f"SSMI-S FOV count {n} is not a multiple of {SSMIS_FOVS_PER_SCAN}; "
            "using ellipse_azimuth_deg=0 (North). For correct conical-scan orientation, "
            "each (timestamp, satellite) group should contain full scans in file order.",
            UserWarning,
            stacklevel=2,
        )
        _SSMIS_BEARING_FALLBACK_WARNED = True
    return np.zeros(n, dtype=np.float64)


def run_rfi_for_channel_5g(
    data: dict,
    density: np.ndarray,
    ecef_by_satellite: dict,
    v_band_antenna,
    emitter_antenna,
    channel_num: int,
    center_freq_hz: float,
    bandwidth_hz: float,
    emitter_fundamental_hz: float,
    out_csv: str,
):
    """Run 5G RFI for one SSMI-S channel; write CSV with constant ``saza`` (fixed conical geometry)."""
    lat = data["lat"]
    lon = data["lon"]
    timestamps = data["timestamps"]
    satellite = data["satellite"]
    n_obs = data["n_obs"]
    obs_valid = data.get("obs_valid")
    if obs_valid is None:
        obs_valid = np.ones(n_obs, dtype=bool)
    else:
        obs_valid = np.asarray(obs_valid, dtype=bool).ravel()
    allowed = set(
        SENSOR_SAID_TO_SATELLITE[SENSOR_NAME][said]
        for said in SENSOR_ALLOWED_SAIDS[SENSOR_NAME]
    )

    rfi_dBW = np.full(n_obs, -300.0)
    rfi_K = np.full(n_obs, 0.0)

    for ts, sat, idx, coords in iter_valid_ts_sat_indices(
        timestamps, satellite, allowed, ecef_by_satellite
    ):
        idx_good = idx[obs_valid[idx]]
        if idx_good.size == 0:
            continue
        sat_ecef_km = np.array(coords, dtype=np.float64)
        rfi_db, rfi_tb = model_rfi_nwp_5g_single_time_ssmis(
            sat_ecef_km,
            lat[idx_good],
            lon[idx_good],
            density[idx_good],
            v_band_antenna,
            emitter_antenna,
            freq_hz=center_freq_hz,
            bandwidth_hz=bandwidth_hz,
            eirp_per_emitter_dbw=EIRP_PER_EMITTER_DBW,
            emitter_fundamental_freq=emitter_fundamental_hz,
            temperature=TEMPERATURE_K,
            pressure=PRESSURE_PA,
            humidity=HUMIDITY_PCT,
            slant_range_km=SSMIS_SLANT_RANGE_KM,
            elevation_deg=SSMIS_ELEVATION_DEG,
        )
        rfi_dBW[idx_good] = rfi_db
        rfi_K[idx_good] = rfi_tb

    df = pd.DataFrame({
        "timestamp": timestamps,
        "satellite": satellite,
        "lat": np.round(lat, 6),
        "lon": np.round(lon, 6),
        "saza": np.round(np.full(n_obs, SSMIS_SAZA_DEG, dtype=np.float64), 6),
        "rfi_power_dBW": np.round(rfi_dBW, 3),
        "rfi_brightness_temperature_K": [f"{x:.3e}" for x in rfi_K],
    })
    df.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv} ({len(df):,} rows)")
    return df


def run_rfi_for_channel_starlink_gateway(
    data: dict,
    gateway_lat: np.ndarray,
    gateway_lon: np.ndarray,
    ecef_by_satellite: dict,
    v_band_antenna,
    gateway_antenna,
    channel_num: int,
    center_freq_hz: float,
    sensor_center_freq_hz: float,
    bandwidth_hz: float,
    out_csv: str,
    sensor_name: str = "SSMI-S",
    chunk_size: int = 200000,
    gateway_bbox_margin_deg: float = 1.0,
    in_fov_first: bool = True,
    gateway_random_boresight: bool = True,
    gateway_boresight_pointing: bool = True,
    gateway_boresight_random_seed: Optional[int] = None,
    gateway_per_antenna_boresight: bool = GATEWAY_PER_ANTENNA_BORESIGHT,
    eirp_reference_bandwidth_hz: float = GATEWAY_EIRP_REFERENCE_BANDWIDTH_HZ,
    gateway_altitude_m: float = GATEWAY_ALTITUDE_M,
    profile_rfi: bool = False,
):
    """Starlink gateway RFI for one channel; OOBE vs. ``sensor_center_freq_hz``; CSV has constant ``saza``."""
    lat = data["lat"]
    lon = data["lon"]
    timestamps = data["timestamps"]
    satellite = data["satellite"]
    altitude_m = data["altitude_m"]
    n_obs = data["n_obs"]
    obs_valid = data.get("obs_valid")
    if obs_valid is None:
        obs_valid = np.ones(n_obs, dtype=bool)
    else:
        obs_valid = np.asarray(obs_valid, dtype=bool).ravel()
    allowed = set(
        SENSOR_SAID_TO_SATELLITE[sensor_name][said]
        for said in SENSOR_ALLOWED_SAIDS[sensor_name]
    )

    rfi_dBW = np.full(n_obs, -300.0)
    rfi_K = np.full(n_obs, 0.0)

    _model = (
        model_rfi_nwp_starlink_gateway_single_time_in_fov_first
        if in_fov_first
        else model_rfi_nwp_starlink_gateway_single_time
    )
    t_model_s = 0.0
    n_epoch_calls = 0

    for ts, sat, idx, coords in iter_valid_ts_sat_indices(
        timestamps, satellite, allowed, ecef_by_satellite
    ):
        idx_good = idx[obs_valid[idx]]
        if idx_good.size == 0:
            continue
        sat_ecef_km = np.array(coords, dtype=np.float64)
        fov_lat = lat[idx_good]
        fov_lon = lon[idx_good]
        n_fov = int(np.asarray(fov_lat).size)
        fov_saza_dummy = np.zeros(n_fov, dtype=np.float64)
        ellipse_az_deg = _ssmis_ellipse_azimuth_deg(fov_lat, fov_lon)

        t0 = time_module.perf_counter() if profile_rfi else 0.0
        rfi_db, rfi_tb = _model(
            sat_ecef_km,
            fov_lat,
            fov_lon,
            fov_saza_dummy,
            altitude_m,
            gateway_lat,
            gateway_lon,
            v_band_antenna,
            gateway_antenna,
            freq_hz=center_freq_hz,
            bandwidth_hz=bandwidth_hz,
            eirp_per_gateway_dbw=EIRP_PER_GATEWAY_DBW,
            n_antennas_per_gateway=N_ANTENNAS_PER_GATEWAY,
            gateway_per_antenna_boresight=gateway_per_antenna_boresight,
            gateway_random_boresight=gateway_random_boresight,
            gateway_boresight_pointing=gateway_boresight_pointing,
            gateway_boresight_random_seed=gateway_boresight_random_seed,
            eirp_reference_bandwidth_hz=eirp_reference_bandwidth_hz,
            gateway_altitude_m=gateway_altitude_m,
            temperature=TEMPERATURE_K,
            pressure=PRESSURE_PA,
            humidity=HUMIDITY_PCT,
            sensor_name="SSMI-S",
            chunk_size=chunk_size,
            gateway_bbox_margin_deg=gateway_bbox_margin_deg,
            ellipse_azimuth_deg=ellipse_az_deg,
        )
        if profile_rfi:
            t_model_s += time_module.perf_counter() - t0
            n_epoch_calls += 1
        rfi_dBW[idx_good] = rfi_db
        rfi_K[idx_good] = rfi_tb

    if profile_rfi:
        print(
            f"  [profile] Starlink ch{channel_num}: model calls={n_epoch_calls}, "
            f"model time={t_model_s:.2f} s"
        )

    apply_starlink_gateway_oobe_in_place(rfi_dBW, rfi_K, sensor_center_freq_hz)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "satellite": satellite,
        "lat": np.round(lat, 6),
        "lon": np.round(lon, 6),
        "saza": np.round(np.full(n_obs, SSMIS_SAZA_DEG, dtype=np.float64), 6),
        "rfi_power_dBW": np.round(rfi_dBW, 3),
        "rfi_brightness_temperature_K": [f"{x:.3e}" for x in rfi_K],
    })
    df.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv} ({len(df):,} rows)")
    return df


def _append_top5_block(
    top5_file,
    ch_header: str,
    df: pd.DataFrame,
    *,
    oobe_atten_db: Optional[float] = None,
    gateway_rfi_computed: bool = True,
):
    top5_file.write(f"\n{ch_header}\n")
    if not gateway_rfi_computed:
        msg = (
            "  OOBE: N/A (gateway carrier out of channel passband; no direct RFI computed)."
        )
        print(msg)
        top5_file.write(msg + "\n")
    elif oobe_atten_db is not None:
        line_a = (
            f"  OOBE attenuation A(f) = {oobe_atten_db:.3f} dB "
            f"(per channel; f = sensor center)."
        )
        print(line_a)
        top5_file.write(line_a + "\n")
        if oobe_atten_db <= 0.0:
            line_b = (
                "  Top-5 Tb and power are post-OOBE (in-band: A=0 dB; same as direct-only "
                "gateway RFI)."
            )
        else:
            line_b = (
                "  Top-5 Tb and power are post-OOBE (direct gateway RFI, then ITU-R SM.1541 / "
                "SM.329-style OOBE mask)."
            )
        print(line_b)
        top5_file.write(line_b + "\n")

    rfi_K_float = df["rfi_brightness_temperature_K"].astype(float)
    idx_top5 = rfi_K_float.abs().nlargest(5).index
    print("  Top 5 by |rfi_brightness_temperature_K|:")
    top5_file.write("  Top 5 by |rfi_brightness_temperature_K|:\n")
    for rank, row_idx in enumerate(idx_top5, start=1):
        row = df.loc[row_idx]
        line = (
            f"    [{rank}] satellite: {row['satellite']}, lat: {row['lat']}, lon: {row['lon']}, "
            f"saza: {row['saza']}, "
            f"rfi_power_dBW: {row['rfi_power_dBW']}, rfi_Tb_K: {row['rfi_brightness_temperature_K']}"
        )
        print(line)
        top5_file.write(line + "\n")


def main():
    t_start = time_module.perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "SSMI-S RFI for NWP: 5G (harmonic) and Starlink gateways "
            "(direct link + uplink OOBE mask); SAID 285/DMSP-F17 when ECEF exists."
        )
    )
    parser.add_argument(
        "--sensor",
        required=True,
        metavar="SENSOR",
        help="Sensor name (e.g. SSMI-S).",
    )
    parser.add_argument(
        "--nc4",
        required=True,
        help="Path to SSMI-S netCDF-4 file. ECEF lookups from same dir.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for RFI CSVs (default: same directory as nc4 file).",
    )
    parser.add_argument(
        "--gateways_csv",
        default=None,
        help=(
            "Path to Starlink gateways CSV (lat, lon). "
            "Default: research_tutorials/data/starlink_gateways_geolocations.csv"
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200000,
        help="FOV chunk size for gateway in-FOV / link budget (default 200000).",
    )
    parser.add_argument(
        "--no_gateway_bbox",
        action="store_true",
        help="Disable gateway lat/lon bounding-box filter (slower; debugging).",
    )
    parser.add_argument(
        "--attenuation_first",
        action="store_true",
        help="Gateway: ITU-R before in-FOV mask. Default is in-FOV-first.",
    )
    parser.add_argument(
        "--legacy_gateway_boresight_at_satellite",
        action="store_true",
        help="Gateway: bore sight at satellite (gain 1). Default: random az/el + sector pattern.",
    )
    parser.add_argument(
        "--gateway_boresight_random_seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Optional RNG seed for gateway random boresight.",
    )
    parser.add_argument(
        "--legacy_co_pointed_gateways",
        action="store_true",
        help=(
            "Gateway: legacy co-pointed site (+10·log₁₀(N), one boresight per site). "
            "Default is per-antenna independent boresight with incoherent tx sum."
        ),
    )
    parser.add_argument(
        "--legacy-per-cell-5g",
        action="store_true",
        help=(
            "Use legacy per-cell ultra-dense GHSL (original GeoTIFF, no metro "
            "contiguity filter). Default is metro-contiguous density."
        ),
    )
    parser.add_argument(
        "--profile_rfi",
        action="store_true",
        help=(
            "Print Starlink gateway model timing per channel. "
            "Also enabled when env RSCSIM_PROFILE_RFI=1."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.nc4):
        print(f"ERROR: nc4 file not found: {args.nc4}")
        sys.exit(1)
    if args.sensor != "SSMI-S":
        print(f"ERROR: This script is for SSMI-S sensor; got --sensor {args.sensor!r}")
        sys.exit(1)

    ecef_by_satellite = load_ecef_lookups_for_nc4(args.nc4)
    if not ecef_by_satellite:
        print(
            f"ERROR: No ECEF lookup CSVs found for {args.nc4} "
            "(expect *_ECEF_lookup_<stem>.csv in same dir)"
        )
        sys.exit(1)
    print(f"  Loaded ECEF lookups for satellites: {list(ecef_by_satellite.keys())}")

    gateway_lat, gateway_lon = load_starlink_gateways(csv_path=args.gateways_csv)
    print(f"  Loaded {len(gateway_lat):,} Starlink gateways")

    data_dir = os.path.join(_script_dir, "data")
    v_band_csv = os.path.join(data_dir, "SSMI-S V-Band absolute antenna pattern.csv")
    if not os.path.isfile(v_band_csv):
        print(
            f"ERROR: V-band antenna CSV required for beam-relative receive gain: {v_band_csv}"
        )
        print(
            "  Comment 2 path requires the symmetric V-band CSV (no ITU 2D fallback)."
        )
        sys.exit(1)
    v_band_antenna = load_weather_sat_antenna_from_csv(
        v_band_csv,
        eta_rad=0.99,
        valid_freqs=(40e9, 60e9),
    )

    emitter_antenna = create_5g_sector_antenna_pattern(
        gain_max=GROUND_EMITTER_GAIN_MAX,
        horiz_beamwidth=GROUND_EMITTER_HORIZ_BW,
        vert_beamwidth=GROUND_EMITTER_VERT_BW,
        eta_rad=GROUND_EMITTER_ETA_RAD,
        valid_freqs=(1e9, 100e9),
    )
    gateway_pattern_csv = os.path.join(data_dir, "starlink_gateway_antenna_pattern.csv")
    if not os.path.exists(gateway_pattern_csv):
        print(
            f"WARNING: Starlink gateway antenna CSV not found: {gateway_pattern_csv}\n"
            "  Using legacy 5G sector pattern (see create_5g_sector_antenna_pattern)."
        )
        gateway_antenna = create_5g_sector_antenna_pattern(
            gain_max=GATEWAY_GAIN_MAX,
            horiz_beamwidth=GATEWAY_HORIZ_BW,
            vert_beamwidth=GATEWAY_VERT_BW,
            eta_rad=GATEWAY_ETA_RAD,
            valid_freqs=(1e9, 100e9),
        )
    else:
        gateway_antenna = load_starlink_gateway_antenna_from_csv(
            gateway_pattern_csv,
            eta_rad=1.0,
            valid_freqs=(1e9, 100e9),
        )
        print(f"  Starlink gateway antenna: {gateway_pattern_csv}")

    print(f"Loading {args.nc4}...")
    data = load_ssmis_nc4_and_build_arrays(args.nc4)
    n_obs = data["n_obs"]
    unique_ts = np.unique(data["timestamps"])
    print(f"  Observations: {n_obs:,}; unique timestamps: {len(unique_ts):,}")
    print(
        "  SSMI-S FOV ellipse: fixed 27×18 km (d_max/d_min in starlink_gateway_mdl); "
        "orientation from calculate_ssmis_fov_bearing_vectorized when n_FOV % 60 == 0, "
        "else North-aligned fallback."
    )

    country_df = load_country_5g_sensor_channel_csv()

    out_base_nc4 = os.path.splitext(os.path.basename(args.nc4))[0]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.nc4))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    top5_path = os.path.join(out_dir, f"{out_base_nc4}_{TOP5_STEM}_top5.txt")
    gateway_bbox_margin_deg = 0.0 if args.no_gateway_bbox else 1.0
    in_fov_first = not args.attenuation_first
    gw_random = not args.legacy_gateway_boresight_at_satellite
    gw_bore_pt = True
    gw_per_antenna = not args.legacy_co_pointed_gateways
    profile_rfi = args.profile_rfi or os.environ.get(
        "RSCSIM_PROFILE_RFI", ""
    ).strip().lower() in ("1", "true", "yes")
    print(
        "  Starlink gateway link order: "
        + (
            "in-FOV first (then ITU-R for gateways in a footprint)"
            if in_fov_first
            else "attenuation first (original)"
        )
    )
    print(
        "  Starlink gateway tx: "
        + (
            f"per-antenna boresight (N={N_ANTENNAS_PER_GATEWAY}, "
            f"B_ref={GATEWAY_EIRP_REFERENCE_BANDWIDTH_HZ/1e6:.0f} MHz)"
            if gw_per_antenna
            else f"legacy co-pointed (+10·log₁₀({N_ANTENNAS_PER_GATEWAY}))"
        )
    )
    print(
        "  Starlink gateway antenna: "
        + (
            "uniform random boresight (az [0,360) deg, el [25,90] deg) + off-boresight gain"
            if gw_random
            else "legacy bore sight toward satellite (relative gain 1)"
        )
    )

    use_metro_5g = not args.legacy_per_cell_5g
    any_harmonic_in_band = any(
        (cfg[1] - cfg[2] / 2.0)
        <= 2.0 * emitter_fundamental_hz_list[i]
        <= (cfg[1] + cfg[2] / 2.0)
        for i, cfg in enumerate(SSMIS_CHANNEL_CONFIGS)
    )
    if use_metro_5g:
        print("5G density: metro-contiguous (default)")
        if any_harmonic_in_band:
            metro_path = resolve_ghsl_metro_tif_path()
            if not metro_path.is_file():
                print(
                    f"ERROR: Metro GHSL GeoTIFF not found: {metro_path}\n"
                    "  Place GHS_POP_*_metro.tif in research_tutorials/data/\n"
                    "  (obtain from project maintainer) or set GHSL_METRO_TIF_PATH.\n"
                    "  Legacy per-cell: --legacy-per-cell-5g"
                )
                sys.exit(1)
    else:
        print("5G density: legacy per-cell (--legacy-per-cell-5g)")

    with open(top5_path, "w") as top5_file:
        top5_file.write("=" * 72 + "\n")
        top5_file.write("5G ground emitters (second harmonic in channel band)\n")
        top5_file.write("=" * 72 + "\n")
        print("\n" + "=" * 72)
        print("5G ground emitters (second harmonic in channel band)")
        print("=" * 72)

        for idx, (ch_num, center_freq_hz, bandwidth_hz) in enumerate(SSMIS_CHANNEL_CONFIGS):
            emitter_fundamental_hz = emitter_fundamental_hz_list[idx]
            harmonic_freq_hz = 2.0 * emitter_fundamental_hz
            freq_min = center_freq_hz - bandwidth_hz / 2.0
            freq_max = center_freq_hz + bandwidth_hz / 2.0
            harmonic_in_band = freq_min <= harmonic_freq_hz <= freq_max

            out_csv = os.path.join(
                out_dir, f"{out_base_nc4}_{RFI_PREFIX_5G}_RFI_ch{ch_num}.csv"
            )
            supported = supported_5g_countries_for_channel(
                country_df, "SSMI-S", ch_num
            )
            countries_line = (
                "Country: " + ", ".join(sorted(supported.values()))
                if supported
                else "Country: (none)"
            )
            first_line = (
                f"[5G] Channel {ch_num}: {center_freq_hz/1e9:.2f} GHz, BW={bandwidth_hz/1e6:.0f} MHz, "
                f"emitter fundamental={emitter_fundamental_hz/1e9:.2f} GHz "
                f"(2nd harmonic={harmonic_freq_hz/1e9:.2f} GHz)"
            )
            print(f"\n{first_line}")
            print(countries_line)

            t0_ch = time_module.perf_counter()
            if not harmonic_in_band:
                print(
                    f"  2nd harmonic out of channel band [{freq_min/1e9:.3f}, {freq_max/1e9:.3f}] GHz "
                    "→ zero RFI for all observations"
                )
                df = pd.DataFrame({
                    "timestamp": data["timestamps"],
                    "satellite": data["satellite"],
                    "lat": np.round(data["lat"], 6),
                    "lon": np.round(data["lon"], 6),
                    "saza": np.round(np.full(n_obs, SSMIS_SAZA_DEG, dtype=np.float64), 6),
                    "rfi_power_dBW": np.round(np.full(n_obs, -300.0), 3),
                    "rfi_brightness_temperature_K": ["0.000e+00"] * n_obs,
                })
                df.to_csv(out_csv, index=False)
                print(f"  Wrote {out_csv} ({len(df):,} rows)")
            else:
                t0_dens = time_module.perf_counter()
                if use_metro_5g:
                    density = get_emitter_density_metro_vectorized(
                        data["lat"],
                        data["lon"],
                        supported_5g_countries=supported,
                    )
                else:
                    density = get_emitter_density_vectorized(
                        data["lat"],
                        data["lon"],
                        supported_5g_countries=supported,
                    )
                print(
                    f"  Emitter density (vectorized) in "
                    f"{time_module.perf_counter() - t0_dens:.1f} s"
                )
                df = run_rfi_for_channel_5g(
                    data,
                    density,
                    ecef_by_satellite,
                    v_band_antenna,
                    emitter_antenna,
                    ch_num,
                    center_freq_hz,
                    bandwidth_hz,
                    emitter_fundamental_hz,
                    out_csv,
                )
            print(f"  [5G] Channel {ch_num} done in {time_module.perf_counter() - t0_ch:.1f} s")

            ch_header = first_line + "\n" + countries_line
            _append_top5_block(top5_file, ch_header, df)

        if use_metro_5g:
            clear_ghsl_metro_raster_cache()
        else:
            clear_ghsl_raster_cache()
        gc.collect()

        top5_file.write("\n" + "=" * 72 + "\n")
        top5_file.write(
            "Starlink ground gateways (direct RFI at gateway freq; OOBE A(f) vs sensor center)\n"
        )
        top5_file.write("=" * 72 + "\n")
        print("\n" + "=" * 72)
        print(
            "Starlink ground gateways (direct RFI at gateway freq; OOBE A(f) vs sensor center)"
        )
        print("=" * 72)

        for idx_ch, (ch_num, ssmis_center_freq_hz, bandwidth_hz) in enumerate(SSMIS_CHANNEL_CONFIGS):
            gateway_center_freq_hz = gateway_center_freq_hz_list[idx_ch]
            freq_min = ssmis_center_freq_hz - bandwidth_hz / 2.0
            freq_max = ssmis_center_freq_hz + bandwidth_hz / 2.0
            gateway_in_band = freq_min <= gateway_center_freq_hz <= freq_max

            out_csv = os.path.join(
                out_dir, f"{out_base_nc4}_{RFI_PREFIX_STARLINK}_RFI_ch{ch_num}.csv"
            )
            print(
                f"\n[Starlink] Channel {ch_num}: gateway {gateway_center_freq_hz/1e9:.2f} GHz, "
                f"sensor center {ssmis_center_freq_hz/1e9:.2f} GHz, "
                f"BW={bandwidth_hz/1e6:.0f} MHz (direct link + OOBE mask)"
            )

            t0_ch = time_module.perf_counter()
            if not gateway_in_band:
                print(
                    f"  Gateway center freq out of channel band [{freq_min/1e9:.3f}, {freq_max/1e9:.3f}] GHz "
                    "→ zero RFI for all observations"
                )
                df = pd.DataFrame({
                    "timestamp": data["timestamps"],
                    "satellite": data["satellite"],
                    "lat": np.round(data["lat"], 6),
                    "lon": np.round(data["lon"], 6),
                    "saza": np.round(np.full(n_obs, SSMIS_SAZA_DEG, dtype=np.float64), 6),
                    "rfi_power_dBW": np.round(np.full(n_obs, -300.0), 3),
                    "rfi_brightness_temperature_K": ["0.000e+00"] * n_obs,
                })
                df.to_csv(out_csv, index=False)
                print(f"  Wrote {out_csv} ({len(df):,} rows)")
                oobe_db = None
                gw_computed = False
            else:
                df = run_rfi_for_channel_starlink_gateway(
                    data,
                    gateway_lat,
                    gateway_lon,
                    ecef_by_satellite,
                    v_band_antenna,
                    gateway_antenna,
                    ch_num,
                    gateway_center_freq_hz,
                    ssmis_center_freq_hz,
                    bandwidth_hz,
                    out_csv,
                    sensor_name="SSMI-S",
                    chunk_size=args.chunk_size,
                    gateway_bbox_margin_deg=gateway_bbox_margin_deg,
                    in_fov_first=in_fov_first,
                    gateway_random_boresight=gw_random,
                    gateway_boresight_pointing=gw_bore_pt,
                    gateway_boresight_random_seed=args.gateway_boresight_random_seed,
                    gateway_per_antenna_boresight=gw_per_antenna,
                    eirp_reference_bandwidth_hz=GATEWAY_EIRP_REFERENCE_BANDWIDTH_HZ,
                    profile_rfi=profile_rfi,
                )
                oobe_db = starlink_gateway_uplink_oobe_attenuation_db(ssmis_center_freq_hz)
                gw_computed = True
            print(
                f"  [Starlink] Channel {ch_num} done in "
                f"{time_module.perf_counter() - t0_ch:.1f} s"
            )

            ch_header = (
                f"[Starlink] Channel {ch_num}: gateway {gateway_center_freq_hz/1e9:.2f} GHz, "
                f"sensor center {ssmis_center_freq_hz/1e9:.2f} GHz, "
                f"BW={bandwidth_hz/1e6:.0f} MHz (direct link + OOBE mask)"
            )
            _append_top5_block(
                top5_file,
                ch_header,
                df,
                oobe_atten_db=oobe_db,
                gateway_rfi_computed=gw_computed,
            )

    print(f"\nTop 5 summary written to {top5_path}")

    # Release large inputs before CSV combine / netCDF phase to lower peak RAM.
    del data
    del ecef_by_satellite
    del gateway_lat, gateway_lon
    del v_band_antenna, emitter_antenna, gateway_antenna
    del country_df
    gc.collect()

    combined_5g = combine_channel_csvs(
        out_dir, out_base_nc4, remove_channel_files=True, rfi_prefix=RFI_PREFIX_5G
    )
    if combined_5g is not None:
        print(f"Combined 5G channel CSVs to {combined_5g}.")

    combined_sl = combine_channel_csvs(
        out_dir, out_base_nc4, remove_channel_files=True, rfi_prefix=RFI_PREFIX_STARLINK
    )
    if combined_sl is not None:
        print(f"Combined Starlink Gateway channel CSVs to {combined_sl}.")

    sum_path = sum_two_rfi_combined_csvs_by_channel(
        out_dir,
        out_base_nc4,
        rfi_prefix_a=RFI_PREFIX_5G,
        rfi_prefix_b=RFI_PREFIX_STARLINK,
        output_rfi_prefix=COMBINED_SUM_PREFIX,
    )
    if sum_path is not None:
        print(f"Wrote summed Tb combined CSV to {sum_path}.")
        rfi_nc4 = Path(out_dir) / f"{out_base_nc4}_RFI.nc4"
        ch_nums = [cfg[0] for cfg in SSMIS_CHANNEL_CONFIGS]
        try:
            csv_5g = Path(out_dir) / f"{out_base_nc4}_{RFI_PREFIX_5G}_RFI_combined.csv"
            csv_sl = Path(out_dir) / f"{out_base_nc4}_{RFI_PREFIX_STARLINK}_RFI_combined.csv"

            df_sum_cr = pd.read_csv(sum_path)
            lat_cr = pd.to_numeric(df_sum_cr["lat"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            lon_cr = pd.to_numeric(df_sum_cr["lon"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            saza_cr = pd.to_numeric(df_sum_cr["saza"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            elevation_deg_cr = 90.0 - np.abs(saza_cr)
            data_dir_cr = os.path.join(_script_dir, "data")
            itu_nc_cr = itu_iclw_rain_info_nc_path(
                os.path.basename(sum_path), data_dir_cr
            )
            center_freqs_ghz_cr = (
                np.array([cfg[1] for cfg in SSMIS_CHANNEL_CONFIGS], dtype=np.float64)
                / 1e9
            )
            rng_cr = np.random.default_rng(None)
            atten_db_cr = compute_cloud_rain_atten_db_for_fovs(
                lat_cr,
                lon_cr,
                elevation_deg_cr,
                itu_nc_cr,
                center_freqs_ghz_cr,
                rng_cr,
                iclw_abs_threshold=CLOUD_RAIN_ICLW_ABS_THRESHOLD,
            )
            bad_geo = ~np.isfinite(lat_cr) | ~np.isfinite(lon_cr)
            if np.any(bad_geo):
                atten_db_cr = np.array(atten_db_cr, copy=True)
                atten_db_cr[bad_geo, :] = 0.0
            atten_by_ch = atten_db_to_by_channel_dict(ch_nums, atten_db_cr)
            del lat_cr, lon_cr, saza_cr, elevation_deg_cr, atten_db_cr
            del itu_nc_cr, center_freqs_ghz_cr, rng_cr, data_dir_cr
            gc.collect()

            top5_att_path = (
                Path(out_dir) / f"{out_base_nc4}_5G_Starlink_Gateway_Attenuation_top5.txt"
            )
            df_5g_cr = pd.read_csv(csv_5g)
            df_sl_cr = pd.read_csv(csv_sl)
            write_attenuated_combined_rfi_top5_file(
                top5_att_path,
                df_sum_cr,
                df_5g_cr,
                df_sl_cr,
                ch_nums,
                atten_by_ch,
            )

            copy_nc4_with_tmbr_plus_rfi(
                args.nc4,
                rfi_nc4,
                sum_path,
                tmbr_channel_numbers_with_rfi=ch_nums,
                n_tmbr_channels=TMBR_N_CHANNELS,
                combined_rfi_csv_5g=csv_5g,
                combined_rfi_csv_starlink=csv_sl,
                cloud_rain_atten_db_by_channel=atten_by_ch,
                combined_rfi_df=df_sum_cr,
                combined_rfi_df_5g=df_5g_cr,
                combined_rfi_df_starlink=df_sl_cr,
            )
            print(
                f"Wrote RFI-augmented netCDF (native TMBR unchanged; TMBR_RFI with cloud/rain-scaled summed RFI, "
                f"CELL_RFI, GATE_RFI, CLOUD_RAIN_ATT on ch 1–5) to {rfi_nc4}."
            )
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            print(f"WARNING: Could not write {rfi_nc4.name}: {e}, likely lack of system memory in the machine")
    else:
        print("WARNING: Could not build 5G+Starlink summed combined CSV (missing or mismatched inputs).")

    elapsed = time_module.perf_counter() - t_start
    print(f"\nOverall execution time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
