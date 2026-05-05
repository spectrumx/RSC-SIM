"""
RFI modeling for ATMS for NWP data biasing (5G + Starlink ground gateways).

Computes both (1) 5G ground-emitter RFI via second harmonic in band and
(2) Starlink ground-gateway RFI (direct RFI at channel center) for all FOVs in
ATMS netCDF-4 files. Each nc4 may contain multiple satellites (SUOMI-NPP, JPSS-1);
SAID identifies the satellite per observation. ECEF lookups are loaded per satellite
from the sensor directory. Parametric study over ATMS channels 3–9.

Execution order: all 5G channels first, then all Starlink gateway channels (same
pattern as running the two original scripts in sequence). Outputs per-source
per-channel CSVs and combined CSVs (``*_5G_RFI_combined.csv``,
``*_Starlink_Gateway_RFI_combined.csv``), a unified top-5 text file
(``*_5G_Starlink_Gateway_top5.txt``), and a summed combined CSV where each
``channelN_rfi_brightness_temperature_K`` is the linear sum of 5G and gateway Tb
(``*_5G_Starlink_Gateway_RFI_combined.csv``).

Full ITU-R P.676 atmospheric absorption; no polarization loss, terrain masking, or
OOBE for 5G. Gateway modeling uses the same atmospheric path as the standalone
gateway script. Cloud/rain slant attenuation (P.840 / P.838) scales summed RFI into ``TMBR_RFI``
and is stored as ``CLOUD_RAIN_ATT`` (dB) in ``*_RFI.nc4`` (see ``attenuation_mdl``).

Usage:
  python ATMS_RFI_modeling.py --sensor ATMS --nc4 path [--out_dir dir] [--gateways_csv path]

  python ATMS_RFI_modeling.py --sensor ATMS --nc4 util/ATMS/atms.2023080112.nc4 --out_dir util/ATMS

Outputs (in out_dir, with nc4 stem as prefix):
  Per-channel: *_5G_RFI_chN.csv, *_Starlink_Gateway_RFI_chN.csv
  Combined: *_5G_RFI_combined.csv, *_Starlink_Gateway_RFI_combined.csv,
            *_5G_Starlink_Gateway_RFI_combined.csv (sum of Tb columns)
  Top 5: *_5G_Starlink_Gateway_top5.txt, *_5G_Starlink_Gateway_Attenuation_top5.txt
  netCDF: <stem>_RFI.nc4 — native ``TMBR`` unchanged; ``TMBR_RFI`` = ``TMBR`` + summed RFI Tb on ch 3–9 (after cloud/rain
    path factor); ``CELL_RFI`` / ``GATE_RFI`` hold 5G-only and gateway-only Tb (K) on a compact channel axis (ch 3–9 only);
    ``CLOUD_RAIN_ATT`` slant attenuation (dB) on the same axis;
    same axis order as ``TMBR``; channel dim ``nchans_rfi`` (name sorts before ``obsNumber`` for
    Panoply like ``nchans``); coordinate ``channel_index_rfi`` gives instrument channel per index.
"""  # noqa: E501

import argparse
import gc
import os
import sys
import time as time_module
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
    load_starlink_gateways,
    load_starlink_gateway_antenna_from_csv,
    model_rfi_nwp_starlink_gateway_single_time,
    model_rfi_nwp_starlink_gateway_single_time_in_fov_first,
)
from weather_sat_mdl import (  # noqa: E402
    create_5g_sector_antenna_pattern,
    load_weather_sat_antenna_from_csv,
)
from weather_sat_nwp import (  # noqa: E402
    clear_ghsl_raster_cache,
    combine_channel_csvs,
    copy_nc4_with_tmbr_plus_rfi,
    get_emitter_density_vectorized,
    iter_valid_ts_sat_indices,
    load_country_5g_sensor_channel_csv,
    load_ecef_lookups_for_nc4,
    model_rfi_nwp_5g_single_time,
    obs_valid_cross_track,
    replace_missing_with_nan,
    said_to_satellite_array,
    scalar_altitude_m_from_hmsl,
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

# V-band (50–55.5 GHz), ATMS Ch 3–9
ATMS_CHANNEL_CONFIGS = [
    (3, 50.3e9, 180e6),
    (4, 51.76e9, 400e6),
    (5, 52.8e9, 400e6),
    (6, 53.6e9, 170e6),
    (7, 54.4e9, 400e6),
    (8, 54.94e9, 400e6),
    (9, 55.5e9, 330e6),
]
# 5G Emitter fundamental (Hz) per channel; 2nd harmonic = 2 * fundamental at channel center.
# One element per ATMS_CHANNEL_CONFIGS entry, same order. Use Hz, not GHz
emitter_fundamental_hz_list = [
    ATMS_CHANNEL_CONFIGS[0][1] / 2.0,  # Ch 3
    ATMS_CHANNEL_CONFIGS[1][1] / 2.0,  # Ch 4
    ATMS_CHANNEL_CONFIGS[2][1] / 2.0,  # Ch 5
    ATMS_CHANNEL_CONFIGS[3][1] / 2.0,  # Ch 6
    ATMS_CHANNEL_CONFIGS[4][1] / 2.0,  # Ch 7
    ATMS_CHANNEL_CONFIGS[5][1] / 2.0,  # Ch 8
    ATMS_CHANNEL_CONFIGS[6][1] / 2.0,  # Ch 9
]

# Starlink Gateway center frequency (Hz) per channel; one element per ATMS_CHANNEL_CONFIGS entry, same order.
# Direct RFI at channel center: default = channel center frequency. Use Hz, not GHz.
gateway_center_freq_hz_list = [
    ATMS_CHANNEL_CONFIGS[0][1] / 1,  # Ch 3
    ATMS_CHANNEL_CONFIGS[1][1] / 1,  # Ch 4
    ATMS_CHANNEL_CONFIGS[2][1] / 1,  # Ch 5
    ATMS_CHANNEL_CONFIGS[3][1] / 1,  # Ch 6
    ATMS_CHANNEL_CONFIGS[4][1] / 1,  # Ch 7
    ATMS_CHANNEL_CONFIGS[5][1] / 1,  # Ch 8
    ATMS_CHANNEL_CONFIGS[6][1] / 1,  # Ch 9
]

# 5G sector antenna
# ITU-R M.2101 and 3GPP standards recommend 24 - 25 dBi for 5G sector antenna with standard 8x8 phased array
GROUND_EMITTER_GAIN_MAX = 24.5  # dBi
GROUND_EMITTER_HORIZ_BW = 65.0
GROUND_EMITTER_VERT_BW = 10.0
GROUND_EMITTER_ETA_RAD = 0.8

# International/FCC regulation of transmit power= -33 dBW
TRANSMIT_POWER_DBW = -33
# EIRP per emitter (dBW): EIRP at bore sight direction
# EIRP (bore sight) = TRANSMIT_POWER_DBW + GROUND_EMITTER_GAIN_MAX = -8.5 dBW
EIRP_PER_EMITTER_DBW = TRANSMIT_POWER_DBW + GROUND_EMITTER_GAIN_MAX

# Starlink ground gateway
EIRP_PER_GATEWAY_DBW = 70.5
N_ANTENNAS_PER_GATEWAY = 40
# Starlink gateway: default is uniform random boresight (see starlink_gateway_mdl); legacy = bore sight at sat
GATEWAY_BORESIGHT_POINTING = True  # if random boresight disabled via CLI
GATEWAY_GAIN_MAX = 24.5
GATEWAY_HORIZ_BW = 65.0
GATEWAY_VERT_BW = 10.0
GATEWAY_ETA_RAD = 0.8

# Atmospheric defaults for ITU-R P.676
TEMPERATURE_K = 288.15
PRESSURE_PA = 101325.0
HUMIDITY_PCT = 50.0

# netCDF-4 (.nc4) variable names
VAR_LAT = "LAT"
VAR_LON = "LON"
VAR_SAZA = "SAZA"
VAR_BEARAZ = "BEARAZ"
VAR_ALT = "HMSL"
VAR_YEAR = "YEAR"
VAR_MONTH = "MONTH"
VAR_DAYS = "DAYS"
VAR_HOUR = "HOUR"
VAR_MINU = "MINUTE"
VAR_SECO = "SECOND"
VAR_SAID = "SAID"
SENSOR_NAME = "ATMS"

RFI_PREFIX_5G = "5G"
RFI_PREFIX_STARLINK = "Starlink_Gateway"
TOP5_STEM = "5G_Starlink_Gateway"
COMBINED_SUM_PREFIX = "5G_Starlink_Gateway"

# TMBR channel dimension size in colleague nc4 (ch 3–9 → 0-based indices 2–8)
TMBR_N_CHANNELS = 22

# ICLW (kg/m^2) below this → cloud attenuation treated as zero in P.840 (script-local threshold).
CLOUD_RAIN_ICLW_ABS_THRESHOLD = 0.05


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


def load_atms_nc4_and_build_arrays(nc4_path: str):
    """
    Load ATMS nc4 and return flat arrays: lat, lon, saza, bearaz, altitude_m,
    year, month, day, hour, minute, second, timestamps_str, satellite (per obs), n_obs.
    SAID is read and mapped to satellite name via SENSOR_SAID_TO_SATELLITE["ATMS"].
    """
    with Dataset(nc4_path, "r") as ds:
        lat = _read_nc4_var(ds, VAR_LAT)
        lon = _read_nc4_var(ds, VAR_LON)
        saza = _read_nc4_var(ds, VAR_SAZA)
        bearaz = _read_nc4_var(ds, VAR_BEARAZ)
        alt = _read_nc4_var(ds, VAR_ALT, "HMSL")
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
        saza = saza.ravel()
        bearaz = bearaz.ravel()
        alt = alt.ravel() if alt.ndim == 2 else _flatten_or_repeat_time(alt, n_fov)
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
        saza = np.atleast_1d(saza).ravel()
        bearaz = np.atleast_1d(bearaz).ravel()
        alt = np.atleast_1d(alt).ravel()
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
    saza = replace_missing_with_nan(saza)
    bearaz = replace_missing_with_nan(bearaz)
    year = replace_missing_with_nan(year)
    month = replace_missing_with_nan(month)
    days = replace_missing_with_nan(days)
    hour = replace_missing_with_nan(hour)
    minu = replace_missing_with_nan(minu)
    seco = replace_missing_with_nan(seco)
    altitude_m = scalar_altitude_m_from_hmsl(alt)
    if said is not None:
        said_f = replace_missing_with_nan(np.asarray(said, dtype=np.float64))
        said_int = np.where(np.isfinite(said_f), said_f, -1.0).astype(np.int64)
        satellite = said_to_satellite_array(said_int, SENSOR_NAME)
    else:
        satellite = np.array([""] * n_obs, dtype=object)
    timestamps = timestamp_from_nc4_vars(year, month, days, hour, minu, seco)
    obs_valid = obs_valid_cross_track(
        lat,
        lon,
        saza,
        bearaz,
        year,
        month,
        days,
        hour,
        minu,
        seco,
        satellite,
    )
    return {
        "lat": lat,
        "lon": lon,
        "saza": saza,
        "bearaz": bearaz,
        "altitude_m": altitude_m,
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
    sensor_name: str = "ATMS",
):
    """Run 5G RFI model for one ATMS channel; write CSV."""
    lat = data["lat"]
    lon = data["lon"]
    saza = data["saza"]
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

    for ts, sat, idx, coords in iter_valid_ts_sat_indices(
        timestamps, satellite, allowed, ecef_by_satellite
    ):
        idx_good = idx[obs_valid[idx]]
        if idx_good.size == 0:
            continue
        sat_ecef_km = np.array(coords, dtype=np.float64)
        rfi_db, rfi_tb = model_rfi_nwp_5g_single_time(
            sat_ecef_km,
            lat[idx_good],
            lon[idx_good],
            saza[idx_good],
            altitude_m,
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
            sensor_name=sensor_name,
        )
        rfi_dBW[idx_good] = rfi_db
        rfi_K[idx_good] = rfi_tb

    df = pd.DataFrame({
        "timestamp": timestamps,
        "satellite": satellite,
        "lat": np.round(lat, 6),
        "lon": np.round(lon, 6),
        "saza": np.round(saza, 6),
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
    bandwidth_hz: float,
    out_csv: str,
    sensor_name: str = "ATMS",
    chunk_size: int = 200000,
    gateway_bbox_margin_deg: float = 1.0,
    in_fov_first: bool = True,
    gateway_random_boresight: bool = True,
    gateway_boresight_pointing: bool = True,
    gateway_boresight_random_seed: Optional[int] = None,
):
    """Run Starlink gateway RFI for one ATMS channel; write CSV (same columns as 5G)."""
    lat = data["lat"]
    lon = data["lon"]
    saza = data["saza"]
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
    for ts, sat, idx, coords in iter_valid_ts_sat_indices(
        timestamps, satellite, allowed, ecef_by_satellite
    ):
        idx_good = idx[obs_valid[idx]]
        if idx_good.size == 0:
            continue
        sat_ecef_km = np.array(coords, dtype=np.float64)
        rfi_db, rfi_tb = _model(
            sat_ecef_km,
            lat[idx_good],
            lon[idx_good],
            saza[idx_good],
            altitude_m,
            gateway_lat,
            gateway_lon,
            v_band_antenna,
            gateway_antenna,
            freq_hz=center_freq_hz,
            bandwidth_hz=bandwidth_hz,
            eirp_per_gateway_dbw=EIRP_PER_GATEWAY_DBW,
            n_antennas_per_gateway=N_ANTENNAS_PER_GATEWAY,
            gateway_random_boresight=gateway_random_boresight,
            gateway_boresight_pointing=gateway_boresight_pointing,
            gateway_boresight_random_seed=gateway_boresight_random_seed,
            temperature=TEMPERATURE_K,
            pressure=PRESSURE_PA,
            humidity=HUMIDITY_PCT,
            sensor_name=sensor_name,
            chunk_size=chunk_size,
            gateway_bbox_margin_deg=gateway_bbox_margin_deg,
        )
        rfi_dBW[idx_good] = rfi_db
        rfi_K[idx_good] = rfi_tb

    df = pd.DataFrame({
        "timestamp": timestamps,
        "satellite": satellite,
        "lat": np.round(lat, 6),
        "lon": np.round(lon, 6),
        "saza": np.round(saza, 6),
        "rfi_power_dBW": np.round(rfi_dBW, 3),
        "rfi_brightness_temperature_K": [f"{x:.3e}" for x in rfi_K],
    })
    df.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv} ({len(df):,} rows)")
    return df


def _append_top5_block(top5_file, ch_header: str, df: pd.DataFrame):
    """Write one channel's top-5 block to file and print."""
    rfi_K_float = df["rfi_brightness_temperature_K"].astype(float)
    idx_top5 = rfi_K_float.abs().nlargest(5).index
    top5_file.write(f"\n{ch_header}\n")
    print("  Top 5 by |rfi_brightness_temperature_K|:")
    top5_file.write("  Top 5 by |rfi_brightness_temperature_K|:\n")
    for rank, row_idx in enumerate(idx_top5, start=1):
        row = df.loc[row_idx]
        line = (
            f"    [{rank}] satellite: {row['satellite']}, lat: {row['lat']}, lon: {row['lon']}, "
            f"saza: {row['saza']}, rfi_power_dBW: {row['rfi_power_dBW']}, "
            f"rfi_Tb_K: {row['rfi_brightness_temperature_K']}"
        )
        print(line)
        top5_file.write(line + "\n")


def main():
    t_start = time_module.perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "ATMS RFI for NWP: 5G ground emitters (harmonic) and Starlink ground gateways "
            "(direct); multi-satellite per nc4."
        )
    )
    parser.add_argument(
        "--sensor",
        required=True,
        metavar="SENSOR",
        help="Sensor name (e.g. ATMS); used for data directory and SAID mapping.",
    )
    parser.add_argument(
        "--nc4",
        required=True,
        help="Path to ATMS netCDF-4 file (e.g. util/ATMS/atms.2023080112.nc4).",
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
        help=(
            "Gateway: use ITU-R / link budget before in-FOV mask. "
            "Default is in-FOV-first."
        ),
    )
    parser.add_argument(
        "--legacy_gateway_boresight_at_satellite",
        action="store_true",
        help=(
            "Gateway: disable random antenna orientation; assume bore sight at the "
            "weather satellite (relative gain 1). Default is uniform random azimuth "
            "[0,360) deg and elevation [25,90] deg with sector off-boresight gain."
        ),
    )
    parser.add_argument(
        "--gateway_boresight_random_seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Optional RNG seed for gateway random boresight (per timestep batch).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.nc4):
        print(f"ERROR: nc4 file not found: {args.nc4}")
        sys.exit(1)
    if args.sensor != "ATMS":
        print(f"ERROR: This script is for ATMS sensor; got --sensor {args.sensor!r}")
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
    # DKDK
    v_band_csv = os.path.join(data_dir, "V-Band 50.3 GHz absolute antenna pattern.csv")
    if not os.path.exists(v_band_csv):
        print(f"WARNING: V-Band antenna CSV not found: {v_band_csv}")
        print("  Using ITU fallback (see tuto_radiomdl_weather_phase3.py).")
        from radio_types import Antenna  # noqa: E402
        from astro_mdl import antenna_mdl_ITU  # noqa: E402
        alphas = np.arange(0, 181, 1)
        betas = np.arange(0, 360, 1)
        gain_max = 50.0
        half_beamwidth = 0.5
        v_band_ant_df = antenna_mdl_ITU(gain_max, half_beamwidth, alphas, betas)
        v_band_antenna = Antenna.from_dataframe(
            v_band_ant_df, 0.99, (40e9, 60e9)
        )
    else:
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
    data = load_atms_nc4_and_build_arrays(args.nc4)
    n_obs = data["n_obs"]
    unique_ts = np.unique(data["timestamps"])
    print(f"  Observations: {n_obs:,}; unique timestamps: {len(unique_ts):,}")

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
    print(
        "  Starlink gateway link order: "
        + (
            "in-FOV first (then ITU-R for gateways in a footprint)"
            if in_fov_first
            else "attenuation first (original)"
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

    with open(top5_path, "w") as top5_file:
        top5_file.write("=" * 72 + "\n")
        top5_file.write("5G ground emitters (second harmonic in channel band)\n")
        top5_file.write("=" * 72 + "\n")
        print("\n" + "=" * 72)
        print("5G ground emitters (second harmonic in channel band)")
        print("=" * 72)

        for idx, (ch_num, center_freq_hz, bandwidth_hz) in enumerate(ATMS_CHANNEL_CONFIGS):
            emitter_fundamental_hz = emitter_fundamental_hz_list[idx]
            harmonic_freq_hz = 2.0 * emitter_fundamental_hz
            freq_min = center_freq_hz - bandwidth_hz / 2.0
            freq_max = center_freq_hz + bandwidth_hz / 2.0
            harmonic_in_band = freq_min <= harmonic_freq_hz <= freq_max

            out_csv = os.path.join(
                out_dir, f"{out_base_nc4}_{RFI_PREFIX_5G}_RFI_ch{ch_num}.csv"
            )
            supported = supported_5g_countries_for_channel(
                country_df, "ATMS", ch_num
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
                    "saza": np.round(data["saza"], 6),
                    "rfi_power_dBW": np.round(np.full(n_obs, -300.0), 3),
                    "rfi_brightness_temperature_K": ["0.000e+00"] * n_obs,
                })
                df.to_csv(out_csv, index=False)
                print(f"  Wrote {out_csv} ({len(df):,} rows)")
            else:
                t0_dens = time_module.perf_counter()
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
                    sensor_name="ATMS",
                )
            print(f"  [5G] Channel {ch_num} done in {time_module.perf_counter() - t0_ch:.1f} s")

            ch_header = first_line + "\n" + countries_line
            _append_top5_block(top5_file, ch_header, df)

        clear_ghsl_raster_cache()
        gc.collect()

        top5_file.write("\n" + "=" * 72 + "\n")
        top5_file.write("Starlink ground gateways (direct RFI at channel center)\n")
        top5_file.write("=" * 72 + "\n")
        print("\n" + "=" * 72)
        print("Starlink ground gateways (direct RFI at channel center)")
        print("=" * 72)

        for idx, (ch_num, atms_center_freq_hz, bandwidth_hz) in enumerate(ATMS_CHANNEL_CONFIGS):
            gateway_center_freq_hz = gateway_center_freq_hz_list[idx]
            freq_min = atms_center_freq_hz - bandwidth_hz / 2.0
            freq_max = atms_center_freq_hz + bandwidth_hz / 2.0
            gateway_in_band = freq_min <= gateway_center_freq_hz <= freq_max

            out_csv = os.path.join(
                out_dir, f"{out_base_nc4}_{RFI_PREFIX_STARLINK}_RFI_ch{ch_num}.csv"
            )
            print(
                f"\n[Starlink] Channel {ch_num}: {gateway_center_freq_hz/1e9:.2f} GHz, "
                f"BW={bandwidth_hz/1e6:.0f} MHz (direct RFI at gateway center freq)"
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
                    "saza": np.round(data["saza"], 6),
                    "rfi_power_dBW": np.round(np.full(n_obs, -300.0), 3),
                    "rfi_brightness_temperature_K": ["0.000e+00"] * n_obs,
                })
                df.to_csv(out_csv, index=False)
                print(f"  Wrote {out_csv} ({len(df):,} rows)")
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
                    bandwidth_hz,
                    out_csv,
                    sensor_name="ATMS",
                    chunk_size=args.chunk_size,
                    gateway_bbox_margin_deg=gateway_bbox_margin_deg,
                    in_fov_first=in_fov_first,
                    gateway_random_boresight=gw_random,
                    gateway_boresight_pointing=gw_bore_pt,
                    gateway_boresight_random_seed=args.gateway_boresight_random_seed,
                )
            print(
                f"  [Starlink] Channel {ch_num} done in "
                f"{time_module.perf_counter() - t0_ch:.1f} s"
            )

            ch_header = (
                f"[Starlink] Channel {ch_num}: {gateway_center_freq_hz/1e9:.2f} GHz, "
                f"BW={bandwidth_hz/1e6:.0f} MHz (direct RFI)"
            )
            _append_top5_block(top5_file, ch_header, df)

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
        ch_nums = [cfg[0] for cfg in ATMS_CHANNEL_CONFIGS]
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
                np.array([cfg[1] for cfg in ATMS_CHANNEL_CONFIGS], dtype=np.float64)
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
                f"CELL_RFI, GATE_RFI, CLOUD_RAIN_ATT on ch 3–9) to {rfi_nc4}."
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
