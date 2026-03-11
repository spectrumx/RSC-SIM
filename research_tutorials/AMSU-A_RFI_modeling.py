"""
RFI modeling for AMSU-A sensor for NWP data biasing.

Computes 5G ground-emitter RFI for all FOVs in AMSU-A netCDF-4 files. Each nc4 contains
data from multiple satellites (NOAA-15/18/19, METOP-B/C); SAID identifies the satellite
per observation. ECEF lookups are loaded per satellite from the sensor directory.
Runs a parametric study over AMSU-A channels 3–8.

Usage:
  python AMSU-A_RFI_modeling.py --sensor AMSU-A --nc4 path [--out_dir dir]
  e.g.:
  python AMSU-A_RFI_modeling.py --sensor AMSU-A --nc4 util/AMSU-A/amsua_2023080112.nc4 --out_dir util/AMSU-A
  Output: <out_dir>/<nc4_stem>_5G_RFI_chN.csv (timestamp, satellite, lat, lon, saza, rfi_dBW, rfi_Tb).

Author: Weather Satellite RFI / NWP Team
"""  # noqa: E501

import argparse
import os
import sys
import time as time_module

import numpy as np

# Add src to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(os.path.dirname(_script_dir), "src")
sys.path.insert(0, _src_dir)

from netCDF4 import Dataset  # noqa: E402

from weather_sat_mdl import (  # noqa: E402
    create_5g_sector_antenna_pattern,
    load_weather_sat_antenna_from_csv,
)
from weather_sat_nwp import (  # noqa: E402
    combine_channel_csvs,
    get_emitter_density_vectorized,
    iter_valid_ts_sat_indices,
    load_ecef_lookups_for_nc4,
    model_rfi_nwp_5g_single_time,
    said_to_satellite_array,
    SENSOR_ALLOWED_SAIDS,
    SENSOR_SAID_TO_SATELLITE,
    timestamp_from_nc4_vars,
)

# =============================================================================
# Configuration (override with CLI)
# =============================================================================

# directory that has data files (relative to this script)
DATA_DIR = os.path.join(_script_dir, "data")

# V-band (50–55.5 GHz), AMSU-A Ch 3–8: parametric study over channels
# Each entry: (channel_number, center_freq_Hz, bandwidth_Hz). Use Hz, not GHz
AMSUA_CHANNEL_CONFIGS = [
    (3, 50.3e9, 180e6),
    (4, 52.8e9, 400e6),
    (5, 53.6e9, 170e6),
    (6, 54.4e9, 400e6),
    (7, 54.94e9, 400e6),
    (8, 55.5e9, 330e6),
]
# Emitter fundamental (Hz) per channel; 2nd harmonic = 2 * fundamental at channel center.
# One element per AMSUA_CHANNEL_CONFIGS entry, same order. Use Hz, not GHz
emitter_fundamental_hz_list = [
    AMSUA_CHANNEL_CONFIGS[0][1] / 2.0,  # Ch 3
    AMSUA_CHANNEL_CONFIGS[1][1] / 2.0,  # Ch 4
    AMSUA_CHANNEL_CONFIGS[2][1] / 2.0,  # Ch 5
    AMSUA_CHANNEL_CONFIGS[3][1] / 2.0,  # Ch 6
    AMSUA_CHANNEL_CONFIGS[4][1] / 2.0,  # Ch 7
    AMSUA_CHANNEL_CONFIGS[5][1] / 2.0,  # Ch 8
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

# Atmospheric defaults for ITU-R P.676
TEMPERATURE_K = 288.15
PRESSURE_PA = 101325.0
HUMIDITY_PCT = 50.0

# nc4 variable names (Satellite_Info_DK1 AMSU-A CSV.csv)
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
SENSOR_NAME = "AMSU-A"


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


def load_amsua_nc4_and_build_arrays(nc4_path: str):
    """
    Load AMSU-A nc4 and return flat arrays: lat, lon, saza, bearaz, altitude_m,
    timestamps_str, satellite (per obs), n_obs. SAID mapped via SENSOR_SAID_TO_SATELLITE.
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

    timestamps = timestamp_from_nc4_vars(year, month, days, hour, minu, seco)
    altitude_m = float(np.nanmean(alt))
    if said is not None:
        satellite = said_to_satellite_array(said, SENSOR_NAME)
    else:
        satellite = np.array([""] * n_obs, dtype=object)
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
    }


def run_rfi_for_channel(
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
    sensor_name: str = "AMSU-A",
):
    """Run RFI model for one AMSU-A channel; write CSV with timestamp, satellite, lat, lon, saza, rfi_dBW, rfi_Tb."""
    lat = data["lat"]
    lon = data["lon"]
    saza = data["saza"]
    timestamps = data["timestamps"]
    satellite = data["satellite"]
    altitude_m = data["altitude_m"]
    n_obs = data["n_obs"]
    allowed = set(SENSOR_SAID_TO_SATELLITE[sensor_name][said] for said in SENSOR_ALLOWED_SAIDS[sensor_name])

    rfi_dBW = np.full(n_obs, -300.0)
    rfi_K = np.full(n_obs, 0.0)

    for ts, sat, idx, coords in iter_valid_ts_sat_indices(
        timestamps, satellite, allowed, ecef_by_satellite
    ):
        sat_ecef_km = np.array(coords, dtype=np.float64)
        rfi_db, rfi_tb = model_rfi_nwp_5g_single_time(
            sat_ecef_km,
            lat[idx],
            lon[idx],
            saza[idx],
            altitude_m,
            density[idx],
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
        rfi_dBW[idx] = rfi_db
        rfi_K[idx] = rfi_tb

    import pandas as pd
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


def main():
    t_start = time_module.perf_counter()
    parser = argparse.ArgumentParser(
        description="AMSU-A RFI modeling for NWP (5G ground emitters only; multi-satellite per nc4)."
    )
    parser.add_argument(
        "--sensor",
        required=True,
        metavar="SENSOR",
        help="Sensor name (e.g. AMSU-A); used for SAID mapping and data directory.",
    )
    parser.add_argument(
        "--nc4",
        required=True,
        help="Path to AMSU-A netCDF-4 file (e.g. util/AMSU-A/amsua_2023080112.nc4). ECEF lookups from same dir.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for RFI CSVs (default: same directory as nc4 file).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.nc4):
        print(f"ERROR: nc4 file not found: {args.nc4}")
        sys.exit(1)
    if args.sensor != "AMSU-A":
        print(f"ERROR: This script is for AMSU-A sensor; got --sensor {args.sensor!r}")
        sys.exit(1)

    ecef_by_satellite = load_ecef_lookups_for_nc4(args.nc4)
    if not ecef_by_satellite:
        print(f"ERROR: No ECEF lookup CSVs found for {args.nc4} (expect *_ECEF_lookup_<stem>.csv in same dir)")
        sys.exit(1)
    print(f"  Loaded ECEF lookups for satellites: {list(ecef_by_satellite.keys())}")

    data_dir = os.path.join(_script_dir, "data")
    v_band_csv = os.path.join(data_dir, "AMSU-A V-Band 50.3 GHz absolute antenna pattern.csv")
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

    # Load nc4 and compute emitter density once (shared across all channels)
    print(f"Loading {args.nc4}...")
    data = load_amsua_nc4_and_build_arrays(args.nc4)
    n_obs = data["n_obs"]
    unique_ts = np.unique(data["timestamps"])
    print(f"  Observations: {n_obs:,}; unique timestamps: {len(unique_ts):,}")
    print("Computing emitter density (vectorized)...")
    t0 = time_module.perf_counter()
    density = get_emitter_density_vectorized(data["lat"], data["lon"])
    print(f"  Done in {time_module.perf_counter() - t0:.1f} s")

    out_base_nc4 = os.path.splitext(os.path.basename(args.nc4))[0]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.nc4))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    top5_path = os.path.join(out_dir, f"{out_base_nc4}_top5.txt")
    import pandas as pd
    with open(top5_path, "w") as top5_file:
        for idx, (ch_num, center_freq_hz, bandwidth_hz) in enumerate(AMSUA_CHANNEL_CONFIGS):
            emitter_fundamental_hz = emitter_fundamental_hz_list[idx]
            harmonic_freq_hz = 2.0 * emitter_fundamental_hz
            freq_min = center_freq_hz - bandwidth_hz / 2.0
            freq_max = center_freq_hz + bandwidth_hz / 2.0
            harmonic_in_band = freq_min <= harmonic_freq_hz <= freq_max

            out_csv = os.path.join(out_dir, f"{out_base_nc4}_5G_RFI_ch{ch_num}.csv")
            print(f"\nChannel {ch_num}: {center_freq_hz/1e9:.2f} GHz, BW={bandwidth_hz/1e6:.0f} MHz, "
                  f"emitter fundamental={emitter_fundamental_hz/1e9:.2f} GHz (2nd harmonic={harmonic_freq_hz/1e9:.2f} GHz)")  # noqa: E501

            if not harmonic_in_band:
                print(f"  2nd harmonic out of channel band [{freq_min/1e9:.3f}, {freq_max/1e9:.3f}] GHz → zero RFI for all observations")  # noqa: E501
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
                df = run_rfi_for_channel(
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
                    sensor_name="AMSU-A",
                )

            rfi_K_float = df["rfi_brightness_temperature_K"].astype(float)
            idx_top5 = rfi_K_float.abs().nlargest(5).index
            ch_header = (f"Channel {ch_num}: {center_freq_hz/1e9:.2f} GHz, BW={bandwidth_hz/1e6:.0f} MHz, "
                         f"emitter fundamental={emitter_fundamental_hz/1e9:.2f} GHz (2nd harmonic={harmonic_freq_hz/1e9:.2f} GHz)")  # noqa: E501
            top5_file.write(f"\n{ch_header}\n")
            print("  Top 5 by |rfi_brightness_temperature_K|:")
            top5_file.write("  Top 5 by |rfi_brightness_temperature_K|:\n")
            for rank, idx in enumerate(idx_top5, start=1):
                row = df.loc[idx]
                line = (f"    [{rank}] satellite: {row['satellite']}, lat: {row['lat']}, lon: {row['lon']}, "
                        f"saza: {row['saza']}, rfi_power_dBW: {row['rfi_power_dBW']}, "
                        f"rfi_Tb_K: {row['rfi_brightness_temperature_K']}")
                print(line)
                top5_file.write(line + "\n")

    print(f"\nTop 5 summary written to {top5_path}")

    # combine channel CSVs: set remove_channel_files=False to keep the channel CSVs
    combined_path = combine_channel_csvs(out_dir, out_base_nc4, remove_channel_files=True)
    if combined_path is not None:
        print(f"Combined channel CSVs to {combined_path}.")

    elapsed = time_module.perf_counter() - t_start
    print(f"\nOverall execution time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
