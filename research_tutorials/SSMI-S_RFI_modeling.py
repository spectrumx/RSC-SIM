"""
RFI modeling for SSMI-S/DMSP-F17 for NWP data biasing.

Computes 5G ground-emitter RFI (no Starlink) for all FOVs in SSMI-S netCDF-4 files,
using ECEF lookup for satellite position per timestamp. Runs a parametric study
over SSMI-S channels 1–5: one output CSV per channel, with emitter fundamental set
so the 2nd harmonic falls at each channel center frequency.

Outputs RFI in dBW and brightness temperature (K). Full ITU-R P.676 atmospheric
absorption; no polarization loss, terrain masking, or OOBE.

Usage:
  python SSMI-S_RFI_modeling.py --sat SATELLITE --nc4 path --ecef path --out_dir dir
  e.g.:
  python SSMI-S_RFI_modeling.py --sat DMSP-F17 --nc4 util/DMSP-F17/ssmis_2023080112.nc4 --ecef util/DMSP-F17/DMSP-F17_ECEF_lookup_ssmis_2023080112.csv --out_dir util/DMSP-F17
  Output: <out_dir>/<satellite>_<nc4_basename>_5G_RFI_chN.csv (no SAZA column).

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
    load_ecef_lookup,
    model_rfi_nwp_5g_single_time_ssmis,
    timestamp_from_nc4_vars,
)

# =============================================================================
# Configuration (override with CLI)
# =============================================================================

# directory that has data files (relative to this script)
DATA_DIR = os.path.join(_script_dir, "data")

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

# nc4 variable names (SSMI-S: LAT, LON, time only; no SAZA, BEARAZ, HMSL)
VAR_LAT = "LAT"
VAR_LON = "LON"
VAR_YEAR = "YEAR"
VAR_MONTH = "MONTH"
VAR_DAYS = "DAYS"
VAR_HOUR = "HOUR"
VAR_MINU = "MINUTE"
VAR_SECO = "SECOND"

# SSMI-S geometry (DMSP-F17): altitude 850 km, slant range 1020 km, elevation 36.9 deg
SSMIS_ALTITUDE_M = 850_000.0
SSMIS_SLANT_RANGE_KM = 1020.0
SSMIS_ELEVATION_DEG = 36.9
SSMIS_FOVS_PER_SCAN = 60


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
    Load SSMI-S nc4 (LAT, LON, time only; no SAZA, BEARAZ, HMSL).
    Return flat arrays: lat, lon, altitude_m (constant), timestamps, n_obs, time vars.
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

    timestamps = timestamp_from_nc4_vars(year, month, days, hour, minu, seco)
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
        "n_obs": n_obs,
    }


def run_rfi_for_channel_ssmis(
    data: dict,
    density: np.ndarray,
    ecef_lookup: dict,
    v_band_antenna,
    emitter_antenna,
    channel_num: int,
    center_freq_hz: float,
    bandwidth_hz: float,
    emitter_fundamental_hz: float,
    out_csv: str,
):
    """Run RFI model for one SSMI-S channel; write CSV (no SAZA)."""
    lat = data["lat"]
    lon = data["lon"]
    timestamps = data["timestamps"]
    n_obs = data["n_obs"]
    unique_ts = np.unique(timestamps)

    rfi_dBW = np.full(n_obs, np.nan)
    rfi_K = np.full(n_obs, np.nan)

    for ts in unique_ts:
        coords = ecef_lookup.get(ts)
        if coords is None:
            continue
        sat_ecef_km = np.array(coords, dtype=np.float64)
        mask = timestamps == ts
        rfi_db, rfi_tb = model_rfi_nwp_5g_single_time_ssmis(
            sat_ecef_km,
            lat[mask],
            lon[mask],
            density[mask],
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
        rfi_dBW[mask] = rfi_db
        rfi_K[mask] = rfi_tb

    import pandas as pd
    df = pd.DataFrame({
        "timestamp": timestamps,
        "lat": np.round(lat, 6),
        "lon": np.round(lon, 6),
        "rfi_power_dBW": np.round(rfi_dBW, 3),
        "rfi_brightness_temperature_K": [f"{x:.3e}" for x in rfi_K],
    })
    df.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv} ({len(df):,} rows)")
    return df


def main():
    t_start = time_module.perf_counter()
    parser = argparse.ArgumentParser(
        description="SSMI-S/DMSP-F17 RFI modeling for NWP (5G ground emitters only)."
    )
    parser.add_argument(
        "--sat",
        required=True,
        metavar="SATELLITE",
        help="Satellite name for output filenames (e.g. DMSP-F17).",
    )
    parser.add_argument(
        "--nc4",
        required=True,
        help="Path to SSMI-S netCDF-4 file (e.g. data/ssmis_2023080112.nc4).",
    )
    parser.add_argument(
        "--ecef",
        required=True,
        help="Path to DMSP-F17 ECEF lookup CSV (e.g., data/DMSP-F17_ECEF_lookup.csv).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output CSV directory path for RFI (dBW and K).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.nc4):
        print(f"ERROR: nc4 file not found: {args.nc4}")
        sys.exit(1)
    if not os.path.isfile(args.ecef):
        print(f"ERROR: ECEF lookup not found: {args.ecef}")
        sys.exit(1)

    ecef_lookup = load_ecef_lookup(args.ecef)

    data_dir = os.path.join(_script_dir, "data")
    v_band_csv = os.path.join(data_dir, "V-Band 50.3 GHz absolute antenna pattern.csv")
    if not os.path.exists(v_band_csv):
        print(f"WARNING: V-Band antenna CSV not found: {v_band_csv}")
        print("  Using ITU fallback (SSMI-S antenna unknown; same as ATMS for now).")
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

    print(f"Loading {args.nc4}...")
    data = load_ssmis_nc4_and_build_arrays(args.nc4)
    n_obs = data["n_obs"]
    unique_ts = np.unique(data["timestamps"])
    print(f"  Observations: {n_obs:,}; unique timestamps: {len(unique_ts):,}")
    print("Computing emitter density (vectorized)...")
    t0 = time_module.perf_counter()
    density = get_emitter_density_vectorized(data["lat"], data["lon"])
    print(f"  Done in {time_module.perf_counter() - t0:.1f} s")

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    satellite_name = args.sat.strip().replace(" ", "_")
    out_base_nc4 = os.path.splitext(os.path.basename(args.nc4))[0]
    top5_path = os.path.join(out_dir, f"{satellite_name}_{out_base_nc4}_top5.txt")
    import pandas as pd
    with open(top5_path, "w") as top5_file:
        for idx, (ch_num, center_freq_hz, bandwidth_hz) in enumerate(SSMIS_CHANNEL_CONFIGS):
            emitter_fundamental_hz = emitter_fundamental_hz_list[idx]
            harmonic_freq_hz = 2.0 * emitter_fundamental_hz
            freq_min = center_freq_hz - bandwidth_hz / 2.0
            freq_max = center_freq_hz + bandwidth_hz / 2.0
            harmonic_in_band = freq_min <= harmonic_freq_hz <= freq_max

            out_csv = os.path.join(
                out_dir, f"{satellite_name}_{out_base_nc4}_5G_RFI_ch{ch_num}.csv"
            )
            print(
                f"\nChannel {ch_num}: {center_freq_hz/1e9:.2f} GHz, BW={bandwidth_hz/1e6:.0f} MHz, "
                f"emitter fundamental={emitter_fundamental_hz/1e9:.2f} GHz "
                f"(2nd harmonic={harmonic_freq_hz/1e9:.2f} GHz)"
            )

            if not harmonic_in_band:
                print(
                    f"  2nd harmonic out of channel band "
                    f"[{freq_min/1e9:.3f}, {freq_max/1e9:.3f}] GHz → zero RFI for all observations"
                )
                df = pd.DataFrame({
                    "timestamp": data["timestamps"],
                    "lat": np.round(data["lat"], 6),
                    "lon": np.round(data["lon"], 6),
                    "rfi_power_dBW": np.round(np.full(n_obs, -300.0), 3),
                    "rfi_brightness_temperature_K": ["0.000e+00"] * n_obs,
                })
                df.to_csv(out_csv, index=False)
                print(f"  Wrote {out_csv} ({len(df):,} rows)")
            else:
                df = run_rfi_for_channel_ssmis(
                    data,
                    density,
                    ecef_lookup,
                    v_band_antenna,
                    emitter_antenna,
                    ch_num,
                    center_freq_hz,
                    bandwidth_hz,
                    emitter_fundamental_hz,
                    out_csv,
                )

            rfi_K_float = df["rfi_brightness_temperature_K"].astype(float)
            idx_top5 = rfi_K_float.abs().nlargest(5).index
            ch_header = (f"Channel {ch_num}: {center_freq_hz/1e9:.2f} GHz, BW={bandwidth_hz/1e6:.0f} MHz, "
                         f"emitter fundamental={emitter_fundamental_hz/1e9:.2f} GHz "
                         f"(2nd harmonic={harmonic_freq_hz/1e9:.2f} GHz)")
            top5_file.write(f"\n{ch_header}\n")
            print("  Top 5 by |rfi_brightness_temperature_K|:")
            top5_file.write("  Top 5 by |rfi_brightness_temperature_K|:\n")
            for rank, idx in enumerate(idx_top5, start=1):
                row = df.loc[idx]
                line = (f"    [{rank}] lat: {row['lat']}, lon: {row['lon']}, "
                        f"rfi_power_dBW: {row['rfi_power_dBW']}, "
                        f"rfi_Tb_K: {row['rfi_brightness_temperature_K']}")
                print(line)
                top5_file.write(line + "\n")

    print(f"\nTop 5 summary written to {top5_path}")

    # Combine channel CSVs into one CSV. Set remove_channel_files=False to keep the per-channel CSVs.
    combined_path = combine_channel_csvs(out_dir, satellite_name, out_base_nc4, remove_channel_files=False)
    if combined_path is not None:
        print(f"Combined channel CSVs to {combined_path} (per-channel CSVs removed).")

    elapsed = time_module.perf_counter() - t_start
    print(f"\nOverall execution time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
