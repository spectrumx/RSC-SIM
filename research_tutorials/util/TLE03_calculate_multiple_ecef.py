"""
Compute ECEF (Earth-Centered, Earth-Fixed) position lookup tables from TLE data
and per-satellite timestamp CSVs in a sensor directory.

This script is the third step in the pre-processor: (1) TLE01 downloads TLE archive,
(2) TLE02 generates per-satellite timestamp CSVs per nc4 file in a sensor directory
(ATMS/, AMSU-A/, SSMI-S/), (3) TLE03 reads all *_timestamp_*.csv in that directory,
infers satellite name from each filename ({satellite}_timestamp_{stem}.csv), loads
the corresponding TLE from the TLE directory, and writes {satellite}_ECEF_lookup_{stem}.csv
in the same directory.

Input arguments (command-line):

  -input-dir INPUT_DIR  Sensor directory containing timestamp CSVs from TLE02
                        (e.g. ATMS, AMSU-A, SSMI-S). ECEF lookups are written here.
  -tle-dir TLE_DIR      Directory containing TLE files (default: data). TLE files
                        are named {satellite}_TLE.txt (e.g. JPSS-1_TLE.txt).

Example:
  python TLE03_calculate_multiple_ecef.py -input-dir ATMS -tle-dir data
  python TLE03_calculate_multiple_ecef.py -input-dir AMSU-A -tle-dir data
  python TLE03_calculate_multiple_ecef.py -input-dir SSMI-S -tle-dir data
"""  # noqa: E501

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skyfield.api import load
from skyfield.framelib import itrs

# Default TLE directory relative to script
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_TLE_DIR = _SCRIPT_DIR / "data"


def generate_ecef_lookup(tle_file_path, input_csv_path, output_csv_path):
    print("Loading Time Scale and TLEs...")
    ts = load.timescale()
    raw_tles = load.tle_file(tle_file_path)

    # 1. Clean and Sort TLEs
    unique_tles = {sat.epoch.tt: sat for sat in raw_tles}
    tles = [unique_tles[k] for k in sorted(unique_tles.keys())]
    tle_epochs = np.array([sat.epoch.tt for sat in tles])
    midpoints = (tle_epochs[:-1] + tle_epochs[1:]) / 2.0

    print(f"Reading timestamps from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    df['parsed_time'] = pd.to_datetime(df['timestamp'])

    print("Converting datetimes to Skyfield time arrays...")
    t = ts.utc(
        df['parsed_time'].dt.year.values,
        df['parsed_time'].dt.month.values,
        df['parsed_time'].dt.day.values,
        df['parsed_time'].dt.hour.values,
        df['parsed_time'].dt.minute.values,
        df['parsed_time'].dt.second.values + (df['parsed_time'].dt.microsecond.values / 1e6)
    )

    print("Assigning timestamps to closest TLEs...")
    timestamp_tts = t.tt
    tle_indices = np.searchsorted(midpoints, timestamp_tts)

    x_results = np.zeros(len(df))
    y_results = np.zeros(len(df))
    z_results = np.zeros(len(df))

    print("Calculating Vectorized ECEF coordinates...")
    unique_indices = np.unique(tle_indices)

    for idx in unique_indices:
        mask = (tle_indices == idx)
        sat = tles[idx]
        t_chunk = t[mask]
        geocentric = sat.at(t_chunk)
        x, y, z = geocentric.frame_xyz(itrs).km
        x_results[mask] = x
        y_results[mask] = y
        z_results[mask] = z

    print("Saving lookup table...")
    df['X_km'] = x_results
    df['Y_km'] = y_results
    df['Z_km'] = z_results
    df_out = df[['timestamp', 'X_km', 'Y_km', 'Z_km']]
    df_out.to_csv(output_csv_path, index=False)
    print(f"Done! Lookup table saved to {output_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ECEF lookup tables from TLE directory and per-satellite timestamp CSVs in a sensor directory."  # noqa: E501
    )
    parser.add_argument(
        '-input-dir',
        dest='input_dir',
        required=True,
        metavar='INPUT_DIR',
        help='Sensor directory with timestamp CSVs from TLE02 (e.g. ATMS, AMSU-A, SSMI-S).',
    )
    parser.add_argument(
        '-tle-dir',
        dest='tle_dir',
        default=None,
        metavar='TLE_DIR',
        help='Directory with TLE files named {satellite}_TLE.txt (default: data under script).',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    tle_dir = Path(args.tle_dir).resolve() if args.tle_dir else _DEFAULT_TLE_DIR
    if not tle_dir.is_dir():
        raise NotADirectoryError(f"TLE directory not found: {tle_dir}")

    # Find all timestamp CSVs: {satellite}_timestamp_{stem}.csv
    timestamp_files = sorted(input_dir.glob("*_timestamp_*.csv"))
    timestamp_files = [f for f in timestamp_files if "_timestamp_" in f.name and f.suffix == ".csv"]

    if not timestamp_files:
        print(f"No *_timestamp_*.csv files found in {input_dir}")
        exit(1)

    print(f"Found {len(timestamp_files):,} timestamp CSV(s) in {input_dir}.")
    for ts_path in timestamp_files:
        # Parse: {satellite}_timestamp_{stem}.csv
        name = ts_path.stem
        if not name.endswith("_timestamp_"):
            idx = name.find("_timestamp_")
            if idx < 0:
                print(f"  Skip (invalid name): {ts_path.name}")
                continue
        idx = name.find("_timestamp_")
        satellite_name = name[:idx]
        stem = name[idx + len("_timestamp_"):]

        tle_path = tle_dir / f"{satellite_name}_TLE.txt"
        if not tle_path.is_file():
            print(f"  Skip {ts_path.name}: TLE not found {tle_path}")
            continue

        out_name = f"{satellite_name}_ECEF_lookup_{stem}.csv"
        out_path = input_dir / out_name
        generate_ecef_lookup(
            tle_file_path=str(tle_path),
            input_csv_path=str(ts_path),
            output_csv_path=str(out_path),
        )
