"""
Compute ECEF (Earth-Centered, Earth-Fixed) position lookup table for a weather
satellite from TLE data and timestamp CSVs.

This script is the third step in a sequential pre-processor: (1) TLE01 downloads
TLE archive, (2) TLE02 generates one timestamp CSV per nc4 file in a directory,
(3) TLE03 reads all timestamp CSVs in that directory and writes one ECEF lookup
CSV per timestamp file, in the same directory. Timestamp files are expected to
be named {satellite_name}_timestamp_{stem}.csv (from TLE02); output files are
{satellite_name}_ECEF_lookup_{stem}.csv. Satellite name is the input directory name.

Input arguments (command-line):

  -tle TLE_FILE_PATH    Path to the TLE text file (e.g. from TLE01).

  -input-dir INPUT_DIR  Directory containing timestamp CSVs from TLE02 (e.g. JPSS-1).
                        Each *_timestamp_*.csv is processed; ECEF lookups are
                        written to the same directory.

Example:
  python TLE03_calculate_multiple_ecef.py -tle data/JPSS-1_TLE.txt -input-dir JPSS-1
"""  # noqa: E501

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skyfield.api import load
from skyfield.framelib import itrs


def generate_ecef_lookup(tle_file_path, input_csv_path, output_csv_path):
    print("Loading Time Scale and TLEs...")
    ts = load.timescale()
    raw_tles = load.tle_file(tle_file_path)

    # 1. Clean and Sort TLEs
    # Occasionally Space-Track has duplicate TLEs. We filter them by their exact epoch time.
    unique_tles = {sat.epoch.tt: sat for sat in raw_tles}
    tles = [unique_tles[k] for k in sorted(unique_tles.keys())]
    tle_epochs = np.array([sat.epoch.tt for sat in tles])

    # Calculate the exact middle points between TLE epochs.
    # This creates the boundaries for our "closest TLE" groupings.
    midpoints = (tle_epochs[:-1] + tle_epochs[1:]) / 2.0

    print(f"Reading timestamps from {input_csv_path}...")
    # pandas parses 3 million fractional datetimes in seconds
    df = pd.read_csv(input_csv_path)
    df['parsed_time'] = pd.to_datetime(df['timestamp'])

    # 2. Vectorized conversion to Skyfield Time array
    # We extract the numpy arrays (.values) for maximum speed.
    # Fractional seconds are captured by combining seconds + microseconds.
    print("Converting datetimes to Skyfield time arrays...")
    t = ts.utc(
        df['parsed_time'].dt.year.values,
        df['parsed_time'].dt.month.values,
        df['parsed_time'].dt.day.values,
        df['parsed_time'].dt.hour.values,
        df['parsed_time'].dt.minute.values,
        df['parsed_time'].dt.second.values + (df['parsed_time'].dt.microsecond.values / 1e6)
    )

    # 3. Bin the timestamps to their closest TLE
    # np.searchsorted instantly finds which TLE boundary every timestamp falls into
    print("Assigning timestamps to closest TLEs...")
    timestamp_tts = t.tt
    tle_indices = np.searchsorted(midpoints, timestamp_tts)

    # Prepare empty arrays for our results
    x_results = np.zeros(len(df))
    y_results = np.zeros(len(df))
    z_results = np.zeros(len(df))

    # 4. Process each TLE chunk simultaneously
    print("Calculating Vectorized ECEF coordinates...")
    unique_indices = np.unique(tle_indices)

    for idx in unique_indices:
        # Find all timestamps that belong to this specific TLE
        mask = (tle_indices == idx)
        sat = tles[idx]

        # Slice the Skyfield time array to only pass the relevant times
        t_chunk = t[mask]

        # Calculate ECEF for this entire chunk in one line!
        geocentric = sat.at(t_chunk)
        x, y, z = geocentric.frame_xyz(itrs).km

        # Map results back to our main arrays
        x_results[mask] = x
        y_results[mask] = y
        z_results[mask] = z

    # 5. Format and Save Output
    print("Saving lookup table...")
    df['X_km'] = x_results
    df['Y_km'] = y_results
    df['Z_km'] = z_results

    # Drop the temporary parsed time column, keep the original string and coordinates
    df_out = df[['timestamp', 'X_km', 'Y_km', 'Z_km']]
    df_out.to_csv(output_csv_path, index=False)

    print(f"Done! Lookup table saved to {output_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ECEF lookup tables from TLE file and timestamp CSVs in a directory."
    )
    parser.add_argument(
        '-tle',
        dest='tle_file_path',
        required=True,
        metavar='TLE_FILE_PATH',
        help='Path to the TLE text file (e.g. data/JPSS-1_TLE.txt)',
    )
    parser.add_argument(
        '-input-dir',
        dest='input_dir',
        required=True,
        metavar='INPUT_DIR',
        help='Directory containing timestamp CSVs from TLE02 (e.g. JPSS-1). ECEF lookups are written here.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")
    satellite_name = input_dir.name

    # Find all timestamp CSVs produced by TLE02: {satellite_name}_timestamp_{stem}.csv
    prefix = f"{satellite_name}_timestamp_"
    suffix = ".csv"
    timestamp_files = sorted(input_dir.glob(f"{prefix}*{suffix}"))
    # Require exact prefix so we do not pick up other CSVs
    timestamp_files = [f for f in timestamp_files if f.name.startswith(prefix) and f.name.endswith(suffix)]

    if not timestamp_files:
        print(f"No timestamp CSVs found in {input_dir} matching {prefix}*{suffix}")
        exit(1)

    print(f"Found {len(timestamp_files):,} timestamp CSV(s) in {input_dir}.")
    for ts_path in timestamp_files:
        stem = ts_path.name[len(prefix) : -len(suffix)]
        out_name = f"{satellite_name}_ECEF_lookup_{stem}.csv"
        out_path = input_dir / out_name
        generate_ecef_lookup(
            tle_file_path=args.tle_file_path,
            input_csv_path=str(ts_path),
            output_csv_path=str(out_path),
        )
