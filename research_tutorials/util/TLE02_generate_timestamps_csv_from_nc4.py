"""
Generate one CSV of unique timestamps per netCDF-4 file (ATMS or similar).

Reads each .nc4 file in the input directory, extracts YEAR, MONTH, DAYS, HOUR,
MINUTE, SECOND, builds ISO 8601-style timestamps (YYYY-MM-DD HH:MM:SS.fff),
collects unique timestamps within that file only, and writes one CSV per nc4
file into the same directory. Output filename: {satellite_name}_timestamp_
{nc4_basename_no_ext}.csv (e.g. JPSS-1_timestamp_atms_2023080112.csv).
Satellite name is taken from the input directory name.

Usage:
    python TLE02_generate_timestamps_csv_from_nc4.py --input-dir JPSS-1
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

# Default paths relative to this script
_SCRIPT_DIR = Path(__file__).resolve().parent

TIME_VARS = ("YEAR", "MONTH", "DAYS", "HOUR", "MINUTE", "SECOND")


def _get_time_array(ds: Dataset, name: str) -> np.ndarray:
    """Get 1D time variable as numpy array (unmasked, flattened)."""
    if name not in ds.variables:
        raise KeyError(f"Variable '{name}' not found in {list(ds.variables.keys())}")
    arr = np.asarray(ds.variables[name][...])
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=float).ravel()


def _timestamps_from_nc4(filepath: Path) -> list[str]:
    """
    Read one .nc4 file, extract YEAR, MONTH, DAYS, HOUR, MINUTE, SECOND,
    return list of timestamp strings. Dataset is closed on exit; time arrays
    are explicitly freed before return to minimize peak memory.
    """
    out = []
    with Dataset(str(filepath), "r") as ds:
        for v in TIME_VARS:
            if v not in ds.variables:
                return out
        year = _get_time_array(ds, "YEAR")
        month = _get_time_array(ds, "MONTH")
        days = _get_time_array(ds, "DAYS")
        hour = _get_time_array(ds, "HOUR")
        minute = _get_time_array(ds, "MINUTE")
        second = _get_time_array(ds, "SECOND")
        n = len(year)
        if not (n == len(month) == len(days) == len(hour) == len(minute) == len(second)):
            return out
        for i in range(n):
            y, mo, d = int(year[i]), int(month[i]), int(days[i])
            h, mi = int(hour[i]), int(minute[i])
            sec = float(second[i])
            sec_whole = int(sec)
            micro = int(round((sec - sec_whole) * 1_000_000))
            if micro >= 1_000_000:
                micro = 0
                sec_whole += 1
            micro = min(micro, 999_999)
            try:
                dt = datetime(y, mo, d, h, mi, sec_whole, micro)
            except ValueError:
                continue
            # ISO 8601 style: YYYY-MM-DD HH:MM:SS.fff
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S") + f".{micro // 1000:03d}"
            out.append(ts_str)
        # Release time arrays before returning so memory is freed while set is updated
        del year, month, days, hour, minute, second
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one CSV of unique timestamps per netCDF-4 file (YEAR, MONTH, DAYS, HOUR, MINUTE, SECOND)."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing .nc4 files (e.g., JPSS-1 or path/to/JPSS-1). Timestamp CSVs are written here. Directory name is used as satellite_name.",  # noqa: E501
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")
    satellite_name = input_dir.name

    nc4_files = sorted(input_dir.glob("*.nc4"))
    print(f"Found {len(nc4_files):,} .nc4 files in {input_dir}. Writing one timestamp CSV per file to same directory.")

    for fp in nc4_files:
        try:
            ts_list = _timestamps_from_nc4(fp)
            unique_ts = sorted(set(ts_list))
            out_name = f"{satellite_name}_timestamp_{fp.stem}.csv"
            out_path = input_dir / out_name
            df = pd.DataFrame({"timestamp": unique_ts})
            df.to_csv(out_path, index=False)
            print(f"  {fp.name} -> {out_name} ({len(unique_ts):,} unique timestamps)")
        except Exception as e:
            print(f"Warning: skipped {fp.name}: {e}")


if __name__ == "__main__":
    main()
