"""
Generate timestamp CSVs per netCDF-4 file, split by satellite (SAID).

Each nc4 file contains data from multiple satellites (distinguished by SAID).
Reads each .nc4 in the input directory (sensor-based: ATMS/, AMSU-A/, SSMI-S/),
extracts YEAR, MONTH, DAYS, HOUR, MINUTE, SECOND and SAID, builds timestamps,
splits by SAID into per-satellite lists, and writes one CSV per (satellite, nc4)
into the same directory. Output filename: {satellite_name}_timestamp_{nc4_stem}.csv.

Usage:
    python TLE02_generate_timestamps_csv_from_nc4.py --input-dir ATMS
    python TLE02_generate_timestamps_csv_from_nc4.py --input-dir AMSU-A
    python TLE02_generate_timestamps_csv_from_nc4.py --input-dir SSMI-S
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

# Default paths relative to this script
_SCRIPT_DIR = Path(__file__).resolve().parent

TIME_VARS = ("YEAR", "MONTH", "DAYS", "HOUR", "MINUTE", "SECOND")
SAID_VAR = "SAID"

# Sensor name (input-dir name) -> SAID value (int) -> satellite name
SENSOR_SAID_TO_SATELLITE = {
    "ATMS": {
        224: "SUOMI-NPP",
        225: "JPSS-1",
    },
    "AMSU-A": {
        206: "NOAA-15",
        209: "NOAA-18",
        223: "NOAA-19",
        3: "METOP-B",
        5: "METOP-C",
    },
    "SSMI-S": {
        285: "DMSP-F17"
    },
}


def _get_time_array(ds: Dataset, name: str) -> np.ndarray:
    """Get 1D time variable as numpy array (unmasked, flattened)."""
    if name not in ds.variables:
        raise KeyError(f"Variable '{name}' not found in {list(ds.variables.keys())}")
    arr = np.asarray(ds.variables[name][...])
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=float).ravel()


def _timestamps_by_satellite_from_nc4(
    filepath: Path,
    said_to_satellite: dict[int, str],
) -> dict[str, list[str]]:
    """
    Read one .nc4 file, extract TIME_VARS and SAID, build timestamps per observation,
    group by SAID -> satellite name, return dict satellite_name -> list of timestamp strings
    (with duplicates per obs; caller will unique/sort).
    """
    with Dataset(str(filepath), "r") as ds:
        for v in TIME_VARS:
            if v not in ds.variables:
                return {}
        if SAID_VAR not in ds.variables:
            raise KeyError(
                f"Variable '{SAID_VAR}' not found in {list(ds.variables.keys())}. "
                "Required to split data by satellite."
            )
        year = _get_time_array(ds, "YEAR")
        month = _get_time_array(ds, "MONTH")
        days = _get_time_array(ds, "DAYS")
        hour = _get_time_array(ds, "HOUR")
        minute = _get_time_array(ds, "MINUTE")
        second = _get_time_array(ds, "SECOND")
        said_arr = np.asarray(ds.variables[SAID_VAR][...]).ravel()
        if hasattr(said_arr, "filled"):
            said_arr = said_arr.filled(-999)
        said_arr = np.asarray(said_arr, dtype=int)

        n = len(year)
        if not (
            n == len(month) == len(days) == len(hour) == len(minute) == len(second) == len(said_arr)
        ):
            return {}

        by_satellite: dict[str, list[str]] = defaultdict(list)
        for i in range(n):
            said_val = int(said_arr[i])
            if said_val not in said_to_satellite:
                continue
            sat_name = said_to_satellite[said_val]
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
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S") + f".{micro // 1000:03d}"
            by_satellite[sat_name].append(ts_str)

    return dict(by_satellite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-satellite timestamp CSVs from nc4 files (sensor dir; split by SAID)."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Sensor directory containing .nc4 files (e.g. ATMS, AMSU-A, SSMI-S). Timestamp CSVs are written here.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")
    sensor_name = input_dir.name
    if sensor_name not in SENSOR_SAID_TO_SATELLITE:
        raise ValueError(
            f"Unknown sensor directory name: {sensor_name!r}. "
            f"Expected one of: {list(SENSOR_SAID_TO_SATELLITE.keys())}"
        )
    said_to_satellite = SENSOR_SAID_TO_SATELLITE[sensor_name]

    nc4_files = sorted(input_dir.glob("*.nc4"))
    print(
        f"Found {len(nc4_files):,} .nc4 files in {input_dir}. "
        "Writing per-satellite timestamp CSVs into same directory."
    )

    for fp in nc4_files:
        try:
            by_sat = _timestamps_by_satellite_from_nc4(fp, said_to_satellite)
            if not by_sat:
                print(f"  Warning: skipped {fp.name} (missing time vars or empty)")
                continue
            for sat_name, ts_list in by_sat.items():
                unique_ts = sorted(set(ts_list))
                out_name = f"{sat_name}_timestamp_{fp.stem}.csv"
                out_path = input_dir / out_name
                df = pd.DataFrame({"timestamp": unique_ts})
                df.to_csv(out_path, index=False)
                print(f"  {fp.name} -> {out_name} ({len(unique_ts):,} unique timestamps)")
        except Exception as e:
            print(f"Warning: skipped {fp.name}: {e}")


if __name__ == "__main__":
    main()
