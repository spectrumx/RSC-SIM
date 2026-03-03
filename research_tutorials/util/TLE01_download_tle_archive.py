"""
Download TLE (Two-Line Element) archive from Space-Track.org for a given satellite
and date range. Requires Space-Track.org account credentials (USERNAME, PASSWORD).

A 5-day window is applied before the start date and after the end date so that the
requested TLE range extends beyond the user dates. This improves epoch coverage
and helps select TLEs with epochs close to (but not after) target times.

Input arguments (command-line):

  -sat SATELLITE_NAME   Satellite name (case-insensitive). Resolved to NORAD ID
                        internally. Allowed values: SUOMI-NPP, JPSS-1, NOAA-15,
                        NOAA-18, NOAA-19, METOP-B, METOP-C, DMSP-F17.

  -start START_DATE     Start date of interest, format YYYY-MM-DD. The TLE query
                        actually starts 5 days earlier to improve epoch coverage.

  -end END_DATE         End date of interest, format YYYY-MM-DD. The TLE query
                        actually ends 5 days later to improve epoch coverage.

Output: writes TLEs to data/<SATELLITE_NAME>_TLE.txt (e.g. data/JPSS-1_TLE.txt).

Example:
  python TLE01_download_tle_archive.py -sat jpss-1 -start 2023-08-01 -end 2023-11-01
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Space-Track.org account credentials: please register an account and use your own credentials
USERNAME = ''
PASSWORD = ''

# Satellite NORAD IDs (name -> ID lookup uses uppercase for -sat)
NORAD_ID_TO_NAME = {
    37849: 'SUOMI-NPP',
    43013: 'JPSS-1',
    25338: 'NOAA-15',
    28654: 'NOAA-18',
    33591: 'NOAA-19',
    38771: 'METOP-B',
    43689: 'METOP-C',
    29522: 'DMSP-F17',
}
# Uppercase name -> NORAD ID (for -sat argument; accepts lowercase input)
NAME_UPPER_TO_NORAD = {name.upper(): nid for nid, name in NORAD_ID_TO_NAME.items()}

# TLE window padding: request 5 extra days before/after for better epoch coverage
TLE_WINDOW_DAYS = 5
DATE_FMT = '%Y-%m-%d'

# Space-Track.org login URL
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download TLE archive from Space-Track.org for a satellite and date range."
    )
    parser.add_argument(
        '-sat',
        required=True,
        metavar='SATELLITE_NAME',
        help=f"Satellite name (case-insensitive). One of: {list(NORAD_ID_TO_NAME.values())}",
    )
    parser.add_argument(
        '-start',
        required=True,
        metavar='START_DATE',
        help='Start date of interest (YYYY-MM-DD). TLE query will start 5 days earlier.',
    )
    parser.add_argument(
        '-end',
        required=True,
        metavar='END_DATE',
        help='End date of interest (YYYY-MM-DD). TLE query will end 5 days later.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sat_upper = args.sat.strip().upper()
    if sat_upper not in NAME_UPPER_TO_NORAD:
        print(f"Unknown satellite '{args.sat}'. Allowed: {list(NORAD_ID_TO_NAME.values())}")
        return 1
    NORAD_ID = NAME_UPPER_TO_NORAD[sat_upper]
    SATELLITE_NAME = NORAD_ID_TO_NAME[NORAD_ID]

    start_dt = datetime.strptime(args.start.strip(), DATE_FMT)
    end_dt = datetime.strptime(args.end.strip(), DATE_FMT)
    query_start = (start_dt - timedelta(days=TLE_WINDOW_DAYS)).strftime(DATE_FMT)
    query_end = (end_dt + timedelta(days=TLE_WINDOW_DAYS)).strftime(DATE_FMT)

    query_url = (
        f"https://www.space-track.org/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{NORAD_ID}/EPOCH/{query_start}--{query_end}/orderby/EPOCH ASC/format/tle"
    )
    output_file = Path(__file__).resolve().parent / "data" / f"{SATELLITE_NAME}_TLE.txt"

    print(f"Fetching TLEs for {SATELLITE_NAME} (NORAD {NORAD_ID}) from {query_start} to {query_end} "
          f"(user range {args.start} to {args.end} + {TLE_WINDOW_DAYS} day window)...")

    with requests.Session() as session:
        session.post(LOGIN_URL, data={'identity': USERNAME, 'password': PASSWORD})
        response = session.get(query_url)

        if response.status_code == 200:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(response.text.strip())
            n_tles = len(response.text.strip().splitlines()) // 2
            print(f"Success! Saved {n_tles} TLEs to {output_file}")
        else:
            print(f"Failed to fetch data. HTTP Status: {response.status_code}")
            return 1
    return 0


if __name__ == '__main__':
    exit(main())
