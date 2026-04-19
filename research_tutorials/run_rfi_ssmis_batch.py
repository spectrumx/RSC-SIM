import os
import sys
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_rfi_script(nc4_path, sensor_name, script_path, out_dir, max_retries=2):
    """
    Worker: run SSMI-S_RFI_modeling.py for one nc4 (5G harmonic + Starlink gateway; default gateways CSV).
    ECEF lookups from the same directory as the nc4/out_dir. Only SAID 285 (DMSP-F17) gets non-zero RFI.
    """
    cmd = [
        sys.executable, str(script_path),
        "--sensor", sensor_name,
        "--nc4", str(nc4_path),
        "--out_dir", str(out_dir),
    ]

    attempt = 0
    while attempt <= max_retries:
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return "SUCCESS", nc4_path.name
        except subprocess.CalledProcessError as e:
            attempt += 1
            if attempt <= max_retries:
                time.sleep(1)
                continue
            return "ERROR", f"{nc4_path.name} failed after {max_retries} retries.\nReason: {e.stderr}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel batch: run SSMI-S_RFI_modeling.py on every .nc4 (5G + Starlink gateway RFI; "
            "e.g. util/SSMI-S). Only SAID 285 (DMSP-F17) is processed. Optional script flags are not passed."
        )
    )
    parser.add_argument("sensor_dir", help="Sensor directory containing .nc4 and ECEF lookups (e.g. util/SSMI-S)")
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (default: CPU count - 1, minimum 1).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / "SSMI-S_RFI_modeling.py"
    sensor_dir = Path(args.sensor_dir).resolve()
    sensor_name = "SSMI-S"

    if not sensor_dir.exists():
        print(f"Error: Directory {sensor_dir} does not exist.")
        return
    if not script_path.is_file():
        print(f"Error: Script not found: {script_path}")
        return

    nc4_files = list(sensor_dir.glob("*.nc4"))
    total_files = len(nc4_files)

    if args.workers is not None:
        use_cores = max(1, args.workers)
    else:
        use_cores = max(1, (os.cpu_count() or 1) - 2)

    log_file = script_dir / f"SSMI-S_{sensor_dir.name}_error_log.txt"
    with open(log_file, "w") as f:
        f.write(f"--- SSMI-S RFI (5G + Starlink gateway) processing exceptions for {sensor_dir} ---\n")
        f.write(f"Started at: {time.ctime()}\n\n")

    print(f"--- Starting parallel SSMI-S RFI run (5G + Starlink gateway; cores: {use_cores}) ---")
    print(f"Sensor directory: {sensor_dir}")
    print(f"Processing {total_files} nc4 file(s).")
    print(f"Logging errors/skips to: {log_file}")

    start_time = time.perf_counter()
    results = {"SUCCESS": 0, "ERROR": 0, "SKIP": 0}
    completed_count = 0

    # Parallel Execution
    with ProcessPoolExecutor(max_workers=use_cores) as executor:
        future_to_nc4 = {
            executor.submit(run_rfi_script, f, sensor_name, script_path, sensor_dir): f
            for f in nc4_files
        }

        for future in as_completed(future_to_nc4):
            nc4_path = future_to_nc4[future]
            status, message = future.result()
            results[status] += 1
            completed_count += 1

            # Progress: [X/total] nc4_filename ... STATUS
            print(f"[{completed_count}/{total_files}] {nc4_path.name} ... {status}")

            if status in ["ERROR", "SKIP"]:
                with open(log_file, "a") as f:
                    f.write(f"[{status}] {message}\n")

    end_time = time.perf_counter()
    duration = end_time - start_time

    # Final Summary Report
    print("\n" + "="*40)
    print("FINAL BATCH SUMMARY")
    print("="*40)
    print(f"Total Files Processed: {total_files}")
    print(f"  - Successful:  {results['SUCCESS']}")
    print(f"  - Failed:      {results['ERROR']}")
    print(f"  - Skipped:     {results['SKIP']}")
    print("-" * 40)
    print(f"Total Duration:  {duration:.2f} seconds")
    if total_files > 0:
        print(f"Avg per file:    {duration/total_files:.2f} seconds")

    if results['ERROR'] > 0 or results['SKIP'] > 0:
        print(f"\n[!] Check '{log_file}' for details on failed/skipped files.")
    print("="*40)


if __name__ == "__main__":
    main()
