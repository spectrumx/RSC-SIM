"""
Pre-render scenario assets for the demo gallery tab.

Run once before the conference. All outputs go to `demo_app/assets/gallery/`,
which is what `panels/gallery.py` reads.

Usage:
    python demo_app/precompute.py
    python demo_app/precompute.py --include-nwp  # also try ATMS/AMSU-A/SSMI-S maps

The default invocation only uses files bundled in `research_tutorials/data/`
and is therefore safe to run on the booth laptop offline. The optional
`--include-nwp` flag also renders TMBR vs TMBR_RFI maps if a `*_RFI.nc4`
output is available alongside the input nc4 (per the NWP pipeline). If those
files are not present, that step is skipped silently.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up imports the same way the Streamlit app does.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "educational_tutorials"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(_REPO_ROOT, "research_tutorials", "data")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gallery")
os.makedirs(ASSETS_DIR, exist_ok=True)


def _save_fig(fig, name: str, *, dpi: int = 150) -> str:
    out_path = os.path.join(ASSETS_DIR, name)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Scenario 1: Doppler waterfall from a single Starlink pass
# ---------------------------------------------------------------------------


def render_doppler_waterfall() -> Optional[str]:
    """
    Build a synthetic spectrogram showing a Starlink carrier sweeping across
    the radio band as it passes overhead. Uses the bundled Starlink Arrow
    file's azimuth/elevation/distance to derive radial velocity, then plots
    the Doppler-shifted carrier power vs time.
    """
    print("Rendering Doppler waterfall...")
    starlink_path = os.path.join(
        DATA_DIR,
        "Starlink_trajectory_Westford_2025-02-18T15_00_00.000_2025-02-18T15_45_00.000.arrow",
    )
    if not os.path.isfile(starlink_path):
        print("  skip: bundled Starlink trajectory missing")
        return None

    import pyarrow as pa

    with pa.memory_map(starlink_path, "r") as src:
        table = pa.ipc.open_file(src).read_all()
    df = table.to_pandas().rename(
        columns={"timestamp": "times", "ranges_westford": "distances"}
    )
    df["times"] = pd.to_datetime(df["times"])
    df = df[~df["sat"].astype(str).str.contains("DTC")]
    df = df[df["elevations"] > 30.0]
    if df.empty:
        print("  skip: no high-elevation Starlinks in trajectory")
        return None

    # Pick the satellite with the most samples in this window.
    sat_name = df["sat"].value_counts().idxmax()
    df_sat = df[df["sat"] == sat_name].sort_values("times").reset_index(drop=True)
    if len(df_sat) < 5:
        print(f"  skip: too few samples for {sat_name}")
        return None

    # Estimate radial velocity from finite-differenced range.
    t = df_sat["times"].astype("int64").to_numpy() / 1e9
    rng = df_sat["distances"].to_numpy()
    dt = np.gradient(t)
    drng = np.gradient(rng)
    radial_vel = drng / np.where(dt == 0, 1, dt)  # m/s

    f0 = 11.325e9  # Hz
    c = 3e8
    df_shift = -f0 * radial_vel / c  # Doppler shift, Hz

    # Build a fake spectrogram: Gaussian centered at df_shift over a 100 kHz band.
    bw = 100e3
    n_freq = 128
    freqs = np.linspace(-bw / 2, bw / 2, n_freq)
    spec = np.zeros((len(t), n_freq))
    sigma = 2e3  # 2 kHz line width
    for i, sh in enumerate(df_shift):
        spec[i] = np.exp(-0.5 * ((freqs - sh) / sigma) ** 2)
    spec_db = 10 * np.log10(spec + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 4))
    times_plot = df_sat["times"].to_numpy()
    pc = ax.pcolormesh(
        times_plot,
        freqs / 1e3,
        spec_db.T,
        cmap="viridis",
        shading="auto",
    )
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("Offset from 11.325 GHz [kHz]")
    ax.set_title(f"Doppler signature — {sat_name}")
    fig.colorbar(pc, ax=ax, label="Spectral power [dB]")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save_fig(fig, "doppler_waterfall.png")


# ---------------------------------------------------------------------------
# Scenario 2: "What 5G + Starlink looks like to a weather sat" infographic
# ---------------------------------------------------------------------------


def render_weather_summary() -> str:
    """
    A purely visual summary that doesn't require Phase 2/3 data. Shows the
    bundled K-Band and V-Band antenna patterns side-by-side, annotated with
    where Starlink harmonics fall.
    """
    print("Rendering weather summary infographic...")

    from weather_sat_mdl import load_weather_sat_antenna_from_csv  # noqa: E402

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), subplot_kw={"projection": "polar"})

    for ax, csv_name, freq, title in [
        (axes[0], "K-Band 23.8 GHz absolute antenna pattern.csv", 23.8e9, "K-Band 23.8 GHz"),
        (axes[1], "V-Band 50.3 GHz absolute antenna pattern.csv", 50.3e9, "V-Band 50.3 GHz"),
    ]:
        path = os.path.join(DATA_DIR, csv_name)
        if not os.path.isfile(path):
            ax.set_title(f"{title} (missing CSV)")
            continue
        ant = load_weather_sat_antenna_from_csv(path, eta_rad=0.99, valid_freqs=(freq * 0.9, freq * 1.1))
        alphas, gains = ant.get_slice_gain(0.0)
        sort_idx = np.argsort(alphas)
        alphas = alphas[sort_idx]
        gains = gains[sort_idx]
        gains_db = 10 * np.log10(np.clip(gains, 1e-30, None))
        elevation_mapped = np.where(alphas < 0, alphas + 360, alphas)
        ax.plot(np.deg2rad(elevation_mapped), gains_db - gains_db.min() + 1, color="#1f77b4")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_title(title, pad=15)

    fig.suptitle("Suomi-NPP K/V-band antenna patterns (Starlink harmonics arrive through sidelobes)")
    fig.tight_layout()
    return _save_fig(fig, "weather_antennas.png")


# ---------------------------------------------------------------------------
# Scenario 3: NWP TMBR vs TMBR_RFI map (optional, requires actual nc4 outputs)
# ---------------------------------------------------------------------------


def render_nwp_maps() -> Optional[str]:
    """
    Look for any *_RFI.nc4 file in `research_tutorials/util/` (sensor dirs)
    and render a global TMBR vs TMBR_RFI comparison. Skips silently if none
    are found.
    """
    print("Looking for NWP TMBR comparison candidates...")
    util_root = Path(_REPO_ROOT) / "research_tutorials" / "util"
    candidates = list(util_root.rglob("*_RFI.nc4"))
    if not candidates:
        print("  skip: no *_RFI.nc4 files found under research_tutorials/util/")
        return None

    try:
        import netCDF4
    except ImportError:
        print("  skip: netCDF4 not installed")
        return None

    rfi_path = candidates[0]
    try:
        with netCDF4.Dataset(rfi_path) as ds:
            # Try a few likely variable names.
            tmbr = ds.variables.get("TMBR")
            tmbr_rfi = ds.variables.get("TMBR_RFI")
            lat = ds.variables.get("lat") or ds.variables.get("Latitude")
            lon = ds.variables.get("lon") or ds.variables.get("Longitude")
            if tmbr is None or tmbr_rfi is None or lat is None or lon is None:
                print(f"  skip: {rfi_path.name} missing required variables")
                return None
            tmbr_arr = np.array(tmbr[:])
            tmbr_rfi_arr = np.array(tmbr_rfi[:])
            lat_arr = np.array(lat[:])
            lon_arr = np.array(lon[:])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  skip: failed to read {rfi_path}: {exc}")
        return None

    # Use the first channel for a visual.
    if tmbr_arr.ndim == 3:
        tmbr_arr = tmbr_arr[..., 0]
        tmbr_rfi_arr = tmbr_rfi_arr[..., 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in [
        (axes[0], tmbr_arr, "Native TMBR"),
        (axes[1], tmbr_rfi_arr, "TMBR + RFI"),
    ]:
        sc = ax.scatter(lon_arr.ravel(), lat_arr.ravel(), c=data.ravel(), s=1, cmap="viridis")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, label="K")

    fig.suptitle(f"NWP brightness temperature comparison ({rfi_path.name})")
    fig.tight_layout()
    return _save_fig(fig, "nwp_tmbr_vs_rfi.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-render demo gallery assets")
    parser.add_argument(
        "--include-nwp",
        action="store_true",
        help="Also try to render TMBR vs TMBR_RFI maps from any *_RFI.nc4 outputs.",
    )
    args = parser.parse_args()

    print(f"Rendering gallery assets to {ASSETS_DIR}")
    paths = {}
    paths["doppler"] = render_doppler_waterfall()
    paths["weather_summary"] = render_weather_summary()
    if args.include_nwp:
        paths["nwp"] = render_nwp_maps()

    # Write a tiny manifest so the gallery panel knows what's available.
    manifest_path = os.path.join(ASSETS_DIR, "manifest.json")
    import json

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": {k: os.path.basename(v) if v else None for k, v in paths.items()},
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
