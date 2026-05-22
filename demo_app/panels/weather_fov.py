"""
Tab 2: Weather satellite single FOV.

Streamlined live version of `research_tutorials/tuto_radiomdl_weather_phase1.py`.

Loads the bundled K-Band / V-Band Suomi-NPP antenna CSVs and the JPSS / Starlink
trajectory Arrow files (if present), then computes a fast per-time-step Tb
budget for a single FOV: Earth main-lobe Tb, sky background, system noise, and
Starlink back-lobe interference (with optional harmonics).

Phase 2/3 features (5G ground emitters, gateways with full link budget) require
extra bulky data files (`GHS_POP_*.tif`, `itu_iclw_rain_info_*.nc`) that aren't
shipped with the repo; when they're missing, the controls are still shown but
the live computation falls back to Phase 1 + a *qualitative* gateway hint.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sim_cache  # noqa: F401  (registers `src/` on sys.path)
from sim_cache import data_file_exists, repo_data_path

# Make the RSC-SIM modules importable (sim_cache already arranged sys.path).
from radio_types import Antenna  # noqa: E402
from astro_mdl import antenna_mdl_ITU, power_to_temperature  # noqa: E402
from weather_sat_mdl import (  # noqa: E402
    calculate_earth_brightness_temperature,
    load_weather_sat_antenna_from_csv,
    starlink_backlobe_to_weather_sat_link_budget,
)


# ---------------------------------------------------------------------------
# Bundled file paths
# ---------------------------------------------------------------------------


_K_BAND_CSV = "K-Band 23.8 GHz absolute antenna pattern.csv"
_V_BAND_CSV = "V-Band 50.3 GHz absolute antenna pattern.csv"

# The bundled JPSS + Starlink trajectories use this 2025-11-01 window.
_JPSS_FILE = "jpss_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow"
_STARLINK_FILE = "Starlink_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow"

_OBS_START = datetime(2025, 11, 1, 8, 15, 0)
_OBS_END = datetime(2025, 11, 1, 8, 21, 0)


_DEMO_DEFAULTS = {
    "wx_band": "K-Band (23.8 GHz)",
    "wx_starlink_freq_ghz": 11.9,  # 2nd harmonic into K-band
    "wx_eirp_dbw": -15 + 10 * np.log10(300),
    "wx_n_starlinks": 30,
    "wx_pol_loss_db": 3.0,
    "wx_time_offset_min": 1.0,
    "wx_show_gateway": False,
    "wx_gateway_lat": 35.0,
    "wx_gateway_lon": -100.0,
    "wx_emitter_density": 0.0,
}


_BANDS = {
    "K-Band (23.8 GHz)": {
        "csv": _K_BAND_CSV,
        "freq_hz": 23.8e9,
        "bandwidth_hz": 270e6,
        "valid_freqs": (20e9, 30e9),
        "system_temp_k": 300.0,
    },
    "V-Band (50.3 GHz)": {
        "csv": _V_BAND_CSV,
        "freq_hz": 50.3e9,
        "bandwidth_hz": 180e6,
        "valid_freqs": (40e9, 60e9),
        "system_temp_k": 400.0,
    },
}


_STARLINK_HARMONICS = [
    (2.0, 0.01),    # -20 dBc
    (3.0, 0.003),   # -25 dBc
    (4.0, 0.001),   # -30 dBc
]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading K-Band antenna...")
def _load_k_band_antenna() -> Optional[Antenna]:
    path = repo_data_path(_K_BAND_CSV)
    if not os.path.isfile(path):
        return None
    return load_weather_sat_antenna_from_csv(path, eta_rad=0.99, valid_freqs=(20e9, 30e9))


@st.cache_resource(show_spinner="Loading V-Band antenna...")
def _load_v_band_antenna() -> Optional[Antenna]:
    path = repo_data_path(_V_BAND_CSV)
    if not os.path.isfile(path):
        return None
    return load_weather_sat_antenna_from_csv(path, eta_rad=0.99, valid_freqs=(40e9, 60e9))


@st.cache_resource(show_spinner="Building Starlink back-lobe model...")
def _starlink_backlobe_antenna() -> Antenna:
    """ITU-mask Starlink antenna (matches `tuto_radiomdl_weather_phase1.py`)."""
    alphas = np.arange(0, 181)
    betas = np.arange(0, 351, 10)
    pat = antenna_mdl_ITU(39.3, 3.0, alphas, betas)
    return Antenna.from_dataframe(pat, 0.5, (10.7e9, 12.7e9))


@st.cache_resource(show_spinner="Loading JPSS trajectory...")
def _load_jpss_trajectory() -> Optional[pd.DataFrame]:
    path = repo_data_path(_JPSS_FILE)
    if not os.path.isfile(path):
        return None
    import pyarrow as pa

    with pa.memory_map(path, "r") as src:
        table = pa.ipc.open_file(src).read_all()
    df = table.to_pandas()
    df = df.rename(
        columns={"timestamp": "times", "ranges_westford": "distances"}
    )
    df["times"] = pd.to_datetime(df["times"])
    return df


@st.cache_resource(show_spinner="Loading Starlink trajectory (Phase 1 window)...")
def _load_starlink_phase1_trajectory() -> Optional[pd.DataFrame]:
    path = repo_data_path(_STARLINK_FILE)
    if not os.path.isfile(path):
        return None
    import pyarrow as pa

    with pa.memory_map(path, "r") as src:
        table = pa.ipc.open_file(src).read_all()
    df = table.to_pandas()
    df = df.rename(columns={"timestamp": "times", "ranges_westford": "distances"})
    df["times"] = pd.to_datetime(df["times"])
    return df


# ---------------------------------------------------------------------------
# Tb-budget computation (fast, single time step)
# ---------------------------------------------------------------------------


def _reset_to_defaults() -> None:
    for key, value in _DEMO_DEFAULTS.items():
        st.session_state[key] = value


def _tb_budget_at_time(
    band_key: str,
    time_plot: datetime,
    starlink_freq_hz: float,
    starlink_eirp_dbw: float,
    n_starlinks: int,
    pol_loss_db: float,
) -> dict:
    """
    Compute the Tb budget at a single time step. Mirrors the inner loop of
    `model_weather_sat_observed_power` but without ECEF transformation -- we
    use the AER -> body-frame angles directly, which is good enough for a
    booth visual.
    """
    band = _BANDS[band_key]
    freq = band["freq_hz"]
    bw = band["bandwidth_hz"]
    sys_temp = band["system_temp_k"]

    if band_key.startswith("K-"):
        antenna = _load_k_band_antenna()
    else:
        antenna = _load_v_band_antenna()
    if antenna is None:
        return {
            "earth_k": np.nan,
            "sky_k": np.nan,
            "system_k": np.nan,
            "starlink_k": np.nan,
            "missing_data": True,
        }
    starlink_ant = _starlink_backlobe_antenna()

    earth_temp = calculate_earth_brightness_temperature(freq, base_temp=280.0)
    earth_gain = antenna.get_gain_value(0.0, 0.0)  # nadir
    earth_temp_eff = earth_temp * earth_gain
    sky_temp_eff = 2.73 * 0.1  # typical sidelobe leakage factor
    system_temp_eff = sys_temp

    starlink_df = _load_starlink_phase1_trajectory()
    starlink_temp_eff = 0.0
    n_visible = 0
    if starlink_df is not None and not starlink_df.empty:
        target = pd.Timestamp(time_plot)
        diffs = (starlink_df["times"] - target).abs()
        nearest_time = starlink_df.loc[diffs.idxmin(), "times"]
        sats_at_t = starlink_df[starlink_df["times"] == nearest_time]
        sats_at_t = sats_at_t[~sats_at_t["sat"].astype(str).str.contains("DTC")]
        sats_at_t = sats_at_t[sats_at_t["elevations"] > 5.0]
        if n_starlinks > 0 and len(sats_at_t) > n_starlinks:
            sats_at_t = sats_at_t.iloc[:n_starlinks]
        n_visible = len(sats_at_t)

        pol_factor = 10 ** (-pol_loss_db / 10.0)
        starlink_power_w = 10 ** (starlink_eirp_dbw / 10.0)
        starlink_temp_per_sat = power_to_temperature(starlink_power_w, bw)

        for _, row in sats_at_t.iterrows():
            # Map AER (azimuth, elevation, range) to a weather-sat-frame angle.
            # The weather sat is overhead at nadir-pointing; we approximate
            # alpha (angle from nadir) as 90° - elevation_observer and beta
            # as the satellite's azimuth. This is a coarse but illustrative
            # mapping and matches the visual "satellites passing under the
            # nadir cone" story we want to tell at the booth.
            alpha_rad = np.deg2rad(90.0 - float(row["elevations"]))
            beta_rad = np.deg2rad(float(row["azimuths"]))
            link_budget = starlink_backlobe_to_weather_sat_link_budget(
                0.0, 0.0,
                antenna,
                alpha_rad, beta_rad,
                float(row["distances"]),
                starlink_ant,
                freq,
                polarization_loss_factor=pol_factor,
                starlink_fundamental_freq=starlink_freq_hz,
                harmonics=_STARLINK_HARMONICS,
                observation_bandwidth=bw,
            )
            starlink_temp_eff += link_budget * starlink_temp_per_sat

    return {
        "earth_k": float(earth_temp_eff),
        "sky_k": float(sky_temp_eff),
        "system_k": float(system_temp_eff),
        "starlink_k": float(starlink_temp_eff),
        "n_visible": int(n_visible),
        "missing_data": False,
        "freq_ghz": freq / 1e9,
        "bandwidth_mhz": bw / 1e6,
    }


@st.cache_data(show_spinner="Computing Tb budget...", max_entries=64)
def _cached_tb_budget(
    band_key: str,
    time_iso: str,
    starlink_freq_hz: float,
    starlink_eirp_dbw: float,
    n_starlinks: int,
    pol_loss_db: float,
) -> dict:
    return _tb_budget_at_time(
        band_key,
        datetime.fromisoformat(time_iso),
        starlink_freq_hz,
        starlink_eirp_dbw,
        n_starlinks,
        pol_loss_db,
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _stacked_budget_figure(budget: dict) -> go.Figure:
    """Stacked bar of effective Tb contributions."""
    labels = ["Earth (main lobe)", "Sky (sidelobe)", "System noise", "Starlink (backlobe)"]
    values = [budget["earth_k"], budget["sky_k"], budget["system_k"], budget["starlink_k"]]
    colors = ["#1f77b4", "#9467bd", "#8c564b", "#d62728"]

    fig = go.Figure()
    for label, value, color in zip(labels, values, colors):
        fig.add_trace(
            go.Bar(
                x=["Tb budget"],
                y=[value],
                name=label,
                marker_color=color,
                hovertemplate=f"{label}<br>%{{y:.2f}} K<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        yaxis_title="Effective brightness temperature [K]",
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _antenna_pattern_figure(band_key: str):
    """Polar slice of the Suomi-NPP antenna pattern (matplotlib for cleanliness)."""
    import matplotlib.pyplot as plt

    if band_key.startswith("K-"):
        antenna = _load_k_band_antenna()
    else:
        antenna = _load_v_band_antenna()
    if antenna is None:
        return None

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    alphas, gains = antenna.get_slice_gain(0.0)
    sort_idx = np.argsort(alphas)
    alphas = alphas[sort_idx]
    gains = gains[sort_idx]
    gains_db = 10 * np.log10(np.clip(gains, 1e-30, None))

    elevation_mapped = np.where(alphas < 0, alphas + 360, alphas)
    theta = np.deg2rad(elevation_mapped)
    r = gains_db - gains_db.min() + 1
    ax.plot(theta, r, color="#1f77b4", linewidth=2)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(f"{band_key} antenna pattern (slice)", pad=18)
    fig.tight_layout()
    return fig


def _fov_map_figure(
    time_plot: datetime,
    band_key: str,
    show_gateway: bool,
    gateway_lat: float,
    gateway_lon: float,
):
    """Scattergeo of the JPSS sub-satellite point + Starlink ground hits."""
    fig = go.Figure()

    jpss_df = _load_jpss_trajectory()
    if jpss_df is not None and not jpss_df.empty:
        # Pick nearest sample.
        target = pd.Timestamp(time_plot)
        diffs = (jpss_df["times"] - target).abs()
        idx = diffs.idxmin()
        row = jpss_df.loc[idx]
        # Approximate sub-satellite point as the observer (Westford); in real
        # life we'd ECEF-trace, but for booth visual we mark "JPSS pass" at
        # observer's location (acceptable since this is Phase 1 cartoon).
        fig.add_trace(
            go.Scattergeo(
                lat=[42.61],
                lon=[-71.49],
                text=["JPSS / Suomi-NPP FOV (Westford)"],
                mode="markers+text",
                marker=dict(size=18, color="#1f77b4", symbol="circle"),
                name="FOV center",
            )
        )

    if show_gateway:
        fig.add_trace(
            go.Scattergeo(
                lat=[gateway_lat],
                lon=[gateway_lon],
                text=["Starlink gateway"],
                mode="markers+text",
                marker=dict(size=14, color="#d62728", symbol="triangle-up"),
                name="Starlink gateway",
            )
        )

    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(50, 50, 50)",
            showocean=True,
            oceancolor="rgb(20, 20, 30)",
            showcountries=True,
            countrycolor="rgb(80, 80, 80)",
            projection_type="natural earth",
        ),
        title=f"{band_key} weather-sat FOV at {time_plot.isoformat(timespec='seconds')}",
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def render() -> None:
    for key, value in _DEMO_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    st.markdown(
        """
        ### Weather satellite single FOV (Looking-Down)
        JPSS / Suomi-NPP looking down at Earth, measuring **brightness
        temperature (K)** with K-Band (23.8 GHz) or V-Band (50.3 GHz) channels.
        Optional Starlink back-lobe interference contributions are shown
        explicitly in the Tb budget (Phase 1 of `tuto_radiomdl_weather_phase1.py`).
        """
    )

    have_jpss = data_file_exists(_JPSS_FILE)
    have_starlink = data_file_exists(_STARLINK_FILE)
    have_k = data_file_exists(_K_BAND_CSV)
    have_v = data_file_exists(_V_BAND_CSV)

    if not (have_k or have_v):
        st.error(
            f"Missing antenna pattern CSVs in `research_tutorials/data/`. "
            f"Expected: `{_K_BAND_CSV}` and/or `{_V_BAND_CSV}`."
        )
        return

    if not have_jpss or not have_starlink:
        st.warning(
            "JPSS or Starlink trajectory file is missing for the 2025-11-01 window. "
            "The Earth map will fall back to a static view; the Tb budget will use "
            "the bundled antennas only."
        )

    # ------------------------------------------------------------------
    # Layout: in-tab controls column on the left, main panel on the right.
    # Same pattern as Tab 1 so booth visitors learn the layout once.
    # ------------------------------------------------------------------
    controls_col, main_col = st.columns([1, 3])

    with controls_col:
        st.markdown("#### Controls")

        if st.button("Reset", key="wx_reset", width="stretch"):
            _reset_to_defaults()
            st.rerun()

        with st.expander("Sensor", expanded=True):
            band_options = [
                k for k, v in _BANDS.items() if data_file_exists(v["csv"])
            ]
            if not band_options:
                st.error("No antenna pattern CSVs available.")
                return
            if st.session_state.wx_band not in band_options:
                st.session_state.wx_band = band_options[0]
            st.selectbox("Band", band_options, key="wx_band")

        with st.expander("Time", expanded=True):
            obs_minutes = (_OBS_END - _OBS_START).total_seconds() / 60.0
            st.slider(
                "Minutes from start of overpass",
                0.0,
                float(obs_minutes),
                key="wx_time_offset_min",
                step=0.25,
            )

        with st.expander("Starlink interference", expanded=True):
            st.slider(
                "Starlink fundamental freq [GHz]",
                10.7,
                12.7,
                key="wx_starlink_freq_ghz",
                step=0.05,
                help="Pick 11.9 GHz to put the 2nd harmonic into K-band, "
                "12.575 GHz for the 4th into V-band.",
            )
            st.slider(
                "Starlink EIRP [dBW]",
                -50.0,
                20.0,
                key="wx_eirp_dbw",
                step=0.5,
            )
            st.slider(
                "Visible Starlinks (cap)",
                0,
                200,
                key="wx_n_starlinks",
                step=1,
            )
            st.slider(
                "Polarization loss [dB]",
                0.0,
                6.0,
                key="wx_pol_loss_db",
                step=0.1,
            )

        with st.expander("Starlink gateway (Phase 2 hint)", expanded=False):
            st.toggle("Show gateway on map", key="wx_show_gateway")
            st.slider(
                "Gateway lat",
                -60.0,
                60.0,
                key="wx_gateway_lat",
                step=0.5,
            )
            st.slider(
                "Gateway lon",
                -180.0,
                180.0,
                key="wx_gateway_lon",
                step=1.0,
            )

    main = main_col

    time_plot = _OBS_START + timedelta(
        minutes=float(st.session_state.wx_time_offset_min)
    )
    main.caption(f"t = {time_plot.isoformat(timespec='seconds')}")

    # ------------------------------------------------------------------
    # Compute and render
    # ------------------------------------------------------------------
    with main:
        budget = _cached_tb_budget(
            st.session_state.wx_band,
            time_plot.isoformat(),
            float(st.session_state.wx_starlink_freq_ghz) * 1e9,
            float(st.session_state.wx_eirp_dbw),
            int(st.session_state.wx_n_starlinks),
            float(st.session_state.wx_pol_loss_db),
        )

    if budget.get("missing_data"):
        main.error("Antenna pattern file for the selected band is missing.")
        return

    metric_cols = main.columns(4)
    metric_cols[0].metric("Channel", f"{budget['freq_ghz']:.1f} GHz")
    metric_cols[1].metric("Bandwidth", f"{budget['bandwidth_mhz']:.0f} MHz")
    metric_cols[2].metric("Visible Starlinks", f"{budget.get('n_visible', 0)}")
    metric_cols[3].metric("Starlink Tb [K]", f"{budget['starlink_k']:.2f}")

    left, right = main.columns([1, 1])
    with left:
        st.markdown("**Effective Tb contributions at FOV**")
        st.plotly_chart(_stacked_budget_figure(budget), width="stretch")

    with right:
        st.markdown("**Antenna pattern**")
        fig = _antenna_pattern_figure(st.session_state.wx_band)
        if fig is not None:
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Antenna pattern unavailable.")

    with main:
        st.markdown("---")
        st.markdown("**FOV ground footprint (Earth view)**")
        geo_fig = _fov_map_figure(
            time_plot,
            st.session_state.wx_band,
            bool(st.session_state.wx_show_gateway),
            float(st.session_state.wx_gateway_lat),
            float(st.session_state.wx_gateway_lon),
        )
        st.plotly_chart(geo_fig, width="stretch")

        st.caption(
            "Phase 1 model: Starlink back-lobe interference + Earth Tb + sky background "
            "+ system noise. Phase 2 (5G ground emitters) and Phase 3 (full multipath) "
            "require the GHS-POP raster and ITU cloud/rain grids; see `research_tutorials/"
            "README_RFI_modeling_for_NWP_simulation.md` to enable them."
        )
