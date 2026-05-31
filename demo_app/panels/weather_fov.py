"""
Tab 2: Weather satellite single FOV (Looking-Down, Phase 2 demo).

Live version of ``research_tutorials/tuto_radiomdl_weather_phase2.py`` with demo
performance limits: 10 s time grid, no DEM, Starlink elev filter (0°),
atmospheric loss disabled, and NWP-style equivalent 5G emitter at FOV center.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import sim_cache  # noqa: F401
from sim_cache import data_file_exists
import weather_phase2_loaders as wx

_GALLERY_DIR = Path(__file__).resolve().parents[1] / "assets" / "gallery"

_DEMO_DEFAULTS = {
    "wx_starlink_freq_ghz": 11.9,
    "wx_eirp_dbw": -15 + 10 * np.log10(300),
    "wx_time_offset_min": 1.0,
    "wx_5g_freq_ghz": 25.15,
    "wx_5g_eirp_dbw": 30.0,
    "wx_emitter_density": 1.0,
}


def _reset_to_defaults() -> None:
    for key, value in _DEMO_DEFAULTS.items():
        st.session_state[key] = value


def _gallery_image(name: str) -> Optional[Path]:
    path = _GALLERY_DIR / name
    return path if path.is_file() else None


def _time_index(offset_min: float, n_points: int) -> int:
    idx = int(round(offset_min * 60.0 / wx.DEMO_TIME_STEP_S))
    return int(min(max(0, idx), max(0, n_points - 1)))


def _peak_rfi_dbw(series: dict) -> dict:
    """Peak (strongest) RFI in dBW over the full cached overpass series."""
    return {
        "k_starlink": float(np.max(series["k_starlink_dbw"])),
        "v_starlink": float(np.max(series["v_starlink_dbw"])),
        "k_5g": float(np.max(series["k_5g_dbw"])),
        "v_5g": float(np.max(series["v_5g_dbw"])),
    }


def _render_peak_rfi_table(container, series: dict) -> None:
    peaks = _peak_rfi_dbw(series)
    t0 = wx.OBS_START.strftime("%H:%M")
    t1 = wx.OBS_END.strftime("%H:%M")
    container.markdown(f"**Peak RFI over overpass ({t0}–{t1} UTC)**")
    container.markdown(
        f"""
| Source | Max K-Band (23.8 GHz) [dBW] | Max V-Band (50.3 GHz) [dBW] |
| --- | ---: | ---: |
| Starlink | {peaks['k_starlink']:.1f} | {peaks['v_starlink']:.1f} |
| 5G | {peaks['k_5g']:.1f} | {peaks['v_5g']:.1f} |
        """
    )
    container.caption(
        "Peak = maximum over the overpass (10 s grid). Updates when Starlink or "
        "5G sliders change. Metrics above are at the selected time. "
        f"Negligible RFI (zero power) is shown as {wx.NEGLIGIBLE_RFI_DBW:.0f} dBW "
        "instead of −1000 dBW for readability."
    )


def _rfi_time_series_figure(
    minutes: np.ndarray,
    k_starlink: np.ndarray,
    k_5g: np.ndarray,
    v_starlink: np.ndarray,
    v_5g: np.ndarray,
    selected_minutes: Optional[float] = None,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("K-Band (23.8 GHz)", "V-Band (50.3 GHz)"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=minutes,
            y=k_starlink,
            name="Starlink backlobe",
            mode="lines",
            line=dict(color="#1f77b4", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=minutes,
            y=k_5g,
            name="5G ground emitters",
            mode="lines",
            line=dict(color="#9467bd", width=2, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=minutes,
            y=v_starlink,
            name="Starlink backlobe",
            mode="lines",
            line=dict(color="#1f77b4", width=2, dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=minutes,
            y=v_5g,
            name="5G ground emitters",
            mode="lines",
            line=dict(color="#9467bd", width=2, dash="dot"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    if selected_minutes is not None:
        fig.add_vline(
            x=selected_minutes,
            line=dict(color="white", width=1),
            opacity=0.6,
        )
    fig.update_xaxes(title_text="Minutes from start of overpass", row=2, col=1)
    fig.update_yaxes(title_text="RFI power [dBW]", row=1, col=1)
    fig.update_yaxes(title_text="RFI power [dBW]", row=2, col=1)
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    return fig


def _fov_map_figure(time_plot: datetime) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=[wx.TARGET_LAT],
            lon=[wx.TARGET_LON],
            text=["Westford FOV center (32 km)"],
            mode="markers+text",
            marker=dict(size=18, color="#1f77b4", symbol="circle"),
            name="FOV center",
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
            center=dict(lat=wx.TARGET_LAT, lon=wx.TARGET_LON),
            projection_scale=8,
        ),
        title=f"FOV at {time_plot.isoformat(timespec='seconds')}",
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _render_antenna_patterns_expander(container) -> None:
    with container.expander("Antenna patterns", expanded=False):
        sl = _gallery_image("starlink_antenna_pattern.png")
        g5 = _gallery_image("ground_emitter_5g_antenna_pattern.png")
        ws = _gallery_image("weather_sat_antenna_patterns.png")
        if sl is None and g5 is None and ws is None:
            st.info(
                "Place pre-rendered PNGs in `demo_app/assets/gallery/`: "
                "`starlink_antenna_pattern.png`, `ground_emitter_5g_antenna_pattern.png`, "
                "`weather_sat_antenna_patterns.png`."
            )
            return
        top_l, top_r = st.columns(2)
        if sl is not None:
            top_l.image(str(sl), use_container_width=True)
        else:
            top_l.info("Missing `starlink_antenna_pattern.png`.")
        if g5 is not None:
            top_r.image(str(g5), use_container_width=True)
        else:
            top_r.info("Missing `ground_emitter_5g_antenna_pattern.png`.")
        if ws is not None:
            st.image(str(ws), use_container_width=True)
        else:
            st.info("Missing `weather_sat_antenna_patterns.png`.")


def _render_satellite_positions_expander(container) -> None:
    with container.expander("Satellite positions", expanded=False):
        img = _gallery_image("satellite_positions.png")
        if img is not None:
            st.image(str(img), use_container_width=True)
        else:
            st.info(
                "Missing `satellite_positions.png` in `demo_app/assets/gallery/`."
            )


def render() -> None:
    for key, value in _DEMO_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    st.markdown(
        """
        ### Weather satellite single FOV (Looking-Down)
        Suomi-NPP / JPSS ATMS **K-Band (23.8 GHz)** and **V-Band (50.3 GHz)** RFI
        from Starlink back/side lobes and **5G mmWave** ground emitters over a single FOV circle of 32 km diameter at Westford for demo purposes. The 2nd harmonic power factor is set to be **0.01** (−20 dBc) for both Starlink and 5G and the 4th harmonic power factor is set to be **0.001** (−30 dBc) for Starlink to V-band. Polarization loss is set to be **3 dB** for Starlink. RFI power of -500 dBW is shown as negligible for readability.
        """
    )

    have_jpss = data_file_exists(wx.JPSS_FILE)
    have_starlink = data_file_exists(wx.STARLINK_FILE)
    have_k = data_file_exists(wx.K_BAND_CSV)
    have_v = data_file_exists(wx.V_BAND_CSV)

    if not (have_k and have_v):
        st.error(
            "Missing K-Band and/or V-Band antenna CSVs in "
            "`research_tutorials/data/`."
        )
        return

    if not have_jpss or not have_starlink:
        st.warning(
            "JPSS or Starlink trajectory file is missing for the 2025-11-01 "
            "window. RFI time series may be empty."
        )

    controls_col, main_col = st.columns([1, 3])

    with controls_col:
        st.markdown("#### Controls")

        if st.button("Reset", key="wx_reset", width="stretch"):
            _reset_to_defaults()
            st.rerun()

        with st.expander("Time", expanded=True):
            obs_minutes = (wx.OBS_END - wx.OBS_START).total_seconds() / 60.0
            st.slider(
                "Minutes from start of overpass",
                0.0,
                float(obs_minutes),
                key="wx_time_offset_min",
                step=0.25,
                help="Moves the marker on the time series; does not recompute.",
            )

        with st.expander("Starlink interference", expanded=True):
            st.slider(
                "Starlink fundamental freq [GHz]",
                10.7,
                12.7,
                key="wx_starlink_freq_ghz",
                step=0.005,
                format="%.3f",
                help="11.9 GHz → 2nd harmonic in K-band; 12.575 GHz → 4th in V-band.",
            )
            st.slider(
                "Starlink EIRP [dBW]",
                -50.0,
                20.0,
                key="wx_eirp_dbw",
                step=0.5,
            )

        with st.expander("5G interference", expanded=True):
            st.slider(
                "5G fundamental freq [GHz]",
                23.8,
                50.3,
                key="wx_5g_freq_ghz",
                step=0.05,
                help=(
                    "Demo what-if range aligned with ATMS K-band (23.8 GHz) and "
                    "V-band (50.3 GHz). Default 25.15 GHz (mmWave n258; 2nd "
                    "harmonic → V-band)."
                ),
            )
            st.slider(
                "5G EIRP [dBW]",
                -8.5,
                40.0,
                key="wx_5g_eirp_dbw",
                step=0.5,
            )
            st.slider(
                "5G emitter density [emitter/km²]",
                1.0,
                50.0,
                key="wx_emitter_density",
                step=1.0,
            )

    main = main_col
    time_plot = wx.OBS_START + timedelta(
        minutes=float(st.session_state.wx_time_offset_min)
    )
    main.caption(f"t = {time_plot.isoformat(timespec='seconds')} UTC")

    series = wx.compute_rfi_time_series(
        float(st.session_state.wx_starlink_freq_ghz),
        float(st.session_state.wx_eirp_dbw),
        float(st.session_state.wx_5g_freq_ghz),
        float(st.session_state.wx_5g_eirp_dbw),
        float(st.session_state.wx_emitter_density),
    )

    if series.get("missing_data"):
        main.error("Could not load weather satellite antenna patterns.")
        return

    n_pts = len(series["minutes"])
    idx = _time_index(float(st.session_state.wx_time_offset_min), n_pts)
    selected_min = float(series["minutes"][idx])
    obs_time = datetime.fromisoformat(series["obs_times_iso"][idx])
    n_sl = wx.starlink_count_at_time(obs_time)

    main.markdown("**Starlink**")
    sl_cols = main.columns(4)
    sl_cols[0].metric("Visible Starlinks", f"{n_sl}")
    sl_cols[1].metric("Fundamental", f"{st.session_state.wx_starlink_freq_ghz:.3f} GHz")
    sl_cols[2].metric("RFI K-Band [dBW]", f"{series['k_starlink_dbw'][idx]:.1f}")
    sl_cols[3].metric("RFI V-Band [dBW]", f"{series['v_starlink_dbw'][idx]:.1f}")

    main.markdown("**5G cellular network (mmWave)**")
    fg_cols = main.columns(5)
    fg_cols[0].metric("Emitters in FOV", f"{series['n_emitters']}")
    fg_cols[1].metric("Fundamental", f"{st.session_state.wx_5g_freq_ghz:.2f} GHz")
    fg_cols[2].metric("EIRP", f"{st.session_state.wx_5g_eirp_dbw:.1f} dBW")
    fg_cols[3].metric("RFI K-Band [dBW]", f"{series['k_5g_dbw'][idx]:.1f}")
    fg_cols[4].metric("RFI V-Band [dBW]", f"{series['v_5g_dbw'][idx]:.1f}")

    main.caption(
        "ATMS: K 23.8 GHz / 270 MHz, V 50.3 GHz / 180 MHz. "
        "5G uses equivalent emitter at FOV center × emitter count."
    )

    _render_peak_rfi_table(main, series)

    main.markdown("**RFI power vs time**")
    ts_fig = _rfi_time_series_figure(
        series["minutes"],
        series["k_starlink_dbw"],
        series["k_5g_dbw"],
        series["v_starlink_dbw"],
        series["v_5g_dbw"],
        selected_minutes=selected_min,
    )
    main.plotly_chart(ts_fig, use_container_width=True)
    main.caption(
        "Demo simplifications: 10 s time grid, no DEM terrain masking, "
        "Starlink elevation > 0° (no DTC; bundled traj ~5° pass mask), "
        "atmospheric loss disabled, "
        "5G = one representative link budget at FOV center scaled by "
        f"n_emitters (density × π×16² km²). OOBE not modeled."
    )

    _render_antenna_patterns_expander(main)
    _render_satellite_positions_expander(main)

    with main:
        st.markdown("**FOV ground footprint (Earth view)**")
        st.plotly_chart(_fov_map_figure(time_plot), use_container_width=True)
