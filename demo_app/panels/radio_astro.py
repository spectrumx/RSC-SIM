"""
Tab 1: Starlink vs. radio telescope.

Live, fully offline RFI sandbox built around the bundled Cas A / Starlink
trajectories. Mirrors the call paths in
`educational_tutorials/02_satellite_interference.py` and `..._03_sky_mapping.py`,
exposed through Streamlit sliders.

This module imports `sim_cache` as a top-level module: `app.py` puts
`demo_app/` on `sys.path` before loading any panel.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sim_cache  # noqa: F401  (registers `src/` on sys.path)
from sim_cache import (
    RadioState,
    compute_observation_result,
    list_starlink_satellite_names,
    load_cas_a_trajectory,
    load_starlink_trajectory_df,
    make_constellation,
    make_observation,
    make_sky_model,
)

# RSC-SIM helpers -- `sim_cache` already added `src/` to `sys.path`.
from radio_types import Observation, Trajectory  # noqa: E402
from astro_mdl import temperature_to_power  # noqa: E402
from obs_mdl import model_observed_temp  # noqa: E402

from shared.config import (  # type: ignore  # noqa: E402
    BANDWIDTH,
    CENTER_FREQUENCY,
    OBSERVATION_END,
    OBSERVATION_START,
    RECEIVER_TEMP,
    TELESCOPE_COORDS,
    TELESCOPE_FREQ_BAND,
)


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------


_DEMO_DEFAULTS = {
    "ra_constellation": True,
    "ra_beam_avoid_deg": 0.0,
    "ra_n_sats": 0,  # 0 = all satellites
    "ra_direct_sat": "(all satellites)",
    "ra_center_freq_ghz": CENTER_FREQUENCY / 1e9,
    "ra_bandwidth_khz": BANDWIDTH / 1e3,
    "ra_min_elevation": 5.0,
    "ra_time_offset_min": 5.0,  # minutes from observation start
    "ra_skymap_step": 10,
}


def _reset_to_defaults() -> None:
    for key, value in _DEMO_DEFAULTS.items():
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Sky-map computation (small grid for live responsiveness)
# ---------------------------------------------------------------------------


def _compute_sky_map(
    state: RadioState,
    time_plot: datetime,
    az_step_deg: int,
    el_step_deg: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a sky temperature map using `model_observed_temp` over a coarse
    az/el grid. Mirrors the loop in `educational_tutorials/03_sky_mapping.py`,
    but with a smaller grid for sub-second response.
    """
    obs, _ = make_observation(state)
    constellation = make_constellation(state, obs)
    sky_mdl = make_sky_model(obs, include_source=True)

    az_grid = np.arange(0, 360, az_step_deg)
    el_grid = np.arange(5, 91, el_step_deg)
    map_grid = np.zeros((len(el_grid), len(az_grid)))

    instrument = obs.get_instrument()

    for i, el in enumerate(el_grid):
        for j, az in enumerate(az_grid):
            point_df = pd.DataFrame(
                {
                    "times": [time_plot],
                    "azimuths": [float(az)],
                    "elevations": [float(el)],
                    "distances": [np.inf],
                }
            )
            point_obs = Observation.from_dates(
                time_plot, time_plot, Trajectory(point_df), instrument
            )
            result = model_observed_temp(point_obs, sky_mdl, constellation)
            map_grid[i, j] = result[0, 0, 0]

    return map_grid, az_grid, el_grid


@st.cache_data(show_spinner="Building sky map...", max_entries=12)
def _cached_sky_map(
    state: RadioState,
    time_iso: str,
    az_step_deg: int,
    el_step_deg: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Streamlit-cacheable wrapper around `_compute_sky_map`."""
    time_plot = datetime.fromisoformat(time_iso)
    return _compute_sky_map(state, time_plot, az_step_deg, el_step_deg)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _polar_sky_map_figure(
    map_grid: np.ndarray,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    sat_az: Optional[np.ndarray] = None,
    sat_el: Optional[np.ndarray] = None,
    src_az: Optional[float] = None,
    src_el: Optional[float] = None,
    pointing_az: Optional[float] = None,
    pointing_el: Optional[float] = None,
):
    """
    Render a polar heatmap of sky temperature in dB(W) with optional satellite
    and source markers. Uses matplotlib (cleanly polar) -- Plotly's polar
    heatmaps are awkward.
    """
    power_grid = temperature_to_power(map_grid, BANDWIDTH)
    z_db = 10 * np.log10(np.clip(power_grid, 1e-30, None))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    pc = ax.pcolormesh(
        np.radians(az_grid),
        90 - el_grid,
        z_db,
        cmap="plasma",
        shading="auto",
    )
    cbar = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.08)
    cbar.set_label("Power [dBW]")

    ax.set_yticks(range(0, 91, 30))
    ax.set_yticklabels(["zenith", "60°", "30°", "horizon"])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    if sat_az is not None and sat_el is not None and len(sat_az) > 0:
        ax.scatter(
            np.radians(sat_az),
            90 - np.asarray(sat_el),
            s=40,
            c="red",
            marker="x",
            label="Satellites",
            zorder=5,
        )
    if src_az is not None and src_el is not None:
        ax.scatter(
            np.radians([src_az]),
            90 - np.asarray([src_el]),
            s=180,
            c="gold",
            marker="*",
            edgecolor="black",
            linewidth=0.5,
            label="Cas A",
            zorder=6,
        )
    if pointing_az is not None and pointing_el is not None:
        ax.scatter(
            np.radians([pointing_az]),
            90 - np.asarray([pointing_el]),
            s=80,
            facecolor="none",
            edgecolor="white",
            linewidth=2,
            label="Pointing",
            zorder=7,
        )
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
    fig.tight_layout()
    return fig


def _tb_to_dbw(tb_k: np.ndarray, bandwidth_hz: float) -> np.ndarray:
    """
    Convert brightness temperature (K) to received power (dBW). RSC-SIM's
    Looking-Up tutorials all report dBW, so the radio-astronomy panel matches
    that convention for direct comparison with the educational tutorials.
    """
    power_w = temperature_to_power(np.asarray(tb_k), bandwidth_hz)
    return 10.0 * np.log10(np.clip(power_w, 1e-300, None))


def _time_series_figure(
    times: np.ndarray,
    p_no_sat_dbw: np.ndarray,
    p_with_sat_dbw: np.ndarray,
    p_with_avoid_dbw: Optional[np.ndarray],
    threshold_dbw: float,
    selected_time: Optional[datetime] = None,
) -> go.Figure:
    """Plotly time series of received power scenarios (dBW)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=p_no_sat_dbw,
            name="No satellites",
            mode="lines",
            line=dict(color="#2ca02c", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=p_with_sat_dbw,
            name="With Starlink",
            mode="lines",
            line=dict(color="#1f77b4", width=2),
        )
    )
    if p_with_avoid_dbw is not None:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=p_with_avoid_dbw,
                name="Beam avoidance",
                mode="lines",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
            )
        )

    fig.add_hline(
        y=threshold_dbw,
        line=dict(color="red", width=1, dash="dash"),
        annotation_text=f"Threshold {threshold_dbw:.1f} dBW",
        annotation_position="top right",
    )
    if selected_time is not None:
        fig.add_vline(
            x=selected_time,
            line=dict(color="white", width=1),
            opacity=0.6,
        )

    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Received power [dBW]",
        hovermode="x unified",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers for satellite positions / source position at a given time
# ---------------------------------------------------------------------------


def _satellite_positions_at_time(
    state: RadioState, time_plot: datetime
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Az/el/names for satellites visible at `time_plot` (nearest stamp)."""
    df = load_starlink_trajectory_df()
    if df.empty:
        return np.array([]), np.array([]), []

    target = pd.Timestamp(time_plot)
    diffs = (df["times"] - target).abs()
    if diffs.empty:
        return np.array([]), np.array([]), []
    nearest_time = df.loc[diffs.idxmin(), "times"]
    df_t = df[df["times"] == nearest_time]

    df_t = df_t[~df_t["sat"].astype(str).str.contains("DTC")]
    df_t = df_t[df_t["elevations"] > 20.0]

    if state.direct_satellite:
        df_t = df_t[df_t["sat"] == state.direct_satellite]
    elif state.n_satellites > 0:
        keep = sorted(df_t["sat"].unique())[: state.n_satellites]
        df_t = df_t[df_t["sat"].isin(keep)]

    return (
        df_t["azimuths"].to_numpy(),
        df_t["elevations"].to_numpy(),
        df_t["sat"].tolist(),
    )


def _source_position_at_time(time_plot: datetime) -> Tuple[Optional[float], Optional[float]]:
    traj = load_cas_a_trajectory().get_traj()
    diffs = (traj["times"] - pd.Timestamp(time_plot)).abs()
    if diffs.empty:
        return None, None
    row = traj.loc[diffs.idxmin()]
    return float(row["azimuths"]), float(row["elevations"])


def _ground_track_dataframe(state: RadioState, max_points: int = 4000) -> pd.DataFrame:
    """
    Approximate Starlink lat/lon ground tracks from azimuth/elevation/distance
    relative to Westford. The bundled trajectory's elevation cuts make the
    AER -> ENU -> ECEF -> WGS84 path good enough for a booth visual.
    """
    try:
        import pyproj
    except ImportError:
        return pd.DataFrame(columns=["sat", "lat", "lon"])

    df = load_starlink_trajectory_df()
    if df.empty:
        return pd.DataFrame(columns=["sat", "lat", "lon"])

    df = df[~df["sat"].astype(str).str.contains("DTC")]
    df = df[df["elevations"] > 5.0]

    if state.direct_satellite:
        df = df[df["sat"] == state.direct_satellite]
    elif state.n_satellites > 0:
        keep = sorted(df["sat"].unique())[: state.n_satellites]
        df = df[df["sat"].isin(keep)]

    if df.empty:
        return pd.DataFrame(columns=["sat", "lat", "lon"])

    if len(df) > max_points:
        df = df.iloc[:: max(1, len(df) // max_points)]

    az = np.radians(df["azimuths"].to_numpy())
    zen = np.radians(90.0 - df["elevations"].to_numpy())
    dist = df["distances"].to_numpy()

    e = dist * np.sin(zen) * np.sin(az)
    n = dist * np.sin(zen) * np.cos(az)
    u = dist * np.cos(zen)

    obs_lat, obs_lon, obs_alt = TELESCOPE_COORDS
    lat_rad = np.radians(obs_lat)
    lon_rad = np.radians(obs_lon)

    R = np.array(
        [
            [-np.sin(lon_rad), -np.sin(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.cos(lon_rad)],
            [np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad) * np.sin(lon_rad)],
            [0, np.cos(lat_rad), np.sin(lat_rad)],
        ]
    )
    enu = np.stack([e, n, u])
    transformer_to_ecef = pyproj.Transformer.from_crs(4979, 4978, always_xy=True)
    ecef_origin = transformer_to_ecef.transform(obs_lon, obs_lat, obs_alt)
    ecef = (R @ enu).T + np.array(ecef_origin)

    transformer_to_wgs = pyproj.Transformer.from_crs(4978, 4326, always_xy=True)
    lon, lat, _ = transformer_to_wgs.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return pd.DataFrame({"sat": df["sat"].to_numpy(), "lat": lat, "lon": lon})


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------


def _build_state() -> RadioState:
    return RadioState(
        center_freq_hz=float(st.session_state.ra_center_freq_ghz) * 1e9,
        bandwidth_hz=float(st.session_state.ra_bandwidth_khz) * 1e3,
        n_channels=1,
        receiver_temp_k=RECEIVER_TEMP,
        beam_avoid_deg=float(st.session_state.ra_beam_avoid_deg),
        min_elevation_deg=float(st.session_state.ra_min_elevation),
        n_satellites=int(st.session_state.ra_n_sats),
        direct_satellite=(
            None
            if st.session_state.ra_direct_sat == "(all satellites)"
            else st.session_state.ra_direct_sat
        ),
        constellation_enabled=bool(st.session_state.ra_constellation),
        obs_start_iso=OBSERVATION_START.isoformat(),
        obs_end_iso=OBSERVATION_END.isoformat(),
    )


def _state_with(state: RadioState, **overrides) -> RadioState:
    return RadioState(**{**state.__dict__, **overrides})


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the radio-astronomy headline tab."""

    for key, value in _DEMO_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    st.markdown(
        """
        ### Starlink vs. radio telescope (Looking-Up)
        Live RFI sandbox using the bundled **Westford / Cas A / Starlink** data
        (2025-02-18). Toggle the constellation, dial in beam avoidance, or pick
        a single satellite for forensic mode. Every plot updates live.

        Outputs are in **received power (dBW)**, matching RSC-SIM's Looking-Up
        tutorials. Brightness temperature is the natural unit for the
        Looking-Down weather-satellite tab.
        """
    )

    # ------------------------------------------------------------------
    # Layout: controls on the left (in-tab), main panel on the right.
    # Keeping the controls inside the tab means switching tabs swaps the
    # whole control set, so a colleague running the booth doesn't have to
    # mentally filter "which tab does this slider belong to?".
    # ------------------------------------------------------------------
    controls_col, main_col = st.columns([1, 3])

    with controls_col:
        st.markdown("#### Controls")

        if st.button("Reset", key="ra_reset", width="stretch"):
            _reset_to_defaults()
            st.rerun()

        with st.expander("Time", expanded=True):
            obs_minutes = (OBSERVATION_END - OBSERVATION_START).total_seconds() / 60.0
            st.slider(
                "Minutes from start of observation",
                0.0,
                float(obs_minutes),
                key="ra_time_offset_min",
                step=0.25,
            )
            time_plot = OBSERVATION_START + pd.Timedelta(
                minutes=float(st.session_state.ra_time_offset_min)
            )
            st.caption(f"t = {time_plot.isoformat(timespec='seconds')}")

        with st.expander("Telescope", expanded=False):
            f_lo, f_hi = TELESCOPE_FREQ_BAND
            st.slider(
                "Center frequency [GHz]",
                float(f_lo / 1e9),
                float(f_hi / 1e9),
                key="ra_center_freq_ghz",
                step=0.05,
            )
            st.slider(
                "Bandwidth [kHz]",
                0.1,
                500.0,
                key="ra_bandwidth_khz",
                step=0.1,
            )
            st.slider(
                "Min elevation [deg]",
                0.0,
                30.0,
                key="ra_min_elevation",
                step=1.0,
            )

        with st.expander("Satellites", expanded=True):
            st.toggle("Include Starlink constellation", key="ra_constellation")
            st.slider(
                "Beam avoidance [deg] (0 = off)",
                0.0,
                20.0,
                key="ra_beam_avoid_deg",
                step=1.0,
            )
            sat_names = list_starlink_satellite_names()
            n_sats_max = max(1, min(60, len(sat_names)))
            st.slider(
                "Number of satellites (0 = all)",
                0,
                n_sats_max,
                key="ra_n_sats",
                step=1,
            )
            sat_options = ["(all satellites)"] + sat_names
            st.selectbox(
                "Direct mode (one satellite)",
                sat_options,
                key="ra_direct_sat",
            )

        with st.expander("Sky map resolution", expanded=False):
            st.slider(
                "Azimuth step [deg]",
                5,
                30,
                key="ra_skymap_step",
                step=5,
                help="Larger steps = faster, coarser maps.",
            )

    # Everything below renders inside `main_col` so it sits to the right of
    # the in-tab control column.
    main = main_col

    state = _build_state()

    # ------------------------------------------------------------------
    # Run simulations: with-current-state + always-on baselines
    # ------------------------------------------------------------------
    no_sat_state = _state_with(state, constellation_enabled=False, beam_avoid_deg=0.0)
    full_with_sat_state = _state_with(state, constellation_enabled=True, beam_avoid_deg=0.0)
    full_avoid_state = _state_with(state, constellation_enabled=True, beam_avoid_deg=10.0)

    with main, st.spinner("Running observations..."):
        result_no = compute_observation_result(no_sat_state)
        result_with = compute_observation_result(full_with_sat_state)
        result_avoid = compute_observation_result(full_avoid_state)
        result_current = compute_observation_result(state)

    times_ns = np.asarray(result_no["times"], dtype="int64")
    times = times_ns.astype("datetime64[ns]")

    # RSC-SIM's Looking-Up tutorials report received power (dBW). Convert the
    # raw Tb-array results from `model_observed_temp` to power here so every
    # downstream display (metrics, time series, threshold) is in dBW.
    bw_hz = state.bandwidth_hz
    p_no_w = temperature_to_power(np.asarray(result_no["tb_k"]), bw_hz)
    p_with_w = temperature_to_power(np.asarray(result_with["tb_k"]), bw_hz)
    p_avoid_w = temperature_to_power(np.asarray(result_avoid["tb_k"]), bw_hz)
    p_current_w = temperature_to_power(np.asarray(result_current["tb_k"]), bw_hz)

    p_no_dbw = 10.0 * np.log10(np.clip(p_no_w, 1e-300, None))
    p_with_dbw = 10.0 * np.log10(np.clip(p_with_w, 1e-300, None))
    p_avoid_dbw = 10.0 * np.log10(np.clip(p_avoid_w, 1e-300, None))
    p_current_dbw = 10.0 * np.log10(np.clip(p_current_w, 1e-300, None))

    # ------------------------------------------------------------------
    # Headline metrics (Looking-Up case: power in dBW)
    # ------------------------------------------------------------------
    metric_cols = main.columns(4)
    metric_cols[0].metric("Visible satellites", f"{result_current['n_sats']}")
    metric_cols[1].metric("Peak power [dBW]", f"{np.nanmax(p_current_dbw):.1f}")
    interference_db = p_current_dbw - p_no_dbw
    metric_cols[2].metric(
        "Peak excess [dB]",
        f"{np.nanmax(interference_db):.1f}",
        delta=f"{np.nanmean(interference_db):.2f} mean",
    )
    # 5σ threshold computed in linear power, then expressed in dBW.
    threshold_w = float(np.nanmean(p_no_w) + 5.0 * (np.nanstd(p_no_w) + 1e-300))
    threshold_dbw = 10.0 * np.log10(max(threshold_w, 1e-300))
    over = float(np.mean(p_current_w > threshold_w) * 100)
    metric_cols[3].metric("% time over threshold", f"{over:.1f}%")

    # ------------------------------------------------------------------
    # Plots: sky map (left) + time series (right)
    # ------------------------------------------------------------------
    left, right = main.columns([1, 1])

    with left:
        st.markdown("**Sky map at selected time**")
        az_step = int(st.session_state.ra_skymap_step)
        el_step = max(2, az_step // 2)
        try:
            map_grid, az_grid, el_grid = _cached_sky_map(
                state,
                time_plot.isoformat(),
                az_step,
                el_step,
            )
        except Exception as exc:  # pragma: no cover - booth safety net
            st.error(f"Sky map failed: {exc}")
            map_grid = np.zeros((1, 1))
            az_grid = np.array([0])
            el_grid = np.array([0])

        sat_az, sat_el, _ = _satellite_positions_at_time(state, time_plot)
        src_az, src_el = _source_position_at_time(time_plot)

        pointing_az = pointing_el = None
        traj = load_cas_a_trajectory().get_traj()
        if not traj.empty:
            diffs = (traj["times"] - pd.Timestamp(time_plot)).abs()
            row = traj.loc[diffs.idxmin()]
            pointing_az = float(row["azimuths"])
            pointing_el = float(row["elevations"])

        fig = _polar_sky_map_figure(
            map_grid,
            az_grid,
            el_grid,
            sat_az=sat_az,
            sat_el=sat_el,
            src_az=src_az,
            src_el=src_el,
            pointing_az=pointing_az,
            pointing_el=pointing_el,
        )
        st.pyplot(fig, clear_figure=True)

    with right:
        st.markdown("**Received power vs time**")
        ts_fig = _time_series_figure(
            times,
            p_no_dbw,
            p_with_dbw,
            p_avoid_dbw,
            threshold_dbw,
            selected_time=time_plot,
        )
        st.plotly_chart(ts_fig, width="stretch")
        st.caption(
            "Power at the receiver in dBW (Looking-Up convention). "
            "Green = clean observation; blue = Starlink on; orange dotted = "
            "10° beam avoidance; red dashed = 5σ threshold above clean baseline."
        )

    # ------------------------------------------------------------------
    # Earth map of ground tracks
    # ------------------------------------------------------------------
    with main.expander("Satellite ground tracks (Earth view)", expanded=False):
        track_df = _ground_track_dataframe(state)
        if track_df.empty:
            st.info("No satellite tracks to plot for the current selection.")
        else:
            try:
                import pydeck as pdk

                view_state = pdk.ViewState(
                    latitude=TELESCOPE_COORDS[0],
                    longitude=TELESCOPE_COORDS[1],
                    zoom=2.5,
                    pitch=0,
                )
                layers = [
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=track_df.rename(
                            columns={"lon": "longitude", "lat": "latitude"}
                        ),
                        get_position="[longitude, latitude]",
                        get_fill_color="[30, 144, 255, 60]",
                        get_radius=20000,
                        pickable=True,
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=pd.DataFrame(
                            [
                                {
                                    "longitude": TELESCOPE_COORDS[1],
                                    "latitude": TELESCOPE_COORDS[0],
                                    "label": "Westford",
                                }
                            ]
                        ),
                        get_position="[longitude, latitude]",
                        get_fill_color="[255, 165, 0, 220]",
                        get_radius=80000,
                        pickable=True,
                    ),
                ]
                st.pydeck_chart(
                    pdk.Deck(
                        layers=layers,
                        initial_view_state=view_state,
                        map_style=None,
                        tooltip={"text": "{sat}"},
                    )
                )
            except Exception as exc:
                st.warning(f"Earth map unavailable ({exc}); falling back to a 2D plot.")
                fig2 = go.Figure(
                    go.Scattergeo(
                        lat=track_df["lat"],
                        lon=track_df["lon"],
                        text=track_df["sat"],
                        mode="markers",
                        marker=dict(size=4, color="blue", opacity=0.4),
                    )
                )
                fig2.add_trace(
                    go.Scattergeo(
                        lat=[TELESCOPE_COORDS[0]],
                        lon=[TELESCOPE_COORDS[1]],
                        text=["Westford"],
                        mode="markers+text",
                        marker=dict(size=10, color="orange"),
                    )
                )
                fig2.update_layout(geo=dict(showland=True), height=420, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig2, width="stretch")
