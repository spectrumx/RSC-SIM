"""
Matplotlib / Plotly rendering helpers shared across both tabs.

All functions return serializable types (str for base64, dict for Plotly JSON).
"""

from __future__ import annotations

import base64
import io
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

_MPL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def matplotlib_to_base64(fig) -> str:
    """Encode a matplotlib Figure as a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def plotly_fig_to_dict(fig: go.Figure) -> dict:
    """Convert a Plotly Figure to a JSON-serializable dict for State storage."""
    return fig.to_dict()


def _to_serializable(obj):
    """Recursively convert numpy types to Python natives for Reflex State."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.datetime64):
        return str(obj)
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Radio astronomy tab helpers (Looking-Up)
# ---------------------------------------------------------------------------


def polar_sky_map_base64(
    map_grid: np.ndarray,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    bandwidth_hz: float,
    *,
    sat_az: Optional[np.ndarray] = None,
    sat_el: Optional[np.ndarray] = None,
    src_az: Optional[float] = None,
    src_el: Optional[float] = None,
    pointing_az: Optional[float] = None,
    pointing_el: Optional[float] = None,
) -> str:
    """
    Render polar sky temperature heatmap with overlay markers. Returns base64 PNG.

    Thread-safe via _MPL_LOCK — sky map computation is CPU-bound and called from
    background tasks.
    """
    from astro_mdl import temperature_to_power

    power_grid = temperature_to_power(map_grid, bandwidth_hz)
    z_db = 10 * np.log10(np.clip(power_grid, 1e-30, None))

    with _MPL_LOCK:
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

        result = matplotlib_to_base64(fig)
        plt.close(fig)

    return result


def satellite_positions_at_time(
    df: pd.DataFrame,
    time_plot: datetime,
    *,
    direct_satellite: Optional[str] = None,
    n_satellites: int = 0,
    min_elevation: float = 20.0,
):
    """
    Return (az_array, el_array, name_list) for visible satellites at time_plot.

    Nearest-timestamp lookup matching radio_astro.py:375-403.
    """
    if df is None or df.empty:
        return np.array([]), np.array([]), []

    target = pd.Timestamp(time_plot)
    diffs = (df["times"] - target).abs()
    nearest_time = df.loc[diffs.idxmin(), "times"]
    df_t = df[df["times"] == nearest_time].copy()

    df_t = df_t[~df_t["sat"].astype(str).str.contains("DTC")]
    df_t = df_t[df_t["elevations"] > min_elevation]

    if direct_satellite:
        df_t = df_t[df_t["sat"] == direct_satellite]
    elif n_satellites > 0:
        keep = sorted(df_t["sat"].unique())[:n_satellites]
        df_t = df_t[df_t["sat"].isin(keep)]

    return (
        df_t["azimuths"].to_numpy(),
        df_t["elevations"].to_numpy(),
        df_t["sat"].tolist(),
    )


def source_position_at_time(cas_a_traj_df: pd.DataFrame, time_plot: datetime):
    """Return (az, el) for Cas A at nearest timestamp. Returns (None, None) if empty."""
    if cas_a_traj_df is None or cas_a_traj_df.empty:
        return None, None
    diffs = (cas_a_traj_df["times"] - pd.Timestamp(time_plot)).abs()
    row = cas_a_traj_df.loc[diffs.idxmin()]
    return float(row["azimuths"]), float(row["elevations"])


def pointing_position_at_time(pointing_traj_df: pd.DataFrame, time_plot: datetime):
    """Return (az, el) for telescope pointing direction at nearest timestamp."""
    if pointing_traj_df is None or pointing_traj_df.empty:
        return None, None
    diffs = (pointing_traj_df["times"] - pd.Timestamp(time_plot)).abs()
    row = pointing_traj_df.loc[diffs.idxmin()]
    return float(row["azimuths"]), float(row["elevations"])


def ground_track_dataframe(
    starlink_df: pd.DataFrame,
    telescope_coords,
    *,
    direct_satellite: Optional[str] = None,
    n_satellites: int = 0,
    max_points: int = 4000,
) -> pd.DataFrame:
    """
    Convert AER→ENU→ECEF→WGS84 for Starlink ground tracks relative to telescope.

    Port of radio_astro.py:415-471. Returns DataFrame with sat/lat/lon columns.
    Returns empty DataFrame if pyproj is unavailable.
    """
    try:
        import pyproj
    except ImportError:
        return pd.DataFrame(columns=["sat", "lat", "lon"])

    if starlink_df is None or starlink_df.empty:
        return pd.DataFrame(columns=["sat", "lat", "lon"])

    df = starlink_df[~starlink_df["sat"].astype(str).str.contains("DTC")].copy()
    df = df[df["elevations"] > 5.0]

    if direct_satellite:
        df = df[df["sat"] == direct_satellite]
    elif n_satellites > 0:
        keep = sorted(df["sat"].unique())[:n_satellites]
        df = df[df["sat"].isin(keep)]

    if df.empty:
        return pd.DataFrame(columns=["sat", "lat", "lon"])

    if len(df) > max_points:
        df = df.iloc[:: max(1, len(df) // max_points)]

    az = np.radians(df["azimuths"].to_numpy())
    zen = np.radians(90.0 - df["elevations"].to_numpy())
    dist = df["distances"].to_numpy()

    e = dist * np.sin(zen) * np.sin(az)
    n_enu = dist * np.sin(zen) * np.cos(az)
    u = dist * np.cos(zen)

    obs_lat, obs_lon, obs_alt = telescope_coords
    lat_rad = np.radians(obs_lat)
    lon_rad = np.radians(obs_lon)

    R = np.array(
        [
            [-np.sin(lon_rad), -np.sin(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.cos(lon_rad)],
            [np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad) * np.sin(lon_rad)],
            [0.0, np.cos(lat_rad), np.sin(lat_rad)],
        ]
    )
    enu = np.stack([e, n_enu, u])
    transformer_to_ecef = pyproj.Transformer.from_crs(4979, 4978, always_xy=True)
    ecef_origin = transformer_to_ecef.transform(obs_lon, obs_lat, obs_alt)
    ecef = (R @ enu).T + np.array(ecef_origin)

    transformer_to_wgs = pyproj.Transformer.from_crs(4978, 4326, always_xy=True)
    lon, lat, _ = transformer_to_wgs.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return pd.DataFrame({"sat": df["sat"].to_numpy(), "lat": lat, "lon": lon})


def add_off_on_source_regions(fig: go.Figure, obs_start, time_on_source, obs_end) -> None:
    """
    Add gray OFF-source and blue ON-source vrects + annotations to a Plotly figure.

    Port of radio_astro.py:258-302.
    """
    t_start = np.datetime64(pd.Timestamp(obs_start))
    t_on = np.datetime64(pd.Timestamp(time_on_source))
    t_end = np.datetime64(pd.Timestamp(obs_end))
    t_mid_off = t_start + (t_on - t_start) // 2
    t_mid_on = t_on + (t_end - t_on) // 2

    fig.add_vrect(
        x0=t_start, x1=t_on,
        fillcolor="gray", opacity=0.12, layer="below", line_width=0,
    )
    fig.add_vrect(
        x0=t_on, x1=t_end,
        fillcolor="steelblue", opacity=0.08, layer="below", line_width=0,
    )
    fig.add_vline(
        x=t_on,
        line=dict(color="gray", width=1, dash="dash"),
        opacity=0.75,
    )
    fig.add_annotation(
        x=t_mid_off, y=1.0, yref="paper",
        text="OFF-source", showarrow=False,
        font=dict(size=11, color="#aaaaaa"),
    )
    fig.add_annotation(
        x=t_mid_on, y=1.0, yref="paper",
        text="ON-source", showarrow=False,
        font=dict(size=11, color="#aaaaaa"),
    )


def radio_time_series_dict(
    times_ns: list,
    p_no_dbw: list,
    p_with_dbw,
    p_avoid_dbw,
    obs_start,
    time_on_source,
    obs_end,
    *,
    selected_time_ns: Optional[int] = None,
    beam_avoid_deg: Optional[float] = None,
) -> dict:
    """
    Build a Plotly figure dict for the radio astronomy time series.

    Returns a JSON-serializable dict (via fig.to_dict()) ready for rx.plotly().
    """
    times = pd.to_datetime(times_ns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=p_no_dbw,
        name="No satellites", mode="lines",
        line=dict(color="#2ca02c", width=2),
    ))
    if p_with_dbw is not None:
        fig.add_trace(go.Scatter(
            x=times, y=p_with_dbw,
            name="With Starlink", mode="lines",
            line=dict(color="#1f77b4", width=2),
        ))
    if p_avoid_dbw is not None:
        avoid_label = (
            f"Beam avoidance ({beam_avoid_deg:g}°)" if beam_avoid_deg else "Beam avoidance"
        )
        fig.add_trace(go.Scatter(
            x=times, y=p_avoid_dbw,
            name=avoid_label, mode="lines",
            line=dict(color="#ff7f0e", width=2, dash="dot"),
        ))

    add_off_on_source_regions(fig, obs_start, time_on_source, obs_end)

    if selected_time_ns is not None:
        sel_ts = pd.Timestamp(selected_time_ns)
        fig.add_vline(
            x=sel_ts,
            line=dict(color="white", width=1),
            opacity=0.6,
        )

    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Received power [dBW]",
        hovermode="x unified",
        height=380,
        margin=dict(l=10, r=10, t=28, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.08),
    )
    return plotly_fig_to_dict(fig)


# ---------------------------------------------------------------------------
# Weather satellite tab helpers (Looking-Down)
# ---------------------------------------------------------------------------


def weather_time_series_dict(
    minutes: list,
    k_starlink_dbw: list,
    k_5g_dbw: list,
    v_starlink_dbw: list,
    v_5g_dbw: list,
    *,
    selected_minute: Optional[float] = None,
) -> dict:
    """
    Build dual-subplot (K-band, V-band) Plotly figure dict for weather sat tab.

    Returns JSON-serializable dict for rx.plotly().
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["K-Band (23.8 GHz)", "V-Band (50.3 GHz)"],
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=minutes, y=k_starlink_dbw,
        name="Starlink (K)", mode="lines",
        line=dict(color="#1f77b4", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=minutes, y=k_5g_dbw,
        name="5G (K)", mode="lines",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=minutes, y=v_starlink_dbw,
        name="Starlink (V)", mode="lines",
        line=dict(color="#9467bd", width=2),
        showlegend=True,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=minutes, y=v_5g_dbw,
        name="5G (V)", mode="lines",
        line=dict(color="#d62728", width=2, dash="dot"),
        showlegend=True,
    ), row=2, col=1)

    if selected_minute is not None:
        for row in [1, 2]:
            fig.add_vline(
                x=selected_minute,
                line=dict(color="white", width=1),
                opacity=0.6,
                row=row, col=1,
            )

    fig.update_yaxes(title_text="RFI [dBW]", row=1, col=1)
    fig.update_yaxes(title_text="RFI [dBW]", row=2, col=1)
    fig.update_xaxes(title_text="Minutes from observation start", row=2, col=1)
    fig.update_layout(
        height=500,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return plotly_fig_to_dict(fig)
