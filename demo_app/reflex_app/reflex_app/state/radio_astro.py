"""
Reactive state for the Radio Astronomy (Looking-Up) tab.

All fields are JSON-serializable primitives (int/float/str/list/dict/bool).
Non-serializable objects live in state/resources.py _ResourceStore singleton.

Key design decisions:
- Generation counter guards against stale background task results (Safeguard 7)
- @rx.event(background=True, debounce=500) prevents rapid-fire recomputes
- Three-observation pattern (Safeguard 0) produces green/blue/orange traces
- Sky map is lazily computed (separate background task, triggered explicitly)
"""

from __future__ import annotations

from typing import Optional

import reflex as rx

import reflex_app.utils.sim_cache_reflex  # noqa: F401 — sets up sys.path for shared.config

from reflex_app.state.cache import (
    RadioParams,
    WeatherParams,
    _tb_to_dbw,
    compute_observation_cached,
    compute_sky_map_cached,
)
from reflex_app.state.resources import RESOURCES
from reflex_app.utils.plot_helpers import (
    _to_serializable,
    polar_sky_map_base64,
    radio_time_series_dict,
    ground_track_dataframe,
    plotly_fig_to_dict,
)

# Config constants — loaded once at import time (sim_cache_reflex sets sys.path)
from shared.config import (  # type: ignore
    OBSERVATION_START,
    OBSERVATION_END,
    TIME_ON_SOURCE,
    BANDWIDTH,
    CENTER_FREQUENCY,
    TELESCOPE_FREQ_BAND,
    RECEIVER_TEMP,
    FREQUENCY_CHANNELS,
    TELESCOPE_COORDS,
)


_OBS_TOTAL_MINUTES = (OBSERVATION_END - OBSERVATION_START).total_seconds() / 60.0
_OBS_START_ISO = OBSERVATION_START.isoformat()
_OBS_END_ISO = OBSERVATION_END.isoformat()
_FREQ_LO_GHZ = TELESCOPE_FREQ_BAND[0] / 1e9
_FREQ_HI_GHZ = TELESCOPE_FREQ_BAND[1] / 1e9


class RadioAstroState(rx.State):
    """Tab 1 reactive state. All fields must be JSON-serializable."""

    # --- Slider values (primitives) ---
    time_offset_min: float = 5.0
    center_freq_ghz: float = CENTER_FREQUENCY / 1e9
    bandwidth_khz: float = BANDWIDTH / 1e3        # 1.0 kHz (BANDWIDTH = 1e3 Hz)
    n_channels: int = FREQUENCY_CHANNELS
    receiver_temp_k: float = float(RECEIVER_TEMP)
    constellation_enabled: bool = False
    beam_avoid_deg: float = 0.0
    n_sats: int = 0
    direct_sat: str = "(all satellites)"
    min_elevation: float = 5.0
    skymap_step: int = 20                          # Coarser default for Reflex (8s vs 32s)

    # --- Live slider display values (updated on_change, no compute triggered) ---
    display_time_offset_min: float = 5.0
    display_center_freq_ghz: float = CENTER_FREQUENCY / 1e9
    display_bandwidth_khz: float = BANDWIDTH / 1e3
    display_min_elevation: float = 5.0
    display_beam_avoid_deg: float = 0.0
    display_n_sats: int = 0
    display_skymap_step: int = 20

    # --- UI state ---
    loading: bool = False
    sky_map_loading: bool = False
    resources_ready: bool = False
    load_error: str = ""

    # --- Satellite list (populated on load, for dropdown + slider max) ---
    satellite_names: list = []

    # --- Computed results (stored as serializable dicts/lists, NOT @rx.var) ---
    time_series_data: dict = {}   # keys: times_ns, p_no_dbw, p_with_dbw, p_avoid_dbw
    time_series_fig: dict = {}    # Plotly figure dict for rx.plotly()
    sky_map_base64: str = ""      # base64 PNG data URI
    ground_track_fig: dict = {}   # Plotly figure dict for Scattergeo rx.plotly()
    metrics: dict = {}            # visible_sats, peak_power_dbw

    # Incremented on Reset; used as key= on sliders to force remount to default_value
    reset_key: int = 0

    # --- Backend vars (underscore prefix = not sent to frontend) ---
    _generation: int = 0          # Stale result guard (Safeguard 7)
    _sky_map_gen: int = 0

    # ---------------------------------------------------------------------------
    # Computed vars for UI (cheap derived values only)
    # ---------------------------------------------------------------------------

    @rx.var
    def n_sats_max(self) -> int:
        return max(1, min(60, len(self.satellite_names)))

    @rx.var
    def satellite_options(self) -> list:
        return ["(all satellites)"] + self.satellite_names

    @rx.var
    def obs_total_minutes(self) -> float:
        return _OBS_TOTAL_MINUTES

    @rx.var
    def freq_lo_ghz(self) -> float:
        return _FREQ_LO_GHZ

    @rx.var
    def freq_hi_ghz(self) -> float:
        return _FREQ_HI_GHZ

    # ---------------------------------------------------------------------------
    # Resource loading (on_load hook)
    # ---------------------------------------------------------------------------

    @rx.event(background=True)
    async def load_resources(self):
        async with self:
            self.loading = True
            self.load_error = ""
        try:
            RESOURCES.load_all()
            sat_names = RESOURCES.satellite_names
            async with self:
                self.satellite_names = list(sat_names)
                self.resources_ready = True
                self.loading = False
        except Exception as exc:
            async with self:
                self.load_error = f"Resource loading failed: {exc}"
                self.resources_ready = False
                self.loading = False
            return

        # Trigger initial computation once resources are ready
        return RadioAstroState.recompute

    # ---------------------------------------------------------------------------
    # Slider event handlers
    # ---------------------------------------------------------------------------
    # on_change  → updates display var only (fires on every drag tick, cheap)
    # on_value_commit → commits value + triggers recompute (fires on release)
    # ---------------------------------------------------------------------------

    def preview_time_offset(self, value):
        self.display_time_offset_min = value[0] if isinstance(value, list) else float(value)

    def preview_center_freq(self, value):
        self.display_center_freq_ghz = value[0] if isinstance(value, list) else float(value)

    def preview_bandwidth(self, value):
        self.display_bandwidth_khz = value[0] if isinstance(value, list) else float(value)

    def preview_min_elevation(self, value):
        self.display_min_elevation = value[0] if isinstance(value, list) else float(value)

    def preview_beam_avoid(self, value):
        self.display_beam_avoid_deg = value[0] if isinstance(value, list) else float(value)

    def preview_n_sats(self, value):
        self.display_n_sats = int(value[0] if isinstance(value, list) else value)

    def preview_skymap_step(self, value):
        self.display_skymap_step = int(value[0] if isinstance(value, list) else value)

    def set_time_offset(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.time_offset_min = v
        self.display_time_offset_min = v
        self._generation += 1
        return RadioAstroState.recompute_sky_map

    def set_center_freq(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.center_freq_ghz = v
        self.display_center_freq_ghz = v
        self._generation += 1
        return RadioAstroState.recompute

    def set_bandwidth(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.bandwidth_khz = v
        self.display_bandwidth_khz = v
        self._generation += 1
        return RadioAstroState.recompute

    def set_min_elevation(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.min_elevation = v
        self.display_min_elevation = v
        self._generation += 1
        return RadioAstroState.recompute

    def set_beam_avoid(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.beam_avoid_deg = v
        self.display_beam_avoid_deg = v
        self._generation += 1
        return RadioAstroState.recompute

    def set_n_sats(self, value):
        v = int(value[0] if isinstance(value, list) else value)
        self.n_sats = v
        self.display_n_sats = v
        self._generation += 1
        return RadioAstroState.recompute

    def set_direct_sat(self, value: str):
        self.direct_sat = value
        self._generation += 1
        return RadioAstroState.recompute

    def set_skymap_step(self, value):
        v = int(value[0] if isinstance(value, list) else value)
        self.skymap_step = v
        self.display_skymap_step = v
        return RadioAstroState.recompute_sky_map

    def toggle_constellation(self, value: bool):
        self.constellation_enabled = value
        if value:
            self.beam_avoid_deg = 0.0
            self.display_beam_avoid_deg = 0.0
            self.n_sats = 0
            self.display_n_sats = 0
            self.direct_sat = "(all satellites)"
        self._generation += 1
        return RadioAstroState.recompute

    def reset_to_defaults(self):
        self.time_offset_min = 5.0
        self.center_freq_ghz = CENTER_FREQUENCY / 1e9
        self.bandwidth_khz = BANDWIDTH / 1e3
        self.n_channels = FREQUENCY_CHANNELS
        self.receiver_temp_k = float(RECEIVER_TEMP)
        self.constellation_enabled = False
        self.beam_avoid_deg = 0.0
        self.n_sats = 0
        self.direct_sat = "(all satellites)"
        self.min_elevation = 5.0
        self.skymap_step = 20
        self.reset_key += 1
        # sync display vars
        self.display_time_offset_min = 5.0
        self.display_center_freq_ghz = CENTER_FREQUENCY / 1e9
        self.display_bandwidth_khz = BANDWIDTH / 1e3
        self.display_min_elevation = 5.0
        self.display_beam_avoid_deg = 0.0
        self.display_n_sats = 0
        self.display_skymap_step = 20
        self._generation += 1
        return RadioAstroState.recompute

    def request_sky_map(self):
        """Explicit user trigger for sky map generation."""
        return RadioAstroState.recompute_sky_map

    # ---------------------------------------------------------------------------
    # Background: time series recompute (3-observation pattern)
    # ---------------------------------------------------------------------------

    @rx.event(background=True, debounce=500)
    async def recompute(self):
        """
        Run three RSC-SIM observations and store serializable results.

        Three-observation pattern matching radio_astro.py:633-646:
          1. no_sat — baseline (green trace)
          2. with_sat_no_avoid — Starlink on, beam_avoid=0 (blue trace)
          3. current — user settings with beam avoidance (orange trace)
        """
        if not RESOURCES.is_loaded():
            return

        import pandas as pd

        async with self:
            self.loading = True
            gen = self._generation
            params = self._build_params()
            time_offset_snapshot = self.time_offset_min

        bw_hz = params.bandwidth_hz
        sel_offset_ns = int(time_offset_snapshot * 60 * 1e9)
        sel_time_ns = int(pd.Timestamp(_OBS_START_ISO).value) + sel_offset_ns

        # Build the three param variants
        from dataclasses import replace
        no_sat_params = replace(params, constellation_enabled=False, beam_avoid_deg=0.0)

        # Run outside state lock (CPU-bound, must not hold lock)
        try:
            result_no = compute_observation_cached(no_sat_params)
        except Exception as exc:
            async with self:
                self.load_error = f"Simulation failed: {exc}"
                self.loading = False
            return

        p_no_dbw = _tb_to_dbw(result_no["tb_k"], bw_hz)
        p_with_dbw = None
        p_avoid_dbw = None
        n_sats = 0

        if params.constellation_enabled:
            try:
                with_sat_params = replace(params, beam_avoid_deg=0.0)
                result_with = compute_observation_cached(with_sat_params)
                result_current = compute_observation_cached(params)
            except Exception as exc:
                async with self:
                    self.load_error = f"Simulation failed: {exc}"
                    self.loading = False
                return

            p_with_dbw = _tb_to_dbw(result_with["tb_k"], bw_hz)
            p_current_dbw = _tb_to_dbw(result_current["tb_k"], bw_hz)
            if params.beam_avoid_deg > 0:
                p_avoid_dbw = p_current_dbw
            p_display_dbw = p_current_dbw if params.beam_avoid_deg > 0 else p_with_dbw
            n_sats = result_current["n_sats"]
        else:
            p_display_dbw = p_no_dbw

        async with self:
            if self._generation != gen:
                return  # Stale result — a newer recompute is in flight

            fig_dict = radio_time_series_dict(
                times_ns=result_no["times"],
                p_no_dbw=p_no_dbw,
                p_with_dbw=p_with_dbw,
                p_avoid_dbw=p_avoid_dbw,
                obs_start=OBSERVATION_START,
                time_on_source=TIME_ON_SOURCE,
                obs_end=OBSERVATION_END,
                selected_time_ns=sel_time_ns,
                beam_avoid_deg=params.beam_avoid_deg if p_avoid_dbw else None,
            )

            self.time_series_data = _to_serializable({
                "times_ns": result_no["times"],
                "p_no_dbw": p_no_dbw,
                "p_with_dbw": p_with_dbw,
                "p_avoid_dbw": p_avoid_dbw,
            })
            self.time_series_fig = fig_dict
            self.metrics = {
                "visible_sats": n_sats,
                "peak_power_dbw": float(max(p_display_dbw)) if p_display_dbw else 0.0,
            }
            self.load_error = ""
            self.loading = False
            # Auto-generate sky map on first load (sky_map_base64 still empty)
            need_sky_map = self.sky_map_base64 == ""

        if need_sky_map:
            return RadioAstroState.recompute_sky_map

    # ---------------------------------------------------------------------------
    # Background: sky map recompute
    # ---------------------------------------------------------------------------

    @rx.event(background=True)
    async def recompute_sky_map(self):
        """Compute polar sky map + satellite/source overlays → base64 PNG."""
        if not RESOURCES.is_loaded():
            return

        async with self:
            self.sky_map_loading = True
            gen = self._sky_map_gen
            self._sky_map_gen += 1
            sky_gen = self._sky_map_gen
            params = self._build_params()
            az_step = self.skymap_step
            time_offset = self.time_offset_min

        import pandas as pd
        time_plot = OBSERVATION_START + pd.Timedelta(minutes=time_offset)
        time_iso = time_plot.isoformat()
        el_step = max(2, az_step // 2)

        try:
            sky_data = compute_sky_map_cached(params, time_iso, az_step, el_step)
        except Exception as exc:
            async with self:
                self.load_error = f"Sky map failed: {exc}"
                self.sky_map_loading = False
            return

        import numpy as np
        b64 = polar_sky_map_base64(
            map_grid=np.asarray(sky_data["map_grid"]),
            az_grid=np.asarray(sky_data["az_grid"]),
            el_grid=np.asarray(sky_data["el_grid"]),
            bandwidth_hz=params.bandwidth_hz,
            sat_az=np.asarray(sky_data["sat_az"]) if sky_data["sat_az"] else None,
            sat_el=np.asarray(sky_data["sat_el"]) if sky_data["sat_el"] else None,
            src_az=sky_data["src_az"],
            src_el=sky_data["src_el"],
            pointing_az=sky_data["pointing_az"],
            pointing_el=sky_data["pointing_el"],
        )

        # Ground tracks (CPU-bound but fast with max_points=4000)
        import plotly.graph_objects as go

        starlink_df = RESOURCES.starlink_df_radio
        track_df = ground_track_dataframe(
            starlink_df,
            TELESCOPE_COORDS,
            direct_satellite=params.direct_satellite,
            n_satellites=params.n_satellites,
        )

        track_fig = go.Figure()
        if not track_df.empty:
            track_fig.add_trace(go.Scattergeo(
                lat=track_df["lat"].tolist(),
                lon=track_df["lon"].tolist(),
                text=track_df["sat"].tolist(),
                mode="markers",
                marker=dict(size=3, color="#1f77b4", opacity=0.6),
                name="Starlink tracks",
            ))
        track_fig.add_trace(go.Scattergeo(
            lat=[TELESCOPE_COORDS[0]],
            lon=[TELESCOPE_COORDS[1]],
            text=["Westford"],
            mode="markers+text",
            marker=dict(size=12, color="red", symbol="star"),
            name="Westford",
            textposition="top center",
        ))
        track_fig.update_layout(
            geo=dict(
                showland=True,
                landcolor="rgb(50, 50, 50)",
                showocean=True,
                oceancolor="rgb(20, 20, 30)",
                showcountries=True,
                countrycolor="rgb(80, 80, 80)",
                projection_type="natural earth",
                center=dict(lat=TELESCOPE_COORDS[0], lon=TELESCOPE_COORDS[1]),
                projection_scale=2,
            ),
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
        )

        async with self:
            if self._sky_map_gen != sky_gen:
                return  # Newer sky map request supersedes this one
            self.sky_map_base64 = b64
            self.ground_track_fig = plotly_fig_to_dict(track_fig)
            self.sky_map_loading = False

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _build_params(self) -> RadioParams:
        return RadioParams(
            center_freq_hz=self.center_freq_ghz * 1e9,
            bandwidth_hz=self.bandwidth_khz * 1e3,
            n_channels=self.n_channels,
            receiver_temp_k=self.receiver_temp_k,
            beam_avoid_deg=self.beam_avoid_deg,
            constellation_enabled=self.constellation_enabled,
            min_elevation_deg=self.min_elevation,
            n_satellites=self.n_sats,
            # CRITICAL: map sentinel to None (truthy string would filter for non-existent sat)
            direct_satellite=(
                None if self.direct_sat == "(all satellites)" else self.direct_sat
            ),
            obs_start_iso=_OBS_START_ISO,
            obs_end_iso=_OBS_END_ISO,
        )
