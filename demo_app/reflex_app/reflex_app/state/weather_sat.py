"""
Reactive state for the Weather Satellite (Looking-Down) tab.

All fields are JSON-serializable primitives. Non-serializable objects in
state/resources.py _ResourceStore.
"""

from __future__ import annotations

import numpy as np
import reflex as rx

from reflex_app.state.cache import WeatherParams, compute_rfi_cached
from reflex_app.state.resources import RESOURCES
from reflex_app.utils.plot_helpers import _to_serializable, weather_time_series_dict
from reflex_app.utils.weather_loaders_reflex import (
    NEGLIGIBLE_RFI_DBW,
    OBS_START,
    OBS_END,
    DEMO_TIME_STEP_S,
    starlink_count_at_time,
)


_OBS_TOTAL_MINUTES = (OBS_END - OBS_START).total_seconds() / 60.0

_STARLINK_EIRP_DEFAULT = float(-15 + 10 * np.log10(300))


class WeatherSatState(rx.State):
    """Tab 2 reactive state. All fields must be JSON-serializable."""

    # --- Slider values ---
    time_offset_min: float = 1.0
    starlink_freq_ghz: float = 11.9
    starlink_eirp_dbw: float = _STARLINK_EIRP_DEFAULT
    five_g_freq_ghz: float = 25.15
    five_g_eirp_dbw: float = 30.0
    emitter_density: float = 1.0

    # --- Live slider display values (updated on_change, no compute) ---
    display_time_offset_min: float = 1.0
    display_starlink_freq_ghz: float = 11.9
    display_starlink_eirp_dbw: float = _STARLINK_EIRP_DEFAULT
    display_five_g_freq_ghz: float = 25.15
    display_five_g_eirp_dbw: float = 30.0
    display_emitter_density: float = 1.0

    # --- UI state ---
    loading: bool = False
    resources_ready: bool = False
    load_error: str = ""

    # --- Computed results ---
    rfi_data: dict = {}          # keys from compute_rfi_cached (all lists)
    time_series_fig: dict = {}   # Plotly figure dict for rx.plotly()
    metrics: dict = {}           # Starlink + 5G metric values at selected time
    peak_rfi: dict = {}          # Max over full overpass

    # Incremented on Reset; used as key= on sliders to force remount to default_value
    reset_key: int = 0

    # Backend var
    _generation: int = 0

    # ---------------------------------------------------------------------------
    # Computed vars
    # ---------------------------------------------------------------------------

    @rx.var
    def obs_total_minutes(self) -> float:
        return _OBS_TOTAL_MINUTES

    @rx.var
    def selected_minute(self) -> float:
        """Minutes value clamped to available data range."""
        return max(0.0, min(self.time_offset_min, _OBS_TOTAL_MINUTES))

    # ---------------------------------------------------------------------------
    # Resource loading
    # ---------------------------------------------------------------------------

    @rx.event(background=True)
    async def load_resources(self):
        async with self:
            self.loading = True
            self.load_error = ""
        try:
            RESOURCES.load_all()
            async with self:
                self.resources_ready = True
                self.loading = False
        except Exception as exc:
            async with self:
                self.load_error = f"Resource loading failed: {exc}"
                self.resources_ready = False
                self.loading = False
            return

        return WeatherSatState.recompute

    # ---------------------------------------------------------------------------
    # Slider event handlers
    # on_change  → display var only (every drag tick)
    # on_value_commit → commit + recompute (on release)
    # ---------------------------------------------------------------------------

    def preview_time_offset(self, value):
        self.display_time_offset_min = value[0] if isinstance(value, list) else float(value)

    def preview_starlink_freq(self, value):
        self.display_starlink_freq_ghz = value[0] if isinstance(value, list) else float(value)

    def preview_starlink_eirp(self, value):
        self.display_starlink_eirp_dbw = value[0] if isinstance(value, list) else float(value)

    def preview_5g_freq(self, value):
        self.display_five_g_freq_ghz = value[0] if isinstance(value, list) else float(value)

    def preview_5g_eirp(self, value):
        self.display_five_g_eirp_dbw = value[0] if isinstance(value, list) else float(value)

    def preview_emitter_density(self, value):
        self.display_emitter_density = value[0] if isinstance(value, list) else float(value)

    def set_time_offset(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.time_offset_min = v
        self.display_time_offset_min = v
        return WeatherSatState.update_time_marker

    def set_starlink_freq(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.starlink_freq_ghz = v
        self.display_starlink_freq_ghz = v
        self._generation += 1
        return WeatherSatState.recompute

    def set_starlink_eirp(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.starlink_eirp_dbw = v
        self.display_starlink_eirp_dbw = v
        self._generation += 1
        return WeatherSatState.recompute

    def set_5g_freq(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.five_g_freq_ghz = v
        self.display_five_g_freq_ghz = v
        self._generation += 1
        return WeatherSatState.recompute

    def set_5g_eirp(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.five_g_eirp_dbw = v
        self.display_five_g_eirp_dbw = v
        self._generation += 1
        return WeatherSatState.recompute

    def set_emitter_density(self, value):
        v = value[0] if isinstance(value, list) else float(value)
        self.emitter_density = v
        self.display_emitter_density = v
        self._generation += 1
        return WeatherSatState.recompute

    def reset_to_defaults(self):
        self.time_offset_min = 1.0
        self.starlink_freq_ghz = 11.9
        self.starlink_eirp_dbw = _STARLINK_EIRP_DEFAULT
        self.five_g_freq_ghz = 25.15
        self.five_g_eirp_dbw = 30.0
        self.emitter_density = 1.0
        self.display_time_offset_min = 1.0
        self.display_starlink_freq_ghz = 11.9
        self.display_starlink_eirp_dbw = _STARLINK_EIRP_DEFAULT
        self.display_five_g_freq_ghz = 25.15
        self.display_five_g_eirp_dbw = 30.0
        self.display_emitter_density = 1.0
        self.reset_key += 1
        self._generation += 1
        return WeatherSatState.recompute

    # ---------------------------------------------------------------------------
    # Background: update only the time marker (no simulation re-run)
    # ---------------------------------------------------------------------------

    @rx.event(background=True)
    async def update_time_marker(self):
        """Redraw time series with updated selected-time marker without recomputing."""
        if not self.rfi_data:
            return

        async with self:
            rfi = self.rfi_data
            sel_min = self.time_offset_min
            params = self._build_params()

        if not rfi.get("minutes"):
            return

        fig_dict = weather_time_series_dict(
            minutes=rfi["minutes"],
            k_starlink_dbw=rfi["k_starlink_dbw"],
            k_5g_dbw=rfi["k_5g_dbw"],
            v_starlink_dbw=rfi["v_starlink_dbw"],
            v_5g_dbw=rfi["v_5g_dbw"],
            selected_minute=sel_min,
        )
        metrics = self._compute_metrics(rfi, sel_min, params)

        async with self:
            self.time_series_fig = fig_dict
            self.metrics = metrics

    # ---------------------------------------------------------------------------
    # Background: full RFI recompute
    # ---------------------------------------------------------------------------

    @rx.event(background=True, debounce=500)
    async def recompute(self):
        if not RESOURCES.is_loaded():
            return

        async with self:
            self.loading = True
            gen = self._generation
            params = self._build_params()
            sel_min = self.time_offset_min

        try:
            rfi = compute_rfi_cached(params)
        except Exception as exc:
            async with self:
                self.load_error = f"Weather simulation failed: {exc}"
                self.loading = False
            return

        if rfi.get("missing_data"):
            async with self:
                self.load_error = "Missing weather satellite antenna pattern files."
                self.loading = False
            return

        fig_dict = weather_time_series_dict(
            minutes=rfi["minutes"],
            k_starlink_dbw=rfi["k_starlink_dbw"],
            k_5g_dbw=rfi["k_5g_dbw"],
            v_starlink_dbw=rfi["v_starlink_dbw"],
            v_5g_dbw=rfi["v_5g_dbw"],
            selected_minute=sel_min,
        )

        metrics = self._compute_metrics(rfi, sel_min, params)

        peak = {
            "k_starlink": float(max(rfi["k_starlink_dbw"])),
            "v_starlink": float(max(rfi["v_starlink_dbw"])),
            "k_5g": float(max(rfi["k_5g_dbw"])),
            "v_5g": float(max(rfi["v_5g_dbw"])),
        }

        async with self:
            if self._generation != gen:
                return
            self.rfi_data = _to_serializable(rfi)
            self.time_series_fig = fig_dict
            self.metrics = metrics
            self.peak_rfi = peak
            self.load_error = ""
            self.loading = False

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _build_params(self) -> WeatherParams:
        return WeatherParams(
            starlink_freq_ghz=self.starlink_freq_ghz,
            starlink_eirp_dbw=self.starlink_eirp_dbw,
            five_g_freq_ghz=self.five_g_freq_ghz,
            five_g_eirp_dbw=self.five_g_eirp_dbw,
            emitter_density=self.emitter_density,
        )

    @staticmethod
    def _compute_metrics(rfi: dict, sel_min: float, params: WeatherParams) -> dict:
        """Compute metric values at the selected time index."""
        from datetime import datetime

        minutes = rfi.get("minutes", [])
        n_pts = len(minutes)
        if n_pts == 0:
            return {}

        idx = int(round(sel_min * 60.0 / DEMO_TIME_STEP_S))
        idx = max(0, min(idx, n_pts - 1))

        obs_times_iso = rfi.get("obs_times_iso", [])
        n_sl = 0
        if obs_times_iso and idx < len(obs_times_iso):
            try:
                obs_time = datetime.fromisoformat(obs_times_iso[idx])
                n_sl = starlink_count_at_time(obs_time)
            except Exception:
                n_sl = 0

        return {
            "n_starlinks": n_sl,
            "starlink_freq_ghz": params.starlink_freq_ghz,
            "k_starlink_dbw": rfi["k_starlink_dbw"][idx],
            "v_starlink_dbw": rfi["v_starlink_dbw"][idx],
            "n_emitters": rfi.get("n_emitters", 0),
            "five_g_freq_ghz": params.five_g_freq_ghz,
            "five_g_eirp_dbw": params.five_g_eirp_dbw,
            "k_5g_dbw": rfi["k_5g_dbw"][idx],
            "v_5g_dbw": rfi["v_5g_dbw"][idx],
            "selected_minute": float(minutes[idx]),
        }
