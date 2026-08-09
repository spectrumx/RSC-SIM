"""
LRU-cached computation layer for Reflex.

Replaces @st.cache_data from the Streamlit app. Frozen dataclasses as
hashable cache keys. All return values are JSON-serializable Python natives
(lists, dicts, floats) — never numpy arrays or custom objects.

CRITICAL: Never mutate lru_cache return values. Cache is process-global
(shared across sessions). _to_serializable() creates new objects (safe).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


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


def _tb_to_dbw(tb_list: list, bandwidth_hz: float) -> list:
    """Convert brightness temperature (K) to received power (dBW)."""
    from astro_mdl import temperature_to_power

    tb = np.asarray(tb_list)
    power_w = temperature_to_power(tb, bandwidth_hz)
    return (10.0 * np.log10(np.clip(power_w, 1e-300, None))).tolist()


# ---------------------------------------------------------------------------
# Radio astronomy cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadioParams:
    """Hashable key for compute_observation_cached()."""
    center_freq_hz: float
    bandwidth_hz: float
    n_channels: int
    receiver_temp_k: float
    beam_avoid_deg: float
    constellation_enabled: bool
    min_elevation_deg: float
    n_satellites: int
    direct_satellite: Optional[str]
    obs_start_iso: str
    obs_end_iso: str


@lru_cache(maxsize=128)
def compute_observation_cached(params: RadioParams) -> dict:
    """
    Run RSC-SIM model_observed_temp for the given params.

    Returns serializable dict with 'times' (ns int list), 'tb_k' (float list),
    'n_sats' (int), 'sat_names' (str list).

    Tb is brightness temperature in Kelvin. Caller converts to dBW via _tb_to_dbw().
    """
    from reflex_app.utils.sim_cache_reflex import (
        RadioState,
        make_observation,
        make_constellation,
        make_sky_model,
    )
    from obs_mdl import model_observed_temp

    state = RadioState(
        center_freq_hz=params.center_freq_hz,
        bandwidth_hz=params.bandwidth_hz,
        n_channels=params.n_channels,
        receiver_temp_k=params.receiver_temp_k,
        beam_avoid_deg=params.beam_avoid_deg,
        min_elevation_deg=params.min_elevation_deg,
        n_satellites=params.n_satellites,
        direct_satellite=params.direct_satellite,
        constellation_enabled=params.constellation_enabled,
        obs_start_iso=params.obs_start_iso,
        obs_end_iso=params.obs_end_iso,
    )

    obs, _ = make_observation(state)
    constellation = make_constellation(state, obs)
    sky_mdl = make_sky_model(obs, include_source=True)

    beam_avoid = params.beam_avoid_deg > 0 and params.constellation_enabled
    result = model_observed_temp(obs, sky_mdl, constellation, beam_avoidance=beam_avoid)

    times = obs.get_time_stamps().reset_index(drop=True)
    tb_series = result[:, 0, 0]
    sat_names = constellation.get_sats_name() if constellation is not None else []

    return {
        "times": times.astype("datetime64[ns]").astype(np.int64).tolist(),
        "tb_k": tb_series.tolist(),
        "n_sats": len(sat_names),
        "sat_names": list(sat_names),
        "obs_start": params.obs_start_iso,
        "obs_end": params.obs_end_iso,
    }


@lru_cache(maxsize=24)
def compute_sky_map_cached(
    params: RadioParams,
    time_iso: str,
    az_step_deg: int,
    el_step_deg: int,
) -> dict:
    """
    Compute polar sky temperature map over az/el grid for sky_map visualization.

    Returns dict with 'map_grid', 'az_grid', 'el_grid' as Python lists, plus
    'sat_az', 'sat_el', 'sat_names', 'src_az', 'src_el', 'pointing_az', 'pointing_el'.
    """
    from datetime import datetime

    import pandas as pd

    from reflex_app.utils.sim_cache_reflex import (
        RadioState,
        make_observation,
        make_constellation,
        make_sky_model,
        load_starlink_trajectory_df,
        load_cas_a_trajectory,
        load_pointing_trajectory,
    )
    from obs_mdl import model_observed_temp
    from radio_types import Observation, Trajectory
    from reflex_app.utils.plot_helpers import (
        satellite_positions_at_time,
        source_position_at_time,
        pointing_position_at_time,
    )

    state = RadioState(
        center_freq_hz=params.center_freq_hz,
        bandwidth_hz=params.bandwidth_hz,
        n_channels=params.n_channels,
        receiver_temp_k=params.receiver_temp_k,
        beam_avoid_deg=params.beam_avoid_deg,
        min_elevation_deg=params.min_elevation_deg,
        n_satellites=params.n_satellites,
        direct_satellite=params.direct_satellite,
        constellation_enabled=params.constellation_enabled,
        obs_start_iso=params.obs_start_iso,
        obs_end_iso=params.obs_end_iso,
    )

    time_plot = datetime.fromisoformat(time_iso)
    obs, _ = make_observation(state)
    constellation = make_constellation(state, obs)
    sky_mdl = make_sky_model(obs, include_source=True)

    az_grid = np.arange(0, 360, az_step_deg)
    el_grid = np.arange(5, 91, el_step_deg)
    map_grid = np.zeros((len(el_grid), len(az_grid)))

    instrument = obs.get_instrument()
    use_beam_avoid = (
        params.beam_avoid_deg > 0
        and params.constellation_enabled
        and constellation is not None
    )

    for i, el in enumerate(el_grid):
        for j, az in enumerate(az_grid):
            point_df = pd.DataFrame({
                "times": [time_plot],
                "azimuths": [float(az)],
                "elevations": [float(el)],
                "distances": [np.inf],
            })
            point_obs = Observation.from_dates(
                time_plot, time_plot, Trajectory(point_df), instrument
            )
            result = model_observed_temp(
                point_obs, sky_mdl, constellation, beam_avoidance=use_beam_avoid,
            )
            map_grid[i, j] = result[0, 0, 0]

    # Overlay data
    starlink_df = load_starlink_trajectory_df()
    sat_az, sat_el, sat_names = satellite_positions_at_time(
        starlink_df, time_plot,
        direct_satellite=params.direct_satellite,
        n_satellites=params.n_satellites,
    )

    cas_a_df = load_cas_a_trajectory().get_traj()
    src_az, src_el = source_position_at_time(cas_a_df, time_plot)

    pointing_df = load_pointing_trajectory().get_traj()
    pointing_az, pointing_el = pointing_position_at_time(pointing_df, time_plot)

    return {
        "map_grid": map_grid.tolist(),
        "az_grid": az_grid.tolist(),
        "el_grid": el_grid.tolist(),
        "sat_az": sat_az.tolist(),
        "sat_el": sat_el.tolist(),
        "sat_names": sat_names,
        "src_az": src_az,
        "src_el": src_el,
        "pointing_az": pointing_az,
        "pointing_el": pointing_el,
    }


# ---------------------------------------------------------------------------
# Weather satellite cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeatherParams:
    """Hashable key for compute_rfi_cached()."""
    starlink_freq_ghz: float
    starlink_eirp_dbw: float
    five_g_freq_ghz: float
    five_g_eirp_dbw: float
    emitter_density: float


@lru_cache(maxsize=64)
def compute_rfi_cached(params: WeatherParams) -> dict:
    """
    Run weather satellite RFI computation and return serializable dict.

    Dict keys: 'minutes', 'obs_times_iso', 'k_starlink_dbw', 'k_5g_dbw',
    'v_starlink_dbw', 'v_5g_dbw', 'n_emitters', 'missing_data'.
    All array values are Python lists (not np.ndarray).
    """
    from reflex_app.utils.weather_loaders_reflex import compute_rfi_time_series

    raw = compute_rfi_time_series(
        starlink_freq_ghz=params.starlink_freq_ghz,
        starlink_eirp_dbw=params.starlink_eirp_dbw,
        five_g_freq_ghz=params.five_g_freq_ghz,
        five_g_eirp_dbw=params.five_g_eirp_dbw,
        emitter_density=params.emitter_density,
    )
    return _to_serializable(raw)
