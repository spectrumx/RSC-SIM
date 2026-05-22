"""
Cached loaders for the RSC-SIM Streamlit demo.

Heavy artifacts (Antenna, Trajectory, Constellation, sky-models) are loaded
exactly once per Streamlit process and pinned with `@st.cache_resource`. The
expensive `model_observed_temp` call is wrapped with `@st.cache_data` and keyed
on the hashable subset of the slider state, so repeated drags become instant.

This module is intentionally side-effect-free at import time: nothing is loaded
until a panel calls one of the `load_*` helpers. The first call therefore
"pre-warms" the cache; subsequent calls (from any tab) are O(1).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Make the RSC-SIM source modules importable regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Also make the educational shared utilities importable so we can reuse the
# Westford / Cas A configuration constants without copy-pasting.
_EDU_DIR = os.path.join(_REPO_ROOT, "educational_tutorials")
if _EDU_DIR not in sys.path:
    sys.path.insert(0, _EDU_DIR)

# RSC-SIM imports (after sys.path mutation).
from radio_types import (  # noqa: E402
    Antenna,
    Constellation,
    Instrument,
    Observation,
    Trajectory,
)
from astro_mdl import (  # noqa: E402
    antenna_mdl_ITU,
    estim_casA_flux,
    estim_temp,
    power_to_temperature,
)
from antenna_pattern import gain_to_effective_aperture  # noqa: E402
from obs_mdl import model_observed_temp  # noqa: E402
from sat_mdl import sat_link_budget_vectorized  # noqa: E402

# Reuse the educational tutorial defaults for parameters that don't need to
# differ between the CLI tutorials and the demo app.
from shared.config import (  # type: ignore  # noqa: E402
    ANTENNA_PATTERN_FILE,
    CAS_A_TRAJECTORY_FILE,
    STARLINK_TRAJECTORY_FILE,
    OBSERVATION_START,
    OBSERVATION_END,
    OFFSET_ANGLES,
    TIME_ON_SOURCE,
    BANDWIDTH,
    CENTER_FREQUENCY,
    TELESCOPE_RADIATION_EFFICIENCY,
    TELESCOPE_FREQ_BAND,
    TELESCOPE_PHYSICAL_TEMP,
    FREQUENCY_CHANNELS,
    RECEIVER_TEMP,
    TELESCOPE_COORDS,
    SATELLITE_RADIATION_EFFICIENCY,
    SATELLITE_MAX_GAIN,
    SATELLITE_HALF_BEAMWIDTH,
    SATELLITE_PHYSICAL_TEMP,
    SATELLITE_FREQUENCY,
    SATELLITE_BANDWIDTH,
    SATELLITE_TRANSMIT_POWER,
    ATMOSPHERIC_TEMP_ZENITH,
    ATMOSPHERIC_OPACITY,
    GALACTIC_TEMP_REF_FREQ,
    GALACTIC_TEMP_REF_VALUE,
    GALACTIC_TEMP_SPECTRAL_INDEX,
    CMB_TEMPERATURE,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


DATA_DIR = os.path.join(_REPO_ROOT, "research_tutorials", "data")


def repo_data_path(*parts: str) -> str:
    """Join components onto the bundled `research_tutorials/data/` directory."""
    return os.path.join(DATA_DIR, *parts)


def data_file_exists(filename: str) -> bool:
    """Return True if `research_tutorials/data/<filename>` is present."""
    return os.path.isfile(repo_data_path(filename))


# ---------------------------------------------------------------------------
# Telescope / antenna / source loaders (radio-astronomy tab)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading Westford antenna pattern...")
def load_westford_antenna() -> Antenna:
    """Load the Westford antenna model from `single_cut_res.cut`."""
    return Antenna.from_file(
        ANTENNA_PATTERN_FILE,
        TELESCOPE_RADIATION_EFFICIENCY,
        TELESCOPE_FREQ_BAND,
        power_tag="power",
        declination_tag="alpha",
        azimuth_tag="beta",
    )


def make_westford_telescope(
    center_freq: float = CENTER_FREQUENCY,
    bandwidth: float = BANDWIDTH,
    receiver_temp: float = RECEIVER_TEMP,
    n_channels: int = FREQUENCY_CHANNELS,
) -> Instrument:
    """
    Build a Westford `Instrument` from the cached antenna and the chosen
    receiver parameters. Cheap; not cached because callers tweak the freq.
    """
    antenna = load_westford_antenna()

    def T_RX(_t, _f):
        return receiver_temp

    return Instrument(
        antenna,
        TELESCOPE_PHYSICAL_TEMP,
        center_freq,
        bandwidth,
        T_RX,
        n_channels,
        TELESCOPE_COORDS,
    )


@st.cache_resource(show_spinner="Building Starlink ITU transmitter...")
def load_starlink_transmitter_itu() -> Instrument:
    """Standard ITU-mask Starlink transmitter (matches `setup_satellite_transmitter`)."""
    alphas = np.arange(0, 181)
    betas = np.arange(0, 351, 10)
    gain_pat = antenna_mdl_ITU(
        SATELLITE_MAX_GAIN, SATELLITE_HALF_BEAMWIDTH, alphas, betas
    )
    sat_ant = Antenna.from_dataframe(
        gain_pat, SATELLITE_RADIATION_EFFICIENCY, TELESCOPE_FREQ_BAND
    )

    def transmit_temp(_t, _f):
        return power_to_temperature(10 ** (SATELLITE_TRANSMIT_POWER / 10), 1.0)

    return Instrument(
        sat_ant,
        SATELLITE_PHYSICAL_TEMP,
        SATELLITE_FREQUENCY,
        SATELLITE_BANDWIDTH,
        transmit_temp,
        1,
        [],
    )


@st.cache_resource(show_spinner="Loading Cas A trajectory...")
def load_cas_a_trajectory() -> Trajectory:
    """Load the bundled Cas A pointing trajectory."""
    return Trajectory.from_file(
        CAS_A_TRAJECTORY_FILE,
        time_tag="time_stamps",
        elevation_tag="altitudes",
        azimuth_tag="azimuths",
        distance_tag="distances",
    )


@st.cache_resource(show_spinner="Building Cas A pointing trajectory...")
def load_pointing_trajectory() -> Trajectory:
    """
    Apply the Cas A ON/OFF source offset (from the educational config) so the
    trajectory matches the tutorials. Cached because the OFFSET / TIME_ON_SOURCE
    are constants.
    """
    base = load_cas_a_trajectory()
    pointing = Trajectory(base.traj.copy())
    mask = (pointing.traj["times"] >= OBSERVATION_START) & (
        pointing.traj["times"] <= TIME_ON_SOURCE
    )
    pointing.traj.loc[mask, "azimuths"] += OFFSET_ANGLES[0]
    pointing.traj.loc[mask, "elevations"] += OFFSET_ANGLES[1]
    return pointing


@st.cache_resource(show_spinner="Loading Starlink trajectory...")
def load_starlink_trajectory_df() -> pd.DataFrame:
    """
    Load the bundled Starlink Arrow trajectory once and keep a pandas DataFrame
    in memory. The Constellation builder takes a slice of this DataFrame, so
    sharing a single cached copy avoids re-reading 100s of MB on every change.
    """
    import pyarrow as pa

    with pa.memory_map(STARLINK_TRAJECTORY_FILE, "r") as source:
        table = pa.ipc.open_file(source).read_all()
    df = table.to_pandas()
    df = df.rename(
        columns={
            "timestamp": "times",
            "ranges_westford": "distances",
        }
    )
    df["times"] = pd.to_datetime(df["times"])
    return df


# ---------------------------------------------------------------------------
# Sky models
# ---------------------------------------------------------------------------


def make_sky_model(
    observation: Observation,
    *,
    include_source: bool = True,
) -> Callable:
    """
    Build a sky temperature model identical to `shared.sky_models.create_sky_model`,
    but with a switch to drop the Cas A source contribution for "no-source"
    sky maps. The model itself is a closure -- not Streamlit-cacheable.
    """
    antenna = observation.get_instrument().get_antenna()
    max_gain = antenna.get_boresight_gain()
    eff_aperture = gain_to_effective_aperture(max_gain, CENTER_FREQUENCY)
    flux_src = estim_casA_flux(CENTER_FREQUENCY)
    src_temp = estim_temp(flux_src, eff_aperture)

    def T_atm(dec):
        return ATMOSPHERIC_TEMP_ZENITH * (1 - np.exp(-ATMOSPHERIC_OPACITY / np.cos(dec)))

    def T_bkg(freq):
        gal = GALACTIC_TEMP_REF_VALUE * (
            freq / GALACTIC_TEMP_REF_FREQ
        ) ** GALACTIC_TEMP_SPECTRAL_INDEX
        return CMB_TEMPERATURE + gal

    def T_src(t):
        if not include_source:
            return 0.0
        if t <= TIME_ON_SOURCE:
            return 0.0
        return src_temp

    def sky_mdl(dec, caz, tim, freq):
        return T_src(tim) + T_atm(dec) + T_bkg(freq)

    return sky_mdl


# ---------------------------------------------------------------------------
# Observation factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadioState:
    """
    Hashable subset of the radio-astronomy panel's slider state. Used as a
    cache key for `compute_observation_result`.
    """
    center_freq_hz: float
    bandwidth_hz: float
    n_channels: int
    receiver_temp_k: float
    beam_avoid_deg: float
    min_elevation_deg: float
    n_satellites: int
    direct_satellite: Optional[str]
    constellation_enabled: bool
    obs_start_iso: str
    obs_end_iso: str


def make_observation(
    state: RadioState,
) -> Tuple[Observation, Trajectory]:
    """Build a fresh Observation from the educational pointing trajectory."""
    pointing = load_pointing_trajectory()
    telescope = make_westford_telescope(
        center_freq=state.center_freq_hz,
        bandwidth=state.bandwidth_hz,
        receiver_temp=state.receiver_temp_k,
        n_channels=state.n_channels,
    )
    obs = Observation.from_dates(
        datetime.fromisoformat(state.obs_start_iso),
        datetime.fromisoformat(state.obs_end_iso),
        pointing,
        telescope,
        filt_funcs=(("elevations", lambda e: e > state.min_elevation_deg),),
    )
    return obs, pointing


def _link_budget_with_beam_avoid(beam_avoid_deg: float) -> Callable:
    """Return a `lnk_bdgt_mdl` closure that applies a beam-avoidance angle."""

    def lnk_bdgt(*args, **kwargs):
        kwargs.pop("beam_avoid", None)
        return sat_link_budget_vectorized(
            *args, beam_avoid=beam_avoid_deg, turn_off=False, **kwargs
        )

    return lnk_bdgt


def make_constellation(
    state: RadioState,
    observation: Observation,
) -> Optional[Constellation]:
    """Build a `Constellation` from the cached Starlink DataFrame."""
    if not state.constellation_enabled:
        return None

    df = load_starlink_trajectory_df().copy()
    transmitter = load_starlink_transmitter_itu()
    lnk_bdgt = _link_budget_with_beam_avoid(state.beam_avoid_deg)

    filt_funcs = [
        ("sat", lambda s: ~s.str.contains("DTC")),
        ("elevations", lambda e: e > 20.0),
    ]
    if state.direct_satellite:
        wanted = state.direct_satellite

        def _name_filter(s, _wanted=wanted):
            return s == _wanted

        filt_funcs.append(("sat", _name_filter))

    constellation = Constellation.from_observation(
        df,
        observation,
        transmitter,
        lnk_bdgt_mdl=lnk_bdgt,
        filt_funcs=tuple(filt_funcs),
    )

    if state.n_satellites > 0 and not state.direct_satellite:
        all_names = constellation.get_sats_name()
        keep = set(all_names[: state.n_satellites])
        constellation.sats = constellation.sats[
            constellation.sats["sat"].isin(keep)
        ].reset_index(drop=True)

    return constellation


@st.cache_data(show_spinner="Running observation simulation...", max_entries=64)
def compute_observation_result(state: RadioState) -> dict:
    """
    Run `model_observed_temp` for the slider state and return a JSON-friendly
    payload. Streamlit will cache the dict by `state`, so repeating a slider
    drag returns instantly.
    """
    obs, _ = make_observation(state)
    constellation = make_constellation(state, obs)

    sky_mdl = make_sky_model(obs, include_source=True)
    beam_avoid = state.beam_avoid_deg > 0 and state.constellation_enabled
    result = model_observed_temp(
        obs,
        sky_mdl,
        constellation,
        beam_avoidance=beam_avoid,
    )

    times = obs.get_time_stamps().reset_index(drop=True)
    tb_series = result[:, 0, 0]
    sat_names = constellation.get_sats_name() if constellation is not None else []

    return {
        "times": times.astype("datetime64[ns]").astype(np.int64).tolist(),
        "tb_k": tb_series.tolist(),
        "n_sats": len(sat_names),
        "sat_names": sat_names,
        "obs_start": state.obs_start_iso,
        "obs_end": state.obs_end_iso,
    }


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def list_starlink_satellite_names(max_count: int = 200) -> list:
    """Names of Starlink satellites in the bundled trajectory (truncated)."""
    df = load_starlink_trajectory_df()
    names = sorted(n for n in df["sat"].dropna().unique() if "DTC" not in str(n))
    return names[:max_count]


@st.cache_data(show_spinner=False)
def starlink_time_bounds() -> Tuple[datetime, datetime]:
    """Min/max timestamps in the bundled Starlink Arrow file."""
    df = load_starlink_trajectory_df()
    return df["times"].min().to_pydatetime(), df["times"].max().to_pydatetime()


@st.cache_data(show_spinner=False)
def cas_a_time_bounds() -> Tuple[datetime, datetime]:
    """Min/max timestamps in the Cas A pointing trajectory."""
    traj = load_cas_a_trajectory()
    return traj.get_traj()["times"].min().to_pydatetime(), traj.get_traj()[
        "times"
    ].max().to_pydatetime()


# ---------------------------------------------------------------------------
# Pre-warm helper
# ---------------------------------------------------------------------------


def prewarm_radio_astro() -> None:
    """
    Touch each cached resource so the first slider drag is instant. Called
    from `app.py` once on startup.
    """
    load_westford_antenna()
    load_starlink_transmitter_itu()
    load_cas_a_trajectory()
    load_pointing_trajectory()
    load_starlink_trajectory_df()
