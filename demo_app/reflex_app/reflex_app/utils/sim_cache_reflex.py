"""
Reflex port of demo_app/sim_cache.py.

All @st.cache_resource / @st.cache_data decorators removed.
Caching is handled by the module-level _ResourceStore singleton (state/resources.py)
and @lru_cache in state/cache.py.

sys.path is adjusted for running from demo_app/reflex_app/ (4 levels up to repo root).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Thread-safe backend — MUST be before any pyplot import

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

# Path from utils/ -> reflex_app/ -> reflex_app/ -> demo_app/ -> RSC-SIM/ (4 levels up)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
_EDU_DIR = os.path.join(_REPO_ROOT, "educational_tutorials")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _EDU_DIR not in sys.path:
    sys.path.insert(0, _EDU_DIR)

# RSC-SIM imports (after sys.path mutation)
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
    temperature_to_power,
)
from antenna_pattern import gain_to_effective_aperture  # noqa: E402
from obs_mdl import model_observed_temp  # noqa: E402
from sat_mdl import sat_link_budget_vectorized  # noqa: E402

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
    """Build a Westford Instrument from the cached antenna and chosen receiver params."""
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


def load_starlink_transmitter_itu() -> Instrument:
    """Standard ITU-mask Starlink transmitter."""
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


def load_cas_a_trajectory() -> Trajectory:
    """Load the bundled Cas A pointing trajectory."""
    return Trajectory.from_file(
        CAS_A_TRAJECTORY_FILE,
        time_tag="time_stamps",
        elevation_tag="altitudes",
        azimuth_tag="azimuths",
        distance_tag="distances",
    )


def load_pointing_trajectory() -> Trajectory:
    """Apply the Cas A ON/OFF source offset so the trajectory matches the tutorials."""
    base = load_cas_a_trajectory()
    pointing = Trajectory(base.traj.copy())
    mask = (pointing.traj["times"] >= OBSERVATION_START) & (
        pointing.traj["times"] <= TIME_ON_SOURCE
    )
    pointing.traj.loc[mask, "azimuths"] += OFFSET_ANGLES[0]
    pointing.traj.loc[mask, "elevations"] += OFFSET_ANGLES[1]
    return pointing


def load_starlink_trajectory_df() -> pd.DataFrame:
    """Load the bundled Starlink Arrow trajectory as a pandas DataFrame."""
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
    """Build a sky temperature model closure."""
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
    """Hashable cache key for compute_observation_result."""
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


def make_observation(state: RadioState) -> Tuple[Observation, Trajectory]:
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
    """Return a lnk_bdgt_mdl closure that applies a beam-avoidance angle."""

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
    """Build a Constellation from the cached Starlink DataFrame."""
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


def compute_observation_result(state: RadioState) -> dict:
    """
    Run model_observed_temp for the given state and return a JSON-friendly payload.

    Returns 'tb_k' as brightness temperature in Kelvin.
    Caller must convert to power dBW using temperature_to_power() + 10*log10().
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
        "sat_names": list(sat_names),
        "obs_start": state.obs_start_iso,
        "obs_end": state.obs_end_iso,
    }


def tb_to_dbw(tb_k, bandwidth_hz: float) -> list:
    """Convert brightness temperature (K) to received power (dBW)."""
    tb = np.asarray(tb_k)
    power_w = temperature_to_power(tb, bandwidth_hz)
    return (10.0 * np.log10(np.clip(power_w, 1e-300, None))).tolist()


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def list_starlink_satellite_names(max_count: int = 200) -> list:
    """Names of Starlink satellites in the bundled trajectory (truncated)."""
    df = load_starlink_trajectory_df()
    names = sorted(n for n in df["sat"].dropna().unique() if "DTC" not in str(n))
    return names[:max_count]


def starlink_time_bounds() -> Tuple[datetime, datetime]:
    """Min/max timestamps in the bundled Starlink Arrow file."""
    df = load_starlink_trajectory_df()
    return df["times"].min().to_pydatetime(), df["times"].max().to_pydatetime()


def cas_a_time_bounds() -> Tuple[datetime, datetime]:
    """Min/max timestamps in the Cas A pointing trajectory."""
    traj = load_cas_a_trajectory()
    return traj.get_traj()["times"].min().to_pydatetime(), traj.get_traj()[
        "times"
    ].max().to_pydatetime()
