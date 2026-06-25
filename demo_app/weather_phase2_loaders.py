"""
Cached Phase-2 loaders and RFI time-series computation for the Weather FOV tab.

Starlink RFI uses ``model_weather_sat_observed_power_phase2`` (full ECEF path).
5G RFI uses equivalent-emitter scaling at the FOV center (NWP Section 5 idea)
with phase2 ``ground_emitter_to_weather_sat_link_budget`` harmonics and
beam-relative weather-sat receive gain (FOV center boresight).
"""

from __future__ import annotations

import contextlib
import io
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import sim_cache  # noqa: F401
from sim_cache import data_file_exists, repo_data_path

from radio_types import Antenna, Constellation, Instrument, Observation, Trajectory  # noqa: E402
from astro_mdl import antenna_mdl_ITU, power_to_temperature, temperature_to_power  # noqa: E402
from weather_sat_mdl import (  # noqa: E402
    compute_weather_sat_ecef_from_trajectory,
    create_5g_sector_antenna_pattern,
    ground_emitter_to_weather_sat_link_budget,
    latlonalt_to_ecef,
    load_weather_sat_antenna_from_csv,
    model_weather_sat_observed_power_phase2,
)

# ---------------------------------------------------------------------------
# Demo constants (match tuto_radiomdl_weather_phase2.py where noted)
# ---------------------------------------------------------------------------

K_BAND_CSV = "K-Band 23.8 GHz absolute antenna pattern.csv"
V_BAND_CSV = "V-Band 50.3 GHz absolute antenna pattern.csv"
JPSS_FILE = (
    "jpss_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow"
)
STARLINK_FILE = (
    "Starlink_trajectory_Westford_2025-11-01T07_45_00.000_2025-11-01T08_45_00.000.arrow"
)

OBS_START = datetime(2025, 11, 1, 8, 15, 0)
OBS_END = datetime(2025, 11, 1, 8, 21, 0)
DEMO_TIME_STEP_S = 10
RESOLUTION_KM = 32.0
# Bundled Starlink trajectories use a ~5° pass mask at creation; 0° keeps all
# samples in-file. Raise if future trajectories include below-horizon rows.
STARLINK_ELEV_MIN_DEG = 0.0
STARLINK_POL_LOSS_DB = 3.0  # circular (Starlink) vs linear (Suomi-NPP), phase2 default

# Demo display: src/ uses 10*log10(1e-100) = -1000 dBW for zero power; remap for plots.
NEGLIGIBLE_RFI_DBW = -500.0
_RFI_ZERO_POWER_W = 1e-50  # 10*log10(1e-50) = -500 dBW (demo 5G path only)
EMITTER_TOWER_ALT_M = 15.0

OBSERVER_LAT = 42.6129479883915
OBSERVER_LON = -71.49379366344017
OBSERVER_ALT = 86.7689687917009
TARGET_LAT = OBSERVER_LAT
TARGET_LON = OBSERVER_LON
TARGET_ALT = OBSERVER_ALT

FREQ_K_HZ = 23.8e9
FREQ_V_HZ = 50.3e9
BW_K_HZ = 270e6
BW_V_HZ = 180e6

STARLINK_HARMONICS = [
    (2.0, 0.01),
    (3.0, 0.003),
    (4.0, 0.001),
]

FIVE_G_HARMONICS = [
    (2.0, 0.01),
    (3.0, 0.003),
    (4.0, 0.001),
]

T_PHY = 280.0


def get_atms_bandwidth(freq_hz: float) -> float:
    if abs(freq_hz - FREQ_K_HZ) < 0.1e9:
        return BW_K_HZ
    if abs(freq_hz - FREQ_V_HZ) < 0.1e9:
        return BW_V_HZ
    return BW_K_HZ


def t_rx_k_band(_tim, freq) -> float:
    return 300.0 if freq < 30e9 else 400.0


def n_emitters_from_density(density_per_km2: float) -> int:
    area_km2 = np.pi * (RESOLUTION_KM / 2.0) ** 2
    return int(np.ceil(area_km2 * density_per_km2))


def remap_negligible_rfi_dbw(dbw: np.ndarray) -> np.ndarray:
    """Map numerical zero-power floor (~-1000 dBW from src/) to demo display floor."""
    out = np.asarray(dbw, dtype=float).copy()
    out[out <= -999.0] = NEGLIGIBLE_RFI_DBW
    return out


# ---------------------------------------------------------------------------
# Resource loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading K-Band antenna...")
def load_k_band_antenna() -> Optional[Antenna]:
    path = repo_data_path(K_BAND_CSV)
    if not os.path.isfile(path):
        return None
    return load_weather_sat_antenna_from_csv(path, eta_rad=0.99, valid_freqs=(20e9, 30e9))


@st.cache_resource(show_spinner="Loading V-Band antenna...")
def load_v_band_antenna() -> Optional[Antenna]:
    path = repo_data_path(V_BAND_CSV)
    if not os.path.isfile(path):
        return None
    return load_weather_sat_antenna_from_csv(path, eta_rad=0.99, valid_freqs=(40e9, 60e9))


@st.cache_resource(show_spinner="Loading JPSS trajectory...")
def load_jpss_trajectory() -> Optional[Trajectory]:
    path = repo_data_path(JPSS_FILE)
    if not os.path.isfile(path):
        return None
    return Trajectory.from_file(
        path,
        time_tag="timestamp",
        elevation_tag="elevations",
        azimuth_tag="azimuths",
        distance_tag="ranges_westford",
    )


@st.cache_resource(show_spinner="Building 5G sector antenna...")
def load_5g_emitter_antenna() -> Antenna:
    return create_5g_sector_antenna_pattern(
        gain_max=18.0,
        horiz_beamwidth=65.0,
        vert_beamwidth=10.0,
        eta_rad=0.8,
        valid_freqs=(1e9, 100e9),
    )


@st.cache_resource(show_spinner="Building Starlink back-lobe model...")
def load_starlink_antenna() -> Antenna:
    alphas = np.arange(0, 181)
    betas = np.arange(0, 351, 10)
    pat = antenna_mdl_ITU(39.3, 3.0, alphas, betas)
    return Antenna.from_dataframe(pat, 0.5, (10.7e9, 12.7e9))


def _make_instrument(antenna: Antenna, freq_hz: float) -> Instrument:
    bw = get_atms_bandwidth(freq_hz)
    return Instrument(antenna, T_PHY, freq_hz, bw, t_rx_k_band, freq_chan=1, coords=[])


@st.cache_resource(show_spinner="Creating weather satellite instruments...")
def load_k_instrument() -> Optional[Instrument]:
    ant = load_k_band_antenna()
    if ant is None:
        return None
    return _make_instrument(ant, FREQ_K_HZ)


@st.cache_resource(show_spinner="Creating weather satellite instruments...")
def load_v_instrument() -> Optional[Instrument]:
    ant = load_v_band_antenna()
    if ant is None:
        return None
    return _make_instrument(ant, FREQ_V_HZ)


def _load_starlink_dataframe() -> Optional[pd.DataFrame]:
    path = repo_data_path(STARLINK_FILE)
    if not os.path.isfile(path):
        return None
    import pyarrow as pa

    with pa.memory_map(path, "r") as src:
        table = pa.ipc.open_file(src).read_all()
    df = table.to_pandas()
    df = df.rename(
        columns={
            "timestamp": "times",
            "ranges_westford": "distances",
        }
    )
    df["times"] = pd.to_datetime(df["times"])
    return df


@st.cache_data(show_spinner=False)
def filter_starlink_obs_data() -> Optional[pd.DataFrame]:
    """Filter Starlink rows to obs window, elev > STARLINK_ELEV_MIN_DEG, no DTC."""
    raw = _load_starlink_dataframe()
    if raw is None or raw.empty:
        return None
    mask = (
        (raw["times"] >= OBS_START)
        & (raw["times"] <= OBS_END)
        & (raw["elevations"] > STARLINK_ELEV_MIN_DEG)
        & (~raw["sat"].astype(str).str.contains("DTC", na=False))
    )
    return raw.loc[mask].copy().reset_index(drop=True)


@st.cache_resource(show_spinner="Building Starlink constellation...")
def load_starlink_constellation() -> Optional[Constellation]:
    starlink_obs_data = filter_starlink_obs_data()
    if starlink_obs_data is None or starlink_obs_data.empty:
        return None

    starlink_ant = load_starlink_antenna()
    starlink_transmit_pow = -15 + 10 * np.log10(300)

    def starlink_transmit_temp(_tim, _freq):
        return power_to_temperature(10 ** (starlink_transmit_pow / 10.0), 1.0)

    starlink_transmitter = Instrument(
        starlink_ant,
        0.0,
        11.9e9,
        250e6,
        starlink_transmit_temp,
        freq_chan=1,
        coords=[],
    )

    dummy_traj_data = pd.DataFrame(
        {
            "times": [OBS_START, OBS_END],
            "azimuths": [0.0, 0.0],
            "elevations": [90.0, 90.0],
            "distances": [1e6, 1e6],
        }
    )
    dummy_traj = Trajectory(dummy_traj_data)
    dummy_instrument = Instrument(
        starlink_ant,
        300.0,
        11.7e9,
        1e6,
        lambda _t, _f: 0.0,
        freq_chan=1,
        coords=[],
    )
    dummy_obs = Observation.from_dates(OBS_START, OBS_END, dummy_traj, dummy_instrument)

    return Constellation.from_observation(
        starlink_obs_data,
        dummy_obs,
        starlink_transmitter,
        lnk_bdgt_mdl=None,
        filt_funcs=(),
    )


@st.cache_data(show_spinner=False)
def observation_time_grid() -> Tuple[List[datetime], np.ndarray]:
    """
    ~10 s grid aligned to bundled trajectory timestamps (1 Hz subsample).
    """
    traj = load_jpss_trajectory()
    if traj is None:
        times = pd.date_range(OBS_START, OBS_END, freq=f"{DEMO_TIME_STEP_S}s")
        minutes = np.array(
            [(t - OBS_START).total_seconds() / 60.0 for t in times.to_pydatetime()]
        )
        return list(times.to_pydatetime()), minutes

    df = traj.get_traj_between(OBS_START, OBS_END)
    if df.empty:
        times = pd.date_range(OBS_START, OBS_END, freq=f"{DEMO_TIME_STEP_S}s")
        minutes = np.array(
            [(t - OBS_START).total_seconds() / 60.0 for t in times.to_pydatetime()]
        )
        return list(times.to_pydatetime()), minutes

    unique_times = df["times"].drop_duplicates().sort_values().reset_index(drop=True)
    indices = list(range(0, len(unique_times), DEMO_TIME_STEP_S))
    if indices[-1] != len(unique_times) - 1:
        indices.append(len(unique_times) - 1)
    obs_times = unique_times.iloc[indices].tolist()
    minutes = np.array([(t - OBS_START).total_seconds() / 60.0 for t in obs_times])
    return obs_times, minutes


def starlink_count_at_time(obs_time: datetime) -> int:
    df = filter_starlink_obs_data()
    if df is None or df.empty:
        return 0
    ts = pd.Timestamp(obs_time)
    at_t = df[df["times"] == ts]
    if at_t.empty:
        diffs = (df["times"] - ts).abs()
        nearest = df.loc[diffs.idxmin(), "times"]
        at_t = df[df["times"] == nearest]
    return len(at_t)


def _ws_ecef_at_time(
    ws_ecef_pos: pd.DataFrame, obs_time: datetime
) -> np.ndarray:
    ws_mask = ws_ecef_pos["times"] == obs_time
    if ws_mask.any():
        return ws_ecef_pos[ws_mask].iloc[0][["x", "y", "z"]].values.astype(float)
    time_diffs = np.abs((ws_ecef_pos["times"] - obs_time).dt.total_seconds())
    nearest_idx = time_diffs.idxmin()
    return ws_ecef_pos.loc[nearest_idx][["x", "y", "z"]].values.astype(float)


def _run_phase2_starlink(
    freq_hz: float,
    instrument: Instrument,
    constellation: Constellation,
    obs_times: List[datetime],
    starlink_freq_hz: float,
    starlink_eirp_dbw: float,
) -> np.ndarray:
    obs_array = np.array(obs_times, dtype=object)
    pol_factor = 10 ** (-STARLINK_POL_LOSS_DB / 10.0)
    traj = load_jpss_trajectory()
    if traj is None:
        return np.full(len(obs_times), np.nan)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = model_weather_sat_observed_power_phase2(
            weather_sat_trajectory=traj,
            weather_sat_instrument=instrument,
            starlink_constellation=constellation,
            observation_times=obs_array,
            observer_lat=OBSERVER_LAT,
            observer_lon=OBSERVER_LON,
            observer_alt=OBSERVER_ALT,
            target_lat=TARGET_LAT,
            target_lon=TARGET_LON,
            target_alt=TARGET_ALT,
            freq_channels=np.array([freq_hz]),
            ground_emitters=None,
            ground_emitter_antenna=None,
            starlink_eirp_dbw=starlink_eirp_dbw,
            enable_terrain_masking=False,
            include_atmospheric_loss=False,
            dem_file=None,
            polarization_loss_factor=pol_factor,
            starlink_fundamental_freq=starlink_freq_hz,
            harmonics=STARLINK_HARMONICS,
        )
    return np.asarray(result["starlink"][:, 0], dtype=float)


def _equivalent_5g_rfi_dbw_series(
    freq_hz: float,
    instrument: Instrument,
    obs_times: List[datetime],
    density: float,
    eirp_dbw: float,
    fundamental_hz: float,
) -> np.ndarray:
    n_emit = n_emitters_from_density(density)
    if n_emit <= 0 or density <= 0:
        return np.full(len(obs_times), -100.0)

    traj = load_jpss_trajectory()
    if traj is None:
        return np.full(len(obs_times), np.nan)

    ws_antenna = instrument.get_antenna()
    emitter_ant = load_5g_emitter_antenna()
    bandwidth = instrument.get_bandwidth()
    emitter_ecef = latlonalt_to_ecef(
        TARGET_LAT, TARGET_LON, TARGET_ALT + EMITTER_TOWER_ALT_M
    )
    # Beam boresight = FOV center (same lat/lon/alt as equivalent emitter).
    boresight_target_ecef = emitter_ecef

    ws_ecef_pos, _ = compute_weather_sat_ecef_from_trajectory(
        traj, OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT
    )

    emitter_power_w = 10 ** (eirp_dbw / 10.0)
    emitter_temp = power_to_temperature(emitter_power_w, bandwidth)

    out = np.zeros(len(obs_times), dtype=float)
    for i, obs_time in enumerate(obs_times):
        ws_ecef = _ws_ecef_at_time(ws_ecef_pos, obs_time)
        link_budget = ground_emitter_to_weather_sat_link_budget(
            emitter_ecef,
            ws_ecef,
            boresight_target_ecef,
            ws_antenna,
            emitter_ant,
            freq_hz,
            include_atmospheric_loss=False,
            emitter_fundamental_freq=fundamental_hz,
            harmonics=FIVE_G_HARMONICS,
            observation_bandwidth=bandwidth,
        )
        temp_total = n_emit * link_budget * emitter_temp
        power_w = temperature_to_power(temp_total, bandwidth)
        out[i] = 10.0 * np.log10(max(power_w, _RFI_ZERO_POWER_W))
    return out


@st.cache_data(show_spinner="Computing RFI time series...", max_entries=32)
def compute_rfi_time_series(
    starlink_freq_ghz: float,
    starlink_eirp_dbw: float,
    five_g_freq_ghz: float,
    five_g_eirp_dbw: float,
    emitter_density: float,
) -> dict:
    """
    Full Starlink + equivalent 5G RFI series for K and V bands.

    Returns dict with minute axis, dBW arrays, n_emitters, and obs_times iso list.
    """
    k_inst = load_k_instrument()
    v_inst = load_v_instrument()
    constellation = load_starlink_constellation()
    obs_times, minutes = observation_time_grid()

    missing = k_inst is None or v_inst is None
    if missing:
        nan = np.full(len(obs_times), np.nan)
        return {
            "minutes": minutes,
            "obs_times_iso": [t.isoformat() for t in obs_times],
            "k_starlink_dbw": nan,
            "k_5g_dbw": nan,
            "v_starlink_dbw": nan,
            "v_5g_dbw": nan,
            "n_emitters": n_emitters_from_density(emitter_density),
            "missing_data": True,
        }

    sl_freq_hz = starlink_freq_ghz * 1e9
    fg_hz = five_g_freq_ghz * 1e9
    n_emit = n_emitters_from_density(emitter_density)

    if constellation is None:
        k_sl = np.full(len(obs_times), -100.0)
        v_sl = np.full(len(obs_times), -100.0)
    else:
        k_sl = _run_phase2_starlink(
            FREQ_K_HZ,
            k_inst,
            constellation,
            obs_times,
            sl_freq_hz,
            starlink_eirp_dbw,
        )
        v_sl = _run_phase2_starlink(
            FREQ_V_HZ,
            v_inst,
            constellation,
            obs_times,
            sl_freq_hz,
            starlink_eirp_dbw,
        )

    k_5g = _equivalent_5g_rfi_dbw_series(
        FREQ_K_HZ,
        k_inst,
        obs_times,
        emitter_density,
        five_g_eirp_dbw,
        fg_hz,
    )
    v_5g = _equivalent_5g_rfi_dbw_series(
        FREQ_V_HZ,
        v_inst,
        obs_times,
        emitter_density,
        five_g_eirp_dbw,
        fg_hz,
    )

    return {
        "minutes": minutes,
        "obs_times_iso": [t.isoformat() for t in obs_times],
        "k_starlink_dbw": remap_negligible_rfi_dbw(k_sl),
        "k_5g_dbw": remap_negligible_rfi_dbw(k_5g),
        "v_starlink_dbw": remap_negligible_rfi_dbw(v_sl),
        "v_5g_dbw": remap_negligible_rfi_dbw(v_5g),
        "n_emitters": n_emit,
        "missing_data": False,
    }
