"""
Module-level singleton holding heavy non-serializable RSC-SIM objects.

Reflex State fields must be JSON-serializable (int/float/str/list/dict).
Custom objects (Antenna, Trajectory, DataFrame, Constellation) live here,
NOT in any rx.State subclass.

Thread-safe: load_all() uses _lock to prevent concurrent initialization.
"""

from __future__ import annotations

import threading


class _ResourceStore:
    """Process-wide singleton for RSC-SIM heavy resources."""

    _lock = threading.Lock()
    _loaded = False
    _load_error: str = ""

    # --- Radio astronomy tab ---
    westford_antenna = None          # Antenna
    starlink_transmitter_itu = None  # Instrument
    cas_a_trajectory = None          # Trajectory
    pointing_trajectory = None       # Trajectory (with ON/OFF offsets)
    starlink_df_radio = None         # pd.DataFrame (16 MB Arrow)

    # --- Weather satellite tab ---
    k_band_antenna = None            # Antenna (K-Band 23.8 GHz)
    v_band_antenna = None            # Antenna (V-Band 50.3 GHz)
    k_instrument = None              # Instrument (K-Band)
    v_instrument = None              # Instrument (V-Band)
    jpss_trajectory = None           # Trajectory (JPSS weather sat)
    starlink_antenna_weather = None  # Antenna (Starlink back-lobe model)
    emitter_5g_antenna = None        # Antenna (5G sector pattern)
    starlink_constellation_wx = None # Constellation (weather tab)
    starlink_obs_data_wx = None      # pd.DataFrame (filtered Starlink obs)
    obs_times = None                 # List[datetime] (time grid)
    obs_minutes = None               # np.ndarray (minutes from obs start)

    # --- Shared / UI ---
    satellite_names: list = []       # For dropdown and slider max

    @classmethod
    def load_all(cls) -> None:
        """Load all heavy resources once. Idempotent and thread-safe."""
        with cls._lock:
            if cls._loaded:
                return
            try:
                cls._load_all_unsafe()
                cls._loaded = True
                cls._load_error = ""
            except Exception as exc:
                cls._load_error = str(exc)
                raise

    @classmethod
    def _load_all_unsafe(cls) -> None:
        from reflex_app.utils.sim_cache_reflex import (
            load_westford_antenna,
            load_starlink_transmitter_itu,
            load_cas_a_trajectory,
            load_pointing_trajectory,
            load_starlink_trajectory_df,
            list_starlink_satellite_names,
        )
        from reflex_app.utils.weather_loaders_reflex import (
            load_k_band_antenna,
            load_v_band_antenna,
            load_k_instrument,
            load_v_instrument,
            load_jpss_trajectory,
            load_starlink_antenna,
            load_5g_emitter_antenna,
            load_starlink_constellation,
            filter_starlink_obs_data,
            observation_time_grid,
        )

        # Radio tab
        cls.westford_antenna = load_westford_antenna()
        cls.starlink_transmitter_itu = load_starlink_transmitter_itu()
        cls.cas_a_trajectory = load_cas_a_trajectory()
        cls.pointing_trajectory = load_pointing_trajectory()
        cls.starlink_df_radio = load_starlink_trajectory_df()

        # Weather tab
        cls.k_band_antenna = load_k_band_antenna()
        cls.v_band_antenna = load_v_band_antenna()
        cls.k_instrument = load_k_instrument()
        cls.v_instrument = load_v_instrument()
        cls.jpss_trajectory = load_jpss_trajectory()
        cls.starlink_antenna_weather = load_starlink_antenna()
        cls.emitter_5g_antenna = load_5g_emitter_antenna()
        cls.starlink_constellation_wx = load_starlink_constellation()
        cls.starlink_obs_data_wx = filter_starlink_obs_data()
        cls.obs_times, cls.obs_minutes = observation_time_grid()

        # UI
        cls.satellite_names = list_starlink_satellite_names()

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded

    @classmethod
    def load_error(cls) -> str:
        return cls._load_error


RESOURCES = _ResourceStore()
