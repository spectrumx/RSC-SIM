"""
Shared utilities for modular RadioMdlPy tutorials.

This package contains common functions and configurations used across
multiple tutorial scripts to avoid code duplication and ensure consistency.
"""

from .instrument_setup import (
    setup_westford_telescope,
    setup_satellite_transmitter,
    setup_constant_gain_satellite,
    setup_psd_instrument,
    setup_frequency_dependent_satellite
)
from .sky_models import (
    create_sky_model,
    create_atmospheric_model,
    create_galactic_background_model,
    create_background_model,
    create_source_model,
    create_rfi_model,
    create_sky_model_without_source,
    create_directional_source_model
)
from .plotting_utils import (
    setup_plotting,
    create_polar_plot,
    plot_antenna_pattern,
    plot_trajectory_comparison,
    plot_sky_temperature_map,
    plot_power_time_series,
    plot_psd_spectrogram,
    plot_satellite_positions,
    plot_satellite_trajectories,
    safe_log10
)
from .config import (
    OBSERVATION_START,
    OBSERVATION_END,
    OFFSET_ANGLES,
    TIME_ON_SOURCE,
    MIN_ELEVATION,
    BANDWIDTH,
    CENTER_FREQUENCY,
    TELESCOPE_COORDS,
    ANTENNA_PATTERN_FILE,
    CAS_A_TRAJECTORY_FILE,
    STARLINK_TRAJECTORY_FILE,
    AZIMUTH_GRID_STEP,
    ELEVATION_GRID_STEP
)

__all__ = [
    # Instrument setup functions
    'setup_westford_telescope',
    'setup_satellite_transmitter',
    'setup_constant_gain_satellite',
    'setup_psd_instrument',
    'setup_frequency_dependent_satellite',

    # Sky model functions
    'create_sky_model',
    'create_atmospheric_model',
    'create_galactic_background_model',
    'create_background_model',
    'create_source_model',
    'create_rfi_model',
    'create_sky_model_without_source',
    'create_directional_source_model',

    # Plotting functions
    'setup_plotting',
    'create_polar_plot',
    'plot_antenna_pattern',
    'plot_trajectory_comparison',
    'plot_sky_temperature_map',
    'plot_power_time_series',
    'plot_psd_spectrogram',
    'plot_satellite_positions',
    'plot_satellite_trajectories',
    'safe_log10',

    # Configuration constants (from config.py)
    'OBSERVATION_START',
    'OBSERVATION_END',
    'OFFSET_ANGLES',
    'TIME_ON_SOURCE',
    'MIN_ELEVATION',
    'BANDWIDTH',
    'CENTER_FREQUENCY',
    'TELESCOPE_COORDS',
    'ANTENNA_PATTERN_FILE',
    'CAS_A_TRAJECTORY_FILE',
    'STARLINK_TRAJECTORY_FILE',
    'AZIMUTH_GRID_STEP',
    'ELEVATION_GRID_STEP'
]
