"""
Configuration parameters for RSC-SIM tutorials.

This module contains all the shared parameters, constants, and configuration
values used across the tutorial scripts.
"""

import os
from datetime import datetime, timedelta

# =============================================================================
# PATHS AND FILES
# =============================================================================

# Base directory for tutorial data
TUTORIAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tutorial", "data")

# Antenna pattern file
ANTENNA_PATTERN_FILE = os.path.join(TUTORIAL_DATA_DIR, "single_cut_res.cut")

# Trajectory files
CAS_A_TRAJECTORY_FILE = os.path.join(
    TUTORIAL_DATA_DIR,
    "casA_trajectory_Westford_2025-02-18T15_00_00.000_2025-02-18T15_45_00.000.arrow"
)

STARLINK_TRAJECTORY_FILE = os.path.join(
    TUTORIAL_DATA_DIR,
    "Starlink_trajectory_Westford_2025-02-18T15_00_00.000_2025-02-18T15_45_00.000.arrow"
)

# =============================================================================
# OBSERVATION PARAMETERS
# =============================================================================

# Observation time window
OBSERVATION_START = datetime.strptime("2025-02-18T15:30:00.000", "%Y-%m-%dT%H:%M:%S.%f")
OBSERVATION_END = datetime.strptime("2025-02-18T15:40:00.000", "%Y-%m-%dT%H:%M:%S.%f")

# Source tracking parameters
OFFSET_ANGLES = (-40, 0.)  # (azimuth, elevation) offset in degrees
TIME_ON_SOURCE = OBSERVATION_START + timedelta(minutes=5)

# Elevation filter
MIN_ELEVATION = 5.0  # degrees

# =============================================================================
# TELESCOPE PARAMETERS
# =============================================================================

# Antenna parameters
TELESCOPE_RADIATION_EFFICIENCY = 0.45
TELESCOPE_FREQ_BAND = (10e9, 12e9)  # Hz
TELESCOPE_PHYSICAL_TEMP = 300.0  # K

# Receiver parameters
CENTER_FREQUENCY = 11.325e9  # Hz
BANDWIDTH = 1e3  # Hz
FREQUENCY_CHANNELS = 1
RECEIVER_TEMP = 80.0  # K

# Telescope coordinates (Westford)
TELESCOPE_COORDS = [42.6129479883915, -71.49379366344017, 86.7689687917009]

# =============================================================================
# SATELLITE PARAMETERS
# =============================================================================

# Satellite antenna parameters
SATELLITE_RADIATION_EFFICIENCY = 0.5
SATELLITE_MAX_GAIN = 39.3  # dBi
SATELLITE_HALF_BEAMWIDTH = 3.0  # degrees
SATELLITE_PHYSICAL_TEMP = 0.0  # K

# Satellite transmission parameters
SATELLITE_FREQUENCY = 11.325e9  # Hz
SATELLITE_BANDWIDTH = 250e6  # Hz
SATELLITE_TRANSMIT_POWER = -15 + 10 * 2.477  # dBW (10*log10(300))

# =============================================================================
# SKY MODEL PARAMETERS
# =============================================================================

# Atmospheric parameters
ATMOSPHERIC_TEMP_ZENITH = 150.0  # K
ATMOSPHERIC_OPACITY = 0.05

# Background temperatures
CMB_TEMPERATURE = 2.73  # K
GALACTIC_TEMP_REF_FREQ = 1.41e9  # Hz
GALACTIC_TEMP_REF_VALUE = 1e-1  # K
GALACTIC_TEMP_SPECTRAL_INDEX = -2.7

# RFI parameters
GROUND_RFI_TEMP = 0.0  # K
VARIABLE_RFI_TEMP = 0.0  # K

# =============================================================================
# PLOTTING PARAMETERS
# =============================================================================

# Figure sizes
STANDARD_FIGURE_SIZE = (10, 6)
LARGE_FIGURE_SIZE = (16, 8)
POLAR_FIGURE_SIZE = (10, 10)

# Color maps
DEFAULT_COLORMAP = "plasma"
TEMPERATURE_COLORMAP = "plasma"
POWER_COLORMAP = "viridis"

# =============================================================================
# COMPUTATION PARAMETERS
# =============================================================================

# Sky mapping grid
AZIMUTH_GRID_STEP = 5  # degrees
ELEVATION_GRID_STEP = 1  # degrees

# PSD parameters
PSD_FREQUENCY_CHANNELS = 164
PSD_BANDWIDTH = 30e6  # Hz

# Beam avoidance
BEAM_AVOIDANCE_ANGLE = 10.0  # degrees
