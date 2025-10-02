"""
Instrument setup utilities for RSC-SIM tutorials.

This module provides functions to set up telescope and satellite instruments
with consistent parameters across all tutorials.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from radio_types import Antenna, Instrument  # noqa: E402
from astro_mdl import antenna_mdl_ITU, power_to_temperature  # noqa: E402
from .config import (  # noqa: E402
    ANTENNA_PATTERN_FILE,
    TELESCOPE_RADIATION_EFFICIENCY,
    TELESCOPE_FREQ_BAND,
    TELESCOPE_PHYSICAL_TEMP,
    CENTER_FREQUENCY,
    BANDWIDTH,
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
    PSD_FREQUENCY_CHANNELS,
    PSD_BANDWIDTH
)  # noqa: E402


def setup_westford_telescope():
    """
    Set up the Westford telescope instrument with standard parameters.

    Returns:
        Instrument: Configured telescope instrument
    """
    # Load telescope antenna from file
    tel_ant = Antenna.from_file(
        ANTENNA_PATTERN_FILE,
        TELESCOPE_RADIATION_EFFICIENCY,
        TELESCOPE_FREQ_BAND,
        power_tag='power',
        declination_tag='alpha',
        azimuth_tag='beta'
    )

    # Define receiver temperature function
    def T_RX(tim, freq):
        return RECEIVER_TEMP

    # Create instrument
    telescope = Instrument(
        tel_ant,
        TELESCOPE_PHYSICAL_TEMP,
        CENTER_FREQUENCY,
        BANDWIDTH,
        T_RX,
        FREQUENCY_CHANNELS,
        TELESCOPE_COORDS
    )

    return telescope


def setup_satellite_transmitter():
    """
    Set up satellite transmitter instrument with ITU gain model.

    Returns:
        Instrument: Configured satellite transmitter
    """
    # Create ITU recommended gain profile
    alphas = np.arange(0, 181)
    betas = np.arange(0, 351, 10)

    gain_pat = antenna_mdl_ITU(
        SATELLITE_MAX_GAIN,
        SATELLITE_HALF_BEAMWIDTH,
        alphas,
        betas
    )

    # Create satellite antenna
    sat_ant = Antenna.from_dataframe(gain_pat, SATELLITE_RADIATION_EFFICIENCY, TELESCOPE_FREQ_BAND)

    # Define transmission temperature function
    def transmit_temp(tim, freq):
        return power_to_temperature(10**(SATELLITE_TRANSMIT_POWER/10), 1.0)

    # Create transmitter instrument
    transmitter = Instrument(
        sat_ant,
        SATELLITE_PHYSICAL_TEMP,
        SATELLITE_FREQUENCY,
        SATELLITE_BANDWIDTH,
        transmit_temp,
        1,
        []
    )

    return transmitter


def setup_constant_gain_satellite():
    """
    Set up satellite transmitter with constant (minimum) gain for comparison.

    Returns:
        Instrument: Configured satellite transmitter with constant gain
    """
    # Create constant gain profile
    alphas = np.arange(0, 181)
    betas = np.arange(0, 351, 10)

    # Get minimum gain from standard satellite antenna
    sat_ant_std = setup_satellite_transmitter().get_antenna()
    sat_gain_min = sat_ant_std.gain_pat['gains'].min()

    # Create gain dataframe with constant minimum gain
    gain_pat = pd.DataFrame({
        'alphas': np.repeat(alphas, len(betas)),
        'betas': np.tile(betas, len(alphas)),
        'gains': np.full(len(alphas) * len(betas), sat_gain_min)
    })

    # Create satellite antenna
    sat_ant = Antenna.from_dataframe(gain_pat, SATELLITE_RADIATION_EFFICIENCY, TELESCOPE_FREQ_BAND)

    # Define transmission temperature function
    def transmit_temp(tim, freq):
        return power_to_temperature(10**(SATELLITE_TRANSMIT_POWER/10), 1.0)

    # Create transmitter instrument
    transmitter = Instrument(
        sat_ant,
        SATELLITE_PHYSICAL_TEMP,
        SATELLITE_FREQUENCY,
        SATELLITE_BANDWIDTH,
        transmit_temp,
        1,
        []
    )

    return transmitter


def setup_psd_instrument():
    """
    Set up instrument for Power Spectral Density analysis.

    Returns:
        Instrument: Configured telescope instrument for PSD
    """
    # Load telescope antenna
    tel_ant = Antenna.from_file(
        ANTENNA_PATTERN_FILE,
        TELESCOPE_RADIATION_EFFICIENCY,
        TELESCOPE_FREQ_BAND,
        power_tag='power',
        declination_tag='alpha',
        azimuth_tag='beta'
    )

    # Define receiver temperature function
    def T_RX(tim, freq):
        return RECEIVER_TEMP

    # Create instrument with PSD parameters
    telescope = Instrument(
        tel_ant,
        TELESCOPE_PHYSICAL_TEMP,
        CENTER_FREQUENCY,
        PSD_BANDWIDTH,
        T_RX,
        PSD_FREQUENCY_CHANNELS,
        TELESCOPE_COORDS
    )

    return telescope


def setup_frequency_dependent_satellite():
    """
    Set up satellite transmitter with frequency-dependent transmission pattern.

    Returns:
        Instrument: Configured satellite transmitter with frequency-dependent pattern
    """
    # Create standard satellite antenna
    sat_ant = setup_satellite_transmitter().get_antenna()

    # Create frequency-dependent transmission profile
    tmt_profile = np.ones(PSD_FREQUENCY_CHANNELS)
    tmt_profile[:PSD_FREQUENCY_CHANNELS//10] = 0.0
    tmt_profile[-PSD_FREQUENCY_CHANNELS//10:] = 0.0
    tmt_profile[PSD_FREQUENCY_CHANNELS//2 - PSD_FREQUENCY_CHANNELS//10:
                PSD_FREQUENCY_CHANNELS//2 + PSD_FREQUENCY_CHANNELS//10] = 0.0
    tmt_profile[PSD_FREQUENCY_CHANNELS//2] = 1.0

    # Get frequency bins
    psd_instrument = setup_psd_instrument()
    freq_bins = psd_instrument.get_center_freq_chans()

    # Define frequency-dependent transmission temperature function
    def transmit_temp_freqs(tim, freq):
        ind_freq = np.argmin(np.abs(freq_bins - freq))
        return tmt_profile[ind_freq] * power_to_temperature(10**(SATELLITE_TRANSMIT_POWER/10), 1.0)

    # Create transmitter instrument
    transmitter = Instrument(
        sat_ant,
        SATELLITE_PHYSICAL_TEMP,
        SATELLITE_FREQUENCY,
        SATELLITE_BANDWIDTH,
        transmit_temp_freqs,
        PSD_FREQUENCY_CHANNELS,
        []
    )

    return transmitter
