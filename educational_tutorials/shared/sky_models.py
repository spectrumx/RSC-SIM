"""
Sky model utilities for RSC-SIM tutorials.

This module provides functions to create sky temperature models including
astronomical sources, atmospheric effects, and background radiation.
"""

import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from astro_mdl import estim_casA_flux, estim_temp  # noqa: E402
from .config import (  # noqa: E402
    ATMOSPHERIC_TEMP_ZENITH,
    ATMOSPHERIC_OPACITY,
    GALACTIC_TEMP_REF_VALUE,
    GALACTIC_TEMP_REF_FREQ,
    GALACTIC_TEMP_SPECTRAL_INDEX,
    CMB_TEMPERATURE,
    GROUND_RFI_TEMP,
    VARIABLE_RFI_TEMP,
    CENTER_FREQUENCY,
    TIME_ON_SOURCE
)


def create_atmospheric_model():
    """
    Create atmospheric temperature model.

    Returns:
        function: Atmospheric temperature function T_atm(dec)
    """
    def T_atm(dec):
        """Atmospheric temperature model varying with elevation angle."""
        return ATMOSPHERIC_TEMP_ZENITH * (1 - np.exp(-ATMOSPHERIC_OPACITY/np.cos(dec)))

    return T_atm


def create_galactic_background_model():
    """
    Create galactic background temperature model.

    Returns:
        function: Galactic temperature function T_gal(freq)
    """
    def T_gal(freq):
        """Galactic background temperature with spectral index."""
        return GALACTIC_TEMP_REF_VALUE * (freq/GALACTIC_TEMP_REF_FREQ)**GALACTIC_TEMP_SPECTRAL_INDEX

    return T_gal


def create_background_model():
    """
    Create combined background temperature model (CMB + Galactic).

    Returns:
        function: Background temperature function T_bkg(freq)
    """
    T_gal = create_galactic_background_model()

    def T_bkg(freq):
        """Combined background temperature (CMB + Galactic)."""
        return CMB_TEMPERATURE + T_gal(freq)

    return T_bkg


def create_source_model(observation):
    """
    Create astronomical source temperature model for Cas A.

    Args:
        observation: Observation object for flux calculation

    Returns:
        function: Source temperature function T_src(t)
    """
    # Calculate source flux
    flux_src = estim_casA_flux(CENTER_FREQUENCY)

    # Get effective aperture from antenna gain
    from antenna_pattern import gain_to_effective_aperture
    antenna = observation.get_instrument().get_antenna()
    max_gain = antenna.get_boresight_gain()
    effective_aperture = gain_to_effective_aperture(max_gain, CENTER_FREQUENCY)

    def T_src(t):
        """Source temperature model with ON/OFF source behavior."""
        if t <= TIME_ON_SOURCE:
            return 0.0  # OFF source
        else:
            return estim_temp(flux_src, effective_aperture)  # ON source

    return T_src


def create_rfi_model():
    """
    Create Radio Frequency Interference model.

    Returns:
        float: RFI temperature (currently set to 0)
    """
    return GROUND_RFI_TEMP + VARIABLE_RFI_TEMP


def create_sky_model(observation):
    """
    Create complete sky temperature model.

    Args:
        observation: Observation object for source calculations

    Returns:
        function: Complete sky model function sky_mdl(dec, caz, tim, freq)
    """
    T_src = create_source_model(observation)
    T_atm = create_atmospheric_model()
    T_bkg = create_background_model()
    T_rfi = create_rfi_model()

    def sky_mdl(dec, caz, tim, freq):
        """
        Complete sky temperature model.

        Args:
            dec: Declination angle (radians)
            caz: Co-azimuth angle (radians)
            tim: Time (datetime)
            freq: Frequency (Hz)

        Returns:
            float: Sky temperature (K)
        """
        return T_src(tim) + T_atm(dec) + T_rfi + T_bkg(freq)

    return sky_mdl


def create_sky_model_without_source():
    """
    Create sky temperature model without astronomical source.

    Returns:
        function: Sky model function without source component
    """
    T_atm = create_atmospheric_model()
    T_bkg = create_background_model()
    T_rfi = create_rfi_model()

    def sky_mdl_no_source(dec, caz, tim, freq):
        """
        Sky temperature model without astronomical source.

        Args:
            dec: Declination angle (radians)
            caz: Co-azimuth angle (radians)
            tim: Time (datetime)
            freq: Frequency (Hz)

        Returns:
            float: Sky temperature (K)
        """
        return T_atm(dec) + T_rfi + T_bkg(freq)

    return sky_mdl_no_source


def create_directional_source_model(observation, source_trajectory, beamwidth=1.0):
    """
    Create directional source model for sky mapping.

    Args:
        observation: Observation object
        source_trajectory: Source trajectory object
        beamwidth: Telescope beamwidth in degrees

    Returns:
        function: Directional source model function
    """
    # Calculate source flux
    flux_src = estim_casA_flux(CENTER_FREQUENCY)

    def T_src_directional(dec, caz, tim, freq):
        """
        Directional source temperature model for sky mapping.

        Args:
            dec: Declination angle (radians)
            caz: Co-azimuth angle (radians)
            tim: Time (datetime)
            freq: Frequency (Hz)

        Returns:
            float: Source temperature (K)
        """
        # Get source position at given time
        src_traj = source_trajectory.get_traj()
        src_at_time = src_traj[src_traj['times'] == tim]

        if len(src_at_time) > 0:
            dec_src = np.pi/2 - np.radians(src_at_time.iloc[0]['elevations'])
            caz_src = -np.radians(src_at_time.iloc[0]['azimuths'])

            # Check if pointing is within beam of source
            mask = (np.abs(dec - dec_src) < np.radians(beamwidth)) & \
                   (np.abs(caz - caz_src) < np.radians(beamwidth))

            # Create output array
            out = np.zeros_like(dec)
            out[mask] = estim_temp(flux_src, observation)
            return out

        return np.zeros_like(dec)

    return T_src_directional
