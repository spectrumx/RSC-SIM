"""
ITU-R atmospheric attenuation utilities for weather satellite RFI modeling.

- **P.676** — Gaseous absorption: ``ITUP676Calculator``, ``get_cached_calculator``.
- **P.840-9 / P.838-3** — Cloud and rain slant attenuation (vectorized NumPy):
  ``cloud_attenuation_db``, ``rain_attenuation_db``, ``compute_cloud_rain_attenuation_db``,
  plus defaults (path-length range, P.838 coefficient tables). Gating threshold
  ``iclw_abs_threshold`` is supplied by the caller.

Usage (P.676):
    from attenuation_mdl import ITUP676Calculator, get_cached_calculator
    calc = ITUP676Calculator()
    attenuation_db = calc.total_slant_attenuation(freq_ghz=50.3, elevation_deg=30.0)

Usage (cloud / rain):
    from attenuation_mdl import compute_cloud_rain_attenuation_db
    # iclw_abs_threshold is required (set in your application / sensor script).

ITU ICLW / rain NetCDF grids and per-FOV slant attenuation for RFI pipelines:
``load_iclw_grid``, ``load_itu_rain_grid``, ``itu_iclw_rain_info_nc_path``,
``compute_cloud_rain_atten_db_for_fovs``, ``atten_db_to_by_channel_dict``.

References:
    ITU-R P.676-13 (08/2022); P.840-9 (cloud); P.838-3 (rain coefficients).

Author: Weather Satellite RFI Modeling Team
"""

import os
import re

import numpy as np
import pandas as pd
from typing import Dict, Sequence

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None  # type: ignore[assignment, misc]

try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None  # type: ignore[assignment, misc]


# =============================================================================
# Equivalent Height Coefficients (ITU-R P.676 Part 1 Data File)
# =============================================================================

# Coefficients for equivalent height calculation
# Format: [c0, c1, c2, c3] for h0 = c0 + c1*T + c2*p + c3*rho
# These are for specific frequencies in the 48-55 GHz range (V-band)
EQUIVALENT_HEIGHT_COEFFS = {
    48.0: [-2.495838e+00, 2.841564e-02, -6.670666e-04, -1.475172e-03],
    48.5: [-2.460026e+00, 2.825847e-02, -6.678321e-04, -1.469660e-03],
    49.0: [-2.407158e+00, 2.802601e-02, -6.676717e-04, -1.459266e-03],
    49.5: [-2.323643e+00, 2.765616e-02, -6.646817e-04, -1.438567e-03],
    50.0: [-2.181135e+00, 2.700265e-02, -6.497764e-04, -1.354873e-03],
    50.5: [-1.965792e+00, 2.597955e-02, -6.115973e-04, -9.553682e-04],
    51.0: [-1.680298e+00, 2.465202e-02, -5.616553e-04, -2.626463e-04],
    51.5: [-1.325805e+00, 2.355260e-02, -6.127006e-04, 1.014900e-03],
    52.0: [-1.127561e+00, 2.199826e-02, -3.966258e-04, 2.428849e-03],
    52.5: [-1.001601e+00, 2.124219e-02, -2.582798e-04, 4.283698e-03],
    53.0: [-9.929553e-01, 2.125811e-02, -1.291180e-04, 6.155893e-03],
    53.5: [-1.080925e+00, 2.194315e-02, 9.481092e-06, 7.793931e-03],
    54.0: [-1.262827e+00, 2.318681e-02, 1.824770e-04, 9.094853e-03],
    54.5: [-1.549949e+00, 2.491395e-02, 4.046114e-04, 9.962579e-03],
    55.0: [-1.952223e+00, 2.718364e-02, 6.617421e-04, 1.033374e-02],
}


# =============================================================================
# Data Loading Functions
# =============================================================================

def _load_oxygen_spectroscopic_data(csv_path: str) -> pd.DataFrame:
    """
    Load oxygen spectroscopic data from CSV file (ITU-R P.676 Table 1).

    Args:
        csv_path: Path to table1_oxygen.csv file

    Returns:
        pd.DataFrame: DataFrame with columns [f0, a1, a2, a3, a4, a5, a6]
    """
    df = pd.read_csv(csv_path)
    expected_cols = ['f0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    if list(df.columns) != expected_cols:
        if len(df.columns) == 7:
            df.columns = expected_cols
    return df


def _load_water_vapor_spectroscopic_data(csv_path: str) -> pd.DataFrame:
    """
    Load water vapor spectroscopic data from CSV file (ITU-R P.676 Table 2).

    Args:
        csv_path: Path to table2_water_vapor.csv file

    Returns:
        pd.DataFrame: DataFrame with columns [f0, b1, b2, b3, b4, b5, b6]
    """
    # The CSV has an empty first row, skip it
    df = pd.read_csv(csv_path, skiprows=1)
    expected_cols = ['f0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    if list(df.columns) != expected_cols:
        if len(df.columns) == 7:
            df.columns = expected_cols
    return df


def _get_default_data_dir() -> str:
    """
    Get the default data directory containing spectroscopic CSV files.

    The data files are located in research_tutorials/data/ relative to
    the project root.
    """
    # Get the src directory (where this file is located)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to project root, then to research_tutorials/data
    project_root = os.path.dirname(src_dir)
    data_dir = os.path.join(project_root, 'research_tutorials', 'data')
    return data_dir


# =============================================================================
# Main Calculator Class
# =============================================================================

class ITUP676Calculator:
    """
    Optimized ITU-R P.676 calculator for high-performance batch calculations.

    This class pre-loads spectroscopic data and caches intermediate results
    to significantly speed up repeated calculations in weather satellite modeling.

    Features:
        - Pre-loads CSV data once at initialization (~10ms)
        - Pre-extracts numpy arrays for fast access (avoids DataFrame overhead)
        - Supports both single-value and batch calculations
        - ~100x faster than function-based API for repeated calls

    Usage:
        # Create calculator once (loads data from disk)
        calc = ITUP676Calculator()

        # Use for many calculations without reloading data
        gamma_o = calc.oxygen_attenuation(freq_ghz, pressure, temp, rho)
        gamma_w = calc.water_vapor_attenuation(freq_ghz, pressure, temp, rho)
        total_db = calc.total_slant_attenuation(freq_ghz, elev_deg, pressure, temp, rho)

    Reference:
        ITU-R P.676-13 (08/2022): Attenuation by atmospheric gases and related effects
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize calculator and pre-load spectroscopic data.

        Args:
            data_dir: Directory containing CSV data files (table1_oxygen.csv,
                      table2_water_vapor.csv). If None, uses default location
                      in research_tutorials/data/.
        """
        if data_dir is None:
            data_dir = _get_default_data_dir()

        self.data_dir = data_dir

        # Pre-load spectroscopic tables (one-time disk I/O)
        oxygen_csv = os.path.join(data_dir, 'table1_oxygen.csv')
        water_vapor_csv = os.path.join(data_dir, 'table2_water_vapor.csv')

        self._oxygen_table = _load_oxygen_spectroscopic_data(oxygen_csv)
        self._water_vapor_table = _load_water_vapor_spectroscopic_data(water_vapor_csv)

        # Pre-extract numpy arrays for faster access (avoid DataFrame overhead)
        self._o2_f0 = self._oxygen_table['f0'].values
        self._o2_a1 = self._oxygen_table['a1'].values
        self._o2_a2 = self._oxygen_table['a2'].values
        self._o2_a3 = self._oxygen_table['a3'].values
        self._o2_a4 = self._oxygen_table['a4'].values
        self._o2_a5 = self._oxygen_table['a5'].values
        self._o2_a6 = self._oxygen_table['a6'].values

        self._h2o_f0 = self._water_vapor_table['f0'].values
        self._h2o_b1 = self._water_vapor_table['b1'].values
        self._h2o_b2 = self._water_vapor_table['b2'].values
        self._h2o_b3 = self._water_vapor_table['b3'].values
        self._h2o_b4 = self._water_vapor_table['b4'].values
        self._h2o_b5 = self._water_vapor_table['b5'].values
        self._h2o_b6 = self._water_vapor_table['b6'].values

        # Pre-compute equivalent height coefficient arrays for interpolation
        self._eq_height_freqs = np.array(sorted(EQUIVALENT_HEIGHT_COEFFS.keys()))
        self._eq_height_coeffs = np.array([
            EQUIVALENT_HEIGHT_COEFFS[f] for f in self._eq_height_freqs
        ])

    def oxygen_attenuation(
        self,
        freq_ghz: float,
        pressure_hpa: float = 1013.25,
        temperature_k: float = 288.15,
        water_vapor_density: float = 7.5
    ) -> float:
        """
        Calculate oxygen specific attenuation (gamma_o) in dB/km.

        Implements ITU-R P.676-13 Annex 1 line-by-line calculation.

        Args:
            freq_ghz: Frequency in GHz
            pressure_hpa: Surface pressure in hPa (default: 1013.25)
            temperature_k: Surface temperature in Kelvin (default: 288.15 = 15°C)
            water_vapor_density: Surface water vapor density in g/m³ (default: 7.5)

        Returns:
            float: Oxygen specific attenuation in dB/km
        """
        # Pre-compute common values
        e_s = (water_vapor_density * temperature_k) / 216.7
        p_dry = pressure_hpa - e_s
        theta = 300.0 / temperature_k
        f = freq_ghz

        # Line strength (Equation 3)
        Si = ((self._o2_a1 * 1e-7) * p_dry * (theta ** 3) *
              np.exp(self._o2_a2 * (1 - theta)))

        # Line width (Equation 6a)
        delta_f = (self._o2_a3 * 1e-4 *
                   (p_dry * (theta ** (0.8 - self._o2_a4)) + 1.1 * e_s * theta))
        delta_f = np.sqrt(delta_f ** 2 + 2.25e-6)

        # Interference factor (Equation 7)
        sigma = ((self._o2_a5 + self._o2_a6 * theta) * 1e-4 *
                 (p_dry + e_s) * (theta ** 0.8))

        # Line shape factor (Equation 5)
        f0 = self._o2_f0
        term1 = (delta_f - sigma * (f0 - f)) / ((f0 - f) ** 2 + delta_f ** 2)
        term2 = (delta_f - sigma * (f0 + f)) / ((f0 + f) ** 2 + delta_f ** 2)
        Fi = (f / f0) * (term1 + term2)

        # Dry continuum (Equation 9)
        d = 5.6e-4 * (p_dry + e_s) * (theta ** 0.8)
        Nd = (f * p_dry * (theta ** 2) *
              ((6.14e-5 / (d * (1 + (f / d) ** 2))) +
               (1.4e-12 * p_dry * (theta ** 1.5) / (1 + 1.9e-5 * (f ** 1.5)))))

        # Sum and convert to attenuation
        N_O2 = np.sum(Si * Fi) + Nd
        return 0.1820 * f * N_O2

    def water_vapor_attenuation(
        self,
        freq_ghz: float,
        pressure_hpa: float = 1013.25,
        temperature_k: float = 288.15,
        water_vapor_density: float = 7.5
    ) -> float:
        """
        Calculate water vapor specific attenuation (gamma_w) in dB/km.

        Implements ITU-R P.676-13 Annex 1 line-by-line calculation.

        Args:
            freq_ghz: Frequency in GHz
            pressure_hpa: Surface pressure in hPa (default: 1013.25)
            temperature_k: Surface temperature in Kelvin (default: 288.15 = 15°C)
            water_vapor_density: Surface water vapor density in g/m³ (default: 7.5)

        Returns:
            float: Water vapor specific attenuation in dB/km
        """
        e_s = (water_vapor_density * temperature_k) / 216.7
        p_dry = pressure_hpa - e_s
        theta = 300.0 / temperature_k
        f = freq_ghz

        # Line strength (Equation 4)
        Si = ((self._h2o_b1 * 1e-1) * e_s * (theta ** 3.5) *
              np.exp(self._h2o_b2 * (1 - theta)))

        # Line width (Equation 6a)
        delta_f = (self._h2o_b3 * 1e-4 *
                   (p_dry * (theta ** self._h2o_b4) +
                    self._h2o_b5 * e_s * (theta ** self._h2o_b6)))
        delta_f = np.sqrt(delta_f ** 2 + 2.25e-6)

        # Line shape factor
        f0 = self._h2o_f0
        term1 = delta_f / ((f0 - f) ** 2 + delta_f ** 2)
        term2 = delta_f / ((f0 + f) ** 2 + delta_f ** 2)
        Fi = (f / f0) * (term1 + term2)

        N_H2O = np.sum(Si * Fi)
        return 0.1820 * f * N_H2O

    def equivalent_height_oxygen(
        self,
        freq_ghz: float,
        temperature_k: float = 288.15,
        pressure_hpa: float = 1013.25,
        water_vapor_density: float = 7.5
    ) -> float:
        """
        Calculate equivalent height for oxygen (h0) in km.

        Uses interpolated coefficients from ITU-R P.676 data file for
        frequencies in V-band (48-55 GHz).

        Args:
            freq_ghz: Frequency in GHz
            temperature_k: Surface temperature in Kelvin
            pressure_hpa: Surface pressure in hPa
            water_vapor_density: Water vapor density in g/m³

        Returns:
            float: Equivalent height in km
        """
        min_freq = self._eq_height_freqs[0]
        max_freq = self._eq_height_freqs[-1]

        if min_freq <= freq_ghz <= max_freq:
            # Fast interpolation using numpy
            coeffs = np.zeros(4)
            for i in range(4):
                coeffs[i] = np.interp(
                    freq_ghz, self._eq_height_freqs, self._eq_height_coeffs[:, i]
                )
            h0 = (coeffs[0] + coeffs[1] * temperature_k +
                  coeffs[2] * pressure_hpa + coeffs[3] * water_vapor_density)
        elif freq_ghz < min_freq:
            h0 = 6.0  # Default for lower frequencies
        else:
            coeffs = self._eq_height_coeffs[-1]
            h0 = (coeffs[0] + coeffs[1] * temperature_k +
                  coeffs[2] * pressure_hpa + coeffs[3] * water_vapor_density)

        return max(0.1, h0)

    def equivalent_height_water_vapor(
        self,
        freq_ghz: float,
        pressure_hpa: float = 1013.25,
        temperature_k: float = 288.15,
        water_vapor_density: float = 7.5
    ) -> float:
        """
        Calculate equivalent height for water vapor (h_w) in km.

        Implements ITU-R P.676-13 Annex 2 formula with frequency-dependent
        corrections for the 22.235 GHz water vapor line.

        Args:
            freq_ghz: Frequency in GHz
            pressure_hpa: Surface pressure in hPa
            temperature_k: Surface temperature in Kelvin
            water_vapor_density: Water vapor density in g/m³

        Returns:
            float: Equivalent height in km (typically 1.5-2.5 km)
        """
        f = freq_ghz
        e_s = (water_vapor_density * temperature_k) / 216.7
        theta = 300.0 / temperature_k
        p_dry = pressure_hpa - e_s

        # Base equivalent height
        h_w0 = 1.66  # km

        # Line width parameter for 22.235 GHz line
        sigma_w = 26.38e-4 * (p_dry * (theta ** 0.76) + 5.087 * e_s * theta)

        # Frequency-dependent correction
        f_22 = 22.235
        if abs(f - f_22) < 0.1:
            line_contribution = 1.0 / (sigma_w + 0.001)
        else:
            line_contribution = sigma_w / ((f - f_22) ** 2 + sigma_w ** 2)

        symmetric_contribution = sigma_w / ((f + f_22) ** 2 + sigma_w ** 2)
        h_w = h_w0 * (1.0 + 1.39 * (line_contribution + symmetric_contribution))

        # Apply atmospheric corrections
        h_w *= (1.0 + 0.03 * (1.0 - pressure_hpa / 1013.25))
        h_w *= (1.0 + 0.02 * (temperature_k / 288.15 - 1.0))
        h_w *= (1.0 - 0.01 * (water_vapor_density / 7.5 - 1.0))

        return max(1.0, min(h_w, 5.0))

    def total_slant_attenuation(
        self,
        freq_ghz: float,
        elevation_deg: float,
        pressure_hpa: float = 1013.25,
        temperature_k: float = 288.15,
        water_vapor_density: float = 7.5,
        include_water_vapor: bool = True
    ) -> float:
        """
        Calculate total slant path attenuation in dB.

        This is the main function for Phase 3 weather satellite modeling.

        Args:
            freq_ghz: Frequency in GHz
            elevation_deg: Elevation angle in degrees (0-90, where 90=zenith)
            pressure_hpa: Surface pressure in hPa (default: 1013.25)
            temperature_k: Surface temperature in Kelvin (default: 288.15)
            water_vapor_density: Water vapor density in g/m³ (default: 7.5)
            include_water_vapor: Whether to include water vapor attenuation

        Returns:
            float: Total slant path attenuation in dB
        """
        # Oxygen attenuation
        gamma_o = self.oxygen_attenuation(
            freq_ghz, pressure_hpa, temperature_k, water_vapor_density
        )
        h0 = self.equivalent_height_oxygen(
            freq_ghz, temperature_k, pressure_hpa, water_vapor_density
        )
        A_o_zenith = gamma_o * h0

        # Water vapor attenuation
        if include_water_vapor:
            gamma_w = self.water_vapor_attenuation(
                freq_ghz, pressure_hpa, temperature_k, water_vapor_density
            )
            h_w = self.equivalent_height_water_vapor(
                freq_ghz, pressure_hpa, temperature_k, water_vapor_density
            )
            A_w_zenith = gamma_w * h_w
        else:
            A_w_zenith = 0.0

        # Air mass factor
        elevation_rad = np.deg2rad(elevation_deg)
        if elevation_deg > 5:
            air_mass = 1.0 / np.sin(elevation_rad)
        else:
            air_mass = 1.0 / (np.sin(elevation_rad) +
                              0.00175 * np.tan(np.deg2rad(90 - elevation_deg)))
            air_mass = min(air_mass, 40.0)

        return (A_o_zenith + A_w_zenith) * air_mass

    def total_slant_attenuation_detailed(
        self,
        freq_ghz: float,
        elevation_deg: float,
        pressure_hpa: float = 1013.25,
        temperature_k: float = 288.15,
        water_vapor_density: float = 7.5,
        include_water_vapor: bool = True
    ) -> Dict[str, float]:
        """
        Calculate total slant path attenuation with detailed breakdown.

        Returns all intermediate values for debugging/analysis.

        Args:
            freq_ghz: Frequency in GHz
            elevation_deg: Elevation angle in degrees
            pressure_hpa: Surface pressure in hPa
            temperature_k: Surface temperature in Kelvin
            water_vapor_density: Water vapor density in g/m³
            include_water_vapor: Whether to include water vapor

        Returns:
            Dict with keys:
                - 'gamma_o': Oxygen specific attenuation (dB/km)
                - 'gamma_w': Water vapor specific attenuation (dB/km)
                - 'h0': Oxygen equivalent height (km)
                - 'h_w': Water vapor equivalent height (km)
                - 'A_o_zenith': Oxygen zenith attenuation (dB)
                - 'A_w_zenith': Water vapor zenith attenuation (dB)
                - 'A_o_slant': Oxygen slant path attenuation (dB)
                - 'A_w_slant': Water vapor slant path attenuation (dB)
                - 'A_total_slant': Total slant path attenuation (dB)
                - 'air_mass_factor': Air mass factor (1/sin(elevation))
        """
        gamma_o = self.oxygen_attenuation(
            freq_ghz, pressure_hpa, temperature_k, water_vapor_density
        )
        h0 = self.equivalent_height_oxygen(
            freq_ghz, temperature_k, pressure_hpa, water_vapor_density
        )
        A_o_zenith = gamma_o * h0

        if include_water_vapor:
            gamma_w = self.water_vapor_attenuation(
                freq_ghz, pressure_hpa, temperature_k, water_vapor_density
            )
            h_w = self.equivalent_height_water_vapor(
                freq_ghz, pressure_hpa, temperature_k, water_vapor_density
            )
            A_w_zenith = gamma_w * h_w
        else:
            gamma_w = 0.0
            h_w = 0.0
            A_w_zenith = 0.0

        elevation_rad = np.deg2rad(elevation_deg)
        if elevation_deg > 5:
            air_mass = 1.0 / np.sin(elevation_rad)
        else:
            air_mass = 1.0 / (np.sin(elevation_rad) +
                              0.00175 * np.tan(np.deg2rad(90 - elevation_deg)))
            air_mass = min(air_mass, 40.0)

        A_o_slant = A_o_zenith * air_mass
        A_w_slant = A_w_zenith * air_mass

        return {
            'gamma_o': gamma_o,
            'gamma_w': gamma_w,
            'h0': h0,
            'h_w': h_w,
            'A_o_zenith': A_o_zenith,
            'A_w_zenith': A_w_zenith,
            'A_o_slant': A_o_slant,
            'A_w_slant': A_w_slant,
            'A_total_slant': A_o_slant + A_w_slant,
            'air_mass_factor': air_mass
        }

    def batch_total_attenuation(
        self,
        freq_ghz_array: np.ndarray,
        elevation_deg: float,
        pressure_hpa: float = 1013.25,
        temperature_k: float = 288.15,
        water_vapor_density: float = 7.5,
        include_water_vapor: bool = True
    ) -> np.ndarray:
        """
        Calculate total slant attenuation for multiple frequencies.

        Optimized for batch processing in Phase 3 modeling.

        Args:
            freq_ghz_array: Array of frequencies in GHz
            elevation_deg: Elevation angle in degrees
            pressure_hpa: Surface pressure in hPa
            temperature_k: Surface temperature in Kelvin
            water_vapor_density: Water vapor density in g/m³
            include_water_vapor: Whether to include water vapor

        Returns:
            np.ndarray: Array of total slant attenuation values in dB
        """
        return np.array([
            self.total_slant_attenuation(
                f, elevation_deg, pressure_hpa, temperature_k,
                water_vapor_density, include_water_vapor
            )
            for f in freq_ghz_array
        ])


# =============================================================================
# Cached Calculator Instance
# =============================================================================

_cached_calculator = None


def get_cached_calculator(data_dir: str = None) -> ITUP676Calculator:
    """
    Get a cached calculator instance to avoid repeated data loading.

    The calculator is created once on first call and reused for subsequent calls.
    This is useful for Phase 3 modeling where many calculations are performed.

    Args:
        data_dir: Optional directory containing CSV data files.
                  Only used on first call when creating the calculator.

    Returns:
        ITUP676Calculator: Cached calculator instance

    Example:
        calc = get_cached_calculator()
        atten1 = calc.total_slant_attenuation(50.3, 30.0)
        atten2 = calc.total_slant_attenuation(52.5, 45.0)
    """
    global _cached_calculator
    if _cached_calculator is None:
        _cached_calculator = ITUP676Calculator(data_dir)
    return _cached_calculator


def reset_cached_calculator():
    """
    Reset the cached calculator instance.

    Call this if you need to reload data files or use a different data directory.
    """
    global _cached_calculator
    _cached_calculator = None


# =============================================================================
# ITU-R P.840-9 (cloud) and P.838-3 (rain) slant attenuation
# =============================================================================

# Rain path length Uniform[low, high) km — typical draw per rainy FOV.
RAIN_PATH_LENGTH_UNIFORM_LOW_KM = 2.0
RAIN_PATH_LENGTH_UNIFORM_HIGH_KM = 8.0

# Cloud liquid water temperature (K), P.840-9 default.
DEFAULT_CLOUD_WATER_TEMP_K = 273.75
# Circular polarization tilt for P.838-3 (degrees).
DEFAULT_RAIN_TAU_DEG = 45.0

# ITU-R P.838-3 frequency table (GHz) and coefficients
P838_FREQ_TABLE_GHZ = np.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0], dtype=np.float64)
P838_K_H_TABLE = np.array([0.6600, 0.6811, 0.7020, 0.7228, 0.7433, 0.7635], dtype=np.float64)
P838_GAMMA_H_TABLE = np.array(
    [0.8084, 0.8034, 0.7987, 0.7941, 0.7896, 0.7853], dtype=np.float64
)
P838_K_V_TABLE = np.array([0.6472, 0.6687, 0.6901, 0.7112, 0.7321, 0.7527], dtype=np.float64)
P838_GAMMA_V_TABLE = np.array(
    [0.7871, 0.7826, 0.7783, 0.7741, 0.7700, 0.7661], dtype=np.float64
)


def _broadcast_arrays_cloud_rain(*args: np.ndarray) -> tuple[np.ndarray, ...]:
    return np.broadcast_arrays(*args)


def cloud_attenuation_db(
    elevation_angle_deg: np.ndarray | float,
    f_ghz: np.ndarray | float,
    iclw_kg_m2: np.ndarray | float,
    T_k: np.ndarray | float = 273.75,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Total cloud attenuation (dB), ITU-R P.840-9 (no rain path model).

    Returns ``(total_cloud_atten_db, K_L)`` with broadcast shapes.
    """
    theta = np.asarray(elevation_angle_deg, dtype=np.float64)
    f_GHz = np.asarray(f_ghz, dtype=np.float64)
    iclw = np.asarray(iclw_kg_m2, dtype=np.float64)
    T = np.asarray(T_k, dtype=np.float64)

    theta, f_GHz, iclw, T = _broadcast_arrays_cloud_rain(theta, f_GHz, iclw, T)

    theta_T = 300.0 / T
    dt = theta_T - 1.0
    eps0 = 77.66 + 103.3 * dt
    eps1 = 0.0671 * eps0
    eps2 = 3.52

    fp = 20.20 - 146.0 * dt + 316.0 * dt**2
    fs = 39.8 * fp

    eps_pp = (f_GHz * (eps0 - eps1) / fp) / (1.0 + (f_GHz / fp) ** 2) + (
        f_GHz * (eps1 - eps2) / fs
    ) / (1.0 + (f_GHz / fs) ** 2)

    eps_p = (eps0 - eps1) / (1.0 + (f_GHz / fp) ** 2) + (eps1 - eps2) / (
        1.0 + (f_GHz / fs) ** 2
    ) + eps2

    eta = (2.0 + eps_p) / eps_pp
    Kl = 0.819 * f_GHz / (eps_pp * (1.0 + eta**2))

    A1 = 0.1522
    A2 = 11.51
    A3 = -10.4912
    f1 = -23.9589
    f2 = 219.2096
    sig1 = 3.2991e3
    sig2 = 2.7595e6

    K_L = Kl * (
        A1 * np.exp((f_GHz - f1) ** 2 / sig1)
        + A2 * np.exp((f_GHz - f2) ** 2 / sig2)
        + A3
    )

    sin_el = np.sin(np.deg2rad(theta))
    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.where(
            np.abs(sin_el) > 1e-15,
            K_L * iclw / sin_el,
            np.inf,
        )
    return total, K_L


def nearest_p838_freq_table_indices(f_ghz: np.ndarray) -> np.ndarray:
    """Index in ``P838_FREQ_TABLE_GHZ`` with smallest |f - table| per element."""
    f = np.asarray(f_ghz, dtype=np.float64)
    dist = np.abs(f[..., np.newaxis] - P838_FREQ_TABLE_GHZ[np.newaxis, :])
    return np.argmin(dist, axis=-1)


def rain_attenuation_db(
    elevation_angle_deg: np.ndarray | float,
    f_ghz: np.ndarray | float,
    rain_rate_mm_hr: np.ndarray | float,
    rain_path_length_km: np.ndarray | float,
    tau_deg: np.ndarray | float = 45.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Total rain attenuation (dB), ITU-R P.838-3 style; coefficients by nearest table frequency.

    Returns ``(tot_rain_atten_db, spec_rain_atten_db_per_km, closest_freq_ghz)``.
    """
    theta = np.asarray(elevation_angle_deg, dtype=np.float64)
    f = np.asarray(f_ghz, dtype=np.float64)
    R = np.asarray(rain_rate_mm_hr, dtype=np.float64)
    L = np.asarray(rain_path_length_km, dtype=np.float64)
    tau = np.asarray(tau_deg, dtype=np.float64)

    theta, f, R, L, tau = _broadcast_arrays_cloud_rain(theta, f, R, L, tau)

    fi = nearest_p838_freq_table_indices(f)
    k_h = P838_K_H_TABLE[fi]
    gamma_h = P838_GAMMA_H_TABLE[fi]
    k_v = P838_K_V_TABLE[fi]
    gamma_v = P838_GAMMA_V_TABLE[fi]
    closest = P838_FREQ_TABLE_GHZ[fi]

    theta_r = np.deg2rad(theta)
    tau_r = np.deg2rad(tau)
    cth2 = np.cos(theta_r) ** 2
    c2t = np.cos(2.0 * tau_r)

    k = (k_h + k_v + (k_h - k_v) * cth2 * c2t) / 2.0
    gamma = (k_h * gamma_h + k_v * gamma_v + (k_h * gamma_h - k_v * gamma_v) * cth2 * c2t) / (
        2.0 * k
    )

    spec = k * np.power(R, gamma)

    sin_el = np.sin(theta_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        tot = np.where(np.abs(sin_el) > 1e-15, (spec * L) / sin_el, np.inf)

    return tot, spec, closest


def compute_cloud_rain_attenuation_db(
    elevation_deg: np.ndarray,
    iclw: np.ndarray,
    rain: np.ndarray,
    rain_rate_mm_hr: np.ndarray,
    freqs_ghz: np.ndarray,
    rng: np.random.Generator,
    *,
    iclw_abs_threshold: float,
    T_k: float = DEFAULT_CLOUD_WATER_TEMP_K,
    tau_deg: float = DEFAULT_RAIN_TAU_DEG,
    path_low_km: float = RAIN_PATH_LENGTH_UNIFORM_LOW_KM,
    path_high_km: float = RAIN_PATH_LENGTH_UNIFORM_HIGH_KM,
) -> np.ndarray:
    """
    Per-FOV, per-channel attenuation (dB), gated on ``|iclw|`` and ``rain``.

    Parameters
    ----------
    iclw_abs_threshold
        |ICLW| threshold (kg/m²); must be passed explicitly by the application
        (e.g. simulation script), not a library default.

    Rules
    -----
    - ``|iclw| <= iclw_abs_threshold`` or non-finite ``iclw`` → 0 dB.
    - ``|iclw| >`` threshold and ``rain == 0`` → P.840-9 cloud only.
    - ``|iclw| >`` threshold and ``rain == 1`` → P.838-3 rain only.

    Cloud and rain formulas run only on the matching FOV subsets (vectorized per subset).

    Returns array shaped ``(n_fov, n_chan)``.
    """
    elevation_deg = np.asarray(elevation_deg, dtype=np.float64)
    iclw = np.asarray(iclw, dtype=np.float64)
    rain = np.asarray(rain, dtype=np.int8)
    rain_rate_mm_hr = np.maximum(np.asarray(rain_rate_mm_hr, dtype=np.float64), 0.0)
    freqs_ghz = np.asarray(freqs_ghz, dtype=np.float64)

    n = elevation_deg.size
    n_ch = freqs_ghz.size
    f_r = freqs_ghz[np.newaxis, :]

    finite = np.isfinite(iclw)
    high = finite & (np.abs(iclw) > iclw_abs_threshold)
    dry_high = high & (rain == 0)
    rain_high = high & (rain == 1)

    out = np.zeros((n, n_ch), dtype=np.float64)

    if np.any(dry_high):
        tot_c, _ = cloud_attenuation_db(
            elevation_deg[dry_high][:, np.newaxis],
            f_r,
            iclw[dry_high][:, np.newaxis],
            T_k=T_k,
        )
        out[dry_high, :] = np.asarray(tot_c, dtype=np.float64)

    if np.any(rain_high):
        n_rh = int(np.sum(rain_high))
        L = rng.uniform(path_low_km, path_high_km, size=n_rh)
        tot_r_sub, _, _ = rain_attenuation_db(
            elevation_deg[rain_high][:, np.newaxis],
            f_r,
            rain_rate_mm_hr[rain_high][:, np.newaxis],
            L[:, np.newaxis],
            tau_deg=tau_deg,
        )
        out[rain_high, :] = np.asarray(tot_r_sub, dtype=np.float64)

    return out


# =============================================================================
# ITU ICLW / rain grids — per-FOV cloud + rain slant attenuation (RFI pipelines)
# =============================================================================


def _nearest_indices_1d(query: np.ndarray, grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    idx_hi = np.searchsorted(grid, q, side="left")
    idx_lo = idx_hi - 1
    idx_lo = np.clip(idx_lo, 0, grid.size - 1)
    idx_hi = np.clip(idx_hi, 0, grid.size - 1)
    dist_lo = np.abs(q - grid[idx_lo])
    dist_hi = np.abs(grid[idx_hi] - q)
    return np.where(dist_lo <= dist_hi, idx_lo, idx_hi)


def _align_lon_to_grid(lon_fov: np.ndarray, lon_grid_1d: np.ndarray) -> np.ndarray:
    x = np.asarray(lon_fov, dtype=np.float64)
    lo = float(np.min(lon_grid_1d))
    hi = float(np.max(lon_grid_1d))
    if lo >= 0.0 and hi > 180.0:
        return np.mod(x, 360.0)
    if lo < 0.0 or hi <= 180.0:
        return (np.mod(x + 180.0, 360.0)) - 180.0
    return x


def load_iclw_grid(nc_path: str):
    if Dataset is None:
        raise ImportError("netCDF4 is required for load_iclw_grid")
    ds = Dataset(nc_path, "r")
    try:

        def _first_var(*candidates: str):
            for n in candidates:
                if n in ds.variables:
                    return ds.variables[n]
            raise KeyError(f"None of {candidates!r} found in {nc_path}")

        mean_v = _first_var("iclw_mean_val", "mean_val")
        std_v = _first_var("iclw_stddev_val", "stddev_val")
        lat_var = _first_var("lat", "latitude", "Latitude")
        lon_var = _first_var("lon", "longitude", "Longitude")

        for _v in (mean_v, std_v, lat_var, lon_var):
            _v.set_auto_maskandscale(False)

        mean_arr = np.asarray(mean_v[:], dtype=np.float64)
        std_arr = np.asarray(std_v[:], dtype=np.float64)

        if lat_var.ndim == 1 and lon_var.ndim == 1:
            lat_1d = np.asarray(lat_var[:], dtype=np.float64)
            lon_1d = np.asarray(lon_var[:], dtype=np.float64)
            if lat_1d.size < 2 or lon_1d.size < 2:
                raise ValueError("lat/lon must have at least 2 points each")

            lat_inc = lat_1d[0] < lat_1d[-1]
            lon_inc = lon_1d[0] < lon_1d[-1]
            if not lat_inc:
                lat_1d = lat_1d[::-1]
                mean_arr = np.flip(mean_arr, axis=0)
                std_arr = np.flip(std_arr, axis=0)
            if not lon_inc:
                lon_1d = lon_1d[::-1]
                mean_arr = np.flip(mean_arr, axis=1)
                std_arr = np.flip(std_arr, axis=1)

            sh_m = mean_arr.shape
            if sh_m == (lat_1d.size, lon_1d.size):
                kind = "rectilinear"
            elif sh_m == (lon_1d.size, lat_1d.size):
                mean_arr = mean_arr.T
                std_arr = std_arr.T
                kind = "rectilinear"
            else:
                raise ValueError(
                    f"ICLW mean shape {sh_m} incompatible with lat {lat_1d.size}, lon {lon_1d.size}"
                )

            return {
                "kind": kind,
                "lat_1d": lat_1d,
                "lon_1d": lon_1d,
                "mean": mean_arr,
                "std": std_arr,
            }

        lat_2d = np.asarray(lat_var[:], dtype=np.float64)
        lon_2d = np.asarray(lon_var[:], dtype=np.float64)
        if lat_2d.shape != lon_2d.shape or mean_arr.shape != lat_2d.shape:
            raise ValueError(
                f"Geo2D mismatch: lat {lat_2d.shape}, lon {lon_2d.shape}, mean {mean_arr.shape}"
            )
        if cKDTree is None:
            raise SystemExit("2D lat/lon grid requires scipy (cKDTree). pip install scipy")
        pts = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])
        tree = cKDTree(pts)
        return {
            "kind": "kd",
            "tree": tree,
            "mean_flat": mean_arr.ravel(),
            "std_flat": std_arr.ravel(),
            "lon_lo": float(np.min(lon_2d)),
            "lon_hi": float(np.max(lon_2d)),
        }
    finally:
        ds.close()


def map_fovs_to_mean_std(
    lat_fov: np.ndarray,
    lon_fov: np.ndarray,
    grid: dict,
) -> tuple[np.ndarray, np.ndarray]:
    lat_fov = np.asarray(lat_fov, dtype=np.float64)
    lon_fov = np.asarray(lon_fov, dtype=np.float64)

    if grid["kind"] == "rectilinear":
        lon_q = _align_lon_to_grid(lon_fov, grid["lon_1d"])
        ilat = _nearest_indices_1d(lat_fov, grid["lat_1d"])
        ilon = _nearest_indices_1d(lon_q, grid["lon_1d"])
        mean_s = grid["mean"][ilat, ilon]
        std_s = grid["std"][ilat, ilon]
        return mean_s, std_s

    lon_q = _align_lon_to_grid(
        lon_fov, np.array([grid["lon_lo"], grid["lon_hi"]], dtype=np.float64)
    )
    q = np.column_stack([lat_fov, lon_q])
    try:
        _, idx = grid["tree"].query(q, workers=-1)
    except TypeError:
        _, idx = grid["tree"].query(q)
    mean_s = grid["mean_flat"][idx]
    std_s = grid["std_flat"][idx]
    return mean_s, std_s


def _geo2d_nearest_indices(lat_fov: np.ndarray, lon_fov: np.ndarray, grid: dict):
    lat_fov = np.asarray(lat_fov, dtype=np.float64)
    lon_fov = np.asarray(lon_fov, dtype=np.float64)

    if grid["kind"] == "rectilinear":
        lon_q = _align_lon_to_grid(lon_fov, grid["lon_1d"])
        ilat = _nearest_indices_1d(lat_fov, grid["lat_1d"])
        ilon = _nearest_indices_1d(lon_q, grid["lon_1d"])
        return ("rectilinear", ilat, ilon)

    lon_q = _align_lon_to_grid(
        lon_fov, np.array([grid["lon_lo"], grid["lon_hi"]], dtype=np.float64)
    )
    q = np.column_stack([lat_fov, lon_q])
    try:
        _, idx = grid["tree"].query(q, workers=-1)
    except TypeError:
        _, idx = grid["tree"].query(q)
    return ("kd", idx, None)


def map_fovs_to_rain_prob_and_rate(
    lat_fov: np.ndarray, lon_fov: np.ndarray, grid: dict
) -> tuple[np.ndarray, np.ndarray]:
    kind, a, b = _geo2d_nearest_indices(lat_fov, lon_fov, grid)
    if kind == "rectilinear":
        return (
            np.asarray(grid["rain_prob"][a, b], dtype=np.float64),
            np.asarray(grid["rain_rate"][a, b], dtype=np.float64),
        )
    return (
        np.asarray(grid["rain_prob_flat"][a], dtype=np.float64),
        np.asarray(grid["rain_rate_flat"][a], dtype=np.float64),
    )


def _coord_var(ds, *names: str):
    for n in names:
        if n in ds.variables:
            return ds.variables[n]
    raise KeyError(f"None of {names!r} found in dataset")


def load_itu_rain_grid(nc_path: str) -> dict:
    if Dataset is None:
        raise ImportError("netCDF4 is required for load_itu_rain_grid")
    ds = Dataset(nc_path, "r")
    try:
        rain_v = _coord_var(ds, "rain_prob")
        try:
            rate_v = _coord_var(ds, "rain_rate")
        except KeyError as e:
            raise KeyError(
                f"Variable 'rain_rate' not found in {nc_path}; "
                f"available: {list(ds.variables.keys())}"
            ) from e
        lat_var = _coord_var(ds, "lat", "latitude", "Latitude")
        lon_var = _coord_var(ds, "lon", "longitude", "Longitude")

        for _v in (rain_v, rate_v, lat_var, lon_var):
            _v.set_auto_maskandscale(False)

        rain_arr = np.asarray(rain_v[:], dtype=np.float64)
        rate_arr = np.asarray(rate_v[:], dtype=np.float64)
        if rain_arr.shape != rate_arr.shape:
            raise ValueError(
                f"rain_prob shape {rain_arr.shape} != rain_rate shape {rate_arr.shape}"
            )

        if lat_var.ndim == 1 and lon_var.ndim == 1:
            lat_1d = np.asarray(lat_var[:], dtype=np.float64)
            lon_1d = np.asarray(lon_var[:], dtype=np.float64)
            if lat_1d.size < 2 or lon_1d.size < 2:
                raise ValueError("lat/lon must have at least 2 points each")

            lat_inc = lat_1d[0] < lat_1d[-1]
            lon_inc = lon_1d[0] < lon_1d[-1]
            if not lat_inc:
                lat_1d = lat_1d[::-1]
                rain_arr = np.flip(rain_arr, axis=0)
                rate_arr = np.flip(rate_arr, axis=0)
            if not lon_inc:
                lon_1d = lon_1d[::-1]
                rain_arr = np.flip(rain_arr, axis=1)
                rate_arr = np.flip(rate_arr, axis=1)

            sh = rain_arr.shape
            if sh == (lat_1d.size, lon_1d.size):
                pass
            elif sh == (lon_1d.size, lat_1d.size):
                rain_arr = rain_arr.T
                rate_arr = rate_arr.T
            else:
                raise ValueError(
                    f"rain fields shape {sh} incompatible with "
                    f"lat {lat_1d.size}, lon {lon_1d.size}"
                )

            return {
                "kind": "rectilinear",
                "lat_1d": lat_1d,
                "lon_1d": lon_1d,
                "rain_prob": rain_arr,
                "rain_rate": rate_arr,
            }

        lat_2d = np.asarray(lat_var[:], dtype=np.float64)
        lon_2d = np.asarray(lon_var[:], dtype=np.float64)
        if lat_2d.shape != lon_2d.shape or rain_arr.shape != lat_2d.shape:
            raise ValueError(
                f"Geo2D mismatch: lat {lat_2d.shape}, lon {lon_2d.shape}, rain fields {rain_arr.shape}"
            )
        if cKDTree is None:
            raise SystemExit("2D lat/lon grid requires scipy (cKDTree). pip install scipy")
        pts = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])
        tree = cKDTree(pts)
        return {
            "kind": "kd",
            "tree": tree,
            "rain_prob_flat": rain_arr.ravel(),
            "rain_rate_flat": rate_arr.ravel(),
            "lon_lo": float(np.min(lon_2d)),
            "lon_hi": float(np.max(lon_2d)),
        }
    finally:
        ds.close()


def itu_iclw_rain_info_nc_path(combined_csv_basename: str, data_dir: str) -> str:
    stem = os.path.splitext(combined_csv_basename)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot parse month from CSV stem {stem!r}; expected e.g. atms_2023080112_..."
        )
    dt = parts[1]
    if not re.match(r"^\d{8,}$", dt):
        raise ValueError(
            f"Second underscore segment {dt!r} is not yyyymmddhh...; stem={stem!r}"
        )
    mm = dt[4:6]
    return os.path.join(data_dir, f"itu_iclw_rain_info_{mm}.nc")


def compute_cloud_rain_atten_db_for_fovs(
    lat: np.ndarray,
    lon: np.ndarray,
    elevation_deg: np.ndarray,
    itu_nc_path: str,
    center_freqs_ghz: np.ndarray,
    rng: np.random.Generator,
    iclw_abs_threshold: float,
) -> np.ndarray:
    """
    Vectorized cloud/rain slant attenuation (dB) per FOV per channel.

    ``center_freqs_ghz`` column ``j`` corresponds to the same channel order used when
    building the dict for ``weather_sat_nwp.copy_nc4_with_tmbr_plus_rfi``.

    If ``itu_nc_path`` is missing, returns zeros (no attenuation) after a warning.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    elevation_deg = np.asarray(elevation_deg, dtype=np.float64)
    freqs = np.asarray(center_freqs_ghz, dtype=np.float64)
    n = lat.size
    n_ch = freqs.size
    if not os.path.isfile(itu_nc_path):
        print(
            f"WARNING: ITU cloud/rain NetCDF not found ({itu_nc_path}); "
            "using 0 dB attenuation for all FOVs/channels."
        )
        return np.zeros((n, n_ch), dtype=np.float64)

    iclg = load_iclw_grid(itu_nc_path)
    mean_s, std_s = map_fovs_to_mean_std(lat, lon, iclg)
    noise = rng.standard_normal(size=mean_s.shape)
    iclw = mean_s + std_s * noise
    bad = ~np.isfinite(mean_s) | ~np.isfinite(std_s)
    if np.any(bad):
        iclw = np.where(bad, np.nan, iclw)
    iclw = np.clip(iclw, 0.0, None)

    rain_grid = load_itu_rain_grid(itu_nc_path)
    rain_prob_pct, rain_rate_mm = map_fovs_to_rain_prob_and_rate(lat, lon, rain_grid)
    u = rng.uniform(0.0, 100.0, size=rain_prob_pct.shape)
    rain = (u < rain_prob_pct).astype(np.int8)
    invalid_rain = ~np.isfinite(rain_prob_pct)
    if np.any(invalid_rain):
        rain = np.where(invalid_rain, 0, rain)

    return compute_cloud_rain_attenuation_db(
        elevation_deg,
        iclw,
        rain,
        rain_rate_mm,
        freqs,
        rng,
        iclw_abs_threshold=iclw_abs_threshold,
    )


def atten_db_to_by_channel_dict(
    channel_numbers: Sequence[int],
    atten_db: np.ndarray,
) -> dict[int, np.ndarray]:
    """Map instrument channel number -> (n_obs,) attenuation (dB); column order matches ``channel_numbers``."""
    ch_list = [int(c) for c in channel_numbers]
    if atten_db.shape[1] != len(ch_list):
        raise ValueError(
            f"atten_db has {atten_db.shape[1]} columns but {len(ch_list)} channel_numbers"
        )
    return {ch_list[j]: np.asarray(atten_db[:, j], dtype=np.float64) for j in range(len(ch_list))}


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing attenuation_mdl.py...")
    print()

    # Create calculator
    calc = ITUP676Calculator()
    print("Calculator initialized successfully.")
    print(f"Data directory: {calc.data_dir}")
    print()

    # atmospheric conditions
    pressure_hpa = 1013.25
    temperature_k = 293.15  # 20°C
    water_vapor_density = 7.5

    print("Atmospheric conditions:")
    print(f"  Pressure: {pressure_hpa} hPa")
    print(f"  Temperature: {temperature_k} K ({temperature_k - 273.15:.1f}°C)")
    print(f"  Water vapor density: {water_vapor_density} g/m³")
    print()

    # Test calculations
    test_cases = [
        (23.8, 90.0, "K-Band, Zenith"),
        (23.8, 30.0, "K-Band, 30°"),
        (50.3, 90.0, "V-Band (50.3 GHz), Zenith"),
        (50.3, 30.0, "V-Band (50.3 GHz), 30°"),
        (52.5, 90.0, "V-Band (52.5 GHz), Zenith"),
        (52.5, 30.0, "V-Band (52.5 GHz), 30°"),
    ]

    print(f"{'Description':<30} {'Freq (GHz)':<12} {'Elev (°)':<10} {'Atten (dB)':<12}")
    print("-" * 70)

    for freq, elev, desc in test_cases:
        atten = calc.total_slant_attenuation(
            freq, elev, pressure_hpa, temperature_k, water_vapor_density
        )
        print(f"{desc:<30} {freq:<12.1f} {elev:<10.0f} {atten:<12.2f}")

    print()
    print("Test complete!")
