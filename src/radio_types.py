"""
Type definitions for radio modeling
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import pyarrow as pa
from radio_io import power_pattern_from_cut_file

# Import functions from other modules to avoid circular imports
import antenna_pattern
import astro_mdl


@dataclass
class Antenna:
    """
    Antenna class for radio modeling.
    alphas are angles from z-axis towards x-y-plane, betas are angles from x-axis
    towards y-axis, all in degrees.
    The gain interpolator coords are in radians.
    """
    gain_pat: pd.DataFrame  # gain pattern
    gain_func: RegularGridInterpolator  # gain interpolator
    rad_eff: float  # radiation efficiency
    valid_freqs: Tuple[float, float]  # min and max valid frequencies for the gain model

    def __post_init__(self):
        """Validate the antenna parameters"""
        required_columns = ['alphas', 'betas', 'gains']
        if not all(col in self.gain_pat.columns for col in required_columns):
            raise ValueError(f"gain_pat must contain columns: {required_columns}")
        if self.valid_freqs[0] >= self.valid_freqs[1]:
            raise ValueError("valid_freqs[0] must be less than valid_freqs[1]")

    @classmethod
    def from_dataframe(cls, gain_ant: pd.DataFrame, rad_eff: float,
                       valid_freqs: Tuple[float, float] = (0.0, 0.0)) -> 'Antenna':
        """Create an Antenna instance from a DataFrame"""

        # Create the gain interpolator
        alphas = gain_ant[gain_ant['betas'] == gain_ant['betas'].iloc[0]]['alphas']
        betas = gain_ant[gain_ant['alphas'] == gain_ant['alphas'].iloc[0]]['betas']
        gain_func = antenna_pattern.interpolate_gain(gain_ant['gains'].values, alphas.values, betas.values)
        return cls(gain_ant, gain_func, rad_eff, valid_freqs)

    @classmethod
    def from_file(cls, file_pattern_path: str, rad_eff: float,
                  valid_freqs: Tuple[float, float] = (0.0, 0.0),
                  power_tag: str = 'gains',
                  declination_tag: str = 'alphas',
                  azimuth_tag: str = 'betas') -> 'Antenna':
        """Create an Antenna instance from a file"""
        if not file_pattern_path.endswith('.cut'):
            raise ValueError("The power pattern file must be a .cut file")

        # Load the antenna power pattern
        gain_ant = power_pattern_from_cut_file(file_pattern_path)

        # Rename angles columns
        gain_ant = gain_ant.rename(columns={
            declination_tag: 'alphas',
            azimuth_tag: 'betas'
        })

        # Convert into gain
        alphas = gain_ant[gain_ant['betas'] == gain_ant['betas'].iloc[0]]['alphas']
        betas = gain_ant[gain_ant['alphas'] == gain_ant['alphas'].iloc[0]]['betas']

        gain_ant[power_tag] = antenna_pattern.radiated_power_to_gain(
            gain_ant[power_tag].values, alphas.values, betas.values, eta_rad=rad_eff
        )
        gain_ant = gain_ant.rename(columns={power_tag: 'gains'})

        return cls.from_dataframe(gain_ant, rad_eff, valid_freqs)

    def get_gain_pattern(self) -> pd.DataFrame:
        return self.gain_pat

    def get_gain_value(self, alpha: float, beta: float) -> float:
        return self.gain_func((alpha, beta))

    def get_gain_values(self, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """
        Vectorized gain lookup for arrays of spherical coordinates (in radians).
        Accepts broadcastable shapes; returns a 1-D array after raveling inputs.
        """
        alphas = np.asarray(alphas)
        betas = np.asarray(betas)
        # Broadcast to a common shape, then stack as (N, 2)
        common_shape = np.broadcast_shapes(alphas.shape, betas.shape)
        a = np.broadcast_to(alphas, common_shape).ravel()
        b = np.broadcast_to(betas, common_shape).ravel()
        pts = np.stack((a, b), axis=1)
        vals = self.gain_func(pts)
        return np.asarray(vals).reshape(common_shape)

    def get_def_angles(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.unique(self.gain_pat['alphas']),
                np.unique(self.gain_pat['betas']))

    def get_boresight_gain(self) -> float:
        return self.gain_pat['gains'].max()

    def get_boresight_point(self) -> Tuple[float, float]:
        gain = self.gain_pat
        i = gain['gains'].idxmax()
        return gain.loc[i, 'alphas'], gain.loc[i, 'betas']

    def get_slice_gain(self, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        gain_pat = self.gain_pat
        g_pos = gain_pat[gain_pat['betas'] == beta]
        g_neg = gain_pat[gain_pat['betas'] == beta + 180]

        alphas = np.concatenate([-g_neg['alphas'].values[:-1][::-1],
                                g_pos['alphas'].values[1:]])
        gains = np.concatenate([g_neg['gains'].values[:-1][::-1],
                               g_pos['gains'].values[1:]])

        return alphas, gains

    def get_rad_eff(self) -> float:
        return self.rad_eff

    def get_valid_freqs(self) -> Tuple[float, float]:
        return self.valid_freqs


@dataclass
class Instrument:
    """
    Instrument class for radio modeling.
    Assumes the frame of antenna is oriented North-West-Up.
    Assumes the output of signal_func is expressed in Kelvin (temperature).
    signal_funcs should depend on the time and the frequency.
    freq_chan is number of frequency channels for spectrograph model instead of full
    bw integration of power.
    """
    antenna: Antenna  # antenna
    phy_temp: float  # antenna physical temperature
    cent_freq: float  # center frequency
    bw: float  # bandwidth
    signal_func: callable  # signal from instrument (noise, transmission, etc.)
    freq_chan: int  # number of frequency channels
    coords: List[float]  # coordinates

    def __post_init__(self):
        """Validate the instrument parameters"""
        ant_fmin, ant_fmax = self.antenna.get_valid_freqs()
        if not (ant_fmin <= self.cent_freq - self.bw/2 and
                self.cent_freq + self.bw/2 <= ant_fmax):
            raise ValueError("Antenna model is not suited for center frequency of instrument")

    @classmethod
    def from_scalar(cls, ant: Antenna, phy_temp: float, center_freq: float,
                    bandwidth: float, signal: float, freq_chan: int = 1,
                    coords: List[float] = None) -> 'Instrument':
        """Create an Instrument instance with a scalar signal"""
        if coords is None:
            coords = []

        def _signal_func(*_args):
            return signal
        return cls(ant, phy_temp, center_freq, bandwidth, _signal_func, freq_chan, coords)

    def get_coords(self) -> List[float]:
        return self.coords

    def get_antenna(self) -> Antenna:
        return self.antenna

    def get_phy_temp(self) -> float:
        return self.phy_temp

    def get_center_freq(self) -> float:
        return self.cent_freq

    def get_bandwidth(self) -> float:
        return self.bw

    def get_nb_freq_chan(self) -> int:
        return self.freq_chan

    def get_center_freq_chans(self) -> np.ndarray:
        freq_chan = self.get_nb_freq_chan()
        bw_RX = self.get_bandwidth()
        delta_freq = bw_RX/freq_chan
        rng_freq = np.linspace(-bw_RX/2 + delta_freq/2,
                               bw_RX/2 - delta_freq/2,
                               freq_chan)
        return self.get_center_freq() + rng_freq

    def get_inst_signal(self) -> callable:
        return self.signal_func


@dataclass
class Transmitter:
    """
    Enhanced transmitter class with polarization and harmonics modeling.

    This class extends the basic Instrument functionality with additional
    transmitter-specific characteristics that are important for accurate
    interference modeling in radio astronomy.
    """
    instrument: Instrument  # Base instrument (antenna, frequency, power, etc.)
    polarization: str  # Polarization type: 'linear', 'circular', 'elliptical'
    polarization_angle: float  # Polarization angle in degrees (0-180)
    harmonics: List[Tuple[float, float]]  # List of (frequency_multiplier, power_reduction_factor)

    def __post_init__(self):
        """Validate the transmitter parameters"""
        if self.polarization not in ['linear', 'circular', 'elliptical']:
            raise ValueError("polarization must be 'linear', 'circular', or 'elliptical'")

        if not (0 <= self.polarization_angle <= 180):
            raise ValueError("polarization_angle must be between 0 and 180 degrees")

        # Validate harmonics format
        for i, (freq_mult, power_red) in enumerate(self.harmonics):
            if freq_mult <= 0:
                raise ValueError(f"harmonic {i}: frequency_multiplier must be positive")
            if power_red <= 0 or power_red > 1:
                raise ValueError(f"harmonic {i}: power_reduction_factor must be between 0 and 1")

    @classmethod
    def from_instrument(cls, instrument: Instrument, polarization: str = 'linear',
                        polarization_angle: float = 0.0,
                        harmonics: List[Tuple[float, float]] = None) -> 'Transmitter':
        """Create a Transmitter from an existing Instrument"""
        if harmonics is None:
            harmonics = []
        return cls(instrument, polarization, polarization_angle, harmonics)

    def get_instrument(self) -> Instrument:
        """Get the base instrument"""
        return self.instrument

    def get_polarization(self) -> str:
        """Get the polarization type"""
        return self.polarization

    def get_polarization_angle(self) -> float:
        """Get the polarization angle in degrees"""
        return self.polarization_angle

    def get_harmonics(self) -> List[Tuple[float, float]]:
        """Get the harmonics list"""
        return self.harmonics

    def add_harmonic(self, frequency_multiplier: float, power_reduction_factor: float):
        """Add a harmonic to the transmitter"""
        if frequency_multiplier <= 0:
            raise ValueError("frequency_multiplier must be positive")
        if power_reduction_factor <= 0 or power_reduction_factor > 1:
            raise ValueError("power_reduction_factor must be between 0 and 1")

        self.harmonics.append((frequency_multiplier, power_reduction_factor))

    def get_harmonic_frequencies(self) -> List[float]:
        """Get list of harmonic frequencies"""
        base_freq = self.instrument.get_center_freq()
        return [base_freq * freq_mult for freq_mult, _ in self.harmonics]

    def get_harmonic_powers(self) -> List[float]:
        """Get list of harmonic power reduction factors"""
        return [power_red for _, power_red in self.harmonics]


@dataclass
class Trajectory:
    """
    Trajectory class for radio modeling.
    traj is a DataFrame where each row is indexed by a datetime and has an elevation
    and azimuth angle(s). It is possible to have vectors of els and azs for a same
    time (e.g. for sky mapping). All measurements are assumed to be given in SI
    units (in degrees for angles, meters for distances).
    """
    traj: pd.DataFrame  # azimuth, elevation and distance info for each sampled time

    def __post_init__(self):
        """Validate the trajectory parameters"""
        required_columns = ['times', 'azimuths', 'elevations', 'distances']
        if not all(col in self.traj.columns for col in required_columns):
            raise ValueError(f"traj must contain columns: {required_columns}")
        if len(self.traj['times'].unique()) != len(self.traj):
            raise ValueError("Times must be unique")
        if not isinstance(self.traj['times'].iloc[0], datetime):
            raise ValueError("Times must be datetime objects")

        # Sort by time
        self.traj = self.traj.sort_values('times')

    @classmethod
    def from_file(cls, file_path: str, time_tag: str = 'times',
                  elevation_tag: str = 'altitudes', azimuth_tag: str = 'azimuths',
                  distance_tag: str = 'distances') -> 'Trajectory':
        """Create a Trajectory instance from a file"""
        if file_path.endswith('.arrow'):
            # read arrow file
            with pa.memory_map(file_path, 'r') as source:
                table = pa.ipc.open_file(source).read_all()
            traj = table.to_pandas()
        elif file_path.endswith('.csv'):
            traj = pd.read_csv(file_path)
        else:
            raise ValueError("The trajectory points must be in Arrow or CSV format")

        # Rename columns
        traj = traj.rename(columns={
            time_tag: 'times',
            azimuth_tag: 'azimuths',
            elevation_tag: 'elevations',
            distance_tag: 'distances'
        })

        # Convert time stamps to datetime
        traj['times'] = pd.to_datetime(traj['times'])

        return cls(traj[['times', 'azimuths', 'elevations', 'distances']])

    def get_traj(self) -> pd.DataFrame:
        return self.traj

    def get_traj_between(self, t0: datetime, t1: datetime,
                         skipmissing: bool = True) -> pd.DataFrame:
        mask = (self.traj['times'] >= t0) & (self.traj['times'] <= t1)
        return self.traj[mask]

    def get_time_bounds(self) -> Tuple[datetime, datetime]:
        return self.traj['times'].iloc[0], self.traj['times'].iloc[-1]

    def get_time_stamps(self) -> pd.Series:
        return self.traj['times']

    def get_azimuths(self) -> pd.Series:
        return self.traj['azimuths']

    def get_elevations(self) -> pd.Series:
        return self.traj['elevations']

    def get_distances(self) -> pd.Series:
        return self.traj['distances']


@dataclass
class Observation:
    """
    Observation class for radio modeling.
    pts can contain different positions for a same time, e.g. for a sky map.
    """
    pts: Trajectory  # trajectory of the observation
    inst: Instrument  # instrument used for observation
    result: np.ndarray  # store the results of the modeling of the observation

    def __post_init__(self):
        """Validate the observation parameters"""
        if len(self.pts.get_time_stamps()) != self.result.shape[0]:
            raise ValueError("Time stamps length must match result first dimension")

        az = self.pts.get_azimuths().iloc[0]
        try:
            len_pos = len(az)
        except TypeError:
            len_pos = 1
        if len_pos != self.result.shape[1]:
            raise ValueError("Azimuths length must match result second dimension")
        if self.inst.get_nb_freq_chan() != self.result.shape[2]:
            raise ValueError("Frequency channels must match result third dimension")

    @classmethod
    def from_dates(cls, start_date: datetime, stop_date: datetime,
                   trajectory: Trajectory, instrument: Instrument,
                   filt_funcs: Tuple = ()) -> 'Observation':
        """Create an Observation instance from dates"""
        # Filter date and other from trajectory
        traj = trajectory.get_traj_between(start_date, stop_date)
        for filt in filt_funcs:
            traj = traj[filt[1](traj[filt[0]])]

        if traj.empty:
            raise ValueError("No pointing positions found for the given time window and custom filters")

        traj = traj.sort_values('times')
        pts = Trajectory(traj)

        # Create result storage
        len_time = len(pts.get_time_stamps())

        az = pts.get_azimuths().iloc[0]
        try:
            len_pos = len(az)
        except TypeError:
            len_pos = 1
        len_freq = instrument.get_nb_freq_chan()
        result = np.full((len_time, len_pos, len_freq), np.nan)

        return cls(pts, instrument, result)

    def get_traj(self) -> pd.DataFrame:
        return self.pts.get_traj()

    def get_time_bounds(self) -> Tuple[datetime, datetime]:
        return self.pts.get_time_bounds()

    def get_time_stamps(self) -> pd.Series:
        return self.pts.get_time_stamps()

    def get_azimuths(self) -> pd.Series:
        return self.pts.get_azimuths()

    def get_elevations(self) -> pd.Series:
        return self.pts.get_elevations()

    def get_distances(self) -> pd.Series:
        return self.pts.get_distances()

    def get_instrument(self) -> Instrument:
        return self.inst

    def get_result(self) -> np.ndarray:
        return self.result


# there exist two estim_temp functions in radio_types.py (here) and astro_mdl.py
def estim_temp(flux: float, obs: Observation) -> float:
    """Estimate temperature from flux and observation"""
    instru = obs.get_instrument()
    frequency = instru.get_center_freq()
    ant = instru.get_antenna()

    max_gain = ant.get_boresight_gain()
    A_eff_max = antenna_pattern.gain_to_effective_aperture(max_gain, frequency)

    return astro_mdl.estim_temp(flux, A_eff_max)


@dataclass
class Constellation:
    """
    Constellation class for radio modeling.
    Assumes the positions of sats are time-synced with the time samples of observation.
    Suppose the frame of satellite antenna is oriented North-East-Nadir. The antenna
    pointing can be any direction from Nadir, defined in the map gain.
    """
    sats: pd.DataFrame
    tmt: Instrument
    lnk_bdgt_mdl: callable

    def __post_init__(self):
        """Validate the constellation parameters"""
        # Check lnk_bdgt_mdl signature is correct
        # TODO: Implement signature checking

    @classmethod
    def from_observation(cls, sats: pd.DataFrame, observation: Observation,
                         sat_tmt: Instrument, lnk_bdgt_mdl: callable = None,
                         filt_funcs: Tuple = ()) -> 'Constellation':
        """Create a Constellation instance from an observation"""
        if lnk_bdgt_mdl is None:
            # Import here to avoid circular import
            import sat_mdl
            lnk_bdgt_mdl = sat_mdl.sat_link_budget

        # Observation window
        start_date, stop_date = observation.get_time_bounds()

        # Apply the custom filters
        sats = sats[(sats['times'] >= start_date) & (sats['times'] <= stop_date)]

        # debug
        for filt in filt_funcs:
            sats = sats[filt[1](sats[filt[0]])]

        # Sort by 'sat' and 'times' for consistency with Julia
        sats = sats.sort_values(['sat', 'times']).reset_index(drop=True)

        return cls(sats, sat_tmt, lnk_bdgt_mdl)

    @classmethod
    def from_file(cls, file_path: str, observation: Observation,
                  sat_tmt: Instrument, lnk_bdgt_mdl: callable = None,
                  name_tag: str = 'sat', time_tag: str = 'time_stamps',
                  elevation_tag: str = 'altitudes', azimuth_tag: str = 'azimuths',
                  distance_tag: str = 'distances', filt_funcs: Tuple = ()) -> 'Constellation':
        """Create a Constellation instance from a file"""
        with pa.memory_map(file_path, 'r') as source:
            table = pa.ipc.open_file(source).read_all()
        sats = table.to_pandas()

        # Rename columns
        sats = sats.rename(columns={
            time_tag: 'times',
            name_tag: 'sat',
            azimuth_tag: 'azimuths',
            elevation_tag: 'elevations',
            distance_tag: 'distances'
        })

        sats['times'] = pd.to_datetime(sats['times'])
        sats = sats.sort_values('times')

        return cls.from_observation(sats, observation, sat_tmt, lnk_bdgt_mdl, filt_funcs)

    def get_antenna(self) -> Antenna:
        return self.tmt.get_antenna()

    def get_transmitter(self) -> Instrument:
        return self.tmt

    def get_sats_name(self) -> List[str]:
        return self.sats['sat'].unique().tolist()

    def get_lnk_bdgt_mdl(self) -> callable:
        return self.lnk_bdgt_mdl

    def get_sat_traj(self, s: str) -> pd.DataFrame:
        return self.sats[self.sats['sat'] == s]

    def get_sats_names_at_time(self, t: datetime) -> List[str]:
        return self.sats[self.sats['times'] == t]['sat'].tolist()


# Helper functions
def power_to_temperature(power: float, bandwidth: float) -> float:
    """Convert power to temperature"""
    k_boltz = 1.380649e-23  # Boltzmann's constant
    return power / (k_boltz * bandwidth)


def temperature_to_power(temperature: float, bandwidth: float) -> float:
    """Convert temperature to power"""
    k_boltz = 1.380649e-23  # Boltzmann's constant
    return k_boltz * temperature * bandwidth
