# Modeling the result of a radio astronomy observation
#
# This notebook uses the Python modules to model the resulting power
# levels seen by a radio telescope - the Westford antenna - when observing an
# astronomical object such as Cas A.
#
# Comprehensive environmental effects modeling for radio astronomy observations including terrain
# masking with DEM data, atmospheric refraction correction, water vapor effects, and limb refraction
# for space-to-space interactions, extending `tuto_radiomdl_transmitter.py`

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from radio_types import Antenna, Instrument, Observation, Constellation, Trajectory, Transmitter  # noqa: E402
from astro_mdl import (  # noqa: E402
    estim_casA_flux, power_to_temperature, temperature_to_power,
    antenna_mdl_ITU, estim_temp
)
from sat_mdl import (  # noqa: E402
    sat_link_budget_vectorized,
    # get_doppler_impact_on_observation,
    lnk_bdgt_with_doppler_correction,
    calculate_radial_velocities_vectorized,
    analyze_doppler_statistics,
    print_doppler_statistical_summary,
    calculate_polarization_mismatch_loss,
    calculate_harmonic_contribution,
    sat_link_budget_comprehensive_vectorized,
    link_budget_doppler_transmitter,
    calculate_comprehensive_environmental_effects_vectorized
)
from obs_mdl import model_observed_temp, model_observed_temp_with_atmospheric_refraction_vectorized  # noqa: E402
from env_mdl import AdvancedEnvironmentalEffects  # noqa: E402
import antenna_pattern  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings('ignore')


# ## Define the instrument used to observe
# ---

# ### The antenna
#
# The first step to define the instrument of observation is to define the antenna
# used. The package has a custom structure `Antenna` that takes as inputs a
# `DataFrame` structure with columns `alphas`, `betas` and `gains`, containing
# the gain values and their angle coordinates (α,β), the estimated
# radiation efficiency and the frequency band where the model of antenna defined
# is valid. α∈[0,180] and β∈[0,360[ are defined such that when
# the antenna is pointing at the horizon, β=0 gives a vertical slice of the
# pattern, with α>0 oriented towards the ground.
#
# A path to a `.cut` file defining the power pattern model can be given in place
# of the `DataFrame` structure to load and format the gain pattern. In that case,
# it is possible to give the tags of the different columns if other names are
# present in the table to load.

# radiation efficiency of telescope antenna
eta_rad = 0.45

# valid frequency band of gain pattern model
freq_band = (10e9, 12e9)  # in Hz

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# load telescope antenna
file_pattern_path = os.path.join(script_dir, "data", "single_cut_res.cut")
tel_ant = Antenna.from_file(
    file_pattern_path,
    eta_rad,
    freq_band,
    power_tag='power',
    declination_tag='alpha',
    azimuth_tag='beta'
)

# The result is an `Antenna` storing the different information and an interpolated
# version of the gain pattern.

# plot gain pattern
nb_curv = 5  # number of slices to plot
alphas, betas = tel_ant.get_def_angles()
step_beta_ind = len(betas) // (2 * nb_curv)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
for i in range(0, len(betas) // 2, step_beta_ind):
    a, g = tel_ant.get_slice_gain(betas[i])
    ax.plot(np.radians(a), 10 * np.log10(g), label=f"β = {betas[i]}deg")
ax.legend()
# plt.show()

# ### The instrument
#
# To define the `Instrument` of observation as a structure, we need an `Antenna`
# structure, as well as other parameters:
# - the physical temperature of the antenna;
# - the frequency of observation;
# - the bandwidth of the instrument;
# - the receiver temperature as a function of time (`datetime` type) and frequency;
# It is also possible to specify:
# - the number of frequency channels to divide the bandwidth of the instrument by
#   (default is 1);
# - the coordinates of the instrument as a vector (not used as of now);

# telescope antenna physical temperature
T_phy = 300.0  # in K

# frequency of observation
cent_freq = 11.325e9  # in Hz

# bandwidth of telescope receiver
bw = 1e3  # in Hz

# number of frequency channels to divide the bandwidth
freq_chan = 1


# telescope receiver temperature (constant over the bandwidth)
def T_RX(tim, freq):
    return 80.0  # in K


# coordinates of telescope
coords = [42.6129479883915, -71.49379366344017, 86.7689687917009]

# environmental: Initialize environmental effects
westford_lat = coords[0]  # degrees
westford_lon = coords[1]  # degrees
westford_elevation = coords[2]  # meters above sea level

# Atmospheric conditions (typical for Westford, MA)
temperature = 288.15  # K (15°C)
pressure = 101325  # Pa (1 atm)
humidity = 60.0  # % (moderate humidity)

# Environmental effects configuration
environmental_config = {
    'dem_file': os.path.join(script_dir, "data", "USGS_OPR_MA_CentralEastern_2021_B21_be_19TBH294720.tif"),
    'antenna_lat': westford_lat,
    'antenna_lon': westford_lon,
    'antenna_elevation': westford_elevation,
    'temperature': temperature,
    'pressure': pressure,
    'humidity': humidity,
    'min_elevation_angle': 5.0,  # degrees
    'apply_terrain_masking': True,
    'apply_atmospheric_effects': True,
    'apply_limb_refraction': True
}

# Initialize environmental effects
print("Initializing environmental effects...")
environment = AdvancedEnvironmentalEffects(
    environmental_config['dem_file'], westford_lat, westford_lon, westford_elevation,
    temperature, pressure, humidity
)

# create instrument
westford = Instrument(tel_ant, T_phy, cent_freq, bw, T_RX, freq_chan, coords)

# ## Define the observation plan
# ---
#
# The next step is to define the observation plan and conditions.

# ### The pointing trajectory during the observation
#
# It is possible to load an already computed table of pointing directions. For
# instance, an `.arrow` file containing the position of Cas A over a window of
# time. The package defines a `Trajectory` structure that formats a `DataFrame` to
# be used thereafter.
#
# `Trajectory` can receive the path to an `.arrow` or `.csv` file instead of the
# `DataFrame` to load an existing file. In that case, it is possible to give the
# tags of the different columns if other names are present in the table to load.

# time window of generated source trajectory
start_window = "2025-02-18T15:00:00.000"
stop_window = "2025-02-18T15:45:00.000"

# replace colon with underscore
start_window_str = start_window.replace(":", "_")
stop_window_str = stop_window.replace(":", "_")

# load telescope antenna
file_traj_obj_path = os.path.join(
    script_dir, "data",
    f"casA_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
)

print(file_traj_obj_path)

# source position over time window
traj_src = Trajectory.from_file(
    file_traj_obj_path,
    time_tag='time_stamps',
    elevation_tag='altitudes',
    azimuth_tag='azimuths',
    distance_tag='distances'
)

# To be more realistic, say the observation lasted 10min, with 5min offset and
# 5min on source, excluding any points that could be below 5deg elevation:

# start-end of observation
dateformat = "%Y-%m-%dT%H:%M:%S.%f"
start_obs = datetime.strptime("2025-02-18T15:30:00.000", dateformat)
stop_obs = datetime.strptime("2025-02-18T15:40:00.000", dateformat)

# offset from source at the beginning of the observation
offset_angles = (-40, 0.)  # (az,el) in degrees

# time of OFF-ON transition
time_off_src = start_obs
time_on_src = time_off_src + timedelta(minutes=5)

# copy trajectory
traj_obj = Trajectory(traj_src.traj.copy())

# apply offset
mask = (traj_obj.traj['times'] >= time_off_src) & (traj_obj.traj['times'] <= time_on_src)
traj_obj.traj.loc[mask, 'azimuths'] += offset_angles[0]
traj_obj.traj.loc[mask, 'elevations'] += offset_angles[1]

# filter points below 5deg elevation
filt_el = ('elevations', lambda e: e > 5.)

# create observation
observ = Observation.from_dates(start_obs, stop_obs, traj_obj, westford, filt_funcs=(filt_el,))

# visualize source and pointing trajectory
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, polar=True)
src_traj = traj_src.get_traj()
obs_traj = observ.get_traj()
ax.plot(np.radians(src_traj['azimuths']), 90 - src_traj['elevations'], label="source")
ax.plot(np.radians(obs_traj['azimuths']), 90 - obs_traj['elevations'], label="pointing")
ax.set_yticks(range(0, 91, 10))
ax.set_yticklabels([str(x) for x in range(90, -1, -10)])
ax.legend()
ax.set_theta_zero_location("N")
# plt.show()

# ### Sky model
#
# Accounting for the sky components is important to have a realistic simulation of
# the power received.
#
# Given the fact that we will point at Cas A for a part of the observation, we
# need to account for the difference of temperature depending on the pointing position:

# source flux
flux_src = estim_casA_flux(cent_freq)  # in Jy


# Pre-calculate effective aperture for performance optimization
max_gain = tel_ant.get_boresight_gain()
A_eff_max = antenna_pattern.gain_to_effective_aperture(max_gain, cent_freq)


# source temperature in K
def T_src(t):
    if t <= time_on_src:
        return 0.0
    else:
        # use A_eff_max instead of observ
        return estim_temp(flux_src, A_eff_max)


# Same for the RFI and the background sources that can be modeled as constants as a first approximation:

# ground temperature in K
T_gnd = 0  # no constant RFI

# various RFI
T_var = 0  # in K (no RFI)

# total RFI temperature
T_rfi = T_gnd + T_var

# CMB temperature
T_CMB = 2.73  # in K


# galaxy temperature
def T_gal(freq): return 1e-1 * (freq/1.41e9)**(-2.7)  # in K


# background
def T_bkg(freq): return T_CMB + T_gal(freq)


# The atmosphere is also important to account for:
# atmospheric temperature at zenith
T_atm_zenith = 150  # in K

# opacity of atmosphere at zenith
tau = 0.05


# atmospheric temperature model
def T_atm(dec): return T_atm_zenith * (1 - np.exp(-tau/np.cos(dec)))  # in K


# Adding up all of these sources gives:
# Total sky model in K
def sky_mdl(dec, caz, tim, freq):
    return T_src(tim) + T_atm(dec) + T_rfi + T_bkg(freq)


# plot of sky model without source
azimuth_grid = np.arange(0, 361, 5)
elevation_grid = np.arange(0, 91, 1)
az_grid, el_grid = np.meshgrid(azimuth_grid, elevation_grid)
sky_temp = np.vectorize(
    lambda el, az: sky_mdl(np.radians(90-el), -np.radians(az), start_obs, cent_freq)
)(el_grid, az_grid)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, polar=True)
pc = ax.pcolormesh(np.radians(azimuth_grid), 90-elevation_grid, sky_temp, cmap="plasma")
cbar = plt.colorbar(pc)
cbar.set_label("Temperature [K]")
ax.set_yticks(range(0, 91, 10))
ax.set_yticklabels([str(x) for x in range(90, -1, -10)])
ax.set_theta_zero_location("N")
# plt.show()

# ## Satellite constellations
# ---
# The package also defines a structure `Constellation` to account for aggregated
# power from satellites.

# ### Antenna model
#
# The first step is to define the antenna model for the type of satellites in the
# constellation, as for instance the ITU recommended gain model.
#
# Note that in the following, the satellites coordinate frame is supposed to be
# following (North, East, Nadir).

# radiation efficiency of telescope antenna
sat_eta_rad = 0.5  # FIXME: check value

# maximum gain of satellite antenna
sat_gain_max = 39.3  # in dBi FIXME: check value in dBi

# create ITU recommended gain profile
# satellite boresight half beamwidth
half_beamwidth = 3.0  # in deg FIXME: check value
# declination angles alpha
alphas = np.arange(0, 181)
# azimuth angles beta
betas = np.arange(0, 351, 10)
# create gain dataframe
gain_pat = antenna_mdl_ITU(sat_gain_max, half_beamwidth, alphas, betas)

# create satellite antenna
sat_ant = Antenna.from_dataframe(gain_pat, sat_eta_rad, freq_band)

# plot gain pattern
nb_curv = 5  # number of slices to plot
alphas, betas = sat_ant.get_def_angles()
step_beta_ind = len(betas) // (2 * nb_curv)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
for i in range(0, len(betas) // 2, step_beta_ind):
    a, g = sat_ant.get_slice_gain(betas[i])
    ax.plot(np.radians(a), 10 * np.log10(g), label=f"β = {betas[i]}deg")
ax.legend()
# plt.show()

# ### Transmitter parameters
#
# The satellites are transmitting information and can be defined as an
# `Instrument` structure:

# telescope antenna physical temperature
sat_T_phy = 0.0  # in K

# frequency of transmission
sat_freq = 11.325e9  # in Hz

# satellite transmission bandwidth
sat_bw = 250e6  # in Hz

# satellite effective isotropically radiated power
transmit_pow = -15 + 10 * np.log10(300)  # in dBW FIXME: check value


def transmit_temp(tim, freq):
    return power_to_temperature(10**(transmit_pow/10), 1.0)  # in K


# create transmitter instrument
sat_transmit = Instrument(sat_ant, sat_T_phy, sat_freq, sat_bw, transmit_temp, 1, [])

# ### Constellation
#
# Taking a pre-generated file containing the trajectory of each satellite at each
# time samples of the observation. The `Constellation` structure takes as input
# the path to this file, the `Observation` structure and the `Instrument` with which
# the satellites are transmitting.
#
# The structure can take the tags of the columns of the table and additional
# filters:

# filter the satellites
filt_name = ('sat', lambda s: ~s.str.contains('DTC'))
filt_el = ('elevations', lambda e: e > 20)


def lnk_bdgt(*args, **kwargs):
    # "Catch" the unexpected argument and remove it from the dictionary.
    # The second argument to pop (e.g., None) is a default value if 'beam_avoid' isn't present.
    beam_avoid_angle = kwargs.pop('beam_avoid', None)

    # Now you can use the value of beam_avoid_angle for any custom logic
    # within this wrapper function if you need to.
    if beam_avoid_angle is not None:
        # For example: print(f"Beam avoidance angle is {beam_avoid_angle} degrees.")
        # Or you could modify one of the *args based on this value before passing them on.
        pass

    # Call the base function with only the arguments it actually expects.
    # `kwargs` no longer contains the 'beam_avoid' key.
    return sat_link_budget_vectorized(*args, **kwargs)


# satellites trajectories during the observation
file_traj_sats_path = os.path.join(
    script_dir, "data",
    f"Starlink_trajectory_Westford_{start_window_str}_{stop_window_str}.arrow"
)

# Load satellite data for analysis
print("Loading satellite data for analysis...")

# Load all satellite data for analysis
try:
    import pyarrow as pa
    with pa.memory_map(file_traj_sats_path, 'r') as source:
        table = pa.ipc.open_file(source).read_all()
    all_sat_data = table.to_pandas()
except ImportError:
    print("ERROR: pyarrow not available. Please install pyarrow to read .arrow files.")
    sys.exit(1)

# Apply column renaming and datetime conversion
all_sat_data = all_sat_data.rename(columns={
    'timestamp': 'times',
    'sat': 'sat',
    'azimuths': 'azimuths',
    'elevations': 'elevations',
    'ranges_westford': 'distances'
})
all_sat_data['times'] = pd.to_datetime(all_sat_data['times'])

# Define observation parameters
observation_band_center = cent_freq  # 11.325 GHz
observation_band_width = bw  # 1 KHz

# Run statistical analysis FIRST
print("\n" + "="*60)
print("STATISTICAL ANALYSIS FOR DOPPLER EFFECT")
print("="*60)
doppler_statistics = analyze_doppler_statistics(
    all_sat_data, observation_band_center, observation_band_width,
    start_obs, stop_obs, cent_freq
)
print_doppler_statistical_summary(doppler_statistics, observation_band_width)

# Decision logic based on risk assessment
contamination_probability = doppler_statistics['contamination_probability']
print(f"\n🔍 RISK-BASED DECISION: Contamination probability = {contamination_probability:.1%}")

# Initialize variables
doppler_filtered_satellites = []
filt_doppler = None
doppler_filter_time = 0

if contamination_probability > 0.4:  # Medium or High risk
    print("⚠️ MEDIUM/HIGH RISK DETECTED - Applying Doppler Correction in frequency-domain for Physics-Based Prediction")

    # Physics-Based Doppler Correction in frequency-domain
    # Calculate radial velocities for all satellites in the observation window
    print("Calculating radial velocities for Doppler correction...")

    # Filter data to observation window and elevation criteria
    obs_window_data = all_sat_data[
        (all_sat_data['times'] >= start_obs) &
        (all_sat_data['times'] <= stop_obs) &
        (all_sat_data['elevations'] > 20)
    ].copy()

    # Calculate radial velocities for all satellites
    radial_velocities = calculate_radial_velocities_vectorized(obs_window_data)

    # Add radial velocities to the data for later use
    obs_window_data['radial_velocities'] = radial_velocities

    # Calculate Doppler statistics for reporting
    max_radial_velocity = np.max(np.abs(radial_velocities)) if len(radial_velocities) > 0 else 0
    max_doppler_shift = (max_radial_velocity / 299792458) * observation_band_center

    print(f"   • Maximum radial velocity: {max_radial_velocity:.1f} m/s")
    print(f"   • Maximum Doppler shift: {max_doppler_shift/1e3:.1f} kHz")
    print(f"   • Satellites with Doppler correction: {len(obs_window_data['sat'].unique())}")

    # Store the corrected data for use in constellation loading
    doppler_corrected_data = obs_window_data
    use_doppler_correction = True

else:
    print("🟢 LOW RISK DETECTED - Using Standard Prediction (No Doppler Correction)")
    print("Using all satellites with standard link budget calculation")
    doppler_corrected_data = None
    use_doppler_correction = False

# Transmitter Characteristics Analysis
# =============================================================================
print("\n" + "="*60)
print("TRANSMITTER CHARACTERISTICS ANALYSIS")
print("="*60)


# Set up satellite transmitters with different characteristics
def setup_enhanced_transmitters(sat_transmit):
    """Set up satellite transmitters with characteristics."""

    print("Setting up satellite transmitters...")

    # Create transmitters with different characteristics
    transmitters = {}

    # 1. Standard transmitter (no enhancements) - baseline for comparison
    transmitters['standard'] = Transmitter.from_instrument(
        sat_transmit,
        polarization='linear',
        polarization_angle=0.0,
        harmonics=[]
    )

    # 2. Linear polarized transmitter with harmonics
    transmitters['linear_harmonics'] = Transmitter.from_instrument(
        sat_transmit,
        polarization='linear',
        polarization_angle=45.0,
        harmonics=[(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]
    )

    # 3. Circular polarized transmitter
    transmitters['circular'] = Transmitter.from_instrument(
        sat_transmit,
        polarization='circular',
        polarization_angle=0.0,
        harmonics=[(2.0, 0.05), (3.0, 0.02)]
    )

    # 4. High harmonic transmitter (realistic for some satellite systems)
    transmitters['high_harmonics'] = Transmitter.from_instrument(
        sat_transmit,
        polarization='linear',
        polarization_angle=45.0,  # Changed from 90.0 to 45.0 to avoid complete signal loss
        harmonics=[(2.0, 0.2), (3.0, 0.1), (4.0, 0.05), (5.0, 0.02), (6.0, 0.01)]
    )

    # 5. REALISTIC SCENARIO: Starlink (circular) + Westford (linear) = 3 dB loss
    transmitters['realistic_starlink'] = Transmitter.from_instrument(
        sat_transmit,
        polarization='circular',  # Starlink uses circular polarization
        polarization_angle=0.0,   # Circular polarization angle is not relevant
        harmonics=[(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]  # Moderate harmonics for realistic satellite
    )

    print("transmitters created:")
    for name, tx in transmitters.items():
        print(f"  - {name}: {tx.get_polarization()} polarization, {len(tx.get_harmonics())} harmonics")

    return transmitters


# Create transmitters
enhanced_transmitters = setup_enhanced_transmitters(sat_transmit)

# Capture the selected transmitter early to avoid closure issues
# REALISTIC SCENARIO: Use Starlink (circular) + Westford (linear) configuration
# This represents the actual polarization mismatch between:
# - Starlink satellites: circular polarization (transmitter)
# - Westford radio telescope: linear polarization (receiver)
# Result: 3 dB polarization loss (50% power reduction)
selected_transmitter = enhanced_transmitters['realistic_starlink']

# Analyze transmitter characteristics impact
print("\nAnalyzing transmitter characteristics impact...")

# Test polarization mismatch loss for different combinations
test_polarizations = [
    ('linear', 0.0, 'linear', 0.0),
    ('linear', 0.0, 'linear', 90.0),
    ('linear', 45.0, 'linear', 45.0),
    ('linear', 0.0, 'circular', 0.0),
    ('circular', 0.0, 'linear', 0.0),
    ('circular', 0.0, 'circular', 0.0)
]

print("\nPolarization Mismatch Loss Analysis:")
print("-" * 50)
print(f"{'TX Pol':<12} {'TX Ang':<8} {'RX Pol':<12} {'RX Ang':<8} {'Loss':<8}")
print("-" * 50)

for tx_pol, tx_ang, rx_pol, rx_ang in test_polarizations:
    loss = calculate_polarization_mismatch_loss(tx_pol, tx_ang, rx_pol, rx_ang)
    loss_db = 10 * np.log10(loss) if loss > 0 else -100
    print(f"{tx_pol:<12} {tx_ang:<8.1f} {rx_pol:<12} {rx_ang:<8.1f} {loss_db:<8.1f}")

# Test harmonic contributions
print("\nHarmonic Contribution Analysis:")
print("-" * 40)

base_freq = cent_freq  # 11.325 GHz
base_power = 1.0
obs_freq = cent_freq
obs_bw = bw  # 1 kHz

test_harmonics = [(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]

total_contribution = calculate_harmonic_contribution(
    base_freq, base_power, test_harmonics, obs_freq, obs_bw
)

if total_contribution > 0:
    print(f"Total harmonic contribution: {total_contribution:.3f} ({10*np.log10(total_contribution):.1f} dB)")
else:
    print(f"Total harmonic contribution: {total_contribution:.3f} (-inf dB)")

print(f"Fundamental + harmonics: {1.0 + total_contribution:.3f} ({10*np.log10(1.0 + total_contribution):.1f} dB)")

# Decision logic for transmitter characteristics
print("\n🔍 TRANSMITTER CHARACTERISTICS DECISION:")

# REALISTIC SCENARIO ANALYSIS: Starlink (circular) + Westford (linear)
realistic_polarization_loss = abs(10*np.log10(calculate_polarization_mismatch_loss('circular', 0.0, 'linear', 0.0)))
print("   • REALISTIC SCENARIO: Starlink (circular) + Westford (linear)")
print(f"   • Polarization effects: {'Significant' if realistic_polarization_loss > 1 else 'Minimal'}")
print(f"      - realistic polarization loss = {realistic_polarization_loss:.2f} dB")
print(f"   • Harmonic effects: {'Significant' if total_contribution > 0.01 else 'Minimal'}")
print(f"      - total harmonics contribution = {total_contribution:.6f}")

# General analysis for comparison
polarization_loss = abs(10*np.log10(calculate_polarization_mismatch_loss('linear', 0.0, 'circular', 0.0)))
print(f"\n   • General analysis (linear-to-circular): {polarization_loss:.2f} dB")
print(f"   • General analysis (circular-to-linear): {realistic_polarization_loss:.2f} dB")

# Use realistic scenario for decision making
use_enhanced_transmitters = total_contribution > 0.01 or realistic_polarization_loss > 1

# Display conclusion based on actual decision logic
print(f"   • Using transmitter characteristics modeling: {'Yes' if use_enhanced_transmitters else 'No (baseline)'}")
print("      - polarization loss > 1 OR total harmonics contribution > 0.01\n")

# New constellation loading with conditional Doppler correction AND Transmitter characteristics
if use_doppler_correction:
    # Apply Doppler correction for physics-based prediction
    print("Loading constellation with Doppler correction...")

    # Create a custom link budget function that uses the combined function
    # KEY IMPROVEMENT: Now using link_budget_doppler_transmitter() from sat_mdl.py
    # This function properly combines BOTH Doppler correction AND transmitter characteristics
    def lnk_bdgt_with_doppler_and_enhanced(*args, **kwargs):
        """
        VECTORIZED version of the custom link budget function with environmental effects.
        This function integrates Doppler effects, transmitter characteristics, and environmental effects.
        """
        # Set beam avoidance parameters to accept custom link budget model_observed_temp
        kwargs['beam_avoid'] = 1e-20
        kwargs['turn_off'] = False

        # VECTORIZED ENVIRONMENTAL EFFECTS CALCULATION
        if environmental_config is not None:
            # Convert to degrees for environmental calculations
            dec_sat, caz_sat, rng_sat, freq = args[3], args[4], args[5], args[7]
            alt_deg = 90 - np.degrees(dec_sat)
            az_deg = -np.degrees(caz_sat)

            # Vectorized environmental factor calculation
            env_factors = calculate_comprehensive_environmental_effects_vectorized(
                alt_deg, az_deg, rng_sat, freq, environment
            )

            # If satellite is completely blocked, return zero
            if env_factors['total_factor'] == 0.0:
                return 0.0
        else:
            env_factors = {'total_factor': 1.0}

        # Extract radial velocities from the corrected data
        if doppler_corrected_data is not None and len(doppler_corrected_data) > 0:
            # Use average radial velocity for now (simplified approach)
            avg_radial_velocity = np.mean(doppler_corrected_data['radial_velocities'])

            # Use the combined function that handles BOTH effects
            if use_enhanced_transmitters:
                new_args = list(args[:6]) + [selected_transmitter] + [args[7]]
                result = link_budget_doppler_transmitter(
                    *new_args,
                    radial_velocities=avg_radial_velocity, **kwargs
                )
            else:
                # Use standard Doppler correction only
                result = lnk_bdgt_with_doppler_correction(*args, radial_velocities=avg_radial_velocity, **kwargs)
        else:
            # No Doppler data available
            if use_enhanced_transmitters:
                # Use the captured transmitter object to avoid closure issues
                if len(args) >= 8:  # Ensure we have enough arguments
                    new_args = list(args[:6]) + [args[7]] + [selected_transmitter]
                    result = sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
                else:
                    # Fallback to basic function if not enough arguments
                    result = sat_link_budget_vectorized(*args, **kwargs)
            else:
                result = sat_link_budget_vectorized(*args, **kwargs)

        # Apply environmental factor
        return result * env_factors['total_factor']

    starlink_constellation = Constellation.from_file(
        file_traj_sats_path, observ, sat_transmit, lnk_bdgt_with_doppler_and_enhanced,
        name_tag='sat',
        time_tag='timestamp',
        elevation_tag='elevations',
        azimuth_tag='azimuths',
        distance_tag='ranges_westford',
        filt_funcs=(filt_name, filt_el)  # No filtering - all satellites included with correction
    )
else:
    # No Doppler correction (low risk case)
    print("Loading constellation with standard link budget...")

    # Still check if transmitter characteristics should be used
    if use_enhanced_transmitters:
        print("   + transmitter characteristics enabled")

        def lnk_bdgt_enhanced(*args, **kwargs):
            # set very small number for beam_avoid to accept lnk_bdgt_enhanced at model_observed_temp
            # Otherwise, model_observed_temp will always use lnk_bdgt
            kwargs['beam_avoid'] = 1e-20
            kwargs['turn_off'] = False

            # VECTORIZED ENVIRONMENTAL EFFECTS CALCULATION
            if environmental_config is not None:
                # Convert to degrees for environmental calculations
                dec_sat, caz_sat, rng_sat, freq = args[3], args[4], args[5], args[7]
                alt_deg = 90 - np.degrees(dec_sat)
                az_deg = -np.degrees(caz_sat)

                # Vectorized environmental factor calculation
                env_factors = calculate_comprehensive_environmental_effects_vectorized(
                    alt_deg, az_deg, rng_sat, freq, environment
                )

                # If satellite is completely blocked, return zero
                if env_factors['total_factor'] == 0.0:
                    return 0.0
            else:
                env_factors = {'total_factor': 1.0}

            if len(args) >= 8:
                # Reorder arguments: move freq to position 6, add transmitter at position 7
                # Use the captured transmitter object to avoid closure issues
                new_args = list(args[:6]) + [args[7]] + [selected_transmitter]
                result = sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
            else:
                # Fallback to basic function if not enough arguments
                result = sat_link_budget_vectorized(*args, **kwargs)

            # Apply environmental factor
            return result * env_factors['total_factor']

        starlink_constellation = Constellation.from_file(
            file_traj_sats_path, observ, sat_transmit, lnk_bdgt_enhanced,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )
    else:
        print("   + Standard transmitter characteristics (baseline)")
        starlink_constellation = Constellation.from_file(
            file_traj_sats_path, observ, sat_transmit, lnk_bdgt,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )

# plot satellites trajectory
list_sats = starlink_constellation.get_sats_name()

sel_sats = list_sats[:len(list_sats)]

for s in sel_sats:
    sat = starlink_constellation.get_sat_traj(s)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, polar=True)
for s in sel_sats:
    sat = starlink_constellation.get_sat_traj(s)
    ax.plot(np.radians(sat['azimuths']), 90 - sat['elevations'])
ax.set_yticks(range(0, 91, 10))
ax.set_yticklabels([str(x) for x in range(90, -1, -10)])
ax.set_theta_zero_location("N")
# plt.show()

# ## Model total power received during observation
# ---
#
# The `model_observed_temp` takes as input the `Observation` and can also take a
# `sky_mdl` function and one or a vector of `Constellation`.

print("Starting result computation (without beam avoidance)...")
# Use atmospheric refraction correction with beam avoidance
atmospheric_refraction_config = {
    'temperature': 288.15,  # K
    'pressure': 101325,     # Pa
    'humidity': 50.0,       # %
    'apply_refraction_correction': True,
    'refraction_model': 'standard'
}

result, refraction_summary = model_observed_temp_with_atmospheric_refraction_vectorized(
    observ, sky_mdl, constellation=starlink_constellation, beam_avoidance=True,
    atmospheric_refraction=atmospheric_refraction_config
)

# The method also have a keyword `beam_avoid` that takes an angle value. If the
# angle between the boresight of a satellite and the telescope pointing direction
# is below this angle, the satellite "steers" away of 45deg.

obs_beam_avoid = Observation.from_dates(start_obs, stop_obs, traj_obj, westford)


def lnk_bdgt_beam_avoid(*args, **kwargs):
    return sat_link_budget_vectorized(*args, beam_avoid=10.0, turn_off=False, **kwargs)


# Beam avoidance constellation with conditional Doppler AND transmitter effects
if use_doppler_correction:
    # Apply both Doppler correction and transmitter characteristics for physics-based prediction
    def lnk_bdgt_beam_avoid_with_doppler_and_transmitter(*args, **kwargs):
        """
        Beam avoidance link budget function with Doppler correction, transmitter characteristics,
        and environmental effects
        """
        # Add beam_avoid and turn_off parameters for beam avoidance functionality
        kwargs['beam_avoid'] = 10.0
        kwargs['turn_off'] = False

        # VECTORIZED ENVIRONMENTAL EFFECTS CALCULATION
        if environmental_config is not None:
            # Convert to degrees for environmental calculations
            dec_sat, caz_sat, rng_sat, freq = args[3], args[4], args[5], args[7]
            alt_deg = 90 - np.degrees(dec_sat)
            az_deg = -np.degrees(caz_sat)

            # Vectorized environmental factor calculation
            env_factors = calculate_comprehensive_environmental_effects_vectorized(
                alt_deg, az_deg, rng_sat, freq, environment
            )

            # If satellite is completely blocked, return zero
            if env_factors['total_factor'] == 0.0:
                return 0.0
        else:
            env_factors = {'total_factor': 1.0}

        if doppler_corrected_data is not None and len(doppler_corrected_data) > 0:
            avg_radial_velocity = np.mean(doppler_corrected_data['radial_velocities'])
            # Use combined function for consistent physics modeling
            if use_enhanced_transmitters:
                # Use the captured transmitter object to avoid closure issues
                new_args = list(args[:6]) + [selected_transmitter] + [args[7]]
                result = link_budget_doppler_transmitter(
                    *new_args,
                    radial_velocities=avg_radial_velocity, **kwargs
                )
            else:
                # Only Doppler correction, no transmitter characteristics
                result = lnk_bdgt_with_doppler_correction(*args, radial_velocities=avg_radial_velocity, **kwargs)
        else:
            # No Doppler but still apply transmitter characteristics
            if use_enhanced_transmitters:
                # Properly handle arguments for comprehensive function
                if len(args) >= 8:  # Ensure we have enough arguments
                    # Use the captured transmitter object to avoid closure issues
                    new_args = list(args[:6]) + [args[7]] + [selected_transmitter]
                    result = sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
                else:
                    # Fallback to basic function if not enough arguments
                    result = lnk_bdgt_beam_avoid(*args, **kwargs)
            else:
                result = lnk_bdgt_beam_avoid(*args, **kwargs)

        # Apply environmental factor
        return result * env_factors['total_factor']

    starlink_const_beam_avoid = Constellation.from_file(
        file_traj_sats_path, observ, sat_transmit, lnk_bdgt_beam_avoid_with_doppler_and_transmitter,
        name_tag='sat',
        time_tag='timestamp',
        elevation_tag='elevations',
        azimuth_tag='azimuths',
        distance_tag='ranges_westford',
        filt_funcs=(filt_name, filt_el)  # No filtering - all satellites included with correction
    )
else:
    # No Doppler correction but still apply transmitter characteristics if enabled
    if use_enhanced_transmitters:
        def lnk_bdgt_beam_avoid_with_transmitter(*args, **kwargs):
            """Beam avoidance link budget function with transmitter characteristics only"""
            # Add beam_avoid and turn_off parameters for beam avoidance functionality
            kwargs['beam_avoid'] = 10.0
            kwargs['turn_off'] = False

            # VECTORIZED ENVIRONMENTAL EFFECTS CALCULATION
            if environmental_config is not None:
                # Convert to degrees for environmental calculations
                dec_sat, caz_sat, rng_sat, freq = args[3], args[4], args[5], args[7]
                alt_deg = 90 - np.degrees(dec_sat)
                az_deg = -np.degrees(caz_sat)

                # Vectorized environmental factor calculation
                env_factors = calculate_comprehensive_environmental_effects_vectorized(
                    alt_deg, az_deg, rng_sat, freq, environment
                )

                # If satellite is completely blocked, return zero
                if env_factors['total_factor'] == 0.0:
                    return 0.0
            else:
                env_factors = {'total_factor': 1.0}

            if len(args) >= 8:  # Ensure we have enough arguments
                # Use the captured transmitter object to avoid closure issues
                new_args = list(args[:6]) + [args[7]] + [selected_transmitter]
                result = sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
            else:
                # Fallback to basic function if not enough arguments
                result = lnk_bdgt_beam_avoid(*args, **kwargs)

            # Apply environmental factor
            return result * env_factors['total_factor']

        starlink_const_beam_avoid = Constellation.from_file(
            file_traj_sats_path, observ, sat_transmit, lnk_bdgt_beam_avoid_with_transmitter,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )
    else:
        # No Doppler, no transmitter characteristics (low risk case)
        starlink_const_beam_avoid = Constellation.from_file(
            file_traj_sats_path, observ, sat_transmit, lnk_bdgt_beam_avoid,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )

print("Starting result_beam_avoid computation (beam avoidance)...")
# Use the integrated model_observed_temp function with beam_avoidance=True
result_beam_avoid, refraction_summary_beam_avoid = model_observed_temp_with_atmospheric_refraction_vectorized(
    obs_beam_avoid, sky_mdl, constellation=starlink_const_beam_avoid, beam_avoidance=True,
    atmospheric_refraction=atmospheric_refraction_config
)

# Without a satellite constellation:

obs_src = Observation.from_dates(start_obs, stop_obs, traj_obj, westford)

print("Starting result_src computation (no satellites)...")

# No satellites still need to have atmospheric refraction correction but no beam_avoidance
result_src, refraction_summary_src = model_observed_temp_with_atmospheric_refraction_vectorized(
    obs_src, sky_mdl, atmospheric_refraction=atmospheric_refraction_config
)

# With a constellation of satellites that are omni-directional and low power:

# create constant gain profile
# declination angles alpha
alphas = np.arange(0, 181)
# azimuth angles beta
betas = np.arange(0, 351, 10)
# minimum gain from sats
sat_gain_min = sat_ant.gain_pat['gains'].min()
# create gain dataframe
gain_pat = pd.DataFrame({
    'alphas': np.repeat(alphas, len(betas)),
    'betas': np.tile(betas, len(alphas)),
    'gains': np.full(len(alphas) * len(betas), sat_gain_min)
})
sat_cst_gain_ant = Antenna.from_dataframe(gain_pat, sat_eta_rad, freq_band)
sat_cst_gain_transmit = Instrument(sat_cst_gain_ant, sat_T_phy, sat_freq, sat_bw, transmit_temp, 1, [])

# Constant gain constellation with conditional Doppler AND transmitter effects
if use_doppler_correction:
    # Apply both Doppler correction and transmitter characteristics for physics-based prediction
    def lnk_bdgt_cst_gain_with_doppler_and_transmitter(*args, **kwargs):
        """Constant gain link budget function with Doppler correction AND transmitter characteristics"""
        # Note: This is for constant gain, not beam avoidance, so no beam_avoid parameters needed

        # set very small number for beam_avoid to accept lnk_bdgt_enhanced at model_observed_temp
        # Otherwise, model_observed_temp will always use lnk_bdgt
        kwargs['beam_avoid'] = 1e-20
        kwargs['turn_off'] = False

        # VECTORIZED ENVIRONMENTAL EFFECTS CALCULATION
        if environmental_config is not None:
            # Convert to degrees for environmental calculations
            dec_sat, caz_sat, rng_sat, freq = args[3], args[4], args[5], args[7]
            alt_deg = 90 - np.degrees(dec_sat)
            az_deg = -np.degrees(caz_sat)

            # Vectorized environmental factor calculation
            env_factors = calculate_comprehensive_environmental_effects_vectorized(
                alt_deg, az_deg, rng_sat, freq, environment
            )

            # If satellite is completely blocked, return zero
            if env_factors['total_factor'] == 0.0:
                return 0.0
        else:
            env_factors = {'total_factor': 1.0}

        if doppler_corrected_data is not None and len(doppler_corrected_data) > 0:
            avg_radial_velocity = np.mean(doppler_corrected_data['radial_velocities'])
            # Use combined function for consistent physics modeling
            if use_enhanced_transmitters:
                # Note: For constant gain, we need to create a Transmitter object from the instrument
                # since link_budget_doppler_transmitter expects a Transmitter object
                temp_transmitter = Transmitter.from_instrument(
                    sat_cst_gain_transmit,
                    polarization='circular',  # Use realistic Starlink characteristics
                    polarization_angle=0.0,   # Circular polarization angle is not relevant
                    harmonics=[(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]  # Realistic harmonics
                )
                # Use the captured transmitter object to avoid closure issues
                new_args = list(args[:6]) + [temp_transmitter] + [args[7]]
                result = link_budget_doppler_transmitter(
                    *new_args,
                    radial_velocities=avg_radial_velocity, **kwargs
                )
            else:
                # Only Doppler correction, no transmitter characteristics
                result = lnk_bdgt_with_doppler_correction(*args, radial_velocities=avg_radial_velocity, **kwargs)
        else:
            # No Doppler but still apply transmitter characteristics
            if use_enhanced_transmitters:
                temp_transmitter = Transmitter.from_instrument(
                    sat_cst_gain_transmit,
                    polarization='circular',  # Use realistic Starlink characteristics
                    polarization_angle=0.0,   # Circular polarization angle is not relevant
                    harmonics=[(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]  # Realistic harmonics
                )
                # Properly handle arguments for comprehensive function
                if len(args) >= 8:  # Ensure we have enough arguments
                    new_args = list(args[:6]) + [args[7]] + [temp_transmitter]
                    result = sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
                else:
                    # Fallback to basic function if not enough arguments
                    result = lnk_bdgt(*args, **kwargs)
            else:
                result = lnk_bdgt(*args, **kwargs)

        # Apply environmental factor
        return result * env_factors['total_factor']

    starlink_cst_gain_constellation = Constellation.from_file(
        file_traj_sats_path, observ, sat_cst_gain_transmit, lnk_bdgt_cst_gain_with_doppler_and_transmitter,
        name_tag='sat',
        time_tag='timestamp',
        elevation_tag='elevations',
        azimuth_tag='azimuths',
        distance_tag='ranges_westford',
        filt_funcs=(filt_name, filt_el)  # No filtering - all satellites included with correction
    )
else:
    # No Doppler correction but still apply transmitter characteristics if enabled
    if use_enhanced_transmitters:
        def lnk_bdgt_cst_gain_with_transmitter(*args, **kwargs):
            """
            Constant gain link budget function with transmitter characteristics and environmental effects
            """
            # Note: This is for constant gain, not beam avoidance, so no beam_avoid parameters needed
            kwargs['beam_avoid'] = 1e-20
            kwargs['turn_off'] = False

            # VECTORIZED ENVIRONMENTAL EFFECTS CALCULATION
            if environmental_config is not None:
                # Convert to degrees for environmental calculations
                dec_sat, caz_sat, rng_sat, freq = args[3], args[4], args[5], args[7]
                alt_deg = 90 - np.degrees(dec_sat)
                az_deg = -np.degrees(caz_sat)

                # Vectorized environmental factor calculation
                env_factors = calculate_comprehensive_environmental_effects_vectorized(
                    alt_deg, az_deg, rng_sat, freq, environment
                )

                # If satellite is completely blocked, return zero
                if env_factors['total_factor'] == 0.0:
                    return 0.0
            else:
                env_factors = {'total_factor': 1.0}

            temp_transmitter = Transmitter.from_instrument(
                sat_cst_gain_transmit,
                polarization='circular',  # Use realistic Starlink characteristics
                polarization_angle=0.0,   # Circular polarization angle is not relevant
                harmonics=[(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]  # Realistic harmonics
            )
            # Properly handle arguments for comprehensive function
            if len(args) >= 8:  # Ensure we have enough arguments
                # Use the captured transmitter object to avoid closure issues
                new_args = list(args[:6]) + [args[7]] + [temp_transmitter]
                result = sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
            else:
                # Fallback to basic function if not enough arguments
                result = lnk_bdgt(*args, **kwargs)

            # Apply environmental factor
            return result * env_factors['total_factor']

        starlink_cst_gain_constellation = Constellation.from_file(
            file_traj_sats_path, observ, sat_cst_gain_transmit, lnk_bdgt_cst_gain_with_transmitter,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )
    else:
        # No Doppler, no transmitter characteristics (low risk case)
        starlink_cst_gain_constellation = Constellation.from_file(
            file_traj_sats_path, observ, sat_cst_gain_transmit, lnk_bdgt,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )

obs_cst_sat_gain = Observation.from_dates(start_obs, stop_obs, traj_obj, westford)

print("Starting result_cst_sat_gain computation (constant satellite gain)...")

# Force beam_avoidance=True to use the link budget function instead of vectorized path
result_cst_sat_gain, refraction_summary_cst_gain = model_observed_temp_with_atmospheric_refraction_vectorized(
    obs_cst_sat_gain, sky_mdl, constellation=starlink_cst_gain_constellation, beam_avoidance=True,
    atmospheric_refraction=atmospheric_refraction_config
)

#
# add original, result_original, without any effects
#

# Define filter for baseline observation
filt_el_observ_original = ('elevations', lambda e: e > 5)

# redefine observ_original to avoid modifying the same array
observ_original = Observation.from_dates(
    start_obs, stop_obs, traj_obj, westford,
    filt_funcs=(filt_el_observ_original,)
)

starlink_constellation_original = Constellation.from_file(
    file_traj_sats_path, observ_original, sat_transmit, lnk_bdgt,
    name_tag='sat',
    time_tag='timestamp',
    elevation_tag='elevations',
    azimuth_tag='azimuths',
    distance_tag='ranges_westford',
    filt_funcs=(filt_name, filt_el)
)

result_original = model_observed_temp(observ_original, sky_mdl, starlink_constellation_original)


# prevent log10 of negative values
def safe_log10(x):
    x = np.array(x)
    x = np.where(x > 0, x, np.nan)
    return np.log10(x)


fig, ax = plt.subplots(figsize=(18, 6))
time_samples = observ.get_time_stamps()

# Observation without beam avoidance
plot_result = temperature_to_power(result[:, 0, 0], bw)
ax.plot(time_samples, 10 * safe_log10(plot_result), label="without beam avoidance")

# Observation with beam avoidance
plot_result = temperature_to_power(result_beam_avoid[:, 0, 0], bw)
ax.plot(time_samples, 10 * safe_log10(plot_result), label="with beam avoidance")

# Observation without constellation
plot_result = temperature_to_power(result_src[:, 0, 0], bw)
ax.plot(time_samples, 10 * safe_log10(plot_result), label="no satellites")

# Observation with constellation of constant gain
plot_result = temperature_to_power(result_cst_sat_gain[:, 0, 0], bw)
ax.plot(time_samples, 10 * safe_log10(plot_result), label="constant satellite gain")

# Observation without any effects: no beam avoidance and no effects
plot_result = temperature_to_power(result_original[:, 0, 0], bw)
ax.plot(time_samples, 10 * safe_log10(plot_result), label="without effects")

ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("Power [dBW]")
ax.grid(True)
fig.tight_layout()
# plt.show()

# # save image
# plot_filename = "tuto_doppler_transmitter_characteristics_environmental_effects.png"
# plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
# print(f"    Plot saved as: {plot_filename}")

# Focusing on a specific time to see where the satellites are located compare to
# the pointing direction of the telescope:

# zoom dates for comparison purpose
start_zoom = datetime.strptime("2025-02-18T15:32:40.000", dateformat)
stop_zoom = datetime.strptime("2025-02-18T15:33:50.000", dateformat)

time_samples = observ.get_time_stamps()
time_zoom = time_samples[(time_samples >= start_zoom) & (time_samples <= stop_zoom)]

fig, ax = plt.subplots(figsize=(10, 5))
# Observation without beam avoidance
zoom_indices = [i for i, t in enumerate(time_samples) if start_zoom <= t <= stop_zoom]
plot_result = temperature_to_power(result[zoom_indices, 0, 0], bw)
ax.plot(time_zoom, 10 * np.log10(plot_result), label="without beam avoidance")
# Observation with beam avoidance
plot_result = temperature_to_power(result_beam_avoid[zoom_indices, 0, 0], bw)
ax.plot(time_zoom, 10 * np.log10(plot_result), label="with beam avoidance")
# Observation without constellation
plot_result = temperature_to_power(result_src[zoom_indices, 0, 0], bw)
ax.plot(time_zoom, 10 * np.log10(plot_result), label="no satellites")
# Observation with constellation of constant gain
plot_result = temperature_to_power(result_cst_sat_gain[zoom_indices, 0, 0], bw)
ax.plot(time_zoom, 10 * np.log10(plot_result), label="constant satellite gain")
# Observation without any effects: no beam avoidance and no effects
plot_result = temperature_to_power(result_original[zoom_indices, 0, 0], bw)
ax.plot(time_zoom, 10 * np.log10(plot_result), label="without effects")

ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("Power [dBW]")
ax.grid(True)
ax.legend()
fig.tight_layout()
# plt.show()

time_study = datetime.strptime("2025-02-18T15:34:29.000", dateformat)
sats_at_t = starlink_constellation.get_sats_names_at_time(time_study)

sel_sats = sats_at_t[:len(sats_at_t)]
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(1, 1, 1, polar=True)
for s in sel_sats:
    sat = starlink_constellation.get_sat_traj(s)
    sat_pt = sat[sat['times'] == time_study]
    ax.scatter(np.radians(sat_pt['azimuths']), 90 - sat_pt['elevations'])
instru_pt = observ.get_traj()[observ.get_traj()['times'] == time_study]
ax.scatter(np.radians(instru_pt['azimuths']), 90 - instru_pt['elevations'],
           marker="*", c="black", s=200)
ax.set_yticks(range(0, 91, 10))
ax.set_yticklabels([str(x) for x in range(90, -1, -10)])
ax.set_theta_zero_location("N")
# plt.show()


# ## Model Power Spectral Density during observation
# ---
#
# The package is also capable of estimating the power for a wider bandwidth and
# different frequency channels (PSD).

# It is possible to increase the number of frequency channels that the simulator
# can compute to visualize the PSD:

# new instrument parameters
new_freq_chan = 164
new_bw = 30e6

# new instrument that simulate the PSD
westford_freqs = Instrument(tel_ant, T_phy, cent_freq, new_bw, T_RX, new_freq_chan, coords)

# new observation
observ_freqs = Observation.from_dates(start_obs, stop_obs, traj_obj, westford_freqs, filt_funcs=(filt_el,))

# Say we define a new transmission pattern from the satellites that depends on the frequency:

# new satellite transmission model that depends on frequency
tmt_profile = np.ones(new_freq_chan)
tmt_profile[:new_freq_chan//10] = 0.0
tmt_profile[-new_freq_chan//10:] = 0.0
tmt_profile[new_freq_chan//2 - new_freq_chan//10:new_freq_chan//2 + new_freq_chan//10] = 0.0
tmt_profile[new_freq_chan//2] = 1.0

freq_bins = westford_freqs.get_center_freq_chans()


def transmit_temp_freqs(tim, freq):
    ind_freq = np.argmin(np.abs(freq_bins - freq))
    return tmt_profile[ind_freq] * power_to_temperature(10**(transmit_pow/10), 1.0)  # in K


plt.figure()
plt.plot(freq_bins, tmt_profile)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Normalized temperature")
# plt.show()

# The rest of the code is run the same way it was before:

# create transmitter instrument
sat_transmit_freqs = Instrument(sat_ant, sat_T_phy, sat_freq, sat_bw, transmit_temp_freqs, new_freq_chan, [])

# Frequency-dependent constellation with conditional Doppler AND transmitter effects
if use_doppler_correction:
    # Apply both Doppler correction and transmitter characteristics for physics-based prediction
    def lnk_bdgt_freqs_with_doppler_and_transmitter(*args, **kwargs):
        """
        Frequency-dependent link budget function with Doppler correction, transmitter characteristics,
        and environmental effects
        """
        # VECTORIZED ENVIRONMENTAL EFFECTS CALCULATION
        if environmental_config is not None:
            # Convert to degrees for environmental calculations
            dec_sat, caz_sat, rng_sat, freq = args[3], args[4], args[5], args[7]
            alt_deg = 90 - np.degrees(dec_sat)
            az_deg = -np.degrees(caz_sat)

            # Vectorized environmental factor calculation
            env_factors = calculate_comprehensive_environmental_effects_vectorized(
                alt_deg, az_deg, rng_sat, freq, environment
            )

            # If satellite is completely blocked, return zero
            if env_factors['total_factor'] == 0.0:
                return 0.0
        else:
            env_factors = {'total_factor': 1.0}

        if doppler_corrected_data is not None and len(doppler_corrected_data) > 0:
            avg_radial_velocity = np.mean(doppler_corrected_data['radial_velocities'])
            if use_enhanced_transmitters:
                # Use combined function for both effects
                # Note: For frequency-dependent case, we need to create a Transmitter object
                temp_transmitter = Transmitter.from_instrument(
                    sat_transmit_freqs,
                    polarization='circular',  # Use realistic Starlink characteristics
                    polarization_angle=0.0,   # Circular polarization angle is not relevant
                    harmonics=[(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]  # Realistic harmonics
                )
                new_args = list(args[:6]) + [temp_transmitter] + [args[7]]
                result = link_budget_doppler_transmitter(
                    *new_args,
                    radial_velocities=avg_radial_velocity, **kwargs
                )
            else:
                # Only Doppler correction, no transmitter characteristics
                result = lnk_bdgt_with_doppler_correction(*args, radial_velocities=avg_radial_velocity, **kwargs)
        else:
            # No Doppler but still apply transmitter characteristics if enabled
            if use_enhanced_transmitters:
                temp_transmitter = Transmitter.from_instrument(
                    sat_transmit_freqs,
                    polarization='circular',  # Use realistic Starlink characteristics
                    polarization_angle=0.0,   # Circular polarization angle is not relevant
                    harmonics=[(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]  # Realistic harmonics
                )
                # Properly handle arguments for comprehensive function
                if len(args) >= 8:  # Ensure we have enough arguments
                    new_args = list(args[:6]) + [args[7]] + [temp_transmitter]
                    result = sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
                else:
                    # Fallback to basic function if not enough arguments
                    result = sat_link_budget_vectorized(*args, **kwargs)
            else:
                result = sat_link_budget_vectorized(*args, **kwargs)

        # Apply environmental factor
        return result * env_factors['total_factor']

    starlink_constellation_freqs = Constellation.from_file(
        file_traj_sats_path, observ_freqs, sat_transmit_freqs, lnk_bdgt_freqs_with_doppler_and_transmitter,
        name_tag='sat',
        time_tag='timestamp',
        elevation_tag='elevations',
        azimuth_tag='azimuths',
        distance_tag='ranges_westford',
        filt_funcs=(filt_name, filt_el)  # No filtering - all satellites included with correction
    )
else:
    # No Doppler correction but still apply transmitter characteristics if enabled
    if use_enhanced_transmitters:
        def lnk_bdgt_freqs_with_transmitter(*args, **kwargs):
            """Frequency-dependent link budget function with transmitter characteristics only"""
            temp_transmitter = Transmitter.from_instrument(
                sat_transmit_freqs,
                polarization='circular',  # Use realistic Starlink characteristics
                polarization_angle=0.0,   # Circular polarization angle is not relevant
                harmonics=[(2.0, 0.1), (3.0, 0.05), (4.0, 0.02)]  # Realistic harmonics
            )
            # Properly handle arguments for comprehensive function
            if len(args) >= 8:  # Ensure we have enough arguments
                new_args = list(args[:6]) + [args[7]] + [temp_transmitter]
                return sat_link_budget_comprehensive_vectorized(*new_args, **kwargs)
            else:
                # Fallback to basic function if not enough arguments
                return sat_link_budget_vectorized(*args, **kwargs)

        starlink_constellation_freqs = Constellation.from_file(
            file_traj_sats_path, observ_freqs, sat_transmit_freqs, lnk_bdgt_freqs_with_transmitter,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )
    else:
        # No Doppler, no transmitter characteristics (low risk case)
        starlink_constellation_freqs = Constellation.from_file(
            file_traj_sats_path, observ_freqs, sat_transmit_freqs, sat_link_budget_vectorized,
            name_tag='sat',
            time_tag='timestamp',
            elevation_tag='elevations',
            azimuth_tag='azimuths',
            distance_tag='ranges_westford',
            filt_funcs=(filt_name, filt_el)
        )

print("Starting result_freqs computation...")

# For now, atmospheric refraction is not considered yet
result_freqs = model_observed_temp(observ_freqs, sky_mdl, starlink_constellation_freqs)
# # Use atmospheric refraction correction for frequency analysis
# result_freqs, refraction_summary_freqs = model_observed_temp_with_atmospheric_refraction_vectorized(
#     observ_freqs, sky_mdl, constellation=starlink_constellation_freqs, beam_avoidance=True,
#     atmospheric_refraction=atmospheric_refraction_config
# )

# spectogram plot
time_samples = observ_freqs.get_time_stamps()
freq_bins = westford_freqs.get_center_freq_chans()
plot_psd = temperature_to_power(result_freqs, bw/freq_chan)
plot_pow = temperature_to_power(result[:, 0, 0], bw)

fig = plt.figure(figsize=(16, 8))
gs = plt.matplotlib.gridspec.GridSpec(2, 2, height_ratios=[1, 0.4],
                                      width_ratios=[1, 0.01])
gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)

ax1 = plt.subplot(gs[0, 0])
psd = ax1.imshow(10 * np.log10(plot_psd[:, 0, :].T), interpolation="nearest",
                 cmap="plasma", aspect="auto")

ax1.set_xlim(-0.5, plot_psd.shape[0] - 0.5)
ax1.set_ylim(-0.5, plot_psd.shape[2] - 0.5)
ax1.set_xlabel("")
ax1.set_xticks(range(plot_psd.shape[0]))
ax1.set_xticklabels([])
ax1.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
ax1.set_yticks(range(plot_psd.shape[2]))
ax1.set_yticklabels([f"{f/1e9:.3f}" for f in freq_bins])
ax1.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
ax1.set_ylabel("Frequency [GHz]")

cbax = plt.subplot(gs[0, 1])
cb = plt.matplotlib.colorbar.Colorbar(ax=cbax, mappable=psd)
cb.set_label("Spectral Power [dB/Hz]")

ax2 = plt.subplot(gs[1, 0])
ax2.plot(range(len(time_samples)), 10 * np.log10(plot_pow), label="without beam avoidance")

ax2.set_xlim(-0.5, plot_psd.shape[0] - 0.5)
ax2.set_xticks(range(plot_psd.shape[0]))
ax2.set_xticklabels(time_samples)
ax2.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
ax2.set_xlabel("Time [UTC]")
ax2.set_ylabel("Power 11.325 GHz [dBW]")
ax2.grid(True)
ax2.legend()
# plt.show()


# ## Model total power over entire sky
# ---
#
# The package also allows modeling the power received at each time samples for
# multiple positions over the sky.

# It is possible to define a `Trajectory` that points at the full sky over each
# time samples:

# sky map
azimuth_grid = np.arange(0, 356, 5)
elevation_grid = np.arange(0, 91, 1)
time_samples = observ.get_time_stamps()

points_data = []
for t in time_samples:
    az_list = []
    el_list = []
    dist_list = []
    for az in azimuth_grid:
        for el in elevation_grid:
            az_list.append(az)
            el_list.append(el)
            dist_list.append(np.inf)
    points_data.append({
        'times': t,
        'azimuths': az_list,
        'elevations': el_list,
        'distances': dist_list
    })

points_df = pd.DataFrame(points_data)
traj_sky = Trajectory(points_df)

# With the new sky model:

# The simulation with satellites gives:

# --- Efficient sky map modeling for a single time_plot ---
# Only compute for the specific time_plot
time_plot = datetime.strptime("2025-02-18T15:34:29.000", dateformat)

# Define your azimuth/elevation grid
azimuth_grid = np.arange(0, 356, 5)
elevation_grid = np.arange(0, 91, 1)
n_az = len(azimuth_grid)
n_el = len(elevation_grid)

# Prepare output array for the case WITH satellites
map_grid = np.zeros((n_el, n_az))

# Loop over the grid for the case WITH satellites
print("Starting map_grid computation...")
for i, el in enumerate(elevation_grid):
    for j, az in enumerate(azimuth_grid):
        point_df = pd.DataFrame({
            'times': [time_plot],
            'azimuths': [az],
            'elevations': [el],
            'distances': [np.inf]
        })
        traj = Trajectory(point_df)
        obs = Observation.from_dates(time_plot, time_plot, traj, westford)
        sky_result = model_observed_temp(obs, sky_mdl, starlink_constellation)
        map_grid[i, j] = sky_result[0, 0, 0]

# Plotting for the case WITH satellites
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(1, 1, 1, polar=True)
pc = ax.pcolormesh(
    np.radians(azimuth_grid),
    90 - elevation_grid,
    10 * np.log10(temperature_to_power(map_grid, bw)),
    cmap="plasma"
)
cbar = plt.colorbar(pc)
cbar.set_label("Power [dBW]")
# Optionally, plot satellite and source positions as before
sats_at_t = starlink_constellation.get_sats_names_at_time(time_plot)
sel_sats = sats_at_t[:len(sats_at_t)]
for s in sel_sats:
    sat = starlink_constellation.get_sat_traj(s)
    sat_pt = sat[sat['times'] == time_plot]
    ax.scatter(np.radians(sat_pt['azimuths']), 90 - sat_pt['elevations'])
src_pt = traj_src.get_traj()[traj_src.get_traj()['times'] == time_plot]
ax.scatter(np.radians(src_pt['azimuths']), 90 - src_pt['elevations'],
           marker="*", c="white", s=10)
ax.set_yticks(range(0, 91, 10))
ax.set_yticklabels([str(x) for x in range(90, -1, -10)])
ax.set_theta_zero_location("N")
# plt.show()

# And without accounting for satellites:

# Prepare output array for the case WITHOUT satellites
map_grid_no_sat = np.zeros((n_el, n_az))

# Loop over the grid for the case WITHOUT satellites
print("Starting map_grid_no_sat computation...")
for i, el in enumerate(elevation_grid):
    for j, az in enumerate(azimuth_grid):
        point_df = pd.DataFrame({
            'times': [time_plot],
            'azimuths': [az],
            'elevations': [el],
            'distances': [np.inf]
        })
        traj = Trajectory(point_df)
        obs = Observation.from_dates(time_plot, time_plot, traj, westford)
        sky_result_no_sat = model_observed_temp(obs, sky_mdl)
        map_grid_no_sat[i, j] = sky_result_no_sat[0, 0, 0]

# Plotting for the case WITHOUT satellites
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(1, 1, 1, polar=True)
pc = ax.pcolormesh(
    np.radians(azimuth_grid),
    90 - elevation_grid,
    10 * np.log10(temperature_to_power(map_grid_no_sat, bw)), cmap="plasma"
)
cbar = plt.colorbar(pc)
cbar.set_label("Power [dBW]")
src_pt = traj_src.get_traj()[traj_src.get_traj()['times'] == time_plot]
ax.scatter(np.radians(src_pt['azimuths']), 90 - src_pt['elevations'],
           marker="*", c="white", s=10)
ax.set_yticks(range(0, 91, 10))
ax.set_yticklabels([str(x) for x in range(90, -1, -10)])
ax.set_theta_zero_location("N")

# Final Summary of EnhancedModeling
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY: ENHANCED MODELING WITH DOPPLER, TRANSMITTER CHARACTERISTICS, AND ENVIRONMENTAL EFFECTS")
print("="*80)

print("📡 Doppler Effect Analysis:")
risk_level = 'Medium/High' if contamination_probability > 0.4 else 'Low'
print(f"   • Risk assessment: {risk_level}")
print(f"   • Contamination probability: {contamination_probability:.1%}")
doppler_status = 'Yes' if use_doppler_correction else 'No'
print(f"   • Doppler correction applied: {doppler_status}")

print("\n🔌 Transmitter Characteristics Analysis:")
enhanced_status = 'Yes' if use_enhanced_transmitters else 'No'
print(f"   • Transmitter characteristics modeling enabled: {enhanced_status}")
if use_enhanced_transmitters:
    print("   • Selected transmitter: realistic_starlink (Starlink circular + Westford linear)")
    pol_type = enhanced_transmitters['realistic_starlink'].get_polarization()
    print(f"   • Polarization: {pol_type} (Starlink satellites)")
    print("   • Receiver polarization: linear (Westford telescope)")
    print(f"   • Polarization mismatch loss: {realistic_polarization_loss:.1f} dB")
    num_harmonics = len(enhanced_transmitters['realistic_starlink'].get_harmonics())
    print(f"   • Harmonics: {num_harmonics}")
    harm_freqs = [f/1e9 for f in enhanced_transmitters['realistic_starlink'].get_harmonic_frequencies()]
    print(f"   • Harmonic frequencies: {harm_freqs} GHz")

print("\n🌍 Environmental Effects Analysis:")
print("   • Terrain masking: Enabled (DEM-based line-of-sight obstruction)")
print(f"   • DEM file: {environmental_config['dem_file']}")
print(f"   • Antenna location: {westford_lat:.6f}°N, {westford_lon:.6f}°W, {westford_elevation:.1f}m")
print(f"   • Atmospheric conditions: {temperature:.1f}K, {pressure/1000:.1f}kPa, {humidity:.0f}% RH")
print("   • Environmental effects included:")
print("     - Terrain masking (line-of-sight obstructions)")
print("     - Atmospheric refraction (signal bending)")
print("     - Atmospheric absorption (frequency-dependent)")
print("     - Water vapor effects (humidity impact)")
print("     - Limb refraction (space-to-space interactions)")
print("     - Antenna mechanical limitations")
print(f"   • Minimum elevation angle: {environmental_config['min_elevation_angle']:.1f}°")
print("   • Terrain ray tracing: Vectorized for performance")
print("   • Fast terrain lookup: Pre-computed 50x50 grid (2.5km radius)")

print("\n⚡ Final Simulation Configuration:")
doppler_enabled = 'Enabled' if use_doppler_correction else 'Disabled'
print(f"   • Doppler correction: {doppler_enabled}")
transmitter_enabled = 'Enabled' if use_enhanced_transmitters else 'Disabled'
print(f"   • Transmitter characteristics: {transmitter_enabled}")
print("   • Environmental effects: Enabled (comprehensive modeling)")

print("\n🎯 Key Differences from Standard Model:")
if use_doppler_correction:
    print("   • Doppler shift compensation in frequency domain")
if use_enhanced_transmitters:
    print("   • Polarization mismatch loss modeling")
    print("   • Harmonic contribution analysis")
    print("   • Enhanced link budget calculations")
    if use_doppler_correction:
        print("   • INTEGRATED: Both effects combined in link_budget_doppler_transmitter()")
print("   • Environmental effects modeling:")
print("     - Terrain masking with DEM data")
print("     - Atmospheric refraction correction")
print("     - Atmospheric absorption and water vapor effects")
print("     - Limb refraction for space-to-space interactions")
print("     - Antenna mechanical limitations")
print("   • COMPREHENSIVE: All three effect categories integrated")
print("     - Doppler effects + Transmitter characteristics + Environmental effects")

print("="*80)
print("Enhanced modeling completed successfully!")
print("="*80)

plt.show()
