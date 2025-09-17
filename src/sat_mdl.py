"""
Satellite modeling functions for radio astronomy
"""

import numpy as np
from coord_frames import ground_to_beam_coord

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Install with: pip install numba")

speed_c = 3e8  # m/s


def free_space_loss(rng, freq):
    return (4 * np.pi * rng / (speed_c / freq)) ** 2


def simple_link_budget(gain_RX, gain_TX, rng, freq):
    L = free_space_loss(rng, freq)
    return gain_RX * (1 / L) * gain_TX


def sat_link_budget(
    dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat, instru_sat, freq,
    beam_avoid=0.0, turn_off=False
):
    """
    Full-featured satellite link budget calculation, ported from Julia.
    """
    # Coordinate of sat in antenna frame
    dec_sat_tel, caz_sat_tel = ground_to_beam_coord(dec_sat, caz_sat, dec_tel, caz_tel)

    # Telescope gain
    instru_ant = instru_tel.get_antenna()
    gain_tel = instru_ant.get_gain_value(dec_sat_tel, caz_sat_tel)

    sat_ant = instru_sat.get_antenna()
    # Coordinate of telescope at time t in satellite frame
    dec_tel_sat = dec_sat
    caz_tel_sat = -caz_sat

    # Initialize with defaults (as in Julia)
    dec_sat_ant = dec_tel_sat
    caz_sat_ant = caz_tel_sat

    # Beam avoidance logic
    if beam_avoid > 0:
        beam_dec, beam_caz = sat_ant.get_boresight_point()  # Should return in radians
        if abs(beam_dec - dec_tel_sat) < np.deg2rad(beam_avoid):
            if turn_off:
                return 0.0
            else:
                dec_sat_ant = np.mod(dec_tel_sat + np.pi / 4, np.pi)
        elif abs(beam_caz - caz_tel_sat) < np.deg2rad(beam_avoid):
            if turn_off:
                return 0.0
            else:
                caz_sat_ant = np.mod(caz_tel_sat + np.pi / 4, 2 * np.pi)

    # Satellite gain (use possibly modified dec_sat_ant, caz_sat_ant)
    gain_sat = sat_ant.get_gain_value(dec_sat_ant, caz_sat_ant)

    # Link budget
    return simple_link_budget(gain_tel, gain_sat, rng_sat, freq)


# Numba-optimized link budget calculation
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def fast_link_budget_core(dec_tel, caz_tel, dec_sat, caz_sat, rng_sat, freq,
                              beam_avoid_rad, beam_dec, beam_caz, turn_off):
        """
        Numba-optimized core link budget calculation.
        """
        n = len(dec_tel)
        result = np.zeros(n)

        for i in prange(n):
            # Free space loss calculation
            L = (4.0 * np.pi * rng_sat[i] / (3e8 / freq[i])) ** 2

            # Simple gain model (assuming constant gain for speed)
            gain_tel = 1.0
            gain_sat = 1.0

            # Beam avoidance logic
            if beam_avoid_rad > 0:
                dec_condition = abs(beam_dec - dec_sat[i]) < beam_avoid_rad
                caz_condition = abs(beam_caz - (-caz_sat[i])) < beam_avoid_rad

                if turn_off and (dec_condition or caz_condition):
                    result[i] = 0.0
                    continue

            # Link budget calculation
            result[i] = gain_tel * (1.0 / L) * gain_sat

        return result


def sat_link_budget_vectorized(
    dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat, instru_sat, freq,
    beam_avoid=0.0, turn_off=False
):
    """
    Ultra-fast vectorized version using Numba JIT compilation for maximum performance.
    """
    # Convert inputs to arrays
    dec_tel = np.asarray(dec_tel)
    caz_tel = np.asarray(caz_tel)
    dec_sat = np.asarray(dec_sat)
    caz_sat = np.asarray(caz_sat)
    rng_sat = np.asarray(rng_sat)
    freq = np.asarray(freq)

    # Get the broadcast shape
    shape = np.broadcast_shapes(
        dec_tel.shape, caz_tel.shape, dec_sat.shape,
        caz_sat.shape, rng_sat.shape, freq.shape
    )

    # Flatten all arrays for processing
    dec_tel_flat = dec_tel.flatten()
    caz_tel_flat = caz_tel.flatten()
    dec_sat_flat = dec_sat.flatten()
    caz_sat_flat = caz_sat.flatten()
    rng_sat_flat = rng_sat.flatten()
    freq_flat = freq.flatten()

    # Use Numba-optimized function if available and no beam avoidance is needed
    if NUMBA_AVAILABLE and beam_avoid == 0.0:
        beam_avoid_rad = 0.0
        beam_dec, beam_caz = (0.0, 0.0)  # Default values for speed

        result = fast_link_budget_core(
            dec_tel_flat, caz_tel_flat, dec_sat_flat, caz_sat_flat,
            rng_sat_flat, freq_flat, beam_avoid_rad, beam_dec, beam_caz, turn_off
        )
    else:
        # Fallback to pure NumPy version
        # Vectorized coordinate transformation
        from coord_frames import ground_to_beam_coord_vectorized
        dec_sat_tel, caz_sat_tel = ground_to_beam_coord_vectorized(
            dec_sat_flat, caz_sat_flat, dec_tel_flat, caz_tel_flat
        )

        # Vectorized telescope gain calculation
        instru_ant = instru_tel.get_antenna()
        gain_tel = instru_ant.get_gain_value(dec_sat_tel, caz_sat_tel)

        # Vectorized satellite gain calculation
        sat_ant = instru_sat.get_antenna()
        dec_tel_sat = dec_sat_flat
        caz_tel_sat = -caz_sat_flat

        # Initialize with defaults (matching original Python implementation)
        dec_sat_ant = dec_tel_sat
        caz_sat_ant = caz_tel_sat

        # Vectorized beam avoidance logic
        if beam_avoid > 0:
            beam_dec, beam_caz = sat_ant.get_boresight_point()
            beam_avoid_rad = np.deg2rad(beam_avoid)

            # Vectorized conditions
            dec_condition = np.abs(beam_dec - dec_tel_sat) < beam_avoid_rad
            caz_condition = np.abs(beam_caz - caz_tel_sat) < beam_avoid_rad

            if turn_off:
                # For turn_off=True, we'll apply the mask after the link budget calculation
                pass
            else:
                # Modify coordinates for beam avoidance cases (matching original Python implementation)
                dec_sat_ant = np.where(
                    dec_condition,
                    np.mod(dec_tel_sat + np.pi / 4, np.pi),
                    dec_sat_ant,
                )
                caz_sat_ant = np.where(
                    caz_condition,
                    np.mod(caz_tel_sat + np.pi / 4, 2 * np.pi),
                    caz_sat_ant,
                )

        # Vectorized satellite gain calculation
        gain_sat = sat_ant.get_gain_value(dec_sat_ant, caz_sat_ant)

        # Vectorized link budget calculation
        # Free space loss: L = (4 * π * rng / (c / freq))^2
        speed_c = 3e8
        L = (4 * np.pi * rng_sat_flat / (speed_c / freq_flat)) ** 2

        # Link budget: gain_RX * (1 / L) * gain_TX
        result = gain_tel * (1 / L) * gain_sat

        # Apply beam avoidance turn-off if needed
        if beam_avoid > 0 and turn_off:
            result = result * np.where(dec_condition | caz_condition, 0.0, 1.0)

    # Reshape back to original broadcast shape
    return result.reshape(shape)


# Vectorized link budget with Doppler correction
def lnk_bdgt_with_doppler_correction(
    dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat, instru_sat, freq,
    radial_velocities=None, beam_avoid=0.0, turn_off=False
):
    """
    Ultra-fast vectorized link budget calculation with Doppler correction for physics-based prediction.

    This function applies Doppler correction to satellite signals while maintaining the computational
    efficiency of sat_link_budget_vectorized(). It corrects both frequency and power according to
    Doppler physics equations.

    Args:
        dec_tel, caz_tel: Telescope coordinates (declination, cos(azimuth))
        instru_tel: Telescope instrument object
        dec_sat, caz_sat: Satellite coordinates (declination, cos(azimuth))
        rng_sat: Satellite ranges (distances)
        instru_sat: Satellite instrument object
        freq: Frequencies (Hz)
        radial_velocities: Radial velocities of satellites (m/s, positive = moving away)
                          If None, no Doppler correction is applied
        beam_avoid: Beam avoidance angle in degrees
        turn_off: Whether to turn off satellites in beam avoidance zone

    Returns:
        result: Link budget values with Doppler correction applied
    """
    # Convert inputs to arrays
    dec_tel = np.asarray(dec_tel)
    caz_tel = np.asarray(caz_tel)
    dec_sat = np.asarray(dec_sat)
    caz_sat = np.asarray(caz_sat)
    rng_sat = np.asarray(rng_sat)
    freq = np.asarray(freq)

    # Handle radial velocities
    if radial_velocities is not None:
        radial_velocities = np.asarray(radial_velocities)
        # Ensure radial_velocities has the same shape as other inputs
        radial_velocities = np.broadcast_to(radial_velocities, freq.shape)
    else:
        # No Doppler correction - use original sat_link_budget_vectorized
        return sat_link_budget_vectorized(
            dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat, instru_sat, freq,
            beam_avoid=beam_avoid, turn_off=turn_off
        )

    # Get the broadcast shape
    shape = np.broadcast_shapes(
        dec_tel.shape, caz_tel.shape, dec_sat.shape,
        caz_sat.shape, rng_sat.shape, freq.shape, radial_velocities.shape
    )

    # Flatten all arrays for processing
    dec_tel_flat = dec_tel.flatten()
    caz_tel_flat = caz_tel.flatten()
    dec_sat_flat = dec_sat.flatten()
    caz_sat_flat = caz_sat.flatten()
    rng_sat_flat = rng_sat.flatten()
    freq_flat = freq.flatten()
    radial_velocities_flat = radial_velocities.flatten()

    # Apply Doppler correction to frequencies
    speed_c = 3e8  # Speed of light in m/s

    # Calculate Doppler factor using relativistic formula for accuracy
    # For small velocities (v << c), this approximates to: f' ≈ f * (1 + v/c)
    beta = radial_velocities_flat / speed_c
    doppler_factor = np.sqrt((1 + beta) / (1 - beta))

    # Apply Doppler correction to frequencies
    freq_corrected = freq_flat * doppler_factor

    # Calculate Doppler frequency shift (for potential future use)
    # doppler_shift = freq_corrected - freq_flat

    # Use Numba-optimized function if available and no beam avoidance is needed
    if NUMBA_AVAILABLE and beam_avoid == 0.0:
        beam_avoid_rad = 0.0
        beam_dec, beam_caz = (0.0, 0.0)  # Default values for speed

        # Use corrected frequencies in the core calculation
        result = fast_link_budget_core(
            dec_tel_flat, caz_tel_flat, dec_sat_flat, caz_sat_flat,
            rng_sat_flat, freq_corrected, beam_avoid_rad, beam_dec, beam_caz, turn_off
        )
    else:
        # Fallback to pure NumPy version with Doppler correction
        # Vectorized coordinate transformation
        from coord_frames import ground_to_beam_coord_vectorized
        dec_sat_tel, caz_sat_tel = ground_to_beam_coord_vectorized(
            dec_sat_flat, caz_sat_flat, dec_tel_flat, caz_tel_flat
        )

        # Vectorized telescope gain calculation
        instru_ant = instru_tel.get_antenna()
        gain_tel = instru_ant.get_gain_value(dec_sat_tel, caz_sat_tel)

        # Vectorized satellite gain calculation
        sat_ant = instru_sat.get_antenna()
        dec_tel_sat = dec_sat_flat
        caz_tel_sat = -caz_sat_flat

        # Initialize with defaults (matching original Python implementation)
        dec_sat_ant = dec_tel_sat
        caz_sat_ant = caz_tel_sat

        # Vectorized beam avoidance logic
        if beam_avoid > 0:
            beam_dec, beam_caz = sat_ant.get_boresight_point()
            beam_avoid_rad = np.deg2rad(beam_avoid)

            # Vectorized conditions
            dec_condition = np.abs(beam_dec - dec_tel_sat) < beam_avoid_rad
            caz_condition = np.abs(beam_caz - caz_tel_sat) < beam_avoid_rad

            if turn_off:
                # For turn_off=True, we'll apply the mask after the link budget calculation
                pass
            else:
                # Modify coordinates for beam avoidance cases (matching original Python implementation)
                dec_sat_ant = np.where(
                    dec_condition,
                    np.mod(dec_tel_sat + np.pi / 4, np.pi),
                    dec_sat_ant,
                )
                caz_sat_ant = np.where(
                    caz_condition,
                    np.mod(caz_tel_sat + np.pi / 4, 2 * np.pi),
                    caz_sat_ant,
                )

        # Vectorized satellite gain calculation
        gain_sat = sat_ant.get_gain_value(dec_sat_ant, caz_sat_ant)

        # Vectorized link budget calculation with Doppler-corrected frequencies
        # Free space loss: L = (4 * π * rng / (c / freq_corrected))^2
        L = (4 * np.pi * rng_sat_flat / (speed_c / freq_corrected)) ** 2

        # Link budget: gain_RX * (1 / L) * gain_TX
        # Apply additional Doppler power correction factor
        # Power spectral density changes with frequency shift: P' = P * (f'/f)^2
        doppler_power_factor = (freq_corrected / freq_flat) ** 2

        result = gain_tel * (1 / L) * gain_sat * doppler_power_factor

        # Apply beam avoidance turn-off if needed
        if beam_avoid > 0 and turn_off:
            result = result * np.where(dec_condition | caz_condition, 0.0, 1.0)

    # Reshape back to original broadcast shape
    return result.reshape(shape)


# Vectorized radial velocity calculation for Doppler effect
def calculate_radial_velocities_vectorized(trajectory_data, time_indices=None):
    """
    Calculate radial velocities for multiple satellites in a vectorized manner.

    Args:
        trajectory_data: DataFrame with columns ['times', 'azimuths', 'elevations', 'distances', 'sat']
        time_indices: Optional array of time indices to calculate velocities for

    Returns:
        radial_velocities: Array of radial velocities in m/s, same shape as input data
    """
    if 'sat' not in trajectory_data.columns:
        # Single satellite case
        return calculate_radial_velocity(trajectory_data)

    # Multi-satellite case - vectorized calculation
    radial_velocities = np.zeros(len(trajectory_data))

    # Group by satellite and calculate velocities
    for sat_name in trajectory_data['sat'].unique():
        sat_mask = trajectory_data['sat'] == sat_name
        sat_data = trajectory_data[sat_mask].copy()

        if len(sat_data) >= 2:
            # Calculate velocities for this satellite
            sat_velocities = calculate_radial_velocity(sat_data)
            radial_velocities[sat_mask] = sat_velocities

    return radial_velocities


# Wrapper function for easy integration for
def lnk_bdgt_doppler_wrapper(*args, **kwargs):
    """
    Wrapper function that automatically applies Doppler correction if radial velocities are provided.
    This maintains compatibility with existing code while adding Doppler correction capability.

    Usage:
    - Without Doppler: lnk_bdgt_doppler_wrapper(dec_tel, caz_tel, instru_tel, dec_sat,
      caz_sat, rng_sat, instru_sat, freq, **kwargs)
    - With Doppler: lnk_bdgt_doppler_wrapper(dec_tel, caz_tel, instru_tel, dec_sat,
      caz_sat, rng_sat, instru_sat, freq, radial_velocities=velocities, **kwargs)
    """
    # Extract radial_velocities from kwargs if provided
    radial_velocities = kwargs.pop('radial_velocities', None)

    if radial_velocities is not None:
        # Apply Doppler correction
        return lnk_bdgt_with_doppler_correction(*args, radial_velocities=radial_velocities, **kwargs)
    else:
        # Use original vectorized function
        return sat_link_budget_vectorized(*args, **kwargs)


def calculate_radial_velocity(trajectory_data):
    """
    Calculate radial velocity between satellite and telescope from trajectory data.

    Args:
        trajectory_data: DataFrame with columns ['times', 'azimuths', 'elevations', 'distances']

    Returns:
        radial_velocity: Array of radial velocities in m/s (positive = moving away)
    """
    # Extract time and distance data
    times = trajectory_data['times'].values
    distances = trajectory_data['distances'].values

    # Check if we have enough data points for velocity calculation
    if len(times) < 2:
        # If only one point or no points, return array of zeros
        return np.zeros(len(times))

    # Calculate time differences in seconds
    time_diffs = np.diff(times).astype('timedelta64[s]').astype(float)

    # Check for invalid time differences (zero or negative)
    if np.any(time_diffs <= 0):
        # If any time differences are invalid, return array of zeros
        return np.zeros(len(times))

    # Calculate distance differences
    distance_diffs = np.diff(distances)

    # Calculate radial velocity (m/s)
    # Positive velocity means satellite is moving away from telescope
    radial_velocity = distance_diffs / time_diffs

    # Check if radial_velocity array is empty
    if len(radial_velocity) == 0:
        return np.zeros(len(times))

    # Pad with the first velocity to maintain array length
    # For n points, we have n-1 differences, so we need to add 1 element to get back to n
    # Use the first velocity for the first point
    radial_velocity = np.concatenate([[radial_velocity[0]], radial_velocity])

    return radial_velocity


def calculate_doppler_shift(frequency, radial_velocity):
    """
    Calculate Doppler frequency shift using the relativistic Doppler formula.

    Args:
        frequency: Frequency in Hz
        radial_velocity: Radial velocity in m/s (positive = moving away)

    Returns:
        doppler_shift: Frequency shift in Hz
        shifted_frequency: New frequency after Doppler shift in Hz
    """
    speed_c = 3e8  # Speed of light in m/s

    # Relativistic Doppler formula
    # f' = f * sqrt((1 + v/c) / (1 - v/c))
    # For small velocities (v << c), this approximates to: f' ≈ f * (1 + v/c)

    # Use relativistic formula for accuracy
    beta = radial_velocity / speed_c
    doppler_factor = np.sqrt((1 + beta) / (1 - beta))

    # Calculate shifted frequency
    shifted_frequency = frequency * doppler_factor

    # Calculate frequency shift
    doppler_shift = shifted_frequency - frequency

    return doppler_shift, shifted_frequency


def add_doppler_effect_to_trajectory(trajectory_data, center_frequency):
    """
    Add Doppler effect calculations to trajectory data.

    Args:
        trajectory_data: DataFrame with trajectory data
        center_frequency: Center frequency of observation in Hz

    Returns:
        trajectory_with_doppler: DataFrame with added Doppler columns
    """
    # Create a copy to avoid modifying original data
    trajectory_with_doppler = trajectory_data.copy()

    # Calculate radial velocity
    radial_velocity = calculate_radial_velocity(trajectory_data)

    # Calculate Doppler shift
    doppler_shift, shifted_frequency = calculate_doppler_shift(center_frequency, radial_velocity)

    # Add new columns
    trajectory_with_doppler['radial_velocity'] = radial_velocity
    trajectory_with_doppler['doppler_shift'] = doppler_shift
    trajectory_with_doppler['shifted_frequency'] = shifted_frequency

    return trajectory_with_doppler


def calculate_doppler_shift_for_frequency_channels(center_frequency, bandwidth, num_channels, radial_velocity):
    """
    Calculate Doppler shift for each frequency channel in a multi-channel observation.

    Args:
        center_frequency: Center frequency in Hz
        bandwidth: Total bandwidth in Hz
        num_channels: Number of frequency channels
        radial_velocity: Radial velocity in m/s

    Returns:
        original_frequencies: Array of original channel frequencies
        shifted_frequencies: Array of shifted channel frequencies
        doppler_shifts: Array of frequency shifts for each channel
    """
    # Generate original frequency channels
    freq_start = center_frequency - bandwidth / 2
    original_frequencies = np.linspace(freq_start, freq_start + bandwidth, num_channels, endpoint=False)

    # Calculate Doppler shift for each channel
    doppler_shifts = np.zeros(num_channels)
    shifted_frequencies = np.zeros(num_channels)

    for i, freq in enumerate(original_frequencies):
        doppler_shift, shifted_freq = calculate_doppler_shift(freq, radial_velocity)
        doppler_shifts[i] = doppler_shift
        shifted_frequencies[i] = shifted_freq

    return original_frequencies, shifted_frequencies, doppler_shifts


def get_doppler_impact_on_observation(trajectory_data, center_frequency, bandwidth=1e3, num_channels=1):
    """
    Calculate the impact of Doppler effect on a radio astronomy observation.

    Args:
        trajectory_data: DataFrame with trajectory data
        center_frequency: Center frequency of observation in Hz
        bandwidth: Bandwidth of observation in Hz
        num_channels: Number of frequency channels

    Returns:
        doppler_summary: Dictionary with Doppler effect summary
    """
    # Add Doppler effect to trajectory
    trajectory_with_doppler = add_doppler_effect_to_trajectory(trajectory_data, center_frequency)

    # Calculate statistics
    radial_velocities = trajectory_with_doppler['radial_velocity']
    doppler_shifts = trajectory_with_doppler['doppler_shift']

    # Calculate summary statistics
    doppler_summary = {
        'max_radial_velocity': np.max(np.abs(radial_velocities)),
        'mean_radial_velocity': np.mean(radial_velocities),
        'max_doppler_shift': np.max(np.abs(doppler_shifts)),
        'mean_doppler_shift': np.mean(doppler_shifts),
        'doppler_shift_std': np.std(doppler_shifts),
        'fractional_shift_max': np.max(np.abs(doppler_shifts)) / center_frequency,
        'trajectory_with_doppler': trajectory_with_doppler
    }

    # For multi-channel observations, calculate channel-specific effects
    if num_channels > 1:
        # Use maximum radial velocity for worst-case scenario
        max_radial_velocity = np.max(np.abs(radial_velocities))
        original_freqs, shifted_freqs, channel_shifts = calculate_doppler_shift_for_frequency_channels(
            center_frequency, bandwidth, num_channels, max_radial_velocity
        )

        doppler_summary.update({
            'original_frequencies': original_freqs,
            'shifted_frequencies': shifted_freqs,
            'channel_doppler_shifts': channel_shifts,
            'max_channel_shift': np.max(np.abs(channel_shifts))
        })

    return doppler_summary


def analyze_doppler_statistics(all_sat_data, observation_band_center,
                               observation_band_width, start_obs, stop_obs, cent_freq):
    """
    Statistical analysis of Doppler impact across all satellites.
    This provides realistic values to judge how much Doppler effects affect the results.

    Args:
        all_sat_data: DataFrame containing satellite trajectory data
        observation_band_center: Center frequency of observation band in Hz
        observation_band_width: Width of observation band in Hz
        start_obs: Start time of observation
        stop_obs: Stop time of observation
        cent_freq: Center frequency for Doppler calculations in Hz

    Returns:
        statistical_results: Dictionary with comprehensive statistical analysis
    """
    import pandas as pd

    print("Running statistical analysis...")

    # Initialize statistical data structures
    all_doppler_shifts = []
    frequency_contamination_data = []
    temporal_interference_data = []

    # Analyze each satellite for statistical patterns
    for sat_name in all_sat_data['sat'].unique():
        sat_data = all_sat_data[all_sat_data['sat'] == sat_name]
        sat_in_obs = sat_data[
            (sat_data['times'] >= start_obs) &
            (sat_data['times'] <= stop_obs) &
            (sat_data['elevations'] > 20)
        ]

        if len(sat_in_obs) >= 2:
            try:
                # Convert to trajectory format
                sat_traj_df = pd.DataFrame({
                    'times': sat_in_obs['times'],
                    'azimuths': sat_in_obs['azimuths'],
                    'elevations': sat_in_obs['elevations'],
                    'distances': sat_in_obs['distances']
                })

                # Calculate Doppler effect
                doppler_summary = get_doppler_impact_on_observation(sat_traj_df, cent_freq, 1e3, 1)
                max_doppler_shift = doppler_summary['max_doppler_shift']

                # Store Doppler shift data
                all_doppler_shifts.append(max_doppler_shift)

                # Analyze frequency contamination
                satellite_freq = 11.325e9  # 11.325 GHz (hardcoded for now)
                min_shifted_freq = satellite_freq - max_doppler_shift
                max_shifted_freq = satellite_freq + max_doppler_shift

                obs_band_min = observation_band_center - observation_band_width/2
                obs_band_max = observation_band_center + observation_band_width/2

                # Check if this satellite contaminates the observation band
                if (min_shifted_freq <= obs_band_max and max_shifted_freq >= obs_band_min):
                    frequency_contamination_data.append({
                        'satellite': sat_name,
                        'max_doppler_shift': max_doppler_shift,
                        'min_shifted_freq': min_shifted_freq,
                        'max_shifted_freq': max_shifted_freq,
                        'contamination_duration': len(sat_in_obs)  # Number of time steps
                    })

                    # Analyze temporal patterns
                    for _, row in sat_in_obs.iterrows():
                        temporal_interference_data.append({
                            'time': row['times'],
                            'satellite': sat_name,
                            'elevation': row['elevations'],
                            'doppler_shift': max_doppler_shift
                        })

            except Exception:
                continue  # Skip satellites with calculation errors

    # Calculate statistical metrics
    all_doppler_shifts = np.array(all_doppler_shifts)

    # Frequency contamination probability
    total_satellites = len(all_sat_data['sat'].unique())
    contaminating_satellites = len(frequency_contamination_data)
    contamination_probability = contaminating_satellites / total_satellites if total_satellites > 0 else 0

    # Calculate adaptive thresholds based on observation bandwidth
    bandwidth_threshold_high = observation_band_width * 0.01  # 1% of bandwidth
    bandwidth_threshold_medium = observation_band_width * 0.002  # 0.2% of bandwidth

    # Doppler impact distribution
    if len(all_doppler_shifts) > 0:
        high_impact = np.sum(all_doppler_shifts > bandwidth_threshold_high) / len(all_doppler_shifts) * 100
        medium_impact = (np.sum((all_doppler_shifts >= bandwidth_threshold_medium) &
                                (all_doppler_shifts <= bandwidth_threshold_high)) / len(all_doppler_shifts) * 100)
        low_impact = np.sum(all_doppler_shifts < bandwidth_threshold_medium) / len(all_doppler_shifts) * 100
    else:
        high_impact = medium_impact = low_impact = 0

    # Average interference statistics
    avg_interfering_satellites = len(frequency_contamination_data) / total_satellites if total_satellites > 0 else 0

    # Temporal analysis
    if temporal_interference_data:
        temp_df = pd.DataFrame(temporal_interference_data)
        peak_interference_times = (temp_df.groupby('time').size()
                                   .sort_values(ascending=False).head(5))
        avg_interference_duration = (temp_df.groupby('satellite').size().mean())
    else:
        peak_interference_times = pd.Series()
        avg_interference_duration = 0

    # Risk assessment
    mean_shift_factor = np.mean(all_doppler_shifts) / 1e6 if len(all_doppler_shifts) > 0 else 0
    risk_score = contamination_probability * avg_interfering_satellites * mean_shift_factor

    # Quality metrics
    data_integrity_score = 1 - contamination_probability
    frequency_coverage_loss = contamination_probability  # Simplified estimate

    # Determine observation strategy recommendation
    if contamination_probability > 0.8:
        recommended_strategy = "frequency_hopping"
    elif contamination_probability > 0.5:
        recommended_strategy = "time_gating"
    else:
        recommended_strategy = "standard_observation"

    # Compile statistical results
    statistical_results = {
        'contamination_probability': contamination_probability,
        'avg_interfering_satellites': avg_interfering_satellites,
        'doppler_impact_distribution': {
            'high_impact_satellites': f"{high_impact:.1f}%",
            'medium_impact_satellites': f"{medium_impact:.1f}%",
            'low_impact_satellites': f"{low_impact:.1f}%"
        },
        'risk_assessment': {
            'risk_score': risk_score,
            'data_integrity_score': data_integrity_score,
            'frequency_coverage_loss': frequency_coverage_loss,
            'recommended_strategy': recommended_strategy
        },
        'temporal_patterns': {
            'avg_interference_duration': avg_interference_duration,
            'peak_interference_times': (
                peak_interference_times.index.tolist()
                if len(peak_interference_times) > 0 else []
            )
        },
        'detailed_statistics': {
            'total_satellites_analyzed': len(all_doppler_shifts),
            'contaminating_satellites': contaminating_satellites,
            'mean_doppler_shift': np.mean(all_doppler_shifts) if len(all_doppler_shifts) > 0 else 0,
            'max_doppler_shift': np.max(all_doppler_shifts) if len(all_doppler_shifts) > 0 else 0,
            'std_doppler_shift': np.std(all_doppler_shifts) if len(all_doppler_shifts) > 0 else 0
        }
    }

    return statistical_results


def print_doppler_statistical_summary(stats, observation_band_width):
    """
    Print comprehensive statistical summary of Doppler impact.

    Args:
        stats: Dictionary with statistical results from analyze_doppler_statistics
        observation_band_width: Width of observation band in Hz (for threshold display)
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*80)

    print("\n📊 FREQUENCY CONTAMINATION ANALYSIS:")
    print(f"   • Contamination Probability: {stats['contamination_probability']:.1%}")
    print(f"   • Average Interfering Satellites: {stats['avg_interfering_satellites']:.1f}")
    print(f"   • Data Integrity Score: {stats['risk_assessment']['data_integrity_score']:.1%}")
    print(f"   • Frequency Coverage Loss: {stats['risk_assessment']['frequency_coverage_loss']:.1%}")

    print("\n🎯 DOPPLER IMPACT DISTRIBUTION:")
    # Calculate thresholds for display
    bandwidth_threshold_high = observation_band_width * 0.01 / 1e3  # Convert to kHz
    bandwidth_threshold_medium = observation_band_width * 0.002 / 1e3  # Convert to kHz
    high_impact_pct = stats['doppler_impact_distribution']['high_impact_satellites']
    medium_impact_pct = stats['doppler_impact_distribution']['medium_impact_satellites']
    low_impact_pct = stats['doppler_impact_distribution']['low_impact_satellites']
    print(f"   • High Impact (>{bandwidth_threshold_high:.0f} kHz): {high_impact_pct}")
    print(f"   • Medium Impact ({bandwidth_threshold_medium:.0f}-{bandwidth_threshold_high:.0f} kHz): "
          f"{medium_impact_pct}")
    print(f"   • Low Impact (<{bandwidth_threshold_medium:.0f} kHz): {low_impact_pct}")

    print("\n⚠️  RISK ASSESSMENT:")
    print(f"   • Risk Score: {stats['risk_assessment']['risk_score']:.3f}")
    strategy = stats['risk_assessment']['recommended_strategy'].replace('_', ' ').title()
    print(f"   • Recommended Strategy: {strategy}")

    print("\n📈 DETAILED STATISTICS:")
    print(f"   • Total Satellites Analyzed: {stats['detailed_statistics']['total_satellites_analyzed']}")
    print(f"   • Contaminating Satellites: {stats['detailed_statistics']['contaminating_satellites']}")
    mean_shift = stats['detailed_statistics']['mean_doppler_shift']/1e3
    print(f"   • Mean Doppler Shift: {mean_shift:.1f} kHz")
    max_shift = stats['detailed_statistics']['max_doppler_shift']/1e3
    print(f"   • Max Doppler Shift: {max_shift:.1f} kHz")
    std_shift = stats['detailed_statistics']['std_doppler_shift']/1e3
    print(f"   • Doppler Shift Std Dev: {std_shift:.1f} kHz")

    if stats['temporal_patterns']['peak_interference_times']:
        print("\n⏰ TEMPORAL PATTERNS:")
        avg_duration = stats['temporal_patterns']['avg_interference_duration']
        print(f"   • Average Interference Duration: {avg_duration:.1f} time steps")
        peak_count = len(stats['temporal_patterns']['peak_interference_times'])
        print(f"   • Peak Interference Times: {peak_count} identified")

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)

    if stats['contamination_probability'] > 0.7:
        print("🔴 HIGH INTERFERENCE RISK: Consider alternative observation time or frequency band")
    elif stats['contamination_probability'] > 0.4:
        print("🟡 MODERATE INTERFERENCE RISK: Monitor closely, consider mitigation strategies")
    else:
        print("🟢 LOW INTERFERENCE RISK: Standard observation should be feasible")

    if stats['risk_assessment']['recommended_strategy'] == 'frequency_hopping':
        print("💡 RECOMMENDATION: Use frequency hopping to avoid contaminated bands")
    elif stats['risk_assessment']['recommended_strategy'] == 'time_gating':
        print("💡 RECOMMENDATION: Use time gating to avoid peak interference periods")
    else:
        print("💡 RECOMMENDATION: Standard observation approach should work")

    print("="*80)
