"""
Plotting utilities for RSC-SIM tutorials.

This module provides common plotting functions and utilities for
visualizing antenna patterns, trajectories, and observation results.
"""

import matplotlib.pyplot as plt
import numpy as np
from .config import (
    STANDARD_FIGURE_SIZE,
    # LARGE_FIGURE_SIZE,
    POLAR_FIGURE_SIZE,
    TEMPERATURE_COLORMAP,
    POWER_COLORMAP
)


def setup_plotting():
    """
    Set up matplotlib plotting parameters for consistent styling.
    """
    plt.rcParams['figure.figsize'] = STANDARD_FIGURE_SIZE
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def create_polar_plot(figsize=POLAR_FIGURE_SIZE):
    """
    Create a polar plot with standard formatting.

    Args:
        figsize: Figure size tuple

    Returns:
        tuple: (figure, axes) objects
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
    ax.set_theta_zero_location("N")
    return fig, ax


def create_antenna_polar_plot(figsize=POLAR_FIGURE_SIZE):
    """
    Create a polar plot specifically for antenna patterns with gain scaling.

    Args:
        figsize: Figure size tuple

    Returns:
        tuple: (figure, axes) objects
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
    ax.set_theta_zero_location("N")
    return fig, ax


def plot_antenna_pattern(antenna, title="Satellite Antenna Pattern"):
    """
    Plot antenna gain pattern using the same approach as the original code.
    Shows elevation cuts for different beta angles.

    Args:
        antenna: Antenna object
        title: Plot title

    Returns:
        tuple: (figure, axes) objects
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE, subplot_kw={'projection': 'polar'})

    # Use the same approach as the original code
    nb_curv = 5  # number of slices to plot
    alphas, betas = antenna.get_def_angles()
    step_beta_ind = len(betas) // (2 * nb_curv)

    for i in range(0, len(betas) // 2, step_beta_ind):
        a, g = antenna.get_slice_gain(betas[i])
        ax.plot(np.radians(a), 10 * np.log10(g), label=f"β = {betas[i]}deg")

    ax.legend()
    ax.set_title(title, pad=20)
    ax.grid(True)

    # Set theta direction and zero location to match the image
    ax.set_theta_zero_location("E")  # 0 degrees to the East (right)
    ax.set_theta_direction(1)  # Counter-clockwise direction

    # Add gain information
    max_gain = antenna.get_boresight_gain()
    max_gain_db = 10 * np.log10(max_gain)
    ax.text(0.02, 0.98, f'Max Gain: {max_gain_db:.1f} dB',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return fig, ax


def plot_trajectory_comparison(source_traj, pointing_traj, title="Source vs Pointing Trajectory"):
    """
    Plot comparison of source and pointing trajectories.

    Args:
        source_traj: Source trajectory data
        pointing_traj: Pointing trajectory data
        title: Plot title

    Returns:
        tuple: (figure, axes) objects
    """
    fig, ax = create_polar_plot()

    ax.plot(np.radians(source_traj['azimuths']),
            90 - source_traj['elevations'],
            label="source", linewidth=2)
    ax.plot(np.radians(pointing_traj['azimuths']),
            90 - pointing_traj['elevations'],
            label="pointing", linewidth=2)

    # Set standard ticks like the original code
    ax.set_yticks(range(0, 91, 10))
    ax.set_yticklabels([str(x) for x in range(90, -1, -10)])

    ax.legend()
    ax.set_title(title)

    return fig, ax


def plot_sky_temperature_map(sky_temp, azimuth_grid, elevation_grid,
                             title="Sky Temperature Map", colormap=TEMPERATURE_COLORMAP):
    """
    Plot sky temperature map in polar coordinates.

    Args:
        sky_temp: 2D array of sky temperatures
        azimuth_grid: Azimuth grid array
        elevation_grid: Elevation grid array
        title: Plot title
        colormap: Colormap name

    Returns:
        tuple: (figure, axes) objects
    """
    fig, ax = create_polar_plot()

    pc = ax.pcolormesh(np.radians(azimuth_grid),
                       90 - elevation_grid,
                       sky_temp,
                       cmap=colormap)

    cbar = plt.colorbar(pc)
    cbar.set_label("Temperature [K]")
    ax.set_title(title)

    return fig, ax


def plot_power_time_series(time_samples, power_data, labels,
                           title="Power vs Time", ylabel="Power [dBW]"):
    """
    Plot power time series for multiple scenarios.

    Args:
        time_samples: Time array
        power_data: List of power arrays
        labels: List of labels for each power array
        title: Plot title
        ylabel: Y-axis label

    Returns:
        tuple: (figure, axes) objects
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)

    for power, label in zip(power_data, labels):
        ax.plot(time_samples, 10 * np.log10(power), label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    return fig, ax


def plot_psd_spectrogram(time_samples, freq_bins, psd_data, single_channel_data=None,
                         title="Power Spectral Density", colormap=POWER_COLORMAP):
    """
    Plot power spectral density spectrogram.

    Args:
        time_samples: Time array
        freq_bins: Frequency bins array
        psd_data: 3D PSD data array
        single_channel_data: Optional single-channel power data for comparison
        title: Plot title
        colormap: Colormap name

    Returns:
        tuple: (figure, axes) objects
    """
    # Match original code structure exactly
    fig = plt.figure(figsize=(16, 8))
    gs = plt.matplotlib.gridspec.GridSpec(2, 2, height_ratios=[1, 0.4],
                                          width_ratios=[1, 0.01])
    gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)

    # Main spectrogram - match original exactly
    ax1 = plt.subplot(gs[0, 0])
    psd = ax1.imshow(10 * np.log10(psd_data[:, 0, :].T),
                     interpolation="nearest",
                     cmap=colormap,
                     aspect="auto")

    ax1.set_xlim(-0.5, psd_data.shape[0] - 0.5)
    ax1.set_ylim(-0.5, psd_data.shape[2] - 0.5)
    ax1.set_xlabel("")
    ax1.set_xticks(range(psd_data.shape[0]))
    ax1.set_xticklabels([])
    ax1.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    ax1.set_yticks(range(psd_data.shape[2]))
    ax1.set_yticklabels([f"{f/1e9:.3f}" for f in freq_bins])
    ax1.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    ax1.set_ylabel("Frequency [GHz]")

    # Colorbar - match original exactly
    cbax = plt.subplot(gs[0, 1])
    cb = plt.matplotlib.colorbar.Colorbar(ax=cbax, mappable=psd)
    cb.set_label("Spectral Power [dB/Hz]")

    # Time series below - show only PSD center frequency (without beam avoidance)
    ax2 = plt.subplot(gs[1, 0])

    # Plot PSD center frequency
    center_freq_idx = psd_data.shape[2] // 2
    psd_center = psd_data[:, 0, center_freq_idx]
    ax2.plot(range(len(time_samples)),
             10 * np.log10(psd_center),
             label="without beam avoidance", linewidth=2)

    ax2.set_xlim(-0.5, psd_data.shape[0] - 0.5)
    ax2.set_xticks(range(psd_data.shape[0]))
    ax2.set_xticklabels(time_samples)
    ax2.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    ax2.set_xlabel("Time [UTC]")
    center_freq_ghz = freq_bins[center_freq_idx] / 1e9
    ax2.set_ylabel(f"Power at {center_freq_ghz:.3f} GHz [dBW]")
    ax2.grid(True)
    ax2.legend()

    # Set the overall figure title
    fig.suptitle(title, fontsize=14, y=0.98)

    return fig, (ax1, ax2), psd


def plot_satellite_positions(constellation, time_study, source_traj=None,
                             title="Satellite Positions"):
    """
    Plot satellite positions at a specific time.

    Args:
        constellation: Constellation object
        time_study: Time to plot positions
        source_traj: Optional source trajectory for comparison
        title: Plot title

    Returns:
        tuple: (figure, axes) objects
    """
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(1, 1, 1, polar=True)

    # Define a color palette for satellites
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12 distinct colors from Set3 colormap

    # Plot satellite positions at the specific time with different colors
    sats_at_t = constellation.get_sats_names_at_time(time_study)
    sel_sats_at_t = sats_at_t[:len(sats_at_t)]

    for i, s in enumerate(sel_sats_at_t):
        sat = constellation.get_sat_traj(s)
        sat_pt = sat[sat['times'] == time_study]
        color = colors[i % len(colors)]  # Cycle through colors if more than 12 satellites
        ax.scatter(np.radians(sat_pt['azimuths']), 90 - sat_pt['elevations'],
                   color=color, s=50, zorder=5, label=f"Satellite {s}" if i < 10 else "")

    # Plot telescope/instrument position (matching original code exactly)
    if source_traj is not None:
        # Use the observation trajectory instead of source trajectory for telescope position
        instru_pt = source_traj.get_traj()[source_traj.get_traj()['times'] == time_study]
        ax.scatter(np.radians(instru_pt['azimuths']), 90 - instru_pt['elevations'],
                   marker="*", c="black", s=200, label="Telescope Pointing", zorder=6)

    # Set up axis formatting (matching original code exactly)
    ax.set_yticks(range(0, 91, 10))
    ax.set_yticklabels([str(x) for x in range(90, -1, -10)])
    ax.set_theta_zero_location("N")
    ax.set_title(title)

    # Add legend to explain the markers (limit to first 10 satellites to avoid clutter)
    ax.legend()

    return fig, ax


def plot_satellite_trajectories(constellation, title="Satellite Trajectories"):
    """
    Plot satellite trajectories over the observation period.

    Args:
        constellation: Constellation object
        title: Plot title

    Returns:
        tuple: (figure, axes) objects
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, polar=True)

    # Get all satellite names and plot their trajectories
    list_sats = constellation.get_sats_name()
    sel_sats = list_sats[:len(list_sats)]

    # Plot satellite trajectories (matching original code exactly)
    for s in sel_sats:
        sat = constellation.get_sat_traj(s)
        ax.plot(np.radians(sat['azimuths']), 90 - sat['elevations'])

    # Set up axis formatting (matching original code exactly)
    ax.set_yticks(range(0, 91, 10))
    ax.set_yticklabels([str(x) for x in range(90, -1, -10)])
    ax.set_theta_zero_location("N")
    ax.set_title(title)

    return fig, ax


def safe_log10(x):
    """
    Safe logarithm function that handles negative/zero values.

    Args:
        x: Input array

    Returns:
        array: Log10 of x with NaN for non-positive values
    """
    x = np.array(x)
    x = np.where(x > 0, x, np.nan)
    return np.log10(x)
