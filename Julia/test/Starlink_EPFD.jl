
#--- Multi-processing and multi-threading ---#
# run in terminal:

#                                nohup julia Starlink_EPFD.jl > output.log 2>&1 &

# check script with:             tail -f output.log

using Distributed

const N_WORKERS = 2
const N_THREADS_PER_WORKER = 5

addprocs(N_WORKERS; exeflags="--threads=$(N_THREADS_PER_WORKER)")

#----------------------------------------------

@everywhere using DataFrames
@everywhere using Dates 
@everywhere using DelimitedFiles
@everywhere using DimensionalData
@everywhere using DSP
@everywhere using FFTW
@everywhere using JLD2
@everywhere using ProgressMeter
@everywhere using Statistics    
@everywhere using RadioAstro
@everywhere using RadioMdl

using Arrow
ENV["MPLBACKEND"] = "Agg"
using PyPlot
const plt = PyPlot



######################
##### INSTRUMENT #####
######################

## Receiver
#---

# frequency of observation in Hz
cent_freq = 10.69e9

# bandwidth of telescope receiver in Hz
bandwidth = div(31.25e6, 4096)

# frequency resolution
nb_freq_bins = 1

# freq resolution in Hz
freq_res = bandwidth

# gain of amplifier chain
gain_amps = 1.

# telescope receiver temperature in K
T_rx = .0

# freq response
freq_resp = ones(nb_freq_bins)

# create receiver
receiver = Receiver(freq_res, cent_freq, bandwidth, gain_amps, T_rx, freq_resp)


## Antenna
#---

# Size of antenna in m
ant_diameter = 18.3

# aperture efficiency
eta_app = .7

# radiation efficiency of telescope antenna
eta_rad = .99

# valid frequency band of gain pattern model
freq_band = (10e9, 12e9) # in Hz

# gain pattern
max_ant_gain = 10^(67/10) 

# telescope antenna physical temperature
T_phy = .0 # in K

# load telescope antenna#TODO: Check with ITU definition
file_pattern_path = "/home/samthe/antenna_mdl/single_cut_res.cut"
tel_ant = Antenna(file_pattern_path, ant_diameter, eta_app, eta_rad, freq_band, T_phy)

#=
alpha_grid = tel_ant.gain_pat.alpha_grid
beta_grid = tel_ant.gain_pat.beta_grid
smap = tel_ant.gain_pat.spheremap
fig, axs = plt.subplots(subplot_kw=Dict("projection"=>"polar"))
im = axs.pcolormesh(deg2rad.(alpha_grid), beta_grid, 10. .*log10.(smap'); cmap="viridis")
fig.colorbar(im, label="Power (dBW)")
axs.set_theta_zero_location("N")
fig.tight_layout()
=#


## Telescope
#---

# coordinates of telescope
coords = Dict(:lat => 42.6129479883915, :lon => -71.49379366344017, 
              :alt => 86.7689687917009)

# create instrument
westford = Instrument(tel_ant, receiver, coords)



#################################
##### SATELLITES PARAMETERS #####
#################################

## Satellite Antenna
#---

# size of antenna in m
sat_ant_diameter = .8

# aperture efficiency
sat_eta_app = 1.

# radiation efficiency of telescope antenna
sat_eta_rad = 1.

# valid frequency band
sat_freq_band = (10e9, 14e9)

# co-azimuth angles alpha
alphas = [0., 180., 359.]

# polar angles beta
betas = 0.:1.:180.

# create gain dataframe
sat_gain_pat = antenna_mdl_cst(1., alphas, betas)

# satellite antenna physical temperature
sat_T_phy = 0. # in K

# create satellite antenna
sat_ant = Antenna(sat_ant_diameter, sat_gain_pat, sat_eta_app, sat_eta_rad, freq_band, 
                  sat_T_phy)

#=
alpha_grid = sat_ant.gain_pat.alpha_grid
beta_grid = sat_ant.gain_pat.beta_grid
smap = sat_ant.gain_pat.spheremap
fig, axs = plt.subplots(subplot_kw=Dict("projection"=>"polar"))
img = axs.pcolormesh(deg2rad.(alpha_grid), beta_grid, 10. .*log10.(smap'); cmap="viridis")
fig.colorbar(img, label="Power (dBW)")
axs.set_theta_zero_location("N")
fig.tight_layout()
=#


## Satellites Transmitter
#---

# frequency of transmition
sat_freq = cent_freq # in Hz

# satellite transmition bandwidth
sat_bw = bandwidth # in Hz

# number of frequency channels to devide the bandwidth
sat_freq_chan = Int(div(sat_bw, freq_res))

# gain of amplifier chain
sat_gain_amps = 1.

# satellite reciever temperature in K
sat_T_rx = 0. # in K

# freq response
freq_resp = ones(sat_freq_chan)

# create transmitter of satellite
sat_receiver = Receiver(freq_res, sat_freq, sat_bw, sat_gain_amps, sat_T_rx, freq_resp)

#=
figure()
plt.plot(freq_range(receiver), receiver.freq_resp, label="receiver")
plt.plot(freq_range(sat_receiver), sat_receiver.freq_resp, label="satellite")
plt.yscale("log")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Frequency response")
plt.tight_layout()
=#

# create instrument
sat_instrument = Instrument(sat_ant, sat_receiver)


## Satellites Orbit and Parameters
#---

# satellites effective isotropically radiated power distribution
@everywhere starlink_eirp_distro = readdlm("/home/samthe/git/RadioMdl/sat/\
                                            eirp_peak_Starlink.csv")
                                   
#=
figure()
plt.hist(dB_scale.(starlink_eirp_distro), bins=15)
plt.xlabel("peak EIRP [dBW/Hz]")
plt.ylabel("Number of satellites")
plt.tight_layout()
=#

# function to select EIRP randomly from the distribution
@everywhere eirp_peak_func(traj::Trajectory) = rand(starlink_eirp_distro)

# fetch satellites orbit information
name_filters = ["STARLINK"]
avoid_names = ["[DTC]"]
sat_info_path = "/home/samthe/git/RadioMdl/sat/2025-04-01Starlink_active.csv"
sats_catalog = fetch_satellites_info(; info_path=sat_info_path, 
                                     name_filters=name_filters, 
                                     avoid_names=avoid_names, save=false, verb=true)



############################
##### OBSERVATION PLAN #####
############################

## Observation Static Parameters
#---

# day of observation
dateformat = "yyyy-mm-dd\\THH:MM:SS.sss"
start_date = DateTime("2025-04-01T00:00:00.000", dateformat)
stop_date = DateTime("2025-04-01T23:26:00.000", dateformat)

# time resolution in milliseconds
integration_time = 1e3
time_res = Second(1)

# sky model
sky_mdl = nothing


## Satellites Static Parameters
#---

# satellites trajectory resolution
sat_time_res = time_res

# minimum elevation filter
min_elevation_filter = 5.

# compute satellites positions
sats_pos = compute_sats_traj(sats_catalog, start_date, stop_date, 
                             westford.coords, sat_time_res; save=false, 
                             el_min=min_elevation_filter)
    

## Sky Grid for Antenna Positions
#---

# grid polar resolution in degrees
nb_elevation_rings = 30
sky_cells = sky_grid(nb_elevation_rings; min_elevation_filter=min_elevation_filter)
sky_cells_bounds = sky_cells[:, [:pol_min, :pol_max, :caz_min, :caz_max]]

#=
fig = figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=true)
for i in axes(sky_cells, 1)
    pol_min, pol_max = sky_cells[i,:pol_min], sky_cells[i,:pol_max]
    caz_min, caz_max = sky_cells[i,:caz_min], sky_cells[i,:caz_max]
    center_cell = deg2rad((caz_min + caz_max) / 2)
    height_cell = pol_max - pol_min
    width_cell = deg2rad(caz_max - caz_min)
    ax.bar(2π - center_cell; height=height_cell, width=width_cell, bottom=pol_min,
           edgecolor="k", linewidth=0.3, alpha=0.6, align="center")
end
ax.set_theta_zero_location("N")   # 0° azimuth points "up"
ax.set_theta_direction(-1)        # clockwise azimuth (common convention)
ax.set_rlabel_position(135)
ax.set_ylim(0., 90. - min_elevation_filter)
tight_layout()
=#



#######################
##### MONTE-CARLO #####
#######################

## MC Function
#---

# check save directory
mkpath("save_MC_results")

@everywhere function mc_func(sky_cells::DataFrame,
    start_date::DateTime,
    stop_date::DateTime,
    time_res::Dates.Period,
    instru::Instrument,
    sats_pos::DataFrame,
    sat_time_res::Dates.Period,
    sat_instrument::Instrument,
    eirp_peak_func::Function,
    lnk_bdgt::Function,
    sky_mdl::Union{Nothing,SkyMdl},
    n::Int;
    plot::Bool=false,
    save::Bool=false)

        
    ## Create Observation
    #---
    
    # random choice of start of observation
    start_obs = DateTime(Dates.UTM(rand(Dates.value(start_date):Dates.value(stop_date))))
    stop_obs = start_obs + Second(2e3)
    time_samples = range(start_obs, stop_obs; step=time_res)
    
    # antenna position over time
    trajs = fill(SphereCoord(0., 0., 0.), length(time_samples), size(sky_cells,1))
    for i in axes(sky_cells, 1)
        pol_min, pol_max = sky_cells[i,:pol_min], sky_cells[i,:pol_max]
        caz_min, caz_max = sky_cells[i,:caz_min], sky_cells[i,:caz_max]
        point = (rand(caz_min:caz_max), rand(pol_min:pol_max))
        @. trajs[:,i] = SphereCoord(point[1], point[2])
    end
    traj_ant = Trajectory(trajs, collect(time_samples))
    
    # create observation
    observ = Observation(traj_ant, instru, start_obs, stop_obs)
    
    
    ## Satellites trajectories
    #---
    
    # create list of satellites
    sats_list = form_satellites_list(sats_pos, sat_instrument, eirp_peak_func,
                                     start_obs, stop_obs; time_tag=:times,
                                     sat_id_tag=:sat, elevation_tag=:elevations,
                                     azimuth_tag=:azimuths, range_tag=:ranges);
    
    # create constellation
    starlink_constellation = Constellation("Starlink", sats_list, lnk_bdgt)
    
    if plot
        list_sats = get_sats_name(starlink_constellation)
        sel_sats = 1:length(list_sats)
        fig, ax = plt.subplots(subplot_kw=Dict("projection"=>"polar"))
        for s in list_sats[sel_sats]
            sat_coords = get_coords(get_sat_traj(starlink_constellation, s))
            ax.plot(deg2rad.(sat_coords[1]), sat_coords[2])
        end
        ax.set_theta_zero_location("N")
    end
    
    
    ## Simulating EPFD for each cell in sky grid
    #---
    
    # run simulation
    model_observ_psd!(observ, sky_mdl, starlink_constellation)
    
    # mean and store
    mean_epfd_cells = mean(observ.result, dims=:times)[times=1,freqs=1]
    
    save && jldsave("save_MC_results/starlink_epfd_iter_$(n).jld2"; n=n, 
                    mean_epfd_cells=mean_epfd_cells)
    
    return mean_epfd_cells
end


## Run Monte-Carlo
#---

# nb monte-carlo loop
nb_loop = 100 * N_WORKERS

# run workers
mc_results = progress_pmap(1:nb_loop) do n
    mc_func(sky_cells_bounds, start_date, stop_date, time_res, westford, sats_pos, 
            sat_time_res, sat_instrument, eirp_peak_func, epfd_link_budget, sky_mdl, n; 
            save=true)
end

# merge results
for mean_epfd_cells in mc_results
    push!.(sky_cells[:,:epfd], mean_epfd_cells)
end

# Save the merged results to disk
jldsave("save_MC_results/starlink_epfd.jld2"; sky_cells=sky_cells)


## Plotting
#---

# EPFD distribution for each sky cell
max_ant_gain = get_boresight_gain(westford.antenna)[1]
pfd_thresh = -240 - dB_scale(max_ant_gain) # from ITU-R RA.769-2

fig1 = figure()
for e in sky_cells[:,:epfd]
    epfd_db = sort(dB_scale.(e))
    plt.plot(epfd_db .- pfd_thresh, 100 .* collect(1:nb_loop) ./ nb_loop)
end
plt.axhline(y=98., color="black")
plt.axvline(x=0., color="black")
plt.xlabel(L"EPFD/PFD_{thresh} for each sky cell [dB]")
plt.ylabel("Percentile")
plt.tight_layout()
fig1.savefig("save_MC_results/starlink_epfd_distribution.pdf")

# mean EPFD for each sky cell
epfd_db = 10 .* log10.(mean.(sky_cells[:,:epfd]))
vmin, vmax = extrema(epfd_db)

cmap = get_cmap("viridis")
norm = plt.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
colors = [cmap(norm(v)) for v in epfd_db]

centers = deg2rad.(2π .- (sky_cells[:,:caz_min] .+ sky_cells[:,:caz_max]) ./ 2)
widths  = deg2rad.(sky_cells[:,:caz_max] .- sky_cells[:,:caz_min])
heights = sky_cells[:,:pol_max] .- sky_cells[:,:pol_min]
bottoms = sky_cells[:,:pol_min]

fig2 = figure(figsize=(8, 8))
ax = fig2.add_subplot(111, polar=true)
ax.bar(centers, heights; width=widths, bottom=bottoms, color=colors,
       edgecolor=colors, linewidth=0.3, align="center");
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rlabel_position(135)
ax.set_ylim(0., 90.)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig2.colorbar(sm, ax=ax, label="EPFD (dBW)", shrink=0.7, pad=0.1)
fig2.tight_layout()
fig2.savefig("save_MC_results/starlink_epfd_map.pdf")
