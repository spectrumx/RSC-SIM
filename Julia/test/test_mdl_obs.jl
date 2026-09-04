
using Dates
using DimensionalData
using DSP
using FFTW
using PyPlot
const plt = PyPlot
using Statistics

using Revise
using RadioMdl


### INSTRUMENT ###

## Antenna
# Size of antenna in m
ant_diameter = 18.3

# aperture efficiency
eta_app = .7

# radiation efficiency of telescope antenna
eta_rad = .99

# valid frequency band of gain pattern model
freq_band = (10e9, 12e9) # in Hz

# telescope antenna physical temperature
T_phy = 300.0 # in K

# load telescope antenna
file_pattern_path = "supp/single_cut_res.cut"
tel_ant = Antenna(file_pattern_path, ant_diameter, eta_app, eta_rad, freq_band, T_phy)

# alphas = [0., 180., 359.]
# betas = 0.:1.:180.
# # tel_gain_pat = antenna_mdl_ITU_SA_509_3(1.553826454066362e6,
# # 8.938834164371842, alphas, betas)
# tel_gain_pat = antenna_mdl_cst(1.553826454066362e6, alphas, betas)
# tel_ant = Antenna(ant_diameter, tel_gain_pat, eta_app, eta_rad, freq_band, T_phy)

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


## Receiver
# freq resolution in Hz
freq_res = 2e6

# frequency of observation
cent_freq = 10.825e9 # in Hz

# bandwidth of telescope receiver
bw = 1024e6 # in Hz

# gain of amplifier chain
gain_amps = 10^(80/10)

# number of frequency channels to devide the bandwidth
freq_chan = Int(div(bw, freq_res))

# fequency channels
freq_bins = freq_range(freq_res, cent_freq, bw)

# telescope receiver temperature in K
adc_vpp = 1. # in Volts
adc_nb_bits = 16
T_adc = adc_noise_temperature(adc_vpp, adc_nb_bits, bw; instru_imp=50.)
T_LNA = 100. # in K
T_rx = friis_noise_temp((T_LNA, gain_amps), (T_adc, 1.)) # in K

# freq response
responsetype = Lowpass(bw/2-freq_res)
designmethod = FIRWindow(hamming(freq_chan+1)) # avoid 0 tap at beginning
filter_design = digitalfilter(responsetype, designmethod; fs=bw)
filter_design ./= sum(filter_design)
freq_resp = abs.(fftshift(fft(filter_design)))[1:end-1].^2

# create receiver
receiver = Receiver(freq_res, cent_freq, bw, gain_amps, T_rx, freq_resp)

#=
figure()
plt.plot(freq_range(receiver), receiver.freq_resp)
plt.yscale("log")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Frequency response")
plt.tight_layout()
=#


## Telescope
# coordinates of telescope
coords = Dict(:lat => 42.6129479883915, :lon => -71.49379366344017, 
              :alt => 86.7689687917009)

# create instrument
westford = Instrument(tel_ant, receiver, coords)



### OBSERVATION PLAN ###

## Source trajectory over observation window
# observation window
dateformat = "yyyy-mm-dd\\THH:MM:SS.sss"
start_window = "2025-02-18T15:00:00.000"
stop_window = "2025-02-18T15:45:00.000"

# source position over time
# to get the trajectory of the source over Westford, launch the Python script
# 'compute_obj_overflights_full_traj.py'#TODO: implement in Julia
file_traj_obj_path = "supp/traj_files/casA_trajectory_Westford_$(start_window)_\
                      $(stop_window).arrow"
traj_src = Trajectory(file_traj_obj_path; time_tag = :time_stamps, 
                      azimuth_tag = :azimuths, elevation_tag = :altitudes,
                      date_format = dateformat)


## Observation Parameters
# start-end of observation
start_obs = DateTime("2025-02-18T15:30:00.000", dateformat)
stop_obs = DateTime("2025-02-18T15:40:00.000", dateformat)

# offset from source at the beginning of the observation
offset_angles = SphereCoord(20., 0.) # (caz,pol) in degrees

# time of OFF-ON transition
time_off_src = start_obs
time_on_src = time_off_src + Minute(5)

# copy trajectory
traj_obs_1 = copy(traj_src)

# apply offset
offset_angle_trajectory!(traj_obs_1, offset_angles, time_off_src, time_on_src;
                         subtract_angles=true)

# add second trajectory
traj_obs_2 = copy(traj_src)
offset_angle_trajectory!(traj_obs_2, SphereCoord(20., 0.), start_obs, stop_obs)

# add third trajectory
traj_obs_3 = Trajectory(SphereCoord(45., 45.), traj_obs_1.times)
offset_angle_trajectory!(traj_obs_3, offset_angles, time_off_src, time_on_src)


# create the multiple trajectory
traj_obs = Trajectory(hcat(traj_obs_1.traj, traj_obs_2.traj, traj_obs_3.traj), 
                      traj_obs_1.times)

# create observation
observ = Observation(traj_obs, westford, start_obs, stop_obs)

#=
az_src = 360. .- [traj_src.traj[i].alpha for i in 1:size(traj_src.traj,1)]
pol_src = [traj_src.traj[i].beta for i in 1:size(traj_src.traj,1)]
fig, axs = plt.subplots(subplot_kw=Dict("projection"=>"polar"))
axs.plot(deg2rad.(az_src), pol_src, label="source", color="black")
for j in axes(observ.antenna_traj.traj,2)
    az_obs = 360. .- [observ.antenna_traj.traj[i,j].alpha
                      for i in axes(observ.antenna_traj.traj,1)]
    pol_obs = [observ.antenna_traj.traj[i,j].beta 
               for i in axes(observ.antenna_traj.traj,1)]
    axs.plot(deg2rad.(az_obs), pol_obs, label="pointing $j")
end
axs.legend()
axs.set_theta_zero_location("N")
axs.set_ylim(0., 90.)
fig.tight_layout()
=#



### SKY COMPONENTS ###

## Background temperature
# CMB temperature
T_CMB = 2.73 # in K

# galaxy temperature
T_gal = galactic_model(cent_freq) # in K

# background
T_bkg = T_CMB + T_gal


## RFI temperature
# ground temperature in K
T_gnd = ground_model([0.], collect(0.:1.:180.), 250.)

# various RFI
T_var = 0. # in K (no RFI)

# total RFI temperature
T_rfi = T_gnd + T_var


## Atmosphere temperature (and Background through it)
# atmospheric temperature at zenith
T_atm_zenith = 273. # in K

# opacity of atmosphere at zenith
tau = .013

# atmospheric temperature model
T_bkg_atm = atmosphere_model(T_atm_zenith, tau, T_bkg)


## Total Fixed Background Model
T_total_bkg = T_bkg_atm + T_rfi

#=
alpha_sky_grid = collect(0.:1.:360.)
beta_sky_grid = T_total_bkg.beta_grid
smap = T_total_bkg.interp_map(alpha_sky_grid, beta_sky_grid)
az_src = 360. .- [traj_src.traj[i].alpha for i in 1:size(traj_src.traj,1)]
pol_src = [traj_src.traj[i].beta for i in 1:size(traj_src.traj,1)]
fig, axs = plt.subplots(subplot_kw=Dict("projection"=>"polar"))
img = axs.pcolormesh(deg2rad.(alpha_sky_grid), beta_sky_grid, smap'; 
                     cmap="gist_earth", shading ="nearest")
fig.colorbar(img, label="Temperature [K]")
axs.plot(deg2rad.(az_src), pol_src, label="source")
for j in axes(observ.antenna_traj.traj,2)
    az_obs = 360. .- [observ.antenna_traj.traj[i,j].alpha
                      for i in axes(observ.antenna_traj.traj,1)]
    pol_obs = [observ.antenna_traj.traj[i,j].beta 
               for i in axes(observ.antenna_traj.traj,1)]
    axs.plot(deg2rad.(az_obs), pol_obs, label="pointing $j")
end
axs.legend()
axs.set_theta_zero_location("N")
fig.tight_layout()
=#


## Source
# source flux in Jy
flux_src = estim_casA_flux.(freq_bins)

# source temperature through atmosphere
flux_atm_src = reduce(vcat, [atmos_opacity_impact(first.(flux_src), tau, t.beta)' 
                             for t in traj_src.traj])
F_src = PointLikeSrcFlux(TiFreqArray(flux_atm_src, traj_src.times, freq_bins), traj_src)

#=
time_bins = F_src.traj.times
freq_bins = Array(dims(F_src.flux, :freqs))
fig, axs = plt.subplots()
plot_extent = [time_bins[1], time_bins[end], freq_bins[1], freq_bins[end]]
img = axs.imshow(Array(F_src.flux), extent=plot_extent, aspect="auto", 
                 cmap="gist_earth", interpolation="none", origin="lower")
cbar = fig.colorbar(img)
cbar.set_label("Flux [Jy]")
axs.set_xlabel("Time [UTC]")
axs.set_ylabel("Frequency [GHz]")
fig.tight_layout()

time_bins = F_src.traj.times
freq_bins = Array(dims(F_src.flux, :freqs))
fig, axs = plt.subplots()
axs.plot(time_bins, Array(F_src.flux[freqs=Near(cent_freq)]), label="source at $(cent_freq/1e9) GHz")
axs.set_xlabel("Time [UTC]")
axs.set_ylabel("Flux [Jy]")
axs.legend()
fig.tight_layout()
=#


## Sky Model
sky_mdl = SkyMdl(T_total_bkg, F_src)



### SATELLITES CONSTELLATION ###

## Satellite Antenna
# size of antenna in m
sat_ant_diameter = .8

# aperture efficiency
sat_eta_app = 1.

# radiation efficiency of telescope antenna
sat_eta_rad = 1.

# maximum gain of satellite antenna
sat_gain_max = 10^(38/10)

# valid frequency band
sat_freq_band = (10e9, 14e9)

# declination angles alpha
alphas = [0., 180., 359.]

# azimuth angles beta
betas = 0.:1.:180.

# create gain dataframe
sat_gain_pat = antenna_mdl_ITU_S_1528(sat_gain_max, sat_ant_diameter,
                                      freq_to_wave(mean(sat_freq_band)), alphas, 
                                      betas; sat_type="LEO")
sat_gain_pat[:,:gains] ./= maximum(sat_gain_pat[:,:gains])
# sat_gain_pat = antenna_mdl_cst(1., alphas, betas)

# satellite antenna physical temperature
sat_T_phy = 0. # in K

# create satellite antenna
sat_ant = Antenna(sat_ant_diameter, sat_gain_pat, sat_eta_app, sat_eta_rad, freq_band, sat_T_phy)

#=
beta_grid = sat_ant.gain_pat.beta_grid
fig, axs = plt.subplots()
axs.plot(beta_grid, 10 .*log10.(sat_ant.gain_pat.spheremap[1,:]),
         color="tab:blue")
fig.tight_layout()
=#


## Satellites Transmitter
# frequency of transmition
sat_freq = cent_freq # in Hz

# satellite transmition bandwidth
sat_bw = bw # in Hz

# number of frequency channels to devide the bandwidth
sat_freq_chan = Int(div(sat_bw, freq_res))

# gain of amplifier chain
sat_gain_amps = 1.

# satellite reciever temperature in K
sat_T_rx = 0. # in K

# satellite transmission model that depends on frequency
tmt_profile = ones(sat_freq_chan)
tmt_profile[div(sat_freq_chan, 2) .+ 
            (-div(sat_freq_chan, 20):div(sat_freq_chan, 20))] .= 1e-8
tmt_profile[div(sat_freq_chan, 2)] = 1.

# freq response
sat_responsetype = Lowpass(125e6-freq_res)
designmethod = FIRWindow(hamming(sat_freq_chan+1)) # avoid 0 tap at beginning
filter_design = digitalfilter(sat_responsetype, designmethod; fs=sat_bw)
freq_resp = abs.(fftshift(fft(filter_design)))[1:end-1].^2

# sat_responsetype2 = Lowpass((125e6-freq_res) / (sat_bw/3.1))
# design_method2 = Butterworth(3)
# filter_design2 = freqresp(digitalfilter(sat_responsetype2, design_method2),
#                          range(0, sat_bw; length=sat_freq_chan+1) .*
#                          (2π / sat_bw))
# freq_resp = abs.(fftshift(filter_design2))[1:end-1].^2

freq_resp .*= tmt_profile
freq_resp ./= maximum(freq_resp)

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

# satellite effective isotropically radiated power to flux
max_EIRP_density = 10^(-50/10)


## Satellite Instrument
# create instrument
sat_instrument = Instrument(sat_ant, sat_receiver)


## List of Satellites
# observation window
start_sat_window, stop_sat_window = start_obs, stop_obs

# fetch satellites orbit information
name_filters = ["STARLINK"]
avoid_names = ["[DTC]"]
sat_info_path = "supp/traj_files/Starlink_active.csv"
sats_catalog = fetch_satellites_info(; info_path=sat_info_path, name_filters=name_filters, 
                                     avoid_names=avoid_names, save=false, verb=true)

# compute satellites positions
sat_time_res = Second(1)
min_elevation_filter = 5.
sats_pos = compute_sats_traj(sats_catalog, start_sat_window, stop_sat_window, 
                             westford.coords, sat_time_res; save=false, 
                             el_min=min_elevation_filter)

# create list of satellites
sats_list = form_satellites_list(sats_pos, sat_instrument, max_EIRP_density,
                                 start_sat_window, stop_sat_window; rotate_beam=true,
                                 time_tag=:times, sat_id_tag=:sat, 
                                 elevation_tag=:elevations, azimuth_tag=:azimuths, 
                                 range_tag=:ranges);


## Constellation of Satellites
# satellite link budget estimator
lnk_bdgt(args...; kwds...) = classic_gain_link_budget(args...;
                                                      beam_avoid_angle=0.,
                                                      turn_off=false, kwds...)

# create constellation
starlink_constellation = Constellation("Starlink", sats_list, lnk_bdgt)

#=
list_sats = get_sats_name(starlink_constellation)
sel_sats = 1:length(list_sats)
fig, ax = plt.subplots(subplot_kw=Dict("projection"=>"polar"))
for s in list_sats[sel_sats]
    sat_coords = get_coords(get_sat_traj(starlink_constellation, s))
    ax.plot(deg2rad.(360. .- sat_coords[1]), sat_coords[2])
end
ax.set_theta_zero_location("N")
=#



### PSD MODEL DURIMG OBSERVATION ###

## Compute PSD model during observation
model_observ_psd!(observ, sky_mdl, starlink_constellation)

time_bins = observ.antenna_traj.times
freq_bins = freq_range(observ.instrument.receiver)
plot_extent = [time_bins[1], time_bins[end], freq_bins[1], freq_bins[end]]
i=1#for i in axes(observ.antenna_traj.traj, 2)
    fig, axs = plt.subplots(1, 1)#, sharex=true)
    img = axs.imshow(10. .*log10.(freq_res .* Array(observ.result[traj_idx=i,freqs=10:end-10])'), 
                        extent=plot_extent, aspect="auto", cmap="gist_ncar", 
                        interpolation="none", origin="lower")
    cbar = fig.colorbar(img)
    cbar.set_label("PSD [dBW/Hz]")
    axs.set_ylabel("Frequency [GHz]")
# end
axs.set_xlabel("Time [UTC]")
fig.tight_layout()

time_bins = observ.antenna_traj.times
fig, axs = plt.subplots()
for i in axes(observ.antenna_traj.traj, 2)
    axs.plot(time_bins, 10. .*log10.(observ.result[traj_idx=i,freqs=Near(10.9e9)]),
             label="pointing $i")
end
axs.set_xlabel("Time [UTC]")
axs.set_ylabel("PSD [dBW/Hz]")
axs.legend()
fig.tight_layout()

# plot close-by satellites passes
sats_close = RadioMdl.SatPos.sats_close_to_pointing(sats_pos, 
                                                    observ.antenna_traj.times,
                                                    observ.antenna_traj.traj)

row_end_times = DateTime[]
cmap = plt.get_cmap("gist_rainbow")
for i in axes(sats_close, 1)
    sat = sats_close[i,:sat]
    min_pass, max_pass = sats_close[i,:t_start], sats_close[i,:t_stop]
    row = findfirst(t -> t <= min_pass, row_end_times)
    if row === nothing
        push!(row_end_times, max_pass)
        row = length(row_end_times)
    else
        row_end_times[row] = max_pass
    end
    ymin = (row - 1) * 0.02
    ymax = ymin + 0.02
    cur_color = cmap((i - 1) / size(sats_close, 1))
    axs.axvspan(min_pass, max_pass, alpha=.5, ymin=ymin, ymax=ymax, color=cur_color,
                label=sats_close[i,:sat])
end


## Compare with beam avoidance
# Initialize new observation
obs_beam_avoid = Observation(traj_obs, westford, start_obs, stop_obs)
obs_beam_off = Observation(traj_obs, westford, start_obs, stop_obs)

# define new link budget and constellation
ang_det = 10.
lnk_bdgt_beam_avoid(args...; 
                    kwds...) = classic_gain_link_budget(args...; 
                                                        beam_avoid_angle = ang_det,
                                                        turn_off = false, kwds...)
lnk_bdgt_beam_off(args...; 
                  kwds...) = classic_gain_link_budget(args...; 
                                                      beam_avoid_angle = ang_det,
                                                      turn_off = true, kwds...)

starlink_const_beam_avoid = Constellation("Starlink", sats_list, lnk_bdgt_beam_avoid)
starlink_const_beam_off = Constellation("Starlink", sats_list, lnk_bdgt_beam_off)

# Compute PSD with beam avoidance
model_observ_psd!(obs_beam_avoid, sky_mdl, starlink_const_beam_avoid)
model_observ_psd!(obs_beam_off, sky_mdl, starlink_const_beam_off)

time_bins = observ.antenna_traj.times
fig, axs = plt.subplots()
fig.suptitle("Beam avoidance strategies - angle thresh = $(ang_det)°")
for i in axes(observ.antenna_traj.traj, 2)
    axs.plot(time_bins, 10. .*log10.(observ.result[traj_idx=i,freqs=Near(10.9e9)]),
             label="pointing $i", linestyle="dotted", color="C$(i-1)")
    axs.plot(time_bins, 10. .*log10.(obs_beam_avoid.result[traj_idx=i,
                                                           freqs=Near(10.9e9)]),
             label="pointing $i with beam avoidance", linestyle="dashed", 
             color="C$(i-1)")
    axs.plot(time_bins, 10. .*log10.(obs_beam_off.result[traj_idx=i,
                                                         freqs=Near(10.9e9)]),
             label="pointing $i with beam off", color="C$(i-1)")
end
axs.set_xlabel("Time [UTC]")
axs.set_ylabel("PSD [dBW/Hz]")
axs.legend()
fig.tight_layout()


## Compare without satellites
# Initialize new observation
observ_no_sat = deepcopy(observ)
observ_no_sat.result .= 0.

# Compute PSD without satellites
model_observ_psd!(observ_no_sat, sky_mdl)

time_bins = observ_no_sat.antenna_traj.times
fig, axs = plt.subplots()
for i in axes(observ_no_sat.antenna_traj.traj, 2)
    axs.plot(time_bins, 10. .*log10.(observ_no_sat.result[traj_idx=i,
                                                          freqs=Near(10.9e9)]),
             label="pointing $i")
end
axs.set_xlabel("Time [UTC]")
axs.set_ylabel("PSD [dBW/Hz]")
axs.legend()
fig.tight_layout()


## Compare with only satellites
# Initialize new observation
observ_only_sat = deepcopy(observ)
observ_only_sat.result .= 0.

# Compute PSD with only satellites
model_observ_psd!(observ_only_sat, nothing, starlink_constellation)

time_bins = observ_only_sat.antenna_traj.times
fig, axs = plt.subplots()
for i in axes(observ_only_sat.antenna_traj.traj, 2)
    axs.plot(time_bins, 10. .*log10.(observ_only_sat.result[traj_idx=i,
                                                            freqs=Near(10.9e9)]),
             label="pointing $i")
end
axs.set_xlabel("Time [UTC]")
axs.set_ylabel("PSD [dBW/Hz]")
axs.legend()
fig.tight_layout()


## Compare for a specific frequency and trajectory
study_freq = 10.9e9
study_traj = 1

time_bins = observ_only_sat.antenna_traj.times
fig, axs = plt.subplots()
axs.plot(time_bins, 10. .*log10.(observ.result[traj_idx=study_traj,
                                               freqs=Near(study_freq)]),
         label="sky and sats")
axs.plot(time_bins, 10. .*log10.(observ_no_sat.result[traj_idx=study_traj,
                                                      freqs=Near(study_freq)]),
         label="no sats")
axs.plot(time_bins, 10. .*log10.(observ_only_sat.result[traj_idx=study_traj,
                                                        freqs=Near(study_freq)]),
         label="only sats")
axs.set_xlabel("Time [UTC]")
axs.set_ylabel("PSD [dBW/Hz]")
axs.legend()
fig.tight_layout()


## Compare for in and out of protected band for given trajectory
study_freq1 = 10.69e9
study_freq2 = 10.9e9
study_traj = 1

time_bins = observ_only_sat.antenna_traj.times
fig, axs = plt.subplots()
axs.plot(time_bins, 10. .*log10.(observ.result[traj_idx=study_traj,
                                               freqs=Near(study_freq2)]),
         label="downlink band")
axs.plot(time_bins, 10. .*log10.(observ.result[traj_idx=study_traj,
                                               freqs=Near(study_freq1)]),
         label="protected band")
axs.set_xlabel("Time [UTC]")
axs.set_ylabel("PSD [dBW/Hz]")
axs.legend()
fig.tight_layout()


## Compare for different transmitters filter
# define more simple frequency_response
sat_responsetype2 = Lowpass((125e6-freq_res) / (sat_bw/3.1))
design_method2 = Chebyshev2(7,55.)#Butterworth(3)
filter_design2 = freqresp(digitalfilter(sat_responsetype2, design_method2),
                         range(0, sat_bw; length=sat_freq_chan+1) .*
                         (2π / sat_bw))
freq_resp2 = abs.(fftshift(filter_design2))[1:end-1].^2

freq_resp2 .*= tmt_profile
freq_resp2 ./= maximum(freq_resp2)

# create transmitter of satellite
sat_receiver2 = Receiver(freq_res, sat_freq, sat_bw, sat_gain_amps, sat_T_rx, freq_resp2)

figure()
plt.plot(freq_range(receiver), receiver.freq_resp, label="receiver")
plt.plot(freq_range(sat_receiver), sat_receiver.freq_resp, label="satellite FIR")
plt.plot(freq_range(sat_receiver2), sat_receiver2.freq_resp, label="satellite butterworth")
plt.yscale("log")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Frequency response")
plt.tight_layout()

# create instrument
sat_instrument2 = Instrument(sat_ant, sat_receiver2)

# create list of satellites
sats_list2 = form_satellites_list(sats_pos, sat_instrument2, max_EIRP_density,
                                 start_sat_window, stop_sat_window; time_tag=:times,
                                 sat_id_tag=:sat, elevation_tag=:elevations,
                                 azimuth_tag=:azimuths, range_tag=:ranges);

# create constellation
starlink_constellation2 = Constellation("Starlink", sats_list2, lnk_bdgt)

# Initialize new observation
observ_sat2 = deepcopy(observ)
observ_sat2.result .= 0.

# Compute observation with satellite simpler frequency response
model_observ_psd!(observ_sat2, sky_mdl, starlink_constellation2)

study_freq1 = 10.69e9
study_freq2 = 10.9e9
study_traj = 1

time_bins = observ_sat2.antenna_traj.times
fig, axs = plt.subplots()
axs.plot(time_bins, 10. .*log10.(observ_sat2.result[traj_idx=study_traj,
                                               freqs=Near(study_freq2)]),
         label="downlink band")
axs.plot(time_bins, 10. .*log10.(observ_sat2.result[traj_idx=study_traj,
                                                    freqs=Near(study_freq1)]),
         label="protected band")
axs.set_xlabel("Time [UTC]")
axs.set_ylabel("PSD [dBW/Hz]")
axs.legend()
fig.tight_layout()


## Model entire sky

# sky grid
sky_grid_cells = sky_grid(90)
# caz_grid = collect(.5:1.:360.)
# pol_grid = collect(.05:.1:90.)

# transform in trajectory
time_samples = observ.antenna_traj.times
sky_grid_cells.pol = (sky_grid_cells.pol_min .+ sky_grid_cells.pol_max) ./ 2
sky_grid_cells.caz = (sky_grid_cells.caz_min .+ sky_grid_cells.caz_max) ./ 2
traj = [SphereCoord(r.caz, r.pol) for r in eachrow(sky_grid_cells)]
# traj = [SphereCoord(caz, pol) for caz in caz_grid for pol in pol_grid]
ant_map = Trajectory([traj[j] for i in 1:length(time_samples), j in eachindex(traj)], 
                     time_samples)

time_plot = DateTime("2025-02-18T15:34:58.000", dateformat)
observ_sky = Observation(ant_map, westford, time_plot - Second(5), time_plot + Second(5))

model_observ_psd!(observ_sky, nothing, starlink_constellation)

psd_dB = 10 .* log10.(observ_sky.result[times=end,freqs=Near(10.9e9)])
vmin, vmax = extrema(psd_dB)

cmap = get_cmap("plasma")
norm = plt.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
colors = [cmap(norm(v)) for v in psd_dB]

centers = deg2rad.(360. .- sky_grid_cells[:,:caz])
widths  = deg2rad.(sky_grid_cells[:,:caz_max] .- sky_grid_cells[:,:caz_min])
heights = sky_grid_cells[:,:pol_max] .- sky_grid_cells[:,:pol_min]
bottoms = sky_grid_cells[:,:pol_min]

fig = figure(figsize=(16,16))
ax = fig.add_subplot(111, polar=true)
ax.bar(centers, heights; width=widths, bottom=bottoms, color=colors,
       edgecolor=colors, linewidth=0.3, align="center");
ax.set_theta_zero_location("N")
ax.set_theta_direction(1)
ax.set_rlabel_position(135)
ax.set_ylim(0., 90.)
ax.set_yticks(0:10:90, string.(Vector(90:-10:0)))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="EPFD (dBW)", shrink=0.7, pad=0.1)
tight_layout()

