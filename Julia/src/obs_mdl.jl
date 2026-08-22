
"""
"""
function model_observ_psd!(obs::Observation{T},
    sky_mdl::Union{AbstractBkg,Nothing} = nothing,
    constellation::Vector{Constellation{T}} = Constellation{T}[]) where T
    
    ## Extract useful information from observation
    # get antenna trajectory during observation
    ant_traj = obs.antenna_traj
    # get time samples of observation
    time_samps = ant_traj.times
    # time resolution of observation
    time_res = minimum(diff(time_samps))
    # get instrument used for observation
    instru = obs.instrument
    # get antenna of instrument
    ant = instru.antenna
    # get reciever of instrument
    rec = instru.receiver
    # get rotation matrices for antenna pointing
    rot_mats = rot_mat.(ant_traj.traj)

    ## Compute instrument temperature components
    # gain of instrument for temperature
    gain_instru = get_psd_gain_coeff(instru)
    # instrument noise temperature
    T_n_instru = rec.T_rx .+ get_antenna_radiation_loss(ant)
    # add to result
    if typeof(T_n_instru) <: DimArray
        obs.result .= broadcast_dims(+, obs.result, T_n_instru)
    else
        obs.result .+= T_n_instru
    end

    ## Compute background antenna temperature
    if !isnothing(sky_mdl)
        # get antenna temperature for antenna trajectories
        obs.result .= broadcast_dims(+, obs.result,
                                     get_antenna_temperature(ant, sky_mdl, ant_traj; 
                                                             pre_load_rot_mat=rot_mats))
    end

    ## Compute constellations contribution#TODO: put in sat_mdl.jl?
    for co in constellation
        # link budget model for constellation
        lnk_bdgt = co.lnk_bdgt_mdl
        # different constellation may have different load hence the :dynamic here
        @threads :dynamic for t in eachindex(time_samps)
            # get pointing positions of antenna
            t_samp = time_samps[t]
            ant_coords = ant_traj.traj[t,:]
            # list of sats visible at time t_samp
            sats_names_at_t = get_sats_names_at_time(co, t_samp; time_res=time_res)
            for s_name in sats_names_at_t
                # satellite
                sat = get_sat(co, s_name)
                # coordinates of sat at time t_samp in topocentric frame
                sat_coord = get_sat_traj(sat)(t_samp-time_res, t_samp+time_res)[1]#(t_samp)[1]#FIXME:
                # satellite instrument
                sat_instru = sat.instrument
                # satellite EIRP_density at time t_samp
                sat_EIRP_den = get_sat_EIRP_density(sat, t_samp; time_res=time_res) ./ 
                               k_boltz
                # loop over antenna positions
                for c_ind in eachindex(ant_coords)
                    ant_c = ant_coords[c_ind]
                    # link budget
                    l = lnk_bdgt(sat_coord, sat_instru, ant_c, instru; 
                                 pre_load_rot_mat=rot_mats[t,c_ind])
                    
                    # update satellite contribution
                    obs.result[times=t,traj_idx=c_ind] .+= l .* sat_EIRP_den
                end
            end
        end
    end

    # add gain effect
    obs.result .= broadcast_dims(*, obs.result, gain_instru)

    return obs.result
end

function model_observ_psd!(obs::Observation{T},
    sky_mdl::Union{AbstractBkg,Nothing},
    constellation::Constellation{T}) where T

    return model_observ_psd!(obs, sky_mdl, [constellation])
end
