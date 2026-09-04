
"""
"""
function model_observ_psd!(obs::Observation{T},
    sky_mdl::Union{<:AbstractBkg,Nothing},
    constellation::AbstractVector{<:Constellation{T}}) where T
    
    ## Extract useful information from observation
    # get antenna trajectory during observation
    ant_traj = obs.antenna_traj
    # get time samples of observation
    time_samps = ant_traj.times
    # time resolution of observation
    time_res = minimum(time_samps[i+1] - time_samps[i] for i in 1:length(time_samps)-1)
    # get instrument used for observation
    instru = obs.instrument
    # get antenna of instrument
    ant = instru.antenna
    # get reciever of instrument
    rec = instru.receiver
    # get rotation matrices for antenna pointing
    rot_mats = unique_rot_mats(ant_traj)

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
        # work with parent view to avoid DimArray overheads
        res = parent(obs.result)
        # link budget model for constellation
        lnk_bdgt = co.lnk_bdgt_mdl
        # get satellites visible at each time sample
        sat_idx = get_sats_idx_at_times(co, time_samps, time_res)
        sats = co.sats
        # different constellation may have different load hence the :dynamic here
        @threads :dynamic for t in eachindex(time_samps)
            @inbounds for e in entries_at(sat_idx, t)
                # satellite
                sat = sats[e.sat_idx]
                # coordinates of sat at time t_samp in topocentric frame
                sat_coord = get_sat_traj(sat).traj[e.traj_idx,1]
                # satellite instrument
                sat_instru = sat.instrument
                # satellite EIRP_density at time t_samp
                sat_EIRP_den = parent(get_sat_EIRP_density(sat, e.traj_idx)) ./ k_boltz
                # loop over antenna positions
                for c_ind in axes(ant_traj.traj, 2)
                    ant_c = ant_traj.traj[t,c_ind]
                    # link budget
                    l = lnk_bdgt(sat_coord, sat_instru, ant_c, instru; 
                                 pre_load_rot_mat=rot_mats[t,c_ind])
                    # update satellite contribution
                    res[t,:,c_ind] .+= l .* sat_EIRP_den
                end
            end
        end
    end

    # add gain effect
    obs.result .= broadcast_dims(*, obs.result, gain_instru)

    return obs.result
end

function model_observ_psd!(obs::Observation{T},
    sky_mdl::Union{<:AbstractBkg,Nothing},
    constellation::Constellation{T}) where T

    return model_observ_psd!(obs, sky_mdl, [constellation])
end

function model_observ_psd!(obs::Observation{T},
    sky_mdl::Union{<:AbstractBkg,Nothing} = nothing) where T

    return model_observ_psd!(obs, sky_mdl, Constellation{T}[])
end
