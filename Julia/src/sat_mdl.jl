
#TODO: see Chapt 9 S.Paine AM package for some refraction accounting


"""
    free_space_loss(rng::T,
                    freq::T) where T

Yields the free space loss for a given range and frequency.

"""
function free_space_loss(rng::T,
    freq::T) where T

    return ( 4*pi*rng / freq_to_wave(freq) )^2
end



"""
    simple_link_budget(gain_RX::T,
                       gain_TX::T,
                       rng::T,
                       freq::T,) where T

Yields the link budget coefficient between a receiver and transmitter according
to the Friis formula:

    gain_RX * gain_TX * (wavelength / (4*pi*rng))^2

"""
function simple_link_budget(gain_RX::T,
    gain_TX::T,
    rng::T,
    freq::T) where T

    L = free_space_loss(rng, freq)

    return gain_RX * gain_TX / L
end



"""
    classic_gain_link_budget(sat_coord::SphereCoord{T},
                             sat_instru::Instrument{T},
                             tel_pointing_coord::SphereCoord{T},
                             tel_instru::Instrument{T};
                             pre_load_rot_mat::Union{AbstractMatrix,Nothing} = nothing,
                             beam_avoid_angle::T = 0.0,
                             turn_off::Bool = false) where T

Yields the link budget coefficient between a satellite and a telescope given
their coordinates and instruments. The link budget is computed using the Friis
formula, accounting for the gains of the satellite and telescope antennas, the
distance between them and the frequency of observation.

It is possible to give a pre-computed rotation matrix to transform from
topocentric frame to telescope's antenna frame.

The satellite coordinates are transformed from the topocentric frame to the
telescope's antenna frame by passing via an intermediate Earth-Centered
Earth-Fixed (ECEF) frame. 

If 'beam_avoid_angle' is greater than 0, the function accounts for beam
avoidance strategies performed by the satellite. If `turn_off` is `true`, if the
satellite is passing close to the telescope pointing direction, its gain is set
to zero as it turns off. If it is `false`, if the satellite boresight is closer
than 'beam_avoid_angle' to the telescope pointing direction, the satellite gain
is reduced by "steering" away the satellite boresight of 45 degrees.

"""
function classic_gain_link_budget(sat_coord::SphereCoord{T},
    sat_instru::Instrument{T,Us,As},
    tel_pointing_coord::SphereCoord{T},
    tel_instru::Instrument{T,U,A};
    pre_load_rot_mat::Union{AbstractMatrix,Nothing} = nothing,
    beam_avoid_angle::T = 0.0,
    turn_off::Bool = false) where {T<:AbstractFloat,Us<:Union{T,AbstractVector{T}}, 
                                   As<:Antenna{T},U<:Union{T,AbstractVector{T}},
                                   A<:Antenna{T}}

    # frequency bins of receiver
    freq_bins = freq_range(sat_instru.receiver)

    # coordinate of sat in telescope frame
    sat_coord_in_tel = pass_frame_to_frame(sat_coord, tel_pointing_coord;
                                           pre_load_rot_mat=pre_load_rot_mat)

    # telescope antenna
    tel_antenna = tel_instru.antenna

    # telescope gain
    gain_tel = get_gain_value(tel_antenna, sat_coord_in_tel)
    
    isnothing(tel_instru.coords) && error("tel_instru.coords (Dict with :lat, :lon, \
                                           :alt) is required when simple_approx is \
                                           false.")
    (tel_coord_in_sat, R_ned_s, 
     R_nwz_t) = tel_dir_in_sat_frame(sat_coord, tel_instru.coords[:lat],
                                     tel_instru.coords[:lon], 
                                     tel_instru.coords[:alt])
    
    # beam avoidance effect
    if beam_avoid_angle > zero(T)
        if turn_off
            ang_sep = angular_separation(sat_coord, tel_pointing_coord)
            if ang_sep < beam_avoid_angle
                return zeros(T, length(freq_bins))
            end
        else
            # get boresight pointing of satellite antenna
            sat_beam_coord = get_boresight_gain_coord(sat_instru.antenna)
            
            # boresight (antenna frame, X=North,Y=East,Z=Nadir) → ECEF → topo
            v_ned  = spher_to_cart_coord(sat_beam_coord.alpha, sat_beam_coord.beta, 
                                         one(T))
            v_ecef = R_ned_s * v_ned
            v_t  = -R_nwz_t' * v_ecef
            alpha_b, beta_b = cart_to_sphe_coord(v_t[1], v_t[2], v_t[3])[1:2]
            sat_beam_coord_topo = SphereCoord(mod(-alpha_b, T(360)), beta_b, one(T))
            
            # angular distance between sat boresight and telescope pointing are
            # closer than beam_avoid_angle
            ang_sep = angular_separation(sat_beam_coord_topo, tel_pointing_coord)
            if ang_sep < beam_avoid_angle
                tel_coord_in_sat = SphereCoord(sat_beam_coord.alpha,
                                               mod(sat_beam_coord.beta + T(45), T(180)),
                                               sat_coord.r)
            end
        end
    end

    # satellite gain
    gain_sat = get_gain_value(sat_instru.antenna, tel_coord_in_sat)

    #link budget
    # return [simple_link_budget(gain_tel, gain_sat, sat_coord.r, f) for f in
    # freq_bins]
    return (gain_tel * gain_sat / (4π * sat_coord.r)^2) .* freq_to_wave.(freq_bins).^2
end



"""
    epfd_link_budget(sat_coord::SphereCoord{T},
                     sat_instru::Instrument{T},
                     tel_pointing_coord::SphereCoord{T},
                     tel_instru::Instrument{T};
                     pre_load_rot_mat::Union{AbstractMatrix,Nothing} = nothing) where T

Compute the EIRP link budget for a satellite and a telescope, that is:

    gain_telescope(in satellite direction) / (4π * satellite_range^2)

"""
function epfd_link_budget(sat_coord::SphereCoord{T},
    sat_instru::Instrument{T,Us,As},
    tel_pointing_coord::SphereCoord{T},
    tel_instru::Instrument{T,U,A};
    pre_load_rot_mat::Union{AbstractMatrix,Nothing} = nothing) where {T<:AbstractFloat,
                                                        Us<:Union{T,AbstractVector{T}},
                                                                      As<:Antenna{T},
                                                        U<:Union{T,AbstractVector{T}},
                                                        A<:Antenna{T}}

    # coordinate of sat in telescope frame
    sat_coord_in_tel = pass_frame_to_frame(sat_coord, tel_pointing_coord;
                                           pre_load_rot_mat=pre_load_rot_mat)

    # telescope antenna
    tel_antenna = tel_instru.antenna

    # telescope gain
    gain_tel = get_gain_value(tel_antenna, sat_coord_in_tel) / 
               get_boresight_gain(tel_instru.antenna)
    
    return gain_tel / (4π * sat_coord.r^2)
end

