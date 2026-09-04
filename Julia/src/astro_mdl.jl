
"""
    freq_to_wave(freq::T) where {T}

Converts frequency to wavelength.

"""
freq_to_wave(freq::T) where {T} = speed_c / freq



"""
    wave_to_freq(wave::T) where {T}

Converts wavelength to frequency.

"""
wave_to_freq(wave::T) where {T} = speed_c / wave



"""
    flux_to_temperature(flux::T,
                        effective_apperture::T) where T

estimates the temperature of a point-like source from its flux and the antenna effective
aperture. flux must be in Jansky

---
    flux_to_temperature(flux::AbstractVector{T},
                        effective_apperture::T) where T

Yields a 'Vector' of temperatures from a vector of fluxes.

"""
function flux_to_temperature(flux::T,
    effective_apperture::T) where T

    return flux*1e-26 / (2*k_boltz) * effective_apperture
end

function flux_to_temperature(flux::AbstractVector{T},
    effective_apperture::T) where T

    return flux_to_temperature.(flux, effective_apperture)
end



"""
in Jansky
"""
function temperature_to_flux(temp::T,
    effective_apperture::T) where T

    return (2*k_boltz) * temp / effective_apperture * 1e26
end



"""
    estim_casA_flux(center_freq::T) where T

estimates the flux of Cas A, given a frequency. Based on Baars et al. 2014

"""
function estim_casA_flux(center_freq::T;
    year::Int = Year(today()).value) where T
    
    log10freq_MHz = log10(center_freq*1e-6)
    log10freq_GHz = log10(center_freq*1e-9)

    if 22e6 < center_freq < 300e6
        a, var_a = 5.625, .021^2
        b, var_b = -.634, .015^2
        c, var_c = -.023, .001^2
    elseif 300e6 < center_freq
        if center_freq > 31e9
            @warn "the model is not valid for frequencies above 31GHz" maxlog=1
        end
        a, var_a = 5.880, .025^2
        b, var_b = -0.792, .007^2
        c, var_c = 0., 0.
    end
    
    # decay
    decay = 0.97 - 0.3*log10freq_GHz # in %/year since 1980
    var_decay = .04^2 + .04^2*log10freq_GHz^2
    
    # log flux
    log_S_Jy = a + b*log10freq_MHz + c*log10freq_MHz^2
    var_log_S_Jy = var_a + var_b*log10freq_MHz^2 + var_c*log10freq_MHz^4

    # constant flux in Jy
    S0 = 10^log_S_Jy
    var_S0 = (log(10) * S0)^2 * var_log_S_Jy

    # decayed flux in Jy
    k = (year - 1980) / 100
    f = 1 - decay * k
    var_f = k^2 * var_decay
    S_Jy = S0 * f
    var_S_Jy = f^2 * var_S0 + S0^2 * var_f

    return S_Jy, var_S_Jy
end



"""
"""
function estim_virgoA_flux(center_freq::T) where T
    return 10^(5.023 - 0.856*log10(center_freq*1e-6))
end



"""
    galactic_model(freq::Real)

Yields the galactic model at frequency 'freq'.

"""
galactic_model(freq::Real) = 1e-1 * (freq / 1.41e9)^(-2.7)



"""
    ground_model(alpha_grid::AbstractVector{T},
                 beta_grid::AbstractVector{T},
                 T_ground::AbstractMatrix{T}) where T

Yields a 'SphereMap' structure representing a ground model with temperature
'T_ground'. This can include ground local sources, terrain, etc.

---
    ground_model(alpha_grid::AbstractVector{T},
                 beta_grid::AbstractVector{T},
                 T_ground::T) where T

Yields a 'SphereMap' structure representing a ground model with constant
temperature 'T_ground'. 

---
    ground_model(T_ground::T) where T

Yields a 'SphereMap' structure representing a ground model with constant
temperature 'T_ground' and default alpha and beta grids sampled at 1 degree
resolution.
"""
function ground_model(alpha_grid::AbstractVector{T},
    beta_grid::AbstractVector{T},
    T_ground::AbstractMatrix{T}) where T

    @assert size(T_ground) == (length(alpha_grid), length(beta_grid))

    return SphereMap(alpha_grid, beta_grid, T_ground)
end

function ground_model(alpha_grid::AbstractVector{T},
    beta_grid::AbstractVector{T},
    T_ground::T) where T

    alpha_grid = alpha_grid[alpha_grid .!= T(360.)]
    
    mat_gnd = T_ground .* ones(T, length(alpha_grid), length(beta_grid))

    bellow_horizon = beta_grid .>= 90.
    mat_gnd .*= bellow_horizon'
    
    return SphereMap(alpha_grid, beta_grid, mat_gnd)
end

function ground_model(T_ground::T) where T

    alpha_grid = [0., 180., 360.]
    beta_grid = [0., 90.,180.]

    return ground_model(alpha_grid, beta_grid, T_ground)
end



"""
    atmos_opacity_impact(temp::T,
                         zenith_opacity::T,
                         zenith_angle::T) where T

Yields the 'temp' temperature altered by the atmosphere opacity at
'zenith_angle' (in degrees).

---
    atmos_opacity_impact(temp::AbstractArray{T},
                         zenith_opacity::T,
                         zenith_angle::T) where T

Yields the 'temp' temperature altered by the atmosphere opacity at
'zenith_angle' for each element of 'temp'. 'temp being a 'AbstractArray', it can
be a 'DimArray' if needed.

---
    atmos_opacity_impact(temp::AbstractArray{T},
                         zenith_opacity::AbstractArray{T},
                         zenith_angle::T) where T

Yields the 'temp' temperature altered by the atmosphere opacity at
'zenith_angle' for each element of 'temp' and associated 'zenith_opacity'.

"""
function atmos_opacity_impact(temp::T,
    zenith_opacity::T,
    zenith_angle::T) where T

    @assert 0 <= zenith_angle < 90 "zenith angle must be at or over 0 and below 90 \
            degrees"

    return temp .* exp(- zenith_opacity / cosd(zenith_angle))
end

function atmos_opacity_impact(temp::AbstractArray{T},
    zenith_opacity::T,
    zenith_angle::T) where T

    return atmos_opacity_impact.(temp, zenith_opacity, zenith_angle)
end

function atmos_opacity_impact(temp::AbstractArray{T},
    zenith_opacity::AbstractArray{T},
    zenith_angle::T) where T

    @assert size(temp) == size(zenith_opacity)
    
    return atmos_opacity_impact.(temp, zenith_opacity, zenith_angle)
end



"""
    atmosphere_model(alpha_grid::AbstractVector{T},
                     beta_grid::AbstractVector{T},
                     T_eff::T,
                     zenith_opacity::T) where T

Yields a 'SphereMap' structure representing an atmosphere model with constant
temperature 'T_eff' and opacity 'zenith_opacity'.

---
    atmosphere_model(T_eff::T,
                     zenith_opacity::T) where T

Yields a 'SphereMap' structure representing an atmosphere model with constant
temperature 'T_eff' and opacity 'zenith_opacity'. The default alpha and beta grids
are sampled at 1 degree resolution.

"""#TODO: use Chapman function for better model at low elevation angles
function atmosphere_model(alpha_grid::AbstractVector{T},
    beta_grid::AbstractVector{T},
    T_eff::T,
    zenith_opacity::T,
    T_bkg::T = zero(T)) where T

    atm_els = zeros(T, length(beta_grid))
    els_horizon = beta_grid .< 90
    atm_els[els_horizon] = atmos_opacity_impact.(T_bkg - T_eff, zenith_opacity, 
                                                 beta_grid[els_horizon]) .+ T_eff
    
    alpha_grid = alpha_grid[alpha_grid .!= T(360.)]
    # beta_grid = beta_grid[beta_grid .!= T(90.)]

    atm_map = reduce(vcat, [atm_els' for _ in 1:length(alpha_grid)])

    return SphereMap(alpha_grid, beta_grid, atm_map)
end

function atmosphere_model(T_eff::T,
    zenith_opacity::T,
    T_bkg::T = zero(T)) where T

    alpha_grid = [0., 180., 360.]
    beta_grid = collect(0.:1.:180.)

    return atmosphere_model(alpha_grid, beta_grid, T_eff, zenith_opacity, T_bkg)
end



"""
    sky_grid(nb_elevation_rings::Int = 30)

Defines a sky grid as detailed in ITU-R S.1586.

"""
function sky_grid(nb_elevation_rings::Int = 30;
    min_elevation_filter::Real = 5.)
    
    # grid polar resolution in degrees
    pol_res = Float64(div(90, nb_elevation_rings))
    
    # pol-caz cells boundaries
    pol_rings = [(pol_res * (p - 1), pol_res * p) for p in 1:nb_elevation_rings]
    pol_rings = pol_rings[last.(pol_rings) .<= (90. - min_elevation_filter)]
    sky_cells = DataFrame(pol_min=Float64[], pol_max=Float64[], caz_min=Float64[],
                          caz_max=Float64[])
    for i in eachindex(pol_rings)
        p_r = pol_rings[i]
        nb_caz_cells_at_pol = Int(div(360, pol_res/cosd(90 - (p_r[1]+pol_res/2))))
        caz_grid = range(0, 360.; length=nb_caz_cells_at_pol+1)
        for a in 1:(length(caz_grid)-1)
            push!(sky_cells, (p_r[1], p_r[2], caz_grid[a], caz_grid[a+1]))
        end
    end

    return sky_cells
end