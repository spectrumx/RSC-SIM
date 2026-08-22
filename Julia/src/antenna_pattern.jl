
"""
    integration_weights(alpha_grid::::AbstractVector{T},
                        beta_grid::AbstractVector{T}) where T

Yeilds a 2D-matrix of solid angle weights for the integration over a sphere. The
weights are defined by 'alpha_grid' and 'beta_grid'.

This function only works for uniform grids.

"""
function integration_weights(alpha_grid::AbstractVector{T},
    beta_grid::AbstractVector{T};
    alpha_periodic::Union{Bool,Nothing} = nothing) where T
    
    @assert issorted(alpha_grid) "alpha_grid must be sorted"
    @assert issorted(beta_grid) "beta_grid must be sorted"

    # convert to radians
    alpha_rads = deg2rad.(alpha_grid)
    beta_rads = deg2rad.(beta_grid)
    
    # Detect closure of azimuthal grid (changes integration weights at the edges
    # as one of the points has a looped neighbor in one case)
    alpha_closed = isnothing(alpha_periodic) ? !(isapprox(first(alpha_grid), 0; 
                                                          atol=1e-9) &&
                                                 isapprox(last(alpha_grid),  360; 
                                                          atol=1e-9)) : alpha_periodic
    
    # dimension weights
    w_alpha = trapz_weights(alpha_rads; periodic = alpha_closed, period = T(2π))
    w_beta  = trapz_weights(beta_rads; periodic = false)   # β never periodic

    return w_alpha * w_beta'
end



"""
    trapz_weights(x; 
                  periodic = false, 
                  period = 2π)

Trapezoid integration weights for a (regular or non-regular) 1D grid `x`. For a
periodic dimension, endpoints receive full weight via wrap-around. 

"""
function trapz_weights(x::AbstractVector{T};
    periodic::Bool = false, 
    period::T = T(2π)) where T

    n = length(x)
    w = similar(x)
    @inbounds for i in eachindex(x)
        lo = i == 1 ? (periodic ? x[1]  - (x[end] - period) : zero(T)) : x[i] - x[i-1]
        hi = i == n ? (periodic ? (x[1] + period) - x[end]  : zero(T)) : x[i+1] - x[i]
        w[i] = (lo + hi) / 2
    end

    return w
end



"""
    sin_beta_weights(x::AbstractVector{T}) where T

Computes the sin(β) weights for a 1D grid `x` in degrees.

"""
function sin_beta_weights(x::AbstractVector{T}) where T

    beta = deg2rad.(x)
    n = length(beta)
    w = zeros(T, n)
    @inbounds for j in 1:n-1
        a, b, h = beta[j], beta[j+1], beta[j+1] - beta[j]
        w[j] += (sin(a) - sin(b) + h * cos(a)) / h
        w[j+1] += (sin(b) - sin(a) + h * cos(b)) / h
    end

    return w
end



"""
    integrate_spheremap(S::SphereMap;
                        beta_window = (0, 180),
                        alpha_window = (0, 360),
                        normalize = true)

Integrates `S.spheremap` over the angular window defined by `beta_window` (in
degrees) and `alpha_window` (in degrees), weighted by sin(β). If `normalize`,
divides by 4π so the result is a sphere fraction; otherwise returns the raw
solid-angle-weighted integral. 

"""#TODO: update to handle non-uniform grids and put in `coord_frames.jl`
function integrate_spheremap(S::SphereMap{T};
    beta_window::Tuple{<:Real,<:Real}  = (0, 180),
    alpha_window::Tuple{<:Real,<:Real} = (0, 360),
    normalize::Bool = true) where T
    
    alpha, beta = S.alpha_grid, S.beta_grid
    w   = integration_weights(alpha, beta)

    alpha_mask = (alpha .>= alpha_window[1]) .& (alpha .<= alpha_window[2])
    beta_mask = (beta .>= beta_window[1])  .& (beta .<= beta_window[2])
    mask   = alpha_mask .* beta_mask'

    integrand = S.spheremap .* sind.(beta') .* w .* mask
    I = sum(integrand)

    return normalize ? I / (4π) : I
end



"""
    radiated_power_to_gain!(rad_pow::AbstractDataFrame,
                           eta_rad::Real = 1.0;

# using DimensionalData                           alpha_col::Symbol = :caz,
                           beta_col::Symbol = :polar,
                           map_col::Symbol = :power) where T

Yields the gain pattern of an antenna, given a radiated power pattern.

"""
function radiated_power_to_gain!(rad_pow::AbstractDataFrame,
    eta_rad::Real = 1.0;
    alpha_col::Symbol = :caz,
    beta_col::Symbol = :polar,
    map_col::Symbol = :power)

    @assert 0. <= eta_rad <= 1.

    # map the radiated power for interpolation
    rad_pow_map, a, b = map_sphere_coords(rad_pow; alpha_col=alpha_col, 
                                          beta_col=beta_col, map_col=map_col)

    # check grids are covering full sphere (as normalization would not be the same)
    solid_angle = integrate_spheremap(SphereMap(a, b, ones(eltype(rad_pow_map), 
                                                length(a), length(b)));
                                      normalize = false)
    @assert isapprox(solid_angle, 4π; rtol = 1e-1) "Grid does not integrate to 4π — \
            full-sphere coverage required."

    # integrate over the sphere
    rad_pow_avg = sum(rad_pow_map .* #=sin_beta_weights=#sind.(b)' .* 
                      integration_weights(a, b)) / (4π)

    # directivity
    rad_pow[:,map_col] ./= rad_pow_avg

    # gain
    rad_pow[:,map_col] .*= eta_rad
    map_col == :gains ? nothing : rename!(rad_pow, map_col => :gains)

    return rad_pow
end



"""
    gain_to_effective_aperture(gain::Real,
                               wavelength::Real)

Yields the effective aperture of an antenna given its gain and wavelength.

"""
function gain_to_effective_aperture(gain::Real,
    wavelength::Real)
    
    return gain * (wavelength^2/(4π))
end



"""
    estim_hpbw(G::Real; 
               K::Real = 31_000)

Returns the half-power beamwidth (in degrees) of an antenna given its peak gain
(not in dB). The constant K is related to the antenna type and efficiency.

`K` presets: 41253 (lossless bound), 33000 (low-loss array), 31000 (typical),
27000 (lossy).

---
    estim_hpbw(D::Real,
               lambda::Real;
               k::Real = 67.6)

Returns the half-power beamwidth (in degrees) of an antenna given its diameter
and wavelength. `k` presets: 50.8 (uniform line/array cut), 58.4 (uniform
circular), 67.6 (~-10 dB taper), 70 (tapered dish). 

!!! warning
    `k` and aperture efficiency η are coupled: K = η·π²·k². Do not mix k=67.6
    with η=1.
    
"""
function estim_hpbw(G::Real; 
    K::Real = 31_000)
    
    G_dBi = 10 * log10(G)
    0 < K ≤ 41253 || throw(DomainError(K, "K must be in (0, 41253]  (4π sr in deg²)"))
    0 ≤ G_dBi ≤ 90 || throw(DomainError(G_dBi, "peak gain outside 0–90 dBi"))
    θ = sqrt(K / exp10(G_dBi / 10))
    θ > 20 && @warn "HPBW = $(round(θ; digits=1))°: pencil-beam approximation is poor \
                     below ~20 dBi"
   
    return θ
end

estim_hpbw(D::Real, lambda::Real; k::Real = 67.6) = k * lambda / D



"""
    effective_aperture_to_gain(effective_aperture::Real,
                               wavelength::Real)

Yields the gain of an antenna given its effective aperture and wavelength.

"""
function get_geometric_effective_aperture(aperture_efficiency::T,
    diameter::T) where T

    @assert T(0) <= aperture_efficiency <= T(1)

    return aperture_efficiency * pi * (diameter / 2)^2
end



"""
    antenna_mdl_ITU_SA_509_3(gain_max::T,
                             half_beamwidth::T,
                             caz::AbstractVector{T},
                             pol::AbstractVector{T};
                             single_rfi::Bool = false) where T

Create ITU recommended gain profile according to ITU-R SA.509-3 "Space research
earth station and radio astronomy reference antenna radiation pattern for use in
interference calculations, including coordination procedures, for frequencies
less than 30 GHz". 

---
    antenna_mdl_ITU_SA_509_3(caz::AbstractVector{T},
                             pol::AbstractVector{T},
                             aperture_eff::T,
                             diameter::T,
                             wavelength::T;
                             kwds...) where T

Create ITU recommended gain profile according to ITU-R SA.509-3, without knowing
the gain_max and half_beamwidth parameters.

"""
function antenna_mdl_ITU_SA_509_3(gain_max::T,
    half_beamwidth::T,
    caz::AbstractVector{T},
    pol::AbstractVector{T};
    single_rfi::Bool = false) where T
    
    @assert gain_max > zero(T) "gain_max must be positive"

    # gain profile container
    gain_profile = zeros(length(pol))

    # select different parts of the gain profile
    gain_max_dB = 10. *log10(gain_max)
    parts = [0, half_beamwidth*sqrt(17/3), 10^((49-gain_max_dB)/25), 48, 80, 120, 180]
    part1 = findall(i -> parts[1] <= i < parts[2], pol)
    part2 = findall(i -> parts[2] <= i < parts[3], pol)
    part3 = findall(i -> parts[3] <= i < parts[4], pol)
    part4 = findall(i -> parts[4] <= i < parts[5], pol)
    part5 = findall(i -> parts[5] <= i < parts[6], pol)
    part6 = findall(i -> parts[6] <= i <= parts[7], pol)

    # calculate gain profile
    gain_profile[part1] .= gain_max_dB .- 3*(pol[part1]./half_beamwidth).^2
    gain_profile[part2] .= gain_max_dB - (single_rfi ? 17 : 20)
    gain_profile[part3] .= (single_rfi ? 32 : 29) .- 25 .*log10.(pol[part3])
    gain_profile[part4] .= (single_rfi ? -10 : -13)
    gain_profile[part5] .= (single_rfi ? -5 : -8)
    gain_profile[part6] .= (single_rfi ? -10 : -13)
    
    # create gain dataframe
    gain_pat = DataFrame(polar=zeros(length(pol)*length(caz)), 
                         caz=zeros(length(pol)*length(caz)), 
                         gains=zeros(length(pol)*length(caz)))
    for b in eachindex(caz)
        gain_pat[((b-1)*length(pol)+1):b*length(pol), :polar] .= pol
        gain_pat[((b-1)*length(pol)+1):b*length(pol), :caz] .= caz[b]
        gain_pat[((b-1)*length(pol)+1):b*length(pol), :gains] .= 10 .^(gain_profile./10)
    end

    return gain_pat
end

function antenna_mdl_ITU_SA_509_3(caz::AbstractVector{T},
    pol::AbstractVector{T},
    aperture_eff::T,
    diameter::T,
    wavelength::T;
    kwds...) where T

    gain_max = aperture_eff * (π * diameter / wavelength)^2
    half_beamwidth = 20 * sqrt(3) * wavelength / diameter

    return antenna_mdl_ITU_SA_509_3(gain_max, half_beamwidth, caz, pol; kwds...)
end



"""
    antenna_mdl_ITU_RA_1631(gain_max::T,
                            ant_diameter::T,
                            wavelength::T,
                            caz::AbstractVector{T},
                            pol::AbstractVector{T}) where T

Create ITU recommended gain profile according to ITU-R RA.1631-1 "Reference
radio astronomy antenna pattern to be used for compatibility analyses between
non-GSO systems and radio astronomy service stations based on the epfd concept".

"""
function antenna_mdl_ITU_RA_1631(gain_max::T,
    ant_diameter::T,
    wavelength::T,
    caz::AbstractVector{T},
    pol::AbstractVector{T}) where {T}
    
    @assert gain_max > zero(T) "gain_max must be positive"

    # gain profile container
    gain_profile = zeros(length(pol))

    # select different parts of the gain profile
    parts = [0., 69.88/(ant_diameter / wavelength), 1., 10., 34.1, 80., 120., 180.]
    part1 = findall(i -> parts[1] < i < parts[2], pol)
    part2 = findall(i -> parts[2] <= i < parts[3], pol)
    part3 = findall(i -> parts[3] <= i < parts[4], pol)
    part4 = findall(i -> parts[4] <= i < parts[5], pol)
    part5 = findall(i -> parts[5] <= i < parts[6], pol)
    part6 = findall(i -> parts[6] <= i <= parts[7], pol)
    part7 = findall(i -> parts[7] <= i <= parts[8], pol)

    # calculate gain profile
    x1 = π * ant_diameter / (360. * wavelength) .* pol[part1]
    x2 = π * ant_diameter / (360. * wavelength) .* pol[part2]
    B = 10^3.2 * π^2 * ((π * ant_diameter / 2) / (180. * wavelength))^2
    gain_profile[1] = gain_max
    gain_profile[part1] .= gain_max .* (besselj1.(2π .* x1) ./ (π .* x1)).^2
    gain_profile[part2] .= B .* (cos.(2π .* x2 .- 3π/4 .+ .0953) ./ (π .* x2)).^2
    gain_profile[part3] .= 10. .^((29. .- 25. .* log10.(pol[part3])) ./ 10.)
    gain_profile[part4] .= 10. .^((34. .- 30. .* log10.(pol[part4])) ./ 10.)
    gain_profile[part5] .= 10. .^(-12. / 10.)
    gain_profile[part6] .= 10. .^(-7. / 10.)
    gain_profile[part7] .= 10. .^(-12. / 10.)

    # create gain dataframe
    gain_pat = DataFrame(polar=zeros(length(pol)*length(caz)), 
                         caz=zeros(length(pol)*length(caz)), 
                         gains=zeros(length(pol)*length(caz)))
    for b in eachindex(caz)
        gain_pat[((b-1)*length(pol)+1):b*length(pol), :polar] .= pol
        gain_pat[((b-1)*length(pol)+1):b*length(pol), :caz] .= caz[b]
        gain_pat[((b-1)*length(pol)+1):b*length(pol), :gains] .= gain_profile
    end
    
    return gain_pat
end



"""
    antenna_mdl_ITU_S_1528()

Create ITU recommended gain profile according to ITU-R S.1528-2 "Satellite
antenna radiation patterns for non-geostationary orbit satellite antennas
operating in the fixed-satellite service below 30 GHz".

"""
function antenna_mdl_ITU_S_1528(gain_max::T,
    ant_diameter::T,
    wavelength::T,
    caz::AbstractVector{T},
    pol::AbstractVector{T};
    sat_type::AbstractString = "LEO") where T

    @assert gain_max > zero(T) "gain_max must be positive"
    @assert ant_diameter > zero(T) "ant_diameter must be positive"
    @assert wavelength > zero(T) "wavelength must be positive"

    hpbw = estim_hpbw(gain_max)
    psi_b = hpbw / 2
    @assert psi_b > pol[1] "hpbw/2 must be larger than the first polar angle"
    lamb_D = ant_diameter / wavelength
    @assert lamb_D < 35. "the ratio diameter-wavelength cannot exceed 35"

    # gain profile container
    gain_profile = zeros(length(pol))

    # select different parts of the gain profile
    if sat_type == "LEO"
        L_f = 5.
        L_s = -6.75
        Y = 1.5 * psi_b
    elseif sat_type == "MEO"
        L_f = 3.
        L_s = -12.
        Y = 2 * psi_b
    else
        @warn "sat_type must be either 'LEO', 'MEO' or defaulting to reference pattern \
               detailed in 1.3 of ITU-R" maxlog=1
        L_f = 0.
        L_s = -25.
        Y = psi_b * (-L_s / 3)^(1/2)
    end
    Z = Y * 10^(.04 * (gain_max + L_s - L_f))
    parts = [0., psi_b, Y, Z, 180.]
    part1 = findall(i -> parts[1] <= i < parts[2], pol)
    part2 = findall(i -> parts[2] <= i < parts[3], pol)
    part3 = findall(i -> parts[3] <= i < parts[4], pol)
    part4 = findall(i -> parts[4] <= i < parts[5], pol)

    # calculate gain profile
    gain_max_dB = 10. *log10(gain_max)
    gain_profile[part1] .= gain_max_dB .- 3 .* (pol[part1] ./ psi_b).^1.5
    gain_profile[part2] .= gain_max_dB .- 3 .* (pol[part2] ./ psi_b).^2
    if sat_type == "LEO"
        gain_profile[part3] .= (20. * log10(lamb_D) + 5.65) .- 
                               25. .* log10.(pol[part3] ./ psi_b)
    elseif sat_type == "MEO"
        gain_profile[part3] .= (20. * log10(lamb_D) + 3.5) .- 
                               25. .* log10.(pol[part3] ./ psi_b)
    else
        gain_profile[part3] .= gain_max_dB .+ L_s .- 25. .* log10.(pol[part3] ./ Y)
    end
    gain_profile[part4] .= L_f

    # create gain dataframe
    gain_pat = DataFrame(polar=zeros(length(pol)*length(caz)), 
                         caz=zeros(length(pol)*length(caz)), 
                         gains=zeros(length(pol)*length(caz)))
    for b in eachindex(caz)
        gain_pat[((b-1)*length(pol)+1):b*length(pol), :polar] .= pol
        gain_pat[((b-1)*length(pol)+1):b*length(pol), :caz] .= caz[b]
        gain_pat[((b-1)*length(pol)+1):b*length(pol), :gains] .= 10 .^(gain_profile./10)
    end
    
    return gain_pat
end



"""
    antenna_mdl_cst(gain::T,
                     caz::AbstractVector{T},
                     pol::AbstractVector{T}) where T

Create constant gain pattern for omni-directional antennas.

"""
function antenna_mdl_cst(gain::T,
    caz::AbstractVector{T},
    pol::AbstractVector{T}) where T

    @assert gain > zero(T) "gain must be positive"

    gain_pat = DataFrame(caz=zeros(length(caz)*length(pol)), 
                         polar=zeros(length(caz)*length(pol)), 
                         gains=zeros(length(caz)*length(pol)))
    for b in eachindex(pol)
        gain_pat[((b-1)*length(caz)+1):b*length(caz), :caz] .= caz
        gain_pat[((b-1)*length(caz)+1):b*length(caz), :polar] .= pol[b]
        gain_pat[((b-1)*length(caz)+1):b*length(caz), :gains] .= gain
    end

    return gain_pat
end