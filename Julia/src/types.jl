
"""
    is_TiFreqArray(d::DimArray; strict::Bool = false)

Yields true if 'd' is a 'DimArray' with dimensions :times and/or :freqs. If 'strict'
is set to true, the array must have both dimensions.

"""
function is_TiFreqArray(d::DimArray;
    strict::Bool = false)
    
    if strict
        @assert all(d -> name(d) in (:times, :freqs), dims(d)) "Array must have \
                :times and :freqs dimensions"
    else
        @assert any(d -> name(d) in (:times, :freqs), dims(d)) "Array must have :times \
            and/or :freqs dimensions"
    end
end



"""
    RegularSampled(r::AbstractRange)

Yields a 'Sampled' object with a regular step.
"""
RegularSampled(r::AbstractRange) = DimensionalData.Sampled(r, 
                                                        DimensionalData.ForwardOrdered(),
                                                        DimensionalData.Regular(step(r)), 
                                                           DimensionalData.Points(),
                                                           DimensionalData.NoMetadata())



"""
    TiFreqArray(data::AbstractArray{T},
                times::AbstractVector{DateTime}) where T

Yields a 'DimArray' with dimension :times. If times is an AbstractRange, it will
convert it in a regular sampled grid (see ['RegularSampled'](@ref)). 

---
    TiFreqArray(data::AbstractArray{T},
                freqs::AbstractVector{<:Real}) where T

Yields a 'DimArray' with dimension :freqs. If freqs is an AbstractRange, it will
convert it in a regular sampled grid (see ['RegularSampled'](@ref)). 

---
    TiFreqArray(data::AbstractArray{T},
                times::AbstractVector{DateTime},
                freqs::AbstractVector{<:Real}) where T

Yields a 'DimArray' with dimensions containing :times and :freqs. If times
and/or freqs is an AbstractRange, it will convert it in a regular sampled grid
(see ['RegularSampled'](@ref)). 

"""
function TiFreqArray(data::AbstractArray{T},
    times::AbstractVector{DateTime}) where T

    if typeof(times) <: AbstractRange
        times = RegularSampled(times)
    end
    return DimArray(data, (Dim{:times}(times)))
end

function TiFreqArray(data::AbstractArray{T},
    freqs::AbstractVector{<:Real}) where T

    if typeof(freqs) <: AbstractRange
        freqs = RegularSampled(freqs)
    end
    return DimArray(data, (Dim{:freqs}(freqs)))
end

function TiFreqArray(data::AbstractArray{T},
    times::AbstractVector{DateTime},
    freqs::AbstractVector{<:Real}) where T

    if typeof(times) <: AbstractRange
        times = RegularSampled(times)
    end
    if typeof(freqs) <: AbstractRange
        freqs = RegularSampled(freqs)
    end
    return DimArray(data, (Dim{:times}(times), Dim{:freqs}(freqs)))
end



"""
    Trajectory(traj::AbstractArray{SphereCoord{T}},
               times::Vector{DateTime}) where T<:AbstractFloat

Yields a 'Trajectory' structure. The trajectory is defined by a AbstractArray of
['SphereCoord'](@ref) (angles in degrees) and a vector of DateTime. This allow
the use of multiple coordinates per time samples, e.g. in case of sky mapping.

'get_time_bounds' can be used to get the first and last times of the trajectory.

From a 'Trajectory' structure 'Tr', it is possible to get the trajectory over two
dates 't0' and 't1' using 'get_traj(Tr, t0, [t1 = t0])'.

It is also possible to offset the trajectory using
['offset_angle_trajectory!'](@ref). 

---
    Trajectory(traj::AbstractDataFrame;
               time_tag::Symbol = :times,
               azimuth_tag::Symbol = :azimuths,
               elevation_tag::Symbol = :elevations)

Yields a 'Trajectory' structure. The columns of the DataFrame 'traj'  contain
the columns of symbols defined by 'time_tag', 'azimuth_tag' and 'elevation_tag'.
It is possible to have rows of same time and different azimuths and/or
elevations (e.g. for sky mapping). Angles are in degrees. It is assumed the
dataframe defines angles as azimuths and elevations. The method converts them in
co-azimuth and polar angles, as defined in ['SphereCoord'](@ref).

---
    Trajectory(file_path::String;
               kwds...)

Yields a 'Trajectory' structure from the file located at 'file_path'. Only '.arrow'
and '.csv' files are supported for now.

---
    Trajectory(coord::SphereCoord, 
               times::Vector{DateTime})

Yields a 'Trajectory' structure from a single 'coord' and 'times'.

---
    Trajectory(T::Trajectory,
               start_date::DateTime,
               stop_date::DateTime)

Yields a 'Trajectory' structure from 'T' between 'start_date' and 'stop_date'.

"""
struct Trajectory{T<:AbstractFloat}
    traj::AbstractArray{SphereCoord{T}}
    times::Vector{DateTime}
    
    function Trajectory(traj::AbstractArray{SphereCoord{T}},
        times::Vector{DateTime}) where {T<:AbstractFloat}

        @assert size(traj,1) == length(times) " first dimension of 'traj' must have \
                same length than 'times'."
        @assert size(traj,3) == 1 " 'traj' must be at most a 2D matrix."
        
        return new{T}(traj, times)
    end
end

function Trajectory(traj::AbstractDataFrame;
    time_tag::Symbol = :times,
    azimuth_tag::Symbol = :azimuths,
    elevation_tag::Symbol = :elevations,
    range_tag::Union{Symbol,Nothing} = nothing,
    date_format::String = "yyyy-mm-dd\\THH:MM:SS.sss")
    
    @info "'traj' must have coordinates defined in azimuth and elevation angles. The \
           angles are converted in co-azimuth and polar angles as defined in \
           ['SphereCoord'](@ref)." maxlog=1

    # check columns
    col_names = propertynames(traj)
    for tag in [time_tag, azimuth_tag, elevation_tag]
        @assert tag in col_names "`traj` must contain the column defined by `$tag`"
    end
    !isnothing(range_tag) && @assert range_tag in col_names "`traj` must contain the \
                                     column defined by `$range_tag`"
    @assert all(length.(traj[!,azimuth_tag]) == length.(traj[!,elevation_tag]) .== 1) "\
            the elements of columns of `traj` must be scalars. Use new row with same \
            time stamp and different angles for sky mapping"
    if !(eltype(traj[!,time_tag]) <: DateTime)
        traj[!,time_tag] = DateTime.(traj[:,time_tag], date_format)
    end

    # Group by time
    grouped = groupby(traj, time_tag)
    
    # Get sorted unique times and max coords
    uniq_times = sort!(unique(traj[!,time_tag]))
    max_coords = maximum(nrow, grouped)
    arr_traj = fill(SphereCoord(NaN, NaN), length(uniq_times), max_coords)
    for (t_idx, t) in enumerate(uniq_times)
        group = grouped[(t,)]
        n_coords = nrow(group)
        arr_traj[t_idx,1:n_coords] = SphereCoord.(-group[!,azimuth_tag],
                                                  90 .- group[!,elevation_tag],
                                                  isnothing(range_tag) ? 1. : 
                                                                      group[!,range_tag])
    end

    return Trajectory(arr_traj, uniq_times)
end

function Trajectory(file_path::String;
    kwds...)

    # load the trajectory as a DataFrame
    if occursin(".arrow", file_path)
        traj = DataFrame(Arrow.Table(file_path))
    elseif occursin(".csv", file_path)
        traj = DataFrame(CSV.File(file_path))
    elseif occursin(".dat", file_path)
        traj = read_VGOS_antenna_traj(file_path)
    else
        error("the trajectory points are not in Arrow or CSV format")
    end

    return Trajectory(traj; kwds...)
end

function Trajectory(coord::SphereCoord, times::Vector{DateTime})

    return Trajectory(fill(coord, length(times), 1), times)
end

function Trajectory(T::Trajectory,
    t0::DateTime,
    t1::DateTime = t0)
    
    ind_sub_times = t0 .<= T.times .<= t1
    
    return Trajectory(T.traj[ind_sub_times,:], T.times[ind_sub_times])
end

Base.copy(T::Trajectory) = Trajectory(copy(T.traj), copy(T.times))

Base.show(io::IO, Tr::Trajectory{T}) where {T<:AbstractFloat} = begin
    print(io, "Trajectory{$T}:\n")
    print(io, "number of time samples: $(length(Tr.times))\n")
    print(io, "time bounds: $(get_time_bounds(Tr))\n")
    print(io, "number of trajectory points per time sample: $(size(Tr.traj, 2))")
end

get_time_bounds(T::Trajectory) = (T.times[1], T.times[end])

(T::Trajectory)(t0::DateTime, t1::DateTime) = T.traj[t0 .<= T.times .< t1,:]

(T::Trajectory)(t0::DateTime) = T.traj[t0 .== T.times,:]

get_coords(T::Trajectory) = [T.traj[i].alpha for i in eachindex(T.times)], 
                            [T.traj[i].beta for i in eachindex(T.times)], 
                            [T.traj[i].r for i in eachindex(T.times)]



"""
    get_unique_coords(T::Trajectory)

Returns the unique coordinates in the trajectory and their row (time) and column
(trajectory point) indices in the trajectory array.

"""
function get_unique_coords(T::Trajectory)

    # get unique coordinates and their indices in trajectory
    unique_coords = unique(T.traj)
    unique_coords_id = [findall(x -> x == c, T.traj) for c in unique_coords]
    if eltype(unique_coords_id[1]) == CartesianIndex{2}
        time_ids = [[a_c[1] for a_c in u_id] for u_id in unique_coords_id]
        traj_ids = [[a_c[2] for a_c in u_id] for u_id in unique_coords_id]
    else
        time_ids = [[a_c for a_c in u_id] for u_id in unique_coords_id]
        traj_ids = [[1 for a_c in u_id] for u_id in unique_coords_id]
    end

    return unique_coords, time_ids, traj_ids
end



"""
    offset_angle_trajectory!(Tr::Trajectory,
                             offset::SphereCoord,
                             t0::DateTime = Tr.times[1],
                             t1::DateTime = Tr.times[end])

Offsets the trajectory 'Tr' by the angle 'offset' between 't0' and 't1'.

"""
function offset_angle_trajectory!(Tr::Trajectory,
    offset::SphereCoord,
    t0::DateTime = Tr.times[1],
    t1::DateTime = Tr.times[end];
    kwds...)
    
    traj_off_ind = findall(t0 .<= Tr.times .<= t1)
    isempty(traj_off_ind) && @warn "No trajectory points found between $t0 and $t1."
    
    for i in traj_off_ind
        for j in eachindex(Tr.traj[i,:])
            Tr.traj[i,j] = add_coords(Tr.traj[i,j], offset; kwds...)
        end
    end
end



"""
    Antenna(ant_diameter::T,
            gain_pat::SphereMap{T},
            ap_eff::T,
            rad_eff::T,
            valid_freqs::Tuple{<:Real,<:Real},
            T_phy::Union{T,DimArray{T}}) where T<:AbstractFloat
        
Yields an 'Antenna' structure that defines the antenna properties, that is its
diameter 'ant_diameter', gain pattern 'gain_pat' as a 'SphereMap' ( see
['SphereMap'](@ref) ), aperture efficiency 'ap_eff', radiation efficiency
'rad_eff', physical temperature 'T_phy' as a constant or a 'DimArray'
and valid frequency range 'valid_freqs'.

If 'T_phy' is a 'DimArray', its dimensions must be named 'times' and/or 'freqs'.

When applying a '[SphereMap'](@ref), 'S', to an 'Antenna', 'A',
'''
A(S)
'''
the map of 'S' is interpolated to the sampling grids of 'A.gain_pat' and
multiplied by the 'gain_pat' map of 'A'. The result is then an array sampled
over the same grids as 'A.gain_pat'.

It is possible to use different get functions:
'''
# returns the gain at (alpha, beta)
get_gain_value(A::Antenna, alpha::Real, beta::Real)
c = SphereCoord(alpha, beta)
get_gain_value(A::Antenna, spherecoord::SphereCoord)

# returns the directivity value at (alpha, beta)
get_directivity_value(A::Antenna, alpha::Real, beta::Real) 
get_directivity_value(A::Antenna, spherecoord::SphereCoord)

# returns the angle girds where the gain pattern is sampled
get_angle_grids(A::Antenna)

# returns the boresight gain
get_boresight_gain(A::Antenna)

# return the half power beamwidths
estim_hpbw(A::Antenna, wavelength::Real)

# return the geometric effective aperture
get_geometric_effective_aperture(A::Antenna)

# return the antenna radiation loss
get_antenna_radiation_loss(A::Antenna)

# return the converted flux into temperature via the geometric effective aperture
flux_to_temperature(flux::T, A::Antenna)
'''

---
    Antenna(ant_diameter::T, 
            gain_pat::AbstractDataFrame,
            ap_eff::T,
            rad_eff::T,
            valid_freqs::Tuple{<:Real,<:Real},
            T_phy::Union{T,DimArray{T}}) where T<:AbstractFloat

Yields an 'Antenna' structure based on the dataframe 'gain_pat'. The columns of
the dataframe 'gain_pat' must contain the columns `:caz`, `:polar` and `:gains`.

---
    Antenna(cut_file_path::String,
            ant_diameter::T,
            ap_eff::T,
            rad_eff::T,
            valid_freqs::Tuple{<:Real,<:Real},
            T_phy::Union{T,DimArray{T}}) where T<:AbstractFloat

Yields an 'Antenna' structure based on the cut file located at 'cut_file_path',
using the ['power_pattern_from_cut_file'](@ref) function.

---
    
"""
struct Antenna{T<:AbstractFloat}
    ant_diameter::T # antenna diameters in m
    gain_pat::SphereMap{T} # gain pattern as a spherical map
    ap_eff::T # aperture efficiency
    rad_eff::T # radiation efficiency
    valid_freqs::Tuple{<:Real,<:Real} # min and max valid frequencies for the gain model
    T_phy::Union{T,DimArray{T}} # physical temperature in K

    function Antenna(ant_diameter::T, 
        gain_pat::SphereMap{T},
        ap_eff::T,
        rad_eff::T,
        valid_freqs::Tuple{<:Real,<:Real},
        T_phy::Union{T,DimArray{T}}) where {T<:AbstractFloat}

        @assert ant_diameter > 0 "antenna diameter must be positive"
        @assert 0 <= rad_eff <= 1 "radiation efficiency must be between 0 and 1"
        @assert 0 <= ap_eff <= 1 "aperture efficiency must be between 0 and 1"
        @assert valid_freqs[1] < valid_freqs[2] "the valid frequency range must be \
                non-empty"
        @assert all(T_phy .>= 0) "physical temperature must be non-negative"
        if typeof(T_phy) <: DimArray
            is_TiFreqArray(T_phy)
            if :freqs in name.(dims(T_phy))
                @assert valid_freqs[1] .<= T_phy[:freqs] .<= valid_freqs[2] "physical \
                        temperature must be defined for the valid frequency range"
            end
        end

        return new{T}(ant_diameter, gain_pat, ap_eff, rad_eff, valid_freqs, T_phy)
    end
end

function Antenna(ant_diameter::T, 
    gain_pat::AbstractDataFrame,
    ap_eff::T,
    rad_eff::T,
    valid_freqs::Tuple{<:Real,<:Real},
    T_phy::Union{T,DimArray{T}}) where T

    # create the sphere map 
    @assert :caz in propertynames(gain_pat) "the gain pattern must have a column \
                                                `:caz`"
    @assert :polar in propertynames(gain_pat) "the gain pattern must have a column \
                                               `:polar`"
    @assert :gains in propertynames(gain_pat) "the gain pattern must have a column \
                                               `:gains`"
    SM = SphereMap(gain_pat; map_col=:gains)
    
    return Antenna(ant_diameter, SM, ap_eff, rad_eff, valid_freqs, T_phy)
end

function Antenna(cut_file_path::String,
    ant_diameter::T,
    ap_eff::T,
    rad_eff::T,
    valid_freqs::Tuple{<:Real,<:Real},
    T_phy::Union{T,DimArray{T}}) where T

    # load the antenna power pattern
    pattern = power_pattern_from_cut_file(cut_file_path)

    # convert radiated power to gain
    radiated_power_to_gain!(pattern, rad_eff)

    return Antenna(ant_diameter, pattern, ap_eff, rad_eff, valid_freqs, T_phy)
end

Base.show(io::IO, A::Antenna{T}) where {T} = begin 
    print(io, "Antenna{$T}:\n")
    print(io, "diameter: $(A.ant_diameter)\n")
    print(io, "gain pattern: $(A.gain_pat)\n")
    print(io, "aperture efficiency: $(A.ap_eff)\n")
    print(io, "radiation efficiency: $(A.rad_eff)\n")
    if typeof(A.T_phy) <: DimArray
        print(io, "physical temperature: $(name.(dims(A.T_phy))) of size \
                   $(size(A.T_phy))\n")
    else
        print(io, "physical temperature: $(A.T_phy)\n")
    end
    print(io, "valid frequency range: $(A.valid_freqs)\n")
end

get_gain_value(A::Antenna, alpha::Real, beta::Real) = A.gain_pat(alpha, beta)

get_gain_value(A::Antenna, spherecoord::SphereCoord) = get_gain_value(A,
                                                                      spherecoord.alpha,
                                                                      spherecoord.beta)

function (ant::Antenna)(S::SphereMap{T},
    point_coord::SphereCoord;
    pre_load_rot_mat::Union{Matrix,Nothing} = nothing) where T
    
    # define the sampling grid as the one of the gain pattern
    samp_alpha_grid = ant.gain_pat.alpha_grid
    samp_beta_grid = ant.gain_pat.beta_grid

    # pass SphereMap to antenna coordinate frame and resample to antenna sampled
    # grids
    map_in_ant = pass_frame_to_frame(S, point_coord, samp_alpha_grid, samp_beta_grid;
                                   pre_load_rot_mat=pre_load_rot_mat)
    # calculate the resulting map
    result_map = ant.gain_pat.spheremap .* map_in_ant

    return SphereMap(samp_alpha_grid, samp_beta_grid, result_map)
end

function get_directivity_value(A::Antenna,
    alpha::Real,
    beta::Real)
    
    return get_gain_value(A, alpha, beta) / A.rad_eff
end

function get_directivity_value(A::Antenna,
    spherecoord::SphereCoord)

    return get_directivity_value(A, spherecoord.alpha, spherecoord.beta)
end

get_angle_grids(A::Antenna) = get_angle_grids(A.gain_pat)

function get_boresight_gain(A::Antenna)

    gain = A.gain_pat
    i = findmax(gain.spheremap)[2]

    return gain.spheremap[i], gain.alpha_grid[i[1]], gain.beta_grid[i[2]]
end

estim_hpbw(A::Antenna, wavelength::Real) = estim_hpbw(A.ant_diameter, wavelength)

function get_geometric_effective_aperture(A::Antenna)
    return get_geometric_effective_aperture(A.ap_eff, A.ant_diameter)
end

get_beam_solid_angle(A::Antenna) = 4π * A.rad_eff / get_boresight_gain(A)[1]

get_antenna_radiation_loss(A::Antenna) = (1 - A.rad_eff) .* A.T_phy

function flux_to_temperature(flux::T,
    A::Antenna) where T

    A_eff = get_geometric_effective_aperture(A)

    return flux_to_temperature(flux, A_eff)
end

function temperature_to_flux(temp::T,
    A::Antenna) where T

    A_eff = get_geometric_effective_aperture(A)

    return temperature_to_flux(temp, A_eff)
end



"""
    gain_to_effective_aperture(A::Antenna,
                               wavelength::Real)

Yields the effective aperture of an antenna given a wavelength.

"""
function gain_to_effective_aperture(A::Antenna,
    wavelength::Real,
    c::SphereCoord)
    
    return A.gain_pat(c) * (wavelength^2/(4π))
end



"""
#FIXME: update doc
    get_antenna_temperature(A::Antenna{T}, 
                            T_b::SphereMap{T}) where T

Return the antenna temperature for a 'SphereMap' temperature 'T_b' (extended
sources). See ['SphereMap'](@ref).

---
    get_antenna_temperature(A::Antenna{T},
                            T_b::T) where T

Returns the antenna temperature for a scalar temperature 'T_b' (point-like
sources in boresight).

"""
function get_antenna_temperature(A::Antenna{T}, 
    T_b::SphereMap{T},
    point_coord::SphereCoord;
    pre_load_rot_mat::Union{Matrix,Nothing} = nothing) where T

    # define sampling grids
    alpha_grid = A.gain_pat.alpha_grid
    beta_grid = A.gain_pat.beta_grid
    
    # compute integration weights
    weights = integration_weights(alpha_grid, beta_grid)

    # apply SphereMap to Antenna
    g_T_b = A(T_b, point_coord; pre_load_rot_mat=pre_load_rot_mat)

    # return trapz((deg2rad.(alpha_grid), deg2rad.(beta_grid)), 
    #              g_T_b.spheremap .* #=sin_beta_weights=#sind.(beta_grid)') / (4*pi)
    return sum(g_T_b.spheremap .* #=sin_beta_weights=#sind.(beta_grid)' .* weights) / (4*pi)
end

function get_antenna_temperature(A::Antenna{T}, 
    T_b::T) where T

    return T_b .* A.rad_eff
end

function get_antenna_temperature(A::Antenna{T},
    T_b::DimArray{T}) where T

    is_TiFreqArray(T_b)

    return T_b .* A.rad_eff
end
#FIXME: inconsistent with  get_antenna_temperature(A::Antenna{T}, 
    # T_b::SphereMap{T},
    # point_coord::SphereCoord)
#check with constant sphere (SphereMap([0.], [0., 90., 180.], 300 .* ones(1,3)))
function get_antenna_temperature(A::Antenna{T},
    T_b::SphereMap{T},
    ant_traj::Trajectory;
    pre_load_rot_mat::Union{<:AbstractArray{Matrix{T}},Nothing} = nothing) where T

    # define sampling grids
    alpha_grid = A.gain_pat.alpha_grid
    beta_grid = A.gain_pat.beta_grid
    
    # solid angle
    weights = integration_weights(alpha_grid, beta_grid)

    # scale gain by integral factor
    scaled_gain = (A.gain_pat.spheremap .* #=sin_beta_weights=#sind.(beta_grid)') ./ (4π) .* weights
    
    # nb of coords per time stamps
    nb_coords = size(ant_traj.traj,2)

    # find unique coordinates to reduce computation
    unique_coords, time_unique_ids, traj_unique_ids = get_unique_coords(ant_traj)

    if isnothing(pre_load_rot_mat)
        pre_load_rot_mat = rot_mat.(ant_traj.traj)#TODO: maybe reduce the nb of rot_mat
    else
        @assert size(pre_load_rot_mat) == size(ant_traj.traj) "pre_load_rot_mat \
                must have same size than ant_traj.traj."
    end

    T_As = DimArray(zeros(T, length(ant_traj.times), nb_coords),
                    (Dim{:times}(ant_traj.times),Dim{:traj_idx}(1:nb_coords)))
    @threads :dynamic for c in eachindex(unique_coords)
        # coord and indices of coord in ant_traj
        ant_coord = unique_coords[c]
        t_c = time_unique_ids[c] # time indices where ant_traj is at ant_coord
        tr_c = traj_unique_ids[c] # traj indices where ant_traj is at ant_coord

        # compute spheremap in antenna frame
        T_b_in_ant = pass_frame_to_frame(T_b, ant_coord, alpha_grid, beta_grid;
                                       pre_load_rot_mat=pre_load_rot_mat[t_c[1],tr_c[1]])

        v = dot(scaled_gain, T_b_in_ant)
        @inbounds for k in eachindex(t_c)
            T_As[times=t_c[k],traj_idx=tr_c[k]] = v
        end
    end

    return T_As
end



"""
    get_antenna_temperature(A::Antenna{T},
        T_b::DimArray{SphereMap{T}},
        ant_traj::Trajectory;
        pre_load_rot_mat::Union{<:AbstractArray{Matrix{T}},Nothing} = nothing) where T

Get the antenna temperature in the antenna frame for a DimArray of SphereMaps.
The antenna trajectory must begin after or at the same time as the first SphereMap
in T_b, as a model needs to be defined for each antenna position.

"""
function get_antenna_temperature(A::Antenna{T},
    T_b::DimArray{SphereMap{T}},
    ant_traj::Trajectory;
    pre_load_rot_mat::Union{<:AbstractArray{Matrix{T}},Nothing} = nothing) where T

    is_TiFreqArray(T_b; strict=true)

    # define sampling grids
    alpha_grid = A.gain_pat.alpha_grid
    beta_grid = A.gain_pat.beta_grid
    
    # solid angle
    weights = integration_weights(alpha_grid, beta_grid)

    # scale gain by integral factor
    scaled_gain = (A.gain_pat.spheremap .* #=sin_beta_weights=#sind.(beta_grid)') ./ (4π) .* weights
    
    # nb of coords per time stamps
    nb_coords = size(ant_traj.traj,2)

    # find unique coordinates to reduce computation
    unique_coords, time_unique_ids, traj_unique_ids = get_unique_coords(ant_traj)

    if isnothing(pre_load_rot_mat)
        pre_load_rot_mat = rot_mat.(ant_traj.traj)
    else
        @assert size(pre_load_rot_mat) == size(ant_traj.traj) "pre_load_rot_mat \
                must have same size than ant_traj.traj."
    end

    T_As = DimArray(zeros(T, length(ant_traj.times), length(dims(T_b, :freqs)), 
                          nb_coords),
                    (Dim{:times}(ant_traj.times), dims(T_b, :freqs), 
                     Dim{:traj_idx}(1:nb_coords)))
    for c in eachindex(unique_coords)
        # coord and indices of coord in ant_traj
        ant_coord = unique_coords[c]
        t_c = time_unique_ids[c] # time indices where ant_traj is at ant_coord
        tr_c = traj_unique_ids[c] # traj indices where ant_traj is at ant_coord

        # latest SphereMap models for current ant_coord
        T_b_time_ids = [searchsortedlast(lookup(T_b, :times), t) 
                        for t in ant_traj.times[t_c]]
        # calcul is done on each unique antenna position accounting for T_b
        # model that may change over the times in ant_traj.times[t_c]
        for t_id in unique(T_b_time_ids)
            # if there is no T_b model for the current antenna time, the antenna
            # temperature is left at zero
            if t_id == 0
                continue
            end
            # antenna times where the antenna position is ant_coord and the T_b
            # model is of :times index t_id
            ant_times = t_c[T_b_time_ids .== t_id]
            ant_tr = tr_c[T_b_time_ids .== t_id]

            # select SphereMap from latest time before current antenna time
            T_b_t = T_b[times=t_id,freqs=1]

            # convert the map coordinates (same for all frequencies) of spheremap
            # to antenna frame using precomputed rot_mat for ant_coord
            # new_map_coords = pass_frame_to_frame(T_b_t, ant_coord, alpha_grid, 
            #                                      beta_grid; grid_only=true,
            #                            pre_load_rot_mat=pre_load_rot_mat[t_c[1],tr_c[1]])
            
            @threads for f in axes(T_b, :freqs)    
                # select SphereMap from latest time before current antenna time
                T_b_t_f = T_b[times=t_id,freqs=f]

                # compute spheremap in antenna frame using precomputed rot_mat
                # for ant_coord
                T_b_in_ant = pass_frame_to_frame(T_b_t_f, ant_coord, alpha_grid, 
                                                 beta_grid;
                                       pre_load_rot_mat=pre_load_rot_mat[t_c[1],tr_c[1]])
                # compute spheremap with new rotated map coordinates
                # T_b_in_ant = Matrix{T}(undef, size(new_map_coords))
                # @inbounds for j in axes(new_map_coords,2)
                #     @simd for i in axes(new_map_coords,1)
                #         T_b_in_ant[i,j] = T_b_t_f(new_map_coords[i,j][1],
                #                                                   new_map_coords[i,j][2])
                #     end
                # end
                # T_b_in_ant = T_b[times=t_id,freqs=f](new_map_coords...)#FIXME: this is costly

                # combine antenna gain
                # wks[threadid()] = scaled_gain .* T_b_in_ant.spheremap
    
                # integrate over sphere
                # T_As[times=t_c,traj_idx=tr_c] .= trapz((deg2rad.(alpha_grid), 
                #                                         deg2rad.(beta_grid)), 
                #                                         wks[threadid()])
                v = dot(scaled_gain, T_b_in_ant)
                @inbounds for k in eachindex(ant_times)
                    T_As[times=ant_times[k],freqs=f,traj_idx=ant_tr[k]] = v
                end
            end
        end
    end

    return T_As
end



"""
    Receiver(freq_res::T,
             cent_freq::T,
             bw::T,
             gain_amps::T,
             T_rx::Union{T,DimArray{T}},
             freq_resp::DimArray{T}) where T

Yields a 'Receiver' structure that defines a receiver composed of a frequency
resolution 'freq_res', a center frequency 'cent_freq', a bandwidth 'bw', an
amplifier gain 'gain_amps', a receiver temperature 'T_rx' and a frequency
response 'freq_resp'. 

'T_rx' can be a scalar or a 'DimArray{T}'. In the later case, 'T_rx' must have
the 'times' and/or the 'freqs' dimension.

'freq_resp' must be normalized such that √(1/N * ∑_i∈[1,N] freq_resp[i]^2) ≈ 1,
thus normalizing the receiver frequency response energy.

It is possible to use the 'get_nb_freq_chan' function to get the number of
frequency channels. 'freq_range' can be used with a 'Receiver' as argument to
get the frequency range vector of the receiver.

---
    Receiver(freq_res::T,
             cent_freq::T,
             bw::T,
             gain_amps::T,
             T_rx::Union{T,DimArray{T}},
             freq_resp::AbstractVector{T}) where T

Yields a 'Receiver' structure taking a vector as argument for 'freq_resp'.

---
    Receiver(freq_res::T,
             cent_freq::T,
             bw::T,
             gain_amps::T,
             T_rx::Union{T,DimArray{T}}) where T

Yields a 'Receiver' structure with a flat frequency response.

"""#TODO: rename in Transceiver
struct Receiver{T<:AbstractFloat}
    freq_res::T # frequency resolution
    cent_freq::T # center frequency
    bw::T # bandwidth
    gain_amps::T # gain of amplifiers
    T_rx::Union{T,DimArray{T}} # receiver temperature
    freq_resp::DimArray{T} # frequency response

    function Receiver(freq_res::T, 
        cent_freq::T,
        bw::T,
        gain_amps::T,
        T_rx::Union{T,DimArray{T}},
        freq_resp::DimArray{T}) where T<:AbstractFloat
        
        @assert freq_res > 0
        @assert cent_freq > 0
        @assert bw > 0
        @assert gain_amps > 0
        @assert all(T_rx .>= 0)
        @assert :freqs in name.(dims(freq_resp))
        if typeof(T_rx) <: DimArray
            is_TiFreqArray(T_rx)
            if :freqs in name.(dims(T_rx))
                @assert size(T_rx,:freqs) == size(freq_resp, :freqs)
            end
        end
        # this holds for wideband pass filters only
        # @assert all(sqrt.(sum(freq_resp.^2, dims=:freqs) ./
        #         length(freq_resp)) .- 1 .<= 1e-10) "the receiver frequency response \
        #         must be normalized"
        
        return new{T}(freq_res, cent_freq, bw, gain_amps, T_rx, freq_resp)
    end
end

function Receiver(freq_res::T,
    cent_freq::T,
    bw::T,
    gain_amps::T,
    T_rx::Union{T,DimArray{T}},
    freq_resp::AbstractVector{T}) where {T<:AbstractFloat}
    
    freq_grid = freq_range(freq_res, cent_freq, bw)
    freq_resp_array = TiFreqArray(freq_resp, collect(freq_grid))

    return Receiver(freq_res, cent_freq, bw, gain_amps, T_rx, freq_resp_array)
end

function Receiver(freq_res::T,
    cent_freq::T,
    bw::T,
    gain_amps::T, 
    T_rx::Union{T,DimArray{T}}) where {T<:AbstractFloat}

    # flat frequency response
    nb_freq_chan = Int(div(bw, freq_res))
    freq_resp = T.(ones(nb_freq_chan))

    return Receiver(freq_res, cent_freq, bw, gain_amps, T_rx, freq_resp)
end

Base.show(io::IO, R::Receiver{T}) where {T} = begin
    print(io, "Receiver{$T}:\n")
    print(io, "frequency resolution: $(R.freq_res)\n")
    print(io, "center frequency: $(R.cent_freq)\n")
    print(io, "bandwidth: $(R.bw)\n")
    if typeof(R.T_rx) <: DimArray
        print(io, "receiver temperature: $(name.(dims(R.T_rx))) of size \
                   $(size(R.T_rx))\n")
    else
        print(io, "receiver temperature: $(R.T_rx)\n")
    end
    print(io, "gain amplifiers: $(R.gain_amps)\n")
end

get_nb_freq_chan(R::Receiver) = Int(div(R.bw, R.freq_res))

freq_range(R::Receiver) = freq_range(R.freq_res, R.cent_freq, R.bw)

function get_psd_gain_coeff(R::Receiver)
    return (R.gain_amps * k_boltz) .* R.freq_resp
end



"""
    Instrument(antenna::Antenna{T},
               receiver::Receiver{T},
               coords::Dict{Symbol,Union{T,AbstractVector{T}}} = 
                       Dict(:lat=>0.,:lon=>0.,:alt=>0.)) where T<:AbstractFloat

Yields an 'Instrument' structure that defines an instrument composed of an
antenna and a receiver. The position of the instrument is defined by the
'coords' composed longitude, latitude and altitude coordinates. Note that
instrument's coordinates can evolve with time.

'get_psd_gain_coeff' can be used with an 'Instrument' as argument to get the
instrument gain coefficient (amplifier gain, frequency resolution, Boltzman
constant, impedance and frequency respopnse).

"""
struct Instrument{T<:AbstractFloat,U<:Union{T,AbstractVector{T}}}
    antenna::Antenna{T} # antenna
    #TODO: Union Transmitter with new struct here?
    receiver::Receiver{T} # receiver of precision T
    coords::Dict{Symbol,U} # coordinates

    function Instrument(antenna::Antenna{T},
        receiver::Receiver{T},
        coords::Dict{Symbol,U} = 
                Dict(:lat => 0.,:lon => 0.,:alt=>0.)) where {T<:AbstractFloat,
                                                        U<:Union{T,AbstractVector{T}}}

        ant_fmin, ant_fmax = antenna.valid_freqs
        cent_freq = receiver.cent_freq
        bw = receiver.bw
        f_res = receiver.freq_res
        @assert (ant_fmin <= cent_freq - bw/2) && 
                (cent_freq + bw/2 <= ant_fmax) "the receiver does not cover the \
                antenna valid frequency range"
        if length(coords[:lat]) > 1
            max_T_length = length(coords[:lat])
            if is_TiFreqArray(receiver.T_rx)
                max_T_length = size(receiver.T_rx,:times)
            end
            if is_TiFreqArray(antenna.T_phy)
                max_T_length = max(max_T_length, size(antenna.T_phy,:times))
            end
            @assert length(coords[:lat]) == max_T_length "the length of coords must \
                    match the first dimension (time) of T_rx and T_phy, if they are \
                    time dependent"
        end
        @assert :lat in keys(coords) "`:lat` must be a key of coords"
        @assert :lon in keys(coords) "`:lon` must be a key of coords"
        @assert :alt in keys(coords) "`:alt` must be a key of coords"

        return new{T,U}(antenna, receiver, coords)
    end
end

Base.show(io::IO, I::Instrument{T}) where {T} = begin
    print(io, "Instrument{$T}:\n")
    print(io, "antenna: $(I.antenna)\n")
    print(io, "receiver: $(I.receiver)\n")
    if length(I.coords[:lat]) > 1
        print(io, "mobile instrument")
    elseif I.coords[:lat] == 0 && I.coords[:lon] == 0 && I.coords[:alt] == 0
        print(io, "instrument coordinates not defined")
    else
        print(io, "instrument at $(I.coords[:lat]), $(I.coords[:lon]), \
                   $(I.coords[:alt])")
    end
end

get_psd_gain_coeff(I::Instrument) = get_psd_gain_coeff(I.receiver)



"""
    instrument_psd_stat(i::Instrument{T},
                        T_b::Union{T,DimArray{T},SphereMap{T}},
                        integration_samp::Real = 1) where T

Yields the power spectral density and its variance for the given instrument and
sky brightness temperature 'T_b'.

"""#FIXME: needs pointing coord of antenna!! Define for Observation, not Instrument
function instrument_psd_stat(I::Instrument{T},
    T_b::Union{T,SphereMap{T}},
    integration_samp::Int) where T

    # get receiver and antenna parameters
    rec = I.receiver
    ant = I.antenna

    # instrument gain coefficient
    gain = get_psd_gain_coeff(I)

    # antenna temperature
    T_a = get_antenna_temperature(ant, T_b)
    
    # instrument noise temperature
    T_n = rec.T_rx .+ get_antenna_radiation_loss(ant)

    # calculate power spectral density
    return instrument_psd_stat(gain, T_a, T_n, integration_samp)
end



"""
    Observation(antenna_traj::Trajectory,
                instrument::Instrument{T},
                result::DimArray{T}) where T

Yields an 'Observation' structure that stores the trajectory of the antenna
during the observation, the instrument used and the observation results.

'result' is a 'DimArray' and must have its dimensions named :times, :freqs and
:traj_idx. The length of the :times dimension must be equal to the number of time
stamps in 'antenna_traj'. The length of the :freqs dimension must be equal to
the number of frequency channels in 'instrument.receiver'. The size of the
:traj_idx dimension must be equal to the number of coordinates per time stamp in
'antenna_traj'.

---
    Observation(trajectory::Trajectory,
                instrument::Instrument{T},
                start_date::DateTime = minimum(trajectory.times),
                stop_date::DateTime = maximum(trajectory.times)) where T

Yields an 'Observation' structure that stores the trajectory of the antenna
during the observation, the instrument used and forms the observation results.
The trajectory is filtered to only keep the points between 'start_date' and
'stop_date'.

"""
struct Observation{T<:AbstractFloat}
    antenna_traj::Trajectory # antenna trajectory during observation
    instrument::Instrument{T} # instrument used for observation
    result::DimArray{T} # store the results of observation

    function Observation(antenna_traj::Trajectory,
        instrument::Instrument{T},
        result::DimArray{T}) where T
        
        dim_names = name.(dims(result))
        for dim in dim_names
            @assert dim in [:times, :freqs, :traj_idx] "'result' must have \
                        dimensions named :times, :freqs and/or :traj_idx"
        end
        @assert size(result, :times) == length(antenna_traj.times) "\
                the :times dimension of result must match the number of time \
                stamps of the antenna trajectory"
        @assert size(result, :freqs) == get_nb_freq_chan(instrument.receiver) "\
                the :freqs dimension of result must match the number of frequency \
                channels of the instrument"
        if length(dim_names) > 2
            @assert size(result, :traj_idx) == size(antenna_traj.traj,2) "the :traj_idx \
                    dimension of result must match the number of coords per time stamp \
                    of the antenna trajectory"
        end
        if size(instrument.antenna.T_phy, 1) > 1
            @assert size(instrument.antenna.T_phy, :times) == size(result, :times) "\
                    the :times dimension of result must match the number of time \
                    stamps of the instrument physical temperature 'T_phy'"
        end
        if size(instrument.receiver.T_rx, 1) > 1
            @assert size(instrument.receiver.T_rx, :times) == size(result, :times) "\
                    the :times dimension of result must match the number of time \
                    stamps of the instrument receiver temperature 'T_rx'"
        end

        return new{T}(antenna_traj, instrument, result)
    end
end

function Observation(trajectory::Trajectory,
    instrument::Instrument{T},
    start_date::DateTime = minimum(trajectory.times),
    stop_date::DateTime = maximum(trajectory.times)) where T

    # filter date and other from trajectory
    traj = Trajectory(trajectory, start_date, stop_date)
    isempty(traj.traj) && error("No pointing positions found for the given time window.")

    # create result storage
    time_stamps = traj.times
    freq_bins = freq_range(instrument.receiver)
    n_coords = size(traj.traj,2)
    result = DimArray(fill(zero(T), length(time_stamps), length(freq_bins), n_coords),
                      (Dim{:times}(time_stamps), Dim{:freqs}(freq_bins), 
                       Dim{:traj_idx}(1:n_coords)))

    return Observation(traj, instrument, result)
end

Base.show(io::IO, O::Observation{T}) where {T<:AbstractFloat} = begin
    print(io, "Observation{$T}:\n")
    print(io, "antenna trajectory: $(O.antenna_traj)\n")
    print(io, "instrument: $(O.instrument)\n")
end



"""
AbstractBkg type for different background models. Requires to define a 

'get_antenna_temperature(A::Antenna, S::AbstractBkg, coords::Trajectory)'

method that returns a 'DimArray'.

See ['MovingSkyMdl'](@ref), ['MovingExtendSrcTemp'](@ref),
['PointLikeSrcFlux'](@ref).

"""
abstract type AbstractBkg end



"""
    get_antenna_temperature(A::Antenna{T},
                            S::AbstractBkg,
                            point_coord::Vector{SphereCoord},
                            t::Vector{DateTime}) where T

Yields the antenna temperature at the given coordinates and times.

"""
function get_antenna_temperature(A::Antenna{T},
    S::AbstractBkg,
    point_coord::Vector{SphereCoord},
    t::Vector{DateTime};
    kwds...) where T

    return get_antenna_temperature(A, S, Trajectory(point_coord, t); kwds...)
end



"""
    get_antenna_temperature(A::Antenna{T},
                            S::AbstractBkg,
                            point_coord::SphereCoord,
                            t::DateTime) where T

Yields the antenna temperature at time 't', position 'point_coord' for the
background model 'S'.

"""
function get_antenna_temperature(A::Antenna{T}, 
    S::AbstractBkg, 
    point_coord::SphereCoord,
    t::DateTime;
    kwds...) where T

    return get_antenna_temperature(A, S, Trajectory([point_coord], [t]); kwds...)
end



# """
#     instrument_psd_stat(i::Instrument{T},
#                         T_b::AbstractBkg,
#                         t::DateTime,
#                         integration_samp::Real = 1) where T

# Yields the power spectral density and its variance for the given instrument and
# background brightness temperature 'T_b' at time 't'.

# """
# function instrument_psd_stat(I::Instrument{T},#FIXME: Observation, not instrument
#     T_b::AbstractBkg,
#     t::DateTime,
#     integration_samp::Int) where T

#     # get receiver and antenna parameters
#     rec = I.receiver
#     ant = I.antenna

#     # instrument gain coefficient
#     gain = get_psd_gain_coeff(I)

#     # antenna temperature
#     T_a = get_antenna_temperature(ant, T_b, t)
    
#     # instrument noise temperature
#     T_n = rec.T_rx .+ get_antenna_radiation_loss(ant)

#     # calculate power spectral density
#     return instrument_psd_stat(gain, T_a, T_n, integration_samp)
# end



"""
    PointLikeSrcFlux{T<:AbstractFloat}
    PointLikeSrcFlux(flux::DimArray{T},
                     traj::Trajectory) where T

Yields a 'PointLikeSrcFlux' structure that defines a point-like source with a
flux defined as a 'DimArray' and a trajectory 'traj' over the sky. The flux is
defined in Jansky for given frequencies.

'get_antenna_temperature' can be used to get the antenna temperature of a
point-like source defined as a 'PointLikeSrcFlux' structure. See
['get_antenna_temperature'](@ref). 

"""
struct PointLikeSrcFlux{T<:AbstractFloat} <: AbstractBkg
    flux::DimArray{T}
    traj::Trajectory

    function PointLikeSrcFlux(flux::DimArray{T},
        traj::Trajectory) where {T<:AbstractFloat}
        
        @assert all(d -> name(d) in (:times, :freqs), dims(flux)) "flux must have \
                :times and :freqs dimensions"

        @assert size(traj.traj, 2) == 1 "traj must have only one coordinate per time"

        return new{T}(flux, traj)
    end
end

function get_antenna_temperature(A::Antenna{T},
    S::PointLikeSrcFlux{T},
    ant_traj::Trajectory;
    pre_load_rot_mat::Union{<:AbstractArray{Matrix{T}},Nothing} = nothing) where T

    if isnothing(pre_load_rot_mat)
        pre_load_rot_mat = rot_mat.(ant_traj.traj)#TODO: maybe reduce the nb of rot_mat
    else
        @assert size(pre_load_rot_mat) == size(ant_traj.traj) "pre_load_rot_mat \
                must have same size than ant_traj.traj."
    end

    # nb of coords per time stamps
    nb_coords = size(ant_traj.traj,2)

    # wavelength grid
    wave_grid = freq_to_wave.(dims(S.flux, :freqs))

    # scaling coef
    scale = wave_grid.^2 ./ (8π * k_boltz) .* 1e-26
    src_flux = broadcast_dims(*, S.flux, scale)

    T_As = DimArray(zeros(T, length(ant_traj.times), length(wave_grid), nb_coords), 
                    (Dim{:times}(ant_traj.times), dims(S.flux, :freqs), 
                     Dim{:traj_idx}(1:nb_coords)))
    @threads for t in eachindex(ant_traj.times)
        # get antenna pointing coord
        ant_time = ant_traj.times[t]
        ant_coords = ant_traj.traj[t,:]

        # get source coord and flux at time t
        src_coord = S.traj(ant_time)[1]
        S_t = src_flux[times=At(ant_time)]
        
        for c in eachindex(ant_coords)
            # find source coord in antenna frame
            src_coord_in_ant = pass_frame_to_frame(src_coord, ant_coords[c];
                                                pre_load_rot_mat=pre_load_rot_mat[t,c])
            
            # get antenna temperature
            gain = get_gain_value(A, src_coord_in_ant)
            T_As[times=t,traj_idx=c] .= gain .* S_t
        end
    end

    return T_As
end



"""
    MovingExtendSrcTemp{T<:AbstractFloat}
    MovingExtendSrcTemp(temp::DimArray{SphereMap{T}},
                        traj::Trajectory) where T

Yields a 'MovingExtendSrcTemp' structure that defines a moving extended source
with a temperature defined as a 'DimArray{SphereMap}' and a trajectory 'traj'. The
temperature is defined in Kelvin for given times. The trajectory defines the
movement of the extended source over the sky, over time, with the first time
sample defining the original position of the SphereMap. 

'get_antenna_temperature' can be used to get the antenna temperature of a
moving extended source defined as a 'MovingExtendSrcTemp' structure. See
['get_antenna_temperature'](@ref).

---
    MovingExtendSrcTemp(temp::SphereMap{T},
                        traj::Trajectory) where T

Yields a 'DimArray{SphereMap}' structure that defines a moving extended source
with a temperature defined as a 'SphereMap' and a trajectory 'traj'. The
temperature is defined in Kelvin for given angles. The trajectory defines the 
movement of the extended source over the sky, which is computed and stored in a
'DimArray{SphereMap}'.

"""
struct MovingExtendSrcTemp{T<:AbstractFloat} <: AbstractBkg
    temp::DimArray{SphereMap{T}}
    traj::Trajectory

    function MovingExtendSrcTemp(temp::DimArray{SphereMap{T}},
        traj::Trajectory) where {T<:AbstractFloat}

        @assert :times in name.(dims(temp)) "temp must have a :times dimension"

        @assert traj.times == Array(dims(temp, :times)) "temp and traj must have the \
                same times"

        @assert length(axes(traj.traj,2)) == 1 "traj must have only one coordinate per \
                time stamp"

        return new{T}(temp, traj)
    end
end

function MovingExtendSrcTemp(temp::SphereMap{T},
    traj::Trajectory) where T
    
    alpha_grid = temp.alpha_grid
    beta_grid = temp.beta_grid
    temp_vec = [temp]
    last_coord = traj.traj[1]
    for t in traj.times[2:end]
        moving_coords = traj(t)[1]
        if moving_coords == last_coord
            push!(temp_vec, temp_vec[end])
        else
            push!(temp_vec, SphereMap(alpha_grid, beta_grid, 
                                      rotate_to(temp, moving_coords)))
        end
        last_coord = moving_coords
    end

    return MovingExtendSrcTemp(DimArray(temp_vec, (Dim{:times}(traj.times))), traj)
end

function get_antenna_temperature(A::Antenna{T},
    S::MovingExtendSrcTemp{T},
    ant_traj::Trajectory;
    kwds...) where T

    return get_antenna_temperature(A, S.temp, ant_traj; kwds...)
end



"""#TODO: add DimArray{SphereMapT} and assert with ptl_srcs_flux :freqs dim
   #TODO: OR AbstractBkg instead of MovingExtendSrcTemp in arg def (keep assert)
    SkyMdl(bkg_srcs_temp::MovingExtendSrcTemp{T},
                 ptl_srcs_flux::Vector{PointLikeSrcFlux{T}}) where T

Yields a 'SkyMdl' structure that defines a sky model with background
sources defined either as a constant temperature 'T', a 'SphereMap' spatially
dependent temperature, or a 'MovingExtendSrcTemp' spatio-temporally dependent
temperature and point-like sources defined as a vector of 'PointLikeSrcFlux'.
The background sources are defined in Kelvin. The point-like sources
are defined by their flux and trajectory.  

The sky model can be used to get the antenna temperature at a given time using
'get_antenna_temperature'. See ['get_antenna_temperature'](@ref).

"""
struct SkyMdl{T<:AbstractFloat} <: AbstractBkg
    # background source temperature (constant, spatially dependent or
    # spatio-temporally dependent)
    bkg_srcs_temp::Union{T,SphereMap{T},AbstractBkg}#MovingExtendSrcTemp{T}}
    ptl_srcs_flux::Vector{PointLikeSrcFlux{T}} # point-like sources fluxes
    
    function SkyMdl(bkg_srcs_temp::Union{T,SphereMap{T},AbstractBkg},#MovingExtendSrcTemp{T}},
        ptl_srcs_flux::Vector{PointLikeSrcFlux{T}} = PointLikeSrcFlux{T}[]) where T
        
        if !isempty(ptl_srcs_flux)
            for s in 2:length(ptl_srcs_flux)
                @assert ptl_srcs_flux[s].traj.times == ptl_srcs_flux[1].traj.times "all \
                        point-like sources must have the same times"
                @assert dims(ptl_srcs_flux[s].flux) == dims(ptl_srcs_flux[1].flux) "all \
                        point-like sources must have the same dimensions"
            end
            if typeof(bkg_srcs_temp) <: MovingExtendSrcTemp
                @assert Array(dims(bkg_srcs_temp.temp, :times)) == 
                        ptl_srcs_flux[1].traj.times "the times of the background \
                        sources and the point-like sources must match"
            end
        end

        return new{T}(bkg_srcs_temp, ptl_srcs_flux)
    end
end

function SkyMdl(bkg_srcs_temp::Union{T,SphereMap{T},AbstractBkg},#MovingExtendSrcTemp{T}},
    ptl_srcs_flux::PointLikeSrcFlux{T}) where T

    return SkyMdl(bkg_srcs_temp, [ptl_srcs_flux])
end

function get_antenna_temperature(A::Antenna{T},
    S::SkyMdl{T}, 
    ant_traj::Trajectory;
    kwds...) where T
    
    # get antenna temperature of background sources
    # FIXME: why seperating MovingExtendSrcTemp and SphereMap?
    if typeof(S.bkg_srcs_temp) <: AbstractBkg #MovingExtendSrcTemp
        T_bkg = get_antenna_temperature(A, S.bkg_srcs_temp, ant_traj; kwds...)
    elseif typeof(S.bkg_srcs_temp) <: SphereMap
        T_bkg = get_antenna_temperature(A, S.bkg_srcs_temp, ant_traj; kwds...)
    else
        T_bkg = get_antenna_temperature(A, S.bkg_srcs_temp; kwds...)
    end
    
    # get antenna temperature of point-like sources
    if !isempty(S.ptl_srcs_flux)
        for s_t in S.ptl_srcs_flux
            T_bkg = broadcast_dims(+, T_bkg, get_antenna_temperature(A, s_t, ant_traj;
                                                                     kwds...))
        end
    end

    return T_bkg
end



"""
    Satellite(sat_name::String,
              instrument::Instrument{T},
              EIRP_density::Union{T,DimArray{T}},
              traj::Trajectory) where T

Yields a 'Satellite' structure that defines a satellite with its name,
instrument, effective isotropic radiated power density in W/Hz and trajectory.
Assumes the frame of satellite antenna is oriented North-East-Nadir. The antenna
pointing can be any direction from Nadir, defined in the 'gain_pat' (e.g. for
beamforming).

As the transmitted power is defined by the 'EIRP_density', the antenna gain
pattern of the satellite must be normalized such that the boresight gain is 1.0.
This is to ensure that the power received at the boresight is equal to the EIRP.

EIRP density level can be defined as a constant or a DimArray. If 'EIRP_density'
is a DimArray, it must have the 'times' and/or the 'freqs' dimension.

Use 'get_sat_EIRP_density' to get the EIRP density at a given time (see
['get_sat_EIRP_density'](@ref)).

"""
struct Satellite{T<:AbstractFloat}
    sat_name::String # name of satellite
    instrument::Instrument{T} # instrument of satellite
    EIRP_density::Union{T,DimArray{T}} # effective isotropic radiated power in W/Hz
    sat_traj::Trajectory # trajectory of satellite

    function Satellite(sat_name::String,
        instrument::Instrument{T},
        EIRP_density::Union{T,DimArray{T}},
        sat_traj::Trajectory) where T
        
        if typeof(EIRP_density) <: DimArray
            is_TiFreqArray(EIRP_density)
            if :freqs in name.(dims(EIRP_density))
                @assert size(EIRP_density, :freqs) == 
                        get_nb_freq_chan(instrument.receiver) "if EIRP_density is a \
                        DimArray with a :freqs dimension, its length must match the \
                        number of frequency channels of the satellite receiver"
            end
            if :times in name.(dims(EIRP_density))
                @assert size(EIRP_density, :times) == length(sat_traj.times) "if \
                        EIRP_density is a DimArray with a :times dimension, its length \
                        must match the number of time stamps of the satellite trajectory"
            end
        end 

        @assert isapprox(get_boresight_gain(instrument.antenna)[1], 1.0; 
                         rtol=1e-3) "the antenna gain pattern of the satellite must be \
                normalized such that the boresight gain is 1.0"

        @assert instrument.receiver.gain_amps == 1. "the receiver gain of the satellite \
                                                     must be 1.0, as the EIRP density is \
                                                     already defined in W/Hz"

        return new{T}(sat_name, instrument, EIRP_density, sat_traj)
    end
end

get_sat_traj(S::Satellite) = S.sat_traj



"""
    get_sat_EIRP_density(S::Satellite, 
                         t::DateTime)

Returns the EIRP density of the satellite at time 't'. If the EIRP density is
defined as a DimArray with a :times dimension, the value at time 't' is
returned. If the EIRP density is defined as a constant, the constant value is
returned.

"""
function get_sat_EIRP_density(S::Satellite, 
    t::DateTime;
    time_res::Union{Nothing,D} = nothing) where D<:Dates.Period
    
    gain_sat = S.instrument.receiver.gain_amps .* S.instrument.receiver.freq_resp

    if typeof(S.EIRP_density) <: DimArray && :times in name.(dims(S.EIRP_density)) 
        if isnothing(time_res)
            return gain_sat .* S.EIRP_density[times=Near(t)]
        else
            return gain_sat .* S.EIRP_density[times=(t - time_res) .. (t + time_res)]
        end
    else
        return gain_sat .* S.EIRP_density
    end
end



"""
    form_satellites_list(sats_info::AbstractDataFrame,
                         sats_instrument::Instrument{T},
                         sat_eirp_density_func::F,
                         start_time::DateTime,
                         stop_time::DateTime;
                         sat_id_tag::Symbol = :sat,
                         time_tag::Symbol = :times,
                         kwds...) where {T,N}

Forms a list of 'Satellite' structs from a DataFrame containing satellite
information and a 'start_time' and 'stop_time'. The DataFrame must contain a
column with 'sat_id_tag'. 'sat_eirp_density_func' is a function that takes a
'Trajectory' and returns a scalar or DimArray of EIRP values. This is to allow
for EIRP satellite, range, elevation, time dependencies.

---
    form_satellites_list(sats_info::AbstractDataFrame,
                         sats_instrument::Instrument{T},
                         sat_eirp::Union{T,DimArray{T}},
                         args...;
                         kwds...) where {T,N}

Forms a list of 'Satellite' structs when the EIRP is already defined (e.g.
constant or the same variations for all satellites).

---
    form_satellites_list(path_sats_info::String,
                         args...;
                         kwds...)

Forms a list of 'Satellite' structs from a file path leading to a 'Arrow' table.

"""
function form_satellites_list(sats_info::AbstractDataFrame,
    sats_instrument::Instrument{T},
    sat_eirp_density_func::F,
    start_time::DateTime,
    stop_time::DateTime;
    rotate_beam::Bool = false,
    sat_id_tag::Symbol = :sat,
    time_tag::Symbol = :times,
    kwds...) where {T,F<:Function}

    @assert sat_id_tag in propertynames(sats_info) "sat_id_tag must be a column of \
            'sats_info"

    @assert hasmethod(sat_eirp_density_func, (Trajectory,)) "sat_eirp must be a \
            function of signature (Trajectory) -> Union{T,DimArray{T}}"

    # filter by time window
    sats_info = subset(sats_info, time_tag => t -> start_time .<= t .<= stop_time; 
                       view=true)
    sats_ant = sats_instrument.antenna
    
    list_sats = unique(sats_info, sat_id_tag)[:,sat_id_tag]
    sats = Vector{Satellite{T}}(undef, length(list_sats))
    @threads :dynamic for i in eachindex(list_sats)
        s = list_sats[i]
        # isolate satellite
        sat_info = subset(sats_info, sat_id_tag => n -> n .== s; view=true)

        # create trajectory
        sat_traj = Trajectory(sat_info; time_tag=time_tag, kwds...)

        # form satellite EIRP density (can depend on time and/or elevation, etc.)
        sat_eirp_density = sat_eirp_density_func(sat_traj)

        # beamform the satellite antenna pattern
        if rotate_beam#TODO: make this more complex
            sat_pat = sats_ant.gain_pat
            sat_map_rot = rotate_to(sat_pat, 0., rand(sat_pat.beta_grid))
            sat_gain_pat_rot = SphereMap(sat_pat.alpha_grid, sat_pat.beta_grid, sat_map_rot)
            sat_ant_rot = Antenna(sats_ant.ant_diameter, sat_gain_pat_rot, 
                                  sats_ant.ap_eff, sats_ant.rad_eff, 
                                  sats_ant.valid_freqs, sats_ant.T_phy)
            sats_instrument = Instrument(sat_ant_rot, sats_instrument.receiver, 
                                         sats_instrument.coords)
        end

        # create satellite
        sats[i] = Satellite(s, sats_instrument, sat_eirp_density, sat_traj)
    end

    return sats
end

function form_satellites_list(sats_info::AbstractDataFrame,
    sats_instrument::Instrument{T},
    sat_eirp_density::Union{T,DimArray{T}},
    args...;
    kwds...) where T

    sat_eirp_density_func(traj::Trajectory) = sat_eirp_density

    return form_satellites_list(sats_info, sats_instrument, sat_eirp_density_func, 
                                args...; kwds...)
end

function form_satellites_list(path_sats_info::String,
    args...;
    time_tag::Symbol = :times,
    kwds...)

    @assert occursin(".arrow", path_sats_info) "the file must be an Arrow table."

    sats_info = DataFrame(Arrow.columntable(Arrow.Table(path_sats_info)))
    @. sats_info[!,time_tag] = Dates.DateTime(sats_info[!,time_tag])

    return form_satellites_list(sats_info, args...; time_tag=time_tag, kwds...)
end



"""
    Constellation(constellation_name::String,
                  sats::Vector{Satellite{T}},
                  lnk_bdgt_mdl::Function) where T
                  
Defines a 'Constellation' structure that contains the name of the constellation,
a vector of 'Satellite' structs and a link budget model function. The link
budget model function must have the signature (SphereCoord, Instrument,
SphereCoord, Instrument).

Use 'get_sat' to get a satellite from the constellation by its name.

"""
struct Constellation{T<:AbstractFloat}
    constellation_name::String # name of constellation
    sats::Vector{Satellite{T}} # satellites in constellation
    lnk_bdgt_mdl::Function # link budget model function

    function Constellation(constellation_name::String,
        sats::Vector{Satellite{T}},
        lnk_bdgt_mdl::Function) where T

        # check lnk_bdgt_mdl signature is correct
        @assert hasmethod(lnk_bdgt_mdl, (SphereCoord{T}, Instrument{T}, SphereCoord{T}, 
                                         Instrument{T})) "lnk_bdgt_mdl must be a \
                function of signature (SphereCoord, Instrument, SphereCoord, Antenna)"

        return new{T}(constellation_name, sats, lnk_bdgt_mdl)
    end
end

Base.show(io::IO, C::Constellation{T}) where {T} = begin
    print(io, "Constellation{$T}:\n")
    print(io, "constellation name: $(C.constellation_name)\n")
    print(io, "number of satellites: $(length(C.sats))")
end

# function Constellation(start_date::DateTime,
#     stop_date::DateTime,
#     constellation_name::String,
#     sats::AbstractDataFrame,
#     sat_inst::Instrument{T},
#     lnk_bdgt_mdl::Function = sat_link_budget;
#     filt_funcs::NTuple{N,Pair} = ()) where {N,T}

#     # sats = Satellite{T}[]
#     # for sat_info in sat_infos
#     #     # create trajectory
#     #     traj = Trajectory(sat_info.traj_file_path; traj_kwds...)

#     #     # create instrument
#     #     inst = Instrument{T}(sat_info.antenna, sat_info.receiver, sat_info.coords)

#     #     # create satellite
#     #     push!(sats, Satellite{T}(sat_info.name, traj, inst))
#     # end

#     return Constellation(constellation_name, sats, lnk_bdgt_mdl)
# end

# function Constellation(file_path::String,
#     observation::Observation,
#     sat_tmt::Instrument{T},
#     lnk_bdgt_mdl::Function = sat_link_budget;
#     name_tag::Symbol = :sat,
#     time_tag::Symbol = :time_stamps,
#     elevation_tag::Symbol = :altitudes,
#     azimuth_tag::Symbol = :azimuths,
#     distance_tag::Symbol = :distances,
#     filt_funcs::NTuple{N,Pair} = ()) where {N,T}

#     # # load the trajectory as a DataFrame
#     # sats = DataFrame(Arrow.columntable(Arrow.Table(file_path)))

#     # # rename columns
#     # rename!(sats, time_tag => :times)
#     # rename!(sats, name_tag => :sat)
#     # rename!(sats, azimuth_tag => :azimuths)
#     # rename!(sats, elevation_tag => :elevations)
#     # rename!(sats, distance_tag => :distances)
    
#     # @. sats[!,:times] = Dates.DateTime(sats[!,:times])

#     # sort!(sats, :times)

#     # return Constellation(sats, observation, sat_tmt, lnk_bdgt_mdl; 
#     #                      filt_funcs=filt_funcs)
# end

get_sat(C::Constellation, s::String) = C.sats[findfirst(sats -> sats.sat_name .== s, C.sats)]

get_sats_name(C::Constellation) = [sat.sat_name for sat in C.sats]


"""
    get_sat_traj(C::Constellation,
                 s::String)

Yields the trajectory of satellite 's' in constellation 'C'.

"""
function get_sat_traj(C::Constellation,
    s::String)
    
    sat_index = findfirst(sats -> sats.sat_name .== s, C.sats)

    return get_sat_traj(C.sats[sat_index])
end



"""
    get_sats_names_at_time(C::Constellation,
                           t::DateTime;
                           time_res::Union{Nothing,D} = nothing) where D<:Dates.Period

Yields the names of satellites that are visible at time 't' in constellation
'C'. If 'time_res' is not nothing, the result includes satellites whose position
samples are close to 't', within the time resolution given.

"""
function get_sats_names_at_time(C::Constellation,
    t::DateTime;
    time_res::Union{Nothing,D} = nothing) where D<:Dates.Period
    
    if isnothing(time_res)
        sats_ind = findall(sat -> t in get_sat_traj(sat).times, C.sats)
    else
        sats_ind = findall(sat -> maximum((get_sat_traj(sat).times .- time_res) .< t .< 
                                          (get_sat_traj(sat).times .+ time_res)), C.sats)
    end

    return [C.sats[i].sat_name for i in sats_ind]
end

