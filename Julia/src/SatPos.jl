
"""
    module SatPos

Defines satellite trajectory fetching functions and conversions to different
coordinate frames. Uses extensively the ['RadioMdl.CoordFrames'](@ref) module
and ['SatelliteToolbox'](@ref) packages.

BEWARE: Claude-helped coded for some Julia parts

"""
module SatPos

export angular_separation,
       compute_sats_traj,
       compute_sats_traj_py,
       fetch_satellites_info,
       gcs_to_ecef,
       ned_to_ecef_rotation,
       nwz_to_ecef_rotation,
       sats_close_to_pointing,
       tel_dir_in_sat_frame,
       url_celestrak,
       url_starlink_celestrak

using ..CoordFrames
using Arrow
using CSV
using DataFrames
using Dates
using EzXML 
using HTTP
using LinearAlgebra
using PyCall
using SatelliteToolbox
using SatelliteToolboxSgp4
using SatelliteToolboxTransformations: geodetic_to_ecef, ecef_to_geodetic
using StaticArrays



""" URL of Celestrak (https://celestrak.org/) in csv format """
const url_celestrak = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&\
                       FORMAT=csv"

""" URL of Starlink via Celestrak (https://celestrak.org/) in csv format """
const url_starlink_celestrak = "https://celestrak.org/NORAD/elements/gp.php?\
                                GROUP=starlink&FORMAT=csv"



### FETCHING SATELLITES INFO ###



""" Numeric OMM/GP fields, coerced to Float64 (EPOCH and metadata stay strings). """
const _OMM_NUMERIC = Set([:MEAN_MOTION, :ECCENTRICITY, :INCLINATION, :RA_OF_ASC_NODE,
                          :ARG_OF_PERICENTER, :MEAN_ANOMALY, :BSTAR, :MEAN_MOTION_DOT,
                          :MEAN_MOTION_DDOT, :NORAD_CAT_ID, :ELEMENT_SET_NO,
                          :REV_AT_EPOCH, :EPHEMERIS_TYPE])



""" Recursively collect leaf elements of an OMM node into `name => text`. """
function _omm_leaves!(d::Dict{Symbol,String}, node)
    for el in eachelement(node)
        if isempty(collect(eachelement(el)))
            d[Symbol(nodename(el))] = strip(nodecontent(el))
        else
            _omm_leaves!(d, el)
        end
    end
    return d
end



"""
    omm_xml_to_dataframe(xml::AbstractString; is_path::Bool = true)

Parse a CCSDS OMM/NDM XML document into a DataFrame whose columns match the
CelesTrak `FORMAT=csv` (GP) schema, so that both formats are interchangeable
downstream.

"""
function omm_xml_to_dataframe(xml::AbstractString; is_path::Bool = true)

    doc  = is_path ? readxml(xml) : parsexml(xml)
    rt   = root(doc)

    # CCSDS NDM/XML normally has no default namespace, but be defensive
    nodes = findall("//omm", rt)
    isempty(nodes) && (nodes = findall("//*[local-name()='omm']", rt))
    isempty(nodes) && error("no <omm> segment found in $(is_path ? xml : "XML string")")

    df = DataFrame()
    for n in nodes
        push!(df, _omm_leaves!(Dict{Symbol,String}(), n); cols = :union)
    end

    # coerce numerics; leave EPOCH / OBJECT_NAME / OBJECT_ID as strings
    for c in intersect(Symbol.(names(df)), _OMM_NUMERIC)
        df[!, c] = [ismissing(v) ? missing : parse(Float64, v) for v in df[!, c]]
    end

    return df
end



"""
    fetch_satellites_info(;
                info_path::String = url_celestrak,
                name_filters::Union{Nothing,String,AbstractVector{<:String}} = nothing,
                avoid_names::Union{Nothing,String,AbstractVector{<:String}} = nothing,
                verb::Bool = false,
                save::Bool = false)

Fetch satellites information from Celestrak website or from a local file, in the
CSV or OMM/XML format. The information is returned in a DataFrame. 

"""
function fetch_satellites_info(;
    info_path::String = url_celestrak,
    name_filters::Union{Nothing,String,AbstractVector{<:String}} = nothing,
    avoid_names::Union{Nothing,String,AbstractVector{<:String}} = nothing,
    verb::Bool = false,
    save::Bool = false)

    if occursin("https", info_path)
        
        @assert occursin("celestrak.org", info_path) "Only Celestrak website is \
                                                      supported for now"

        body = String(HTTP.get(info_path).body)

        if occursin("FORMAT=csv", info_path)
            sats_catalog = CSV.read(IOBuffer(body), DataFrame)

            # save the fetched csv to reduce access to website
            save && CSV.write("sats_catalog_$(now()).csv", sats_catalog)
            
        elseif occursin("FORMAT=xml", info_path)
            sats_catalog = omm_xml_to_dataframe(body; is_path=false)

            save && write("sats_catalog_$(now()).xml", body)
        else 
            error("online satellites info must be either in csv or xml format")
        end
    else
        if occursin(".csv", info_path)
            sats_catalog = CSV.read(info_path, DataFrame)
        elseif occursin(".xml", info_path)
            sats_catalog = omm_xml_to_dataframe(info_path)
        else
            error("local satellites info must be in csv or xml format")
        end
    end

    if !isnothing(name_filters)
        typeof(name_filters) == String && (name_filters = [name_filters])
        for filt in name_filters
            filter!(row -> occursin(filt, row.OBJECT_NAME), sats_catalog)
        end
    end
    if !isnothing(avoid_names)
        typeof(avoid_names) == String && (avoid_names = [avoid_names])
        for filt in avoid_names
            filter!(row -> !(occursin(filt, row.OBJECT_NAME)), sats_catalog)
        end
    end
    
    @assert allunique(sats_catalog.NORAD_CAT_ID)

    verb && println("Found $(size(sats_catalog,1)) satellites with the given filters")

    return sats_catalog
end



#### ORBIT PROPAGATION OF SATELLITES ####



""" Process-wide cache for the IERS EOP data (network IO is done at most once). """
const _EOP_CACHE = Ref{Any}(nothing)



"""
    omm_epoch(x) -> DateTime

Parse an OMM `EPOCH` field. Accepts a `DateTime` (already typed by CSV.jl) or an
ISO 8601 string with up to 6 fractional digits (truncated to milliseconds, the
`DateTime` resolution).

"""
function omm_epoch(x)
    x isa DateTime && return x
    s = strip(String(x))
    endswith(s, "Z") && (s = s[1:end-1])
    i = findfirst('.', s)
    isnothing(i) || (s = s[1:min(length(s), i + 3)])
    return DateTime(s)
end



"""
    sgp4_elements(row) -> NamedTuple

Extract the eight SGP4 inputs from one OMM/GP record as **plain Julia numbers**
(no mutable state, safe to share across threads). Angles are converted to
radians and the mean motion to radians per minute.

"""
function sgp4_elements(row)

    theory = hasproperty(row, :MEAN_ELEMENT_THEORY) ? String(row.MEAN_ELEMENT_THEORY) :
                                                     "SGP4"
    theory == "SGP4" || error("unsupported mean element theory '$theory' for \
                               $(row.OBJECT_NAME): BSTAR carries a different \
                               quantity in SGP4-XP")
    abs(Float64(row.BSTAR)) < 1.0 ||
        error("$(row.OBJECT_NAME): |BSTAR| = $(row.BSTAR) suggests SGP4-XP AGOM")

    return (name  = String(row.OBJECT_NAME),
            norad = Int(row.NORAD_CAT_ID),
            jd    = datetime2julian(omm_epoch(row.EPOCH)),
            n_0   = Float64(row.MEAN_MOTION) * 2π / 1440.,   # rev/day -> rad/min
            e_0   = Float64(row.ECCENTRICITY),
            i_0   = deg2rad(Float64(row.INCLINATION)),
            Ω_0   = deg2rad(Float64(row.RA_OF_ASC_NODE)),
            ω_0   = deg2rad(Float64(row.ARG_OF_PERICENTER)),
            M_0   = deg2rad(Float64(row.MEAN_ANOMALY)),
            bstar = Float64(row.BSTAR))
end



""" Topocentric elevation / azimuth / range from one SGP4 propagation. """
@inline function _topo(sgp4d, jd::Float64, jd_ep::Float64,
                       R_e::SMatrix{3,3,Float64,9}, R_T::SMatrix{3,3,Float64,9},
                       r_tel::SVector{3,Float64})

    r_teme, _ = sgp4!(sgp4d, (jd - jd_ep) * 1440.)          # km, TEME
    (all(isfinite, r_teme) && 6400.0 < norm(r_teme) < 1.0e6) ||
        return (-91.0, 0.0, NaN)          # el = -91 → always fails el_min
    d = R_T * (R_e * (SVector{3,Float64}(r_teme) .* 1000.) .- r_tel)   # (N, W, Z) [m]

    ρ = hypot(d[1], d[2])
    return atand(d[3], ρ),                  # elevation [deg]
           mod(-atand(d[2], d[1]), 360.),   # azimuth N->E [deg] (E = -W)
           sqrt(ρ * ρ + d[3] * d[3])        # range [m]
end



"""
    compute_sats_traj(sats_info::AbstractDataFrame,
                      start_time::DateTime,
                      stop_time::DateTime,
                      tel_coords::Dict{Symbol,<:Real},
                      time_res::Dates.Period;
                      el_min::Real = 5.,
                      eop = nothing,
                      use_eop::Bool = true,
                      sgp4c = sgp4c_wgs72,
                      coarse::Int = 1,
                      ntasks::Int = Threads.nthreads(),
                      save::Bool = false)

Compute the trajectories of the satellites whose OMM/GP orbital information is
in the DataFrame `sats_info` (as returned by ['fetch_satellites_info'](@ref),
CSV or XML) between `start_time` and `stop_time`. Returns a DataFrame with
columns :times, :sat, :elevations, :azimuths, :ranges.

Pure Julia, thread-safe: each task owns its own `Sgp4Propagator`, and the
per-epoch TEME->ITRF rotations are precomputed once and shared read-only.

RE-BEWARE: trajectories are expressed as azimuth and **elevation** angles
(azimuth measured from North toward East), contrary to the convention taken in
the rest of the 'RadioMdl' package.

Set `coarse = n > 1` to pre-screen visibility on a grid `n` times sparser
(large catalogues); `use_eop = false` skips the IERS download and uses PEF
instead of ITRF (sub-arcsecond difference, no network access).

BEWARE: Claude-helped coded

"""
function compute_sats_traj(sats_info::AbstractDataFrame,
    start_time::DateTime,
    stop_time::DateTime,
    tel_coords::Dict{Symbol,<:Real},
    time_res::Dates.Period;
    el_min::Real = 5.,
    eop = nothing,
    use_eop::Bool = true,
    sgp4c = sgp4c_wgs72,
    coarse::Int = 1,
    ntasks::Int = Threads.nthreads(),
    save::Bool = false)
    
    @assert :lat in keys(tel_coords) "`:lat` required"
    @assert :lon in keys(tel_coords) "`:lon` required"
    @assert :alt in keys(tel_coords) "`:alt` required"
    @assert start_time < stop_time "`start_time` must precede `stop_time`"
    
    grid = start_time:time_res:stop_time
    nt = length(grid)
    jds = datetime2julian.(collect(grid))
    
    # Resolve EOP once
    eop_data = if !use_eop
        nothing
    elseif !isnothing(eop)
        eop
    else
        isnothing(_EOP_CACHE[]) && (_EOP_CACHE[] = fetch_iers_eop())
        _EOP_CACHE[]
    end

    # force any lazy initialisation inside the rotation code on a single thread,
    # so the @threads loop below only ever *reads* fully-built structures
    let _ = use_eop ? r_eci_to_ecef(TEME(), ITRF(), first(jds), eop_data) :
                      r_eci_to_ecef(TEME(), PEF(),  first(jds))
    end
    
    R = Vector{SMatrix{3,3,Float64,9}}(undef, nt)
    Threads.@threads :dynamic for i in 1:nt
        R[i] = SMatrix{3,3,Float64,9}(use_eop ?
                                      r_eci_to_ecef(TEME(), ITRF(), jds[i], eop_data) :
                                      r_eci_to_ecef(TEME(), PEF(),  jds[i]))
    end
    
    φ, λ, h = deg2rad(tel_coords[:lat]), deg2rad(tel_coords[:lon]), tel_coords[:alt]
    r_tel = SVector{3,Float64}(geodetic_to_ecef(φ, λ, h))
    R_T = transpose(SMatrix{3,3,Float64,9}(nwz_to_ecef_rotation(φ, λ)))  # ECEF->NWZ
    
    # mean elements as immutable plain data
    elems = sgp4_elements.(eachrow(sats_info))
    
    # conservative elevation margin for the coarse pre-screen (max rate ~1.2 deg/s)
    res_s  = Dates.value(Dates.Nanosecond(time_res)) / 1e9
    margin = coarse > 1 ? min(90., 1.2 * coarse * res_s) : 0.
    
    chunks = collect(Iterators.partition(eachindex(elems),
                                         max(1, cld(length(elems), ntasks))))
    
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            times = DateTime[]; sat = String[]
            els   = Float64[];  azs = Float64[]; rgs = Float64[]
            
            for k in chunk
                el = elems[k]
                sgp4d = sgp4_init(el.jd, el.n_0, el.e_0, el.i_0,
                                  el.Ω_0, el.ω_0, el.M_0, el.bstar; sgp4c)  # THREAD-LOCAL
                
                period_min = 2π / el.n_0 # n_0 in rad/min
                deep = period_min > 225.0 # Vallado's SGP4/SDP4 switch
                cand = if coarse > 1 && !deep
                    ci = collect(1:coarse:nt); last(ci) == nt || push!(ci, nt)
                    ce = [_topo(sgp4d, jds[i], el.jd, R[i], R_T, r_tel)[1] for i in ci]
                    keep = falses(nt)
                    thr  = el_min - margin
                    for j in 1:length(ci)-1
                        (ce[j] > thr || ce[j+1] > thr) && (keep[ci[j]:ci[j+1]] .= true)
                    end
                    # re-init: the coarse pass mutated sgp4d and the fine pass
                    # goes BACKWARD 
                    sgp4d = sgp4_init(el.jd, el.n_0, el.e_0, el.i_0, el.Ω_0, el.ω_0, 
                                      el.M_0, el.bstar; sgp4c)
                    findall(keep)
                else
                    1:nt
                end
                
                @inbounds for i in cand
                    e, a, r = _topo(sgp4d, jds[i], el.jd, R[i], R_T, r_tel)
                    e < el_min && continue
                    push!(times, grid[i]); push!(sat, el.name)
                    push!(els, e); push!(azs, a); push!(rgs, r)
                end
            end
            (; times, sat, els, azs, rgs)
        end
    end
    
    parts = fetch.(tasks)
    
    traj_sats = DataFrame(times = reduce(vcat, getfield.(parts, :times)),
                          sat = reduce(vcat, getfield.(parts, :sat)),
                          elevations = reduce(vcat, getfield.(parts, :els)),
                          azimuths = reduce(vcat, getfield.(parts, :azs)),
                          ranges = reduce(vcat, getfield.(parts, :rgs)))
    
    if save
        Arrow.write("sats_traj_$(start_time)_$(stop_time)_$(el_min)elmin_\
                     loaded$(now()).arrow", traj_sats)
    end
    
    return traj_sats
end



""" Index of the element of the sorted vector `ts` closest to `t`. """
function nearest_idx(ts::AbstractVector, t)
    i = searchsortedfirst(ts, t)
    i == firstindex(ts) && return firstindex(ts)
    i > lastindex(ts)   && return lastindex(ts)
    return (t - ts[i-1]) <= (ts[i] - t) ? i - 1 : i
end



"""
    sats_close_to_pointing(traj_sats::DataFrame, 
                           ant_traj::Trajectory;
                           angle_detect::Real = 5.)

Select the satellites that are close to the pointing direction of the antenna.

"""
function sats_close_to_pointing(traj_sats::DataFrame, 
    ant_times::AbstractVector{DateTime},
    ant_traj::AbstractArray{T};
    angle_detect::Real = 5.) where T

    # antenna trajectory
    time_res = minimum(diff(ant_times))
    ant_az = mod.(360. .- getproperty.(ant_traj, :alpha), 360.)
    ant_el = 90. .- getproperty.(ant_traj, :beta)

    # align timestamps
    idx = [nearest_idx(ant_times, t) for t in traj_sats.times]
    keep = abs.(traj_sats.times .- ant_times[idx]) .<= (time_res / 2)

    # select sats
    df = traj_sats[keep,:]
    df.i_ant = idx[keep]
    sort!(df, [:sat, :times])

    sats_close = DataFrame(sat = String[], beam = Int[], t_start = DateTime[], 
                           t_stop = DateTime[], sep_min = Float64[], 
                           t_closest = DateTime[], n = Int[])
    for p in axes(ant_az, 2)
        az_p, el_p = view(ant_az, :, p), view(ant_el, :, p)
        sep = angular_separation.(df.azimuths, df.elevations, az_p[df.i_ant], 
                                  el_p[df.i_ant])
        rows = findall(<=(angle_detect), sep)
        isempty(rows) && continue

        hits = DataFrame(sat=df.sat[rows], times=df.times[rows], sep=sep[rows])
        # a new window starts wherever the row index jumps, or the satellite changes
        hits.win = cumsum([true; (diff(rows) .!= 1) .|
                          (@view(hits.sat[2:end]) .!= @view(hits.sat[1:end-1]))])

        w = combine(groupby(hits, [:sat, :win]), :times => first => :t_start,
                    :times => last => :t_stop, :sep => minimum => :sep_min,
                    [:times, :sep] => ((t, s) -> t[argmin(s)]) => :t_closest,
                    nrow => :n)

        w.beam .= p
        append!(sats_close, select(w, Not(:win)))
    end

    return sats_close
end



#### CHANGE SATELLITE FRAME ####



"""
    gcs_to_ecef(coords::Dict)

Geodetic (lat, lon in degrees; alt in metres) → ECEF.

"""
function gcs_to_ecef(coords::Dict{Symbol,<:Real})

    @assert :lat in keys(coords) "`:lat` required"
    @assert :lon in keys(coords) "`:lon` required"
    @assert :alt in keys(coords) "`:alt` required"

    return geodetic_to_ecef(deg2rad(coords[:lat]), deg2rad(coords[:lon]),
                            coords[:alt])
end



"""
    nwz_to_ecef_rotation(lat, lon)   # radians

North-West-Zenith → ECEF rotation. Columns are N̂, Ŵ, Ẑ(=Up).
Right-handed: N̂ x Ŵ = Ẑ. Consistent with the package's `SphereCoord`
convention, where `spher_to_cart_coord(α,β,r)` yields (N, W, Z).

"""
function nwz_to_ecef_rotation(lat::Real, 
    lon::Real)

    sφ, cφ = sincos(lat)
    sλ, cλ = sincos(lon)
    N̂ = SVector(-sφ*cλ, -sφ*sλ,  cφ)     # North
    Ŵ = SVector( sλ,    -cλ,      0.0)   # West = -East
    Ẑ = SVector( cφ*cλ,  cφ*sλ,   sφ)    # Zenith = Up
    
    return hcat(N̂, Ŵ, Ẑ)
end



"""
    ned_to_ecef_rotation(lat, lon)   # radians

North-East-Down(Nadir) → ECEF. Columns N̂, Ê, D̂. Right-handed (N̂×Ê=D̂).
Matches the satellite antenna frame: boresight (β=0) along Nadir.
"""
function ned_to_ecef_rotation(lat::Real, lon::Real)
    sφ, cφ = sincos(lat)
    sλ, cλ = sincos(lon)
    N̂ = SVector(-sφ*cλ, -sφ*sλ,  cφ)      # North
    Ê = SVector(-sλ,     cλ,      0.0)     # East
    D̂ = SVector(-cφ*cλ, -cφ*sλ, -sφ)      # Down = Nadir = -Up
    return hcat(N̂, Ê, D̂)
end



"""
    angular_separation()

Angular separation (deg) between two (az, el) directions, in degrees.

---
    angular_separation(a::SphereCoord,
                        b::SphereCoord)

Great-circle angular separation (degrees) between two SphereCoords.

"""
@inline function angular_separation(az1::T,
    el1::T,
    az2::T,
    el2::T) where T<:Real
    
    c = sind(el1) * sind(el2) + cosd(el1) * cosd(el2) * cosd(az1 - az2)

    return acosd(clamp(c, -one(c), one(c)))
end

function angular_separation(a::SphereCoord{T},
    b::SphereCoord{T}) where T
    
    va = spher_to_cart_coord(a.alpha, a.beta, one(T))
    vb = spher_to_cart_coord(b.alpha, b.beta, one(T))
    
    return acosd(clamp(dot(va, vb), -one(T), one(T)))
end



"""
    tel_dir_in_sat_frame(sat_coord, tel_lat, tel_lon, tel_alt)

Given the satellite's topocentric position relative to the telescope
(`sat_coord`, in `SphereCoord` convention) and the telescope's 
geodetic coordinates (degrees, degrees, metres), returns:

  * `tel_in_sat` : SphereCoord of the telescope in the satellite's antenna
                   (North-East-Nadir) frame,
  * `R_ned`      : NED→ECEF rotation at the satellite (for beam-avoidance),
  * `R_nwz`      : ENU→ECEF rotation at the telescope.

The satellite ECEF position is reconstructed from the telescope position plus
the topocentric range vector, then the satellite's geodetic position (hence its
local nadir) is obtained via `ecef_to_geodetic`, accounting for Earth curvature.

"""
function tel_dir_in_sat_frame(sat_coord::SphereCoord{T},
    tel_lat::Real, tel_lon::Real, tel_alt::Real) where T

    φ_t, λ_t = deg2rad(tel_lat), deg2rad(tel_lon)

    # telescope ECEF position
    r_tel = geodetic_to_ecef(φ_t, λ_t, tel_alt)
    R_nwz_t = nwz_to_ecef_rotation(φ_t, λ_t)

    # satellite ECEF position
    d_nwz_t = spher_to_cart_coord(sat_coord)
    r_sat = r_tel .+ R_nwz_t * d_nwz_t

    # satellite geodetic position → its local NED (nadir) frame
    φ_s, λ_s, _ = ecef_to_geodetic(r_sat)
    R_ned_s = ned_to_ecef_rotation(φ_s, λ_s)

    # direction satellite → telescope, expressed in the satellite NED frame
    d_ned_s = R_ned_s' * (r_tel .- r_sat)        # (North, East, Down)

    # antenna frame is North-East-Nadir
    α, β, _ = cart_to_sphe_coord(d_ned_s[1], d_ned_s[2], d_ned_s[3])

    return SphereCoord(T(α), T(β), sat_coord.r), R_ned_s, R_nwz_t
end



#### PYTHON VERSION ####



""" Load python packages. """
const sgp4_omm = PyNULL()
const sgp4_api = PyNULL()
const skyfield_api = PyNULL()
const io_module = PyNULL()

function __init__()
    copy!(sgp4_omm, pyimport("sgp4.omm"))
    copy!(sgp4_api, pyimport("sgp4.api"))
    copy!(skyfield_api, pyimport("skyfield.api"))
    copy!(io_module, pyimport("io"))
end



"""
    compute_sats_traj_py(sats_info::AbstractDataFrame,
                         start_time::DateTime,
                         stop_time::DateTime,
                         tel_coords::Dict{Symbol,<:Real},
                         time_res::Dates.Period;
                         save::Bool = false,
                         el_min::Real = 5.)

Compute the trajectories of the satellites which orbital information are in the
DataFrame 'sats_info' between the times 'start_time' and 'stop_time'. The
trajectories are stored in a DataFrame with columns :times, :sat, :elevations,
:azimuths, :ranges.

BEWARE: This function uses Python packages.
RE-BEWARE: The resulted satellites trajectories are expressed in terms of
azimuth and elevation angles, contrary to the convention taken in the rest of
the 'RadioMdl' package.

"""
function compute_sats_traj_py(sats_info::AbstractDataFrame,
    start_time::DateTime,
    stop_time::DateTime,
    tel_coords::Dict{Symbol,<:Real},
    time_res::Dates.Period;
    save::Bool = false,
    el_min::Real = 5.)
    
    @assert :lat in keys(tel_coords) "`:lat` required"
    @assert :lon in keys(tel_coords) "`:lon` required"
    @assert :alt in keys(tel_coords) "`:alt` required"
    @assert start_time < stop_time "`start_time` must precede `stop_time`"

    # create observer
    observer = skyfield_api.wgs84.latlon(tel_coords[:lat], tel_coords[:lon], 
                                         tel_coords[:alt])

    # load the time scale
    ts = skyfield_api.load.timescale()
    
    # convert CSV file for Python
    buff = IOBuffer()
    CSV.write(buff, sats_info)
    csv_file = String(take!(buff))
    f = io_module.StringIO(csv_file)

    # store the satellites as PyObjects
    list_sats = PyObject[]
    try
        for fields in sgp4_omm.parse_csv(f)
            sat = sgp4_api.Satrec()
            sgp4_omm.initialize(sat, fields)
            e = skyfield_api.EarthSatellite.from_satrec(sat, ts)
            e.name = get(fields, "OBJECT_NAME", nothing)
            push!(list_sats, e)
        end
    finally
        f.close()
    end
    
    # time of observation
    t0 = ts.utc(year(start_time), month(start_time), day(start_time),
                hour(start_time), minute(start_time), second(start_time))
    t1 = ts.utc(year(stop_time), month(stop_time), day(stop_time),
                hour(stop_time), minute(stop_time), second(stop_time))
    
    # storage of sats passing at telescope
    traj_sats = Tuple[]
    
    # overflight trajectories of the satellites seen by telescope
    for sat in list_sats
        # distance from sat to telescope over time
        diff_w = sat - observer
        
        # rise/culm/set times (t_w) and event codes (events_w)
        t_w, events_w = sat.find_events(observer, t0, t1, altitude_degrees=el_min)
        n_events = length(t_w)

        # get special cases of sats
        if n_events == 0
            # zero events = sat is either always visible or below horizon between t0 and t1
            alt_at_t0 = diff_w.at(t0).altaz()[1].degrees
            if alt_at_t0 < el_min
                continue
            end
            passes = [(t0, t1)]
        else
            passes = Tuple{PyObject,PyObject}[]
            idx = 1
            while idx <= n_events
                if events_w[idx] == 0
                    rise_time = t_w[idx]
                    idx += 1
                else
                    # first event isn't a rise -> satellite was already up at t0
                    rise_time = t0
                    # don't advance idx yet; this event may still be the set
                end

                set_pos = findfirst(i -> events_w[i] == 2, idx:n_events)
                if isnothing(set_pos)
                    # no set event found -> satellite is still up at t1
                    set_time = t1
                    idx = n_events + 1
                else
                    set_idx = idx - 1 + set_pos
                    set_time = t_w[set_idx]
                    idx = set_idx + 1
                end

                push!(passes, (rise_time, set_time))
            end
        end

        for (t_w_r, t_w_s) in passes
            # parse skyfield's own formatted string straight into a Julia DateTime
            dt_w_r = DateTime(t_w_r.utc_strftime("%Y-%m-%dT%H:%M:%S.%f")[1:end-3],
                              dateformat"yyyy-mm-ddTHH:MM:SS.sss")
            dt_w_s = DateTime(t_w_s.utc_strftime("%Y-%m-%dT%H:%M:%S.%f")[1:end-3],
                              dateformat"yyyy-mm-ddTHH:MM:SS.sss")
            
            # round the start to the nearest second (replaces pandas .round('1000ms'))
            dt_beg_sync = round(dt_w_r, time_res)
            
            # temporal grid, native Julia range instead of pd.date_range
            rise_to_set_range = dt_beg_sync:time_res:dt_w_s
            
            for t_rs in rise_to_set_range
                # build the skyfield Time directly from the Julia DateTime fields
                t_at = ts.utc(year(t_rs), month(t_rs), day(t_rs), hour(t_rs), 
                              minute(t_rs), second(t_rs))

                diff_t_rs = diff_w.at(t_at)
                ang_t_rs = diff_t_rs.altaz()
                
                els = ang_t_rs[1].degrees
                azs = ang_t_rs[2].degrees
                dis_w = ang_t_rs[3].m
                
                push!(traj_sats, (t_rs, sat.name, els, azs, dis_w))
            end
        end
    end
    
    # convert to DataFrames
    traj_sats = DataFrame(traj_sats, [:times, :sat, :elevations, :azimuths, :ranges])

    if save
        Arrow.write("sats_traj_$(start_time)_$(stop_time)_$(el_min)elmin_\
                     loaded$(now()).arrow", traj_sats)
    end

    return traj_sats
end


end # module

