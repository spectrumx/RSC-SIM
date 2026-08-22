
"""
    power_pattern_from_cut_file(file_path::String;
                                free_sp_imp::Real = 377,
                                verb::Bool = false)
                                
Yields the radiated power pattern, in W, of an antenna, times the radiation
efficiency already included in the `.cut` file containing co- and
cross-polarization E-field. 

Headers in the file are below a line starting with `Field`. It is composed of
the starting value of the declination angle, the step and number of samples of
the declination angle and the value of the azimuthal angle.

"""
function power_pattern_from_cut_file(file_path::String;
    free_sp_imp::Real = 377,
    verb::Bool = false)
    
    @assert occursin(".cut", file_path) "the power pattern file must be a .cut file"

    # parse file
    Es = readdlm(file_path)
    pattern = DataFrame(polar=Float64[], caz=Float64[], power=Float64[])
    k = 1
    dec_step = 0.
    while k <= size(Es,1)
        header_line = k + findfirst(x -> x == "Field", Es[k:end,:])[1]
        header = Es[header_line,:]
        verb && println(header)
        dec_start = header[1]
        dec_step = header[2]
        nb_dec = header[3]
        for t in 1:nb_dec
            polar = dec_start + (t-1)*dec_step
            θ = header[4]
            # power pattern, given in dBW, is the sum of the magnitude (squared
            # modulus) of the co- and cross-polarization complex electric field,
            # devided by twice the free-space impedance
            @inbounds u = sum(Es[header_line+t,1:4].^2)/(2*free_sp_imp)
            push!(pattern, [polar, θ, u])
        end
        k = header_line+nb_dec+1
    end
    res_dec = dec_step - round(dec_step)
    if res_dec == 0.
        decimal_places = 0
    else
        decimal_places = max(0, -floor(Int, log10(abs(res_dec))))
    end
    pattern[!,:polar] .= round.(pattern[!,:polar]; digits=decimal_places)
    
    # !!!!!!!!!!!!!!! THIS IS ONLY THE CASE WITH DANIEL'S FORMAT !!!!!!!!!!!!!!!
    
    @warn "This function assumes TICRA generated files"

    # check that polar ∈ [-180,180] and caz ∈ [0, 180[
    subset!(pattern, :polar => p -> -180. .<= p .<= 180., 
            :caz => a -> 0. .<= a .< 180.)

    # at this point, when the telescope is pointed at the horizon, caz = 0 gives
    # an horizontal slice, with polar > 0 oriented towards co-azimuth angles..

    # change interval so that caz ∈ [0,360[ and polar ∈ [0,180]
    pattern[pattern.polar .<= 0.,:caz] .+= 180.
    pattern[pattern.polar .< 0.,:polar] .*= -1.
    append!(pattern, [(;polar = zero(eltype(pattern.polar)), caz = i, 
                       power = pattern[pattern.polar .== 0.,:power][1])
                      for i in pattern[pattern.polar .== maximum(pattern.polar) .&& 
                                       pattern.caz .< 180.,:caz]])
    
    # move the origin of caz so that, when telescope points at the horizon, the
    # first slice (for the new caz = 0) is vertical with polar > 0 oriented towards
    # the ground.
    pattern[:,:caz] = mod.(pattern[!,:caz] .- 90., 360.)
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    sort!(pattern, [:caz, :polar])
    
    return pattern
end



"""
    read_VGOS_antenna_traj(file_path::String;
                           nb_targets_read::UnitRange{Int} = 1:Inf)

Yields the trajectory of a VGOS antenna.

"""
function read_VGOS_antenna_traj(file_path::String;
    kwds...)
    
    @assert occursin(".dat", file_path) "the trajectory file must be a .dat file"

    antenna_pos = CSV.read(file_path, DataFrame; delim=" ", ignorerepeated=true, 
                           header=false, kwds...)
    rename!(antenna_pos, names(antenna_pos)[end-1:end] .=> ["azimuths", "elevations"])
    antenna_pos[:,:times] .= DateTime.(antenna_pos.Column1) .+ 
                            Day.(antenna_pos.Column2 .- 1) .+ 
                            Hour.(antenna_pos.Column3) .+ Minute.(antenna_pos.Column4) .+
                            Second.(antenna_pos.Column5)
    select!(antenna_pos, [:times, :azimuths, :elevations])
    sort!(antenna_pos, :times)
    
    return antenna_pos
end

