
"""
    freq_range(freq_res::T,
               freq_center::T,
               bandwidth::T) where T

Yields the frequency range centered on freq_center and of length nb_freq_bins.
The range is created such as the grid frequencies are always integer multiples
of the frequency resolution 'freq_res'. Hence, some frequencies might be dropped.

"""
function freq_range(freq_res::T,
    freq_center::T,
    bandwidth::T) where T

    # rng = range(freq_res/2, bandwidth/2 - freq_res, length=nb_freq_bins)
    # return freq_center .+ rng
    # return range(freq_center - bandwidth/2 + freq_res, freq_center + bandwidth/2;
    #              length=nb_freq_bins)
    # return range(freq_center - bandwidth/2, freq_center + bandwidth/2;
            #  length=nb_freq_bins)
    nb_freq_bins = div(bandwidth, freq_res)
    f_c = round(Int, freq_center / freq_res)
    fmin = f_c - div(nb_freq_bins-1, 2)
    fmax = fmin + nb_freq_bins - 1
    
    return freq_res .* (fmin:fmax)
end



"""
    power_to_temperature(power::T,
                         bandwidth::T) where T

Yields the temperature of a source given its power in watt, bandwidth in hertz.
"""
function power_to_temperature(power::T,
    bandwidth::T) where T

    return power / (k_boltz * bandwidth)
end



"""
    temperature_to_power(temp::T,
                         bandwidth::T) where T

Yields the power of a source given its temperature in kelvin, bandwidth in
hertz.

"""
function temperature_to_power(temp::T,
    bandwidth::T) where T

    return k_boltz * bandwidth * temp
end



"""
    adc_noise_temperature(Vfs::T,
                          nb_bits::Int,
                          samp_rate::T;
                          instru_imp::T = impedance) where T

Yields the noise temperature of the ADC for a given full-scale voltage 'Vfs',
number of bits 'nb_bits' and number of frequency bins 'nb_freq_bins'. The 
instrument impedance 'instru_imp' is set to 50 Ohm by default. Note that the
result is the total noise temperature in each nb_freq_bins, and not the frequency
dependent noise temperature.

"""
function adc_noise_temperature(Vfs::T,#Full scale ADC voltage
    nb_bits::Int,
    samp_rate::T;
    instru_imp::T = impedance) where T
    
    # noise power of the ADC
    P_adc = Vfs^2 / (12*instru_imp) * 2^(-2. *nb_bits)

    return power_to_temperature(P_adc, samp_rate)
end



"""
    friis_noise_temp(stages::Tuple{T,T}...) where T<:AbstractFloat

Yields the total noise temperature of a receiver given the noise temperatures
and gains of each stage.

"""
function friis_noise_temp(stages::Tuple{T,T}...) where T<:AbstractFloat
    
    # Check that at least one stage is provided
    if isempty(stages)
        return zero(T)
    end

    T_total = zero(T)
    current_cumulative_gain = one(T)
    
    # Iterate through each stage (T_i, G_i)
    for i in eachindex(stages)
        T_i, G_i = stages[i]
        
        # The first stage adds its full noise temperature.
        # Subsequent stages' noise temperatures are divided by the gain 
        # of ALL preceding stages.
        T_total += T_i / current_cumulative_gain

        # Update the cumulative gain for the *next* stage calculation
        current_cumulative_gain *= G_i
    end

    return T_total
end



"""
    intrument_psd_stat(psd_inst_gain::T,
                       T_antenna::T,
                       T_instrument::T,
                       integration_samp::Real = 1) where T

Yields the power spectral density and its variance of an instrument given
the instrument gain 'psd_inst_gain', antenna temperature 'T_antenna',
instrument temperature 'T_instrument' and number of integration samples
'integration_samp'.

---
    instrument_psd_stat(psd_inst_gain::T,
                        T_antenna::AbstractArray{T},
                        T_instrument::AbstractArray{T},
                        integration_samp::Real = 1) where T

Uses 'instrument_psd_stat' on arrays.

"""
function instrument_psd_stat(psd_inst_gain::G,
    T_antenna::T,
    T_instrument::I,
    integration_samp::Real = 1) where {G,T,I}
    
    # calculate power spectral density
    psd = psd_inst_gain .* (T_antenna .+ T_instrument)
    var_psd = psd .* psd ./ integration_samp

   return psd, var_psd
end

# function instrument_psd_stat(psd_inst_gain::AbstractArray{T},
#     T_antenna::AbstractArray{T},
#     T_instrument::AbstractArray{T},
#     integration_samp::Real = 1) where T
    
#     return instrument_psd_stat.(psd_inst_gain, T_antenna, T_instrument, integration_samp)
# end

# function instrument_psd_stat(psd_inst_gain::T,
#     T_antenna::AbstractArray{T},
#     T_instrument::AbstractArray{T},
#     integration_samp::Real = 1) where T
    
#     return instrument_psd_stat.(psd_inst_gain, T_antenna, T_instrument, integration_samp)
# end

# function instrument_psd_stat(psd_inst_gain::AbstractArray{T},
#     T_antenna::T,
#     T_instrument::AbstractArray{T},
#     integration_samp::Real = 1) where T
    
#     return instrument_psd_stat.(psd_inst_gain, T_antenna, T_instrument, integration_samp)
# end

# function instrument_psd_stat(psd_inst_gain::AbstractArray{T},
#     T_antenna::AbstractArray{T},
#     T_instrument::T,
#     integration_samp::Real = 1) where T
    
#     return instrument_psd_stat.(psd_inst_gain, T_antenna, T_instrument, integration_samp)
# end