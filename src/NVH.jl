#************************************************#
# Author: SERIN THOMAS
# Date: 2026-Oct-14
# Description: Core module implementing the Serin Number (Se) and the 
# 4-Zone Serin Method for automated NVH signal classification and routing.
# License: MIT License
#************************************************#

module NVH

using Statistics
using DataFrames

export SERIN_CONSTANT, SerinConfig, serin_number, serin_classify_frame, analyze_nvh_stream

"""
    SERIN_CONSTANT

Fundamental normalization constant scaling Gaussian Kurtosis (3.0) and Sinusoidal Crest Factor (sqrt920).
"""
const SERIN_CONSTANT = 3.0 * sqrt(2.0)

"""
    SerinConfig

Configuration struct for the Serin Method 4-Zone NVH classification system.
"""
Base.@kwdef struct SerinConfig
    se_tonal::Float64      = 0.75
    se_stationary::Float64 = 3.50
    se_shock::Float64      = 35.00
    noise_floor::Float64   = 1e-6
end

"""
    serin_number(signal::AbstractVector{<:Real}; config::SerinConfig = SerinConfig())::Float64

Calculates the dimensionless Serin Number (Se) for a time-series vibration or acoustic block.
Uses Bayesian noise-floor regularization to prevent denominator explosion during sensor silence.
"""
function serin_number(signal::AbstractVector{<:Real}; config::SerinConfig = SerinConfig())::Float64
    n = length(signal)
    if n < 4
        throw(ArgumentError("Signal block must contain at least 4 samples to compute Kurtosis."))
    end

    mu = mean(signal)
    
    sum_sq = 0.0
    sum_fourth = 0.0
    peak_val = 0.0

    @inbounds @simd for i in eachindex(signal)
        dev = Float64(signal[i]) - mu
        abs_dev = abs(dev)
        if abs_dev > peak_val
            peak_val = abs_dev
        end
        dev2 = dev * dev
        sum_sq += dev2
        sum_fourth += dev2 * dev2
    end

    var_raw = sum_sq / n
    
    # Guard against dead-channel denominator trap
    if var_raw <= config.noise_floor^2
        return 0.0
    end

    rms_reg = sqrt(var_raw + config.noise_floor^2)
    
    kurtosis_val = (sum_fourth / n) / (rms_reg^4)
    crest_factor = peak_val / rms_reg

    return (kurtosis_val * crest_factor) / SERIN_CONSTANT
end

"""
    serin_classify_frame(signal::AbstractVector{<:Real}; config::SerinConfig = SerinConfig())

Evaluates a signal block, computes its Serin Number (Se), classifies it into 
Zones S1–S4, and prescribes the optimal spectral window and analysis method.
"""
function serin_classify_frame(signal::AbstractVector{<:Real}; config::SerinConfig = SerinConfig())
    se = serin_number(signal; config=config)

    if se == 0.0
        return (Se = se, Zone = "S0", Regime = "Sensor Silence / Noise Floor", Window = :none, Method = :SkipFrame)
    elseif se <= config.se_tonal
        return (Se = se, Zone = "S1", Regime = "Harmonic / Tonal", Window = :flattop, Method = :AmplitudeSpectrum)
    elseif se <= config.se_stationary
        return (Se = se, Zone = "S2", Regime = "Stationary Random", Window = :hanning, Method = :WelchPSD)
    elseif se <= config.se_shock
        return (Se = se, Zone = "S3", Regime = "Impulsive / Rattling", Window = :force_exp, Method = :HybridTransientPSD)
    else
        return (Se = se, Zone = "S4", Regime = "Severe Shock", Window = :rectangular, Method = :ShockEnergySpectrum)
    end
end

"""
    analyze_nvh_stream(signals::Vector{<:AbstractVector{<:Real}}, labels::Vector{String}; config::SerinConfig = SerinConfig())::DataFrame

Processes a batch of signal blocks and outputs a structured DataFrame for journal reporting.
"""
function analyze_nvh_stream(signals::Vector{<:AbstractVector{<:Real}}, labels::Vector{String}; config::SerinConfig = SerinConfig())::DataFrame
    results = DataFrame(
        FrameID = String[],
        RMS = Float64[],
        Kurtosis = Float64[],
        CrestFactor = Float64[],
        SerinNumber = Float64[],
        Zone = String[],
        Regime = String[],
        OptimalWindow = Symbol[],
        RoutedMethod = Symbol[]
    )

    for (sig, lbl) in zip(signals, labels)
        mu = mean(sig)
        devs = sig .- mu
        rms_val = std(sig; corrected=false)
        
        k_raw = rms_val > 1e-12 ? mean(devs.^4) / (rms_val^4) : Inf
        cf_raw = rms_val > 1e-12 ? maximum(abs.(devs)) / rms_val : Inf
        
        res = serin_classify_frame(sig; config=config)
        push!(results, (lbl, rms_val, round(k_raw, digits=3), round(cf_raw, digits=3), 
                        round(res.Se, digits=4), res.Zone, res.Regime, res.Window, res.Method))
    end

    return results
end

end # module