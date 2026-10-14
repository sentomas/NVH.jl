#************************************************#
# Author: SERIN THOMAS
# Date: 2026-Oct-14
# Description: Reproducible journal validation & benchmark script for the 
# Serin Number (Se) and 4-Zone Serin Method in NVH signal processing.
# License: MIT License
#************************************************#

using Random
using Statistics
using DataFrames
using BenchmarkTools
using NVH

Random.seed!(42) # Fixed seed for 100% reproducible journal results

const FS = 25600.0 # Standard 25.6 kHz NVH sampling rate
const N  = 4096    # 4096-sample FFT block size
t = (0:N-1) ./ FS

# 1. Generate 5 Canonical Physical NVH Signals
# Case 0: Dead/Idling Sensor with a single 1-bit ADC electronic glitch (1e-8 V)
sig_silent = zeros(Float64, N)
sig_silent[2048] = 1e-8 

# Case 1 (S1): Pure Harmonic Sine Wave (100 Hz shaft order)
sig_sine = 2.0 .* sin.(2π .* 100.0 .* t)

# Case 2 (S2): Stationary Gaussian Road Noise
sig_gauss = 0.5 .* randn(N)

# Case 3 (S3): Impulsive Bearing Rattle (Stationary noise + periodic impact spikes)
sig_rattle = 0.5 .* randn(N)
for idx in 500:500:N
    sig_rattle[idx] += 4.2
end

# Case 4 (S4): Severe Pothole Shock (High-energy transient burst)
sig_shock = 0.2 .* randn(N)
sig_shock[2000:2005] .+= [6.0, -9.5, 12.0, -8.0, 4.5, -2.0]

signals = [sig_silent, sig_sine, sig_gauss, sig_rattle, sig_shock]
labels  = [
    "0. Silent Channel (1-bit Glitch)",
    "1. Pure Harmonic (100 Hz Sine)",
    "2. Stationary Road Noise (Gaussian)",
    "3. Bearing Rattle (Periodic Impacts)",
    "4. Severe Pothole Strike (Shock)"
]

println("==========================================================================")
println(" TABLE 1: ANALYTICAL & EMPIRICAL VALIDATION OF THE SERIN NUMBER (Se)")
println("==========================================================================")
df_validation = analyze_nvh_stream(signals, labels)
show(stdout, df_validation; allcols=true, allrows=true)
println("\n")

println("==========================================================================")
println(" PROOF 2: COMPILER TYPE STABILITY (@code_warntype CHECK)")
println("==========================================================================")
# Verify return type is strictly inferred as Float64
inferred_type = Base.return_types(serin_number, (Vector{Float64},))[1]
println("Compiler Inferred Return Type for serin_number: ", inferred_type)

println("\n==========================================================================")
println(" PROOF 3: LLVM BARE-METAL EXECUTION BENCHMARK (4096 SAMPLES @ 25.6 kHz)")
println("==========================================================================")
print("Execution Time & Allocations: ")
@btime serin_number($sig_gauss)