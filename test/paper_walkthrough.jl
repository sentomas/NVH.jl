using NVH
using DataFrames
using DelimitedFiles
using Statistics
using Random

Random.seed!(42)

# ==============================================================================
# PART 1: CREATE A REAL-WORLD ACCELEROMETER CSV FILE (real_bearing_run.csv)
# Sampling Rate: fs = 25,600 Hz | Total Samples: 20,480 (5 frames of N = 4096)
# ==============================================================================
fs = 25600.0
N  = 4096
t_block = (0:(N-1)) ./ fs

# Frame 1 (0.00s - 0.16s): Machine Off / Silent Sensor with a single 10^-8 V ADC glitch
f1 = zeros(Float64, N)
f1[2048] = 1e-8

# Frame 2 (0.16s - 0.32s): Electric Motor Smooth Sine Hum at 100 Hz (Amplitude = 2.0 g)
f2 = 2.0 .* sin.(2π * 100.0 .* t_block)

# Frame 3 (0.32s - 0.48s): Steady Random Road / Airflow Noise (Gaussian, sigma = 0.5 g)
f3 = 0.5 .* randn(N)

# Frame 4 (0.48s - 0.64s): Outer-Race Bearing Spall Rattle (Periodic 4.2g ticks every 500 samples)
f4 = 0.5 .* randn(N)
for idx in 250:500:N
    f4[idx:min(idx+3, N)] .+= 4.2
end

# Frame 5 (0.64s - 0.80s): Severe Pothole / Mechanical Crash (12.0g spike over 6 samples)
f5 = 0.3 .* randn(N)
f5[2000:2005] .+= 12.0

# Combine into a continuous 2-column sensor stream [Time_s, Accel_g] and save to CSV
all_accel = vcat(f1, f2, f3, f4, f5)
all_time  = (0:(length(all_accel)-1)) ./ fs
csv_path  = joinpath(@__DIR__, "real_bearing_run.csv")

open(csv_path, "w") do io
    println(io, "Time_s,Acceleration_g")
    writedlm(io, hcat(all_time, all_accel), ',')
end
println("✔ Saved example sensor recording to: $csv_path (20,480 rows)\n")

# ==============================================================================
# PART 2: READ THE CSV FILE & APPLY THE PAPER'S EQUATIONS STEP-BY-STEP
# ==============================================================================
raw_data, header = readdlm(csv_path, ',', Float64; header=true)
signal_stream = raw_data[:, 2] # Column 2: Acceleration_g

regime_names = [
    "Frame 1 (0.00-0.16s): Machine Off (1-bit ADC Glitch)",
    "Frame 2 (0.16-0.32s): Pure 100 Hz Motor Hum (Harmonic)",
    "Frame 3 (0.32-0.48s): Steady Road Noise (Gaussian)",
    "Frame 4 (0.48-0.64s): Bearing Spall Rattle (Impulsive)",
    "Frame 5 (0.64-0.80s): Severe Pothole Crash (Shock)"
]

println("==========================================================================")
println(" STEP-BY-STEP MATHEMATICAL WALKTHROUGH (FOLLOWING THE SADHANA MANUSCRIPT)")
println("==========================================================================\n")

frames = Vector{Vector{Float64}}(undef, 5)
epsilon_sq = 1e-6^2
lambda_se  = 3.0 * sqrt(2.0)

for k in 1:5
    idx_start = (k - 1) * N + 1
    idx_end   = k * N
    x = signal_stream[idx_start:idx_end]
    frames[k] = x

    # Paper Eq. (1): Classical Unregularized Variance, Kurtosis (K), and Crest Factor (CF)
    mu      = mean(x)
    var_raw = mean((x .- mu).^2)
    rms_raw = sqrt(var_raw)
    m4_raw  = mean((x .- mu).^4)
    peak    = maximum(abs.(x .- mu))

    K_raw  = m4_raw / (var_raw^2)
    CF_raw = peak / rms_raw
    Se_unregularized = (K_raw * CF_raw) / lambda_se

    # Paper Eq. (3): Regularized Serin Number (Se) via NVH.jl
    Se_reg = serin_number(x)
    res    = serin_classify_frame(x)

    println("--------------------------------------------------------------------------")
    println("▶ ", regime_names[k])
    println("--------------------------------------------------------------------------")
    println("  1. Raw Sensor Energy (RMS)          : ", round(rms_raw, sigdigits=4), " g")
    println("  2. Paper Eq.(1) Raw Kurtosis (K)    : ", round(K_raw, digits=3))
    println("  3. Paper Eq.(1) Raw Crest Factor    : ", round(CF_raw, digits=3))
    println("  4. WITHOUT Regularization (Trap!)   : Se_raw = ", round(Se_unregularized, digits=2))
    println("  5. WITH Paper Eq.(3) Regularization : Se     = ", round(Se_reg, digits=4))
    println("  6. 4-Zone Serin Method Decision     : Zone ", res.zone, " (", res.regime, ")")
    println("  7. Automated FFT Window & Estimator : :", res.optimal_window, " -> :", res.routed_method, "\n")
end

println("==========================================================================")
println(" FINAL SUMMARY TABLE GENERATED FROM CSV STREAM")
println("==========================================================================")
df_out = analyze_nvh_stream(frames, regime_names)
show(stdout, df_out; allcols=true, allrows=true)
println("\n")
