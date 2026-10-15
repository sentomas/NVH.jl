# NVH.jl

**Automated Non-Stationary Signal Classification and Spectral Window Routing via the Serin Number ($\mathrm{Se}$).**

`NVH.jl` is a high-performance Julia package for Noise, Vibration, and Harshness (NVH) engineering. It implements the dimensionless **Serin Number (`serin_number`)** and the **4-Zone Serin Method (`serin_classify_frame`)** to automatically route raw DAQ time-series frames to their mathematically optimal spectral windows (`:flattop`, `:hanning`, `:force_exp`, or `:rectangular`) while preventing silent-channel division-by-zero explosions via Bayesian noise-floor regularization.

## Reproducible Journal Validation & Benchmarks
```julia
using NVH
include("test/journal_validation.jl")









