"""
    scaling_config.jl

The problem the scaling benchmarks time, included by `benchmark_worker.jl` and `gpu/collect_benchmark_assets.jl`.

Environment overrides:
- `N_STRONG` — anchor N, the CPU strong-scaling problem size (default 4000)
- `N_LIST` — N values of the GPU problem-size figure
- `N_SLICE` — N of the GPU slice-batch figure (default 1000)
- `T_LIST` — slice counts of the GPU slice-batch figure
- `SCALING_SEED` — RNG seed (default 42)
"""

using Random: Random
using StructureFunctions: StructureFunctionTypes as SFT

"""Anchor N; the `N_STRONG` of `benchmark_scaling.jl`."""
const SCALING_N_ANCHOR = parse(Int, get(ENV, "N_STRONG", "4000"))

"""N values of the GPU problem-size figure (one GPU against the serial CPU)."""
const SCALING_N_LIST = parse.(Int, split(get(ENV, "N_LIST", "4000,6000,8000,12000,16000,20000"), ","))

"""N of the GPU slice-batch figure."""
const SCALING_N_SLICE = parse(Int, get(ENV, "N_SLICE", "1000"))

"""Slice counts of the GPU slice-batch figure."""
const SCALING_T_LIST = parse.(Int, split(get(ENV, "T_LIST", "1,2,4,8,16,32,64"), ","))

"""Distance bin edges, 20 bins."""
const SCALING_BINS = collect(range(0.0, 1.5, length = 21))

"""RNG seed of the synthetic fields."""
const SCALING_SEED = parse(Int, get(ENV, "SCALING_SEED", "42"))

"""The operator timed: the longitudinal second-order structure function."""
const SCALING_SFT = SFT.LongitudinalSecondOrderStructureFunctionType()

"""
    scaling_synthetic_data(N, FT=Float64) -> (x_arr, u_arr)

Uniform random `(3, N)` positions and fields from `SCALING_SEED`.
"""
function scaling_synthetic_data(N::Int, ::Type{FT} = Float64) where {FT}
    Random.seed!(SCALING_SEED)
    x_arr = Matrix{FT}(undef, 3, N)
    u_arr = Matrix{FT}(undef, 3, N)
    for d in 1:3
        x_arr[d, :] .= rand(FT, N)
        u_arr[d, :] .= rand(FT, N)
    end
    return x_arr, u_arr
end

"""
    scaling_bins(FT) -> Vector

Distance bin edges cast to element type `FT`.
"""
scaling_bins(::Type{FT}) where {FT} = collect(FT, SCALING_BINS)
