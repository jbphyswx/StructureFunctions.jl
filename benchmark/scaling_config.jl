"""
    scaling_config.jl

Shared problem definition for CPU thread scaling and GPU doc-asset benchmarks.
Included by `benchmark_worker.jl`, `benchmark_scaling.jl`, and `gpu/collect_benchmark_assets.jl`.

Environment overrides:
- `N_STRONG` — anchor N for CPU strong-scaling figure (default 4000)
- `N_LIST` — N values for **problem-size scaling** (1 GPU vs CPU, vary N)
- `N_SLICE` — N for slice-batch figure only (default 1000)
- `T_LIST` — T values for slice-batch figure at `N_SLICE`
- `SCALING_SEED` — RNG seed (default 42)

CPU strong/weak scaling (vary threads): `benchmark/benchmark_scaling.jl` only.
Multi-GPU strong/weak scaling: not implemented (`gpu/collect_multi_gpu_scaling.jl` stub).
"""

using Random: Random
using StructureFunctions: StructureFunctions as SF, StructureFunctionTypes as SFT

"""Anchor N; matches `N_STRONG` in `benchmark_scaling.jl` (CPU strong-scaling figure)."""
const SCALING_N_ANCHOR = parse(Int, get(ENV, "N_STRONG", "4000"))

"""N values for problem-size scaling (1 GPU + serial CPU, sweep N)."""
const SCALING_N_LIST = parse.(Int, split(get(ENV, "N_LIST", "4000,6000,8000,12000,16000,20000"), ","))

"""N for slice-batch doc figure (smaller N keeps CPU per-slice loop fast)."""
const SCALING_N_SLICE = parse(Int, get(ENV, "N_SLICE", "1000"))

"""T values for slice-batch figure at `SCALING_N_SLICE`."""
const SCALING_T_LIST = parse.(Int, split(get(ENV, "T_LIST", "1,2,4,8,16,32,64"), ","))

"""Distance bin edges — same as `benchmark_worker.jl` (20 bins)."""
const SCALING_BINS = collect(range(0.0, 1.5, length = 21))

"""RNG seed for reproducible synthetic fields."""
const SCALING_SEED = parse(Int, get(ENV, "SCALING_SEED", "42"))

"""Structure-function operator — longitudinal 2nd-order (same as CPU scaling harness)."""
const SCALING_SFT = SFT.LongitudinalSecondOrderStructureFunctionType()

"""
    scaling_synthetic_data(N, FT=Float64) -> (x_arr, u_arr)

Build 3D synthetic `(x, u)` arrays with shape `(3, N)` from `SCALING_SEED`.
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
