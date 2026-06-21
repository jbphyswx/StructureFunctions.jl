#!/usr/bin/env julia
"""
    benchmark_value_axis_dispatch.jl

Timed comparison of single-pass 2D GPU value-axis digitize plans on CUDA.

Run inside a GPU allocation:

    julia --project=gpu gpu/benchmark_value_axis_dispatch.jl
    N=30000 REPEAT=5 julia --project=gpu gpu/benchmark_value_axis_dispatch.jl

Scenarios (log distance bins, N≈20k–30k):
  - `inflinear_shared` — one InfPadded grid, O(1) inf-linear digitize
  - `inflinear_cols`   — eight InfPadded columns, O(1) per column
  - `linear_cols`      — eight LinearBinEdges columns
  - `vector_cols`      — eight raw edge vectors (binary search fallback)
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using Printf: @printf
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges

function _bench(f, warmup::Int, repeat_::Int)
    for _ in 1:warmup
        f()
    end
    CUDA.synchronize()
    elapsed = 0.0
    for _ in 1:repeat_
        elapsed += @elapsed begin
            f()
            CUDA.synchronize()
        end
    end
    return elapsed / repeat_
end

function _inflinear_template(n_inner::Int, ::Type{FT}) where {FT <: AbstractFloat}
    return InfPaddedBinEdges(LinearBinEdges(range(FT(-1), FT(2); length = n_inner + 1)))
end

function _linear_template(n_inner::Int, ::Type{FT}) where {FT <: AbstractFloat}
    return LinearBinEdges(range(FT(-1), FT(2); length = n_inner + 1))
end

function _vector_template(n_inner::Int, ::Type{FT}) where {FT <: AbstractFloat}
    edges = collect(FT, range(-1, 2; length = n_inner + 1))
    return vcat(FT(-Inf), edges, FT(Inf))
end

function _scenario_value_bins(name::AbstractString, n_inner::Int, ::Type{FT}) where {FT}
    if name == "inflinear_shared"
        return _inflinear_template(n_inner, FT)
    elseif name == "inflinear_cols"
        tpl = _inflinear_template(n_inner, FT)
        return ntuple(_ -> tpl, 6)
    elseif name == "linear_cols"
        tpl = _linear_template(n_inner, FT)
        return ntuple(_ -> tpl, 6)
    elseif name == "vector_cols"
        tpl = _vector_template(n_inner, FT)
        return ntuple(_ -> copy(tpl), 6)
    else
        error("unknown scenario: $name")
    end
end

function main()
    CUDA.functional() || error("CUDA not functional — run inside srun --gres=gpu:1")

    N = parse(Int, get(ENV, "N", "20000"))
    n_inner = parse(Int, get(ENV, "N_INNER", "50"))
    warmup = parse(Int, get(ENV, "WARMUP", "2"))
    repeat_ = parse(Int, get(ENV, "REPEAT", "5"))
    FT = get(ENV, "FT", "Float32") == "Float64" ? Float64 : Float32
    ka_backend = CUDA.CUDABackend()
    gpu_backend = SF.GPUBackend(ka_backend)

    println("=" ^ 72)
    println("Single-pass 2D value-axis dispatch benchmark (CUDA)")
    println("Device: ", CUDA.name(CUDA.device()))
    @printf("N=%d  n_inner=%d  dtype=%s  warmup=%d  repeat=%d\n", N, n_inner, FT, warmup, repeat_)
    println("=" ^ 72)

    Random.seed!(42)
    dist_vec = LogBinEdges(Vector{FT}(exp.(range(log(FT(1000)), log(FT(50000)); length = 51))))
    x = rand(FT, 2, N) .* FT(50000)
    u = randn(FT, 2, N) .* FT(0.5)

    scenarios = ("inflinear_shared", "inflinear_cols", "linear_cols", "vector_cols")
    for name in scenarios
        value_bins = _scenario_value_bins(name, n_inner, FT)
        vb0 = value_bins isa Tuple ? value_bins[1] : value_bins
        n_val = length(vb0) - 1
        n_dist = length(dist_vec) - 1

        ws = SFC.GPUSFWorkspace(ka_backend, dist_vec, value_bins; kind = :single_pass_2d)
        plan = ws.val_plan
        sums = zeros(FT, 6, n_dist, n_val)
        counts = zeros(UInt32, 6, n_dist, n_val)

        run! = () -> SFC.gpu_calculate_structure_functions_single_pass_2d!(
            sums, counts, ka_backend, x, u, dist_vec, value_bins;
            workspace = ws,
        )
        t = _bench(run!, warmup, repeat_)
        pairs = N * (N - 1) ÷ 2
        @printf(
            "%-18s  plan=%-24s  %.3f ms  (%.2e pairs/s)\n",
            name,
            typeof(plan),
            1_000t,
            pairs / t,
        )
    end
    println("=" ^ 72)
end

main()
