#!/usr/bin/env julia
"""
    profile_joint2d.jl

Stable GPU workload for joint 2D profiling (Nsight Systems / manual timing).

Run on a GPU allocation:

    julia --project=gpu gpu/profile_joint2d.jl

Wrap with Nsight Systems:

    gpu/run_nsys_joint2d.sh
    NSYS_OUT=my_run gpu/run_nsys_joint2d.sh

Environment:

| Variable | Default | Meaning |
|----------|---------|---------|
| `N` | `20000` | number of points |
| `N_DIST` | `20` | distance bin count (inner) |
| `N_VAL` | `20` | value bin count (inner, before Inf padding) |
| `VALUE_ROUTE` | `inflinear` | `inflinear`, `general` (raw vector edges), or `both` |
| `WARMUP` | `2` | warmup iterations per scenario |
| `REPEAT` | `5` | timed iterations per scenario |
| `SEED` | `42` | RNG seed |
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using Printf: @printf
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT

const _GPUExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
_GPUExt === nothing && error("StructureFunctionsKernelAbstractionsExt not loaded — use julia --project=gpu")

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

function _dist_bins(n_dist::Int, ::Type{FT}) where {FT}
    return LogBinEdges(Vector{FT}(exp.(range(log(FT(1000)), log(FT(50000)); length = n_dist + 1))))
end

function _value_inflinear(n_val_inner::Int, ::Type{FT}) where {FT}
    return InfPaddedBinEdges(LinearBinEdges(range(FT(-1), FT(2); length = n_val_inner + 1)))
end

"""Raw edge vector → joint `val_plan === nothing` → `:general` kernel route."""
function _value_general(n_val_inner::Int, ::Type{FT}) where {FT}
    inner = collect(FT, range(-1, 2; length = n_val_inner + 1))
    return vcat(FT(-Inf), inner, FT(Inf))
end

function _scenario_value_bins(route::Symbol, n_val_inner::Int, ::Type{FT}) where {FT}
    route == :inflinear && return _value_inflinear(n_val_inner, FT), :inflinear
    route == :general && return _value_general(n_val_inner, FT), :general
    error("unknown VALUE_ROUTE=$route (use inflinear, general, or both)")
end

function _profile_one!(
    label::AbstractString,
    route::Symbol,
    backend::KA.Backend,
    sft,
    x,
    u,
    dist,
    n_val_inner::Int,
    FT::Type;
    warmup::Int,
    repeat_::Int,
)
    value_bins, route_sym = _scenario_value_bins(route, n_val_inner, FT)
    val_plan = _GPUExt._joint2d_build_val_plan(backend, value_bins)
    reported = _GPUExt._joint2d_val_route(val_plan)
    n_dist = length(dist) - 1
    n_val = length(value_bins) - 1
    NB2 = n_dist * n_val

    # Use typed value_bins (not _gpu_host_edge_vector) so workspace.val_plan matches kernel route.
    ws = SFC.GPUSFWorkspace(backend, dist, value_bins; kind = :joint2d)
    run! = () -> SFC.gpu_calculate_structure_function_2d(
        sft, backend, x, u, dist, value_bins; workspace = ws,
    )
    t = _bench(run!, warmup, repeat_)

    @printf(
        "%-12s  route=%-10s  plan=%-26s  %8.3f ms  [NB2=%d compile_cells=%d]\n",
        label,
        reported,
        val_plan === nothing ? "nothing (general)" : typeof(val_plan),
        1_000t,
        NB2,
        ws.joint2d_compile_cells,
    )
    return t
end

function main()
    CUDA.functional() || error("CUDA not functional — run inside srun --gres=gpu:1")

    N = parse(Int, get(ENV, "N", "20000"))
    n_dist = parse(Int, get(ENV, "N_DIST", "20"))
    n_val_inner = parse(Int, get(ENV, "N_VAL", "20"))
    warmup = parse(Int, get(ENV, "WARMUP", "2"))
    repeat_ = parse(Int, get(ENV, "REPEAT", "5"))
    seed = parse(Int, get(ENV, "SEED", "42"))
    route_env = lowercase(get(ENV, "VALUE_ROUTE", "inflinear"))
    FT = Float32
    backend = CUDA.CUDABackend()

    routes = if route_env == "both"
        (:inflinear, :general)
    else
        (Symbol(route_env),)
    end
    for r in routes
        r in (:inflinear, :general) ||
            error("VALUE_ROUTE must be inflinear, general, or both (got $route_env)")
    end

    dist = _dist_bins(n_dist, FT)
    dist_route = _GPUExt._joint2d_dist_route(_GPUExt._gpu_normalize_bins(dist))

    println("=" ^ 72)
    println("joint 2D profile workload")
    println("Device: ", CUDA.name(CUDA.device()))
    @printf(
        "N=%d  n_dist=%d  n_val_inner=%d  dist_route=%s  warmup=%d  repeat=%d\n",
        N, n_dist, n_val_inner, dist_route, warmup, repeat_,
    )
    println("=" ^ 72)

    Random.seed!(seed)
    x = rand(FT, 2, N) .* FT(50000)
    u = randn(FT, 2, N) .* FT(0.5)
    sft = SFT.L2SFType()

    times = Dict{Symbol, Float64}()
    for route in routes
        label = string(route)
        times[route] = _profile_one!(
            label, route, backend, sft, x, u, dist, n_val_inner, FT;
            warmup = warmup, repeat_ = repeat_,
        )
    end

    if length(routes) == 2
        t_fast = min(times[:inflinear], times[:general])
        t_slow = max(times[:inflinear], times[:general])
        @printf(
            "\ninflinear vs general: %.2f×  (%.3f ms vs %.3f ms)\n",
            t_slow / t_fast, 1_000times[:inflinear], 1_000times[:general],
        )
    end

    println("=" ^ 72)
    println("Nsight: gpu/run_nsys_joint2d.sh   (writes joint2d.nsys-rep in repo root)")
    return nothing
end

main()
