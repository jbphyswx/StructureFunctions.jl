#!/usr/bin/env julia
"""
    benchmark_joint_value_route_ab.jl

A/B joint 2D: typed value digitize (`inflinear`) vs raw vector edges (`general`).

Same distance bins, same N, same inner grid — only the value-axis kernel route differs.
Use this to measure how much value digitize matters inside the full joint tiled kernel.

Run on GPU:

    julia --project=gpu gpu/benchmark_joint_value_route_ab.jl
    N=20000 N_VAL=50 julia --project=gpu gpu/benchmark_joint_value_route_ab.jl

For isolated digitize impact at larger B, also run:

    julia --project=gpu gpu/benchmark_value_axis_dispatch.jl
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using Printf: @printf
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT

const _GPUExt = Base.get_extension(SF, :StructureFunctionsGPUExt)
_GPUExt === nothing && error("StructureFunctionsGPUExt not loaded — use julia --project=gpu")

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

function _value_general(n_val_inner::Int, ::Type{FT}) where {FT}
    inner = collect(FT, range(-1, 2; length = n_val_inner + 1))
    return vcat(FT(-Inf), inner, FT(Inf))
end

function main()
    CUDA.functional() || error("CUDA not functional")

    N = parse(Int, get(ENV, "N", "20000"))
    n_dist = parse(Int, get(ENV, "N_DIST", "20"))
    n_val_inner = parse(Int, get(ENV, "N_VAL", "20"))
    warmup = parse(Int, get(ENV, "WARMUP", "2"))
    repeat_ = parse(Int, get(ENV, "REPEAT", "5"))
    check_parity = get(ENV, "CHECK_PARITY", "0") == "1"
    FT = Float32
    backend = CUDA.CUDABackend()

    dist = _dist_bins(n_dist, FT)
    val_typed = _value_inflinear(n_val_inner, FT)
    val_general = _value_general(n_val_inner, FT)
    n_val = length(val_typed) - 1
    NB2 = n_dist * n_val

    plan_typed = _GPUExt._joint2d_build_val_plan(backend, val_typed)
    plan_general = _GPUExt._joint2d_build_val_plan(backend, val_general)
    route_typed = _GPUExt._joint2d_val_route(plan_typed)
    route_general = _GPUExt._joint2d_val_route(plan_general)

    println("=" ^ 72)
    println("joint 2D value-route A/B (full kernel, not digitize microbench)")
    println("Device: ", CUDA.name(CUDA.device()))
    @printf(
        "N=%d  n_dist=%d  n_val=%d  NB2=%d  n_val_edges=%d\n",
        N, n_dist, n_val, NB2, n_val + 1,
    )
    @printf("dist_route=%s\n", _GPUExt._joint2d_dist_route(_GPUExt._gpu_normalize_bins(dist)))
    @printf("typed:   value_route=%s  plan=%s\n", route_typed, typeof(plan_typed))
    @printf("general: value_route=%s  plan=%s\n", route_general, plan_general === nothing ? "nothing" : typeof(plan_general))
    println("=" ^ 72)

    Random.seed!(42)
    x = rand(FT, 2, N) .* FT(50000)
    u = randn(FT, 2, N) .* FT(0.5)
    sft = SFT.L2SFType()

    # Typed InfPadded vs raw Vector — pass the same object to workspace and gpu_calculate.
    ws_typed = SFC.GPUSFWorkspace(backend, dist, val_typed; kind = :joint2d)
    ws_general = SFC.GPUSFWorkspace(backend, dist, val_general; kind = :joint2d)

    run_typed! = () -> SFC.gpu_calculate_structure_function_2d(
        sft, backend, x, u, dist, val_typed; workspace = ws_typed,
    )
    run_general! = () -> SFC.gpu_calculate_structure_function_2d(
        sft, backend, x, u, dist, val_general; workspace = ws_general,
    )

    t_typed = _bench(run_typed!, warmup, repeat_)
    t_general = _bench(run_general!, warmup, repeat_)

    @printf("inflinear (typed)   %8.3f ms\n", 1_000t_typed)
    @printf("general (vector)    %8.3f ms\n", 1_000t_general)
    faster = t_typed < t_general ? "inflinear" : "general"
    t_fast = min(t_typed, t_general)
    t_slow = max(t_typed, t_general)
    @printf(
        "delta: %s faster by %.1f%%  (%.2f×)\n",
        faster, 100 * (t_slow - t_fast) / t_slow, t_slow / t_fast,
    )

    if check_parity
        ref = SFC.calculate_structure_function(
            sft, x, u, dist, val_typed;
            backend = SFC.SerialBackend(), verbose = false, show_progress = false,
        )
        gpu_t = SFC.gpu_calculate_structure_function_2d(sft, backend, x, u, dist, val_typed; workspace = ws_typed)
        gpu_g = SFC.gpu_calculate_structure_function_2d(sft, backend, x, u, dist, val_general; workspace = ws_general)
        @printf("parity typed vs CPU:   counts %s  sums max err %.3e\n",
            gpu_t.counts == ref.counts ? "OK" : "MISMATCH",
            maximum(abs.(gpu_t.sums .- ref.sums)),
        )
        @printf("parity general vs CPU: counts %s  sums max err %.3e\n",
            gpu_g.counts == ref.counts ? "OK" : "MISMATCH",
            maximum(abs.(gpu_g.sums .- ref.sums)),
        )
    end

    println("=" ^ 72)
    return nothing
end

main()
