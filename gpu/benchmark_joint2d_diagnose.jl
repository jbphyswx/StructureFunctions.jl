#!/usr/bin/env julia
"""
    benchmark_joint2d_diagnose.jl

Fairer joint-2D micro-benchmark: kernel-only timing, swapped A/B order, compile_cells sweep.

Use when value-route or exact-vs-max numbers look suspicious.

    julia --project=gpu gpu/benchmark_joint2d_diagnose.jl
    N_DIST=50 N_VAL=50 julia --project=gpu gpu/benchmark_joint2d_diagnose.jl
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using Printf: @printf
using Random: Random
using Statistics: Statistics
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges, joint2d_smem_max
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT

const _GPUExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
_GPUExt === nothing && error("StructureFunctionsKernelAbstractionsExt not loaded")

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

function _kernel_only_launch!(sft, backend, x, u, dist, value_bins, ws)
    SFC.reset_histogram!(ws)
    _GPUExt._launch_gpu_joint2d!(
        sft, backend, x, u, dist, value_bins;
        workspace = ws, synchronize = true,
    )
    return nothing
end

function _e2e_launch!(sft, backend, x, u, dist, value_bins, ws)
    SFC.gpu_calculate_structure_function_2d(
        sft, backend, x, u, dist, value_bins; workspace = ws,
    )
    return nothing
end

function _time_samples!(f, warmup::Int, repeat_::Int)
    for _ in 1:warmup
        f()
    end
    CUDA.synchronize()
    samples = Float64[]
    for _ in 1:repeat_
        push!(samples, @elapsed begin
            f()
            CUDA.synchronize()
        end)
    end
    return samples
end

function _report_samples(label::AbstractString, samples::Vector{Float64})
    med = Statistics.median(samples)
    @printf(
        "%-22s  med %7.3f ms  min %7.3f  max %7.3f  σ %6.3f ms\n",
        label, 1_000med, 1_000minimum(samples), 1_000maximum(samples), 1_000Statistics.std(samples),
    )
    return med
end

function _build_ws(backend, dist, value_bins; compile_cells=nothing)
    if compile_cells === nothing
        return SFC.GPUSFWorkspace(backend, dist, value_bins; kind = :joint2d)
    end
    return SFC.GPUSFWorkspace(
        backend, dist, value_bins;
        kind = :joint2d, joint2d_compile_cells = compile_cells,
    )
end

function _route_label(ws)
    dist_r = _GPUExt._joint2d_dist_route(ws.dist_bins)
    val_r = _GPUExt._joint2d_val_route(ws.val_plan)
    return "$(dist_r)/$(val_r)"
end

function main()
    CUDA.functional() || error("CUDA not functional")

    N = parse(Int, get(ENV, "N", "20000"))
    n_dist = parse(Int, get(ENV, "N_DIST", "20"))
    n_val_inner = parse(Int, get(ENV, "N_VAL", "20"))
    warmup = parse(Int, get(ENV, "WARMUP", "5"))
    repeat_ = parse(Int, get(ENV, "REPEAT", "15"))
    FT = Float32
    backend = CUDA.CUDABackend()

    dist = _dist_bins(n_dist, FT)
    val_typed = _value_inflinear(n_val_inner, FT)
    val_general = _value_general(n_val_inner, FT)
    nb2 = (length(dist) - 1) * (length(val_typed) - 1)

    println("=" ^ 72)
    println("joint 2D diagnose — kernel-only + e2e, swapped order")
    println("Device: ", CUDA.name(CUDA.device()))
    @printf("N=%d  n_dist=%d  n_val=%d  NB2=%d  warmup=%d  repeat=%d\n", N, n_dist, length(val_typed) - 1, nb2, warmup, repeat_)
    println("=" ^ 72)

    Random.seed!(42)
    x = rand(FT, 2, N) .* FT(50000)
    u = randn(FT, 2, N) .* FT(0.5)
    sft = SFT.L2SFType()

    ws_t = _build_ws(backend, dist, val_typed)
    ws_g = _build_ws(backend, dist, val_general)
    @printf("inflinear workspace: compile_cells=%d  route=%s\n", ws_t.joint2d_compile_cells, _route_label(ws_t))
    @printf("general workspace:   compile_cells=%d  route=%s\n", ws_g.joint2d_compile_cells, _route_label(ws_g))

    println("\n--- kernel-only (no host download) ---")
    for (label, val, ws) in (("inflinear→general", val_typed, ws_t), ("general→inflinear", val_general, ws_g))
        f = () -> _kernel_only_launch!(sft, backend, x, u, dist, val, ws)
        _report_samples(label, _time_samples!(f, warmup, repeat_))
    end
    # swap order block: build fresh workspaces so JIT order doesn't matter
    ws_t2 = _build_ws(backend, dist, val_typed)
    ws_g2 = _build_ws(backend, dist, val_general)
    s1 = _time_samples!(() -> _kernel_only_launch!(sft, backend, x, u, dist, val_general, ws_g2), warmup, repeat_)
    s2 = _time_samples!(() -> _kernel_only_launch!(sft, backend, x, u, dist, val_typed, ws_t2), warmup, repeat_)
    _report_samples("order: general first", s1)
    _report_samples("order: inflinear first", s2)

    println("\n--- end-to-end (gpu_calculate_structure_function_2d + download) ---")
    s_e2e_t = _time_samples!(() -> _e2e_launch!(sft, backend, x, u, dist, val_typed, ws_t), warmup, repeat_)
    s_e2e_g = _time_samples!(() -> _e2e_launch!(sft, backend, x, u, dist, val_general, ws_g), warmup, repeat_)
    t_med = _report_samples("inflinear e2e", s_e2e_t)
    g_med = _report_samples("general e2e", s_e2e_g)
    faster = t_med < g_med ? "inflinear" : "general"
    t_fast, t_slow = min(t_med, g_med), max(t_med, g_med)
    @printf("e2e delta: %s faster by %.1f%% (%.3f×)\n", faster, 100 * (t_slow - t_fast) / t_slow, t_slow / t_fast)

    println("\n--- compile_cells: exact NB2 vs max (inflinear, kernel-only) ---")
    ws_exact = _build_ws(backend, dist, val_typed)
    ws_max = _build_ws(backend, dist, val_typed; compile_cells=joint2d_smem_max())
    @printf("exact compile_cells=%d  max compile_cells=%d\n", ws_exact.joint2d_compile_cells, ws_max.joint2d_compile_cells)
    e_med = _report_samples("exact kernel-only", _time_samples!(
        () -> _kernel_only_launch!(sft, backend, x, u, dist, val_typed, ws_exact), warmup, repeat_,
    ))
    m_med = _report_samples("max kernel-only", _time_samples!(
        () -> _kernel_only_launch!(sft, backend, x, u, dist, val_typed, ws_max), warmup, repeat_,
    ))
    cf = min(e_med, m_med)
    cs = max(e_med, m_med)
    winner = e_med < m_med ? "exact" : "max"
    @printf("compile_cells delta: %s faster by %.1f%% (%.3f×)\n", winner, 100 * (cs - cf) / cs, cs / cf)

    println("=" ^ 72)
    println("If kernel-only ≈ but e2e differs → host download overhead.")
    println("If max ≪ exact kernel-only → lazy cNB2 codegen vs preload 4096 (investigate with ncu).")
    println("=" ^ 72)
end

main()
