#!/usr/bin/env julia
"""
    benchmark_batch_profile_fixed_x.jl

Phase timing for fixed-x batch accum paths on real tiled128 kernels.
Run on GPU (SLURM or ALLOW_CUDA_BENCH=1):

    PROFILE=reference N=20000 BATCH=8064 julia --project=gpu gpu/benchmark_batch_profile_fixed_x.jl
    PROFILE=quick julia --project=gpu gpu/benchmark_batch_profile_fixed_x.jl
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctionTypes as SFT, LinearBinEdges
using Printf: @printf

include(joinpath(@__DIR__, "batch_prototypes", "BatchPrototypes.jl"))
using .BatchPrototypes: BatchPrototypes as BP

function _resolve_N_B()
    profile = lowercase(strip(get(ENV, "PROFILE", "quick")))
    if profile == "reference" || haskey(ENV, "N") || haskey(ENV, "BATCH")
        N = parse(Int, get(ENV, "N", "20000"))
        batch_shape = BP.parse_batch_env((parse(Int, get(ENV, "BATCH", "8064")),))
        return N, batch_shape, profile
    end
    return 512, (64,), profile
end

function _sync!(backend)
    if backend isa CUDA.CUDABackend
        CUDA.synchronize()
    end
    return nothing
end

function main()
    FT = Float32
    N, batch_shape, profile = _resolve_N_B()
    B = prod(batch_shape)
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    NB = length(bin_edges.edges) - 1
    n_repeat = parse(Int, get(ENV, "N_REPEAT", profile == "quick" ? "3" : "1"))

    CUDA.functional() || error("CUDA required for phase profile")
    backend = CUDA.CUDABackend()
    println("device: ", CUDA.name(CUDA.device()))
    println("PROFILE=$profile  N=$N  B=$B  NB=$NB  n_repeat=$n_repeat")
    est = BP.estimate_batch_priv_bytes(N, B, NB, FT)
    @printf("partial_bytes=%.3f GiB  n_tile_blocks=%d\n", est.partial_bytes / (1024^3), est.n_priv)
    flush(stdout)

    x, u = BP.make_random_batch_problem(FT, N, batch_shape; fixed_x = true, seed = 1)
    sums_h = zeros(FT, NB, batch_shape...)
    counts_h = zeros(UInt32, NB, batch_shape...)
    ws = BP.BatchGPUWorkspace(backend, FT, N, B, NB; fixed_x = true)
    BP.upload_batch!(ws, backend, x, u)
    x_dev, u_dev = ws.x_dev, ws.u_dev

    n_strips = cld(B, BP.BATCH_TILED_STRIP_W)
    println("strip host launches per call: $n_strips (W=$(BP.BATCH_TILED_STRIP_W))")
    flush(stdout)

    # warmup
    BP.gpu_batch_tiled_fixed_x!(
        backend, sums_h, counts_h, x, u, sft, bin_edges;
        workspace = ws, download = false,
    )
    BP._launch_fused_fixed_x_in_kernel_timed!(
        backend, ws.sums_dev, ws.counts_dev, x_dev, u_dev, sft, N, B, bin_edges,
    )
    BP._launch_fused_fixed_x_direct_timed!(
        backend, ws.sums_dev, ws.counts_dev, x_dev, u_dev, sft, N, B, bin_edges,
    )
    BP._launch_fused_fixed_x_priv_timed!(
        backend, ws.sums_dev, ws.counts_dev, BP.ensure_partial_dev!(ws, backend), x_dev, u_dev, sft, N, B, bin_edges,
    )
    _sync!(backend)

    strip_times = Float64[]
    fused_times = Float64[]
    direct_times = Float64[]
    vram_fill = Float64[]
    vram_accum = Float64[]
    vram_merge = Float64[]

    for _ in 1:n_repeat
        BP.reset_output!(ws)
        t = @elapsed begin
            BP.gpu_batch_tiled_fixed_x!(
                backend, sums_h, counts_h, x, u, sft, bin_edges;
                workspace = ws, download = false,
            )
        end
        push!(strip_times, t)

        BP.reset_output!(ws)
        ph = BP._launch_fused_fixed_x_in_kernel_timed!(
            backend, ws.sums_dev, ws.counts_dev, x_dev, u_dev, sft, N, B, bin_edges,
        )
        push!(fused_times, ph.accum_s)

        BP.reset_output!(ws)
        ph = BP._launch_fused_fixed_x_direct_timed!(
            backend, ws.sums_dev, ws.counts_dev, x_dev, u_dev, sft, N, B, bin_edges,
        )
        push!(direct_times, ph.accum_s)

        BP.reset_output!(ws)
        ph = BP._launch_fused_fixed_x_priv_timed!(
            backend, ws.sums_dev, ws.counts_dev, BP.ensure_partial_dev!(ws, backend), x_dev, u_dev, sft, N, B, bin_edges,
        )
        push!(vram_fill, ph.fill_s)
        push!(vram_accum, ph.accum_s)
        push!(vram_merge, ph.merge_s)
    end

    t_strip = minimum(strip_times)
    t_fused = minimum(fused_times)
    t_direct = minimum(direct_times)
    t_vfill = minimum(vram_fill)
    t_vacc = minimum(vram_accum)
    t_vmerge = minimum(vram_merge)
    t_vram = t_vfill + t_vacc + t_vmerge

    println()
    println("--- phase timing (workspace reused, no D2H) ---")
    @printf("fused_1xlaunch         total=%.4fs\n", t_fused)
    @printf("strip_host_%dxlaunch   total=%.4fs  (%.3f ms/launch)\n",
        n_strips, t_strip, 1000 * t_strip / n_strips)
    @printf("direct_global_1x       accum=%.4fs\n", t_direct)
    @printf("vram_private_1x        total=%.4fs  fill=%.4fs  accum=%.4fs  merge=%.4fs\n",
        t_vram, t_vfill, t_vacc, t_vmerge)
    @printf("fused / strip          = %.3fx\n", t_fused / t_strip)
    @printf("fused / direct         = %.3fx\n", t_fused / t_direct)
    @printf("strip / direct         = %.3fx\n", t_strip / t_direct)
    @printf("strip / vram_total     = %.3fx\n", t_strip / t_vram)
    @printf("direct / vram_accum    = %.3fx  (accum only)\n", t_direct / t_vacc)
    @printf("fill+merge as %% vram   = %.2f%%\n", 100 * (t_vfill + t_vmerge) / t_vram)
    println("done")
end

main()
