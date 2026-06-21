#!/usr/bin/env julia
"""
    benchmark_batch_prototypes.jl

Fast-by-default Phase 0 benchmark. Compares `gpu_batch` prototype vs production slice loop.

**Default (no env): ~1–2 min on GPU** — N=512, BATCH=64, full production baseline (64 kernels).

Profiles (`PROFILE=...`):
  `quick`     — defaults; small problem, parity + all timings
  `scaled`    — N=4096 BATCH=512; ~5 min, trend at medium scale
  `reference` — your N/B; sampled production extrapolation (~8 min at N=20k B=8064)
  `reference_full` — exact B×production (**~1 h** at N=20k B=8064; opt-in only)

    julia --project=gpu gpu/benchmark_batch_prototypes.jl
    PROFILE=scaled julia --project=gpu gpu/benchmark_batch_prototypes.jl
    PROFILE=reference N=20000 BATCH=8064 julia --project=gpu gpu/benchmark_batch_prototypes.jl
    PROFILE=reference_full N=20000 BATCH=8064 julia --project=gpu gpu/benchmark_batch_prototypes.jl
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, LinearBinEdges
using Printf: @printf

include(joinpath(@__DIR__, "benchmark_scaling_helpers.jl"))
include(joinpath(@__DIR__, "batch_prototypes", "BatchPrototypes.jl"))
using .BatchPrototypes: BatchPrototypes as BP, BatchVariantConfig

const _PROFILE_DEFAULTS = Dict(
    :quick => (N = 512, batch = (64,)),
    :scaled => (N = 4096, batch = (512,)),
)

function _large_scale(N::Int, batch_shape)
    return N >= 4096 || prod(batch_shape) >= 512
end

function _resolve_profile(N::Int, batch_shape)
    raw = lowercase(strip(get(ENV, "PROFILE", "")))
    if isempty(raw)
        return _large_scale(N, batch_shape) ? :reference : :quick
    end
    sym = Symbol(raw)
    sym in (:quick, :scaled, :reference, :reference_full) ||
        error("PROFILE must be quick, scaled, reference, or reference_full; got $(repr(raw))")
    return sym
end

function _resolve_problem_size(profile::Symbol)
    if haskey(ENV, "N") || haskey(ENV, "BATCH")
        N = parse(Int, get(ENV, "N", string(_PROFILE_DEFAULTS[:quick].N)))
        batch_shape = BP.parse_batch_env(_PROFILE_DEFAULTS[:quick].batch)
        return N, batch_shape
    end
    defs = get(_PROFILE_DEFAULTS, profile, _PROFILE_DEFAULTS[:quick])
    return defs.N, defs.batch
end

function _run_production_slice!(
    backend,
    x,
    u,
    bin_edges,
    sft,
    sums,
    counts,
    ws;
    fixed_x::Bool,
    batch_indices::AbstractVector{Int},
    label::String = "",
)
    bd = BP.batch_dims(u)
    B = prod(bd)
    sums_f = reshape(sums, size(sums, 1), :)
    counts_f = reshape(counts, size(counts, 1), :)
    for b in batch_indices
        if fixed_x
            x_slice = x
            u_slice = BP.batch_field_slice(u, b)
        else
            x_slice = BP.batch_field_slice(x, b)
            u_slice = BP.batch_field_slice(u, b)
        end
        x_dev = CUDA.cu(x_slice)
        u_dev = CUDA.cu(u_slice)
        res = SFC.gpu_calculate_structure_function(
            sft, backend, x_dev, u_dev, bin_edges;
            return_sums_and_counts = true,
            workspace = ws,
        )
        sums_f[:, b] .= res.sums
        counts_f[:, b] .= res.counts
    end
    gpu_sync!(backend)
    return B
end

function _production_batch_indices(B::Int, n_sample::Int)
    n = min(n_sample, B)
    return round.(Int, range(1, B; length = n))
end

function _parse_geometry_env(profile::Symbol)
    default = "both"
    s = lowercase(strip(get(ENV, "GEOMETRY", default)))
    if s == "both"
        return (:fixed_x, :varying_x)
    elseif s == "fixed_x"
        return (:fixed_x,)
    elseif s == "varying_x"
        return (:varying_x,)
    else
        error("GEOMETRY must be fixed_x, varying_x, or both; got $(repr(s))")
    end
end

function _geometry_cases(geo_syms, x_fix, u_fix, x_var, u_var)
    out = NamedTuple[]
    for sym in geo_syms
        if sym === :fixed_x
            push!(out, (geo = BP.FixedX, x = x_fix, u = u_fix, label = "fixed_x"))
        else
            push!(out, (geo = BP.VaryingX, x = x_var, u = u_var, label = "varying_x"))
        end
    end
    return out
end

function _gpu_backend_if_available()
    if CUDA.functional()
        return CUDA.CUDABackend(), true
    end
    return KA.CPU(), false
end

function _time_variant!(
    cfg::BatchVariantConfig,
    sums,
    counts,
    x,
    u,
    sft,
    bin_edges;
    backend = nothing,
    n_warmup::Int = 1,
    n_repeat::Int = 3,
    max_vram::Int = 0,
    workspace = nothing,
    download::Bool = true,
)
    for _ in 1:n_warmup
        fill!(sums, zero(eltype(sums)))
        fill!(counts, zero(eltype(counts)))
        BP.run_variant!(
            cfg, sums, counts, x, u, sft, bin_edges;
            backend = backend, max_vram = max_vram,
            workspace = workspace, download = download,
        )
        workspace !== nothing && backend !== nothing && gpu_sync!(backend)
    end
    times = Float64[]
    for _ in 1:n_repeat
        fill!(sums, zero(eltype(sums)))
        fill!(counts, zero(eltype(counts)))
        workspace !== nothing && BP.reset_output!(workspace)
        t = BP.run_variant!(
            cfg, sums, counts, x, u, sft, bin_edges;
            backend = backend, max_vram = max_vram,
            workspace = workspace, download = download,
        )
        workspace !== nothing && backend !== nothing && gpu_sync!(backend)
        push!(times, t)
    end
    return minimum(times)
end

"""End-to-end (alloc + H2D + kernels + D2H each call) vs kernel-only (reused workspace)."""
function _time_gpu_e2e_and_kernel!(
    cfg::BatchVariantConfig,
    sums,
    counts,
    x,
    u,
    sft,
    bin_edges,
    backend,
    FT;
    n_warmup::Int,
    n_repeat::Int,
    max_vram::Int,
)
    t_e2e = _time_variant!(
        cfg, sums, counts, x, u, sft, bin_edges;
        backend = backend, n_warmup = n_warmup, n_repeat = n_repeat,
        max_vram = max_vram, workspace = nothing, download = true,
    )
    ws = BP.BatchGPUWorkspace(backend, FT, size(x, 2), prod(BP.batch_dims(u)), length(bin_edges.edges) - 1;
        fixed_x = cfg.geometry === BP.FixedX)
    BP.upload_batch!(ws, backend, x, u)
    t_kernel = _time_variant!(
        cfg, sums, counts, x, u, sft, bin_edges;
        backend = backend, n_warmup = n_warmup, n_repeat = n_repeat,
        max_vram = max_vram, workspace = ws, download = false,
    )
    BP.download_batch!(sums, counts, ws)
    return t_e2e, t_kernel
end

function main()
    FT = Float32
    profile_hint = Symbol(lowercase(strip(get(ENV, "PROFILE", "quick"))))
    profile_for_size = profile_hint in keys(_PROFILE_DEFAULTS) ? profile_hint : :quick
    N, batch_shape = _resolve_problem_size(profile_for_size)
    profile = _resolve_profile(N, batch_shape)
    if !haskey(ENV, "N") && !haskey(ENV, "BATCH")
        N, batch_shape = _resolve_problem_size(profile)
    end

    n_warmup = parse(Int, get(ENV, "WARMUP", profile === :quick ? "1" : "1"))
    n_repeat = parse(Int, get(ENV, "N_REPEAT", profile === :quick ? "3" : "1"))
    max_vram = parse(Int, get(ENV, "MAX_VRAM", "0"))
    geo_syms = _parse_geometry_env(profile)

    prod_sample = if profile === :reference_full
        0  # 0 => full B
    elseif haskey(ENV, "PROD_SAMPLE")
        parse(Int, ENV["PROD_SAMPLE"])
    elseif profile === :reference
        8
    else
        0
    end

    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    lp = BP.linear_bin_params(bin_edges)
    NB = lp.n_bins - 1

    x_fix, u_fix = BP.make_random_batch_problem(FT, N, batch_shape; fixed_x = true)
    x_var, u_var = BP.make_random_batch_problem(FT, N, batch_shape; fixed_x = false)

    gpu_backend, cuda_ok = _gpu_backend_if_available()
    large = _large_scale(N, batch_shape)
    B = prod(batch_shape)
    geom_cases = _geometry_cases(geo_syms, x_fix, u_fix, x_var, u_var)

    if cuda_ok && hasproperty(CUDA, :name)
        println("device: ", CUDA.name(CUDA.device()))
    end
    println("=== batch prototypes Phase 0 ===")
    println("prototype: $(BP.batch_prototype_variant())")
    println("  $(strip(BP.batch_prototype_variant_description()))")
    println("PROFILE=$profile  N=$N  batch_shape=$batch_shape  B=$B  NB=$NB")
    if :fixed_x in geo_syms
        est = BP.estimate_batch_priv_bytes(N, B, NB, FT)
        @printf("  fused partial_bytes=%.3f GiB  n_tile_blocks=%d  per_block_slab=%.2f MiB  MAX_VRAM=%d\n",
            est.partial_bytes / (1024^3), est.n_priv, est.per_block_slab_bytes / (1024^2), max_vram)
    end
    println("cuda=$cuda_ok  warmup=$n_warmup  repeat=$n_repeat  geometry=$(join(string.(geo_syms), ","))")
    if profile === :quick
        println("mode: fast default (~1–2 min GPU); cpu_slice parity included")
    elseif profile === :scaled
        println("mode: medium scale trend (~5 min GPU)")
    elseif profile === :reference
        println("mode: reference N/B; production baseline = sample $prod_sample slices × extrapolate")
        println("      (set PROFILE=reference_full for exact B×production — ~1 h at N=20k B=8064)")
    else
        println("mode: REFERENCE FULL — exact B×production per geometry; may take ~1 h")
    end
    large && println("cpu_slice gold: skipped (days at this scale; use PROFILE=quick for parity)")
    flush(stdout)

    if large && get(ENV, "CHECK_PARITY", "0") == "1"
        error("CHECK_PARITY=1 at large scale would take days. Use PROFILE=quick instead.")
    end

    if cuda_ok
        println("--- GPU warmup (one launch per geometry) ---")
        flush(stdout)
        for case in geom_cases
            geo, x, u, label = case.geo, case.x, case.u, case.label
            bd = BP.batch_dims(u)
            sums_w = zeros(FT, NB, bd...)
            counts_w = zeros(UInt32, NB, bd...)
            cfg = BatchVariantConfig("gpu_batch_warmup", geo, :gpu_batch_fused, 0)
            tw = @elapsed _time_variant!(
                cfg, sums_w, counts_w, x, u, sft, bin_edges;
                backend = gpu_backend, n_warmup = 0, n_repeat = 1, max_vram = max_vram,
            )
            @printf("  gpu_batch_fused %-10s warmup=%.4fs\n", label, tw)
            flush(stdout)
        end
    end

    if profile === :quick
        println("--- parity vs cpu_slice gold ---")
        flush(stdout)
        results = BP.run_parity_suite(
            x_fix, u_fix, x_var, u_var, sft, bin_edges;
            backend = cuda_ok ? gpu_backend : nothing,
            max_vram = max_vram,
        )
        BP.print_parity_results(results)
    end

    ws = cuda_ok ? SFC.GPUSFWorkspace(gpu_backend, bin_edges) : nothing
    println()
    println("--- timed variants ---")
    flush(stdout)

    for case in geom_cases
        geo, x, u, label = case.geo, case.x, case.u, case.label
        bd = BP.batch_dims(u)
        sums = zeros(FT, NB, bd...)
        counts = zeros(UInt32, NB, bd...)
        fixed_x = geo === BP.FixedX

        if profile === :quick
            cfg = BatchVariantConfig("cpu_slice", geo, :cpu_slice, 0)
            t_slice = _time_variant!(
                cfg, sums, counts, x, u, sft, bin_edges;
                n_warmup = n_warmup, n_repeat = n_repeat,
            )
            @printf("  cpu_slice_gold %-8s  total=%.4fs\n", label, t_slice)
            flush(stdout)

            sums_b = zeros(FT, NB, bd...)
            counts_b = zeros(UInt32, NB, bd...)
            batch_cfg = BatchVariantConfig("cpu_batch", geo, :cpu_batch, 32)
            t_batch = _time_variant!(
                batch_cfg, sums_b, counts_b, x, u, sft, bin_edges;
                n_warmup = n_warmup, n_repeat = n_repeat,
            )
            ok, _, _ = BP.check_parity(sums, counts, sums_b, counts_b)
            @printf("  cpu_batch %-11s  total=%.4fs  speedup_vs_cpu_slice=%.2fx  ok=%s\n",
                label, t_batch, t_slice / t_batch, ok)
            flush(stdout)
        end

        if cuda_ok
            t_prod = NaN
            t_prod_note = ""
            if prod_sample == 0
                sums_p = zeros(FT, NB, bd...)
                counts_p = zeros(UInt32, NB, bd...)
                idx = collect(1:B)
                println("  gpu_production_slice $label  full B=$B ...")
                flush(stdout)
                t_prod = run_timed_gpu(
                    () -> _run_production_slice!(
                        gpu_backend, x, u, bin_edges, sft, sums_p, counts_p, ws;
                        fixed_x = fixed_x, batch_indices = idx, label = label,
                    ),
                    gpu_backend; warmup = n_warmup,
                )
                t_prod_note = "full"
            else
                idx = _production_batch_indices(B, prod_sample)
                sums_p = zeros(FT, NB, bd...)
                counts_p = zeros(UInt32, NB, bd...)
                println("  gpu_production_slice $label  sample $(length(idx))/$B batches ...")
                flush(stdout)
                t_sample = run_timed_gpu(
                    () -> _run_production_slice!(
                        gpu_backend, x, u, bin_edges, sft, sums_p, counts_p, ws;
                        fixed_x = fixed_x, batch_indices = idx, label = label,
                    ),
                    gpu_backend; warmup = n_warmup,
                )
                t_prod = t_sample * (B / length(idx))
                t_prod_note = "extrap_from_$(length(idx))"
            end
            @printf("  gpu_production_slice %-8s  total=%.4fs (%s)  per_batch=%.3fms\n",
                label, t_prod, t_prod_note, 1000 * t_prod / B)
            if profile === :quick
                ok_slice_prod, Δ_sp, ce_sp = BP.check_parity(sums, counts, sums_p, counts_p)
                @printf("  production_vs_cpu_slice %-8s  max|Δsums|=%.4g  counts=%s  ok=%s\n",
                    label, Δ_sp, ce_sp, ok_slice_prod)
            end
            flush(stdout)

            # Fused tiled (integration candidate)
            sums_f = zeros(FT, NB, bd...)
            counts_f = zeros(UInt32, NB, bd...)
            fused_cfg = BatchVariantConfig("gpu_batch_fused", geo, :gpu_batch_fused, 0)
            t_fused_e2e, t_fused_kernel = _time_gpu_e2e_and_kernel!(
                fused_cfg, sums_f, counts_f, x, u, sft, bin_edges, gpu_backend, FT;
                n_warmup = 0, n_repeat = n_repeat, max_vram = max_vram,
            )
            ok_prod = missing
            ok_f = missing
            if profile === :quick
                ok_prod, Δ_prod, ce_prod = BP.check_parity(sums_p, counts_p, sums_f, counts_f)
                @printf("  gpu_fused_vs_production %-8s  max|Δsums|=%.4g  counts=%s  ok=%s\n",
                    label, Δ_prod, ce_prod, ok_prod)
                flush(stdout)
                # Integration gate on CUDA: production 2D tiled, not cpu_slice 3D gold.
                ok_f = cuda_ok ? ok_prod : BP.check_parity(sums, counts, sums_f, counts_f)[1]
            elseif !large
                sums_ref = zeros(FT, NB, bd...)
                counts_ref = zeros(UInt32, NB, bd...)
                ref_cfg = BatchVariantConfig("cpu_batch_ref", geo, :cpu_batch, 32)
                BP.run_variant!(ref_cfg, sums_ref, counts_ref, x, u, sft, bin_edges)
                ok_f, Δ_f, ce_f = BP.check_parity(sums_ref, counts_ref, sums_f, counts_f)
                @printf("  gpu_fused_vs_cpu_batch %-8s  max|Δsums|=%.4g  counts=%s\n", label, Δ_f, ce_f)
                flush(stdout)
            end
            ok_label = ok_f === missing ? "parity_skipped" : string(ok_f)
            @printf("  gpu_batch_fused %-8s  e2e=%.4fs  kernel_only=%.4fs  speedup_vs_production(e2e)=%.2fx  ok=%s\n",
                label, t_fused_e2e, t_fused_kernel, t_prod / t_fused_kernel, ok_label)
            flush(stdout)

            # v1 strip baseline (fixed-x only meaningful)
            if fixed_x
                sums_g = zeros(FT, NB, bd...)
                counts_g = zeros(UInt32, NB, bd...)
                strip_cfg = BatchVariantConfig("gpu_batch_strip_host", geo, :gpu_batch_tiled, 0)
                t_strip_e2e, t_strip_kernel = _time_gpu_e2e_and_kernel!(
                    strip_cfg, sums_g, counts_g, x, u, sft, bin_edges, gpu_backend, FT;
                    n_warmup = 0, n_repeat = n_repeat, max_vram = max_vram,
                )
                ok_s = profile === :quick ? BP.check_parity(sums, counts, sums_g, counts_g)[1] : missing
                ok_s_label = ok_s === missing ? "parity_skipped" : string(ok_s)
                @printf("  gpu_batch_strip_host %-8s  e2e=%.4fs  kernel_only=%.4fs  speedup_vs_production(e2e)=%.2fx  ok=%s  [baseline]\n",
                    label, t_strip_e2e, t_strip_kernel, t_prod / t_strip_kernel, ok_s_label)
                flush(stdout)
            end

            if profile in (:quick, :scaled)
                sums_v0 = zeros(FT, NB, bd...)
                counts_v0 = zeros(UInt32, NB, bd...)
                v0_cfg = BatchVariantConfig("gpu_batch_v0", geo, :gpu_batch_v0, 0)
                t_v0 = _time_variant!(
                    v0_cfg, sums_v0, counts_v0, x, u, sft, bin_edges;
                    backend = gpu_backend, n_warmup = 0, n_repeat = n_repeat,
                )
                ok_v0 = if profile === :quick && cuda_ok
                    BP.check_parity(sums_p, counts_p, sums_v0, counts_v0)[1]
                elseif profile === :quick
                    BP.check_parity(sums, counts, sums_v0, counts_v0)[1]
                else
                    false
                end
                @printf("  gpu_batch_v0_floor %-8s  total=%.4fs  speedup_vs_production=%.2fx  ok=%s\n",
                    label, t_v0, t_prod / t_v0, ok_v0)
                flush(stdout)
            end
        end
    end

    cuda_ok && ws !== nothing && SFC.release!(ws)
    println()
    println("done (PROFILE=$profile)")
    return nothing
end

main()
