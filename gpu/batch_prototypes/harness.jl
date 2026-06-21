# Harness: variant registry, parity checks, timing helpers.

using Printf: @printf

"""
    batch_prototype_variant() -> String

Human-readable label for the Phase 0 GPU/CPU batch kernel in this tree.
"""
function batch_prototype_variant()
    return "fused_tiled128__fixed_x_one_pass_inner_strip__varying_x_direct_inner_b"
end

function batch_prototype_variant_description()
    return """
    primary: gpu_batch_fused_tiled — one tiled128 launch, geometry once per pair, inner batch strips (Case A fixed);
    baseline: gpu_batch_tiled fixed-x host smem strip (252× pair replay); varying-x per-b slice;
    experimental: gpu_batch_fused_tiled_fixed_x_vram! (block-private partial + merge);
    v0 grid-stride floor only."""
end

"""Prototype variant descriptor for Phase 0 matrix."""
struct BatchVariantConfig
    name::String
    geometry::BatchGeometryCase
    backend::Symbol  # :cpu_slice, :cpu_batch, :gpu_batch_tiled, :gpu_batch_fused, :gpu_batch_v0
    strip_width::Int
end

function default_variants()
    return [
        BatchVariantConfig("cpu_slice_fixed", FixedX, :cpu_slice, 0),
        BatchVariantConfig("cpu_batch_fixed_w32", FixedX, :cpu_batch, 32),
        BatchVariantConfig("cpu_slice_varying", VaryingX, :cpu_slice, 0),
        BatchVariantConfig("cpu_batch_varying_w32", VaryingX, :cpu_batch, 32),
        BatchVariantConfig("gpu_batch_fused_fixed", FixedX, :gpu_batch_fused, 0),
        BatchVariantConfig("gpu_batch_usmem_direct_fixed", FixedX, :gpu_batch_usmem_direct, 0),
        BatchVariantConfig("gpu_batch_usmem_priv_fixed", FixedX, :gpu_batch_usmem_priv, 0),
        BatchVariantConfig("gpu_batch_usmem_atomic_fixed", FixedX, :gpu_batch_usmem_atomic, 0),
        BatchVariantConfig("gpu_batch_vectorized_fixed", FixedX, :gpu_batch_vectorized, 0),
        BatchVariantConfig("gpu_batch_coalesced_fixed", FixedX, :gpu_batch_coalesced, 0),
        BatchVariantConfig("gpu_batch_fused_varying", VaryingX, :gpu_batch_fused, 0),
        BatchVariantConfig("gpu_batch_strip_host_fixed", FixedX, :gpu_batch_tiled, BATCH_TILED_STRIP_W),
        BatchVariantConfig("gpu_batch_tiled_varying", VaryingX, :gpu_batch_tiled, BATCH_TILED_STRIP_W),
        BatchVariantConfig("gpu_batch_v0_fixed", FixedX, :gpu_batch_v0, 0),
        BatchVariantConfig("gpu_batch_v0_varying", VaryingX, :gpu_batch_v0, 0),
    ]
end

function _run_cpu_variant!(
    cfg::BatchVariantConfig,
    sums,
    counts,
    x,
    u,
    sf_type,
    bin_edges,
)
    if cfg.geometry === FixedX
        if cfg.backend === :cpu_slice
            cpu_slice_baseline!(sums, counts, x, u, sf_type, bin_edges; fixed_x = true)
        elseif cfg.backend === :cpu_batch
            cpu_batch_fixed_x!(sums, counts, x, u, sf_type, bin_edges; strip_width = cfg.strip_width)
        else
            error("unsupported CPU backend $(cfg.backend) for FixedX")
        end
    elseif cfg.geometry === VaryingX
        if cfg.backend === :cpu_slice
            cpu_slice_baseline!(sums, counts, x, u, sf_type, bin_edges; fixed_x = false)
        elseif cfg.backend === :cpu_batch
            cpu_batch_varying_x!(sums, counts, x, u, sf_type, bin_edges; strip_width = cfg.strip_width)
        else
            error("unsupported CPU backend $(cfg.backend) for VaryingX")
        end
    end
    return nothing
end

function _run_gpu_variant!(
    cfg::BatchVariantConfig,
    sums,
    counts,
    x,
    u,
    sf_type,
    bin_edges,
    backend;
    max_vram::Int = 0,
    workspace = nothing,
    download::Bool = true,
)
    if cfg.backend === :gpu_batch_fused
        gpu_batch_fused_tiled!(
            backend, sums, counts, x, u, sf_type, bin_edges;
            max_vram = max_vram, workspace = workspace, download = download,
        )
    elseif cfg.backend === :gpu_batch_usmem_direct
        gpu_batch_fused_tiled_fixed_x_usmem!(
            backend, sums, counts, x, u, sf_type, bin_edges;
            workspace = workspace, download = download,
        )
    elseif cfg.backend === :gpu_batch_usmem_priv
        gpu_batch_fused_tiled_fixed_x_usmem_priv!(
            backend, sums, counts, x, u, sf_type, bin_edges;
            workspace = workspace, download = download,
        )
    elseif cfg.backend === :gpu_batch_usmem_atomic
        gpu_batch_fused_tiled_fixed_x_usmem_atomic_flush!(
            backend, sums, counts, x, u, sf_type, bin_edges;
            workspace = workspace, download = download,
        )
    elseif cfg.backend === :gpu_batch_vectorized
        gpu_batch_fused_tiled_fixed_x_vectorized!(
            backend, sums, counts, x, u, sf_type, bin_edges;
            workspace = workspace, download = download,
        )
    elseif cfg.backend === :gpu_batch_coalesced
        gpu_batch_fused_tiled_fixed_x_coalesced!(
            backend, sums, counts, x, u, sf_type, bin_edges;
            workspace = workspace, download = download,
        )
    elseif cfg.backend === :gpu_batch_tiled
        if cfg.geometry === FixedX
            gpu_batch_tiled_fixed_x!(
                backend, sums, counts, x, u, sf_type, bin_edges;
                workspace = workspace, download = download,
            )
        else
            gpu_batch_tiled_varying_x!(
                backend, sums, counts, x, u, sf_type, bin_edges;
                workspace = workspace, download = download,
            )
        end
    elseif cfg.backend === :gpu_batch_v0
        if cfg.geometry === FixedX
            gpu_batch_fixed_x!(backend, sums, counts, x, u, sf_type, bin_edges)
        else
            gpu_batch_varying_x!(backend, sums, counts, x, u, sf_type, bin_edges)
        end
    else
        error("unsupported GPU backend $(cfg.backend)")
    end
    return nothing
end

"""Run one variant; return elapsed seconds (includes alloc/H2D/D2H unless `download=false`)."""
function run_variant!(
    cfg::BatchVariantConfig,
    sums,
    counts,
    x,
    u,
    sf_type,
    bin_edges;
    backend = nothing,
    max_vram::Int = 0,
    workspace = nothing,
    download::Bool = true,
)
    t = @elapsed begin
        if cfg.backend in (:gpu_batch_tiled, :gpu_batch_fused, :gpu_batch_v0, :gpu_batch_usmem_direct, :gpu_batch_usmem_priv, :gpu_batch_usmem_atomic, :gpu_batch_coalesced, :gpu_batch_vectorized)
            backend !== nothing || error("GPU batch variant requires backend")
            _run_gpu_variant!(
                cfg, sums, counts, x, u, sf_type, bin_edges, backend;
                max_vram = max_vram, workspace = workspace, download = download,
            )
        else
            _run_cpu_variant!(cfg, sums, counts, x, u, sf_type, bin_edges)
        end
    end
    return t
end

"""
Compare candidate histograms to slice baseline for the same geometry case.
Returns (ok, max_sums_diff, counts_equal).
"""
function check_parity(
    sums_ref,
    counts_ref,
    sums_cand,
    counts_cand;
    rtol = 1e-4,
    atol = 1f-3,
)
    ok = histograms_equal(sums_ref, counts_ref, sums_cand, counts_cand; rtol = rtol, atol = atol)
    Δ = max_abs_diff(sums_ref, sums_cand)
    ce = counts_ref == counts_cand
    return ok, Δ, ce
end

"""Run parity matrix for small problem; print summary.

GPU tiled paths use production **2D** tiled128 math; `cpu_slice` gold uses **3D**
`cpu_gold_histogram`. Fixed-x Δ_slice is usually within tolerance; varying-x on CUDA
often shows large Δ_slice while still matching production — use `ok` (vs `cpu_batch`)
and the benchmark's `gpu_fused_vs_production` line as the integration gate.
"""
function run_parity_suite(
    x_fixed,
    u_fixed,
    x_vary,
    u_vary,
    sf_type,
    bin_edges;
    backend = nothing,
    rtol = 1e-4,
    atol = 1f-3,
    max_vram::Int = 0,
)
    lp = linear_bin_params(bin_edges)
    NB = lp.n_bins - 1
    results = NamedTuple[]

    for geo in (FixedX, VaryingX)
        x = geo === FixedX ? x_fixed : x_vary
        u = geo === FixedX ? u_fixed : u_vary
        bd = batch_dims(u)
        sums_slice = zeros(eltype(x), NB, bd...)
        counts_slice = zeros(UInt32, NB, bd...)
        slice_cfg = BatchVariantConfig(
            geo === FixedX ? "cpu_slice_fixed" : "cpu_slice_varying",
            geo, :cpu_slice, 0,
        )
        run_variant!(slice_cfg, sums_slice, counts_slice, x, u, sf_type, bin_edges)

        sums_batch = zeros(eltype(x), NB, bd...)
        counts_batch = zeros(UInt32, NB, bd...)
        batch_cfg = BatchVariantConfig(
            geo === FixedX ? "cpu_batch_fixed_w32" : "cpu_batch_varying_w32",
            geo, :cpu_batch, 32,
        )
        run_variant!(batch_cfg, sums_batch, counts_batch, x, u, sf_type, bin_edges)

        for cfg in default_variants()
            cfg.geometry === geo || continue
            cfg.backend === :cpu_slice && continue
            cfg.backend in (:gpu_batch_tiled, :gpu_batch_fused, :gpu_batch_v0, :gpu_batch_usmem_direct, :gpu_batch_usmem_priv, :gpu_batch_usmem_atomic, :gpu_batch_coalesced, :gpu_batch_vectorized) && backend === nothing && continue
            sums = zeros(eltype(x), NB, bd...)
            counts = zeros(UInt32, NB, bd...)
            t = run_variant!(
                cfg, sums, counts, x, u, sf_type, bin_edges;
                backend = backend, max_vram = max_vram,
            )
            ok_slice, Δ_slice, ce_slice = check_parity(
                sums_slice, counts_slice, sums, counts; rtol = rtol, atol = atol,
            )
            is_gpu = cfg.backend in (:gpu_batch_tiled, :gpu_batch_fused, :gpu_batch_v0, :gpu_batch_usmem_direct, :gpu_batch_usmem_priv, :gpu_batch_usmem_atomic, :gpu_batch_coalesced, :gpu_batch_vectorized)
            if is_gpu
                ok, Δ, ce = check_parity(
                    sums_batch, counts_batch, sums, counts; rtol = rtol, atol = atol,
                )
            else
                ok, Δ, ce = ok_slice, Δ_slice, ce_slice
            end
            push!(results, (
                name = cfg.name,
                geometry = geo,
                ok = ok,
                ok_slice = ok_slice,
                Δ = Δ,
                Δ_slice = Δ_slice,
                counts_equal = ce,
                counts_equal_slice = ce_slice,
                time_s = t,
                gpu = is_gpu,
            ))
        end
    end
    return results
end

function print_parity_results(results)
    println("=== batch prototype parity ===")
    println("  gate: CPU vs cpu_slice; GPU vs cpu_batch (production 2D; see timed gpu_fused_vs_production)")
    for r in results
        if r.gpu
            @printf(
                "  %-32s geo=%-8s ok=%5s  max|Δbatch|=%.4g  max|Δslice|=%.4g  counts=%s  t=%.4fs\n",
                r.name, string(r.geometry), r.ok, r.Δ, r.Δ_slice, r.counts_equal, r.time_s,
            )
        else
            @printf(
                "  %-32s geo=%-8s ok=%5s  max|Δsums|=%.4g  counts=%s  t=%.4fs\n",
                r.name, string(r.geometry), r.ok, r.Δ, r.counts_equal, r.time_s,
            )
        end
    end
    return nothing
end

function parse_batch_env(default::NTuple{1, Int} = (8,))
    s = get(ENV, "BATCH", "")
    if isempty(s)
        return default
    end
    parts = parse.(Int, split(s, ','))
    return Tuple(parts)
end

function make_random_batch_problem(
    FT,
    N::Int,
    batch_shape::Dims;
    fixed_x::Bool = true,
    seed::Int = 42,
)
    Random.seed!(seed)
    if fixed_x
        x = rand(FT, 2, N)
        u = rand(FT, 2, N, batch_shape...)
        return x, u
    else
        x = rand(FT, 2, N, batch_shape...)
        u = rand(FT, 2, N, batch_shape...)
        return x, u
    end
end
