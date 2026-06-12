# Shared helpers for `benchmark_prototypes.jl` and `gpu_full_benchmark.jl`.

using Printf: Printf, @printf, @sprintf
using Random: Random

"""Load prototype kernels into a fresh module (safe to re-`include` benchmark scripts)."""
function load_prototype_kernels()
    m = Module()
    Base.include(m, joinpath(@__DIR__, "GPUPrototypeKernels.jl"))
    return m
end

const TIMING_BASELINE = "baseline_linear_global"
const PARITY_BASELINE = "blockshared_256k_w256_u32"
const BASELINE_NAMES = (TIMING_BASELINE, "baseline_grid_N2")
const U32_VARIANTS = ("baseline_linear_global_u32", "blockshared_256k_w256_u32")

is_baseline(name::AbstractString) = name in BASELINE_NAMES
is_u32_variant(name::AbstractString) = name in U32_VARIANTS

function reference_config(variants)
    idx = findfirst(v -> v.name == TIMING_BASELINE, variants)
    idx === nothing && error("baseline_linear_global missing in prototype_variants")
    return variants[idx]
end

function parity_baseline_result(rows)
    idx = findfirst(r -> r.name == PARITY_BASELINE, rows)
    return idx === nothing ? nothing : rows[idx]
end

function result_counts_eltype(res)
    return hasproperty(res, :counts_eltype) ? res.counts_eltype : :f32
end

function device_count_tag(res, name::AbstractString)
    el = result_counts_eltype(res)
    return el == :u32 ? "u32" : "f32"
end

"""
    gold_parity(gold, res, name; sum_ref=nothing)

Compare GPU histogram vs CPU gold.

**Counts:** exact match vs `gold.counts_i64` (serial Int64 from `cpu_gold_histogram`).

**Sums:** vs `sum_ref` when given — must come from **validated** `cpu_f64_serial_sums`
(run `validate_f64_references` first). Default `sum_ref=gold.sums` is serial Float32.
"""
function gold_parity(gold, res, name::AbstractString; sum_ref = nothing)
    sum_reference = sum_ref === nothing ? gold.sums : sum_ref
    gold_counts = gold.counts_i64
    got_counts = res.counts_i64
    length(got_counts) == length(gold_counts) ||
        error("count vector length mismatch for $name")

    Δcnt = res.sum_counts_i64 - gold.n_in
    max_Δcnt_bin = maximum(abs.(got_counts .- gold_counts))
    max_Δsum = maximum(abs.(res.sums .- sum_reference))
    counts_bin_exact = got_counts == gold_counts
    counts_sum_exact = Δcnt == 0
    sums_ok = isapprox(res.sums, sum_reference; rtol = 1f-4, atol = 500.0f0)

    cnt_tag = device_count_tag(res, name)

    if name == TIMING_BASELINE
        label = "timing-ref"
        ok = false
    elseif name == "baseline_grid_N2"
        label = counts_sum_exact && sums_ok ? "ok(vs gold)" : "FAIL(vs gold)"
        ok = counts_sum_exact && sums_ok
    elseif name == PARITY_BASELINE
        label = "parity-ref"
        ok = counts_bin_exact && sums_ok
    elseif is_baseline(name)
        label = "baseline"
        ok = false
    elseif counts_bin_exact && sums_ok
        label = "ok"
        ok = true
    elseif !counts_bin_exact && !sums_ok
        label = max_Δcnt_bin == 0 && Δcnt == 0 ?
            "FAIL(sum)" :
            @sprintf("FAIL(cnt max_bin=%d sum)", max_Δcnt_bin)
        ok = false
    elseif !counts_bin_exact
        label = @sprintf("FAIL(cnt Δ=%+d max_bin=%d)", Δcnt, max_Δcnt_bin)
        ok = false
    else
        label = @sprintf("FAIL(sum max=%.4g)", max_Δsum)
        ok = false
    end

    return (;
        label,
        ok,
        cnt_tag,
        Δcnt,
        max_Δcnt_bin,
        max_Δsum,
        counts_bin_exact,
        counts_sum_exact,
        sums_ok,
    )
end

function parity_label(gold, res, name::AbstractString)
    p = gold_parity(gold, res, name)
    return p.label, p.ok
end

function parity_diagnostics(gold, res)
    p = gold_parity(gold, res, "diagnostic")
    return (
        counts_ok = p.counts_bin_exact,
        sums_ok = p.sums_ok,
        gold_i64 = gold.n_in,
        sum_counts_i64 = res.sum_counts_i64,
        max_abs_count_diff = p.max_Δcnt_bin,
        max_abs_sum_diff = p.max_Δsum,
        Δcnt = p.Δcnt,
    )
end

"""Legacy float-count parity (grid vs linear float baseline only)."""
function parity_check(ref, res)
    sums_ok = isapprox(ref.sums, res.sums; rtol = 1f-4, atol = 500.0f0)
    counts_ok = isapprox(sum(ref.counts), sum(res.counts); rtol = 1f-5, atol = 1f3) &&
        isapprox(ref.counts, res.counts; rtol = 1f-4, atol = 512.0f0)
    return sums_ok && counts_ok, sums_ok, counts_ok
end

"""Run one prototype variant `n_repeat` times; return minimum kernel/staging times + last result."""
function time_prototype_variant(
    ProtoGP,
    cfg,
    backend,
    sft,
    bin_edges,
    x_cpu,
    u_cpu,
    bufs;
    n_warmup = 1,
    n_repeat = 3,
)
    for _ in 1:n_warmup
        ProtoGP.run_prototype!(
            cfg, backend, sft, bin_edges, x_cpu, u_cpu;
            bufs = cfg.device_resident ? bufs : nothing,
        )
    end
    best = nothing
    best_kernel = Inf
    best_staging = Inf
    for _ in 1:n_repeat
        res = ProtoGP.run_prototype!(
            cfg, backend, sft, bin_edges, x_cpu, u_cpu;
            bufs = cfg.device_resident ? bufs : nothing,
        )
        if res.kernel_s < best_kernel
            best = res
            best_kernel = res.kernel_s
            best_staging = res.staging_s
        end
    end
    return (
        sums = best.sums,
        counts = best.counts,
        counts_i64 = best.counts_i64,
        sum_counts_i64 = best.sum_counts_i64,
        counts_eltype = best.counts_eltype,
        kernel_s = best_kernel,
        staging_s = best_staging,
        total_s = best_kernel + best_staging,
        sum_counts = sum(best.counts),
    )
end

function benchmark_all_variants(
    ProtoGP,
    backend,
    sft,
    bin_edges,
    x_cpu,
    u_cpu;
    n_repeat = 3,
)
    N = size(x_cpu, 2)
    variants = ProtoGP.prototype_variants(N)
    bufs = ProtoGP.prepare_device_buffers(backend, x_cpu, u_cpu, length(bin_edges.edges))

    ref_cfg = reference_config(variants)
    for _ in 1:2
        ProtoGP.run_prototype!(ref_cfg, backend, sft, bin_edges, x_cpu, u_cpu)
    end

    rows = NamedTuple[]
    for cfg in variants
        r = time_prototype_variant(
            ProtoGP, cfg, backend, sft, bin_edges, x_cpu, u_cpu, bufs;
            n_repeat = n_repeat,
        )
        push!(rows, (;
            name = cfg.name,
            variant = cfg.variant,
            nworkers = cfg.nworkers,
            workgroup_size = cfg.workgroup_size,
            device_resident = cfg.device_resident,
            kernel_s = r.kernel_s,
            staging_s = r.staging_s,
            total_s = r.total_s,
            sum_counts = r.sum_counts,
            sum_counts_i64 = r.sum_counts_i64,
            counts_eltype = r.counts_eltype,
            sums = r.sums,
            counts = r.counts,
            counts_i64 = r.counts_i64,
        ))
    end
    return rows, variants, bufs
end

function row_by_name(rows, name)
    idx = findfirst(r -> r.name == name, rows)
    return idx === nothing ? nothing : rows[idx]
end

function summarize_n(rows, gold)
    timing_ref = row_by_name(rows, TIMING_BASELINE)
    parity_ref = row_by_name(rows, PARITY_BASELINE)
    timing_ref === nothing && error("missing timing reference row")
    parity_ref === nothing && error("missing parity reference row")

    base_total = timing_ref.total_s
    out = Dict{String, Any}[]
    for r in rows
        p = gold_parity(gold, r, r.name)
        push!(out, Dict{String, Any}(
            "name" => r.name,
            "variant" => String(r.variant),
            "nworkers" => r.nworkers,
            "workgroup_size" => r.workgroup_size,
            "device_resident" => r.device_resident,
            "kernel_s" => r.kernel_s,
            "staging_s" => r.staging_s,
            "total_s" => r.total_s,
            "speedup_vs_timing" => base_total / r.total_s,
            "cnt_dev" => p.cnt_tag,
            "sum_counts" => r.sum_counts,
            "Δcnt" => p.Δcnt,
            "max_Δcnt_bin" => p.max_Δcnt_bin,
            "max_Δsum" => p.max_Δsum,
            "parity" => p.label,
            "parity_ok" => p.ok,
        ))
    end

    u32_row = row_by_name(rows, "baseline_linear_global_u32")
    timing_p = gold_parity(gold, timing_ref, TIMING_BASELINE)

    return (
        rows_json = out,
        timing_ref = timing_ref,
        parity_ref = parity_ref,
        gold_i64 = gold.n_in,
        float32_undercount = timing_p.Δcnt,
        u32_global_kernel_s = u32_row === nothing ? nothing : u32_row.kernel_s,
    )
end
