"""
    generate_gpu_figures.jl

Plot GPU benchmark JSON into docs/src/assets (no GPU required).

Run from repo root:
    julia --project=docs/generate_assets docs/generate_assets/generate_gpu_figures.jl

Requires `gpu/benchmark_results/assets_latest.json` from:
    julia --project=gpu gpu/collect_benchmark_assets.jl

Figures produced (honest names — not HPC strong/weak GPU scaling):
  • gpu_problem_size_scaling.png  — 1 GPU vs serial CPU, sweep N
  • gpu_slice_batch_scaling.png   — slice-batch API, sweep T at fixed N_SLICE

True GPU strong/weak scaling (multi-GPU) is not generated; see gpu/collect_multi_gpu_scaling.jl.
"""

using JSON: JSON
using CairoMakie: CairoMakie as CM

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const JSON_PATH = joinpath(REPO_ROOT, "gpu", "benchmark_results", "assets_latest.json")
const ASSETS_DIR = joinpath(REPO_ROOT, "docs", "src", "assets")

function load_payload()
    isfile(JSON_PATH) || error("Missing $JSON_PATH — run gpu/collect_benchmark_assets.jl on a GPU allocation first")
    return JSON.parsefile(JSON_PATH)
end

"""Read JSON section; accept legacy keys from older collector runs."""
function _section(payload, key::String, legacy_keys...)
    haskey(payload, key) && return payload[key]
    for lk in legacy_keys
        haskey(payload, lk) && return payload[lk]
    end
    return []
end

function _speedup(row)
    return get(row, "speedup_cpu_over_gpu", get(row, "speedup", NaN))
end

function _rows_by_dtype(rows, dtype::String)
    return sort(filter(r -> r["dtype"] == dtype, rows), by = r -> r["N"])
end

function _sorted_by_key(rows, key::String)
    return sort(rows, by = r -> r[key])
end

function _log_ylim(vals...)
    flat = filter(x -> x > 0, vcat(collect.(vals)...))
    isempty(flat) && return nothing
    lo, hi = extrema(flat)
    return (lo * 0.7, hi * 1.4)
end

"""Linear x-axis with ticks only at measured abscissae (integer labels, no 10^n)."""
function _axis_data_ticks!(ax, xs; pad::Float64 = 0.06)
    u = sort(unique(Float64.(xs)))
    length(u) == 1 && (u = [u[1] * 0.9, u[1], u[1] * 1.1])
    lo, hi = extrema(u)
    span = max(hi - lo, lo * 0.05)
    CM.xlims!(ax, lo - span * pad, hi + span * pad)
    ax.xticks = (u, string.(Int.(u)))
    return u
end

"""Warn when Float32 GPU time looks like first-call JIT (>> Float64 at same N)."""
function _warn_gpu_jit_outliers!(f32, f64)
    f64_by_n = Dict(r["N"] => r["gpu_elapsed_s"] for r in f64)
    for r in f32
        n = r["N"]
        t32 = r["gpu_elapsed_s"]
        t64 = get(f64_by_n, n, nothing)
        t64 === nothing && continue
        if t32 > 5 * t64 && t32 > 0.01
            @warn "GPU Float32 at N=$n looks like JIT/outlier" gpu_f32_s=t32 gpu_f64_s=t64 hint="Re-run collect_benchmark_assets.jl (session warmup + median timing)."
        end
    end
    return nothing
end

function _config_note(payload)
    dev = get(get(payload, "device", Dict()), "device", "GPU")
    return "1× $dev vs serial CPU — thread scaling: see CPU strong/weak figures"
end

"""Place title (+ optional subtitle) in a nested header row; return first axis row index."""
function _figure_header!(fig, title::String, subtitle::String = "")
    if isempty(subtitle)
        CM.Label(fig[0, 1:2], title; fontsize = 15, font = :bold, halign = :left)
        CM.rowsize!(fig.layout, 0, CM.Fixed(38))
        return 1
    end
    header = CM.GridLayout()
    fig[0, 1:2] = header
    CM.Label(header[1, 1], title; fontsize = 15, font = :bold, halign = :left, tellwidth = false)
    CM.Label(header[2, 1], subtitle; fontsize = 10, halign = :left, tellwidth = false)
    CM.rowgap!(header, 10)
    CM.rowsize!(fig.layout, 0, CM.Auto())
    return 1
end

function plot_problem_size_scaling!(payload)
    rows = _section(payload, "problem_size_scaling", "strong_scaling")
    isempty(rows) && return

    n_unique = length(unique(r["N"] for r in rows))
    if n_unique < 6
        @warn "Problem-size scaling has only $n_unique N value(s)" n_list=get(get(payload, "config", Dict()), "N_list", nothing) hint="Re-run collect_benchmark_assets.jl (default N: 4000,6000,8000,12000,16000,20000)."
    end

    f32 = _rows_by_dtype(rows, "Float32")
    f64 = _rows_by_dtype(rows, "Float64")
    isempty(f32) && isempty(f64) && return
    _warn_gpu_jit_outliers!(f32, f64)

    N_all = sort(unique(vcat([r["N"] for r in f32], [r["N"] for r in f64])))
    N_f32 = Float64[r["N"] for r in f32]
    N_f64 = Float64[r["N"] for r in f64]
    cpu_f32 = Float64[r["cpu_elapsed_s"] for r in f32]
    gpu_f32 = Float64[r["gpu_elapsed_s"] for r in f32]
    cpu_f64 = Float64[r["cpu_elapsed_s"] for r in f64]
    gpu_f64 = Float64[r["gpu_elapsed_s"] for r in f64]
    sp_f32 = Float64[_speedup(r) for r in f32]
    sp_f64 = Float64[_speedup(r) for r in f64]

    fig = CM.Figure(size = (980, 800), fontsize = 13)
    r0 = _figure_header!(
        fig,
        "Problem-size scaling (single GPU, 3D longitudinal 2nd-order SF)",
        _config_note(payload),
    )

    ax_t = CM.Axis(fig[r0, 1:2],
        xlabel = "N (points)",
        ylabel = "Elapsed time (s)",
        yscale = CM.log10,
        title = "Wall time vs N",
    )
    ylims_t = _log_ylim(cpu_f32, gpu_f32, cpu_f64, gpu_f64)
    ylims_t !== nothing && CM.ylims!(ax_t, ylims_t...)

    ms = 9
    lw = 2.25
    l_cpu_f32 = CM.lines!(ax_t, N_f32, cpu_f32; color = (:steelblue, 0.95), linewidth = lw)
    CM.scatter!(ax_t, N_f32, cpu_f32; color = :steelblue, markersize = ms, marker = :circle)
    l_gpu_f32 = CM.lines!(ax_t, N_f32, gpu_f32; color = (:darkorange, 0.95), linewidth = lw)
    CM.scatter!(ax_t, N_f32, gpu_f32; color = :darkorange, markersize = ms, marker = :rect)
    l_cpu_f64 = CM.lines!(ax_t, N_f64, cpu_f64; color = (:steelblue, 0.55), linewidth = lw, linestyle = :dash)
    CM.scatter!(ax_t, N_f64, cpu_f64; color = (:steelblue, 0.55), markersize = ms, marker = :circle)
    l_gpu_f64 = CM.lines!(ax_t, N_f64, gpu_f64; color = (:darkorange, 0.55), linewidth = lw, linestyle = :dash)
    CM.scatter!(ax_t, N_f64, gpu_f64; color = (:darkorange, 0.55), markersize = ms, marker = :rect)

    CM.Legend(fig[r0, 3],
        [l_cpu_f32, l_gpu_f32, l_cpu_f64, l_gpu_f64],
        ["CPU serial (Float32)", "1 GPU + workspace (Float32)",
         "CPU serial (Float64)", "1 GPU + workspace (Float64)"],
        fontsize = 11,
    )

    ax_sp = CM.Axis(fig[2, 1],
        xlabel = "N (points)",
        ylabel = "CPU time / GPU time",
        title = "Crossover (not parallel efficiency)",
    )
    sp_all = vcat(sp_f32, sp_f64)
    !isempty(sp_all) && CM.ylims!(ax_sp, 0.0, maximum(sp_all) * 1.08)

    l_sp_f32 = CM.lines!(ax_sp, N_f32, sp_f32; color = :seagreen, linewidth = lw)
    CM.scatter!(ax_sp, N_f32, sp_f32; color = :seagreen, markersize = ms)
    l_sp_f64 = CM.lines!(ax_sp, N_f64, sp_f64; color = (:purple, 0.75), linewidth = lw, linestyle = :dash)
    CM.scatter!(ax_sp, N_f64, sp_f64; color = (:purple, 0.75), markersize = ms, marker = :diamond)
    CM.hlines!(ax_sp, [1.0]; color = (:gray, 0.45), linestyle = :dot, linewidth = 1)
    CM.Legend(fig[2, 3], [l_sp_f32, l_sp_f64], ["Float32", "Float64"]; fontsize = 11)

    f32_map = Dict(r["N"] => r["gpu_elapsed_s"] for r in f32)
    f64_map = Dict(r["N"] => r["gpu_elapsed_s"] for r in f64)
    N_common = sort(collect(intersect(keys(f32_map), keys(f64_map))))

    ax_r = CM.Axis(fig[2, 2],
        xlabel = "N (points)",
        ylabel = "GPU Float64 / Float32 time",
        title = "Float64 penalty (1 GPU)",
    )
    if !isempty(N_common)
        ratio = Float64[f64_map[n] / f32_map[n] for n in N_common]
        CM.hlines!(ax_r, [1.0]; color = (:gray, 0.45), linestyle = :dot, linewidth = 1)
        CM.lines!(ax_r, Float64.(N_common), ratio; color = :purple, linewidth = lw)
        CM.scatter!(ax_r, Float64.(N_common), ratio; color = :purple, markersize = ms)
        r_lo, r_hi = extrema(ratio)
        CM.ylims!(ax_r, max(0.85, r_lo * 0.92), r_hi * 1.08)
    end

    _axis_data_ticks!(ax_t, N_all)
    _axis_data_ticks!(ax_sp, N_all)
    !isempty(N_common) && _axis_data_ticks!(ax_r, N_common)

    CM.rowsize!(fig.layout, 1, CM.Fixed(280))
    CM.rowsize!(fig.layout, 2, CM.Fixed(240))
    CM.colsize!(fig.layout, 3, CM.Fixed(210))

    out = joinpath(ASSETS_DIR, "gpu_problem_size_scaling.png")
    CM.save(out, fig, px_per_unit = 2)
    println("Saved: $out")
end

function plot_slice_batch_scaling!(payload)
    rows = _section(payload, "slice_batch_scaling", "weak_slice_scaling")
    isempty(rows) && return

    N_slice = get(get(payload, "config", Dict()), "N_slice", "?")

    f32 = _sorted_by_key(filter(r -> r["dtype"] == "Float32", rows), "T")
    f64 = _sorted_by_key(filter(r -> r["dtype"] == "Float64", rows), "T")
    isempty(f32) && isempty(f64) && return

    T_f32 = Float64[r["T"] for r in f32]
    cpu_f32 = Float64[r["cpu_loop_elapsed_s"] for r in f32]
    naive_f32 = Float64[r["gpu_naive_elapsed_s"] for r in f32]
    slice_f32 = Float64[r["gpu_slice_elapsed_s"] for r in f32]
    T_f64 = Float64[r["T"] for r in f64]
    cpu_f64 = Float64[r["cpu_loop_elapsed_s"] for r in f64]
    slice_f64 = Float64[r["gpu_slice_elapsed_s"] for r in f64]
    sp_f32 = Float64[r["speedup_slice_vs_cpu"] for r in f32]
    sp_f64 = Float64[r["speedup_slice_vs_cpu"] for r in f64]

    T_all = sort(unique(vcat(T_f32, T_f64)))

    fig = CM.Figure(size = (980, 700), fontsize = 13)
    r0 = _figure_header!(
        fig,
        "Slice-batch scaling (fixed N = $N_slice, vary T)",
        "1 GPU — not HPC weak scaling; CPU serial per-slice loop vs GPU batch driver",
    )

    ax_t = CM.Axis(fig[r0, 1:2],
        xlabel = "T (time slices)",
        ylabel = "Total elapsed (s)",
        yscale = CM.log10,
        title = "CPU loop vs GPU paths (1 GPU, shared y-axis)",
    )
    ylims_t = _log_ylim(cpu_f32, naive_f32, slice_f32, cpu_f64, slice_f64)
    ylims_t !== nothing && CM.ylims!(ax_t, ylims_t...)

    lw = 2.25
    ms = 8
    l1 = CM.lines!(ax_t, T_f32, cpu_f32; color = (:steelblue, 0.9), linewidth = lw)
    CM.scatter!(ax_t, T_f32, cpu_f32; color = :steelblue, markersize = ms)
    l2 = CM.lines!(ax_t, T_f32, naive_f32; color = (:crimson, 0.7), linewidth = lw, linestyle = :dash)
    CM.scatter!(ax_t, T_f32, naive_f32; color = (:crimson, 0.7), markersize = ms, marker = :rect)
    l3 = CM.lines!(ax_t, T_f32, slice_f32; color = :darkorange, linewidth = lw)
    CM.scatter!(ax_t, T_f32, slice_f32; color = :darkorange, markersize = ms, marker = :diamond)
    l4 = CM.lines!(ax_t, T_f64, cpu_f64; color = (:steelblue, 0.45), linewidth = lw, linestyle = :dashdot)
    CM.scatter!(ax_t, T_f64, cpu_f64; color = (:steelblue, 0.45), markersize = ms)
    l5 = CM.lines!(ax_t, T_f64, slice_f64; color = (:darkorange, 0.45), linewidth = lw, linestyle = :dashdot)
    CM.scatter!(ax_t, T_f64, slice_f64; color = (:darkorange, 0.45), markersize = ms, marker = :diamond)

    CM.Legend(fig[r0, 3],
        [l1, l2, l3, l4, l5],
        ["CPU loop (f32)", "GPU naive loop (f32)", "GPU slice driver (f32)",
         "CPU loop (f64)", "GPU slice driver (f64)"],
        fontsize = 10,
    )

    ax_sp = CM.Axis(fig[2, 1:2],
        xlabel = "T (time slices)",
        ylabel = "CPU loop / GPU slice driver",
        title = "Batch driver vs CPU loop",
    )
    sp_all = vcat(sp_f32, sp_f64)
    !isempty(sp_all) && CM.ylims!(ax_sp, 0.0, maximum(sp_all) * 1.06)
    l_sp1 = CM.lines!(ax_sp, T_f32, sp_f32; color = :seagreen, linewidth = lw)
    CM.scatter!(ax_sp, T_f32, sp_f32; color = :seagreen, markersize = ms)
    l_sp2 = CM.lines!(ax_sp, T_f64, sp_f64; color = (:purple, 0.7), linewidth = lw, linestyle = :dash)
    CM.scatter!(ax_sp, T_f64, sp_f64; color = (:purple, 0.7), markersize = ms, marker = :diamond)
    CM.hlines!(ax_sp, [1.0]; color = (:gray, 0.45), linestyle = :dot)
    CM.Legend(fig[2, 3], [l_sp1, l_sp2], ["Float32", "Float64"]; fontsize = 11)

    _axis_data_ticks!(ax_t, T_all)
    _axis_data_ticks!(ax_sp, T_all)

    CM.rowsize!(fig.layout, 1, CM.Fixed(280))
    CM.rowsize!(fig.layout, 2, CM.Fixed(220))
    CM.colsize!(fig.layout, 3, CM.Fixed(210))

    out = joinpath(ASSETS_DIR, "gpu_slice_batch_scaling.png")
    CM.save(out, fig, px_per_unit = 2)
    println("Saved: $out")
end

function main()
    mkpath(ASSETS_DIR)
    payload = load_payload()
    plot_problem_size_scaling!(payload)
    plot_slice_batch_scaling!(payload)
    println("Done.")
end

main()
