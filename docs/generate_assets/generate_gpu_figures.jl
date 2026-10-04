"""
    generate_gpu_figures.jl

Plot `gpu/benchmark_results/assets_latest.json` into `docs/src/assets`; needs no GPU.

      pkg> activate gpu 
    julia> include("gpu/collect_benchmark_assets.jl")                          # on a GPU allocation
      pkg> activate docs/generate_assets 
    julia> include("docs/generate_assets/generate_gpu_figures.jl")

Figures:
  • gpu_problem_size_scaling.png — one GPU against one CPU core, over the number of points
  • gpu_slice_batch_scaling.png  — T snapshots: the batch entry on one CPU core and on the GPU, and one GPU call per
    host snapshot
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

_rows(rows, dtype::String, key::String) = sort(filter(r -> r["dtype"] == dtype, rows), by = r -> r[key])
_col(rows, key::String) = Float64[r[key] for r in rows]

function _log_ylim(vals...)
    lo, hi = extrema(filter(x -> x > 0, vcat(collect.(vals)...)))
    return (lo * 0.7, hi * 1.4)
end

"""Logarithmic x-axis with ticks at the measured abscissae."""
function _log_ticks!(ax, xs)
    u = sort(unique(Float64.(xs)))
    CM.xlims!(ax, first(u) / 1.15, last(u) * 1.15)
    ax.xticks = (u, string.(Int.(u)))
    return nothing
end

_device(payload) = payload["device"]["device"]

"""Title and subtitle in a header row above the axes."""
function _figure_header!(fig, title::String, subtitle::String)
    header = CM.GridLayout()
    fig[0, 1:2] = header
    CM.Label(header[1, 1], title; fontsize = 15, font = :bold, halign = :left, tellwidth = false)
    CM.Label(header[2, 1], subtitle; fontsize = 10, halign = :left, tellwidth = false)
    CM.rowgap!(header, 10)
    CM.rowsize!(fig.layout, 0, CM.Auto())
    return nothing
end

function plot_problem_size_scaling!(payload)
    rows = payload["problem_size_scaling"]
    f32, f64 = _rows(rows, "Float32", "N"), _rows(rows, "Float64", "N")
    N32, N64 = _col(f32, "N"), _col(f64, "N")
    cpu32, gpu32 = _col(f32, "cpu_elapsed_s"), _col(f32, "gpu_elapsed_s")
    cpu64, gpu64 = _col(f64, "cpu_elapsed_s"), _col(f64, "gpu_elapsed_s")

    fig = CM.Figure(size = (980, 800), fontsize = 13)
    _figure_header!(fig, "One GPU and one CPU core over the number of points",
                    "$(_device(payload)); 3-D longitudinal S₂, 20 distance bins; the GPU calls reuse a workspace")

    ax_t = CM.Axis(fig[1, 1:2], xlabel = "N (points)", ylabel = "seconds per call", xscale = CM.log10,
                   yscale = CM.log10, title = "Time per call")
    CM.ylims!(ax_t, _log_ylim(cpu32, gpu32, cpu64, gpu64)...)
    ms, lw = 9, 2.25
    l1 = CM.scatterlines!(ax_t, N32, cpu32; color = :steelblue, linewidth = lw, markersize = ms)
    l2 = CM.scatterlines!(ax_t, N32, gpu32; color = :darkorange, linewidth = lw, markersize = ms, marker = :rect)
    l3 = CM.scatterlines!(ax_t, N64, cpu64; color = (:steelblue, 0.55), linewidth = lw, markersize = ms,
                          linestyle = :dash)
    l4 = CM.scatterlines!(ax_t, N64, gpu64; color = (:darkorange, 0.55), linewidth = lw, markersize = ms,
                          marker = :rect, linestyle = :dash)
    CM.Legend(fig[1, 3], [l1, l2, l3, l4],
              ["CPU, one core (Float32)", "GPU (Float32)", "CPU, one core (Float64)", "GPU (Float64)"]; fontsize = 11)

    ax_sp = CM.Axis(fig[2, 1], xlabel = "N (points)", ylabel = "CPU time / GPU time", xscale = CM.log10,
                    title = "CPU time over GPU time")
    sp32, sp64 = cpu32 ./ gpu32, cpu64 ./ gpu64
    CM.ylims!(ax_sp, 0.0, maximum(vcat(sp32, sp64)) * 1.08)
    s1 = CM.scatterlines!(ax_sp, N32, sp32; color = :seagreen, linewidth = lw, markersize = ms)
    s2 = CM.scatterlines!(ax_sp, N64, sp64; color = (:purple, 0.75), linewidth = lw, markersize = ms,
                          marker = :diamond, linestyle = :dash)
    CM.Legend(fig[2, 3], [s1, s2], ["Float32", "Float64"]; fontsize = 11)

    gpu64_at = Dict(zip(N64, gpu64))
    N_both = filter(n -> haskey(gpu64_at, n), N32)
    ratio = Float64[gpu64_at[n] / gpu32[findfirst(==(n), N32)] for n in N_both]
    ax_r = CM.Axis(fig[2, 2], xlabel = "N (points)", ylabel = "GPU Float64 / Float32 time", xscale = CM.log10,
                   title = "GPU time in Float64 over Float32")
    CM.hlines!(ax_r, [1.0]; color = (:gray, 0.45), linestyle = :dot, linewidth = 1)
    CM.scatterlines!(ax_r, N_both, ratio; color = :purple, linewidth = lw, markersize = ms)
    CM.ylims!(ax_r, min(0.9, minimum(ratio) * 0.92), maximum(ratio) * 1.08)

    foreach(ax -> _log_ticks!(ax, vcat(N32, N64)), (ax_t, ax_sp))
    _log_ticks!(ax_r, N_both)
    CM.rowsize!(fig.layout, 1, CM.Fixed(280))
    CM.rowsize!(fig.layout, 2, CM.Fixed(240))
    CM.colsize!(fig.layout, 3, CM.Fixed(210))

    out = joinpath(ASSETS_DIR, "gpu_problem_size_scaling.png")
    CM.save(out, fig, px_per_unit = 2)
    println("Saved: $out")
end

function plot_slice_batch_scaling!(payload)
    rows = payload["slice_batch_scaling"]
    N_slice = payload["config"]["N_slice"]
    f32, f64 = _rows(rows, "Float32", "T"), _rows(rows, "Float64", "T")
    T32, T64 = _col(f32, "T"), _col(f64, "T")
    cpu32, naive32, batch32 = _col(f32, "cpu_batch_elapsed_s"), _col(f32, "gpu_naive_elapsed_s"),
                              _col(f32, "gpu_batch_elapsed_s")
    cpu64, naive64, batch64 = _col(f64, "cpu_batch_elapsed_s"), _col(f64, "gpu_naive_elapsed_s"),
                              _col(f64, "gpu_batch_elapsed_s")

    fig = CM.Figure(size = (980, 700), fontsize = 13)
    _figure_header!(fig, "T snapshots of N = $N_slice points",
                    "$(_device(payload)); the batch entry on one CPU core and on the GPU, and one GPU call per host " *
                    "snapshot")

    ax_t = CM.Axis(fig[1, 1:2], xlabel = "T (snapshots)", ylabel = "seconds", xscale = CM.log2, yscale = CM.log10,
                   title = "Time for all T snapshots")
    CM.ylims!(ax_t, _log_ylim(cpu32, naive32, batch32, cpu64, naive64, batch64)...)
    lw, ms = 2.25, 8
    l1 = CM.scatterlines!(ax_t, T32, cpu32; color = :steelblue, linewidth = lw, markersize = ms)
    l2 = CM.scatterlines!(ax_t, T32, naive32; color = :crimson, linewidth = lw, markersize = ms, marker = :rect)
    l3 = CM.scatterlines!(ax_t, T32, batch32; color = :darkorange, linewidth = lw, markersize = ms, marker = :diamond)
    l4 = CM.scatterlines!(ax_t, T64, cpu64; color = (:steelblue, 0.5), linewidth = lw, markersize = ms,
                          linestyle = :dash)
    l5 = CM.scatterlines!(ax_t, T64, naive64; color = (:crimson, 0.5), linewidth = lw, markersize = ms,
                          marker = :rect, linestyle = :dash)
    l6 = CM.scatterlines!(ax_t, T64, batch64; color = (:darkorange, 0.5), linewidth = lw, markersize = ms,
                          marker = :diamond, linestyle = :dash)
    CM.Legend(fig[1, 3], [l1, l2, l3, l4, l5, l6],
              ["CPU batch, one core (Float32)", "GPU call per snapshot (Float32)", "GPU batch (Float32)",
               "CPU batch, one core (Float64)", "GPU call per snapshot (Float64)", "GPU batch (Float64)"];
              fontsize = 10)

    ax_sp = CM.Axis(fig[2, 1:2], xlabel = "T (snapshots)", ylabel = "CPU batch time / GPU batch time",
                    xscale = CM.log2, title = "CPU batch time over GPU batch time")
    sp32, sp64 = cpu32 ./ batch32, cpu64 ./ batch64
    CM.ylims!(ax_sp, 0.0, maximum(vcat(sp32, sp64)) * 1.06)
    s1 = CM.scatterlines!(ax_sp, T32, sp32; color = :seagreen, linewidth = lw, markersize = ms)
    s2 = CM.scatterlines!(ax_sp, T64, sp64; color = (:purple, 0.7), linewidth = lw, markersize = ms,
                          marker = :diamond, linestyle = :dash)
    CM.Legend(fig[2, 3], [s1, s2], ["Float32", "Float64"]; fontsize = 11)

    foreach(ax -> _log_ticks!(ax, vcat(T32, T64)), (ax_t, ax_sp))
    CM.rowsize!(fig.layout, 1, CM.Fixed(280))
    CM.rowsize!(fig.layout, 2, CM.Fixed(220))
    CM.colsize!(fig.layout, 3, CM.Fixed(250))

    out = joinpath(ASSETS_DIR, "gpu_slice_batch_scaling.png")
    CM.save(out, fig, px_per_unit = 2)
    println("Saved: $out")
end

function main()
    mkpath(ASSETS_DIR)
    payload = load_payload()
    plot_problem_size_scaling!(payload)
    plot_slice_batch_scaling!(payload)
end

main()
