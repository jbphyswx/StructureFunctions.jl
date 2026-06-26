"""
Generate static figure assets for StructureFunctions.jl docs and README.md.

Run from the repo root:
    julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl
"""

using StructureFunctions: StructureFunctions as SF, Calculations as SFC
using OhMyThreads: OhMyThreads   # loads StructureFunctionsOhMyThreadsExt → enables ThreadedBackend
using KernelAbstractions: KernelAbstractions as KA
using CairoMakie: CairoMakie as CM
using Statistics: Statistics
using Random: Random

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS_DIR)

"""Bin midpoints from flat edge vector `[e₀, e₁, …, eₙ]` (N edges → N−1 midpoints)."""
bin_midpoints(edges) = [(edges[i] + edges[i + 1]) / 2 for i in 1:(length(edges) - 1)]

# ─── Shared synthetic field ────────────────────────────────────────────────

function make_synthetic_field(; N=2000, seed=42)
    Random.seed!(seed)
    # 2D scattered points in a 1000×1000 km domain
    # API expects (D, N) layout where D=2 is spatial dimension
    x = Random.rand(2, N) .* 1000.0
    # Velocity field with broad-spectrum turbulent character:
    # u ~ Σ_k A_k cos(2π k·x/L + φ_k), with amplitude A_k ~ k^(-5/6) → E(k) ~ k^(-5/3)
    L = 1000.0
    u = zeros(2, N)
    for k in 1:30
        amp = k^(-5.0/6.0)
        φ_u = 2π * Random.rand()
        φ_v = 2π * Random.rand()
        kx = Float64(k)
        ky = Float64(k ÷ 2 + 1)
        for i in 1:N
            arg = 2π * (kx * x[1,i] + ky * x[2,i]) / L
            u[1, i] += amp * cos(arg + φ_u)
            u[2, i] += amp * sin(arg + φ_v)
        end
    end
    return x, u
end

# ─── Figure 1: S2 Kolmogorov scaling ──────────────────────────────────────

function generate_kolmogorov_figure()
    x, u = make_synthetic_field(N=1500)

    r_min, r_max = 5.0, 400.0
    bins = exp.(range(log(r_min), log(r_max); length=29))

    op = SF.SecondOrderStructureFunctionType()
    result = SF.calculate_structure_function(op, x, u, bins;
        backend=SF.SerialBackend(), show_progress=false, verbose=false)

    sf2   = result.values
    rdist = bin_midpoints(result.distance)
    valid = sf2 .> 0

    fig = CM.Figure(size=(820, 520), fontsize=14)
    CM.Label(fig[0, 1],
        "2nd-Order Structure Function — Kolmogorov Scaling",
        fontsize=16, font=:bold)

    ax = CM.Axis(fig[1, 1],
        xlabel="Separation r  [km]",
        ylabel="S₂(r)",
        xscale=CM.log10, yscale=CM.log10,
        title="S₂(r) vs K41 prediction r^(2/3)")

    CM.scatterlines!(ax, rdist[valid], sf2[valid],
        label="S₂(r)  (computed)", color=:steelblue,
        markersize=7, linewidth=1.5)

    # K41 reference line fit through the middle third
    n = sum(valid)
    if n >= 4
        i0 = max(1, n ÷ 3); i1 = min(n, 2*n ÷ 3)
        r_mid = rdist[valid][i0:i1]
        sf_mid = sf2[valid][i0:i1]
        A = exp(Statistics.mean(log.(sf_mid) .- (2/3) .* log.(r_mid)))
        r_ref = range(rdist[valid][1], rdist[valid][end], length=100)
        CM.lines!(ax, collect(r_ref), A .* collect(r_ref).^(2/3),
            label="K41:  A·r^(2/3)", color=:crimson,
            linewidth=2, linestyle=:dash)
    end

    CM.axislegend(ax, position=:lt)

    outpath = joinpath(ASSETS_DIR, "sf_kolmogorov.png")
    CM.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 2: Longitudinal vs Transverse ─────────────────────────────────

function generate_long_vs_trans_figure()
    x, u = make_synthetic_field(N=1500)

    r_min, r_max = 5.0, 400.0
    bins = exp.(range(log(r_min), log(r_max); length=29))

    op_L = SF.LongitudinalSecondOrderStructureFunctionType()
    op_T = SF.TransverseSecondOrderStructureFunctionType()

    res_L = SF.calculate_structure_function(op_L, x, u, bins;
        backend=SF.SerialBackend(), show_progress=false, verbose=false)
    res_T = SF.calculate_structure_function(op_T, x, u, bins;
        backend=SF.SerialBackend(), show_progress=false, verbose=false)

    sf_L = res_L.values
    sf_T = res_T.values
    rd   = bin_midpoints(res_L.distance)
    vL   = sf_L .> 0
    vT   = sf_T .> 0

    fig = CM.Figure(size=(820, 520), fontsize=14)
    CM.Label(fig[0, 1],
        "Longitudinal vs Transverse 2nd-Order Structure Functions",
        fontsize=16, font=:bold)

    ax = CM.Axis(fig[1, 1],
        xlabel="Separation r  [km]", ylabel="S₂(r)",
        xscale=CM.log10, yscale=CM.log10,
        title="L2SF and T2SF on the same synthetic turbulent field")

    CM.lines!(ax, rd[vL], sf_L[vL],
        label="Longitudinal  L2SF", color=:steelblue, linewidth=2)
    CM.lines!(ax, rd[vT], sf_T[vT],
        label="Transverse    T2SF", color=:darkorange, linewidth=2, linestyle=:dash)
    CM.axislegend(ax, position=:lt)

    outpath = joinpath(ASSETS_DIR, "sf_long_vs_trans.png")
    CM.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 3: Backend parity (Serial vs Threaded) ────────────────────────

function generate_parity_figure()
    Random.seed!(7)
    N = 800
    x = Random.rand(2, N) .* 500.0
    u = randn(2, N)
    bins = exp.(range(log(5.0), log(200.0); length=21))

    op = SF.SecondOrderStructureFunctionType()

    res_serial = SF.calculate_structure_function(op, x, u, bins;
        backend=SF.SerialBackend(), show_progress=false, verbose=false)

    res_thread = SF.calculate_structure_function(op, x, u, bins;
        backend=SF.ThreadedBackend(), show_progress=false, verbose=false)

    sf_s = res_serial.values
    sf_t = res_thread.values
    rd   = bin_midpoints(res_serial.distance)
    diff = abs.(sf_s .- sf_t)
    rel  = diff ./ (abs.(sf_s) .+ 1e-300)

    fig = CM.Figure(size=(1100, 480), fontsize=14)
    CM.Label(fig[0, 1:2],
        "Backend Parity: Serial vs Threaded",
        fontsize=16, font=:bold)

    ax1 = CM.Axis(fig[1, 1],
        xlabel="Separation r", ylabel="S₂(r)",
        xscale=CM.log10, yscale=CM.log10,
        title="S₂(r) — Serial and Threaded (overlapping)")
    CM.lines!(ax1, rd, sf_s, label="Serial",   color=:steelblue, linewidth=2)
    CM.lines!(ax1, rd, sf_t, label="Threaded", color=:crimson,
        linewidth=1.5, linestyle=:dash)
    CM.axislegend(ax1, position=:lt)

    ax2 = CM.Axis(fig[1, 2],
        xlabel="Separation r", ylabel="|Serial − Threaded| / |Serial|",
        xscale=CM.log10,
        title="Relative difference (should be ≈ 0)")
    v = sf_s .> 0
    CM.scatterlines!(ax2, rd[v], rel[v],
        color=:darkorange, markersize=6, linewidth=1)
    CM.hlines!(ax2, [1e-14]; color=:black, linewidth=0.8, linestyle=:dot)

    outpath = joinpath(ASSETS_DIR, "sf_backend_parity.png")
    CM.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 4: GPU kernel parity (Serial vs KA.CPU) ───────────────────────

function generate_gpu_parity_figure()
    Random.seed!(7)
    N = 80
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0, 1.4; length = 11))
    sft = SF.L2SFType()

    res_serial = SFC.calculate_structure_function(
        sft, x, u, bin_edges;
        verbose = false, show_progress = false, output_type = SF.StructureFunctionSumsAndCounts,
    )
    res_gpu = SFC.gpu_calculate_structure_function(
        sft, KA.CPU(), x, u, bin_edges,
    )

    rd = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in 1:(length(bin_edges) - 1)]
    sf_s = res_serial.sums ./ max.(res_serial.counts, 1)
    sf_g = res_gpu.sums ./ max.(res_gpu.counts, 1)
    rel = abs.(sf_s .- sf_g) ./ (abs.(sf_s) .+ 1e-300)

    fig = CM.Figure(size = (1100, 480), fontsize = 14)
    CM.Label(fig[0, 1:2],
        "GPU Kernel Parity: Serial CPU vs KA.CPU()",
        fontsize = 16, font = :bold)

    ax1 = CM.Axis(fig[1, 1],
        xlabel = "Bin center", ylabel = "Mean SF sample",
        title = "Per-bin means (overlapping)")
    CM.lines!(ax1, rd, sf_s, label = "Serial CPU", color = :steelblue, linewidth = 2)
    CM.lines!(ax1, rd, sf_g, label = "KA.CPU GPU kernel", color = :crimson,
        linewidth = 1.5, linestyle = :dash)
    CM.axislegend(ax1, position = :lt)

    ax2 = CM.Axis(fig[1, 2],
        xlabel = "Bin center", ylabel = "|Serial − KA.CPU| / |Serial|",
        title = "Relative difference (should be ≈ 0)")
    CM.scatterlines!(ax2, rd, rel, color = :darkorange, markersize = 6, linewidth = 1)
    CM.hlines!(ax2, [1e-12]; color = :black, linewidth = 0.8, linestyle = :dot)

    outpath = joinpath(ASSETS_DIR, "sf_gpu_parity.png")
    CM.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Execute ──────────────────────────────────────────────────────────────

# ─── Figure 5: Single-pass invariants + Helmholtz ─────────────────────────

function generate_single_pass_figure()
    x, u = make_synthetic_field(N=1500)

    r_min, r_max = 5.0, 400.0
    bins = exp.(range(log(r_min), log(r_max); length=29))
    rdist = bin_midpoints(bins)

    # One O(N²) pass → NamedTuple of the six isotropic invariants; point-field input also
    # yields a `:helmholtz` entry (rotational/divergent decomposition).
    res = SF.calculate_structure_functions_single_pass(x, u, bins; backend=SF.SerialBackend())

    fig = CM.Figure(size=(900, 560), fontsize=14)
    CM.Label(fig[0, 1],
        "Single-Pass Invariants + Helmholtz Decomposition",
        fontsize=16, font=:bold)
    ax = CM.Axis(fig[1, 1],
        xlabel="Separation r  [km]", ylabel="|S(r)|",
        xscale=CM.log10, yscale=CM.log10,
        title="Six isotropic invariants + rotational/divergent — one pass")

    for k in (:S2, :L2, :T2, :S3, :L3, :L1T2)
        v = abs.(getproperty(res, k).values) .+ 1e-12
        CM.scatterlines!(ax, rdist, v; label=string(k), markersize=5, linewidth=1.3)
    end
    h = res.helmholtz
    rot = abs.(h.rotational_sums ./ max.(h.rotational_counts, 1)) .+ 1e-12
    div = abs.(h.divergent_sums ./ max.(h.divergent_counts, 1)) .+ 1e-12
    CM.lines!(ax, rdist, rot; label="Rotational", color=:black, linestyle=:dash, linewidth=2)
    CM.lines!(ax, rdist, div; label="Divergent", color=:gray, linestyle=:dashdot, linewidth=2)
    CM.axislegend(ax, position=:rb, nbanks=2)

    outpath = joinpath(ASSETS_DIR, "sf_single_pass.png")
    CM.save(outpath, fig)
    println("Saved: $outpath")
end

println("Generating StructureFunctions.jl figure assets...")
generate_kolmogorov_figure()
generate_long_vs_trans_figure()
generate_single_pass_figure()
generate_parity_figure()
generate_gpu_parity_figure()

const GPU_JSON = joinpath(@__DIR__, "..", "..", "gpu", "benchmark_results", "assets_latest.json")
if isfile(GPU_JSON)
    println("Found GPU benchmark JSON — run generate_gpu_figures.jl for scaling plots.")
else
    @warn "No gpu/benchmark_results/assets_latest.json — skip GPU scaling figures (run collect_benchmark_assets.jl on GPU)"
end
println("Done.")
