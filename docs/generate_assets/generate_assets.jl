"""
Generate static figure assets for StructureFunctions.jl docs and README.md.

Run from the repo root:
    julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl
"""

using StructureFunctions: StructureFunctions as SF
using OhMyThreads: OhMyThreads   # loads StructureFunctionsOhMyThreadsExt → enables ThreadedBackend
using CairoMakie: CairoMakie as CM
using Statistics: Statistics
using Random: Random

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS_DIR)

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
    edges = exp.(range(log(r_min), log(r_max); length=29))
    bins  = [(edges[i], edges[i+1]) for i in 1:length(edges)-1]

    op = SF.SecondOrderStructureFunctionType()
    result = SF.calculate_structure_function(op, x, u, bins;
        backend=SF.SerialBackend(), show_progress=false, verbose=false)

    sf2   = result.values
    rdist = [(t[1]+t[2])/2 for t in result.distance]
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
    edges = exp.(range(log(r_min), log(r_max); length=29))
    bins  = [(edges[i], edges[i+1]) for i in 1:length(edges)-1]

    op_L = SF.LongitudinalSecondOrderStructureFunctionType()
    op_T = SF.TransverseSecondOrderStructureFunctionType()

    res_L = SF.calculate_structure_function(op_L, x, u, bins;
        backend=SF.SerialBackend(), show_progress=false, verbose=false)
    res_T = SF.calculate_structure_function(op_T, x, u, bins;
        backend=SF.SerialBackend(), show_progress=false, verbose=false)

    sf_L = res_L.values
    sf_T = res_T.values
    rd   = [(t[1]+t[2])/2 for t in res_L.distance]
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
    edges = exp.(range(log(5.0), log(200.0); length=21))
    bins  = [(edges[i], edges[i+1]) for i in 1:length(edges)-1]

    op = SF.SecondOrderStructureFunctionType()

    res_serial = SF.calculate_structure_function(op, x, u, bins;
        backend=SF.SerialBackend(), show_progress=false, verbose=false)

    res_thread = SF.calculate_structure_function(op, x, u, bins;
        backend=SF.ThreadedBackend(), show_progress=false, verbose=false)

    sf_s = res_serial.values
    sf_t = res_thread.values
    rd   = [(t[1]+t[2])/2 for t in res_serial.distance]
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

# ─── Execute ──────────────────────────────────────────────────────────────

println("Generating StructureFunctions.jl figure assets...")
generate_kolmogorov_figure()
generate_long_vs_trans_figure()
generate_parity_figure()
println("Done.")
