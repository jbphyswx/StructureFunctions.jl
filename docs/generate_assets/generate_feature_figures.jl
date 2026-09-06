"""
Generate figure assets for the gridded, directional, multi-channel and spectral-transform features.

Run from the repo root:
    julia --project=docs/generate_assets docs/generate_assets/generate_feature_figures.jl
"""

using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using SpectralBackends: SpectralBackends as SB
using Bessels: Bessels
using FFTW: FFTW
using CairoMakie: CairoMakie as CM
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using LinearAlgebra: LinearAlgebra
using Random: Random

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS_DIR)

# ─── Shared fields ─────────────────────────────────────────────────────────

"""Divergence-free 2-D field on a periodic grid with a prescribed `k^(-5/3)` energy spectrum."""
function spectral_grid_field(n; kmin = 2, kmax = 100, seed = 5)
    rng = Random.MersenneTwister(seed)
    Fx = zeros(ComplexF64, n ÷ 2 + 1, n)
    Fy = zeros(ComplexF64, n ÷ 2 + 1, n)
    for ix in 1:(n ÷ 2 + 1), iy in 1:n
        kx = ix - 1
        ky = iy - 1 <= n ÷ 2 ? iy - 1 : iy - 1 - n
        k = sqrt(kx^2 + ky^2)
        (kmin <= k <= kmax) || continue
        amp = k^(-4 / 3) * n^2
        ph = exp(im * 2π * rand(rng))
        Fx[ix, iy] = amp * ph * (-ky / k)
        Fy[ix, iy] = amp * ph * (kx / k)
    end
    u = zeros(2, n, n)
    u[1, :, :] = FFTW.irfft(Fx, n)
    u[2, :, :] = FFTW.irfft(Fy, n)
    return u
end

# ─── Figure: canonical spectra recovered from a structure function ─────────

function generate_spectra_figure()
    n = 256
    dx = 2π / n
    u = spectral_grid_field(n)
    sched = SFC.UniformLagSchedule((n, n), (dx, dx), (true, true))
    kaxes, density = SFC.gridded_spectrum(u, sched, Val(2),
                                          SB.FastFourierTransformSpectralBackend())
    edges = collect(10 .^ range(log10(1.5), log10(90.0); length = 26))
    mids, E = SFC.shell_average(kaxes, density, edges)
    ok = E .> 0

    # closed-form check: a Gaussian correlation has an analytic spectral density in every dimension
    σ2, ℓ = 1.7, 0.8
    r = collect(range(0.0, 20ℓ; length = 20_000))
    s2 = @. 2σ2 * (1 - exp(-r^2 / (2ℓ^2)))
    kq = collect(range(1e-3, 12.0; length = 400))

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "wavenumber k",
                  ylabel = "E(k)", title = "Spectrum recovered from S₂ on a 256² grid")
    CM.lines!(ax1, mids[ok], E[ok]; linewidth = 3, label = "from the structure function")
    ref = mids[ok] .^ (-5 / 3)
    ref .*= E[ok][6] / ref[6]
    CM.lines!(ax1, mids[ok], ref; linestyle = :dash, linewidth = 2, color = :black,
              label = "prescribed k^(-5/3)")
    CM.axislegend(ax1; position = :lb)

    # Reporting the error directly, rather than overlaying two curves that would sit on top of one
    # another: it states the accuracy, and it shows honestly where quadrature noise takes over.
    ax2 = CM.Axis(fig[1, 2]; yscale = log10, xlabel = "wavenumber k",
                  ylabel = "|P − P_exact| / max(P_exact)",
                  title = "Isotropic transform vs a closed form (Gaussian correlation)")
    for (D, col) in ((1, :dodgerblue), (2, :seagreen), (3, :crimson))
        P = SFC.isotropic_spectrum(SFT.S2SFType(), r, s2, kq, Val(D); asymptote = 2σ2)
        exact = @. σ2 * ℓ^D * exp(-kq^2 * ℓ^2 / 2) / (2π)^(D / 2)
        rel = abs.(P .- exact) ./ maximum(exact)
        keep = rel .> 0
        CM.lines!(ax2, kq[keep], rel[keep]; color = col, linewidth = 3, label = "D = $D")
    end
    CM.axislegend(ax2; position = :rb)

    out = joinpath(ASSETS_DIR, "sf_spectra.png")
    CM.save(out, fig)
    println("  wrote $out")
end

# ─── Figure: a spectrum survives missing data, a direct transform does not ──

function generate_missing_data_figure()
    n = 64
    dx = 2π / n
    u = spectral_grid_field(n; kmin = 2, kmax = 24, seed = 11)
    scal = u[1, :, :]
    sched = SFC.UniformLagSchedule((n, n), (dx, dx), (true, true))
    kaxes, full = SFC.gridded_spectrum(u, sched, Val(2),
                                       SB.FastFourierTransformSpectralBackend())
    dk = (kaxes[1][2] - kaxes[1][1]) * (kaxes[2][2] - kaxes[2][1])
    edges = collect(10 .^ range(log10(1.5), log10(24.0); length = 18))
    mids, Efull = SFC.shell_average(kaxes, full, edges)

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "wavenumber k",
                  ylabel = "E(k)", title = "Spectrum with cells missing")
    okf = Efull .> 0
    CM.lines!(ax1, mids[okf], Efull[okf]; color = :black, linewidth = 3, label = "complete field")

    fracs = [0.1, 0.3, 0.5]
    sf_err = Float64[]
    naive_err = Float64[]
    for (i, frac) in pairs(fracs)
        Random.seed!(7)
        valid = rand(n * n) .> frac
        _, masked = SFC.gridded_spectrum(u, sched, Val(2),
                                         SB.FastFourierTransformSpectralBackend(); valid = valid)
        _, Em = SFC.shell_average(kaxes, masked, edges)
        okm = Em .> 0
        CM.lines!(ax1, mids[okm], Em[okm]; linewidth = 2, label = "$(round(Int, 100frac))% missing")
        push!(sf_err, maximum(abs, masked .- full) / maximum(full))

        naive = copy(scal)
        naive[.!reshape(valid, n, n)] .= 0.0
        nspec = abs2.(FFTW.fft(naive)) ./ (n * n)^2
        nspec[1, 1] = 0.0
        push!(naive_err, maximum(abs, nspec .- full .* dk) / maximum(full .* dk))
    end
    CM.axislegend(ax1; position = :lb)

    ax2 = CM.Axis(fig[1, 2]; xlabel = "fraction of cells missing", ylabel = "relative spectral error",
                  title = "Why go through the structure function")
    CM.barplot!(ax2, (1:3) .- 0.18, sf_err; width = 0.34, label = "via the structure function")
    CM.barplot!(ax2, (1:3) .+ 0.18, naive_err; width = 0.34, label = "zero-fill the gaps and FFT")
    ax2.xticks = (1:3, ["10%", "30%", "50%"])
    CM.axislegend(ax2; position = :lt)

    out = joinpath(ASSETS_DIR, "sf_missing_data.png")
    CM.save(out, fig)
    println("  wrote $out")
end

# ─── Figure: directional output ────────────────────────────────────────────

function generate_directional_figure()
    Random.seed!(21)
    N = 4000
    x = 2π .* rand(2, N)
    u = zeros(2, N)
    for p in 1:N
        u[1, p] = sin(5 * x[1, p])          # varies along x only
    end
    dist = collect(range(0.0, 2.0; length = 15))
    ang = collect(range(0, π; length = 25))
    j = SFC.serial_calculate_structure_function(
        SFT.L2SFType(), x, u, dist, ang;
        second_axis = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)),
        verbose = false, show_progress = false)
    prof = [sum(j.sums[:, a]) / max(sum(j.counts[:, a]), 1) for a in 1:(length(ang) - 1)]
    θ = [(ang[a] + ang[a + 1]) / 2 for a in 1:(length(ang) - 1)]

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; xlabel = "angle θ between separation and x̂ (rad)",
                  ylabel = "⟨δu_L²⟩", title = "S(r, θ) for a field varying along x only")
    CM.lines!(ax1, θ, prof; linewidth = 3)
    CM.vlines!(ax1, [π / 2]; linestyle = :dash, color = :black)
    CM.text!(ax1, π / 2 + 0.05, maximum(prof) * 0.5; text = "separations ⟂ to the variation\ncarry no increment")

    avg = j.sums ./ max.(j.counts, 1)
    ax2 = CM.Axis(fig[1, 2]; xlabel = "separation r", ylabel = "angle θ (rad)",
                  title = "the same, as a joint histogram")
    CM.heatmap!(ax2, [(dist[i] + dist[i + 1]) / 2 for i in 1:(length(dist) - 1)], θ, avg)

    out = joinpath(ASSETS_DIR, "sf_directional.png")
    CM.save(out, fig)
    println("  wrote $out")
end

# ─── Figure: the Helmholtz split, in separation and in wavenumber ──────────

function generate_helmholtz_spectra_figure()
    n_bins = 60
    edges = collect(10 .^ range(-2, 0.5; length = n_bins + 1))
    mids = SF.midpoints(edges)
    counts = ones(UInt32, n_bins)
    kq = collect(range(1.0, 60.0; length = 250))

    D_LL = [r^(2 / 3) for r in mids]
    D_TT = (5 / 3) .* D_LL                       # solenoidal: D_div ≡ 0
    h_rot = SFC.helmholtz_decompose_2d(edges, D_LL, counts, D_TT, counts)
    h_div = SFC.helmholtz_decompose_2d(edges, D_TT, counts, D_LL, counts)   # irrotational

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; xscale = log10, xlabel = "separation r",
                  ylabel = "structure function", title = "Helmholtz split of a solenoidal field")
    CM.lines!(ax1, mids, D_LL; label = "D_LL", linewidth = 2)
    CM.lines!(ax1, mids, D_TT; label = "D_TT", linewidth = 2)
    CM.lines!(ax1, mids, h_rot.rotational_sums; label = "rotational", linewidth = 3,
              linestyle = :dash)
    CM.lines!(ax1, mids, h_rot.divergent_sums; label = "divergent (true value 0)", linewidth = 3,
              color = :crimson)
    CM.axislegend(ax1; position = :lt)

    sr = SFC.helmholtz_spectra(h_rot, kq)
    sd = SFC.helmholtz_spectra(h_div, kq)
    ax2 = CM.Axis(fig[1, 2]; xlabel = "wavenumber k", ylabel = "spectral density",
                  title = "and the spectra it transforms to")
    CM.lines!(ax2, kq, sr.rotational; linewidth = 3, label = "E_rot, solenoidal field")
    CM.lines!(ax2, kq, sr.divergent; linewidth = 2, color = :crimson,
              label = "E_div, solenoidal field")
    CM.lines!(ax2, kq, sd.rotational; linewidth = 2, linestyle = :dot,
              label = "E_rot, irrotational field")
    CM.lines!(ax2, kq, sd.divergent; linewidth = 3, linestyle = :dash,
              label = "E_div, irrotational field")
    CM.axislegend(ax2; position = :rt)

    out = joinpath(ASSETS_DIR, "sf_helmholtz_spectra.png")
    CM.save(out, fig)
    println("  wrote $out")
end

# ─── Figure: scalar and multi-channel structure functions ──────────────────

function generate_channels_figure()
    Random.seed!(33)
    N = 4000
    x = 2π .* rand(2, N)
    u = zeros(2, N)
    θ = zeros(1, N)
    for p in 1:N
        u[1, p] = cos(3 * x[1, p]) + 0.5cos(7 * x[2, p])
        u[2, p] = -sin(3 * x[2, p])
        θ[1, p] = cos(4 * x[1, p] + 0.3) + 0.4sin(9 * x[2, p])
    end
    fields = SF.Fields(vectors = (u,), scalars = (θ,))
    bins = collect(10 .^ range(log10(0.05), log10(2.5); length = 22))
    mids = SF.midpoints(bins)

    scal = SFC.calculate_structure_function(SFT.ScalarSFType{2}(), x, fields, bins;
                                            output_type = SF.StructureFunction)
    vel = SFC.calculate_structure_function(SFT.S2SFType(), x, fields, bins;
                                           output_type = SF.StructureFunction)
    yag = SFC.calculate_structure_function(SFT.MixedSFType{1, 0, 2}(), x, fields, bins;
                                           output_type = SF.StructureFunction)

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "separation r",
                  ylabel = "structure function",
                  title = "One pass over a velocity + tracer bundle")
    CM.lines!(ax1, mids, vel.values; linewidth = 3, label = "⟨‖δu‖²⟩  (velocity)")
    CM.lines!(ax1, mids, scal.values; linewidth = 3, label = "⟨(δθ)²⟩  (tracer)")
    CM.axislegend(ax1; position = :lt)

    ax2 = CM.Axis(fig[1, 2]; xscale = log10, xlabel = "separation r",
                  ylabel = "⟨δu_L (δθ)²⟩", title = "the mixed moment Yaglom's law inverts")
    CM.lines!(ax2, mids, yag.values; linewidth = 3, color = :seagreen)
    CM.hlines!(ax2, [0.0]; color = :black, linestyle = :dash)

    out = joinpath(ASSETS_DIR, "sf_channels.png")
    CM.save(out, fig)
    println("  wrote $out")
end

# ─── Figure: the two gridded algorithms, against each other and on cost ────

function generate_gridded_algorithms_figure()
    dx(n) = 2π / n
    bins(n) = collect(range(0.0, 2.6; length = 41))

    # agreement on one grid, then cost across sizes
    n0 = 96
    u0 = spectral_grid_field(n0; kmin = 2, kmax = 30, seed = 3)
    s0 = SFC.UniformLagSchedule((n0, n0), (dx(n0), dx(n0)), (true, true))
    b0 = bins(n0)
    plan0 = SFC.squared_digitize_plan(b0)
    nb0 = SFC.n_histogram_bins(plan0)
    sweep_s = zeros(Float64, nb0); sweep_c = zeros(Int, nb0)
    SFC.gridded_lag_sweep!(sweep_s, sweep_c, SFT.L2SFType(), u0, s0, b0, Val(2))
    fft_s = zeros(Float64, nb0); fft_c = zeros(Int, nb0)
    SFC.gridded_sweep!(fft_s, fft_c, SFT.L2SFType(), u0, s0, b0, Val(2),
                       SB.FastFourierTransformSpectralBackend())
    mids0 = SF.midpoints(b0)
    ok = sweep_c .> 0
    rel = abs.(fft_s[ok] ./ fft_c[ok] .- sweep_s[ok] ./ sweep_c[ok]) ./
          maximum(abs, sweep_s[ok] ./ sweep_c[ok])

    ns = [32, 48, 64, 96, 128]
    t_sweep = Float64[]; t_fft = Float64[]
    for n in ns
        u = spectral_grid_field(n; kmin = 2, kmax = min(30, n ÷ 3), seed = 3)
        s = SFC.UniformLagSchedule((n, n), (dx(n), dx(n)), (true, true))
        b = bins(n)
        nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(b))
        ss = zeros(Float64, nb); sc = zeros(Int, nb)
        fs = zeros(Float64, nb); fc = zeros(Int, nb)
        SFC.gridded_lag_sweep!(ss, sc, SFT.L2SFType(), u, s, b, Val(2))         # warm up
        SFC.gridded_sweep!(fs, fc, SFT.L2SFType(), u, s, b, Val(2),
                           SB.FastFourierTransformSpectralBackend())
        fill!(ss, 0); fill!(sc, 0); fill!(fs, 0); fill!(fc, 0)
        t1 = time(); SFC.gridded_lag_sweep!(ss, sc, SFT.L2SFType(), u, s, b, Val(2))
        push!(t_sweep, time() - t1)
        t2 = time(); SFC.gridded_sweep!(fs, fc, SFT.L2SFType(), u, s, b, Val(2),
                                        SB.FastFourierTransformSpectralBackend())
        push!(t_fft, time() - t2)
    end

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; yscale = log10, xlabel = "separation r",
                  ylabel = "relative difference",
                  title = "Transform vs lag sweep: two algorithms, one definition")
    CM.lines!(ax1, collect(mids0)[ok], max.(rel, 1e-17); linewidth = 3)
    CM.hlines!(ax1, [1e-16]; color = :black, linestyle = :dash)
    CM.text!(ax1, collect(mids0)[ok][3], 3e-16; text = "double-precision round-off")

    ax2 = CM.Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = "grid side n (n² cells)",
                  ylabel = "time per call (s)", title = "and what each one costs")
    CM.scatterlines!(ax2, Float64.(ns), t_sweep; linewidth = 3, label = "lag sweep")
    CM.scatterlines!(ax2, Float64.(ns), t_fft; linewidth = 3, label = "transform")
    CM.axislegend(ax2; position = :lt)

    out = joinpath(ASSETS_DIR, "sf_gridded_algorithms.png")
    CM.save(out, fig)
    println("  wrote $out  (speedup at n=$(ns[end]): $(round(t_sweep[end]/t_fft[end]; digits=1))x)")
end

# ─── Figure: advective structure functions and the Bessel spectral flux ────

"""Multi-mode 2-D solenoidal field whose wavevectors form triads, with `(u·∇)u` in closed form."""
function triad_field(x)
    kv = [[1, 0], [0, 1], [1, 1], [2, 1], [1, -1], [2, 0], [3, 1], [2, 2],
          [3, 0], [1, 2], [3, 2], [2, -1]]
    cm = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38, 0.35, 0.32]
    Random.seed!(3)
    ph = 2π .* rand(length(kv))
    ev = [[-k[2], k[1]] ./ sqrt(k[1]^2 + k[2]^2) for k in kv]
    N = size(x, 2)
    u = zeros(2, N)
    a = zeros(2, N)
    for p in 1:N
        for m in eachindex(kv)
            c = cm[m] * cos(kv[m][1] * x[1, p] + kv[m][2] * x[2, p] + ph[m])
            u[1, p] += c * ev[m][1]
            u[2, p] += c * ev[m][2]
        end
        for m in eachindex(kv), q in eachindex(kv)
            s = sin(kv[m][1] * x[1, p] + kv[m][2] * x[2, p] + ph[m])
            c = cos(kv[q][1] * x[1, p] + kv[q][2] * x[2, p] + ph[q])
            w = -cm[m] * cm[q] * (ev[q][1] * kv[m][1] + ev[q][2] * kv[m][2]) * c * s
            a[1, p] += w * ev[m][1]
            a[2, p] += w * ev[m][2]
        end
    end
    return u, a
end

function generate_advective_figure()
    Random.seed!(17)
    N = 9000
    x = 2π .* rand(2, N)
    u, a = triad_field(x)
    bins = collect(range(0.0, 3.0; length = 46))
    mids = SF.midpoints(bins)

    asf = SFC.calculate_structure_function(
        SFT.VectorDotSFType(1, 2), x, SF.Fields(vectors = (u, a)), bins;
        output_type = SF.StructureFunctionSumsAndCounts)
    Ks = collect(range(1.5, 9.0; length = 60))
    flux = SFC.spectral_flux(asf, Ks)

    vals = asf.sums ./ max.(asf.counts, 1)
    okb = asf.counts .> 0

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; xlabel = "separation r", ylabel = "⟨δu · δ𝓐ᵤ⟩",
                  title = "Advective structure function, and the flux it gives")
    CM.lines!(ax1, collect(mids)[okb], vals[okb]; linewidth = 3, label = "⟨δu · δ𝓐ᵤ⟩")
    CM.hlines!(ax1, [0.0]; color = :black, linestyle = :dash)
    ax1r = CM.Axis(fig[1, 1]; yaxisposition = :right, ylabel = "spectral flux Π(K)",
                   xlabel = "wavenumber K", xaxisposition = :top, ygridvisible = false)
    CM.lines!(ax1r, Ks, flux; linewidth = 3, color = :seagreen, linestyle = :dash)
    CM.text!(ax1r, Ks[35], flux[35]; text = "  Π(K)", color = :seagreen)

    # The kernel and its prefactor are exact against a closed form: ∫₀^R J₁(Kr)dr = (1 − J₀(KR))/K,
    # so a constant advective structure function `c` must give Π_K = −(c/2)(1 − J₀(KR)).
    c, R = 0.8, 60.0
    rq = collect(range(0.0, R; length = 200_000))
    Kq = collect(range(0.2, 20.0; length = 300))
    got = SFC.spectral_flux(SFT.VectorDotSFType(1, 2), rq, fill(c, length(rq)), Kq)
    want = [-(c / 2) * (1 - Bessels.besselj0(K * R)) for K in Kq]

    ax2 = CM.Axis(fig[1, 2]; xlabel = "wavenumber K", ylabel = "Π(K) for a constant SF_A",
                  title = "the flux kernel against a closed form")
    CM.lines!(ax2, Kq, got; linewidth = 4, label = "−(K/2)∫₀^R SF_A J₁(Kr) dr")
    CM.lines!(ax2, Kq, want; linewidth = 2, color = :black, linestyle = :dash,
              label = "−(c/2)(1 − J₀(KR)), exact")
    CM.hlines!(ax2, [-c / 2]; color = :gray, linestyle = :dot)
    CM.text!(ax2, 12.0, -c / 2 + 0.012; text = "−c/2, the whole-line value", color = :gray)
    CM.axislegend(ax2; position = :rt)

    out = joinpath(ASSETS_DIR, "sf_advective_flux.png")
    CM.save(out, fig)
    println("  wrote $out")
end

# ─── Figure: the third-order exact laws ────────────────────────────────────

function generate_exact_laws_figure()
    # (a) the inversions, on data that obeys each law exactly
    r = collect(range(0.2, 4.0; length = 60))
    ε, εθ = 0.85, 0.42
    eps45 = SF.KHM.epsilon_from_four_fifths(r, -(4 / 5) * ε .* r)
    eps43 = SF.KHM.epsilon_from_four_thirds(r, -(4 / 3) * ε .* r)
    epsY  = SF.KHM.epsilon_theta_from_yaglom(r, -(4 / 3) * εθ .* r)
    # the classic trap: the 4/5 law applied to the scalar moment is off by exactly 5/3
    wrong = SF.KHM.epsilon_from_four_fifths(r, -(4 / 3) * εθ .* r)

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; xlabel = "separation r", ylabel = "recovered dissipation",
                  title = "Each law inverts the moment it is stated for")
    CM.lines!(ax1, r, eps45; linewidth = 3, label = "4/5 law from ⟨δu_L³⟩ → ε")
    CM.lines!(ax1, r, eps43; linewidth = 3, linestyle = :dash,
              label = "4/3 law from ⟨δu_L‖δu‖²⟩ → ε")
    CM.lines!(ax1, r, epsY; linewidth = 3, label = "Yaglom from ⟨δu_L(δθ)²⟩ → ε_θ")
    CM.lines!(ax1, r, wrong; linewidth = 2, color = :crimson, linestyle = :dot,
              label = "4/5 law on the scalar moment (wrong by 5/3)")
    CM.hlines!(ax1, [ε, εθ]; color = :black, linestyle = :dash)
    CM.axislegend(ax1; position = :rc)

    # (b) the cascade sign, on fields with and without one
    # Many modes with random directions, so the random-phase field is close to Gaussian and its odd
    # moments vanish; the ramp-cliff field differs only in the waveform, not the spectrum.
    Random.seed!(5)
    N = 6000
    xs = 2π .* rand(2, N)
    sym = zeros(2, N)
    casc = zeros(2, N)
    saw(θ) = mod(θ, 2π) / π - 1
    for _ in 1:60
        k = rand(1:12)
        ang = 2π * rand()
        kx, ky = k * cos(ang), k * sin(ang)
        amp = k^(-5 / 6) * randn()
        φ = 2π * rand()
        ex, ey = -sin(ang), cos(ang)          # transverse, so both fields are solenoidal
        for p in 1:N
            arg = kx * xs[1, p] + ky * xs[2, p] + φ
            sym[1, p] += amp * cos(arg) * ex
            sym[2, p] += amp * cos(arg) * ey
            casc[1, p] += amp * saw(arg) * ex
            casc[2, p] += amp * saw(arg) * ey
        end
    end
    bins = collect(10 .^ range(log10(0.08), log10(2.5); length = 22))
    mids = SF.midpoints(bins)
    skew(u) = begin
        l3 = SFC.calculate_structure_function(SFT.L3SFType(), xs, u, bins;
                                              output_type = SF.StructureFunction,
                                              verbose = false, show_progress = false)
        l2 = SFC.calculate_structure_function(SFT.L2SFType(), xs, u, bins;
                                              output_type = SF.StructureFunction,
                                              verbose = false, show_progress = false)
        l3.values ./ (l2.values .^ 1.5)
    end
    ax2 = CM.Axis(fig[1, 2]; xscale = log10, xlabel = "separation r",
                  ylabel = "⟨δu_L³⟩ / ⟨δu_L²⟩^(3/2)",
                  title = "Third order carries the cascade's sign")
    CM.lines!(ax2, mids, skew(sym); linewidth = 3, label = "random phases (no cascade)")
    CM.lines!(ax2, mids, skew(casc); linewidth = 3, label = "ramp-cliff (forward cascade)")
    CM.hlines!(ax2, [0.0]; color = :black, linestyle = :dash)
    CM.axislegend(ax2; position = :lb)

    out = joinpath(ASSETS_DIR, "sf_exact_laws.png")
    CM.save(out, fig)
    println("  wrote $out")
end

# ─── Figure: spherical geometry — parallel transport, and the zonal fast path ──

function generate_spherical_figure()
    Random.seed!(64)
    N = 3000
    Rearth = 6.371e6
    lon = 360 .* rand(N) .- 180
    lat = 120 .* rand(N) .- 60
    x = permutedims(hcat(lon, lat))
    Ω = 7.292e-5
    u = permutedims(hcat(Ω * Rearth .* cosd.(lat), zeros(N)))   # solid-body rotation

    bins = collect(range(0.0, 8.0e6; length = 21))
    mids = SF.midpoints(bins)
    sphere = SFC.calculate_structure_functions_single_pass(
        x, u, bins; distance_metric = DI.Haversine(Rearth))
    # the same data with no transport: treat lon/lat as if they were a plane
    flatb = collect(range(0.0, 80.0; length = 21))
    flat = SFC.calculate_structure_functions_single_pass(
        x, u, flatb; distance_metric = DI.Euclidean())

    ratio(res) = begin
        occ = isfinite.(res.L2.values) .& isfinite.(res.S2.values) .& (res.S2.values .> 0)
        occ, abs.(res.L2.values[occ]) ./ res.S2.values[occ]
    end
    o1, r1 = ratio(sphere)
    o2, r2 = ratio(flat)

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; yscale = log10, xlabel = "separation (fraction of the range)",
                  ylabel = "⟨δu_L²⟩ / ⟨‖δu‖²⟩",
                  title = "Solid-body rotation has no longitudinal increment")
    CM.lines!(ax1, range(0, 1; length = count(o1)), max.(r1, 1e-30); linewidth = 3,
              label = "transported (geodesic frame)")
    CM.lines!(ax1, range(0, 1; length = count(o2)), max.(r2, 1e-30); linewidth = 3,
              color = :crimson, label = "untransported (lon/lat as a plane)")
    CM.axislegend(ax1; position = :rc)

    # zonal lat-lon fast path vs the unstructured pair loop, on the same grid
    nlon, nlat = 48, 24
    lons = collect(range(0, 2π * (1 - 1 / nlon); length = nlon))
    lats = collect(range(-1.2, 1.2; length = nlat))
    ug = zeros(2, nlon, nlat)
    for i in 1:nlon, j in 1:nlat
        ug[1, i, j] = cos(3 * lons[i]) * cos(lats[j])
        ug[2, i, j] = sin(2 * lats[j])
    end
    sbins = collect(range(0.0, 2.4; length = 25))
    sched = SFC.ZonalLagSchedule(lats, nlon, lons[2] - lons[1], 1.0, true)
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(sbins))
    zs = zeros(Float64, nb); zc = zeros(Int, nb)
    t1 = time()
    SFC.gridded_lag_sweep!(zs, zc, SFT.L2SFType(), ug, sched, sbins, Val(2))
    t_zonal = time() - t1

    xs = zeros(2, nlon * nlat)
    us = zeros(2, nlon * nlat)
    for i in 1:nlon, j in 1:nlat
        q = (j - 1) * nlon + i
        xs[1, q] = lons[i]; xs[2, q] = lats[j]
        us[1, q] = ug[1, i, j]; us[2, q] = ug[2, i, j]
    end
    t2 = time()
    un = SFC.calculate_structure_function(SFT.L2SFType(), xs, us, sbins;
                                          distance_metric = DI.SphericalAngle(),
                                          output_type = SF.StructureFunctionSumsAndCounts,
                                          verbose = false, show_progress = false)
    t_unstr = time() - t2

    smids = SF.midpoints(sbins)
    okz = (zc .> 0) .& (un.counts .> 0)
    ax2 = CM.Axis(fig[1, 2]; xlabel = "separation (radians on the sphere)",
                  ylabel = "⟨δu_L²⟩",
                  title = "Lat-lon fast path vs the pair loop ($(round(t_unstr/t_zonal; digits=1))× faster)")
    CM.lines!(ax2, collect(smids)[okz], (zs ./ max.(zc, 1))[okz]; linewidth = 4,
              label = "zonal lag schedule")
    CM.lines!(ax2, collect(smids)[okz], (un.sums ./ max.(un.counts, 1))[okz]; linewidth = 2,
              color = :black, linestyle = :dash, label = "unstructured pair loop")
    CM.axislegend(ax2; position = :lt)

    out = joinpath(ASSETS_DIR, "sf_spherical.png")
    CM.save(out, fig)
    println("  wrote $out  (zonal $(round(t_unstr/t_zonal; digits=1))x, counts equal: $(zc == un.counts))")
end

# ─── Figure: culling changes the cost, never the answer ────────────────────

function generate_culling_figure()
    Random.seed!(88)
    N = 20_000
    x = rand(2, N)
    u = randn(2, N)
    fracs = [0.5, 0.2, 0.1, 0.05, 0.03]
    speed = Float64[]
    exact = Bool[]
    for f in fracs
        bins = collect(range(0.0, f; length = 16))
        a = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins;
                culling = SFC.NoCulling(), output_type = SF.StructureFunctionSumsAndCounts,
                verbose = false, show_progress = false)
        t1 = time()
        a = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins;
                culling = SFC.NoCulling(), output_type = SF.StructureFunctionSumsAndCounts,
                verbose = false, show_progress = false)
        tn = time() - t1
        b = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins;
                culling = SFC.AlwaysCulling(), output_type = SF.StructureFunctionSumsAndCounts,
                verbose = false, show_progress = false)
        t2 = time()
        b = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins;
                culling = SFC.AlwaysCulling(), output_type = SF.StructureFunctionSumsAndCounts,
                verbose = false, show_progress = false)
        tc = time() - t2
        push!(speed, tn / tc)
        push!(exact, a.counts == b.counts)
    end

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; xscale = log10, yscale = log10,
                  xlabel = "r_max / domain size", ylabel = "speedup from culling",
                  title = "Culling: cost falls with the cutoff")
    CM.scatterlines!(ax1, fracs, speed; linewidth = 3, markersize = 14)
    CM.hlines!(ax1, [1.0]; color = :black, linestyle = :dash)

    ax2 = CM.Axis(fig[1, 2]; xlabel = "r_max / domain size", ylabel = "pair counts identical?",
                  title = "and never changes the answer")
    CM.barplot!(ax2, 1:length(fracs), Float64.(exact); color = :seagreen)
    ax2.xticks = (1:length(fracs), string.(fracs))
    CM.ylims!(ax2, 0, 1.3)
    CM.text!(ax2, 1.0, 1.1; text = "exact pair-for-pair at every cutoff")

    out = joinpath(ASSETS_DIR, "sf_culling.png")
    CM.save(out, fig)
    println("  wrote $out  (speedups: $(round.(speed; digits=1)), all exact: $(all(exact)))")
end

# ─── Figure: covariance from a structure function ──────────────────────────

function generate_covariance_figure()
    σ2, ℓ = 1.7, 0.8
    edges = collect(range(0.0, 6.0; length = 61))
    mids = SF.midpoints(edges)
    d = [2σ2 * (1 - exp(-r^2 / (2ℓ^2))) for r in mids]
    res = SF.StructureFunction(SFT.S2SFType(), edges, d)
    r, C = SFC.covariance(res, σ2)

    fig = CM.Figure(size = (1100, 430))
    ax1 = CM.Axis(fig[1, 1]; xlabel = "separation r", ylabel = "value",
                  title = "C(r) = C(0) − D(r)/2")
    CM.lines!(ax1, mids, d; linewidth = 3, label = "structure function D(r)")
    CM.lines!(ax1, r, C; linewidth = 3, label = "recovered covariance C(r)")
    CM.lines!(ax1, r, [σ2 * exp(-q^2 / (2ℓ^2)) for q in r]; linewidth = 2, color = :black,
              linestyle = :dash, label = "true covariance")
    CM.hlines!(ax1, [σ2]; color = :gray, linestyle = :dot)
    CM.text!(ax1, 3.5, σ2 * 1.02; text = "variance, which D(r) cannot supply", color = :gray)
    CM.axislegend(ax1; position = :rc)

    # interpolating a positive-definite kernel does not preserve positive-definiteness
    Random.seed!(31)
    pts = 3.0 .* rand(2, 60)
    gauss(s) = 1.4 * exp(-s^2 / (2 * 0.9^2))
    nss = [60, 120, 250, 500, 1000, 2500, 5000]
    λmin = Float64[]
    for ns in nss
        seps = collect(range(0.0, 6.0; length = ns))
        Σ = SFC.covariance_matrix(pts, seps, gauss.(seps); check_posdef = false)
        push!(λmin, minimum(LinearAlgebra.eigvals(LinearAlgebra.Symmetric(Σ))))
    end
    ax2 = CM.Axis(fig[1, 2]; xscale = log10, yscale = log10,
                  xlabel = "samples of C(r) used to build the matrix",
                  ylabel = "|most negative eigenvalue|",
                  title = "Positive-definiteness is checked, not assumed")
    CM.scatterlines!(ax2, Float64.(nss), abs.(λmin); linewidth = 3, markersize = 12)
    CM.text!(ax2, 100.0, 3e-4; text = "under-resolved:\nnot a valid covariance matrix")
    CM.text!(ax2, 1200.0, 2e-8; text = "resolved")

    out = joinpath(ASSETS_DIR, "sf_covariance.png")
    CM.save(out, fig)
    println("  wrote $out")
end

println("Generating StructureFunctions.jl feature figures...")
generate_spectra_figure()
generate_missing_data_figure()
generate_directional_figure()
generate_helmholtz_spectra_figure()
generate_channels_figure()
generate_gridded_algorithms_figure()
generate_advective_figure()
generate_exact_laws_figure()
generate_spherical_figure()
generate_culling_figure()
generate_covariance_figure()
println("Done.")
