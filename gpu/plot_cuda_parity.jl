"""
    plot_cuda_parity.jl

Optional CUDA parity figure for docs (run on GPU allocation).

    julia --project=gpu gpu/plot_cuda_parity.jl

Writes `docs/src/assets/sf_gpu_cuda_parity.png` comparing serial CPU vs CUDA GPU
on the same small problem as `test_cuda_parity.jl`.
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random

function main()
    CUDA.functional() || error("CUDA not functional — run on a GPU allocation")

    Random.seed!(42)
    N = 500
    FT = Float32
    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)
    bins = collect(FT, range(0.0f0, 1.4f0; length = 11))
    sft = SFT.L2SFType()

    ref = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, bins;
        return_sums_and_counts = true, verbose = false, show_progress = false,
    )
    gpu_res = SFC.gpu_calculate_structure_function(
        sft, CUDA.CUDABackend(), x_gpu, u_gpu, bins; return_sums_and_counts = true,
    )
    CUDA.synchronize()

    rd = [(bins[i] + bins[i + 1]) / 2 for i in 1:(length(bins) - 1)]
    sf_ref = ref.sums ./ max.(ref.counts, 1)
    sf_gpu = gpu_res.sums ./ max.(gpu_res.counts, 1)
    rel = abs.(sf_ref .- sf_gpu) ./ (abs.(sf_ref) .+ 1e-300)

    @eval using CairoMakie: CairoMakie as CM
    assets = normpath(joinpath(@__DIR__, "..", "docs", "src", "assets"))
    mkpath(assets)

    fig = CM.Figure(size = (1100, 480), fontsize = 14)
    CM.Label(fig[0, 1:2],
        "CUDA Parity: Serial CPU vs CUDABackend (N=$N, Float32)",
        fontsize = 16, font = :bold)

    ax1 = CM.Axis(fig[1, 1], xlabel = "Bin center", ylabel = "Mean SF sample",
        title = "Per-bin means")
    CM.lines!(ax1, rd, sf_ref, label = "Serial CPU", color = :steelblue, linewidth = 2)
    CM.lines!(ax1, rd, sf_gpu, label = "CUDA GPU", color = :crimson,
        linewidth = 1.5, linestyle = :dash)
    CM.axislegend(ax1, position = :lt)

    ax2 = CM.Axis(fig[1, 2], xlabel = "Bin center",
        ylabel = "|Serial − CUDA| / |Serial|",
        title = "Relative difference")
    CM.scatterlines!(ax2, rd, rel, color = :darkorange, markersize = 6, linewidth = 1)

    out = joinpath(assets, "sf_gpu_cuda_parity.png")
    CM.save(out, fig, px_per_unit = 2)
    println("Saved: $out")
end

main()
