# T time slices on one GPU: one call per host slice, one call per device slice with a workspace, and one batch call.
#
# Run on a GPU allocation:
#   ] activate gpu 
#   > include("gpu/benchmark_slices.jl")
#   > main(; T=8000, N=20000)

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random

include(joinpath(@__DIR__, "benchmark_scaling_helpers.jl"))

function bench_manual_loop!(backend, x_batch, u_batch, bins, sft, sums, counts, ws; T::Int)
    for t in 1:T
        res = SFC.calculate_structure_function(sft, view(x_batch, :, :, t), view(u_batch, :, :, t), bins, UInt32,
                                               StructureFunctionSumsAndCounts; backend = CB.GPUBackend(backend),
                                               workspace = ws)
        sums[:, t] .= Array(res.sums)
        counts[:, t] .= Array(res.counts)
    end
    return nothing
end

function _d2h_bytes_per_slice(NB::Int, ::Type{FT}) where {FT}
    return NB * (sizeof(FT) + sizeof(UInt32))
end

function main()
    Random.seed!(42)
    N = parse(Int, get(ENV, "N", "20000"))
    T = parse(Int, get(ENV, "T", "8000"))
    FT = Float32

    if !CUDA.functional()
        error("CUDA not functional — run on a GPU allocation")
    end
    backend = CUDA.CUDABackend()
    println("backend: ", typeof(backend))
    if hasproperty(CUDA, :name)
        println("device: ", CUDA.name(CUDA.device()))
    end
    println("N_points=$N  T=$T")

    bins = collect(FT, range(FT(0), FT(1.5), length = 65))
    NB = length(bins) - 1
    sft = SFT.L2SFType()
    ws = SFC.GPUSFWorkspace(backend, bins)

    bytes_slice = _d2h_bytes_per_slice(NB, FT)
    println(
        "D2H per slice (histogram only): $(bytes_slice) B " *
        "(NB=$NB sums+counts; not N_points=$N)",
    )
    println("Total histogram D2H if all slices read back: $(round(bytes_slice * T / 1024^2; digits=3)) MiB")
    h2d_input = 2 * 2 * N * T * sizeof(FT)
    println("One-time H2D for x,u (2, N, T): $(round(h2d_input / 1024^3; digits=3)) GiB")

    x_host = rand(FT, 2, N, T)
    u_host = rand(FT, 2, N, T)
    x_batch = CUDA.cu(x_host)
    u_batch = CUDA.cu(u_host)

    sums_a = zeros(FT, NB, T)
    counts_a = zeros(UInt32, NB, T)
    sums_b = zeros(FT, NB, T)
    counts_b = zeros(UInt32, NB, T)
    sums_c = KA.zeros(backend, FT, NB, T)
    counts_c = KA.zeros(backend, UInt32, NB, T)

    println()
    println("--- naive_loop: one call per host slice ---")
    t_naive = bench_naive_slice_loop!(backend, x_host, u_host, bins, sft, sums_a, counts_a; T = T)

    println("--- manual_loop_ws: one call per device slice, with a GPUSFWorkspace ---")
    t_manual = run_timed_gpu(
        () -> bench_manual_loop!(backend, x_batch, u_batch, bins, sft, sums_b, counts_b, ws; T = T), backend,
    )

    println("--- slice_driver: calculate_structure_function_batch! ---")
    t_slice = bench_slice_driver!(backend, x_batch, u_batch, bins, sft, sums_c, counts_c, ws)

    sums_c_host, counts_c_host = Array(sums_c), Array(counts_c)
    maxΔ_manual = maximum(abs.(sums_b .- sums_c_host))
    maxΔ_naive = maximum(abs.(sums_a .- sums_c_host))
    counts_ok = counts_a == counts_c_host && counts_b == counts_c_host
    println()
    println("naive_loop:     total=$(round(t_naive, digits=3))s  per_slice=$(round(1000 * t_naive / T; digits=3))ms")
    println("manual_loop_ws: total=$(round(t_manual, digits=3))s  per_slice=$(round(1000 * t_manual / T; digits=3))ms")
    println("slice_driver:   total=$(round(t_slice, digits=3))s  per_slice=$(round(1000 * t_slice / T; digits=3))ms")
    println("speedup slice vs naive:  $(round(t_naive / t_slice; digits=2))×")
    println("speedup slice vs manual: $(round(t_manual / t_slice; digits=2))×")
    println("parity vs slice: max|Δsums| naive=$(round(maxΔ_naive, digits=4)) manual=$(round(maxΔ_manual, digits=4))  counts_equal=$counts_ok")

    SFC.release!(ws)
    return nothing
end

main()
