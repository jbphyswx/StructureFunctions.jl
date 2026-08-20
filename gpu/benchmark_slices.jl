# Slice-batch benchmark: naive sequential GPU calls vs optimized manual vs slice driver
#
# Run on a GPU node (SLURM):
#   julia --project=gpu gpu/benchmark_slices.jl
#   T=8000 N=20000 julia --project=gpu gpu/benchmark_slices.jl

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random

include(joinpath(@__DIR__, "benchmark_scaling_helpers.jl"))

function bench_manual_loop!(
    backend,
    x_batch,
    u_batch,
    bins,
    sft,
    sums,
    counts,
    ws;
    T::Int,
)
    for t in 1:T
        res = SFC.gpu_calculate_structure_function(
            sft, backend, view(x_batch, :, :, t), view(u_batch, :, :, t), bins;
            workspace = ws,
        )
        sums[:, t] .= res.sums
        counts[:, t] .= res.counts
    end
    gpu_sync!(backend)
    return nothing
end

function _d2h_bytes_per_slice(NB::Int, ::Type{FT}) where {FT}
    return NB * (sizeof(FT) + sizeof(UInt32))
end

function main()
    Random.seed!(42)
    N = parse(Int, get(ENV, "N", "20000"))
    T = parse(Int, get(ENV, "T", "8000"))
    warmup = parse(Int, get(ENV, "WARMUP", "1"))
    FT = Float32

    if !CUDA.functional()
        error("CUDA not functional — run on a GPU allocation")
    end
    backend = CUDA.CUDABackend()
    println("backend: ", typeof(backend))
    if hasproperty(CUDA, :name)
        println("device: ", CUDA.name(CUDA.device()))
    end
    println("N_points=$N  T=$T  warmup=$warmup")

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
    sums_c = zeros(FT, NB, T)
    counts_c = zeros(UInt32, NB, T)

    println()
    println("--- naive_loop: host slice each t, fresh device alloc every call ---")
    t_naive = run_timed_gpu(
        () -> bench_naive_slice_loop!(backend, x_host, u_host, bins, sft, sums_a, counts_a; T = T),
        backend; warmup = warmup,
    )

    println("--- manual_loop_ws: CuArray batch + views + GPUSFWorkspace (expert setup) ---")
    t_manual = run_timed_gpu(
        () -> bench_manual_loop!(backend, x_batch, u_batch, bins, sft, sums_b, counts_b, ws; T = T),
        backend; warmup = warmup,
    )

    println("--- slice_driver: calculate_structure_function_batch! ---")
    t_slice = run_timed_gpu(
        () -> bench_slice_driver!(backend, x_batch, u_batch, bins, sft, sums_c, counts_c, ws),
        backend; warmup = warmup,
    )

    maxΔ_manual = maximum(abs.(sums_b .- sums_c))
    maxΔ_naive = maximum(abs.(sums_a .- sums_c))
    counts_ok = counts_a == counts_c && counts_b == counts_c
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
