"""
    benchmark_cuda.jl

GPU vs CPU timing at the **same** N, bins, and SF type.

Compares three backends:
  - `ThreadedBackend()` — production CPU path (same as `benchmark/benchmark_worker.jl`)
  - `KA.CPU()` — GPU extension kernel on CPU (parity reference only)
  - `CUDA.CUDABackend()` — GPU

Run once per SLURM session. See `gpu/README.md`:

    include(joinpath("/path/to/StructureFunctions.jl", "gpu", "benchmark_cuda.jl"))
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, LinearBinEdges
using Random: Random

function main()
    if !CUDA.functional()
        println("CUDA not functional — skipping benchmark.")
        return
    end

    Random.seed!(42)
    N = parse(Int, get(ENV, "N", "20000"))
    FT = Float32
    nthreads = Threads.nthreads()

    println("Device: ", CUDA.name(CUDA.device()))
    println("Julia threads: ", nthreads)
    println("N = $N  (O(N²) pairs, 2D, Float32, L2SFType)")
    println()

    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    sft = SFT.L2SFType()
    x_tup = (x_cpu[1, :], x_cpu[2, :])
    u_tup = (u_cpu[1, :], u_cpu[2, :])

    threaded_backend = SFC.ThreadedBackend()

    # Warmup
    SFC.calculate_structure_function(
        sft, x_tup, u_tup, bin_edges;
        backend = threaded_backend,
        verbose = false,
        show_progress = false,
    )
    SFC.gpu_calculate_structure_function(sft, KA.CPU(), x_cpu, u_cpu, bin_edges)
    SFC.gpu_calculate_structure_function(sft, CUDA.CUDABackend(), x_gpu, u_gpu, bin_edges)
    CUDA.synchronize()

    t_threaded = @elapsed begin
        SFC.calculate_structure_function(
            sft, x_tup, u_tup, bin_edges;
            backend = threaded_backend,
            verbose = false,
            show_progress = false,
        )
    end

    t_ka_cpu = @elapsed begin
        SFC.gpu_calculate_structure_function(sft, KA.CPU(), x_cpu, u_cpu, bin_edges)
    end

    t_cuda = @elapsed begin
        SFC.gpu_calculate_structure_function(sft, CUDA.CUDABackend(), x_gpu, u_gpu, bin_edges)
        CUDA.synchronize()
    end

    println("ThreadedBackend ($nthreads threads): $(round(t_threaded; digits = 3)) s")
    println("KA.CPU() (GPU kernel on CPU):     $(round(t_ka_cpu; digits = 3)) s")
    println("CUDA:                             $(round(t_cuda; digits = 3)) s")
    println()
    println("CUDA vs ThreadedBackend:  $(round(t_threaded / t_cuda; digits = 2))×")
    println("CUDA vs KA.CPU():         $(round(t_ka_cpu / t_cuda; digits = 2))×")
    return nothing
end

Base.invokelatest(main)
