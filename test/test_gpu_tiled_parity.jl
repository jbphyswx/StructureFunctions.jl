using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LinearBinEdges, LogBinEdges
using Random: Random
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA

Random.seed!(42)

function _cpu_ref(sft, x, u, bin_edges)
    return SFC.calculate_structure_function(
        sft, x, u, bin_edges;
        verbose = false, show_progress = false, output_type = SF.StructureFunctionSumsAndCounts,
    )
end

function _gpu_tiled(sft, x, u, bin_edges)
    return SFC.gpu_calculate_structure_function(
        sft, KA.CPU(), x, u, bin_edges,
    )
end

Test.@testset "GPU tiled parity — linear 2D N=50" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0, 1.4; length = 11))
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, x, u, bin_edges)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — linear 3D N=50" begin
    N = 50
    FT = Float64
    x = rand(FT, 3, N)
    u = rand(FT, 3, N)
    bin_edges = collect(FT, range(0.0, 2.0; length = 11))
    sft = SFT.L2SFType()
    ref = _cpu_ref(
        sft,
        x,
        u,
        bin_edges,
    )
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — log bins 2D" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = rand(FT, 2, N)
    log_vec = exp.(range(log(FT(0.05)), log(FT(1.4)); length = 11))
    bin_edges = LogBinEdges(log_vec)
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, x, u, log_vec)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — general monotone bins 2D" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    # Non-uniform, non-log monotone edges
    bin_edges = FT[0.0, 0.05, 0.12, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.35]
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, x, u, bin_edges)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

# A struct carrying a device array into a kernel must adapt that field, or the launch dies with
# "KernelError: passing non-bitstype argument" on CUDA. KA.CPU cannot reach that failure — adapt is
# a no-op there — so assert the rule itself recurses.
struct _EdgeAdaptProbe end
KA.Adapt.adapt_storage(::_EdgeAdaptProbe, ::AbstractArray) = :adapted

Test.@testset "device-array kernel args recurse through adapt" begin
    GPUExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)

    dig = GPUExt.SFGeneralDigitizer(rand(Float32, 6), 6)
    adapted = KA.Adapt.adapt(_EdgeAdaptProbe(), dig)
    Test.@test adapted.edges === :adapted
    Test.@test adapted.n_edges == 6

    vcols = GPUExt.GPUValueVectorCols{Float32}(rand(Float32, 6, 6))
    Test.@test KA.Adapt.adapt(_EdgeAdaptProbe(), vcols).edges_dev === :adapted
end

Test.@testset "GPU tiled parity — medium N linear 2D" begin
    N = 500
    FT = Float32
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0f0, 1.4f0; length = 11))
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, x, u, bin_edges)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    max_Δ = maximum(abs, gpu.sums .- ref.sums)
    Test.@test max_Δ < 0.05f0
end

Test.@testset "GPU in-place !() parity — linear 2D" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0, 1.4; length = 11))
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, x, u, bin_edges)
    n_bins = length(bin_edges) - 1
    sums = zeros(FT, n_bins)
    counts = zeros(UInt32, n_bins)
    SFC.gpu_calculate_structure_function!(sums, counts, sft, KA.CPU(), x, u, bin_edges)
    Test.@test counts == ref.counts
    Test.@test sums ≈ ref.sums atol = 1e-10
    SFC.gpu_calculate_structure_function!(sums, counts, sft, KA.CPU(), x, u, bin_edges)
    Test.@test counts == ref.counts .* 2
    Test.@test sums ≈ ref.sums .* 2 atol = 1e-10
end

Test.@testset "GPU joint 2D parity — L2SF linear bins N=50" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    distance_bins = collect(FT, range(0.0, 1.4; length = 11))
    value_bins = collect(FT, range(0.0, 2.0; length = 11))
    sft = SFT.L2SFType()
    ref = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = CB.SerialBackend(), verbose = false, show_progress = false,
    )
    gpu = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = CB.GPUBackend(KA.CPU()), verbose = false, show_progress = false,
    )
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU joint 2D parity — L3SF log distance bins" begin
    N = 40
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = randn(FT, 2, N)
    distance_bins = exp.(range(log(0.05), log(2.0); length = 8))
    value_bins = collect(FT, range(-1.0, 1.0; length = 9))
    sft = SFT.L3SFType()
    ref = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = CB.SerialBackend(), verbose = false, show_progress = false,
    )
    gpu = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = CB.GPUBackend(KA.CPU()), verbose = false, show_progress = false,
    )
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — NB > SF_GPU_MAX_BINS errors" begin
    N = 20
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0, 2.0; length = 130))  # 129 bins (> 128 cap)
    sft = SFT.L2SFType()
    Test.@test_throws ErrorException SFC.gpu_calculate_structure_function(
        sft, KA.CPU(), x, u, bin_edges,
    )
end

# The tiled kernels size `@localmem` from the compile-time SF_GPU_MAX_BINS but index it by the
# runtime NB under `@inbounds`, so an unguarded NB > 128 writes out of bounds in shared memory and
# corrupts the histogram silently. The batch entries had no guard at all; these pin it down.
Test.@testset "GPU batch — NB > SF_GPU_MAX_BINS errors (no silent shared-mem overrun)" begin
    FT = Float32
    N, B = 20, 3
    sft = SFT.L2SFType()
    gpu_be = CB.GPUBackend(KA.CPU())
    over = collect(FT, range(0.0f0, 2.0f0; length = 130))   # 129 bins (> 128 cap)
    under = collect(FT, range(0.0f0, 2.0f0; length = 65))   # 64 bins (must still run)

    x_fixed = rand(FT, 2, N)
    x_vary = rand(FT, 2, N, B)
    u = rand(FT, 2, N, B)

    for (name, x) in (("varying-x", x_vary), ("fixed-x", x_fixed))
        Test.@testset "1D individual batch $name" begin
            Test.@test_throws ErrorException SFC.gpu_calculate_structure_function_batch(
                sft, KA.CPU(), x, u, over,
            )
            Test.@test SFC.gpu_calculate_structure_function_batch(
                sft, KA.CPU(), x, u, under,
            ) isa Any
        end
        Test.@testset "single-pass 1D batch $name" begin
            Test.@test_throws ErrorException SFC._dispatch_single_pass(gpu_be, x, u, over)
            Test.@test SFC._dispatch_single_pass(gpu_be, x, u, under) isa Any
        end
    end
end
