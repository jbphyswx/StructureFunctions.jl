using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using StructureFunctions.Calculations: joint2d_smem_max
using Random: Random

Random.seed!(42)

function _ref_joint(sft, x, u, dist, val)
    return SFC.calculate_structure_function(
        sft, x, u, dist, val;
        backend = CB.SerialBackend(),
    )
end

function _gpu_joint(sft, x, u, dist, val; kwargs...)
    return SFC.calculate_structure_function(
        sft, x, u, dist, val; backend = CB.GPUBackend(KA.CPU()), kwargs...,
    )
end

# A workspace compiled wider than its histogram, at the widest width that fits, matches serial; one narrower is refused.
Test.@testset "GPU joint2d compile width: the widest fitting matches, one below the histogram is refused" begin
    N = 60
    FT = Float64
    x = rand(FT, 2, N)
    u = randn(FT, 2, N)
    dist = collect(FT, range(0.0, 1.4; length = 11))
    val = collect(FT, range(0.0, 2.0; length = 11))
    sft = SFT.L2SFType()
    ref = _ref_joint(sft, x, u, dist, val)
    widest = joint2d_smem_max(KA.CPU(), 2, 2, FT, FT, UInt32)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val; joint2d_compile_cells = widest)
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
    Test.@test_throws ArgumentError SFC.GPUSFWorkspace(KA.CPU(), dist, val; joint2d_compile_cells = 10 * 10 - 1)
end

const GPU_JOINT2D_PAST_FIT_CASES = ((Float32, :unweighted), (Float64, :weighted))

# A compile width past the shared-memory fit takes the global-atomic joint kernel, which matches serial weighted or not.
Test.@testset "GPU joint2d past the shared-memory fit" begin
    N = 60
    Random.seed!(7)
    sft = SFT.L2SFType()
    Test.@testset "$FT, $weighting" for (FT, weighting) in GPU_JOINT2D_PAST_FIT_CASES
        x = rand(FT, 2, N)
        u = randn(FT, 2, N)
        dist = collect(FT, range(0.0, 1.4; length = 11))
        val = collect(FT, range(-3.0, 3.0; length = 11))
        ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val; joint2d_compile_cells = 20_000)
        if weighting === :weighted
            w = FT(0.5) .+ rand(FT, N)
            wref = SFC.calculate_structure_function(sft, x, u, dist, val, FT;
                backend = CB.SerialBackend(), weights = w)
            wgpu = SFC.calculate_structure_function(sft, x, u, dist, val, FT;
                backend = CB.GPUBackend(KA.CPU()), weights = w, workspace = ws)
            Test.@test collect(wgpu.counts) ≈ collect(wref.counts) rtol = 1e-5
            Test.@test collect(wgpu.sums) ≈ collect(wref.sums) rtol = 1e-5 atol = 1e-5
        else
            ref = _ref_joint(sft, x, u, dist, val)
            gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
            Test.@test gpu.counts == ref.counts
            Test.@test gpu.sums ≈ ref.sums rtol = 1e-5 atol = 1e-5
        end
    end
end

# A signed operator with log distance bins and both inf-padded catch-alls in use, through a workspace, matches serial.
Test.@testset "GPU joint2d with log distance and inf-padded value bins" begin
    N = 80
    FT = Float64
    x = rand(FT, 2, N)
    u = randn(FT, 2, N)
    dist = LogBinEdges(FT(0.02), FT(1.4), 21)
    val = InfPaddedBinEdges(LinearBinEdges(range(-1.0, 2.0; length = 23)))
    sft = SFT.L3SFType()
    ref = _ref_joint(sft, x, u, dist, val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val; kind = :joint2d)
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test sum(ref.counts[:, 1]) > 0 && sum(ref.counts[:, end]) > 0
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

# One workspace built from bins alone serves both coordinate widths, W = 2 and W = 3, over two point tiles.
Test.@testset "joint2d workspace is width-agnostic" begin
    FT = Float64
    N = 200
    dist = collect(FT, range(0.05, 2.0; length = 11))
    val = collect(FT, range(-5.0, 5.0; length = 9))
    sft = SFT.L2SFType()
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val)
    for D in (2, 3)
        x = rand(FT, D, N) .* FT(1.5)
        u = randn(FT, D, N)
        ref = _ref_joint(sft, x, u, dist, val)
        gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
        Test.@test (D, gpu.counts == ref.counts, isapprox(gpu.sums, ref.sums; atol = 1e-10)) == (D, true, true)
    end
end
