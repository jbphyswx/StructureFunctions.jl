using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    InfPaddedBinEdges, LinearBinEdges, LogBinEdges,
    joint2d_smem_max, joint2d_smem_exact, joint2d_smem_align256
using Random: Random

const GPUExt = Base.get_extension(SF, :StructureFunctionsGPUExt)
GPUExt === nothing && error("StructureFunctionsGPUExt not loaded")

Random.seed!(42)

function _ref_joint(sft, x, u, dist, val)
    return SFC.calculate_structure_function(
        sft, x, u, dist, val;
        backend = SFC.SerialBackend(), verbose = false, show_progress = false,
    )
end

function _gpu_joint(sft, x, u, dist, val; kwargs...)
    return SFC.gpu_calculate_structure_function_2d(
        sft, KA.CPU(), x, u, dist, val; kwargs...,
    )
end

Test.@testset "joint2d smem helpers" begin
    Test.@test joint2d_smem_max() == GPUExt.SF_GPU_MAX_2D_HIST
    Test.@test joint2d_smem_exact(20, 22) == 440
    Test.@test joint2d_smem_align256(20, 22) == 512
    Test.@test joint2d_smem_align256(50, 52) == 2816  # cld(2600, 256) * 256
    Test.@test_throws ArgumentError GPUExt._joint2d_resolve_compile_cells(100, 50)
    Test.@test GPUExt._joint2d_resolve_compile_cells(100, nothing) == 100
    Test.@test GPUExt._joint2d_resolve_compile_cells(100, 256) == 256
end

Test.@testset "GPU joint2d exact smem parity — NB2=100" begin
    N = 60
    FT = Float64
    x = rand(FT, 2, N)
    u = randn(FT, 2, N)
    dist = exp.(range(log(0.05), log(2.0); length = 11))
    val = collect(FT, range(-1.0, 1.0; length = 11))
    sft = SFT.L2SFType()
    ref = _ref_joint(sft, x, u, dist, val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val)
    Test.@test ws.joint2d_nb2 == 100
    Test.@test ws.joint2d_compile_cells == 100
    Test.@test ws.joint2d_kernel !== nothing
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU joint2d exact smem parity — log dist NB2=440" begin
    N = 80
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = randn(FT, 2, N)
    dist = LogBinEdges(exp.(range(log(FT(100)), log(FT(5000)); length = 21)))
    val = collect(FT, range(-1.0, 2.0; length = 23))
    sft = SFT.L3SFType()
    ref = _ref_joint(sft, x, u, dist, val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val)
    Test.@test ws.joint2d_compile_cells == 440
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU joint2d max smem parity — NB2=100 compile_cells=4096" begin
    N = 60
    FT = Float64
    x = rand(FT, 2, N)
    u = randn(FT, 2, N)
    dist = collect(FT, range(0.0, 1.4; length = 11))
    val = collect(FT, range(0.0, 2.0; length = 11))
    sft = SFT.L2SFType()
    ref = _ref_joint(sft, x, u, dist, val)
    ws = SFC.GPUSFWorkspace(
        KA.CPU(), dist, val; joint2d_compile_cells = joint2d_smem_max(),
    )
    Test.@test ws.joint2d_compile_cells == joint2d_smem_max()
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU joint2d typed workspace dispatch — LogBinEdges + InfPadded" begin
    N = 80
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = randn(FT, 2, N)
    dist = LogBinEdges(exp.(range(log(FT(100)), log(FT(5000)); length = 21)))
    val = InfPaddedBinEdges(LinearBinEdges(range(-1.0, 2.0; length = 23)))
    sft = SFT.L2SFType()
    ref = _ref_joint(sft, x, u, dist, val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val; kind = :joint2d)
    Test.@test ws.kind == :joint2d
    Test.@test ws.val_plan isa GPUExt.GPUValueInfLinearShared
    Test.@test GPUExt._joint2d_val_route(ws.val_plan) == :inflinear
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU joint2d align256 smem parity — NB2=440" begin
    N = 80
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = randn(FT, 2, N)
    dist = LogBinEdges(exp.(range(log(FT(100)), log(FT(5000)); length = 21)))
    val = collect(FT, range(-1.0, 2.0; length = 23))
    n_dist = length(dist) - 1
    n_val = length(val) - 1
    sft = SFT.L2SFType()
    ref = _ref_joint(sft, x, u, dist, val)
    ws = SFC.GPUSFWorkspace(
        KA.CPU(), dist, val;
        joint2d_compile_cells = joint2d_smem_align256(n_dist, n_val),
    )
    Test.@test ws.joint2d_compile_cells == 512
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end
