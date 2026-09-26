using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using StructureFunctions.Calculations: joint2d_smem_max, joint2d_smem_exact, joint2d_smem_align256
using Random: Random

const GPUExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
GPUExt === nothing && error("StructureFunctionsKernelAbstractionsExt not loaded")

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

Test.@testset "joint2d smem helpers" begin
    caps = SFC.gpu_device_caps(KA.CPU())
    for (W, F, XT, OT, CT) in ((2, 2, Float32, Float32, UInt32), (2, 2, Float64, Float64, UInt32),
                               (3, 3, Float64, Float64, Float64), (6, 6, Float32, Float64, Float32),
                               (3, 2, Float64, Float64, UInt32))
        m = joint2d_smem_max(KA.CPU(), W, F, XT, OT, CT)
        Test.@test m > 0
        Test.@test GPUExt._gpu_joint_2d_tiled_eligible(caps, W, F, XT, OT, CT, m)
        Test.@test !GPUExt._gpu_joint_2d_tiled_eligible(caps, W, F, XT, OT, CT, m + 1)
    end
    Test.@test joint2d_smem_exact(20, 22) == 440
    Test.@test joint2d_smem_align256(20, 22) == 512
    Test.@test joint2d_smem_align256(50, 52) == 2816  # cld(2600, 256) * 256
    Test.@test joint2d_smem_align256(128, 129) == 16640
    Test.@test_throws ArgumentError GPUExt._joint2d_resolve_compile_cells(100, 50)
    Test.@test GPUExt._joint2d_resolve_compile_cells(100, nothing) == 100
    Test.@test GPUExt._joint2d_resolve_compile_cells(100, 256) == 256
    Test.@test GPUExt._joint2d_resolve_compile_cells(100, 20_000) == 20_000
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
    Test.@test ws.joint2d_compile_cells == 100
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU joint2d exact smem parity — log dist NB2=440" begin
    N = 80
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = randn(FT, 2, N)
    dist = LogBinEdges(FT(100), FT(5000), 21)
    val = collect(FT, range(-1.0, 2.0; length = 23))
    sft = SFT.L3SFType()
    ref = _ref_joint(sft, x, u, dist, val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val)
    Test.@test ws.joint2d_compile_cells == 440
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU joint2d max smem parity — NB2=100 at the widest fitting width" begin
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
    Test.@test ws.joint2d_compile_cells == widest
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

# A histogram, or a compile width, whose tiled kernel does not fit the device's shared memory takes
# the global-atomic joint kernel; weighted counts are a pair mass of the call's count type there too.
Test.@testset "GPU joint2d past the shared-memory fit" begin
    N = 120
    Random.seed!(7)
    for FT in (Float32, Float64)
        x = rand(FT, 2, N)
        u = randn(FT, 2, N)
        w = FT(0.5) .+ rand(FT, N)
        dist = collect(FT, range(0.0, 1.4; length = 101))
        val = collect(FT, range(-3.0, 3.0; length = 101))
        sft = SFT.L2SFType()
        caps = SFC.gpu_device_caps(KA.CPU())
        Test.@test !GPUExt._gpu_joint_2d_tiled_eligible(caps, 2, 2, FT, FT, UInt32, 100 * 100)
        ref = _ref_joint(sft, x, u, dist, val)
        gpu = _gpu_joint(sft, x, u, dist, val)
        Test.@test gpu.counts == ref.counts
        Test.@test gpu.sums ≈ ref.sums rtol = 1e-5 atol = 1e-5
        wref = SFC.calculate_structure_function(sft, x, u, dist, val, FT;
            backend = CB.SerialBackend(), weights = w)
        wgpu = SFC.calculate_structure_function(sft, x, u, dist, val, FT;
            backend = CB.GPUBackend(KA.CPU()), weights = w)
        Test.@test collect(wgpu.counts) ≈ collect(wref.counts) rtol = 1e-5
        Test.@test collect(wgpu.sums) ≈ collect(wref.sums) rtol = 1e-5 atol = 1e-5

        small_dist = collect(FT, range(0.0, 1.4; length = 11))
        small_val = collect(FT, range(-3.0, 3.0; length = 11))
        ws = SFC.GPUSFWorkspace(KA.CPU(), small_dist, small_val; joint2d_compile_cells = 20_000)
        Test.@test !GPUExt._gpu_joint_2d_tiled_eligible(caps, 2, 2, FT, FT, UInt32, 20_000)
        sref = _ref_joint(sft, x, u, small_dist, small_val)
        sgpu = _gpu_joint(sft, x, u, small_dist, small_val; workspace = ws)
        Test.@test sgpu.counts == sref.counts
        Test.@test sgpu.sums ≈ sref.sums rtol = 1e-5 atol = 1e-5
    end
end

Test.@testset "GPU joint2d typed workspace dispatch — LogBinEdges + InfPadded" begin
    N = 80
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = randn(FT, 2, N)
    dist = LogBinEdges(FT(100), FT(5000), 21)
    val = InfPaddedBinEdges(LinearBinEdges(range(-1.0, 2.0; length = 23)))
    sft = SFT.L2SFType()
    ref = _ref_joint(sft, x, u, dist, val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val; kind = :joint2d)
    Test.@test typeof(ws).parameters[1] === :joint2d
    Test.@test ws.val_plan isa InfPaddedBinEdges{FT, <:LinearBinEdges}
    Test.@test ws.dist_digitizer isa SF.BucketedBinEdges
    gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU joint2d align256 smem parity — NB2=440" begin
    N = 80
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = randn(FT, 2, N)
    dist = LogBinEdges(FT(100), FT(5000), 21)
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

# A workspace is built from bins alone, so it must serve both coordinate widths: flat D=2 gives
# W=2 and flat D=3 gives W=3, and the kernel is selected per launch, not fixed at construction.
Test.@testset "joint2d workspace is width-agnostic" begin
    FT = Float64
    N = 200
    dist = collect(FT, range(0.05, 2.0; length = 11))
    val = collect(FT, range(-5.0, 5.0; length = 9))
    sft = SFT.L2SFType()
    for D in (2, 3)
        x = rand(FT, D, N) .* FT(1.5)
        u = randn(FT, D, N)
        ref = _ref_joint(sft, x, u, dist, val)
        ws = SFC.GPUSFWorkspace(KA.CPU(), dist, val)
        gpu = _gpu_joint(sft, x, u, dist, val; workspace = ws)
        Test.@test gpu.counts == ref.counts
        Test.@test gpu.sums ≈ ref.sums atol = 1e-10
    end
end

Test.@testset "GPUSFWorkspace is immutable with kind as a type parameter" begin
    dist = collect(range(0.05, 2.0; length = 11))
    val = collect(range(-1.0, 1.0; length = 9))
    for (args, kind) in (
        ((KA.CPU(), dist), :sf1d),
        ((KA.CPU(), dist), :single_pass),
        ((KA.CPU(), dist, val), :joint2d),
    )
        ws = SFC.GPUSFWorkspace(args...; kind = kind)
        Test.@test !ismutabletype(typeof(ws))
        Test.@test typeof(ws).parameters[1] === kind
        Test.@test !hasfield(typeof(ws), :kind)
        Test.@test all(isconcretetype, fieldtypes(typeof(ws)))
    end
end
