using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, HelperFunctions as SFH,
    LinearBinEdges, LogBinEdges
using Random: Random
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA

Random.seed!(42)

function _cpu_ref(sft, x, u, bin_edges)
    return SFC.calculate_structure_function(
        sft, x, u, bin_edges, SF.StructureFunctionSumsAndCounts,
    )
end

function _gpu_tiled(sft, x, u, bin_edges)
    return SFC.calculate_structure_function(
        sft, x, u, bin_edges, SF.StructureFunctionSumsAndCounts;
        backend = CB.GPUBackend(KA.CPU()),
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
    bin_edges = LogBinEdges(FT(0.05), FT(1.4), 11)
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, x, u, collect(bin_edges))
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

# Every digitizer plan adapts its array fields, checked with an adaptor that rewrites arrays to views.
struct _EdgeAdaptProbe end
KA.Adapt.adapt_storage(::_EdgeAdaptProbe, a::Array) = view(a, :)

Test.@testset "device digitizers recurse through adapt" begin
    KAExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
    to = _EdgeAdaptProbe()
    gen = SF.BinEdges(Float32[0, 0.2, 0.5, 1])
    bucket = SF.digitize_plan(LogBinEdges(0.01f0, 1.0f0, 9))
    table = KAExt._device_plan(LogBinEdges(0.01f0, 1.0f0, 9), Val(:sf1d))
    padded_table = KAExt._device_plan(SF.InfPaddedBinEdges(LogBinEdges(0.01f0, 1.0f0, 9)), Val(:sf1d))
    Test.@test table isa SF.LogTableBinEdges
    Test.@test padded_table.edges isa SF.LogTableBinEdges
    logb = LogBinEdges(0.01f0, 1.0f0, 9)
    Test.@test SFC.GPUSFWorkspace(KA.CPU(), logb; kind = :sf1d).dist_digitizer isa SF.LogTableBinEdges
    Test.@test SFC.GPUSFWorkspace(KA.CPU(), logb; kind = :single_pass).dist_digitizer isa SF.BucketedBinEdges
    for plan in (gen, bucket, table, SF.InfPaddedBinEdges(gen), SF.InfPaddedBinEdges(bucket), padded_table)
        adapted = KA.Adapt.adapt(to, plan)
        inner = adapted isa SF.InfPaddedBinEdges ? adapted.edges : adapted
        Test.@test inner.edges isa SubArray
        inner isa SF.BucketedBinEdges && Test.@test inner.cells isa SubArray
        Test.@test collect(adapted) == collect(plan)
        Test.@test all(x -> searchsortedfirst(adapted, x) == searchsortedfirst(plan, x), 0.0f0:0.01f0:1.1f0)
    end
    lin = LinearBinEdges(0.0f0, 1.0f0, 5)
    Test.@test KA.Adapt.adapt(to, lin) === lin
    per_moment = KAExt._gpu_digitizer(KA.CPU(), (gen, LogBinEdges(0.01f0, 1.0f0, 9), lin, gen, lin, gen), Val(:value))
    Test.@test per_moment[1] isa SF.BucketedBinEdges
    Test.@test per_moment[2] isa SF.BucketedBinEdges
    Test.@test per_moment[3] === lin
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
    SFC.gpu_calculate_structure_function!(sums, counts, sft, KA.CPU(), x, u, bin_edges; geometry = SFH.FlatGeometry{2}())
    Test.@test counts == ref.counts
    Test.@test sums ≈ ref.sums atol = 1e-10
    SFC.gpu_calculate_structure_function!(sums, counts, sft, KA.CPU(), x, u, bin_edges; geometry = SFH.FlatGeometry{2}())
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
        backend = CB.SerialBackend(),
    )
    gpu = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = CB.GPUBackend(KA.CPU()),
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
        backend = CB.SerialBackend(),
    )
    gpu = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = CB.GPUBackend(KA.CPU()),
    )
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — signed transverse operators keep the operator's convention" begin
    for (D, N) in ((2, 50), (3, 40))
        x = rand(Float64, D, N)
        u = randn(Float64, D, N)
        bin_edges = collect(range(0.0, 1.4; length = 8))
        ops = (SFT.T3SFType(), SFT.L2T1SFType())
        D == 3 && (ops = (ops..., SFT.ProjectedStructureFunctionType{0, 3}(
            SFH.ReferenceAxisTransverseBasis(LA.normalize(SA.SVector(1.0, sqrt(2.0), sqrt(3.0)))))))
        for sft in ops
            ref = _cpu_ref(sft, x, u, bin_edges)
            gpu = _gpu_tiled(sft, x, u, bin_edges)
            Test.@test gpu.counts == ref.counts
            Test.@test gpu.sums ≈ ref.sums atol = 1e-10
            Test.@test any(!iszero, ref.sums)
        end
    end
end

Test.@testset "GPU parity above the tiled kernel's shared-memory cap" begin
    # Bin counts at and above `SF_GPU_MAX_BINS` give the serial counts and sums.
    N = 200
    FT = Float64
    Random.seed!(4242)
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    sft = SFT.L2SFType()
    for nb in (128, 129, 4000)
        bin_edges = collect(FT, range(0.0, 2.0; length = nb + 1))
        ref = SFC.calculate_structure_function(sft, x, u, bin_edges, FT, SF.StructureFunctionSumsAndCounts;
            backend = CB.SerialBackend())
        got = SFC.calculate_structure_function(sft, x, u, bin_edges, SF.StructureFunctionSumsAndCounts;
            backend = CB.GPUBackend(KA.CPU()))
        Test.@test sum(ref.counts) > 0
        Test.@test collect(got.counts) == collect(ref.counts)
        Test.@test isapprox(collect(got.sums), collect(ref.sums); rtol = 1e-10, atol = 1e-12)
    end
end

# Batch entries with 64 and 129 bins (either side of the 128-bin cap) match the serial result.
Test.@testset "GPU batch — bin counts either side of SF_GPU_MAX_BINS match serial" begin
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
            for bins in (under, over)
                nb = length(bins) - 1
                ref_s = zeros(FT, nb, B)
                ref_c = zeros(Int, nb, B)
                for b in 1:B
                    xb = ndims(x) == 2 ? x : x[:, :, b]
                    r = SFC.calculate_structure_function(sft, xb, u[:, :, b], bins, FT,
                        SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend())
                    ref_s[:, b] .= r.sums
                    ref_c[:, b] .= r.counts
                end
                g = SFC.calculate_structure_function(sft, x, u, bins, SF.StructureFunctionSumsAndCounts;
                    backend = gpu_be)
                Test.@test reshape(collect(g.counts), nb, B) == ref_c
                Test.@test isapprox(reshape(collect(g.sums), nb, B), ref_s; rtol = 1e-5)
            end
        end
        Test.@testset "single-pass 1D batch $name" begin
            for bins in (under, over)
                ref = SFC.calculate_structure_functions_single_pass(x, u, bins; backend = CB.SerialBackend())
                got = SFC.calculate_structure_functions_single_pass(x, u, bins; backend = gpu_be)
                Test.@test keys(got) == keys(ref)
                for k in keys(ref)
                    Test.@test collect(got[k].counts) == collect(ref[k].counts)
                    Test.@test isapprox(collect(got[k].sums), collect(ref[k].sums); rtol = 1e-5, atol = 1e-5)
                end
            end
        end
    end
end
