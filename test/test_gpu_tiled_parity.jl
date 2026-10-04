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

Test.@testset "GPU tiled parity — log bins 2D" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = rand(FT, 2, N)
    bin_edges = LogBinEdges(FT(0.05), FT(1.4), 11)
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, x, u, collect(bin_edges))
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

# Non-uniform, non-log monotone edges.
Test.@testset "GPU tiled parity — general monotone bins 2D" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = FT[0.0, 0.05, 0.12, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.35]
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, x, u, bin_edges)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

struct _EdgeAdaptProbe end
KA.Adapt.adapt_storage(::_EdgeAdaptProbe, a::Array) = view(a, :)

_arrays_adapted(p::SF.InfPaddedBinEdges) = _arrays_adapted(p.edges)
_arrays_adapted(p::SF.BucketedBinEdges) = p.edges isa SubArray && p.cells isa SubArray
_arrays_adapted(p) = p.edges isa SubArray

# Every digitizer plan adapts its arrays and digitizes as before, checked with an adaptor that rewrites arrays to views.
Test.@testset "device digitizers recurse through adapt" begin
    KAExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
    gen = SF.BinEdges(Float32[0, 0.2, 0.5, 1])
    bucket = SF.digitize_plan(LogBinEdges(0.01f0, 1.0f0, 9))
    table = KAExt._device_plan(LogBinEdges(0.01f0, 1.0f0, 9), Val(:sf1d))
    padded_table = KAExt._device_plan(SF.InfPaddedBinEdges(LogBinEdges(0.01f0, 1.0f0, 9)), Val(:sf1d))
    plans = (gen, bucket, table, SF.InfPaddedBinEdges(gen), SF.InfPaddedBinEdges(bucket), padded_table)
    adapted = map(p -> KA.Adapt.adapt(_EdgeAdaptProbe(), p), plans)
    same_bins(p, a) = collect(a) == collect(p) &&
                      all(x -> searchsortedfirst(a, x) == searchsortedfirst(p, x), 0.0f0:0.01f0:1.1f0)
    Test.@test all(_arrays_adapted, adapted)
    Test.@test all(splat(same_bins), zip(plans, adapted))
end

# N spans two point tiles, the second partial, in Float32.
Test.@testset "GPU tiled parity — two tiles, Float32" begin
    N = 200
    FT = Float32
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0f0, 1.4f0; length = 11))
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, x, u, bin_edges)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts == ref.counts
    Test.@test maximum(abs, gpu.sums .- ref.sums) < 0.05f0
end

# The in-place entry adds into the caller's buffers.
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

const GPU_TILED_TRANSVERSE_CASES = (
    (2, 50, SFT.T3SFType()),
    (3, 40, SFT.L2T1SFType()),
    (3, 40, SFT.ProjectedStructureFunctionType{0, 3}(
        SFH.ReferenceAxisTransverseBasis(LA.normalize(SA.SVector(1.0, sqrt(2.0), sqrt(3.0)))))),
)

Test.@testset "GPU tiled parity — signed transverse operators keep the operator's convention" begin
    for (D, N, sft) in GPU_TILED_TRANSVERSE_CASES
        x = rand(Float64, D, N)
        u = randn(Float64, D, N)
        bin_edges = collect(range(0.0, 1.4; length = 8))
        ref = _cpu_ref(sft, x, u, bin_edges)
        gpu = _gpu_tiled(sft, x, u, bin_edges)
        Test.@test gpu.counts == ref.counts
        Test.@test gpu.sums ≈ ref.sums atol = 1e-10
        Test.@test any(!iszero, ref.sums)
    end
end

# Bin counts at and just above `SF_GPU_MAX_BINS` give the serial counts and sums.
Test.@testset "GPU parity at and above the tiled kernel's bin cap" begin
    N = 100
    FT = Float64
    Random.seed!(4242)
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    sft = SFT.L2SFType()
    cap = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt).SF_GPU_MAX_BINS
    for nb in (cap, cap + 1)
        bin_edges = collect(FT, range(0.0, 1.0; length = nb + 1))
        ref = SFC.calculate_structure_function(sft, x, u, bin_edges, SF.StructureFunctionSumsAndCounts;
            backend = CB.SerialBackend())
        got = SFC.calculate_structure_function(sft, x, u, bin_edges, SF.StructureFunctionSumsAndCounts;
            backend = CB.GPUBackend(KA.CPU()))
        Test.@test sum(ref.counts) > 0
        Test.@test collect(got.counts) == collect(ref.counts)
        Test.@test isapprox(collect(got.sums), collect(ref.sums); rtol = 1e-10, atol = 1e-12)
    end
end

const GPU_TILED_BATCH_CAP_CASES = (
    (:varying, :individual, :under),
    (:varying, :single_pass, :over),
    (:fixed, :individual, :over),
    (:fixed, :single_pass, :under),
)

# Batch entries with bin counts either side of `SF_GPU_MAX_BINS` match the serial result.
Test.@testset "GPU batch — bin counts either side of SF_GPU_MAX_BINS match serial" begin
    FT = Float32
    N, B = 20, 3
    sft = SFT.L2SFType()
    gpu_be = CB.GPUBackend(KA.CPU())
    cap = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt).SF_GPU_MAX_BINS
    bin_sets = (under = collect(FT, range(0.0f0, 2.0f0; length = 65)),
                over = collect(FT, range(0.0f0, 2.0f0; length = cap + 2)))
    positions = (fixed = rand(FT, 2, N), varying = rand(FT, 2, N, B))
    u = rand(FT, 2, N, B)

    Test.@testset "$kind batch, $xkind x, $side the cap" for (xkind, kind, side) in GPU_TILED_BATCH_CAP_CASES
        x = positions[xkind]
        bins = bin_sets[side]
        nb = length(bins) - 1
        if kind === :individual
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
        else
            ref_s = zeros(FT, SFC.SINGLE_PASS_N, nb, B)
            ref_c = zeros(UInt32, SFC.SINGLE_PASS_N, nb, B)
            for b in 1:B
                xb = ndims(x) == 2 ? x : x[:, :, b]
                one_s = zeros(FT, SFC.SINGLE_PASS_N, nb)
                one_c = zeros(UInt32, SFC.SINGLE_PASS_N, nb)
                SFC.calculate_structure_functions_single_pass!(one_s, one_c, xb, u[:, :, b], bins;
                    backend = CB.SerialBackend())
                ref_s[:, :, b] .= one_s
                ref_c[:, :, b] .= one_c
            end
            got = SFC.calculate_structure_functions_single_pass(x, u, bins; backend = gpu_be)
            inv = enumerate((:S2, :L2, :T2, :S3, :L3, :L1T2))
            Test.@test all(((t, k),) -> collect(got[k].counts) == ref_c[t, :, :], inv)
            Test.@test all(((t, k),) -> isapprox(collect(got[k].sums), ref_s[t, :, :]; rtol = 1e-5, atol = 1e-5), inv)
        end
    end
end
