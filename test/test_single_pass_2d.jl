using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionObjects as SFO,
    StructureFunctionTypes as SFT, HelperFunctions as SFH, InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using OhMyThreads: OhMyThreads  # load extension for ThreadedBackend / AutoBackend when nthreads() > 1
using KernelAbstractions: KernelAbstractions as KA
using Test: Test
using Random: Random

# Single-pass 2D returns a NamedTuple keyed by invariant, in the stacked-row order of these six.
const SP2D_INV = (:S2, :L2, :T2, :S3, :L3, :L1T2)

"""Wide synthetic value-bin edges for unit tests only."""
function _synthetic_value_bins(n_bins::Int; pad_infinite::Bool = true)
    inner = LinearBinEdges(range(-1.0, 2.0, length = n_bins + 1))
    return pad_infinite ? InfPaddedBinEdges(inner) : inner
end

# The stacked accumulator equals each invariant's joint histogram, marginalizes to the 1D pass, and is the same threaded.
Test.@testset "Single-Pass 2D Core Correctness & Parity" begin
    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5

    distance_bins = LogBinEdges(1000.0, 50000.0, 6)
    value_bins = _synthetic_value_bins(10; pad_infinite = true)
    n_val = length(value_bins) - 1
    n_bins = length(distance_bins) - 1

    sums_2d = zeros(Float64, 6, n_bins, n_val)
    counts_2d = zeros(UInt32, 6, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_2d, counts_2d, x, u, distance_bins, value_bins;
        backend = CB.SerialBackend(),
    )

    joints = [SFC.calculate_structure_function(SFC.SINGLE_PASS_OPERATORS[k], x, u, distance_bins, value_bins;
                                               backend = CB.SerialBackend()) for k in SP2D_INV]
    Test.@test all(j -> j isa SFO.StructureFunction2DSumsAndCounts, joints)
    Test.@test all(sums_2d[t, :, :] ≈ joints[t].sums for t in 1:6)
    Test.@test all(counts_2d[t, :, :] == joints[t].counts for t in 1:6)

    sp_1d = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins, SF.StructureFunctionSumsAndCounts;
        backend = CB.SerialBackend(),
    )
    Test.@test all(vec(sum(sums_2d[t, :, :]; dims = 2)) ≈ sp_1d[k].sums for (t, k) in enumerate(SP2D_INV))
    Test.@test all(vec(sum(counts_2d[t, :, :]; dims = 2)) == sp_1d[k].counts for (t, k) in enumerate(SP2D_INV))

    sums_post, counts_post = SFC.marginalize_sp2d_then_append_helmholtz_rows(
        sums_2d, counts_2d, distance_bins,
    )
    Test.@test all(sums_post[t, :] ≈ sp_1d[k].sums for (t, k) in enumerate(SP2D_INV))
    Test.@test all(counts_post[t, :] == sp_1d[k].counts for (t, k) in enumerate(SP2D_INV))
    Test.@test sums_post[7, :] ≈ sp_1d.helmholtz.rotational_sums
    Test.@test counts_post[7, :] == sp_1d.helmholtz.rotational_counts
    Test.@test sums_post[8, :] ≈ sp_1d.helmholtz.divergent_sums
    Test.@test counts_post[8, :] == sp_1d.helmholtz.divergent_counts

    threaded = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins;
        backend = CB.ThreadedBackend(),
    )
    Test.@test all(threaded[k].sums ≈ sums_2d[t, :, :] for (t, k) in enumerate(SP2D_INV))
    Test.@test all(threaded[k].counts == counts_2d[t, :, :] for (t, k) in enumerate(SP2D_INV))
end

# A plain range and a tuple mixing ranges and LinearBinEdges bin the same edges alike, on serial and on the device.
Test.@testset "Single-Pass 2D value-bin accepted shapes" begin
    Random.seed!(15)
    x = rand(2, 12)
    u = randn(2, 12)
    distance_bins = LinearBinEdges(range(0.0, 2.0; length = 5))
    nd = length(distance_bins) - 1

    shared_range_bins = range(-2.0, 3.0; length = 9)
    s_shared = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, shared_range_bins; backend = CB.SerialBackend(),
    )
    Test.@test keys(s_shared) == SP2D_INV
    Test.@test size(s_shared.S2.sums) == size(s_shared.S2.counts) == (nd, length(shared_range_bins) - 1)

    mixed_bins = ntuple(6) do t
        isodd(t) ? range(-2.0, 3.0; length = 9) :
        LinearBinEdges(range(-2.0, 3.0; length = 9))
    end
    s_mixed = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, mixed_bins; backend = CB.SerialBackend(),
    )
    Test.@test all(s_mixed[k].counts == s_shared[k].counts && s_mixed[k].sums ≈ s_shared[k].sums for k in SP2D_INV)

    s_gpu = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, mixed_bins; backend = CB.GPUBackend(KA.CPU()),
    )
    Test.@test all(s_gpu[k].counts == s_mixed[k].counts && s_gpu[k].sums ≈ s_mixed[k].sums for k in SP2D_INV)
end

# The device single pass 2D on log distance edges and infinitely padded value edges equals the serial one.
Test.@testset "Single-Pass 2D GPU (KA.CPU) parity vs Serial" begin
    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5
    distance_bins = LogBinEdges(1000.0, 50000.0, 6)
    value_bins = _synthetic_value_bins(10; pad_infinite = true)
    n_val = length(value_bins) - 1
    n_bins = length(distance_bins) - 1

    sums_ref = zeros(Float64, 6, n_bins, n_val)
    counts_ref = zeros(UInt32, 6, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, counts_ref, x, u, distance_bins, value_bins;
        backend = CB.SerialBackend(),
    )

    sp_gpu = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins;
        backend = CB.GPUBackend(KA.CPU()),
    )
    Test.@test all(sp_gpu[k].sums ≈ sums_ref[t, :, :] for (t, k) in enumerate(SP2D_INV))
    Test.@test all(sp_gpu[k].counts == counts_ref[t, :, :] for (t, k) in enumerate(SP2D_INV))
end

# A tuple of mixed value-bin types bins each invariant by its own edges, allocating nothing per pair.
Test.@testset "Single-Pass 2D heterogeneous value-bin tuple" begin
    FT = Float64
    N, nv = 60, 6
    Random.seed!(99)
    x, u = rand(FT, 2, N), randn(FT, 2, N)
    db = collect(FT, range(0.0, 2.0; length = 7))
    nd = length(db) - 1

    lin = LinearBinEdges(range(FT(-10), FT(10); length = nv + 1))
    lg = LogBinEdges(FT(1e-4), FT(10), nv + 1)
    raw = collect(FT, range(FT(-10), FT(10); length = nv + 1))
    het = (lg, lg, lg, lin, raw, lin)

    got = SFC.calculate_structure_functions_single_pass_2d(
        x, u, db, het; backend = CB.SerialBackend(),
    )
    refs = [SFC.calculate_structure_functions_single_pass_2d(x, u, db, ntuple(_ -> het[t], 6);
                                                             backend = CB.SerialBackend()) for t in 1:6]
    Test.@test all(got[k].sums ≈ refs[t][k].sums for (t, k) in enumerate(SP2D_INV))
    Test.@test all(got[k].counts == refs[t][k].counts for (t, k) in enumerate(SP2D_INV))

    sums = zeros(FT, 6, nd, nv)
    counts = zeros(UInt32, 6, nd, nv)
    f() = SFC.serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, db, het;
                                                                   geometry = SFH.FlatGeometry{2}())
    f()
    Test.@test (@allocated f()) < 100_000
end

# Counts accumulate in the requested count type: pair masses 2^25, 2 and 1 in one cell sum to 2^25 + 3, which no Float32 accumulator holds.
Test.@testset "Single-Pass 2D counts in the count type, not the sum type" begin
    x = Float32[0 1f-4 2f-4; 0 0 0]
    u = zeros(Float32, 2, 3)
    w = Float32[2^13, 2^12, 2^-12]
    for backend in (CB.SerialBackend(), CB.ThreadedBackend())
        r = SFC.calculate_structure_functions_single_pass_2d(x, u, Float32[0, 1], Float32[-1, 1], Float64;
                                                             backend, weights = w)
        Test.@test all(k -> r[k].counts[1, 1] == 2^25 + 3, SP2D_INV)
    end
end
