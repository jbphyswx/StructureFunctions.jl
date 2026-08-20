using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionObjects as SFO,
    StructureFunctionTypes as SFT, InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using OhMyThreads: OhMyThreads  # load extension for ThreadedBackend / AutoBackend when nthreads() > 1
using KernelAbstractions: KernelAbstractions as KA
using Test
using Random

# Single-pass 2D now returns a NamedTuple keyed by invariant (each entry a
# StructureFunction2DSumsAndCounts). Canonical invariant order matches the stacked-row order.
const SP2D_INV = (:S2, :L2, :T2, :S3, :L3, :L1T2)

"""Wide synthetic value-bin edges for unit tests only."""
function _synthetic_value_bins(n_bins::Int; pad_infinite::Bool = true)
    inner = LinearBinEdges(range(-1.0, 2.0, length = n_bins + 1))
    return pad_infinite ? InfPaddedBinEdges(inner) : inner
end

function _synthetic_value_bins_ntuple(n_bins::Int; pad_infinite::Bool = true)
    template = _synthetic_value_bins(n_bins; pad_infinite = pad_infinite)
    return ntuple(_ -> copy(template), 6)
end

Test.@testset "Single-Pass 2D Core Correctness & Parity" begin
    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5

    distance_bins = LogBinEdges(collect(exp.(range(log(1000.0), log(50000.0), length = 6))))
    value_bins = _synthetic_value_bins(10; pad_infinite = true)
    n_val = length(value_bins) - 1
    n_bins = length(distance_bins) - 1

    # Mutating API still fills the stacked (6, n_bins, n_val) accumulator.
    sums_2d = zeros(Float64, 6, n_bins, n_val)
    counts_2d = zeros(UInt32, 6, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_2d, counts_2d, x, u, distance_bins, value_bins;
        backend = CB.SerialBackend(),
    )
    Test.@test size(sums_2d) == (6, n_bins, n_val)
    Test.@test size(counts_2d) == (6, n_bins, n_val)

    # Per-invariant equivalence against standard 2D structure function calls.
    for (t, k) in enumerate(SP2D_INV)
        sf2d = SFC.calculate_structure_function(
            SFC.SINGLE_PASS_OPERATORS[k], x, u, distance_bins, value_bins;
            backend = CB.SerialBackend(), verbose = false, show_progress = false,
        )
        Test.@test sf2d isa SFO.StructureFunction2DSumsAndCounts
        Test.@test sums_2d[t, :, :] ≈ sf2d.sums
        Test.@test counts_2d[t, :, :] ≈ sf2d.counts
    end

    # 1D single-pass (keyed, raw) for marginalization parity.
    sp_1d = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = CB.SerialBackend(), output_type = SF.StructureFunctionSumsAndCounts,
    )
    for (t, k) in enumerate(SP2D_INV)
        marg_sums = vec(dropdims(sum(sums_2d[t:t, :, :], dims = 3), dims = 1))
        marg_counts = vec(dropdims(sum(counts_2d[t:t, :, :], dims = 3), dims = 1))
        Test.@test marg_sums ≈ sp_1d[k].sums
        Test.@test marg_counts == sp_1d[k].counts
    end

    # marginalize-then-append-Helmholtz should reproduce the direct 1D single-pass (incl. Helmholtz).
    sums_post, counts_post = SFC.marginalize_sp2d_then_append_helmholtz_rows(
        sums_2d, counts_2d, distance_bins,
    )
    for (t, k) in enumerate(SP2D_INV)
        Test.@test sums_post[t, :] ≈ sp_1d[k].sums
        Test.@test counts_post[t, :] == sp_1d[k].counts
    end
    Test.@test sums_post[7, :] ≈ sp_1d.helmholtz.rotational_sums
    Test.@test counts_post[7, :] == sp_1d.helmholtz.rotational_counts
    Test.@test sums_post[8, :] ≈ sp_1d.helmholtz.divergent_sums
    Test.@test counts_post[8, :] == sp_1d.helmholtz.divergent_counts

    # AutoBackend parity vs the SerialBackend mutating result.
    fill!(sums_2d, 0.0)
    fill!(counts_2d, 0)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_2d, counts_2d, x, u, distance_bins, value_bins;
        backend = CB.AutoBackend(),
    )
    sp2_auto = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins;
        backend = CB.AutoBackend(),
    )
    for (t, k) in enumerate(SP2D_INV)
        Test.@test sp2_auto[k].sums ≈ sums_2d[t, :, :]
        Test.@test sp2_auto[k].counts == counts_2d[t, :, :]
    end
end


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
    Test.@test size(s_shared.S2.sums) == (nd, length(shared_range_bins) - 1)
    Test.@test size(s_shared.S2.counts) == size(s_shared.S2.sums)

    mixed_bins = ntuple(6) do t
        isodd(t) ? range(-2.0, 3.0; length = 9) :
        LinearBinEdges(range(-2.0, 3.0; length = 9))
    end
    s_mixed = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, mixed_bins; backend = CB.SerialBackend(),
    )
    for k in SP2D_INV
        Test.@test size(s_mixed[k].sums) == size(s_shared[k].sums)
    end

    s_gpu = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, mixed_bins; backend = CB.GPUBackend(KA.CPU()),
    )
    for k in SP2D_INV
        Test.@test s_gpu[k].sums ≈ s_mixed[k].sums
        Test.@test s_gpu[k].counts == s_mixed[k].counts
    end
end

Test.@testset "Single-Pass 2D value bins with 3D point fields" begin
    x = Float32[0.0 1.0 0.0 0.3;
                0.0 0.0 1.0 0.4;
                0.0 0.0 0.0 1.0]
    u = Float32[0.0 0.5 0.0 0.1;
                0.0 0.0 0.5 0.2;
                0.0 0.0 0.0 0.5]
    distance_bins = LinearBinEdges(Float32[0.1, 1.0, 2.0])
    value_bins = range(-2.0f0, 2.0f0; length = 9)

    sp_ref = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins; backend = CB.SerialBackend(),
    )
    sp_gpu = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins; backend = CB.GPUBackend(KA.CPU()),
    )
    for k in SP2D_INV
        Test.@test isapprox(sp_gpu[k].sums, sp_ref[k].sums; atol = 1.0f-4)
        Test.@test sp_gpu[k].counts == sp_ref[k].counts
    end
end

Test.@testset "Single-Pass 2D GPU (KA.CPU) parity vs Serial" begin
    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5
    distance_bins = LogBinEdges(collect(exp.(range(log(1000.0), log(50000.0), length = 6))))
    value_bins = _synthetic_value_bins(10; pad_infinite = true)
    n_val = length(value_bins) - 1
    n_bins = length(distance_bins) - 1

    # Mutating Serial reference (stacked accumulator).
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
    for (t, k) in enumerate(SP2D_INV)
        Test.@test sp_gpu[k].sums ≈ sums_ref[t, :, :]
        Test.@test sp_gpu[k].counts == counts_ref[t, :, :]
    end

    # Mutating GPU(KA.CPU) parity (stacked accumulator) — unchanged API.
    sums_gpu2 = zeros(Float64, 6, n_bins, n_val)
    counts_gpu2 = zeros(UInt32, 6, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu2, counts_gpu2, x, u, distance_bins, value_bins;
        backend = CB.GPUBackend(KA.CPU()),
    )
    Test.@test sums_gpu2 ≈ sums_ref
    Test.@test counts_gpu2 == counts_ref
end

# `value_bins` may be a heterogeneous NTuple{6} — log bins for the three non-negative invariants and
# linear/raw-vector bins for the three signed ones is the natural choice. The scatter is unrolled so
# each `vb` stays concretely typed; indexing the tuple with a runtime `t` would make it a `Union` and
# box a `digitize` dispatch on every pair x invariant.
Test.@testset "Single-Pass 2D heterogeneous value-bin tuple" begin
    FT = Float64
    N, nv = 60, 6
    Random.seed!(99)
    x, u = rand(FT, 2, N), randn(FT, 2, N)
    db = collect(FT, range(0.0, 2.0; length = 7))
    nd = length(db) - 1

    lin = LinearBinEdges(range(FT(-10), FT(10); length = nv + 1))
    lg = LogBinEdges(collect(FT, 10 .^ range(-4, 1; length = nv + 1)))
    raw = collect(FT, range(FT(-10), FT(10); length = nv + 1))
    het = (lg, lg, lg, lin, raw, lin)

    Test.@test !isconcretetype(eltype(het))   # the case the unroll exists for

    # Correct: each invariant must match a run with that invariant's bins used uniformly.
    got = SFC.calculate_structure_functions_single_pass_2d(
        x, u, db, het; backend = CB.SerialBackend(), verbose = false, show_progress = false,
    )
    for (t, k) in enumerate(SP2D_INV)
        ref = SFC.calculate_structure_functions_single_pass_2d(
            x, u, db, ntuple(_ -> het[t], 6);
            backend = CB.SerialBackend(), verbose = false, show_progress = false,
        )
        Test.@test got[k].sums ≈ ref[k].sums
        Test.@test got[k].counts == ref[k].counts
    end

    # Type-stability proxy: a boxed Union `vb` allocates per pair x invariant. At N=60 that is
    # 10_620 * 6 scatters, so per-op boxing would cost megabytes; the unrolled form costs ~nothing.
    sums = zeros(FT, 6, nd, nv)
    counts = zeros(UInt32, 6, nd, nv)
    f() = SFC.serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, db, het)
    f()
    Test.@test (@allocated f()) < 100_000
end
