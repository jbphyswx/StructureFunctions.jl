using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions:
    StructureFunctions as SF, StructureFunctionTypes as SFT, Calculations as SFC
using OhMyThreads: OhMyThreads  # load extension for ThreadedBackend / AutoBackend when nthreads() > 1
using KernelAbstractions: KernelAbstractions as KA
using Test: Test
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA

# Single-pass now returns a NamedTuple keyed by invariant (each entry a single-operator result).
# These are the six invariants in canonical order, and the matching reference operators.
const SP_INV = (:S2, :L2, :T2, :S3, :L3, :L1T2)
const SP_REF_TYPES = (
    SFT.SecondOrderStructureFunctionType(),
    SFT.LongitudinalSecondOrderStructureFunctionType(),
    SFT.TransverseSecondOrderStructureFunctionType(),
    SFT.ThirdOrderStructureFunctionType(),
    SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
    SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
)

# NaN-safe elementwise approx (single-pass leaves NaN in empty averaged bins).
_sp_nan_safe(a, b; atol) =
    all(((isnan(a[i]) && isnan(b[i])) || isapprox(a[i], b[i]; atol = atol)) for i in eachindex(a, b))

Test.@testset "Single-Pass Core Correctness & Helmholtz Parity" begin
    # 3 points: (0,0), (1,0), (0,1)
    # Velocities: (0,0), (1,0), (0,1)
    x = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]
    u = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]

    # Use strictly positive bins to prevent log(0.0) -> -Inf
    distance_bins = Float64[0.1, 1.0, 2.0]

    # 1. Test SerialBackend single-pass execution (raw output for sums/counts comparison)
    sp = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = CB.SerialBackend(), output_type = SF.StructureFunctionSumsAndCounts,
    )

    # Keyed-collection shape: six invariants + a helmholtz entry for point-field input.
    Test.@test keys(sp) == (SP_INV..., :helmholtz)
    Test.@test sp.S2 isa SF.StructureFunctionSumsAndCounts
    Test.@test length(sp.S2.sums) == 2
    # Entries are zero-copy views into the stacked accumulator (AbstractVector, not Vector).
    Test.@test sp.S2.sums isa AbstractVector{Float64}
    Test.@test sp.S2.counts isa AbstractVector{UInt32}
    Test.@test sp.helmholtz isa SF.HelmholtzDecomposition2D

    # 2. Test equivalence against standard multi-pass structure function calls
    distance_bins_ref = Float64[0.1, 1.0, 2.0]
    for t in 1:6
        res = SFC.calculate_structure_function(
            SP_REF_TYPES[t], x, u, distance_bins_ref;
            verbose = false, show_progress = false, output_type = SF.StructureFunctionSumsAndCounts,
        )
        entry = sp[SP_INV[t]]
        Test.@test isapprox(entry.sums, res.sums, atol = 1e-12)
        Test.@test entry.counts == res.counts
    end

    # 3. Test Helmholtz Decomposition Parity
    # Rotational + Divergent should sum up to Longitudinal + Transverse.
    h = sp.helmholtz
    D_rot = h.rotational_sums ./ max.(h.rotational_counts, 1)
    D_div = h.divergent_sums ./ max.(h.divergent_counts, 1)
    D_LL = sp.L2.sums ./ max.(sp.L2.counts, 1)
    D_TT = sp.T2.sums ./ max.(sp.T2.counts, 1)
    # The decomposition's stored long/trans values should match the raw L2/T2 means.
    Test.@test h.longitudinal_values ≈ D_LL
    Test.@test h.transverse_values ≈ D_TT

    valid_mask = sp.L2.counts .> 0
    if any(valid_mask)
        Test.@test isapprox(
            D_rot[valid_mask] + D_div[valid_mask], D_LL[valid_mask] + D_TT[valid_mask], atol = 1e-12,
        )
    end

    # 4. Compare AutoBackend against SerialBackend (per invariant, NaN-safe)
    sp_auto = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = CB.AutoBackend(), output_type = SF.StructureFunctionSumsAndCounts,
    )
    for k in SP_INV
        Test.@test _sp_nan_safe(sp_auto[k].sums, sp[k].sums; atol = 1e-12)
        Test.@test sp_auto[k].counts == sp[k].counts
    end
end

Test.@testset "Single-Pass 3D point parity" begin
    x = Float32[0.0 1.0 0.0 0.3;
                0.0 0.0 1.0 0.4;
                0.0 0.0 0.0 1.0]
    u = Float32[0.0 0.5 0.0 0.0;
                0.0 0.0 0.5 0.0;
                0.0 0.0 0.0 0.5]
    distance_bins = Float32[0.1, 1.0, 2.0]

    sp = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    Test.@test keys(sp) == (SP_INV..., :helmholtz)
    Test.@test length(sp.S2.sums) == 2

    sp_auto = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins; backend = CB.AutoBackend(),
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    sp_gpu = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins; backend = CB.GPUBackend(KA.CPU()),
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    for k in SP_INV
        Test.@test sp_auto[k].sums ≈ sp[k].sums
        Test.@test sp_auto[k].counts == sp[k].counts
        Test.@test isapprox(sp_gpu[k].sums, sp[k].sums; atol = 1.0f-4)
        Test.@test sp_gpu[k].counts == sp[k].counts
    end
end

Test.@testset "Single-Pass 3D auxiliary axes" begin
    x = rand(Float32, 3, 8)
    u = rand(Float32, 3, 8, 2)
    distance_bins = Float32[0.0, 0.75, 1.5, 3.0]

    batched = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    # Batched (auxiliary-axis) input has no Helmholtz entry — just the six invariants.
    Test.@test keys(batched) == SP_INV
    for b in 1:2
        spb = SFC.calculate_structure_functions_single_pass(
            x, @view(u[:, :, b]), distance_bins; backend = CB.SerialBackend(),
            output_type = SF.StructureFunctionSumsAndCounts,
        )
        for k in SP_INV
            Test.@test batched[k].sums[:, b] ≈ spb[k].sums
            Test.@test batched[k].counts[:, b] == spb[k].counts
        end
    end
end


# `helmholtz_decompose_2d` used to reconstruct bin midpoints from only the first and last edge via
# exp(min_log + (k-0.5)*log_step), i.e. it assumed log-uniform spacing unconditionally. On linear
# edges that placed midpoints outside their own bins (up to ~59% off) and silently returned a wrong
# decomposition. Midpoints now come from the actual edges as sqrt(e[k]*e[k+1]).
Test.@testset "Helmholtz decomposition — bin midpoints from actual edges" begin
    FT = Float64
    n_bins = 8
    lin_edges = collect(FT, range(0.5, 8.5; length = n_bins + 1))
    log_edges = collect(FT, 10 .^ range(log10(0.1), log10(10.0); length = n_bins + 1))

    # Non-constant D_TT - D_LL so the cumulative integral (and hence the midpoints) actually matter.
    counts = ones(UInt32, n_bins)
    L2 = FT[0.5k for k in 1:n_bins]
    T2 = FT[0.5k + 0.25k^2 for k in 1:n_bins]

    function expected(edges)
        mids = FT[sqrt(edges[k] * edges[k + 1]) for k in 1:n_bins]
        I = zeros(FT, n_bins)
        for k in 2:n_bins
            f_prev = (T2[k - 1] - L2[k - 1]) / mids[k - 1]
            f_curr = (T2[k] - L2[k]) / mids[k]
            I[k] = I[k - 1] + 0.5 * (f_prev + f_curr) * (mids[k] - mids[k - 1])
        end
        return mids, I
    end

    for edges in (lin_edges, log_edges)
        h = SFC.helmholtz_decompose_2d(edges, L2, counts, T2, counts)
        mids, I = expected(edges)
        Test.@test h.rotational_sums ≈ T2 .+ mids .* I
        Test.@test h.divergent_sums ≈ L2 .- mids .* I
        # The invariant the old closed form violated: each midpoint lies inside its own bin.
        for k in 1:n_bins
            Test.@test edges[k] <= mids[k] <= edges[k + 1]
        end
    end

    # Empty bins must not enter the cumulative integral as a fabricated zero.
    sparse_counts = copy(counts)
    sparse_counts[3] = 0
    h_sparse = SFC.helmholtz_decompose_2d(lin_edges, L2, sparse_counts, T2, sparse_counts)
    Test.@test isnan(h_sparse.longitudinal_values[3])
    Test.@test isnan(h_sparse.transverse_values[3])

    # A bin touching r <= 0 has no geometric midpoint and no r^-1 integrand: NaN, never a number.
    # The six invariants stay valid, so this must not error — bins starting at 0.0 are legitimate.
    h_zero = SFC.helmholtz_decompose_2d(
        collect(FT, range(0.0, 8.0; length = n_bins + 1)), L2, counts, T2, counts,
    )
    Test.@test all(isnan, h_zero.rotational_sums)
    Test.@test all(isnan, h_zero.divergent_sums)
end
