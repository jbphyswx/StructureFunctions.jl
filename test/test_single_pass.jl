using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions:
    StructureFunctions as SF, StructureFunctionTypes as SFT, Calculations as SFC
using Test: Test

# Single-pass returns a NamedTuple keyed by invariant; these are the six invariants in canonical order and their reference operators.
const SP_INV = (:S2, :L2, :T2, :S3, :L3, :L1T2)
const SP_REF_TYPES = (
    SFT.SecondOrderStructureFunctionType(),
    SFT.LongitudinalSecondOrderStructureFunctionType(),
    SFT.TransverseSecondOrderStructureFunctionType(),
    SFT.ThirdOrderStructureFunctionType(),
    SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
    SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
)

# The single pass equals each invariant's own entry, and its Helmholtz parts read L2 and T2 and sum to L2 + T2.
Test.@testset "Single-Pass Core Correctness & Helmholtz Parity" begin
    x = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]
    u = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]

    distance_bins = Float64[0.1, 1.0, 2.0]

    sp = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins, SF.StructureFunctionSumsAndCounts;
        backend = CB.SerialBackend(),
    )

    Test.@test keys(sp) == (SP_INV..., :helmholtz)
    Test.@test sp.S2 isa SF.StructureFunctionSumsAndCounts && sp.helmholtz isa SF.HelmholtzDecomposition2D

    refs = [SFC.calculate_structure_function(t, x, u, distance_bins, SF.StructureFunctionSumsAndCounts;
                                             backend = CB.SerialBackend()) for t in SP_REF_TYPES]
    Test.@test all(isapprox(sp[k].sums, r.sums; atol = 1e-12) for (k, r) in zip(SP_INV, refs))
    Test.@test all(sp[k].counts == r.counts for (k, r) in zip(SP_INV, refs))

    h = sp.helmholtz
    D_rot = h.rotational_sums ./ max.(h.rotational_counts, 1)
    D_div = h.divergent_sums ./ max.(h.divergent_counts, 1)
    D_LL = sp.L2.sums ./ max.(sp.L2.counts, 1)
    D_TT = sp.T2.sums ./ max.(sp.T2.counts, 1)
    Test.@test h.longitudinal_values ≈ D_LL
    Test.@test h.transverse_values ≈ D_TT

    valid_mask = sp.L2.counts .> 0
    Test.@test any(valid_mask)
    Test.@test isapprox(D_rot[valid_mask] + D_div[valid_mask], D_LL[valid_mask] + D_TT[valid_mask], atol = 1e-12)
end

# A batch over shared positions has no Helmholtz entry, and each slice equals the point single pass.
Test.@testset "Single-Pass 3D auxiliary axes" begin
    x = rand(3, 8)
    u = rand(3, 8, 2)
    distance_bins = [0.0, 0.75, 1.5, 3.0]

    batched = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
    )
    Test.@test keys(batched) == SP_INV
    slices = [SFC.calculate_structure_functions_single_pass(
        x, u[:, :, b], distance_bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
    ) for b in 1:2]
    Test.@test all(batched[k].sums[:, b] ≈ slices[b][k].sums for k in SP_INV, b in 1:2)
    Test.@test all(batched[k].counts[:, b] == slices[b][k].counts for k in SP_INV, b in 1:2)
end

# The two components sum to L2 + T2 on linear, log and zero-based edges, an empty bin giving NaN, not zero.
Test.@testset "Helmholtz decomposition — quadrature inputs" begin
    FT = Float64
    n_bins = 8
    lin_edges = collect(FT, range(0.5, 8.5; length = n_bins + 1))
    log_edges = SF.LogBinEdges(FT(0.1), FT(10), n_bins + 1)

    counts = ones(UInt32, n_bins)
    L2 = FT[0.5k for k in 1:n_bins]
    T2 = FT[0.5k + 0.25k^2 for k in 1:n_bins]

    for edges in (lin_edges, log_edges)
        h = SFC.helmholtz_decompose_2d(edges, L2, counts, T2, counts)
        Test.@test h.rotational_sums .+ h.divergent_sums ≈ L2 .+ T2
        Test.@test all(edges[k] <= m <= edges[k + 1] for (k, m) in enumerate(SF.midpoints(edges)))
    end

    sparse_counts = copy(counts)
    sparse_counts[3] = 0
    h_sparse = SFC.helmholtz_decompose_2d(lin_edges, L2, sparse_counts, T2, sparse_counts)
    Test.@test isnan(h_sparse.longitudinal_values[3])
    Test.@test isnan(h_sparse.transverse_values[3])

    h_zero = SFC.helmholtz_decompose_2d(
        collect(FT, range(0.0, 8.0; length = n_bins + 1)), L2, counts, T2, counts,
    )
    Test.@test all(isfinite, h_zero.rotational_sums)
    Test.@test all(isfinite, h_zero.divergent_sums)
    Test.@test h_zero.rotational_sums .+ h_zero.divergent_sums ≈ L2 .+ T2
end

# A solenoidal or irrotational D ∝ r^(2/3) puts its energy in one component, in any unit of length.
Test.@testset "Helmholtz decomposition — one-component fields" begin
    FT = Float64
    n_bins = 40
    edges = collect(FT, 10 .^ range(-3, 0; length = n_bins + 1))
    mids = FT[sqrt(edges[k] * edges[k + 1]) for k in 1:n_bins]
    counts = ones(UInt32, n_bins)
    base = mids .^ (2 / 3)

    h_rot = SFC.helmholtz_decompose_2d(edges, base, counts, (5 / 3) .* base, counts)
    Test.@test maximum(abs, h_rot.divergent_sums) < 0.05 * maximum(base)
    h_div = SFC.helmholtz_decompose_2d(edges, (5 / 3) .* base, counts, base, counts)
    Test.@test maximum(abs, h_div.rotational_sums) < 0.05 * maximum(base)

    Test.@test h_rot.rotational_sums .+ h_rot.divergent_sums ≈ base .+ (5 / 3) .* base

    λ = 1000.0
    h_scaled = SFC.helmholtz_decompose_2d(λ .* edges, base, counts, (5 / 3) .* base, counts)
    Test.@test h_scaled.divergent_sums ≈ h_rot.divergent_sums
    Test.@test h_scaled.rotational_sums ≈ h_rot.rotational_sums
end
