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


Test.@testset "Helmholtz decomposition — quadrature inputs" begin
    FT = Float64
    n_bins = 8
    lin_edges = collect(FT, range(0.5, 8.5; length = n_bins + 1))
    log_edges = SF.LogBinEdges(collect(FT, 10 .^ range(log10(0.1), log10(10.0); length = n_bins + 1)))

    # Non-constant D_TT - D_LL so the cumulative integral, and hence the abscissae, matter.
    counts = ones(UInt32, n_bins)
    L2 = FT[0.5k for k in 1:n_bins]
    T2 = FT[0.5k + 0.25k^2 for k in 1:n_bins]

    for edges in (lin_edges, log_edges)
        h = SFC.helmholtz_decompose_2d(edges, L2, counts, T2, counts)
        # The two components carry the total energy between them, whatever the abscissae.
        Test.@test h.rotational_sums .+ h.divergent_sums ≈ L2 .+ T2
        # Each abscissa lies inside its own bin.
        for (k, m) in enumerate(SF.midpoints(edges))
            Test.@test edges[k] <= m <= edges[k + 1]
        end
    end

    # Empty bins must not enter the cumulative integral as a fabricated zero.
    sparse_counts = copy(counts)
    sparse_counts[3] = 0
    h_sparse = SFC.helmholtz_decompose_2d(lin_edges, L2, sparse_counts, T2, sparse_counts)
    Test.@test isnan(h_sparse.longitudinal_values[3])
    Test.@test isnan(h_sparse.transverse_values[3])

    # Uniform edges starting at 0 have abscissae at 0.5, 1.5, …, so the r^-1 integrand is finite and
    # the decomposition is defined throughout.
    h_zero = SFC.helmholtz_decompose_2d(
        collect(FT, range(0.0, 8.0; length = n_bins + 1)), L2, counts, T2, counts,
    )
    Test.@test all(isfinite, h_zero.rotational_sums)
    Test.@test all(isfinite, h_zero.divergent_sums)
    Test.@test h_zero.rotational_sums .+ h_zero.divergent_sums ≈ L2 .+ T2
end

# Physical gates on the decomposition, independent of how it is implemented.
#
# In 2D a solenoidal field satisfies D_TT = D_LL + r dD_LL/dr and an irrotational one satisfies
# D_LL = D_TT + r dD_TT/dr. Feeding either exactly must put all the energy in one component and
# none in the other. For D ∝ r^(2/3) the surviving error is the quadrature's, ~1% of D_LL.
Test.@testset "Helmholtz decomposition — one-component fields" begin
    FT = Float64
    n_bins = 40
    edges = collect(FT, 10 .^ range(-3, 0; length = n_bins + 1))
    mids = FT[sqrt(edges[k] * edges[k + 1]) for k in 1:n_bins]
    counts = ones(UInt32, n_bins)
    base = mids .^ (2 / 3)

    # Solenoidal: D_div == 0.
    h_rot = SFC.helmholtz_decompose_2d(edges, base, counts, (5 / 3) .* base, counts)
    Test.@test maximum(abs, h_rot.divergent_sums) < 0.05 * maximum(base)
    # Irrotational: D_rot == 0.
    h_div = SFC.helmholtz_decompose_2d(edges, (5 / 3) .* base, counts, base, counts)
    Test.@test maximum(abs, h_div.rotational_sums) < 0.05 * maximum(base)

    # Both components sum back to the total energy.
    Test.@test h_rot.rotational_sums .+ h_rot.divergent_sums ≈ base .+ (5 / 3) .* base

    # I(r) = ∫(D_TT - D_LL)/s ds is invariant under r -> λr at fixed velocities, so every output is
    # too: a change of length unit cannot move a velocity-squared quantity.
    λ = 1000.0
    h_scaled = SFC.helmholtz_decompose_2d(λ .* edges, base, counts, (5 / 3) .* base, counts)
    Test.@test h_scaled.divergent_sums ≈ h_rot.divergent_sums
    Test.@test h_scaled.rotational_sums ≈ h_rot.rotational_sums
end
