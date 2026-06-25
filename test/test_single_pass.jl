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
        backend = SFC.SerialBackend(), output_type = SF.StructureFunctionSumsAndCounts,
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
        backend = SFC.AutoBackend(), output_type = SF.StructureFunctionSumsAndCounts,
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
        x, u, distance_bins; backend = SFC.SerialBackend(),
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    Test.@test keys(sp) == (SP_INV..., :helmholtz)
    Test.@test length(sp.S2.sums) == 2

    sp_auto = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins; backend = SFC.AutoBackend(),
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    sp_gpu = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins; backend = SF.GPUBackend(KA.CPU()),
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
        x, u, distance_bins; backend = SFC.SerialBackend(),
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    # Batched (auxiliary-axis) input has no Helmholtz entry — just the six invariants.
    Test.@test keys(batched) == SP_INV
    for b in 1:2
        spb = SFC.calculate_structure_functions_single_pass(
            x, @view(u[:, :, b]), distance_bins; backend = SFC.SerialBackend(),
            output_type = SF.StructureFunctionSumsAndCounts,
        )
        for k in SP_INV
            Test.@test batched[k].sums[:, b] ≈ spb[k].sums
            Test.@test batched[k].counts[:, b] == spb[k].counts
        end
    end
end

Test.@testset "Single-Pass with Custom Distance Metric (Cityblock)" begin
    using Distances: Distances as DI

    x = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]
    u = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]

    distance_bins = Float64[0.1, 1.0, 2.0]
    metric = DI.Cityblock()

    sp = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.SerialBackend(), distance_metric = metric,
        output_type = SF.StructureFunctionSumsAndCounts,
    )

    distance_bins_ref = Float64[0.1, 1.0, 2.0]
    for t in 1:6
        res = SFC.calculate_structure_function(
            SP_REF_TYPES[t], x, u, distance_bins_ref;
            distance_metric = metric,
            verbose = false, show_progress = false, output_type = SF.StructureFunctionSumsAndCounts,
        )
        entry = sp[SP_INV[t]]
        Test.@test isapprox(entry.sums, res.sums, atol = 1e-12)
        Test.@test entry.counts == res.counts
    end

    # Compare AutoBackend against SerialBackend under Cityblock metric (per invariant, NaN-safe)
    sp_auto = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.AutoBackend(), distance_metric = metric,
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    for k in SP_INV
        Test.@test _sp_nan_safe(sp_auto[k].sums, sp[k].sums; atol = 1e-12)
        Test.@test sp_auto[k].counts == sp[k].counts
    end
end
