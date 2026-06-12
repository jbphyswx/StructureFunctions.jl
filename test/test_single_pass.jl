using StructureFunctions:
    StructureFunctions as SF, StructureFunctionTypes as SFT, Calculations as SFC
using OhMyThreads: OhMyThreads  # load extension for ThreadedBackend / AutoBackend when nthreads() > 1
using Test: Test
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA

Test.@testset "Single-Pass Core Correctness & Helmholtz Parity" begin
    # 3 points: (0,0), (1,0), (0,1)
    # Velocities: (0,0), (1,0), (0,1)
    x = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]
    u = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]
                
    # Use strictly positive bins to prevent log(0.0) -> -Inf
    distance_bins = Float64[0.1, 1.0, 2.0]
    
    # 1. Test SerialBackend single-pass execution
    sums, counts = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.SerialBackend()
    )
    
    # Check dimensions
    Test.@test size(sums) == (10, 2)
    Test.@test size(counts) == (10, 2)
    Test.@test sums isa Matrix{Float64}
    Test.@test counts isa Matrix{UInt32}
    
    # 2. Test equivalence against standard multi-pass structure function calls
    distance_bins_ref = Float64[0.1, 1.0, 2.0]
    
    # Let's map single-pass index -> SFT type
    sft_types = [
        SFT.SecondOrderStructureFunctionType(),
        SFT.LongitudinalSecondOrderStructureFunctionType(),
        SFT.TransverseSecondOrderStructureFunctionType(),
        SFT.ThirdOrderStructureFunctionType(),
        SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
        SFT.DiagonalInconsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalConsistentThirdOrderStructureFunctionType()
    ]
    
    x_tuple = (x[1, :], x[2, :])
    u_tuple = (u[1, :], u[2, :])
    
    for t in 1:8
        # Standard calculation
        res = SFC.calculate_structure_function(
            sft_types[t], x_tuple, u_tuple, distance_bins_ref;
            verbose = false, show_progress = false, return_sums_and_counts = true
        )
        # Compare sums with an absolute tolerance for floating-point underflow differences
        Test.@test isapprox(sums[t, :], res.sums, atol=1e-12)
        # Compare counts
        Test.@test counts[t, :] == res.counts
    end
    
    # 3. Test Helmholtz Decomposition Parity
    # Rotational (9) + Divergent (10) should sum up to Longitudinal (2) + Transverse (3)
    D_rot = sums[9, :] ./ max.(counts[9, :], 1)
    D_div = sums[10, :] ./ max.(counts[10, :], 1)
    D_LL = sums[2, :] ./ max.(counts[2, :], 1)
    D_TT = sums[3, :] ./ max.(counts[3, :], 1)
    
    # Only compare where counts are non-zero
    valid_mask = counts[2, :] .> 0
    if any(valid_mask)
        Test.@test isapprox(D_rot[valid_mask] + D_div[valid_mask], D_LL[valid_mask] + D_TT[valid_mask], atol=1e-12)
    end
    
    # 4. Compare ThreadedBackend against SerialBackend
    t_sums, t_counts = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.AutoBackend()
    )
    
    # Handle NaNs safely during parity verification
    nan_safe_match = all(
        (isnan(t_sums[i]) && isnan(sums[i])) || isapprox(t_sums[i], sums[i], atol=1e-12)
        for i in eachindex(sums)
    )
    Test.@test nan_safe_match
    Test.@test t_counts == counts
end

Test.@testset "Single-Pass with Custom Distance Metric (Cityblock)" begin
    using Distances: Distances as DI

    x = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]
    u = Float64[0.0 1.0 0.0;
                0.0 0.0 1.0]
    
    distance_bins = Float64[0.1, 1.0, 2.0]
    metric = DI.Cityblock()

    # 1. Test SerialBackend single-pass execution with Cityblock metric
    sums, counts = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.SerialBackend(),
        distance_metric = metric
    )

    # 2. Test equivalence against standard multi-pass structure function calls with Cityblock
    distance_bins_ref = Float64[0.1, 1.0, 2.0]
    
    sft_types = [
        SFT.SecondOrderStructureFunctionType(),
        SFT.LongitudinalSecondOrderStructureFunctionType(),
        SFT.TransverseSecondOrderStructureFunctionType(),
        SFT.ThirdOrderStructureFunctionType(),
        SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
        SFT.DiagonalInconsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalConsistentThirdOrderStructureFunctionType()
    ]
    
    x_tuple = (x[1, :], x[2, :])
    u_tuple = (u[1, :], u[2, :])
    
    for t in 1:8
        # Standard calculation with distance_metric
        res = SFC.calculate_structure_function(
            sft_types[t], x_tuple, u_tuple, distance_bins_ref;
            distance_metric = metric,
            verbose = false, show_progress = false, return_sums_and_counts = true
        )
        Test.@test isapprox(sums[t, :], res.sums, atol=1e-12)
        Test.@test counts[t, :] == res.counts
    end

    # 3. Compare ThreadedBackend against SerialBackend under Cityblock metric
    t_sums, t_counts = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.AutoBackend(),
        distance_metric = metric
    )
    nan_safe_match = all(
        (isnan(t_sums[i]) && isnan(sums[i])) || isapprox(t_sums[i], sums[i], atol=1e-12)
        for i in eachindex(sums)
    )
    Test.@test nan_safe_match
    Test.@test t_counts == counts
end
