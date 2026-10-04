module Test2DBinning

using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using Test: Test
using Random: Random
using OhMyThreads: OhMyThreads
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionObjects as SFO, StructureFunctionTypes as SFT

Test.@testset "2D Joint-Probability Binning Tests" begin
    # 1. Generate clean synthetic test dataset
    Random.seed!(1234)
    n_points = 50
    # Coordinates in 2D (meters)
    x_coords = rand(n_points) .* 50000.0
    y_coords = rand(n_points) .* 50000.0
    x_mat = [x_coords'; y_coords']

    # Velocities in 2D (m/s)
    u_coords = randn(n_points) .* 0.5
    v_coords = randn(n_points) .* 0.5
    u_mat = [u_coords'; v_coords']

    distance_bins = [0.0, 10000.0, 20000.0, 30000.0, 50000.0]
    n_dist = length(distance_bins) - 1

    # Use extremely wide value bins to verify exact mathematical mass conservation (no clipping)
    l2_value_bins = range(0.0, 1000.0, length = 11) # 10 bins
    l3_value_bins = range(-1000.0, 1000.0, length = 11) # 10 bins

    # 2. Test L2SF (Longitudinal Second Order Structure Function)
    Test.@testset "L2SF Joint-Probability Binning & Mass Conservation" begin
        # 1D Baseline Calculation
        sf1d = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins,
            SF.StructureFunctionSumsAndCounts;
            backend = CB.SerialBackend()
        )

        # 2D Joint-Probability Calculation
        sf2d = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins,
            l2_value_bins;
            backend = CB.SerialBackend()
        )

        Test.@test sf2d isa SFO.StructureFunction2DSumsAndCounts
        Test.@test size(sf2d.sums) == (n_dist, 10)
        Test.@test size(sf2d.counts) == (n_dist, 10)

        # Assert Mass Conservation
        for b in 1:n_dist
            # Sum counts along the value bins dimension
            counts_sum = sum(sf2d.counts[b, :])
            # Sum sums along the value bins dimension
            value_sum = sum(sf2d.sums[b, :])

            # In 1D, sums are in sf1d.sums, counts in sf1d.counts
            Test.@test counts_sum ≈ sf1d.counts[b]
            Test.@test value_sum ≈ sf1d.sums[b]
        end
    end

    # 3. Test L3SF (Longitudinal Third Order Structure Function)
    Test.@testset "L3SF Symmetrical Joint-Probability Binning" begin
        # 1D Baseline
        sf1d = SFC.calculate_structure_function(
            SFT.L3SF,
            x_mat,
            u_mat,
            distance_bins,
            SF.StructureFunctionSumsAndCounts;
            backend = CB.SerialBackend()
        )

        # 2D Joint
        sf2d = SFC.calculate_structure_function(
            SFT.L3SF,
            x_mat,
            u_mat,
            distance_bins,
            l3_value_bins;
            backend = CB.SerialBackend()
        )

        Test.@test sf2d isa SFO.StructureFunction2DSumsAndCounts
        Test.@test size(sf2d.sums) == (n_dist, 10)

        # Assert Mass Conservation
        for b in 1:n_dist
            Test.@test sum(sf2d.counts[b, :]) ≈ sf1d.counts[b]
            Test.@test sum(sf2d.sums[b, :]) ≈ sf1d.sums[b]
        end
    end

    # 4. Test Threaded Backend Equivalence (OhMyThreads)
    Test.@testset "Serial vs Threaded Equivalence" begin
        sf_serial = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins,
            l2_value_bins;
            backend = CB.SerialBackend()
        )

        sf_threaded = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins,
            l2_value_bins;
            backend = CB.ThreadedBackend()
        )

        Test.@test sf_serial.sums ≈ sf_threaded.sums
        Test.@test sf_serial.counts == sf_threaded.counts
    end

    # 5. Test Algebraic Operator Support (+)
    Test.@testset "Base algebraic addition (+)" begin
        sf1 = SFC.calculate_structure_function(SFT.L2SF, x_mat, u_mat, distance_bins, l2_value_bins)
        sf2 = SFC.calculate_structure_function(SFT.L2SF, x_mat, u_mat, distance_bins, l2_value_bins)
        
        combined = sf1 + sf2
        Test.@test combined.sums == sf1.sums .+ sf2.sums
        Test.@test combined.counts == sf1.counts .+ sf2.counts
    end

    # 6. The device joint histogram matches serial for 6 and 201 value bins at widths 2 and 3.
    Test.@testset "the device global-atomic joint route agrees at every width" begin
        Random.seed!(4321)
        n = 150
        dbins = collect(range(0.0, 1.2; length = 7))
        routes = (("tiled", collect(range(-3.0, 3.0; length = 6))),
                  ("global atomic", collect(range(-3.0, 3.0; length = 201))))
        for D in (2, 3), (route, vbins) in routes
            x, u = rand(D, n), randn(D, n)
            ref = SFC.calculate_structure_function(SFT.L2SF, x, u, dbins, vbins;
                backend = CB.SerialBackend())
            dev = SFC.calculate_structure_function(SFT.L2SF, x, u, dbins, vbins;
                backend = CB.GPUBackend(KA.CPU()))
            Test.@test sum(ref.counts) > 0
            Test.@test dev.counts == ref.counts
            Test.@test dev.sums ≈ ref.sums
        end
    end
end

end # module
