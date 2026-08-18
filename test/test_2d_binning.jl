module Test2DBinning

using ComputationalBackends: ComputationalBackends as CB
using Test
using Random
using OhMyThreads
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionObjects as SFO, StructureFunctionTypes as SFT

@testset "2D Joint-Probability Binning Tests" begin
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
    @testset "L2SF Joint-Probability Binning & Mass Conservation" begin
        # 1D Baseline Calculation
        sf1d = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins;
            output_type = SF.StructureFunctionSumsAndCounts,
            backend = CB.SerialBackend(),
            verbose = false,
            show_progress = false
        )

        # 2D Joint-Probability Calculation
        sf2d = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins,
            l2_value_bins;
            backend = CB.SerialBackend(),
            verbose = false,
            show_progress = false
        )

        @test sf2d isa SFO.StructureFunction2DSumsAndCounts
        @test size(sf2d.sums) == (n_dist, 10)
        @test size(sf2d.counts) == (n_dist, 10)

        # Assert Mass Conservation
        for b in 1:n_dist
            # Sum counts along the value bins dimension
            counts_sum = sum(sf2d.counts[b, :])
            # Sum sums along the value bins dimension
            value_sum = sum(sf2d.sums[b, :])

            # In 1D, sums are in sf1d.sums, counts in sf1d.counts
            @test counts_sum ≈ sf1d.counts[b]
            @test value_sum ≈ sf1d.sums[b]
        end
    end

    # 3. Test L3SF (Longitudinal Third Order Structure Function)
    @testset "L3SF Symmetrical Joint-Probability Binning" begin
        # 1D Baseline
        sf1d = SFC.calculate_structure_function(
            SFT.L3SF,
            x_mat,
            u_mat,
            distance_bins;
            output_type = SF.StructureFunctionSumsAndCounts,
            backend = CB.SerialBackend(),
            verbose = false,
            show_progress = false
        )

        # 2D Joint
        sf2d = SFC.calculate_structure_function(
            SFT.L3SF,
            x_mat,
            u_mat,
            distance_bins,
            l3_value_bins;
            backend = CB.SerialBackend(),
            verbose = false,
            show_progress = false
        )

        @test sf2d isa SFO.StructureFunction2DSumsAndCounts
        @test size(sf2d.sums) == (n_dist, 10)

        # Assert Mass Conservation
        for b in 1:n_dist
            @test sum(sf2d.counts[b, :]) ≈ sf1d.counts[b]
            @test sum(sf2d.sums[b, :]) ≈ sf1d.sums[b]
        end
    end

    # 4. Test Array Input Equivalence
    @testset "Array Equivalence" begin
        sf_serial = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins,
            l2_value_bins;
            backend = CB.SerialBackend(),
            verbose = false,
            show_progress = false
        )

        sf_array = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins,
            l2_value_bins;
            backend = CB.SerialBackend(),
            verbose = false,
            show_progress = false
        )

        @test sf_serial.sums == sf_array.sums
        @test sf_serial.counts == sf_array.counts
    end

    # 5. Test Threaded Backend Equivalence (OhMyThreads)
    @testset "Serial vs Threaded Equivalence" begin
        sf_serial = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins,
            l2_value_bins;
            backend = CB.SerialBackend(),
            verbose = false,
            show_progress = false
        )

        sf_threaded = SFC.calculate_structure_function(
            SFT.L2SF,
            x_mat,
            u_mat,
            distance_bins,
            l2_value_bins;
            backend = CB.ThreadedBackend(),
            verbose = false,
            show_progress = false
        )

        @test sf_serial.sums ≈ sf_threaded.sums
        @test sf_serial.counts == sf_threaded.counts
    end

    # 6. Test Algebraic Operator Support (+)
    @testset "Base algebraic addition (+)" begin
        sf1 = SFC.calculate_structure_function(SFT.L2SF, x_mat, u_mat, distance_bins, l2_value_bins; verbose=false, show_progress=false)
        sf2 = SFC.calculate_structure_function(SFT.L2SF, x_mat, u_mat, distance_bins, l2_value_bins; verbose=false, show_progress=false)
        
        combined = sf1 + sf2
        @test combined.sums == sf1.sums .* 2
        @test combined.counts == sf1.counts .* 2
    end
end

end # module
