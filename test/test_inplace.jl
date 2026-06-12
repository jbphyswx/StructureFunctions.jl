module TestInplace

using Test
using Random
using OhMyThreads
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionObjects as SFO, StructureFunctionTypes as SFT

@testset "In-place / Pre-allocated Buffer API Tests" begin
    # Generate synthetic test dataset
    Random.seed!(1234)
    n_points = 60
    x_coords = rand(n_points) .* 50000.0
    y_coords = rand(n_points) .* 50000.0
    x_tuple = (x_coords, y_coords)
    x_mat = [x_coords'; y_coords']

    u_coords = randn(n_points) .* 0.5
    v_coords = randn(n_points) .* 0.5
    u_tuple = (u_coords, v_coords)
    u_mat = [u_coords'; v_coords']

    distance_bins = [(0.0, 10000.0), (10000.0, 20000.0), (20000.0, 30000.0), (30000.0, 50000.0)]
    n_dist = length(distance_bins)
    value_bins = range(-1.0, 1.0, length = 11)
    n_vals = length(value_bins) - 1

    # 1. 1D Serial mutating Tuple tests
    @testset "1D Serial Mutating Tuple Correctness & Accumulation" begin
        # Baselines
        bas = SFC.calculate_structure_function(
            SFT.L2SF, x_tuple, u_tuple, distance_bins;
            return_sums_and_counts = true, backend = SFC.SerialBackend(), verbose = false, show_progress = false
        )

        sums = zeros(Float64, n_dist)
        counts = zeros(UInt32, n_dist)

        # Mutate
        SFC.serial_calculate_structure_function!(sums, counts, SFT.L2SF, x_tuple, u_tuple, distance_bins; verbose=false, show_progress=false)
        @test sums == bas.sums
        @test counts == bas.counts

        # Accumulation (calling twice should double the values)
        SFC.serial_calculate_structure_function!(sums, counts, SFT.L2SF, x_tuple, u_tuple, distance_bins; verbose=false, show_progress=false)
        @test sums ≈ bas.sums .* 2
        @test counts == bas.counts .* 2
    end

    # 2. 1D Serial mutating Array tests
    @testset "1D Serial Mutating Array Correctness & Accumulation" begin
        bas = SFC.calculate_structure_function(
            SFT.L2SF, x_mat, u_mat, distance_bins;
            return_sums_and_counts = true, backend = SFC.SerialBackend(), verbose = false, show_progress = false
        )

        sums = zeros(Float64, n_dist)
        counts = zeros(UInt32, n_dist)

        SFC.serial_calculate_structure_function!(sums, counts, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)
        @test sums == bas.sums
        @test counts == bas.counts

        SFC.serial_calculate_structure_function!(sums, counts, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)
        @test sums ≈ bas.sums .* 2
        @test counts == bas.counts .* 2
    end

    # 3. 2D Serial mutating Tuple and Array tests
    @testset "2D Serial Mutating Tuple & Array Correctness & Accumulation" begin
        bas_tup = SFC.calculate_structure_function(
            SFT.L2SF, x_tuple, u_tuple, distance_bins, value_bins;
            backend = SFC.SerialBackend(), verbose = false, show_progress = false
        )

        sums_tup = zeros(Float64, n_dist, n_vals)
        counts_tup = zeros(UInt32, n_dist, n_vals)

        # Mutate Tuple
        SFC.serial_calculate_structure_function!(sums_tup, counts_tup, SFT.L2SF, x_tuple, u_tuple, distance_bins, value_bins; verbose=false, show_progress=false)
        @test sums_tup == bas_tup.sums
        @test counts_tup == bas_tup.counts

        # Accumulation
        SFC.serial_calculate_structure_function!(sums_tup, counts_tup, SFT.L2SF, x_tuple, u_tuple, distance_bins, value_bins; verbose=false, show_progress=false)
        @test sums_tup ≈ bas_tup.sums .* 2
        @test counts_tup == bas_tup.counts .* 2

        # Mutate Array
        sums_arr = zeros(Float64, n_dist, n_vals)
        counts_arr = zeros(UInt32, n_dist, n_vals)
        SFC.serial_calculate_structure_function!(sums_arr, counts_arr, SFT.L2SF, x_mat, u_mat, distance_bins, value_bins; verbose=false, show_progress=false)
        @test sums_arr == bas_tup.sums
        @test counts_arr == bas_tup.counts
    end

    # 4. Threaded mutating tests
    @testset "1D & 2D Threaded Mutating Parity & Low Allocation" begin
        # 1D Tuple Threaded
        sums_ser = zeros(Float64, n_dist)
        counts_ser = zeros(UInt32, n_dist)
        SFC.serial_calculate_structure_function!(sums_ser, counts_ser, SFT.L2SF, x_tuple, u_tuple, distance_bins; verbose=false, show_progress=false)

        sums_thr = zeros(Float64, n_dist)
        counts_thr = zeros(UInt32, n_dist)
        SFC.threaded_calculate_structure_function!(sums_thr, counts_thr, SFT.L2SF, x_tuple, u_tuple, distance_bins; verbose=false, show_progress=false)

        @test sums_ser ≈ sums_thr
        @test counts_ser == counts_thr

        # 1D Array Threaded
        sums_thr_arr = zeros(Float64, n_dist)
        counts_thr_arr = zeros(UInt32, n_dist)
        SFC.threaded_calculate_structure_function!(sums_thr_arr, counts_thr_arr, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)
        @test sums_ser ≈ sums_thr_arr
        @test counts_ser == counts_thr_arr

        # 2D Tuple Threaded
        sums_2d_ser = zeros(Float64, n_dist, n_vals)
        counts_2d_ser = zeros(UInt32, n_dist, n_vals)
        SFC.serial_calculate_structure_function!(sums_2d_ser, counts_2d_ser, SFT.L2SF, x_tuple, u_tuple, distance_bins, value_bins; verbose=false, show_progress=false)

        sums_2d_thr = zeros(Float64, n_dist, n_vals)
        counts_2d_thr = zeros(UInt32, n_dist, n_vals)
        SFC.threaded_calculate_structure_function!(sums_2d_thr, counts_2d_thr, SFT.L2SF, x_tuple, u_tuple, distance_bins, value_bins; verbose=false, show_progress=false)

        @test sums_2d_ser ≈ sums_2d_thr
        @test counts_2d_ser == counts_2d_thr

        # Allocation checks: chunked OhMyThreads must allocate O(n_threads) which is extremely lightweight
        # We check that it runs without errors or excessive allocation.
        alloc1 = @allocated SFC.threaded_calculate_structure_function!(sums_thr, counts_thr, SFT.L2SF, x_tuple, u_tuple, distance_bins; verbose=false, show_progress=false)
        @info "Threaded 1D Tuple Mutating call allocation: $alloc1 bytes"
        @test alloc1 < 250_000 # extremely lightweight compared to O(N_points)
    end

    # 5. Public backend dispatch calculate_structure_function! tests
    @testset "Public Entrypoints & Backend Dispatch" begin
        # 1D Public AutoBackend (resolves to threaded or serial)
        sums_pub = zeros(Float64, n_dist)
        counts_pub = zeros(UInt32, n_dist)
        SF.calculate_structure_function!(sums_pub, counts_pub, SFT.L2SF, x_tuple, u_tuple, distance_bins; backend=SF.AutoBackend(), verbose=false, show_progress=false)

        sums_bas = zeros(Float64, n_dist)
        counts_bas = zeros(UInt32, n_dist)
        SFC.serial_calculate_structure_function!(sums_bas, counts_bas, SFT.L2SF, x_tuple, u_tuple, distance_bins; verbose=false, show_progress=false)

        @test sums_pub ≈ sums_bas
        @test counts_pub == counts_bas

        # 2D Public AutoBackend
        sums_2d_pub = zeros(Float64, n_dist, n_vals)
        counts_2d_pub = zeros(UInt32, n_dist, n_vals)
        SF.calculate_structure_function!(sums_2d_pub, counts_2d_pub, SFT.L2SF, x_tuple, u_tuple, distance_bins, value_bins; backend=SF.AutoBackend(), verbose=false, show_progress=false)

        sums_2d_bas = zeros(Float64, n_dist, n_vals)
        counts_2d_bas = zeros(UInt32, n_dist, n_vals)
        SFC.serial_calculate_structure_function!(sums_2d_bas, counts_2d_bas, SFT.L2SF, x_tuple, u_tuple, distance_bins, value_bins; verbose=false, show_progress=false)

        @test sums_2d_pub ≈ sums_2d_bas
        @test counts_2d_pub == counts_2d_bas
    end
end

end # module
