module TestInplace

using ComputationalBackends: ComputationalBackends as CB
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
    x_mat = [x_coords'; y_coords']

    u_coords = randn(n_points) .* 0.5
    v_coords = randn(n_points) .* 0.5
    u_mat = [u_coords'; v_coords']

    distance_bins = [0.0, 10000.0, 20000.0, 30000.0, 50000.0]
    n_dist = length(distance_bins) - 1
    value_bins = range(-1.0, 1.0, length = 11)
    n_vals = length(value_bins) - 1

    # 1. 1D Serial mutating Array tests
    @testset "1D Serial Mutating Array Correctness & Accumulation" begin
        # Baselines
        bas = SFC.calculate_structure_function(
            SFT.L2SF, x_mat, u_mat, distance_bins;
            output_type = SF.StructureFunctionSumsAndCounts, backend = CB.SerialBackend(), verbose = false, show_progress = false
        )

        sums = zeros(Float64, n_dist)
        counts = zeros(UInt32, n_dist)

        # Mutate
        SFC.serial_calculate_structure_function!(sums, counts, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)
        @test sums == bas.sums
        @test counts == bas.counts

        # Accumulation (calling twice should double the values)
        SFC.serial_calculate_structure_function!(sums, counts, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)
        @test sums ≈ bas.sums .* 2
        @test counts == bas.counts .* 2
    end

    # 2. 1D Serial mutating Array tests
    @testset "1D Serial Mutating Array Correctness & Accumulation" begin
        bas = SFC.calculate_structure_function(
            SFT.L2SF, x_mat, u_mat, distance_bins;
            output_type = SF.StructureFunctionSumsAndCounts, backend = CB.SerialBackend(), verbose = false, show_progress = false
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

    # 3. 2D Serial mutating Array and Array tests
    @testset "2D Serial Mutating Array & Array Correctness & Accumulation" begin
        bas_arr = SFC.calculate_structure_function(
            SFT.L2SF, x_mat, u_mat, distance_bins, value_bins;
            backend = CB.SerialBackend(), verbose = false, show_progress = false
        )

        sums_arr = zeros(Float64, n_dist, n_vals)
        counts_arr = zeros(UInt32, n_dist, n_vals)

        # Mutate Array
        SFC.serial_calculate_structure_function!(sums_arr, counts_arr, SFT.L2SF, x_mat, u_mat, distance_bins, value_bins; verbose=false, show_progress=false)
        @test sums_arr == bas_arr.sums
        @test counts_arr == bas_arr.counts

        # Accumulation
        SFC.serial_calculate_structure_function!(sums_arr, counts_arr, SFT.L2SF, x_mat, u_mat, distance_bins, value_bins; verbose=false, show_progress=false)
        @test sums_arr ≈ bas_arr.sums .* 2
        @test counts_arr == bas_arr.counts .* 2

        # Mutate Array
        sums_arr = zeros(Float64, n_dist, n_vals)
        counts_arr = zeros(UInt32, n_dist, n_vals)
        SFC.serial_calculate_structure_function!(sums_arr, counts_arr, SFT.L2SF, x_mat, u_mat, distance_bins, value_bins; verbose=false, show_progress=false)
        @test sums_arr == bas_arr.sums
        @test counts_arr == bas_arr.counts
    end

    # 4. Threaded mutating tests
    @testset "1D & 2D Threaded Mutating Parity & Low Allocation" begin
        # 1D Array Threaded
        sums_ser = zeros(Float64, n_dist)
        counts_ser = zeros(UInt32, n_dist)
        SFC.serial_calculate_structure_function!(sums_ser, counts_ser, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)

        sums_thr = zeros(Float64, n_dist)
        counts_thr = zeros(UInt32, n_dist)
        SFC.threaded_calculate_structure_function!(sums_thr, counts_thr, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)

        @test sums_ser ≈ sums_thr
        @test counts_ser == counts_thr

        # 1D Array Threaded
        sums_thr_arr = zeros(Float64, n_dist)
        counts_thr_arr = zeros(UInt32, n_dist)
        SFC.threaded_calculate_structure_function!(sums_thr_arr, counts_thr_arr, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)
        @test sums_ser ≈ sums_thr_arr
        @test counts_ser == counts_thr_arr

        # 2D Array Threaded
        sums_2d_ser = zeros(Float64, n_dist, n_vals)
        counts_2d_ser = zeros(UInt32, n_dist, n_vals)
        SFC.serial_calculate_structure_function!(sums_2d_ser, counts_2d_ser, SFT.L2SF, x_mat, u_mat, distance_bins, value_bins; verbose=false, show_progress=false)

        sums_2d_thr = zeros(Float64, n_dist, n_vals)
        counts_2d_thr = zeros(UInt32, n_dist, n_vals)
        SFC.threaded_calculate_structure_function!(sums_2d_thr, counts_2d_thr, SFT.L2SF, x_mat, u_mat, distance_bins, value_bins; verbose=false, show_progress=false)

        @test sums_2d_ser ≈ sums_2d_thr
        @test counts_2d_ser == counts_2d_thr

        # Allocation checks: chunked OhMyThreads must allocate O(n_threads) which is extremely lightweight
        # We check that it runs without errors or excessive allocation.
        alloc1 = @allocated SFC.threaded_calculate_structure_function!(sums_thr, counts_thr, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)
        @info "Threaded 1D Array Mutating call allocation: $alloc1 bytes"
        @test alloc1 < 250_000 # extremely lightweight compared to O(N_points)
    end

    # 5. Public backend dispatch calculate_structure_function! tests
    @testset "Public Entrypoints & Backend Dispatch" begin
        # 1D Public AutoBackend (resolves to threaded or serial)
        sums_pub = zeros(Float64, n_dist)
        counts_pub = zeros(UInt32, n_dist)
        SF.calculate_structure_function!(sums_pub, counts_pub, SFT.L2SF, x_mat, u_mat, distance_bins; backend=CB.AutoBackend(), verbose=false, show_progress=false)

        sums_bas = zeros(Float64, n_dist)
        counts_bas = zeros(UInt32, n_dist)
        SFC.serial_calculate_structure_function!(sums_bas, counts_bas, SFT.L2SF, x_mat, u_mat, distance_bins; verbose=false, show_progress=false)

        @test sums_pub ≈ sums_bas
        @test counts_pub == counts_bas

        # 2D Public AutoBackend
        sums_2d_pub = zeros(Float64, n_dist, n_vals)
        counts_2d_pub = zeros(UInt32, n_dist, n_vals)
        SF.calculate_structure_function!(sums_2d_pub, counts_2d_pub, SFT.L2SF, x_mat, u_mat, distance_bins, value_bins; backend=CB.AutoBackend(), verbose=false, show_progress=false)

        sums_2d_bas = zeros(Float64, n_dist, n_vals)
        counts_2d_bas = zeros(UInt32, n_dist, n_vals)
        SFC.serial_calculate_structure_function!(sums_2d_bas, counts_2d_bas, SFT.L2SF, x_mat, u_mat, distance_bins, value_bins; verbose=false, show_progress=false)

        @test sums_2d_pub ≈ sums_2d_bas
        @test counts_2d_pub == counts_2d_bas
    end
end

# Every `!` entry ACCUMULATES into the caller's buffers; zeroing belongs to the non-mutating
# wrappers. Batch-leading drivers used to overwrite (via `permutedims!`) and the tensor driver used
# to `fill!` the caller's arrays, so which contract you got depended only on the rank of the input.
@testset "Mutating API accumulates for every input shape" begin
    Random.seed!(4242)
    FT = Float64
    N, B = 24, 3
    bins = collect(FT, range(0.0, 2.0; length = 7))
    nb = length(bins) - 1
    sft = SFT.L2SFType()
    x2, u2 = rand(FT, 2, N), rand(FT, 2, N)
    u3 = rand(FT, 2, N, B)

    # (a) point-field: the family that already accumulated.
    s1, c1 = zeros(FT, nb), zeros(UInt32, nb)
    SFC.calculate_structure_function!(s1, c1, sft, x2, u2, bins; verbose = false, show_progress = false)
    s2, c2 = copy(s1), copy(c1)
    SFC.calculate_structure_function!(s2, c2, sft, x2, u2, bins; verbose = false, show_progress = false)
    @test s2 ≈ 2 .* s1
    @test c2 == 2 .* c1

    # (b) batch / auxiliary axes: previously overwrote.
    bs1, bc1 = zeros(FT, nb, B), zeros(UInt32, nb, B)
    SFC.calculate_structure_function!(bs1, bc1, sft, x2, u3, bins; verbose = false, show_progress = false)
    bs2, bc2 = copy(bs1), copy(bc1)
    SFC.calculate_structure_function!(bs2, bc2, sft, x2, u3, bins; verbose = false, show_progress = false)
    @test bs2 ≈ 2 .* bs1
    @test bc2 == 2 .* bc1

    # (c) tensor: previously `fill!`ed the caller's arrays, so it could never accumulate.
    ts1, tc1 = zeros(FT, 2, 2, nb), zeros(UInt32, nb)
    SFC.calculate_structure_function_tensor!(ts1, tc1, Val(2), x2, u2, bins; verbose = false, show_progress = false)
    ts2, tc2 = copy(ts1), copy(tc1)
    SFC.calculate_structure_function_tensor!(ts2, tc2, Val(2), x2, u2, bins; verbose = false, show_progress = false)
    @test ts2 ≈ 2 .* ts1
    @test tc2 == 2 .* tc1

    # The non-mutating wrappers still own the zeroing, so repeated calls are independent.
    r1 = SFC.calculate_structure_function_tensor(Val(2), x2, u2, bins;
                                                output_type = SFO.StructureFunctionTensorSumsAndCounts)
    r2 = SFC.calculate_structure_function_tensor(Val(2), x2, u2, bins;
                                                output_type = SFO.StructureFunctionTensorSumsAndCounts)
    @test r1.sums ≈ r2.sums
    @test r1.counts == r2.counts
end

end # module
