using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LogBinEdges, LinearBinEdges
using Test: Test
using StaticArrays: StaticArrays as SA
using Distributed: Distributed
using SharedArrays: SharedArrays

# Setup distributed environment if needed
if Distributed.nprocs() == 1
    Distributed.addprocs(2)
end

Distributed.@everywhere using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LogBinEdges, LinearBinEdges
Distributed.@everywhere using StaticArrays: StaticArrays as SA
Distributed.@everywhere using SharedArrays: SharedArrays
Distributed.@everywhere using OhMyThreads: OhMyThreads  # for hybrid DistributedBackend(ThreadedBackend())

Test.@testset "Parallel Equivalence Verification" begin
    # Dataset
    N = 50
    x = rand(2, N)
    u = rand(2, N)
    bins = SA.SVector(0.0, 0.5, 1.0)
    sf_type = SFT.LongitudinalSecondOrderStructureFunction

    # 1. Serial
    res_serial = SFC.calculate_structure_function(
        sf_type,
        x,
        u,
        bins;
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )
    out_serial, counts_serial = res_serial.sums, res_serial.counts

    # 2. Threaded
    res_thread = SFC.calculate_structure_function(
        sf_type,
        x,
        u,
        bins;
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )
    out_thread, counts_thread = res_thread.sums, res_thread.counts

    Test.@testset "Serial vs Threaded" begin
        Test.@test out_serial[1] ≈ out_thread[1]
        Test.@test out_serial[2] ≈ out_thread[2]
        Test.@test counts_serial == counts_thread
    end

    # 3. Distributed
    sx = SharedArrays.SharedArray{eltype(x)}(size(x))
    su = SharedArrays.SharedArray{eltype(u)}(size(u))
    sx .= x
    su .= u

    res_dist = SFC.calculate_structure_function(
        sf_type,
        sx,
        su,
        bins;
        backend = SFC.DistributedBackend(),
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )
    out_dist, counts_dist = res_dist.sums, res_dist.counts

    Test.@testset "Serial vs Distributed" begin
        Test.@test out_serial[1] ≈ out_dist[1]
        Test.@test out_serial[2] ≈ out_dist[2]
        Test.@test counts_serial == counts_dist
    end

    # 3b. Hybrid: DistributedBackend(ThreadedBackend()) — each worker threads over its share.
    res_hybrid = SFC.calculate_structure_function(
        sf_type,
        sx,
        su,
        bins;
        backend = SFC.DistributedBackend(SFC.ThreadedBackend()),
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )

    Test.@testset "Serial vs Distributed(Threaded) hybrid" begin
        Test.@test out_serial ≈ res_hybrid.sums
        Test.@test counts_serial == res_hybrid.counts
    end

    # 4. Distributed with bin count (Int) and LogBinEdges
    res_dist_int = SFC.calculate_structure_function(
        sf_type,
        sx,
        su,
        2;  # n_bins = 2
        backend = SFC.DistributedBackend(),
        bin_spacing = LogBinEdges,
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )

    res_serial_int = SFC.calculate_structure_function(
        sf_type,
        x,
        u,
        2;
        bin_spacing = LogBinEdges,
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )

    Test.@testset "Serial vs Distributed (Int/LogBinEdges)" begin
        Test.@test res_serial_int.sums ≈ res_dist_int.sums
        Test.@test res_serial_int.counts == res_dist_int.counts
    end
end
