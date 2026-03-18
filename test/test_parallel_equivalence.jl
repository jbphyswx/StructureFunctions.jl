using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Test: Test
using StaticArrays: StaticArrays as SA
using Distributed: Distributed
using SharedArrays: SharedArrays

# Setup distributed environment if needed
if Distributed.nprocs() == 1
    Distributed.addprocs(2)
end

Distributed.@everywhere using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
Distributed.@everywhere using StaticArrays: StaticArrays as SA
Distributed.@everywhere using SharedArrays: SharedArrays

Test.@testset "Parallel Equivalence Verification" begin
    # Dataset
    N = 50
    x = ([rand() for _ in 1:N], [rand() for _ in 1:N])
    u = ([rand() for _ in 1:N], [rand() for _ in 1:N])
    bins = SA.SVector(((0.0, 0.5), (0.5, 1.0)))
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
    sx = (SharedArrays.SharedArray(x[1]), SharedArrays.SharedArray(x[2]))
    su = (SharedArrays.SharedArray(u[1]), SharedArrays.SharedArray(u[2]))

    res_dist = SFC.parallel_calculate_structure_function(
        sf_type,
        sx,
        su,
        bins;
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
end
