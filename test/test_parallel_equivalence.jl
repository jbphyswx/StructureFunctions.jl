using StructureFunctions
using Test
using StaticArrays
using Distributed
using SharedArrays

# Setup distributed environment if needed
if nprocs() == 1
    addprocs(2)
end

@everywhere using StructureFunctions
@everywhere using StaticArrays
@everywhere using SharedArrays

const SF = StructureFunctions
const SFC = StructureFunctions.Calculations
const SFT = StructureFunctions.StructureFunctionTypes

@testset "Parallel Equivalence Verification" begin
    # Dataset
    N = 50
    x = ([rand() for _ in 1:N], [rand() for _ in 1:N])
    u = ([rand() for _ in 1:N], [rand() for _ in 1:N])
    bins = SVector(((0.0, 0.5), (0.5, 1.0)))
    sf_type = SFT.LongitudinalSecondOrderStructureFunction()

    # 1. Serial
    out_serial, counts_serial = SFC.calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false, return_sums_and_counts=true)

    # 2. Threaded
    out_thread, counts_thread = SFC.calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false, return_sums_and_counts=true)
    
    @testset "Serial vs Threaded" begin
        @test out_serial[1] ≈ out_thread[1]
        @test out_serial[2] ≈ out_thread[2]
        @test counts_serial == counts_thread
    end

    # 3. Distributed
    sx = (SharedArray(x[1]), SharedArray(x[2]))
    su = (SharedArray(u[1]), SharedArray(u[2]))
    
    out_dist, counts_dist = SFC.parallel_calculate_structure_function(sx, su, bins, sf_type, verbose=false, show_progress=false, return_sums_and_counts=true)

    @testset "Serial vs Distributed" begin
        @test out_serial[1] ≈ out_dist[1]
        @test out_serial[2] ≈ out_dist[2]
        @test counts_serial == counts_dist
    end
end
