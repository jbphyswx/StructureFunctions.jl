using Test: Test
using CUDA: CUDA
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.StructureFunctionObjects: StructureFunctionObjects as SFO
using StructureFunctions.HelperFunctions: HelperFunctions as SFH
using ComputationalBackends: ComputationalBackends as CB
using Distances: Distances as DI

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const OP = SFT.L2SFType()
const RAW = SFO.StructureFunctionSumsAndCounts
const N, T, NB = 500, 3, 8

function compare(gs, gc, rs, rc; rtol = 1e-9)
    gs, gc = Array(gs), Array(gc)
    Test.@test maximum(abs.(gs .- rs)) / (maximum(abs, rs) + eps()) < rtol
    Test.@test maximum(abs.(float.(gc) .- float.(rc))) / (maximum(abs, float.(rc)) + eps()) < rtol
end

"""Serial sums and counts `(nb, T)` of each slice of `u` over the shared positions `x`."""
function serial_slices(x, u, bins)
    nb = length(bins) - 1
    rs, rc = zeros(nb, size(u, 3)), zeros(Int, nb, size(u, 3))
    for t in axes(u, 3)
        r = SFC.calculate_structure_function(OP, x, u[:, :, t], bins, Float64, RAW; backend = SER)
        rs[:, t] .= r.sums
        rc[:, t] .= r.counts
    end
    return rs, rc
end

# (coordinate width, distance-bin type)
const SHARED_CASES = ((2, :linear), (3, :vector))

Test.@testset "batch routing across widths" begin
    Random.seed!(20260918)

    # A batch over shared positions at the coordinate width and the bin type the device route reads.
    Test.@testset "shared-position batch D=$D $btype" for (D, btype) in SHARED_CASES
        x = rand(D, N)
        u = randn(D, N, T)
        edges = range(0.0, 1.0; length = NB + 1)
        bins = btype === :linear ? SF.LinearBinEdges(edges) : collect(edges)
        rs, rc = serial_slices(x, u, collect(edges))
        g = SFC.gpu_calculate_structure_function_batch(OP, CUDA.CUDABackend(), x, u, bins, UInt32;
                                                       geometry = SFH.FlatGeometry{D}())
        compare(reshape(collect(g.sums), NB, T), reshape(collect(g.counts), NB, T), rs, rc)
    end

    # A spherical batch: coordinates two wide, the field `F` wide.
    Test.@testset "spherical batch field=$F" for F in (2, 3)
        m = DI.Haversine(6.371e6)
        x = permutedims(hcat(300 .* rand(N) .- 150, 100 .* rand(N) .- 50))
        db = collect(range(0.0, 9.0e6; length = NB + 1))
        u = randn(F, N, T)
        r = SFC.calculate_structure_function(OP, x, u, db, RAW; backend = SER, distance_metric = m)
        g = SFC.calculate_structure_function(OP, x, u, db, RAW; backend = DEV, distance_metric = m)
        compare(g.sums, g.counts, r.sums, r.counts)
    end

    # A histogram at the native kernels' bin cap and past it, through the point and the batch entries.
    Test.@testset "point histogram nb=$nb" for nb in (128, 129, 4000)
        x, u = rand(2, N), randn(2, N)
        bins = collect(range(0.0, 1.0; length = nb + 1))
        r = SFC.calculate_structure_function(OP, x, u, bins, Float64, RAW; backend = SER)
        g = SFC.calculate_structure_function(OP, x, u, bins, Float64, RAW; backend = DEV)
        compare(g.sums, g.counts, r.sums, r.counts)
    end
    Test.@testset "shared-position batch histogram nb=$nb" for nb in (128, 4000)
        x, u = rand(2, N), randn(2, N, T)
        bins = collect(range(0.0, 1.0; length = nb + 1))
        rs, rc = serial_slices(x, u, bins)
        g = SFC.gpu_calculate_structure_function_batch(OP, CUDA.CUDABackend(), x, u, bins, UInt32;
                                                       geometry = SFH.FlatGeometry{2}())
        compare(reshape(collect(g.sums), nb, T), reshape(collect(g.counts), nb, T), rs, rc)
    end
end
