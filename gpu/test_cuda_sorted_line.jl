using Test: Test
using CUDA: CUDA
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.MultiFields: MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const RAW = SF.StructureFunctionSumsAndCounts
const N = 5000
const BINS = collect(range(0.0, 2.0; length = 33))
const NB = length(BINS) - 1

function compare(gs, gc, rs, rc; rtol = 1e-10)
    Test.@test maximum(abs.(Array(gs) .- rs)) / (maximum(abs, rs) + eps()) < rtol
    Test.@test maximum(abs.(float.(Array(gc)) .- float.(rc))) / (maximum(abs, float.(rc)) + eps()) < rtol
end

Test.@testset "sorted line on the device" begin
    Random.seed!(20260926)
    x = reshape(rand(N) .* 100.0, 1, :)
    u = randn(1, N)
    θ = randn(N)
    w = 0.5 .+ rand(N)

    # A polynomial operator over points on a line, unweighted from device inputs and weighted from host inputs.
    Test.@testset "$name" for (name, sf) in (("S2", SFT.S2SFType()), ("L3", SFT.L3SFType()),
                                            ("L4", SFT.ProjectedStructureFunctionType{4, 0}()))
        Test.@testset "unweighted" begin
            ref = SFC.calculate_structure_function(sf, x, u, BINS, RAW; backend = SER)
            got = SFC.calculate_structure_function(sf, CUDA.CuArray(x), CUDA.CuArray(u), BINS, RAW; backend = DEV)
            compare(got.sums, got.counts, ref.sums, ref.counts)
        end
        Test.@testset "weighted" begin
            ref = SFC.calculate_structure_function(sf, x, u, BINS, Float64, RAW; backend = SER, weights = w)
            got = SFC.calculate_structure_function(sf, x, u, BINS, Float64, RAW; backend = DEV, weights = w)
            compare(got.sums, got.counts, ref.sums, ref.counts)
        end
    end

    Test.@testset "multi-field $name" for (name, sf) in (("Mixed{1,0,2}", SFT.MixedSFType{1, 0, 2}()),
                                                         ("Scalar{3}", SFT.ScalarSFType{3}()))
        f = MF.Fields(vectors = (u,), scalars = (θ,))
        ref = SFC.calculate_structure_function(sf, x, f, BINS, RAW; backend = SER)
        got = SFC.calculate_structure_function(sf, x, f, BINS, RAW; backend = DEV)
        compare(got.sums, got.counts, ref.sums, ref.counts)
    end

    # Float32 values on an offset far above their increments, added twice, against the Float64 answer on them.
    Test.@testset "Float32 on an offset, added twice" begin
        x32, u32, b32 = Float32.(x), Float32.(1000 .+ 0.01 .* u), Float32.(BINS)
        s, c = CUDA.zeros(Float32, NB), CUDA.zeros(UInt32, NB)
        for _ in 1:2
            SFC.calculate_structure_function!(s, c, SFT.L3SFType(), CUDA.CuArray(x32), CUDA.CuArray(u32), b32;
                                              backend = DEV)
        end
        ref = SFC.calculate_structure_function(SFT.L3SFType(), Float64.(x32), Float64.(u32), Float64.(b32), RAW;
                                               backend = SER)
        compare(s, c, 2 .* ref.sums, 2 .* ref.counts; rtol = 1e-3)
    end
end
