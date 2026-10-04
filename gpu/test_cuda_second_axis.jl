using Test: Test
using CUDA: CUDA
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const OP = SFT.L2SFType()
const NP = 1000
const DBINS = collect(range(0.0, 1.0; length = 9)) .+ 0.0137
const ABINS = collect(range(prevfloat(0.0), π; length = 5))
const SRC = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))

function compare(got, ref; rtol = 1e-9)
    gs, gc = Array(got.sums), Array(got.counts)
    rs, rc = Array(ref.sums), Array(ref.counts)
    Test.@test maximum(abs.(gs .- rs)) / (maximum(abs, rs) + eps()) < rtol
    Test.@test maximum(abs.(float.(gc) .- float.(rc))) / (maximum(abs, float.(rc)) + eps()) < rtol
end

Test.@testset "separation-angle second axis" begin
    Random.seed!(20260918)

    # The joint histogram over the separation angle at both coordinate widths.
    Test.@testset "angle axis D=$D" for D in (2, 3)
        x, u = rand(D, NP), rand(D, NP)
        src = SFC.SeparationAngleAxis(D == 2 ? SA.SVector(1.0, 0.0) : SA.SVector(1.0, 0.0, 0.0))
        ref = SFC.calculate_structure_function(OP, x, u, DBINS, ABINS; backend = SER, second_axis = src)
        got = SFC.calculate_structure_function(OP, x, u, DBINS, ABINS; backend = DEV, second_axis = src)
        compare(got, ref)
    end

    # The value axis through the same kernels.
    Test.@testset "value axis" begin
        vbins = collect(range(0.0, 2.0; length = 5)) .+ 0.011
        x, u = rand(2, NP), rand(2, NP)
        ref = SFC.calculate_structure_function(OP, x, u, DBINS, vbins; backend = SER)
        got = SFC.calculate_structure_function(OP, x, u, DBINS, vbins; backend = DEV)
        compare(got, ref)
    end

    # The slice batch bins the angle too: shared positions once per pair, varying positions per slice.
    Test.@testset "angle axis slice batch shared=$shared" for shared in (true, false)
        x = shared ? rand(2, NP ÷ 4) : rand(2, NP ÷ 4, 3)
        u = rand(2, NP ÷ 4, 3)
        ref = SFC.calculate_structure_function(OP, x, u, DBINS, ABINS; backend = SER, second_axis = SRC)
        got = SFC.calculate_structure_function(OP, x, u, DBINS, ABINS; backend = DEV, second_axis = SRC)
        compare(got, ref)
    end

    # The angle is read off `X2 - X1`, which is the separation only on a flat metric.
    Test.@testset "curved metric refused" begin
        x, u = rand(2, 64), rand(2, 64)
        Test.@test_throws ArgumentError SFC.calculate_structure_function(OP, x, u, DBINS, ABINS; backend = DEV,
                                                                         second_axis = SRC,
                                                                         distance_metric = SFC.DI.SphericalAngle())
    end
end
