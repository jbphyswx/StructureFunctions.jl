using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO, MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using StaticArrays: StaticArrays as SA
using FFTW: FFTW
using AbstractFFTs: AbstractFFTs
using Random: Random

Random.seed!(77)

const AL_OP = SFT.L2SFType()
const AL_SER = CB.SerialBackend()
const AL_NB, AL_NV = 8, 5
const AL_BINS = collect(range(0.0, 1.0; length = AL_NB + 1))
const AL_VBINS = collect(range(-3.0, 3.0; length = AL_NV + 1))
const AL_ABINS = collect(range(prevfloat(0.0), π; length = 4))
const AL_AX = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
const AL_RAW = SF.StructureFunctionSumsAndCounts
const AL_TRAW = SFO.StructureFunctionTensorSumsAndCounts
const AL_FFT = SB.FastFourierTransformSpectralBackend()

# Below the fourfold growth of a per-pair allocation when the points double.
const AL_MAX_GROWTH = 3.0

"""Bytes a compiled call allocates, the least of a few runs."""
function alloc_bytes(f, reps::Int = 3)
    f()
    return minimum(@allocated(f()) for _ in 1:reps)
end

"""Positions and velocities of `n` points in `d` dimensions, from one seed so two sizes are comparable."""
function al_points(n::Int, d::Int = 2)
    Random.seed!(1234)
    return rand(d, n), randn(d, n)
end

const AL_POINT_ROUTES = (
    ("point 1D", (n -> begin
        x, u = al_points(n)
        () -> SFC.calculate_structure_function(AL_OP, x, u, AL_BINS, AL_RAW; backend = AL_SER)
    end)),
    ("point joint value", (n -> begin
        x, u = al_points(n)
        () -> SFC.calculate_structure_function(AL_OP, x, u, AL_BINS, AL_VBINS; backend = AL_SER)
    end)),
    ("point joint angle", (n -> begin
        x, u = al_points(n)
        () -> SFC.calculate_structure_function(AL_OP, x, u, AL_BINS, AL_ABINS; backend = AL_SER,
            second_axis = AL_AX)
    end)),
    ("point multi-field", (n -> begin
        x, u = al_points(n)
        f = MF.Fields(vectors = (u,))
        () -> SFC.calculate_structure_function(AL_OP, x, f, AL_BINS, AL_RAW; backend = AL_SER)
    end)),
    ("moment tensor", (n -> begin
        x, u = al_points(n)
        () -> SFC.calculate_structure_function_tensor(Val(2), x, u, AL_BINS, AL_TRAW; backend = AL_SER)
    end)),
    ("single-pass 1D", (n -> begin
        x, u = al_points(n)
        s = zeros(SFC.SINGLE_PASS_N, AL_NB)
        c = zeros(Int, SFC.SINGLE_PASS_N, AL_NB)
        () -> SFC.calculate_structure_functions_single_pass!(s, c, x, u, AL_BINS; backend = AL_SER)
    end)),
    ("single-pass 2D", (n -> begin
        x, u = al_points(n)
        s = zeros(SFC.SINGLE_PASS_N, AL_NB, AL_NV)
        c = zeros(Int, SFC.SINGLE_PASS_N, AL_NB, AL_NV)
        () -> SFC.calculate_structure_functions_single_pass_2d!(s, c, x, u, AL_BINS, AL_VBINS;
            backend = AL_SER)
    end)),
    ("slice batch 1D", (n -> begin
        Random.seed!(1234)
        x, u = rand(2, n, 3), randn(2, n, 3)
        s, c = zeros(AL_NB, 3), zeros(Int, AL_NB, 3)
        () -> SFC.calculate_structure_function_batch!(s, c, AL_OP, x, u, AL_BINS; backend = AL_SER)
    end)),
)

# Each route's allocation grows less than its pair count when the points double.
Test.@testset "allocation follows the inputs, not the pair count" begin
    for (name, build) in AL_POINT_ROUTES
        Test.@testset "$name" begin
            Test.@test alloc_bytes(build(400)) <= AL_MAX_GROWTH * alloc_bytes(build(200))
        end
    end
end

# A grid's allocation grows at most in proportion to its cells when its sides double.
Test.@testset "a gridded route's allocation follows its grid, not its lag count" begin
    edges = collect(range(0.0, 0.2; length = AL_NB + 1))
    for (name, run) in (
        ("gridded lag sweep", ((u, s) -> begin
            a, c = zeros(AL_NB), zeros(Int, AL_NB)
            () -> SFC.gridded_lag_sweep!(a, c, AL_OP, u, s, edges, Val(2), Val(1), Val(0); backend = AL_SER)
        end)),
        ("gridded transform", ((u, s) -> begin
            a, c = zeros(AL_NB), zeros(Int, AL_NB)
            () -> SFC.gridded_sweep!(a, c, AL_OP, u, s, edges, Val(2), Val(1), Val(0), AL_FFT; backend = AL_SER)
        end)),
    )
        Test.@testset "$name" begin
            function bytes(dims)
                u = reshape(Float64[sin(d + 0.3i + 0.7j) for d in 1:2, i in 1:dims[1], j in 1:dims[2]], 2, :)
                s = SFC.UniformLagSchedule(dims, (1 / dims[1], 1 / dims[2]), (true, true))
                return alloc_bytes(run(u, s))
            end
            Test.@test bytes((32, 32)) <= 1.5 * 4 * bytes((16, 16))
        end
    end
end
