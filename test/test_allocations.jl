using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO, MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using StaticArrays: StaticArrays as SA
using FFTW: FFTW
using AbstractFFTs: AbstractFFTs
using Random: Random

# Allocation must follow the inputs and the output, never the pair count.
#
# Doubling the point count **quadruples** the pairs while at most doubling the staging a route
# needs, so a route that allocates per pair — a boxed accumulator in the inner loop, a per-lag
# temporary, a closure capture that escapes — shows up as a ratio near four, and one that only
# stages its inputs shows a ratio near two. That is the defect class this file exists for, and it
# is a property of the code rather than of the machine or the Julia version, so it needs no
# recorded byte count to compare against.

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

"""
Allowed growth when the point count doubles. Two would be exact for a route that only stages its
inputs; the margin covers the bin-independent scratch a route may size from the cull grid. Four is
what allocating per pair costs, so the bound separates the two cases with room to spare.
"""
const AL_MAX_GROWTH = 3.0

"""Bytes a compiled route allocates; the minimum over a few runs drops GC noise."""
function alloc_bytes(f, reps::Int = 3)
    f()
    return minimum(@allocated(f()) for _ in 1:reps)
end

"""Point data of `n` points in `d` dimensions, from one seed so the two sizes are comparable."""
function al_points(n::Int, d::Int = 2)
    Random.seed!(1234)
    return rand(d, n), randn(d, n)
end

# Each route is built from the point count so the same closure serves both sizes.
const AL_POINT_ROUTES = (
    ("point 1D", (n -> begin
        x, u = al_points(n)
        () -> SFC.calculate_structure_function(AL_OP, x, u, AL_BINS, AL_RAW; backend = AL_SER,
            verbose = false)
    end)),
    ("point joint value", (n -> begin
        x, u = al_points(n)
        () -> SFC.calculate_structure_function(AL_OP, x, u, AL_BINS, AL_VBINS; backend = AL_SER,
            verbose = false)
    end)),
    ("point joint angle", (n -> begin
        x, u = al_points(n)
        () -> SFC.calculate_structure_function(AL_OP, x, u, AL_BINS, AL_ABINS; backend = AL_SER,
            verbose = false, second_axis = AL_AX)
    end)),
    ("point multi-field", (n -> begin
        x, u = al_points(n)
        f = MF.Fields(vectors = (u,))
        () -> SFC.calculate_structure_function(AL_OP, x, f, AL_BINS, AL_RAW; backend = AL_SER,
            verbose = false)
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

Test.@testset "allocation follows the inputs, not the pair count" begin
    for (name, build) in AL_POINT_ROUTES
        Test.@testset "$name" begin
            small = alloc_bytes(build(200))
            large = alloc_bytes(build(400))
            # A route that allocated nothing beyond its result would make the ratio meaningless.
            Test.@test small > 0
            growth = large / small
            Test.@test (name, growth <= AL_MAX_GROWTH) == (name, true)
        end
    end
end

Test.@testset "a gridded route's allocation follows its grid, not its lag count" begin
    # Doubling each side quadruples the cells and the transform's buffers with them, while the
    # lags inside the same radius grow with the cells too — so the honest bound here is on the
    # cells, and a per-lag temporary would push it past that.
    dims_small, dims_large = (32, 32), (64, 64)
    edges = collect(range(0.0, 0.2; length = AL_NB + 1))
    for (name, run) in (
        ("gridded lag sweep", ((dims, u, s) -> begin
            a, c = zeros(AL_NB), zeros(Int, AL_NB)
            () -> SFC.gridded_lag_sweep!(a, c, AL_OP, u, s, edges, Val(2), Val(1), Val(0);
                backend = AL_SER)
        end)),
        ("gridded transform", ((dims, u, s) -> begin
            a, c = zeros(AL_NB), zeros(Int, AL_NB)
            () -> SFC.gridded_sweep!(a, c, AL_OP, u, s, edges, Val(2), Val(1), Val(0), AL_FFT;
                backend = AL_SER)
        end)),
    )
        Test.@testset "$name" begin
            function bytes(dims)
                u = reshape(Float64[sin(d + 0.3i + 0.7j) for d in 1:2, i in 1:dims[1], j in 1:dims[2]],
                            2, :)
                s = SFC.UniformLagSchedule(dims, (1 / dims[1], 1 / dims[2]), (true, true))
                return alloc_bytes(run(dims, u, s))
            end
            small = bytes(dims_small)
            large = bytes(dims_large)
            cells = prod(dims_large) / prod(dims_small)
            Test.@test small > 0
            # Four times the cells may cost four times the buffers; per-lag allocation costs more.
            Test.@test (name, large / small <= 1.5 * cells) == (name, true)
        end
    end
end
