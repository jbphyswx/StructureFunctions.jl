using Test: Test
using CUDA: CUDA
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO
using StaticArrays: StaticArrays as SA
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const OP = SFT.L2SFType()
const RAW = SFO.StructureFunctionSumsAndCounts
const N, T, NB, NV = 400, 3, 8, 5
const BINS = collect(range(0.0, 1.0; length = NB + 1))
const VBINS = collect(range(-3.0, 3.0; length = NV + 1))
const ABINS = collect(range(prevfloat(0.0), Float64(π); length = 5))
const NI = SFC.SINGLE_PASS_N

function compare(gs, gc, rs, rc; rtol = 1e-9)
    gs, gc = Array(gs), Array(gc)
    Test.@test maximum(abs.(gs .- rs)) / (maximum(abs, rs) + eps()) < rtol
    Test.@test maximum(abs.(float.(gc) .- float.(rc))) / (maximum(abs, float.(rc)) + eps()) < rtol
end

"""Run `route` at coordinate width `D` on the device and serially, and compare."""
function check_route(route::Symbol, D::Int)
    x, u = rand(D, N), randn(D, N)
    xb, ub = rand(D, N, T), randn(D, N, T)
    if route === :point
        r = SFC.calculate_structure_function(OP, x, u, BINS, Float64, RAW; backend = SER)
        g = SFC.calculate_structure_function(OP, x, u, BINS, Float64, RAW; backend = DEV)
        compare(g.sums, g.counts, r.sums, r.counts)
    elseif route === :joint_value
        r = SFC.calculate_structure_function(OP, x, u, BINS, VBINS; backend = SER)
        g = SFC.calculate_structure_function(OP, x, u, BINS, VBINS; backend = DEV)
        compare(g.sums, g.counts, r.sums, r.counts)
    elseif route === :joint_angle
        ax = SFC.SeparationAngleAxis(SA.SVector(ntuple(d -> d == 1 ? 1.0 : 0.0, D)))
        r = SFC.calculate_structure_function(OP, x, u, BINS, ABINS; backend = SER, second_axis = ax)
        g = SFC.calculate_structure_function(OP, x, u, BINS, ABINS; backend = DEV, second_axis = ax)
        compare(g.sums, g.counts, r.sums, r.counts)
        rv = SFC.calculate_structure_function(OP, x, u, BINS, ABINS; backend = SER)
        Test.@test collect(r.counts) != collect(rv.counts)
    elseif route === :single_pass
        rs, rc = zeros(NI, NB), zeros(Int, NI, NB)
        SFC.calculate_structure_functions_single_pass!(rs, rc, x, u, BINS; backend = SER)
        gs, gc = CUDA.zeros(Float64, NI, NB), CUDA.zeros(Int, NI, NB)
        SFC.calculate_structure_functions_single_pass!(gs, gc, x, u, BINS; backend = DEV)
        compare(gs, gc, rs, rc)
    elseif route === :single_pass_2d
        rs, rc = zeros(NI, NB, NV), zeros(Int, NI, NB, NV)
        SFC.calculate_structure_functions_single_pass_2d!(rs, rc, x, u, BINS, VBINS; backend = SER)
        gs, gc = CUDA.zeros(Float64, NI, NB, NV), CUDA.zeros(Int, NI, NB, NV)
        SFC.calculate_structure_functions_single_pass_2d!(gs, gc, x, u, BINS, VBINS; backend = DEV)
        compare(gs, gc, rs, rc)
    elseif route === :batch
        rs, rc = zeros(NB, T), zeros(Int, NB, T)
        SFC.calculate_structure_function_batch!(rs, rc, OP, xb, ub, BINS; backend = SER)
        gs, gc = CUDA.zeros(Float64, NB, T), CUDA.zeros(Int, NB, T)
        SFC.calculate_structure_function_batch!(gs, gc, OP, xb, ub, BINS; backend = DEV)
        compare(gs, gc, rs, rc)
    elseif route === :batch_joint
        rs, rc = zeros(NB, NV, T), zeros(Int, NB, NV, T)
        SFC.calculate_structure_function_2d_batch!(rs, rc, OP, xb, ub, BINS, VBINS; backend = SER)
        gs, gc = CUDA.zeros(Float64, NB, NV, T), CUDA.zeros(Int, NB, NV, T)
        SFC.calculate_structure_function_2d_batch!(gs, gc, OP, xb, ub, BINS, VBINS; backend = DEV)
        compare(gs, gc, rs, rc)
    elseif route === :batch_single_pass
        rs, rc = zeros(NI, NB, T), zeros(Int, NI, NB, T)
        SFC.calculate_structure_functions_single_pass_batch!(rs, rc, xb, ub, BINS; backend = SER)
        gs, gc = CUDA.zeros(Float64, NI, NB, T), CUDA.zeros(Int, NI, NB, T)
        SFC.calculate_structure_functions_single_pass_batch!(gs, gc, xb, ub, BINS; backend = DEV)
        compare(gs, gc, rs, rc)
    elseif route === :batch_single_pass_2d
        rs, rc = zeros(NI, NB, NV, T), zeros(Int, NI, NB, NV, T)
        SFC.calculate_structure_functions_single_pass_2d_batch!(rs, rc, xb, ub, BINS, VBINS; backend = SER)
        gs, gc = CUDA.zeros(Float64, NI, NB, NV, T), CUDA.zeros(Int, NI, NB, NV, T)
        SFC.calculate_structure_functions_single_pass_2d_batch!(gs, gc, xb, ub, BINS, VBINS; backend = DEV)
        compare(gs, gc, rs, rc)
    else
        error("not implemented: route $route")
    end
    return nothing
end

# (coordinate width, route): every route at a width other than 2 and 3, and every width
const CASES = (
    (2, :point), (3, :joint_value),
    (4, :point), (5, :joint_value), (6, :joint_angle), (7, :single_pass), (4, :single_pass_2d),
    (5, :batch), (6, :batch_joint), (7, :batch_single_pass), (4, :batch_single_pass_2d),
)

Test.@testset "coordinate widths" begin
    # Each route compiles at the coordinate width it is launched at and gives the serial answer.
    Test.@testset "$route D=$D" for (D, route) in CASES
        Random.seed!(4400 + D)
        check_route(route, D)
    end

    # Above every tiled kernel's staging budget the 1-D and 2-D batches take the wide kernels.
    Test.@testset "wide kernels D=64" begin
        D = 64
        xb, ub = rand(D, 32, 2), randn(D, 32, 2)
        Test.@testset "slice batch" begin
            s, c = CUDA.zeros(Float64, NB, 2), CUDA.zeros(Int, NB, 2)
            SFC.calculate_structure_function_batch!(s, c, OP, xb, ub, BINS; backend = DEV)
            rs, rc = zeros(NB, 2), zeros(Int, NB, 2)
            SFC.calculate_structure_function_batch!(rs, rc, OP, xb, ub, BINS; backend = SER)
            compare(s, c, rs, rc)
        end
        Test.@testset "slice batch joint" begin
            js, jc = CUDA.zeros(Float64, NB, NV, 2), CUDA.zeros(Int, NB, NV, 2)
            SFC.calculate_structure_function_2d_batch!(js, jc, OP, xb, ub, BINS, VBINS; backend = DEV)
            rjs, rjc = zeros(NB, NV, 2), zeros(Int, NB, NV, 2)
            SFC.calculate_structure_function_2d_batch!(rjs, rjc, OP, xb, ub, BINS, VBINS; backend = SER)
            compare(js, jc, rjs, rjc)
        end
    end
end
