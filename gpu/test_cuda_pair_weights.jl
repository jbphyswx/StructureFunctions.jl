using Test: Test
using CUDA: CUDA
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.StructureFunctionObjects: StructureFunctionObjects as SFO
using StructureFunctions.MultiFields: MultiFields as MF
using StructureFunctions.HelperFunctions: HelperFunctions as SFH
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const OP = SFT.S2SFType()
const RAW = SFO.StructureFunctionSumsAndCounts
const RAW2 = SFO.StructureFunction2DSumsAndCounts
const NP = 1000
const NT = 4
const NI = SFC.SINGLE_PASS_N
const BINS = collect(range(0.0, 1.0; length = 9)) .+ 0.0137
const VB = collect(range(0.0, 2.0; length = 5)) .+ 0.011
const NB, NV = length(BINS) - 1, length(VB) - 1

function compare(got_s, got_c, ref_s, ref_c; rtol = 1e-9)
    gs, gc = Array(got_s), Array(got_c)
    rs, rc = Array(ref_s), Array(ref_c)
    Test.@test maximum(abs.(gs .- rs)) / (maximum(abs, rs) + eps()) < rtol
    Test.@test maximum(abs.(gc .- rc)) / (maximum(abs, rc) + eps()) < rtol
end

"""Run the weighted `route` at coordinate width `D` on the device and serially, and compare."""
function check_route(route::Symbol, D::Int, w)
    x, u = rand(D, NP), rand(D, NP)
    xb, ub = rand(D, NP, NT), rand(D, NP, NT)
    geom = SFH.FlatGeometry{D}()
    if route === :point
        r = SFC.calculate_structure_function(OP, x, u, BINS, Float64, RAW; backend = SER, weights = w)
        g = SFC.calculate_structure_function(OP, x, u, BINS, Float64, RAW; backend = DEV, weights = w)
        compare(g.sums, g.counts, r.sums, r.counts)
    elseif route === :joint
        r = SFC.calculate_structure_function(OP, x, u, BINS, VB, Float64, RAW2; backend = SER, weights = w)
        g = SFC.calculate_structure_function(OP, x, u, BINS, VB, Float64, RAW2; backend = DEV, weights = w)
        compare(g.sums, g.counts, r.sums, r.counts)
    elseif route === :single_pass
        r = SFC._dispatch_single_pass(SER, SFC.PointField{D}(), x, u, BINS, Float64; weights = w, geometry = geom)
        g = SFC._dispatch_single_pass(DEV, SFC.PointField{D}(), x, u, BINS, Float64; weights = w, geometry = geom)
        compare(g.sums, g.counts, r.sums, r.counts)
    elseif route === :batch_fixed
        r = SFC.calculate_structure_function(OP, x, ub, BINS, Float64, RAW; backend = SER, weights = w)
        g = SFC.calculate_structure_function(OP, x, ub, BINS, Float64, RAW; backend = DEV, weights = w)
        compare(g.sums, g.counts, r.sums, r.counts)
    elseif route === :batch_varying
        r = SFC.calculate_structure_function(OP, xb, ub, BINS, Float64, RAW; backend = SER, weights = w)
        g = SFC.calculate_structure_function(OP, xb, ub, BINS, Float64, RAW; backend = DEV, weights = w)
        compare(g.sums, g.counts, r.sums, r.counts)
    elseif route === :joint_batch_fixed
        r = SFC.calculate_structure_function(OP, x, ub, BINS, VB, Float64, RAW2; backend = SER, weights = w)
        g = SFC.calculate_structure_function(OP, x, ub, BINS, VB, Float64, RAW2; backend = DEV, weights = w)
        compare(g.sums, g.counts, r.sums, r.counts)
    elseif route === :joint_batch_varying
        r = SFC.calculate_structure_function(OP, xb, ub, BINS, VB, Float64, RAW2; backend = SER, weights = w)
        g = SFC.calculate_structure_function(OP, xb, ub, BINS, VB, Float64, RAW2; backend = DEV, weights = w)
        compare(g.sums, g.counts, r.sums, r.counts)
    elseif route === :single_pass_2d
        rs, rc = zeros(NI, NB, NV), zeros(NI, NB, NV)
        SFC.calculate_structure_functions_single_pass_2d!(rs, rc, x, u, BINS, VB; backend = SER, weights = w)
        gs, gc = CUDA.zeros(Float64, NI, NB, NV), CUDA.zeros(Float64, NI, NB, NV)
        SFC.calculate_structure_functions_single_pass_2d!(gs, gc, x, u, BINS, VB; backend = DEV, weights = w)
        compare(gs, gc, rs, rc)
    elseif route === :single_pass_batch_fixed
        rs, rc = zeros(NI, NB, NT), zeros(NI, NB, NT)
        SFC.calculate_structure_functions_single_pass_batch!(rs, rc, x, ub, BINS; backend = SER, weights = w)
        gs, gc = CUDA.zeros(Float64, NI, NB, NT), CUDA.zeros(Float64, NI, NB, NT)
        SFC.calculate_structure_functions_single_pass_batch!(gs, gc, x, ub, BINS; backend = DEV, weights = w)
        compare(gs, gc, rs, rc)
    elseif route === :single_pass_2d_batch_varying
        rs, rc = zeros(NI, NB, NV, NT), zeros(NI, NB, NV, NT)
        SFC.calculate_structure_functions_single_pass_2d_batch!(rs, rc, xb, ub, BINS, VB; backend = SER, weights = w)
        gs, gc = CUDA.zeros(Float64, NI, NB, NV, NT), CUDA.zeros(Float64, NI, NB, NV, NT)
        SFC.calculate_structure_functions_single_pass_2d_batch!(gs, gc, xb, ub, BINS, VB; backend = DEV, weights = w)
        compare(gs, gc, rs, rc)
    else
        error("not implemented: route $route")
    end
    return nothing
end

# (coordinate width, weighted route): every route and every width
const CASES = (
    (2, :point), (2, :single_pass_2d),
    (3, :joint), (3, :batch_varying),
    (5, :single_pass), (5, :joint_batch_varying),
    (6, :batch_fixed), (6, :single_pass_batch_fixed),
    (7, :joint_batch_fixed), (7, :single_pass_2d_batch_varying),
)

# (coordinate width, tensor order)
const TENSOR_CASES = ((3, 2), (2, 3), (3, 4), (2, 5))

Test.@testset "pair weights on the device" begin
    Random.seed!(20260918)
    w = 0.3 .+ rand(NP)

    # Every weighted route compiles at its width and gives the serial weighted answer.
    Test.@testset "$route D=$D" for (D, route) in CASES
        check_route(route, D, w)
    end

    # Float32 input against the serial Float64 answer, within a random walk of roundings over a bin's pairs.
    Test.@testset "Float32" begin
        x, u, w32 = rand(Float32, 2, NP), rand(Float32, 2, NP), 0.3f0 .+ rand(Float32, NP)
        b32, v32 = Float32.(BINS), Float32.(VB)
        x64, u64, w64 = Float64.(x), Float64.(u), Float64.(w32)
        Test.@testset "point" begin
            r = SFC.calculate_structure_function(OP, x64, u64, BINS, Float64, RAW; backend = SER, weights = w64)
            g = SFC.calculate_structure_function(OP, x, u, b32, Float64, RAW; backend = DEV, weights = w32)
            compare(g.sums, g.counts, r.sums, r.counts; rtol = 10 * sqrt(maximum(r.counts)) * eps(Float32))
        end
        Test.@testset "joint" begin
            r = SFC.calculate_structure_function(OP, x64, u64, BINS, VB, Float64, RAW2; backend = SER, weights = w64)
            g = SFC.calculate_structure_function(OP, x, u, b32, v32, Float64, RAW2; backend = DEV, weights = w32)
            compare(g.sums, g.counts, r.sums, r.counts; rtol = 10 * sqrt(maximum(r.counts)) * eps(Float32))
        end
    end

    Test.@testset "multi-field" begin
        x = rand(2, NP)
        fields = MF.Fields(vectors = (rand(2, NP),), scalars = (rand(NP),))
        mixed = SFT.MixedSFType{1, 0, 2}()
        rs, rc = zeros(Float64, NB), zeros(Float64, NB)
        SFC.serial_calculate_structure_function!(rs, rc, mixed, x, fields, BINS; weights = w,
                                                 geometry = SFH.FlatGeometry{2}())
        gs, gc = CUDA.zeros(Float64, NB), CUDA.zeros(Float64, NB)
        SFC.calculate_structure_function!(gs, gc, mixed, x, fields, BINS; backend = DEV, weights = w)
        compare(gs, gc, rs, rc)
    end

    # The moment tensor through the tiled families, at every order and with varying positions.
    Test.@testset "moment tensor D=$D P=$P" for (D, P) in TENSOR_CASES
        x, u = rand(D, NP), rand(D, NP)
        rs, rc = zeros(Float64, ntuple(_ -> D, P)..., NB), zeros(Float64, NB)
        SFC.calculate_structure_function_tensor!(rs, rc, Val(P), x, u, BINS; backend = SER, weights = w)
        gs, gc = CUDA.zeros(Float64, ntuple(_ -> D, P)..., NB), CUDA.zeros(Float64, NB)
        SFC.calculate_structure_function_tensor!(gs, gc, Val(P), x, u, BINS; backend = DEV, weights = w)
        compare(gs, gc, rs, rc)
    end
    Test.@testset "moment tensor varying positions" begin
        D, P = 2, 2
        x, u = rand(D, NP, 3), rand(D, NP, 3)
        rs, rc = zeros(Float64, D, D, NB, 3), zeros(Float64, NB, 3)
        SFC.calculate_structure_function_tensor!(rs, rc, Val(P), x, u, BINS; backend = SER, weights = w)
        gs, gc = CUDA.zeros(Float64, D, D, NB, 3), CUDA.zeros(Float64, NB, 3)
        SFC.calculate_structure_function_tensor!(gs, gc, Val(P), x, u, BINS; backend = DEV, weights = w)
        compare(gs, gc, rs, rc)
    end

    # A constant weight k scales every sum and count by exactly k².
    Test.@testset "moment tensor k^2 scaling" begin
        D, P, k = 2, 2, 3.0
        x, u = rand(D, NP), rand(D, NP)
        us, uc = CUDA.zeros(Float64, D, D, NB), CUDA.zeros(Float64, NB)
        SFC.calculate_structure_function_tensor!(us, uc, Val(P), x, u, BINS; backend = DEV)
        ks, kc = CUDA.zeros(Float64, D, D, NB), CUDA.zeros(Float64, NB)
        SFC.calculate_structure_function_tensor!(ks, kc, Val(P), x, u, BINS; backend = DEV, weights = fill(k, NP))
        compare(ks, kc, k^2 .* us, k^2 .* uc)
    end

    # The unweighted device answer.
    Test.@testset "unweighted D=$D" for D in (2, 3)
        x, u = rand(D, NP), rand(D, NP)
        r = SFC.calculate_structure_function(OP, x, u, BINS, UInt32, RAW; backend = SER)
        g = SFC.calculate_structure_function(OP, x, u, BINS, UInt32, RAW; backend = DEV)
        compare(g.sums, g.counts, r.sums, r.counts)
    end
end
