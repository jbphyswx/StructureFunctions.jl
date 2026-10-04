using Test: Test
using CUDA: CUDA
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.MultiFields: MultiFields as MF
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB

const BE = CUDA.CUDABackend()
const DEV = CB.GPUBackend(BE)
const SER = CB.SerialBackend()
const OP = SFT.L2SFType()
const POLICIES = (SFC.AutoCulling(), SFC.AlwaysCulling())
const N, B = 3000, 4
const VAL = collect(range(-4.0, 4.0; length = 9))
const VB = ntuple(_ -> VAL, SFC.SINGLE_PASS_N)
const NV, NI = length(VAL) - 1, SFC.SINGLE_PASS_N

"""The workspace's cull memo is engaged and the device histogram equals the serial one."""
function check(got_s, got_c, ref_s, ref_c, ws)
    gs, gc = Array(got_s), Array(got_c)
    Test.@test ws.lazy.cull isa SFC.GPUCullMemo
    Test.@test sum(ref_c) > 0
    Test.@test eltype(ref_c) <: Integer ? gc == ref_c : isapprox(gc, ref_c; rtol = 1e-10)
    Test.@test isapprox(gs, ref_s; rtol = 1e-9, atol = 1e-12)
end

"""Shape, workspace constructor and batch entry `run!(sums, counts, backend, workspace, policy)` of family `name`."""
function family(name, x, u, bins, kw)
    nb = length(bins) - 1
    name === :batch1d && return ((nb, B), () -> SFC.GPUSFWorkspace(BE, bins),
        (s, c, be, ws, pol) -> SFC.calculate_structure_function_batch!(
            s, c, OP, x, u, bins; backend = be, workspace = ws, culling = pol, kw...))
    name === :joint && return ((nb, NV, B), () -> SFC.GPUSFWorkspace(BE, bins, VAL; kind = :joint2d),
        (s, c, be, ws, pol) -> SFC.calculate_structure_function_2d_batch!(
            s, c, OP, x, u, bins, VAL; backend = be, workspace = ws, culling = pol, kw...))
    name === :sp1d && return ((NI, nb, B), () -> SFC.GPUSFWorkspace(BE, bins; kind = :single_pass),
        (s, c, be, ws, pol) -> SFC.calculate_structure_functions_single_pass_batch!(
            s, c, x, u, bins; backend = be, workspace = ws, culling = pol, kw...))
    name === :sp2d && return ((NI, nb, NV, B), () -> SFC.GPUSFWorkspace(BE, bins, VB; kind = :single_pass_2d),
        (s, c, be, ws, pol) -> SFC.calculate_structure_functions_single_pass_2d_batch!(
            s, c, x, u, bins, VB; backend = be, workspace = ws, culling = pol, kw...))
    error("not implemented: family $name")
end

# (family, coordinate width, last distance edge, shared positions, weighted)
const CASES = (
    (:batch1d, 2, 0.06, false, true),
    (:batch1d, 7, 0.3, true, false),
    (:joint, 2, 0.06, true, false),
    (:joint, 7, 0.3, false, true),
    (:sp1d, 2, 0.06, true, false),
    (:sp1d, 7, 0.3, false, true),
    (:sp2d, 2, 0.06, true, true),
    (:sp2d, 7, 0.3, false, false),
)

# (tensor order, shared positions, weighted)
const TENSOR_CASES = ((2, true, false), (3, false, true), (4, true, true))

# (operator, weighted)
const FIELD_CASES = ((SFT.MixedSFType{1, 0, 2}(), false), (SFT.VectorDotSFType(1, 2), true))

Test.@testset "culling on the device slice batch" begin
    Random.seed!(20260926)

    # Each family culls shared positions once and varying positions per slice, under each policy in turn.
    Test.@testset "$name D=$D shared=$shared weighted=$weighted" for (name, D, hi, shared, weighted) in CASES
        bins = collect(range(0.0, hi; length = 9))
        x = shared ? rand(D, N) : rand(D, N, B)
        u = randn(D, N, B)
        kw = weighted ? (; weights = 0.5 .+ rand(N)) : (;)
        CT = weighted ? Float64 : UInt32
        shape, workspace, run! = family(name, x, u, bins, kw)
        rs, rc = zeros(shape), zeros(CT, shape)
        run!(rs, rc, SER, nothing, SFC.NoCulling())
        Test.@testset "$(nameof(typeof(pol)))" for pol in POLICIES
            ws = workspace()
            gs, gc = CUDA.zeros(Float64, shape...), CUDA.zeros(CT, shape...)
            run!(gs, gc, DEV, ws, pol)
            check(gs, gc, rs, rc, ws)
        end
    end

    # The moment tensor runs the same tiled kernels through the same workspace memo.
    Test.@testset "tensor P=$P shared=$shared weighted=$weighted" for (P, shared, weighted) in TENSOR_CASES
        D, bins = 2, collect(range(0.0, 0.06; length = 9))
        nb = length(bins) - 1
        kw = weighted ? (; weights = 0.5 .+ rand(N)) : (;)
        CT = weighted ? Float64 : UInt32
        x = shared ? rand(D, N) : rand(D, N, B)
        u = randn(D, N, B)
        rs, rc = zeros(ntuple(_ -> D, P)..., nb, B), zeros(CT, nb, B)
        SFC.calculate_structure_function_tensor!(rs, rc, Val(P), x, u, bins; backend = SER, kw...)
        ws = SFC.GPUSFWorkspace(BE, bins)
        gs, gc = CUDA.zeros(Float64, size(rs)...), CUDA.zeros(CT, size(rc)...)
        SFC.calculate_structure_function_tensor!(gs, gc, Val(P), x, u, bins; backend = DEV, workspace = ws,
                                                 culling = SFC.AlwaysCulling(), kw...)
        check(gs, gc, rs, rc, ws)
    end

    # The multi-field runs the same tiled kernels through the same workspace memo.
    Test.@testset "multi-field $(nameof(typeof(op))) weighted=$weighted" for (op, weighted) in FIELD_CASES
        D, bins = 2, collect(range(0.0, 0.06; length = 9))
        nb = length(bins) - 1
        kw = weighted ? (; weights = 0.5 .+ rand(N)) : (;)
        CT = weighted ? Float64 : UInt32
        x = rand(D, N)
        f = MF.Fields(vectors = (randn(D, N), randn(D, N)), scalars = (randn(N),))
        rs, rc = zeros(nb), zeros(CT, nb)
        SFC.calculate_structure_function!(rs, rc, op, x, f, bins; backend = SER, kw...)
        ws = SFC.GPUSFWorkspace(BE, bins)
        gs, gc = CUDA.zeros(Float64, nb), CUDA.zeros(CT, nb)
        SFC.calculate_structure_function!(gs, gc, op, x, f, bins; backend = DEV, workspace = ws,
                                          culling = SFC.AlwaysCulling(), kw...)
        check(gs, gc, rs, rc, ws)
    end
end
