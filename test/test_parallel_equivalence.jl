using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, LogBinEdges
using Test: Test
using StaticArrays: StaticArrays as SA
using Distributed: Distributed
using SharedArrays: SharedArrays

# The keywords the Distributed extension forwards on its pmap and batch-leading paths, each row on one inner backend.
const PE_SER, PE_THR = CB.DistributedBackend(), CB.DistributedBackend(CB.ThreadedBackend())
const PE_DISTRIBUTED_ROWS = (
    ("in-place weighted point", PE_SER, (be, d) -> begin
        s, c = zeros(d.NB), zeros(d.NB)
        SFC.calculate_structure_function!(s, c, d.op, d.xp, d.up, d.bins; backend = be, weights = d.wp)
        (s, c)
    end),
    ("joint slice batch over the angle axis, open last bin", PE_THR, (be, d) -> begin
        s, c = zeros(d.NO, d.NA, d.T), zeros(UInt32, d.NO, d.NA, d.T)
        SFC.calculate_structure_function_2d_batch!(s, c, d.op, d.xb, d.ub, d.open, d.abins; backend = be,
            second_axis = d.angle)
        (s, c)
    end),
    ("in-place joint point over the angle axis", PE_SER, (be, d) -> begin
        s, c = zeros(d.NB, d.NA), zeros(UInt32, d.NB, d.NA)
        SFC.calculate_structure_function!(s, c, d.op, d.xp, d.up, d.bins, d.abins; backend = be,
            second_axis = d.angle)
        (s, c)
    end),
    ("joint point on a sphere", PE_SER, (be, d) -> begin
        r = SFC.calculate_structure_function(d.op, d.xs, d.us, d.sbins, d.vbins; backend = be,
            distance_metric = SF.HelperFunctions.SphericalDistance(d.R))
        (r.sums, r.counts)
    end),
)

# (partial family, inner backend): every family culled, the inner backends alternating.
const PE_SHARE_CASES = (("1d", CB.SerialBackend()), ("joint", CB.ThreadedBackend()), ("sp1d", CB.SerialBackend()),
                        ("sp2d", CB.ThreadedBackend()), ("tensor", CB.SerialBackend()),
                        ("multi-field", CB.ThreadedBackend()))

const _WORKERS_ADDED_HERE =
    Distributed.nprocs() == 1 ? Distributed.addprocs(2) : Int[]
try

Distributed.@everywhere using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, LogBinEdges
Distributed.@everywhere using StaticArrays: StaticArrays as SA
Distributed.@everywhere using SharedArrays: SharedArrays
Distributed.@everywhere using OhMyThreads: OhMyThreads  # for hybrid CB.DistributedBackend(CB.ThreadedBackend())

Test.@testset "Distributed over shared arrays is the serial answer" begin
    # Bins from a count on a log grid, each worker threading its share.
    N = 50
    x = rand(2, N)
    u = rand(2, N)
    sf_type = SFT.LongitudinalSecondOrderStructureFunction
    sx = SharedArrays.SharedArray{eltype(x)}(size(x))
    su = SharedArrays.SharedArray{eltype(u)}(size(u))
    sx .= x
    su .= u
    res_dist_int = SFC.calculate_structure_function(sf_type, sx, su, 2, SF.StructureFunctionSumsAndCounts;
        backend = CB.DistributedBackend(CB.ThreadedBackend()), bin_spacing = LogBinEdges)
    res_serial_int = SFC.calculate_structure_function(sf_type, x, u, 2, SF.StructureFunctionSumsAndCounts;
        backend = CB.SerialBackend(), bin_spacing = LogBinEdges)
    Test.@test res_serial_int.sums ≈ res_dist_int.sums
    Test.@test res_serial_int.counts == res_dist_int.counts
end

Test.@testset "Distributed forwards weights, the angle axis and a sphere's geometry" begin
    N, T, NB, NV, NA = 40, 3, 6, 5, 4
    d = (; N, T, NB, NV, NA, xp = rand(2, N), up = randn(2, N), wp = rand(N) .+ 0.5, xb = rand(2, N, T),
         ub = randn(2, N, T), bins = collect(range(0.0, 1.0; length = NB + 1)),
         vbins = collect(range(-3.0, 3.0; length = NV + 1)), abins = collect(range(prevfloat(0.0), π; length = NA + 1)),
         angle = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)), op = SFT.L2SFType(), R = 6.371e6,
         xs = permutedims(hcat(rand(N) .* 0.4, rand(N) .* 0.4 .- 0.2)), us = randn(2, N),
         sbins = collect(range(0.0, 1.2e6; length = NB + 1)))
    d = (; d..., open = SF.InfPaddedBinEdges(d.bins), NO = NB + 2)

    # Integer counts are exact; a weighted count is a float mass, compared like a sum.
    counts_agree(a, b) = (eltype(a) <: Integer && eltype(b) <: Integer) ? a == b :
        maximum(abs, a .- b) <= 1e-10 * max(maximum(abs, b), 1e-10)

    for (name, be, run) in PE_DISTRIBUTED_ROWS
        Test.@testset "$name" begin
            ser_s, ser_c = run(CB.SerialBackend(), d)
            dis_s, dis_c = run(be, d)
            Test.@test counts_agree(dis_c, ser_c)
            Test.@test maximum(abs, dis_s .- ser_s) <= 1e-10 * max(maximum(abs, ser_s), 1e-10)
        end
    end

    # An unbounded last bin cannot be culled, so an explicit AlwaysCulling() raising shows the policy arrived.
    Test.@testset "an explicit culling policy reaches the distributed batch kernels" begin
        open_bins, no = d.open, d.NO
        always = SFC.AlwaysCulling()
        be = PE_THR
        Test.@test_throws ArgumentError SFC.calculate_structure_function_batch!(zeros(no, T),
            zeros(UInt32, no, T), d.op, d.xb, d.ub, open_bins; backend = be, culling = always)
        Test.@test_throws ArgumentError SFC.calculate_structure_function_2d_batch!(zeros(no, NV, T),
            zeros(UInt32, no, NV, T), d.op, d.xb, d.ub, open_bins, d.vbins; backend = be, culling = always)
        Test.@test_throws ArgumentError SFC.calculate_structure_functions_single_pass_batch!(
            zeros(SFC.SINGLE_PASS_N, no, T), zeros(UInt32, SFC.SINGLE_PASS_N, no, T), d.xb, d.ub,
            open_bins; backend = be, culling = always)
        Test.@test_throws ArgumentError SFC.calculate_structure_functions_single_pass_2d_batch!(
            zeros(SFC.SINGLE_PASS_N, no, NV, T), zeros(UInt32, SFC.SINGLE_PASS_N, no, NV, T), d.xb,
            d.ub, open_bins, d.vbins; backend = be, culling = always)
    end
end

Test.@testset "the shares of every partial family add to the culled whole sweep" begin
    N, k = 60, 3
    x, u = rand(2, N), randn(2, N)
    f = SF.MultiFields.Fields(vectors = (u,))
    xv, uv = (x[1, :], x[2, :]), (u[1, :], u[2, :])
    vb = collect(range(-3.0, 3.0; length = 6))
    op = SFT.L2SFType()
    g2 = SF.HelperFunctions.FlatGeometry{2}()
    family = Dict(
        "1d" => (be, db, sh, kw) -> (r = SFC._partial_sums_counts(be, op, xv, uv, db, sh, UInt32; kw...);
                                     (r.sums, r.counts)),
        "joint" => (be, db, sh, kw) -> SFC._partial_2d_sums_counts(be, op, xv, uv, db, vb, sh, UInt32; kw...),
        "sp1d" => (be, db, sh, kw) -> SFC._partial_single_pass_1d(be, x, u, db, sh, UInt32; kw...),
        "sp2d" => (be, db, sh, kw) -> SFC._partial_single_pass_2d(be, x, u, db, vb, sh, UInt32; kw...),
        "tensor" => (be, db, sh, kw) -> SFC.tensor_partial(be, Val(2), SFC.PointField{2}(), x, u, db, sh, UInt32;
                                                           kw...),
        "multi-field" => (be, db, sh, kw) -> SFC.field_partial(be, op, x, f, db, sh, UInt32; kw...),
    )
    db = collect(range(0.0, 0.1; length = 7))
    for (name, be) in PE_SHARE_CASES
        part = family[name]
        whole = part(CB.SerialBackend(), db, (1, 1), (; geometry = g2, culling = SFC.NoCulling()))
        shares = [part(be, db, (w, k), (; geometry = g2, culling = SFC.AutoCulling())) for w in 1:k]
        Test.@test (name, sum(s[2] for s in shares) == whole[2]) == (name, true)
        Test.@test (name, isapprox(sum(s[1] for s in shares), whole[1]; rtol = 1e-12)) == (name, true)
    end
end

finally
    isempty(_WORKERS_ADDED_HERE) || Distributed.rmprocs(_WORKERS_ADDED_HERE; waitfor = 30)
end
