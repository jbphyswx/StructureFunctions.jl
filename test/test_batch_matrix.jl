using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LinearBinEdges, pair_from_linear

Random.seed!(2025)

const RAW = SF.StructureFunctionSumsAndCounts
const RAW2 = SF.StructureFunction2DSumsAndCounts
const SER = CB.SerialBackend()
const OP = SFT.L2SFType()
const SP_INV = (:S2, :L2, :T2, :S3, :L3, :L1T2)

"""Sums equal to within rounding, counts exactly or, for a weighted pair mass, to within rounding."""
same_sums(a, b) = isapprox(a, b; rtol = 1e-10, atol = 1e-10 * maximum(abs, b; init = 0.0))
same_counts(a, b) = eltype(a) <: Integer ? a == b : same_sums(a, b)
same((s, c), (rs, rc)) = same_sums(s, rs) && same_counts(c, rc)

# pair_from_linear enumerates the upper triangle row by row, each pair i < j once.
Test.@testset "pair_from_linear" begin
    Test.@test all(N -> [pair_from_linear(k, N) for k in 1:(N * (N - 1) ÷ 2)] ==
                        [(i, j) for i in 1:(N - 1) for j in (i + 1):N], (2, 3, 50))
end

const BM_N = 100
const BM_DB = collect(range(0.0, 0.4; length = 9))
const BM_VB = collect(range(-0.2, 0.6; length = 7))
const BM_ND, BM_NV = length(BM_DB) - 1, length(BM_VB) - 1

"""Whether the `family` batch over `B` slices, positions `shared` or per slice, `weighted` or not, with value edges
`vbins` and keywords `axis`, equals one point call per slice."""
function batch_is_slices(family, shared, B, weighted, vbins = BM_VB, axis = (;))
    Random.seed!(hash((family, shared, B, weighted)))
    x = shared ? rand(2, BM_N) : rand(2, BM_N, B)
    u = rand(2, BM_N, B)
    CT = weighted ? Float64 : UInt32
    kw = weighted ? (; backend = SER, weights = rand(BM_N) .+ 0.5) : (; backend = SER)
    xs(t) = shared ? x : x[:, :, t]
    out(dims...) = (zeros(dims..., B), zeros(CT, dims..., B))
    if family === :sf1d
        got, ref = out(BM_ND), out(BM_ND)
        SFC.calculate_structure_function_batch!(got..., OP, x, u, BM_DB; kw...)
        for t in 1:B
            r = SFC.calculate_structure_function(OP, xs(t), u[:, :, t], BM_DB, CT, RAW; kw...)
            ref[1][:, t] .= r.sums
            ref[2][:, t] .= r.counts
        end
    elseif family === :joint
        got, ref = out(BM_ND, BM_NV), out(BM_ND, BM_NV)
        SFC.calculate_structure_function_2d_batch!(got..., OP, x, u, BM_DB, vbins; axis..., kw...)
        for t in 1:B
            r = SFC.calculate_structure_function(OP, xs(t), u[:, :, t], BM_DB, vbins, CT, RAW2; axis..., kw...)
            ref[1][:, :, t] .= r.sums
            ref[2][:, :, t] .= r.counts
        end
    elseif family === :sp1d
        got, ref = out(6, BM_ND), out(6, BM_ND)
        SFC.calculate_structure_functions_single_pass_batch!(got..., x, u, BM_DB; kw...)
        for t in 1:B
            r = SFC.calculate_structure_functions_single_pass(xs(t), u[:, :, t], BM_DB, CT, RAW; kw...)
            for (k, inv) in enumerate(SP_INV)
                ref[1][k, :, t] .= r[inv].sums
                ref[2][k, :, t] .= r[inv].counts
            end
        end
    else
        got, ref = out(6, BM_ND, BM_NV), out(6, BM_ND, BM_NV)
        SFC.calculate_structure_functions_single_pass_2d_batch!(got..., x, u, BM_DB, vbins; kw...)
        for t in 1:B
            r = SFC.calculate_structure_functions_single_pass_2d(xs(t), u[:, :, t], BM_DB, vbins, CT, RAW2; kw...)
            for (k, inv) in enumerate(SP_INV)
                ref[1][k, :, :, t] .= r[inv].sums
                ref[2][k, :, :, t] .= r[inv].counts
            end
        end
    end
    return same(got, ref)
end

# (family, shared, B, weighted, value edges, axis keywords): every family and value-edge form on shared positions over
# few slices (the rows kernels); the lanes kernels from BL_SLICE_LANES_MIN shared slices in the two families that take
# them on flat geometry; one slice; positions per slice and pair weights in every family.
const CPU_BATCH_CASES = (
    (:sf1d, true, 3, false, BM_VB, (;)),
    (:joint, true, 3, false, BM_VB, (;)),
    (:joint, true, 3, false, LinearBinEdges(-0.2, 0.6, 7), (;)),
    (:joint, true, 3, false, collect(range(-0.01, π / 2 + 0.01; length = 7)),
     (; second_axis = SFC.SeparationAngleAxis(ones(2)))),
    (:sp1d, true, 3, false, BM_VB, (;)),
    (:sp2d, true, 3, false, BM_VB, (;)),
    (:sp2d, true, 3, false, LinearBinEdges(-0.2, 0.6, 7), (;)),
    (:sp2d, true, 3, false, ntuple(k -> LinearBinEdges(-0.3 * k, 0.6, 7), 6), (;)),
    (:sf1d, true, SFC.BL_SLICE_LANES_MIN, false, BM_VB, (;)),
    (:sp1d, true, SFC.BL_SLICE_LANES_MIN, false, BM_VB, (;)),
    (:sf1d, true, 1, false, BM_VB, (;)),
    (:sf1d, false, 3, false, BM_VB, (;)), (:joint, false, 3, false, BM_VB, (;)),
    (:sp1d, false, 3, false, BM_VB, (;)), (:sp2d, false, 3, false, BM_VB, (;)),
    (:sf1d, true, 3, true, BM_VB, (;)), (:joint, true, 3, true, BM_VB, (;)),
    (:sp1d, true, 3, true, BM_VB, (;)), (:sp2d, true, 3, true, BM_VB, (;)),
)

Test.@testset "a CPU batch equals one call per slice" begin
    for (family, shared, B, weighted, vbins, axis) in CPU_BATCH_CASES
        case = (family, shared, B, weighted, typeof(vbins))
        Test.@test (case, batch_is_slices(family, shared, B, weighted, vbins, axis)) == (case, true)
    end
end

# On a sphere a CPU batch takes the curved-geometry kernels, culled per slice or once for shared positions: positions
# varying per slice in every family, shared in the joint.
Test.@testset "a CPU batch on a sphere equals one call per slice" begin
    Random.seed!(77)
    N, B = 40, 2
    kw = (; backend = SER, distance_metric = SFC.DI.Haversine(6.371e6), culling = SFC.AlwaysCulling())
    lonlat() = vcat(300 .* rand(1, N) .- 150, 100 .* rand(1, N) .- 50)
    u = randn(2, N, B)
    db = collect(range(0.0, 3.0e6; length = 9))
    vb = collect(range(-2.0, 6.0; length = 7))
    one(r) = ((r.sums, r.counts),)
    six(r) = map(k -> (r[k].sums, r[k].counts), SP_INV)
    families = (
        ("1-D", (x, uu) -> one(SFC.calculate_structure_function(OP, x, uu, db, RAW; kw...))),
        ("joint", (x, uu) -> one(SFC.calculate_structure_function(OP, x, uu, db, vb; kw...))),
        ("single pass", (x, uu) -> six(SFC.calculate_structure_functions_single_pass(x, uu, db, RAW; kw...))),
        ("single pass 2D", (x, uu) -> six(SFC.calculate_structure_functions_single_pass_2d(x, uu, db, vb; kw...))),
    )
    for (x, fams) in ((cat(lonlat(), lonlat(); dims = 3), families), (lonlat(), families[2:2]))
        for (family, f) in fams
            got = f(x, u)
            refs = [f(ndims(x) == 3 ? x[:, :, t] : x, u[:, :, t]) for t in 1:B]
            agrees = all(t -> all(k -> isapprox(selectdim(got[k][1], ndims(got[k][1]), t), refs[t][k][1]; rtol = 1e-10) &&
                                       selectdim(got[k][2], ndims(got[k][2]), t) == refs[t][k][2], eachindex(got)), 1:B)
            Test.@test (family, ndims(x), agrees) == (family, ndims(x), true)
        end
    end
end

# Every batch family refuses counts past the count type's range, from a nonzero start and from zero, writing nothing.
Test.@testset "batch counts past the count type are refused before anything is written" begin
    for N in (4, 24)
        u = randn(2, N, 2)
        x = rand(2, N)
        bins = [0.0, 2.0]
        value_bins = [-1e6, 1e6]
        initial = N == 4 ? UInt8(250) : UInt8(0)
        for (dims, call!) in (
            ((1, 2), (s, c) -> SFC.calculate_structure_function_batch!(s, c, OP, x, u, bins; backend = SER)),
            ((1, 1, 2), (s, c) -> SFC.calculate_structure_function_2d_batch!(s, c, OP, x, u, bins, value_bins;
                                                                             backend = SER)),
            ((6, 1, 2), (s, c) -> SFC.calculate_structure_functions_single_pass_batch!(s, c, x, u, bins; backend = SER)),
            ((6, 1, 1, 2), (s, c) -> SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, x, u, bins,
                                                                                             value_bins; backend = SER)))
            sums, counts = zeros(dims), fill(initial, dims)
            Test.@test_throws ArgumentError call!(sums, counts)
            Test.@test all(iszero, sums) && all(==(initial), counts)
        end
    end
end
