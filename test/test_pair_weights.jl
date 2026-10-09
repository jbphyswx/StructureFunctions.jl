using Test: Test
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO, MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB

# A constant weight `k` on every point scales every sum and count by exactly `k²`, so a route that drops its weights fails.
const K_CONST = 0.6

"""Whether `run` under the constant weight `k` gives `k²` times the unweighted answer `unweighted`."""
function honours_weights(run, unweighted, n_points::Int; k::Real = K_CONST, rtol::Real = 1e-12)
    a_s, a_c = unweighted
    b_s, b_c = run(fill(k, n_points))
    return isapprox(b_s, k^2 .* a_s; rtol = rtol) && isapprox(b_c, k^2 .* a_c; rtol = rtol)
end

"""The weighted 1-D pair sum, written out so it shares no code with the kernels."""
function weighted_pair_loop(op, x, u, w, bins)
    D, N = size(u)
    nb = length(bins) - 1
    s = zeros(Float64, nb)
    c = zeros(Float64, nb)
    for i in 1:(N - 1), j in (i + 1):N
        dx = x[:, j] .- x[:, i]
        r = sqrt(sum(abs2, dx))
        b = searchsortedlast(bins, r)
        (b >= 1 && b <= nb && r > bins[1]) || continue
        pw = w[i] * w[j]
        s[b] += pw * op(u[:, j] .- u[:, i], dx ./ r)
        c[b] += pw
    end
    return s, c
end

# test_device.jl and test_parallel_equivalence.jl check the other backends' weighted routes against serial.
Test.@testset "pair weights reach every route that accepts them" begin
    Random.seed!(2026)
    np = 120
    bins = collect(range(0.0, 1.0; length = 7))
    value_bins = collect(range(0.0, 2.0; length = 5))
    op = SFT.L2SFType()
    ser = CB.SerialBackend()
    x = rand(2, np)
    u = rand(2, np)
    w = 0.3 .+ rand(np)
    nb = length(bins) - 1
    raw(r) = (collect(r.sums), collect(r.counts))
    invariants = (:S2, :L2, :T2, :S3, :L3, :L1T2)
    stacked(r) = (mapreduce(k -> collect(r[k].sums), vcat, invariants),
                  mapreduce(k -> collect(r[k].counts), vcat, invariants))
    sf1d(uu = u) = ws -> raw(SFC.calculate_structure_function(op, x, uu, bins, Float64, SFO.StructureFunctionSumsAndCounts;
                                                              backend = ser, weights = ws))
    joint2d(uu = u) = ws -> raw(SFC.calculate_structure_function(op, x, uu, bins, value_bins, Float64,
                                                                 SFO.StructureFunction2DSumsAndCounts; backend = ser,
                                                                 weights = ws))
    single_pass(uu = u) = ws -> stacked(SFC.calculate_structure_functions_single_pass(x, uu, bins, Float64;
                                                                                      backend = ser, weights = ws))
    single_pass_2d = ws -> stacked(SFC.calculate_structure_functions_single_pass_2d(x, u, bins, value_bins, Float64;
                                                                                    backend = ser, weights = ws))
    tensor = ws -> (s = zeros(Float64, 2, 2, nb); c = zeros(Float64, nb);
                    SFC.calculate_structure_function_tensor!(s, c, Val(2), x, u, bins; backend = ser, weights = ws);
                    (s, c))
    fields = MF.Fields(vectors = (rand(2, np),), scalars = (rand(np),))
    multifield = ws -> (s = zeros(Float64, nb); c = zeros(Float64, nb);
                        SFC.calculate_structure_function!(s, c, SFT.MixedSFType{1, 0, 2}(), x, fields, bins;
                                                          backend = ser, weights = ws); (s, c))

    Test.@testset "the 1-D point histogram, against an independent weighted pair loop" begin
        ref_s, ref_c = weighted_pair_loop(op, x, u, w, bins)
        got = sf1d()(w)
        Test.@test isapprox(got[1], ref_s; rtol = 1e-10)
        Test.@test isapprox(got[2], ref_c; rtol = 1e-10)
    end

    Test.@testset "a constant weight scales every accepting route by k²" begin
        for route in (sf1d(), single_pass(), joint2d(), single_pass_2d, tensor, multifield)
            Test.@test honours_weights(route, route(nothing), np)
        end
    end

    Test.@testset "the auxiliary-axis batches take weights" begin
        # Each batch's unweighted reference is its point route's, slice by slice.
        nt = 3
        ub = rand(2, np, nt)
        slices(route) = [route(ub[:, :, b])(nothing) for b in 1:nt]
        along(f, rs, dims) = (cat((f(r[1]) for r in rs)...; dims), cat((f(r[2]) for r in rs)...; dims))
        rows(v) = permutedims(reshape(v, nb, SFC.SINGLE_PASS_N))
        batch1d = ws -> raw(SFC.calculate_structure_function(op, x, ub, bins, Float64, SFO.StructureFunctionSumsAndCounts;
                                                             backend = ser, weights = ws))
        batch_joint = ws -> raw(SFC.calculate_structure_function(op, x, ub, bins, value_bins, Float64,
                                                                 SFO.StructureFunction2DSumsAndCounts; backend = ser,
                                                                 weights = ws))
        batch_sp1d = ws -> (s = zeros(Float64, SFC.SINGLE_PASS_N, nb, nt); c = zeros(Float64, SFC.SINGLE_PASS_N, nb, nt);
                            SFC.calculate_structure_functions_single_pass_batch!(s, c, x, ub, bins; backend = ser,
                                                                                 weights = ws); (s, c))
        Test.@test honours_weights(batch1d, along(identity, slices(sf1d), 2), np)
        Test.@test honours_weights(batch_joint, along(identity, slices(joint2d), 3), np)
        Test.@test honours_weights(batch_sp1d, along(rows, slices(single_pass), 3), np)
    end

    Test.@testset "weights need a floating-point count type" begin
        Test.@test_throws ArgumentError SFC.calculate_structure_function(op, x, u, bins, UInt32; backend = ser,
                                                                         weights = w)
    end
end
