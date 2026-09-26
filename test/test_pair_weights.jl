using Test: Test
using Random: Random
using LinearAlgebra: LinearAlgebra
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO, MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads

# A route either applies pair weights or refuses; none may accept them and compute the unweighted
# answer. `honours` pins the first by a property no kernel can satisfy by accident: a constant
# weight `k` on every point multiplies each pair's contribution by exactly `k²`, so every sum and
# every count scales by `k²` whatever the operator, the binning or the pair reading.
const K_CONST = 0.6

"""Whether `run(weights)` scales by `k²` under a constant weight, i.e. the weights reach the kernel."""
function honours_weights(run, n_points::Int; k::Real = K_CONST, rtol::Real = 1e-12)
    a_s, a_c = run(nothing)
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

Test.@testset "pair weights reach every route that accepts them" begin
    Random.seed!(2026)
    np = 120
    bins = collect(range(0.0, 1.0; length = 7))
    value_bins = collect(range(0.0, 2.0; length = 5))
    op = SFT.S2SFType()
    device = CB.GPUBackend(KA.CPU())
    x = rand(2, np)
    u = rand(2, np)
    w = 0.3 .+ rand(np)

    Test.@testset "the 1-D point histogram, against an independent weighted pair loop" begin
        ref_s, ref_c = weighted_pair_loop(op, x, u, w, bins)
        for backend in (CB.SerialBackend(), CB.ThreadedBackend(), device)
            got = SFC.calculate_structure_function(op, x, u, bins, Float64, SFO.StructureFunctionSumsAndCounts;
                backend = backend, weights = w)
            Test.@test isapprox(collect(got.sums), ref_s; rtol = 1e-10)
            Test.@test isapprox(collect(got.counts), ref_c; rtol = 1e-10)
        end
    end

    Test.@testset "a constant weight scales every accepting route by k²" begin
        sf1d(backend) = ws -> begin
            r = SFC.calculate_structure_function(op, x, u, bins, Float64, SFO.StructureFunctionSumsAndCounts;
                backend = backend, weights = ws)
            (collect(r.sums), collect(r.counts))
        end
        joint2d(backend) = ws -> begin
            r = SFC.calculate_structure_function(op, x, u, bins, value_bins, Float64,
                SFO.StructureFunction2DSumsAndCounts; backend = backend, weights = ws)
            (collect(r.sums), collect(r.counts))
        end
        invariants = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        stacked(r) = (mapreduce(k -> collect(r[k].sums), vcat, invariants),
                      mapreduce(k -> collect(r[k].counts), vcat, invariants))
        single_pass(backend) = ws -> stacked(SFC.calculate_structure_functions_single_pass(x, u, bins, Float64;
            backend = backend, weights = ws))
        single_pass_2d(backend) = ws -> stacked(SFC.calculate_structure_functions_single_pass_2d(x, u, bins,
            value_bins, Float64; backend = backend, weights = ws))
        tensor(backend) = ws -> begin
            nb = length(bins) - 1
            s = zeros(Float64, 2, 2, nb)
            c = zeros(Float64, nb)
            SFC.calculate_structure_function_tensor!(s, c, Val(2), x, u, bins;
                backend = backend, weights = ws)
            (s, c)
        end

        for backend in (CB.SerialBackend(), CB.ThreadedBackend(), device)
            Test.@test honours_weights(sf1d(backend), np)
            Test.@test honours_weights(single_pass(backend), np)
            Test.@test honours_weights(joint2d(backend), np)
            Test.@test honours_weights(single_pass_2d(backend), np)
            Test.@test honours_weights(tensor(backend), np)
        end
    end

    Test.@testset "the auxiliary-axis batches take weights on every backend that runs them" begin
        nt = 3
        ub = rand(2, np, nt)
        batch1d(backend) = ws -> begin
            r = SFC.calculate_structure_function(op, x, ub, bins, Float64, SFO.StructureFunctionSumsAndCounts;
                backend = backend, weights = ws)
            (collect(r.sums), collect(r.counts))
        end
        batch_joint(backend) = ws -> begin
            r = SFC.calculate_structure_function(op, x, ub, bins, value_bins, Float64,
                SFO.StructureFunction2DSumsAndCounts; backend = backend, weights = ws)
            (collect(r.sums), collect(r.counts))
        end
        batch_sp1d = ws -> begin
            nb = length(bins) - 1
            s = zeros(Float64, SFC.SINGLE_PASS_N, nb, nt)
            c = zeros(Float64, SFC.SINGLE_PASS_N, nb, nt)
            SFC.calculate_structure_functions_single_pass_batch!(s, c, x, ub, bins;
                backend = CB.SerialBackend(), weights = ws)
            (s, c)
        end
        for backend in (CB.SerialBackend(), CB.ThreadedBackend(), device)
            Test.@test honours_weights(batch1d(backend), np)
            Test.@test honours_weights(batch_joint(backend), np)
        end
        Test.@test honours_weights(batch_sp1d, np)
    end

    Test.@testset "multi-field sweeps take weights" begin
        fields = MF.Fields(vectors = (rand(2, np),), scalars = (rand(np),))
        mixed = SFT.MixedSFType{1, 0, 2}()
        nb = length(bins) - 1
        multifield(run!) = ws -> begin
            s = zeros(Float64, nb)
            c = zeros(Float64, nb)
            run!(s, c, ws)
            (s, c)
        end
        Test.@test honours_weights(multifield((s, c, ws) ->
            SFC.calculate_structure_function!(s, c, mixed, x, fields, bins; backend = CB.SerialBackend(),
                weights = ws)), np)
        Test.@test honours_weights(multifield((s, c, ws) ->
            SFC.calculate_structure_function!(s, c, mixed, x, fields, bins; backend = device,
                weights = ws)), np)
    end

    Test.@testset "the device single-pass 2D point path equals the serial weighted answer" begin
        # `honours_weights` proves the weight reaches the kernel; this proves it reaches it the
        # same way the CPU applies it, over both the shared-histogram and the value-column routes.
        for D in (2, 3)
            xd = rand(D, np); ud = rand(D, np)
            r = SFC._dispatch_single_pass_2d(CB.SerialBackend(), SFC.PointField{D}(), xd, ud,
                bins, value_bins, Float64; weights = w)
            g = SFC._dispatch_single_pass_2d(device, SFC.PointField{D}(), xd, ud,
                bins, value_bins, Float64; weights = w)
            Test.@test isapprox(collect(g[1]), collect(r[1]); rtol = 1e-9)
            Test.@test isapprox(collect(g[2]), collect(r[2]); rtol = 1e-9)
        end
    end

    Test.@testset "weights need a floating-point count type" begin
        Test.@test_throws ArgumentError SFC.calculate_structure_function(op, x, u, bins, UInt32;
            backend = CB.SerialBackend(), weights = w)
        Test.@test_throws ArgumentError SFC.calculate_structure_function(op, x, u, bins, UInt32;
            backend = device, weights = w)
    end
end
