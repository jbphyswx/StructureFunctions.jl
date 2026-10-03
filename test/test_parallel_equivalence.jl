using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LogBinEdges, LinearBinEdges
using Test: Test
using SpectralBackends: SpectralBackends as SB
using StaticArrays: StaticArrays as SA
using Distributed: Distributed
using SharedArrays: SharedArrays

const _WORKERS_ADDED_HERE =
    Distributed.nprocs() == 1 ? Distributed.addprocs(2) : Int[]
try

Distributed.@everywhere using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LogBinEdges, LinearBinEdges
Distributed.@everywhere using StaticArrays: StaticArrays as SA
Distributed.@everywhere using SharedArrays: SharedArrays
Distributed.@everywhere using OhMyThreads: OhMyThreads  # for hybrid CB.DistributedBackend(CB.ThreadedBackend())

Test.@testset "Parallel Equivalence Verification" begin
    # Dataset
    N = 50
    x = rand(2, N)
    u = rand(2, N)
    bins = SA.SVector(0.0, 0.5, 1.0)
    sf_type = SFT.LongitudinalSecondOrderStructureFunction

    # 1. Serial
    res_serial = SFC.calculate_structure_function(
        sf_type,
        x,
        u,
        bins,
        SF.StructureFunctionSumsAndCounts;
        backend = CB.SerialBackend(),
    )
    out_serial, counts_serial = res_serial.sums, res_serial.counts

    # 2. Threaded
    res_thread = SFC.calculate_structure_function(
        sf_type,
        x,
        u,
        bins,
        SF.StructureFunctionSumsAndCounts;
        backend = CB.ThreadedBackend(),
    )
    out_thread, counts_thread = res_thread.sums, res_thread.counts

    Test.@testset "Serial vs Threaded" begin
        Test.@test out_serial[1] ≈ out_thread[1]
        Test.@test out_serial[2] ≈ out_thread[2]
        Test.@test counts_serial == counts_thread
    end

    # The workers this file adds are `LocalManager` workers on this node: they reach cores this process does not
    # use only when it runs one thread.
    Test.@testset "AutoBackend selection" begin
        Test.@test Distributed.nworkers() > 1
        one_thread = Threads.nthreads() == 1
        Test.@test SFC.distributed_adds_hardware(Val(:distributed)) == one_thread
        Test.@test SFC.resolve_auto_backend(; nthreads = 4) isa
                   (one_thread ? CB.AbstractDistributedBackend : CB.AbstractThreadedBackend)
        Test.@test SFC.resolve_auto_backend(; nthreads = 1) isa
                   (one_thread ? CB.AbstractDistributedBackend : CB.AbstractSerialBackend)
    end

    # 3. Distributed
    sx = SharedArrays.SharedArray{eltype(x)}(size(x))
    su = SharedArrays.SharedArray{eltype(u)}(size(u))
    sx .= x
    su .= u

    res_dist = SFC.calculate_structure_function(
        sf_type,
        sx,
        su,
        bins,
        SF.StructureFunctionSumsAndCounts;
        backend = CB.DistributedBackend(),
    )
    out_dist, counts_dist = res_dist.sums, res_dist.counts

    Test.@testset "Serial vs Distributed" begin
        Test.@test out_serial[1] ≈ out_dist[1]
        Test.@test out_serial[2] ≈ out_dist[2]
        Test.@test counts_serial == counts_dist
    end

    # 3b. Hybrid: CB.DistributedBackend(CB.ThreadedBackend()) — each worker threads over its share.
    res_hybrid = SFC.calculate_structure_function(
        sf_type,
        sx,
        su,
        bins,
        SF.StructureFunctionSumsAndCounts;
        backend = CB.DistributedBackend(CB.ThreadedBackend()),
    )

    Test.@testset "Serial vs Distributed(Threaded) hybrid" begin
        Test.@test out_serial ≈ res_hybrid.sums
        Test.@test counts_serial == res_hybrid.counts
    end

    # 3c. Batched distributed (distribute the batch axis across workers), serial+hybrid inner.
    xb = rand(2, N, 4)
    ub = rand(2, N, 4)
    res_ser_b = SFC.calculate_structure_function(
        sf_type, xb, ub, bins, SF.StructureFunctionSumsAndCounts;
        backend = CB.SerialBackend(),
    )
    Test.@testset "Serial vs Distributed batched" begin
        for inner in (CB.SerialBackend(), CB.ThreadedBackend())
            res_db = SFC.calculate_structure_function(
                sf_type, xb, ub, bins, SF.StructureFunctionSumsAndCounts;
                backend = CB.DistributedBackend(inner),
            )
            Test.@test res_ser_b.counts == res_db.counts
            Test.@test res_ser_b.sums ≈ res_db.sums
        end
    end

    # 4. Distributed with bin count (Int) and LogBinEdges
    res_dist_int = SFC.calculate_structure_function(
        sf_type,
        sx,
        su,
        2,  # n_bins = 2
        SF.StructureFunctionSumsAndCounts;
        backend = CB.DistributedBackend(),
        bin_spacing = LogBinEdges,
    )

    res_serial_int = SFC.calculate_structure_function(
        sf_type,
        x,
        u,
        2,
        SF.StructureFunctionSumsAndCounts;
        bin_spacing = LogBinEdges,
    )

    Test.@testset "Serial vs Distributed (Int/LogBinEdges)" begin
        Test.@test res_serial_int.sums ≈ res_dist_int.sums
        Test.@test res_serial_int.counts == res_dist_int.counts
    end
end

Test.@testset "Distributed covers every entry family" begin
    # One row per entry that takes `backend`; each is its own serial answer computed on workers.
    # An entry family with no distributed method fails here rather than reaching a user.
    N, T, NB, NV = 40, 3, 6, 5
    xp, up = rand(2, N), randn(2, N)
    wp = rand(N) .+ 0.5
    xb, ub = rand(2, N, T), randn(2, N, T)
    bins = collect(range(0.0, 1.0; length = NB + 1))
    vbins = collect(range(-3.0, 3.0; length = NV + 1))
    abins = collect(range(prevfloat(0.0), π; length = 5))
    op = SFT.L2SFType()
    R = 6.371e6
    xs = permutedims(hcat(rand(N) .* 0.4, rand(N) .* 0.4 .- 0.2))
    us = randn(2, N)
    sbins = collect(range(0.0, 1.2e6; length = NB + 1))

    entries = (
        ("non-mutating point", be -> begin
            r = SFC.calculate_structure_function(op, xp, up, bins, SF.StructureFunctionSumsAndCounts;
                backend = be)
            (r.sums, r.counts)
        end),
        ("in-place point", be -> begin
            s, c = zeros(NB), zeros(UInt32, NB)
            SFC.calculate_structure_function!(s, c, op, xp, up, bins; backend = be)
            (s, c)
        end),
        ("non-mutating auxiliary axes", be -> begin
            r = SFC.calculate_structure_function(op, xb, ub, bins, SF.StructureFunctionSumsAndCounts;
                backend = be)
            (r.sums, r.counts)
        end),
        ("in-place auxiliary axes", be -> begin
            s, c = zeros(NB, T), zeros(UInt32, NB, T)
            SFC.calculate_structure_function!(s, c, op, xb, ub, bins; backend = be)
            (s, c)
        end),
        ("non-mutating joint point", be -> begin
            r = SFC.calculate_structure_function(op, xp, up, bins, vbins; backend = be)
            (r.sums, r.counts)
        end),
        ("non-mutating joint auxiliary axes", be -> begin
            r = SFC.calculate_structure_function(op, xb, ub, bins, vbins; backend = be)
            (r.sums, r.counts)
        end),
        ("in-place joint point", be -> begin
            s, c = zeros(NB, NV), zeros(UInt32, NB, NV)
            SFC.calculate_structure_function!(s, c, op, xp, up, bins, vbins; backend = be)
            (s, c)
        end),
        ("in-place joint auxiliary axes", be -> begin
            s, c = zeros(NB, NV, T), zeros(UInt32, NB, NV, T)
            SFC.calculate_structure_function!(s, c, op, xb, ub, bins, vbins; backend = be)
            (s, c)
        end),
        ("slice batch", be -> begin
            s, c = zeros(NB, T), zeros(UInt32, NB, T)
            SFC.calculate_structure_function_batch!(s, c, op, xb, ub, bins; backend = be)
            (s, c)
        end),
        ("joint slice batch", be -> begin
            s, c = zeros(NB, NV, T), zeros(UInt32, NB, NV, T)
            SFC.calculate_structure_function_2d_batch!(s, c, op, xb, ub, bins, vbins; backend = be)
            (s, c)
        end),
        ("single-pass slice batch", be -> begin
            s = zeros(SFC.SINGLE_PASS_N, NB, T)
            c = zeros(UInt32, SFC.SINGLE_PASS_N, NB, T)
            SFC.calculate_structure_functions_single_pass_batch!(s, c, xb, ub, bins; backend = be)
            (s, c)
        end),
        ("single-pass 2D slice batch", be -> begin
            s = zeros(SFC.SINGLE_PASS_N, NB, NV, T)
            c = zeros(UInt32, SFC.SINGLE_PASS_N, NB, NV, T)
            SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, xb, ub, bins, vbins;
                backend = be)
            (s, c)
        end),
        ("in-place single-pass", be -> begin
            s, c = zeros(SFC.SINGLE_PASS_N, NB), zeros(UInt32, SFC.SINGLE_PASS_N, NB)
            SFC.calculate_structure_functions_single_pass!(s, c, xp, up, bins; backend = be)
            (s, c)
        end),
        ("in-place single-pass 2D", be -> begin
            s = zeros(SFC.SINGLE_PASS_N, NB, NV)
            c = zeros(UInt32, SFC.SINGLE_PASS_N, NB, NV)
            SFC.calculate_structure_functions_single_pass_2d!(s, c, xp, up, bins, vbins;
                backend = be)
            (s, c)
        end),
        # Pair weights ride as a keyword too; a route that drops them returns the unweighted answer.
        ("weighted point", be -> begin
            r = SFC.calculate_structure_function(op, xp, up, bins, Float64, SF.StructureFunctionSumsAndCounts;
                backend = be, weights = wp)
            (r.sums, r.counts)
        end),
        ("weighted single-pass", be -> begin
            r = SFC.calculate_structure_functions_single_pass(xp, up, bins, Float64; backend = be, weights = wp)
            (r.L3.sums, r.S2.counts)
        end),
        ("weighted single-pass 2D", be -> begin
            r = SFC.calculate_structure_functions_single_pass_2d(xp, up, bins, vbins, Float64; backend = be,
                weights = wp)
            (r.T2.sums, r.T2.counts)
        end),
        ("harmonic direct sum", be -> begin
            hθ = acos.(clamp.(range(-0.95, 0.95; length = N), -1, 1))
            hφ = [2π * (i * 0.6180339887498949 % 1) for i in 1:N]
            hx = permutedims(hcat(collect(hφ), π / 2 .- collect(hθ)))
            hu = Float64[sin(d + 2i) for d in 1:2, i in 1:N]
            nodes = SF.HarmonicNodes(collect(range(0.2, 2.6; length = 9)), 16)
            r = SFC.calculate_structure_function(op, hx, hu, nodes, SB.DirectSumSpectralBackend(),
                SF.StructureFunctionSumsAndCounts; backend = be)
            (r.sums, r.counts)
        end),
        ("gridded lag sweep", be -> begin
            gs = SFC.UniformLagSchedule((8, 8), (1 / 8, 1 / 8), (true, true))
            gu = reshape(Float64[sin(d + 3i + 7j) for d in 1:2, i in 1:8, j in 1:8], 2, :)
            gb = collect(range(0.0, 0.5; length = NB + 1))
            s, c = zeros(NB), zeros(Int, NB)
            SFC.gridded_lag_sweep!(s, c, op, gu, gs, gb, Val(2), Val(1), Val(0); backend = be)
            (s, c)
        end),
        # The second axis rides as a keyword, so a backend that forwards only the keys it knows
        # about bins the pair value instead and returns a plausible, wrong answer.
        ("joint point over the angle axis", be -> begin
            r = SFC.calculate_structure_function(op, xp, up, bins,
                collect(range(prevfloat(0.0), π; length = 5)); backend = be,
                second_axis = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)))
            (r.sums, r.counts)
        end),
        ("joint slice batch over the angle axis", be -> begin
            s, c = zeros(NB, 4, T), zeros(UInt32, NB, 4, T)
            SFC.calculate_structure_function_2d_batch!(s, c, op, xb, ub, bins, abins; backend = be,
                second_axis = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)))
            (s, c)
        end),
        # The culled batch kernels: varying positions sort per slice, shared ones once.
        ("slice batch culled", be -> begin
            s, c = zeros(NB, T), zeros(UInt32, NB, T)
            SFC.calculate_structure_function_batch!(s, c, op, xb, ub, bins; backend = be,
                culling = SFC.AlwaysCulling())
            (s, c)
        end),
        ("slice batch culled, shared positions", be -> begin
            s, c = zeros(NB, T), zeros(UInt32, NB, T)
            SFC.calculate_structure_function_batch!(s, c, op, xp, ub, bins; backend = be,
                culling = SFC.AlwaysCulling())
            (s, c)
        end),
        ("joint slice batch culled over the angle axis", be -> begin
            s, c = zeros(NB, 4, T), zeros(UInt32, NB, 4, T)
            SFC.calculate_structure_function_2d_batch!(s, c, op, xb, ub, bins, abins; backend = be,
                culling = SFC.AlwaysCulling(), second_axis = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)))
            (s, c)
        end),
        ("single-pass slice batch culled", be -> begin
            s = zeros(SFC.SINGLE_PASS_N, NB, T)
            c = zeros(UInt32, SFC.SINGLE_PASS_N, NB, T)
            SFC.calculate_structure_functions_single_pass_batch!(s, c, xb, ub, bins; backend = be,
                culling = SFC.AlwaysCulling())
            (s, c)
        end),
        ("single-pass 2D slice batch culled", be -> begin
            s = zeros(SFC.SINGLE_PASS_N, NB, NV, T)
            c = zeros(UInt32, SFC.SINGLE_PASS_N, NB, NV, T)
            SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, xb, ub, bins, vbins;
                backend = be, culling = SFC.AlwaysCulling())
            (s, c)
        end),
        ("rank-2 tensor point", be -> begin
            s, c = zeros(2, 2, NB), zeros(UInt32, NB)
            SFC.calculate_structure_function_tensor!(s, c, Val(2), xp, up, bins; backend = be)
            (s, c)
        end),
        ("multi-field point", be -> begin
            s, c = zeros(NB), zeros(UInt32, NB)
            SFC.calculate_structure_function!(s, c, op, xp, SF.MultiFields.Fields(vectors = (up,)), bins; backend = be)
            (s, c)
        end),
        # The joint kernels take a geometry, not a metric; a sphere must not be read as flat.
        ("joint point on a sphere", be -> begin
            r = SFC.calculate_structure_function(op, xs, us, sbins, vbins; backend = be,
                distance_metric = SF.HelperFunctions.SphericalDistance(R))
            (r.sums, r.counts)
        end),
    )

    # Integer counts are exact; a kernel-weighted count is a float mass whose last digits follow the
    # reduction order, so it is compared like a sum.
    counts_agree(a, b) = (eltype(a) <: Integer && eltype(b) <: Integer) ? a == b :
        maximum(abs, a .- b) <= 1e-10 * max(maximum(abs, b), 1e-10)

    for (name, run) in entries
        Test.@testset "$name" begin
            ser_s, ser_c = run(CB.SerialBackend())
            for be in (CB.DistributedBackend(), CB.DistributedBackend(CB.ThreadedBackend()))
                dis_s, dis_c = run(be)
                Test.@test counts_agree(dis_c, ser_c)
                Test.@test maximum(abs, dis_s .- ser_s) <= 1e-10 * max(maximum(abs, ser_s), 1e-10)
            end
        end
    end

    # Culling never changes an answer, so only a request it cannot honour shows the policy arrived:
    # an unbounded last bin reports every far pair, and an explicit AlwaysCulling() must raise.
    Test.@testset "an explicit culling policy reaches the distributed batch kernels" begin
        open_bins = SF.InfPaddedBinEdges(bins)
        no = SF.n_histogram_bins(open_bins)
        always = SFC.AlwaysCulling()
        for be in (CB.SerialBackend(), CB.DistributedBackend())
            Test.@test_throws ArgumentError SFC.calculate_structure_function_batch!(zeros(no, T),
                zeros(UInt32, no, T), op, xb, ub, open_bins; backend = be, culling = always)
            Test.@test_throws ArgumentError SFC.calculate_structure_function_2d_batch!(zeros(no, NV, T),
                zeros(UInt32, no, NV, T), op, xb, ub, open_bins, vbins; backend = be, culling = always)
            Test.@test_throws ArgumentError SFC.calculate_structure_functions_single_pass_batch!(
                zeros(SFC.SINGLE_PASS_N, no, T), zeros(UInt32, SFC.SINGLE_PASS_N, no, T), xb, ub,
                open_bins; backend = be, culling = always)
            Test.@test_throws ArgumentError SFC.calculate_structure_functions_single_pass_2d_batch!(
                zeros(SFC.SINGLE_PASS_N, no, NV, T), zeros(UInt32, SFC.SINGLE_PASS_N, no, NV, T), xb,
                ub, open_bins, vbins; backend = be, culling = always)
            # the same calls without the request run
            s, c = zeros(no, T), zeros(UInt32, no, T)
            SFC.calculate_structure_function_batch!(s, c, op, xb, ub, open_bins; backend = be)
            Test.@test sum(c) == T * N * (N - 1) ÷ 2
        end
    end
end

Test.@testset "the shares of every partial family add to the whole sweep, culled or not" begin
    N, k = 1500, 3
    x, u = rand(2, N), randn(2, N)
    f = SF.MultiFields.Fields(vectors = (u,))
    xv, uv = (x[1, :], x[2, :]), (u[1, :], u[2, :])
    vb = collect(range(-3.0, 3.0; length = 6))
    op = SFT.L2SFType()
    g2 = SF.HelperFunctions.FlatGeometry{2}()
    family = (
        ("1d", (be, db, sh, kw) -> (r = SFC._partial_sums_counts(be, op, xv, uv, db, sh, UInt32; kw...);
                                    (r.sums, r.counts))),
        ("joint", (be, db, sh, kw) -> SFC._partial_2d_sums_counts(be, op, xv, uv, db, vb, sh, UInt32; kw...)),
        ("sp1d", (be, db, sh, kw) -> SFC._partial_single_pass_1d(be, x, u, db, sh, UInt32; kw...)),
        ("sp2d", (be, db, sh, kw) -> SFC._partial_single_pass_2d(be, x, u, db, vb, sh, UInt32; kw...)),
        ("tensor", (be, db, sh, kw) -> SFC.tensor_partial(be, Val(2), SFC.PointField{2}(), x, u, db, sh, UInt32;
                                                          kw...)),
        ("multi-field", (be, db, sh, kw) -> SFC.field_partial(be, op, x, f, db, sh, UInt32; kw...)),
    )
    for (db, culls) in ((collect(range(0.0, 1.5; length = 7)), false), (collect(range(0.0, 0.1; length = 7)), true))
        Test.@test (SFC.cull_grid_for((x[1, :], x[2, :]), g2, db,
                                      SFC.AutoCulling()) !== nothing) == culls
        for (name, part) in family, be in (CB.SerialBackend(), CB.ThreadedBackend())
            whole = part(CB.SerialBackend(), db, (1, 1), (; geometry = g2, culling = SFC.NoCulling()))
            shares = [part(be, db, (w, k), (; geometry = g2, culling = SFC.AutoCulling())) for w in 1:k]
            Test.@test (name, culls, sum(s[2] for s in shares) == whole[2]) == (name, culls, true)
            Test.@test (name, culls, isapprox(sum(s[1] for s in shares), whole[1]; rtol = 1e-12)) ==
                       (name, culls, true)
        end
    end
end

finally
    isempty(_WORKERS_ADDED_HERE) || Distributed.rmprocs(_WORKERS_ADDED_HERE; waitfor = 30)
end
