using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LogBinEdges, LinearBinEdges
using Test: Test
using SpectralBackends: SpectralBackends as SB
using StaticArrays: StaticArrays as SA
using Distributed: Distributed
using SharedArrays: SharedArrays

# Workers added here are removed at the end of this file
const _WORKERS_ADDED_HERE =
    Distributed.nprocs() == 1 ? Distributed.addprocs(2) : Int[]

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
        bins;
        backend = CB.SerialBackend(),
        verbose = false,
        show_progress = false,
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    out_serial, counts_serial = res_serial.sums, res_serial.counts

    # 2. Threaded
    res_thread = SFC.calculate_structure_function(
        sf_type,
        x,
        u,
        bins;
        backend = CB.ThreadedBackend(),
        verbose = false,
        show_progress = false,
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    out_thread, counts_thread = res_thread.sums, res_thread.counts

    Test.@testset "Serial vs Threaded" begin
        Test.@test out_serial[1] ≈ out_thread[1]
        Test.@test out_serial[2] ≈ out_thread[2]
        Test.@test counts_serial == counts_thread
    end

    # The workers this file adds are `LocalManager` workers on this node, so they reach no core the
    # threaded backend does not already have. `Auto` therefore keeps the local backend and an
    # explicit `DistributedBackend()` — exercised below — is the only way to reach them.
    Test.@testset "AutoBackend selection" begin
        Test.@test Distributed.nworkers() > 1
        Test.@test SFC.distributed_adds_hardware(Val(:distributed)) == false
        for shape in (SFC.PointField{2}(), SFC.SharedPositionField{2}())
            Test.@test SFC.resolve_auto_backend(shape, () -> true; nthreads = 4) isa
                       CB.AbstractThreadedBackend
            Test.@test SFC.resolve_auto_backend(shape, () -> true; nthreads = 1) isa
                       CB.AbstractSerialBackend
            Test.@test SFC.resolve_auto_backend(shape, () -> false; nthreads = 4) isa
                       CB.AbstractSerialBackend
        end
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
        bins;
        backend = CB.DistributedBackend(),
        verbose = false,
        show_progress = false,
        output_type = SF.StructureFunctionSumsAndCounts,
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
        bins;
        backend = CB.DistributedBackend(CB.ThreadedBackend()),
        verbose = false,
        show_progress = false,
        output_type = SF.StructureFunctionSumsAndCounts,
    )

    Test.@testset "Serial vs Distributed(Threaded) hybrid" begin
        Test.@test out_serial ≈ res_hybrid.sums
        Test.@test counts_serial == res_hybrid.counts
    end

    # 3c. Batched distributed (distribute the batch axis across workers), serial+hybrid inner.
    xb = rand(2, N, 4)
    ub = rand(2, N, 4)
    res_ser_b = SFC.calculate_structure_function(
        sf_type, xb, ub, bins;
        backend = CB.SerialBackend(), verbose = false, show_progress = false,
        output_type = SF.StructureFunctionSumsAndCounts,
    )
    Test.@testset "Serial vs Distributed batched" begin
        for inner in (CB.SerialBackend(), CB.ThreadedBackend())
            res_db = SFC.calculate_structure_function(
                sf_type, xb, ub, bins;
                backend = CB.DistributedBackend(inner), verbose = false, show_progress = false,
                output_type = SF.StructureFunctionSumsAndCounts,
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
        2;  # n_bins = 2
        backend = CB.DistributedBackend(),
        bin_spacing = LogBinEdges,
        verbose = false,
        show_progress = false,
        output_type = SF.StructureFunctionSumsAndCounts,
    )

    res_serial_int = SFC.calculate_structure_function(
        sf_type,
        x,
        u,
        2;
        bin_spacing = LogBinEdges,
        verbose = false,
        show_progress = false,
        output_type = SF.StructureFunctionSumsAndCounts,
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
    xb, ub = rand(2, N, T), randn(2, N, T)
    bins = collect(range(0.0, 1.0; length = NB + 1))
    vbins = collect(range(-3.0, 3.0; length = NV + 1))
    op = SFT.L2SFType()
    R = 6.371e6
    xs = permutedims(hcat(rand(N) .* 0.4, rand(N) .* 0.4 .- 0.2))
    us = randn(2, N)
    sbins = collect(range(0.0, 1.2e6; length = NB + 1))

    entries = (
        ("non-mutating point", be -> begin
            r = SFC.calculate_structure_function(op, xp, up, bins; backend = be, verbose = false,
                show_progress = false, output_type = SF.StructureFunctionSumsAndCounts)
            (r.sums, r.counts)
        end),
        ("in-place point", be -> begin
            s, c = zeros(NB), zeros(UInt32, NB)
            SFC.calculate_structure_function!(s, c, op, xp, up, bins; backend = be,
                verbose = false, show_progress = false)
            (s, c)
        end),
        ("non-mutating auxiliary axes", be -> begin
            r = SFC.calculate_structure_function(op, xb, ub, bins; backend = be, verbose = false,
                show_progress = false, output_type = SF.StructureFunctionSumsAndCounts)
            (r.sums, r.counts)
        end),
        ("in-place auxiliary axes", be -> begin
            s, c = zeros(NB, T), zeros(UInt32, NB, T)
            SFC.calculate_structure_function!(s, c, op, xb, ub, bins; backend = be,
                verbose = false, show_progress = false)
            (s, c)
        end),
        ("non-mutating joint point", be -> begin
            r = SFC.calculate_structure_function(op, xp, up, bins, vbins; backend = be,
                verbose = false, show_progress = false)
            (r.sums, r.counts)
        end),
        ("non-mutating joint auxiliary axes", be -> begin
            r = SFC.calculate_structure_function(op, xb, ub, bins, vbins; backend = be,
                verbose = false, show_progress = false)
            (r.sums, r.counts)
        end),
        ("in-place joint point", be -> begin
            s, c = zeros(NB, NV), zeros(UInt32, NB, NV)
            SFC.calculate_structure_function!(s, c, op, xp, up, bins, vbins; backend = be,
                verbose = false, show_progress = false)
            (s, c)
        end),
        ("in-place joint auxiliary axes", be -> begin
            s, c = zeros(NB, NV, T), zeros(UInt32, NB, NV, T)
            SFC.calculate_structure_function!(s, c, op, xb, ub, bins, vbins; backend = be,
                verbose = false, show_progress = false)
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
        ("harmonic direct sum", be -> begin
            hθ = acos.(clamp.(range(-0.95, 0.95; length = N), -1, 1))
            hφ = [2π * (i * 0.6180339887498949 % 1) for i in 1:N]
            hx = permutedims(hcat(collect(hφ), π / 2 .- collect(hθ)))
            hu = Float64[sin(d + 2i) for d in 1:2, i in 1:N]
            nodes = SF.HarmonicNodes(collect(range(0.2, 2.6; length = 9)), 16)
            r = SFC.calculate_structure_function(op, hx, hu, nodes, SB.DirectSumSpectralBackend();
                backend = be, verbose = false,
                output_type = SF.StructureFunctionSumsAndCounts)
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
                collect(range(prevfloat(0.0), π; length = 5)); backend = be, verbose = false,
                show_progress = false,
                second_axis = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)))
            (r.sums, r.counts)
        end),
        # The joint kernels take a geometry, not a metric; a sphere must not be read as flat.
        ("joint point on a sphere", be -> begin
            r = SFC.calculate_structure_function(op, xs, us, sbins, vbins; backend = be,
                verbose = false, show_progress = false,
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
            dis_s, dis_c = run(CB.DistributedBackend())
            Test.@test counts_agree(dis_c, ser_c)
            Test.@test maximum(abs, dis_s .- ser_s) <= 1e-10 * max(maximum(abs, ser_s), 1e-10)
        end
    end
end

isempty(_WORKERS_ADDED_HERE) || Distributed.rmprocs(_WORKERS_ADDED_HERE; waitfor = 30)
