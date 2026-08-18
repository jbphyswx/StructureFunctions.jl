# Partition balance for O(N²) triangle pair loops (OhMyThreads outer chunks).
using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using Random: Random
using OhMyThreads: OhMyThreads as OMT
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, StructureFunctionObjects as SFO

function _pair_work(n::Int, chunk)
    return sum(n - i for i in chunk)
end

function _triangle_outer_chunks(indices, n_tasks::Integer)
    return OMT.chunks(indices; n = n_tasks, split = OMT.RoundRobin())
end

function _manual_roundrobin_indices(n::Int, n_tasks::Int)
    return [collect(tid:n_tasks:n) for tid in 1:n_tasks]
end

Test.@testset "Triangle outer chunks (OMT RoundRobin)" begin
    n_prod = 32_575
    nthr = 24

    rr_chunks = collect(_triangle_outer_chunks(1:n_prod, nthr))
    manual = _manual_roundrobin_indices(n_prod, nthr)
    Test.@test [collect(c) for c in rr_chunks] == manual

    rr_work = [_pair_work(n_prod, c) for c in rr_chunks]
    Test.@test maximum(rr_work) / minimum(rr_work) < 1.05

    contig_chunks = collect(OMT.chunks(1:n_prod; n = nthr, split = OMT.Consecutive()))
    contig_work = [_pair_work(n_prod, c) for c in contig_chunks]
    Test.@test maximum(contig_work) / minimum(contig_work) > 10

    seen = Int[]
    for c in rr_chunks
        append!(seen, collect(c))
    end
    Test.@test sort(seen) == collect(1:n_prod)
    Test.@test length(seen) == n_prod

    # Threaded vs serial parity (medium N, 2D single-pass)
    if Threads.nthreads() > 1
        n_pts = 120
        x = rand(2, n_pts) .* 50_000.0
        u = randn(2, n_pts)
        distance_bins = exp.(range(log(1000.0), log(50_000.0), length = 11))
        value_bins = ntuple(_ -> collect(range(-1.0, 2.0, length = 12)), 6)

        inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        sp_ref = SFC.calculate_structure_functions_single_pass_2d(
            x, u, distance_bins, value_bins; backend = CB.SerialBackend(),
        )
        sp_thr = SFC.calculate_structure_functions_single_pass_2d(
            x, u, distance_bins, value_bins; backend = CB.ThreadedBackend(),
        )
        for k in inv
            Test.@test sp_thr[k].counts == sp_ref[k].counts
            Test.@test sp_thr[k].sums ≈ sp_ref[k].sums
        end

        sf_type = SFT.LongitudinalSecondOrderStructureFunctionType()
        r1 = SFC.calculate_structure_function(
            sf_type, x, u, distance_bins;
            backend = CB.SerialBackend(), verbose = false, show_progress = false,
            output_type = SFO.StructureFunctionSumsAndCounts,
        )
        r2 = SFC.calculate_structure_function(
            sf_type, x, u, distance_bins;
            backend = CB.ThreadedBackend(), verbose = false, show_progress = false,
            output_type = SFO.StructureFunctionSumsAndCounts,
        )
        Test.@test r2.counts == r1.counts
        Test.@test r2.sums ≈ r1.sums
    end
end
