# Partition balance for O(N²) triangle pair loops (OhMyThreads outer chunks).
using Test: Test
using Random: Random
using OhMyThreads: OhMyThreads as OMT
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT

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
        value_bins = [collect(range(-1.0, 2.0, length = 12)) for _ in 1:8]

        s_ref, c_ref = SFC.calculate_structure_functions_single_pass_2d(
            x, u, distance_bins, value_bins; backend = SFC.SerialBackend(),
        )
        s_thr, c_thr = SFC.calculate_structure_functions_single_pass_2d(
            x, u, distance_bins, value_bins; backend = SFC.ThreadedBackend(),
        )
        Test.@test c_thr == c_ref
        Test.@test s_thr ≈ s_ref

        sf_type = SFT.LongitudinalSecondOrderStructureFunctionType()
        bins = [(distance_bins[i], distance_bins[i + 1]) for i in 1:(length(distance_bins) - 1)]
        x_t = (x[1, :], x[2, :])
        u_t = (u[1, :], u[2, :])
        r1 = SFC.calculate_structure_function(
            sf_type, x_t, u_t, bins;
            backend = SFC.SerialBackend(), verbose = false, show_progress = false,
            return_sums_and_counts = true,
        )
        r2 = SFC.calculate_structure_function(
            sf_type, x_t, u_t, bins;
            backend = SFC.ThreadedBackend(), verbose = false, show_progress = false,
            return_sums_and_counts = true,
        )
        Test.@test r2.counts == r1.counts
        Test.@test r2.sums ≈ r1.sums
    end
end
