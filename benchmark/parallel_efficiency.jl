# Parallel efficiency floors, per route, within one allocation.
#
# Gated, not part of the suite: it spends minutes of wall clock and needs reserved cores to mean
# anything. Ratios measured inside one job are portable; absolute seconds are not, so nothing here
# pins a time.
#
# The load-bearing assertion is the one that needs no tuned constant: **a parallel backend must not
# be slower than serial on the same input**. A route can decompose into the right number of work
# items, spread them over every worker, and still lose — the auxiliary-axis distributed path did
# exactly that for as long as it split by slice instead of by outer pair index, recomputing each
# pair's geometry once per slice. Item counts cannot see that; a wall-clock floor can.
#
# Run: julia -t 8 --project=test benchmark/parallel_efficiency.jl
#      SF_EFF_WORKERS=4 to choose the worker count (default 4).

using Distributed

const NWORKERS = parse(Int, get(ENV, "SF_EFF_WORKERS", "4"))
const TEST_PROJ = joinpath(@__DIR__, "..", "test")

# A worker inherits `JULIA_EXCLUSIVE`, under which each single-threaded worker pins its one thread
# to the first CPU of the mask — the same CPU for all of them, so the pool time-shares one core.
if nworkers() < NWORKERS
    addprocs(NWORKERS - (nprocs() == 1 ? 0 : nworkers());
             exeflags = ["--project=$(abspath(TEST_PROJ))", "-t", "1"],
             env = ["JULIA_EXCLUSIVE" => "0"])
end

@everywhere begin
    using StructureFunctions
    using StructureFunctions.Calculations: Calculations as SFC
    using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
    using StructureFunctions.StructureFunctionObjects: StructureFunctionObjects as SFO
    using ComputationalBackends: ComputationalBackends as CB
    using OhMyThreads: OhMyThreads
end

using Printf, Random

"""Minimum over `reps` runs: on a shared node a single sample carries noise the size of the effect."""
function fastest(f, reps::Int = 3)
    f()
    return minimum(begin
        t0 = time()
        f()
        time() - t0
    end for _ in 1:reps)
end

"""
A parallel backend that is slower than serial has lost whatever its decomposition bought. The
tolerance is for measurement noise on a shared node, not for a real regression: it admits a backend
that is at parity and rejects one that is meaningfully behind.
"""
const NOT_SLOWER_THAN_SERIAL = 1.25

"""
Floor on threaded speed-up, well under what the routes measure so ordinary noise cannot trip it:
the point and batch kernels run near-ideal to 8 threads, so half of the thread count is a wide
margin and still fails a route that has quietly gone serial.
"""
threaded_floor(nthreads::Int) = max(1.5, nthreads / 2)

const OP = SFT.S2SFType()
const BINS = collect(range(0.0, 1.0; length = 21))
const VALUE_BINS = collect(range(0.0, 2.0; length = 6))

sf1d(backend, x, u) = SFC.calculate_structure_function(OP, x, u, BINS, Float64;
    backend = backend, verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
joint2d(backend, x, u) = SFC.calculate_structure_function(OP, x, u, BINS, VALUE_BINS, Float64;
    backend = backend, verbose = false, output_type = SFO.StructureFunction2DSumsAndCounts)
"""The public single-pass entry, mutating so the comparison allocates nothing per repeat."""
function single_pass(backend, x, u)
    nb = length(BINS) - 1
    sums = zeros(Float64, SFC.SINGLE_PASS_N, nb)
    counts = zeros(Float64, SFC.SINGLE_PASS_N, nb)
    SFC.calculate_structure_functions_single_pass!(sums, counts, x, u, BINS; backend = backend)
    return sums, counts
end

"""One route × one problem: time every backend, check the floors, report the row."""
function check_route(name, call, x, u; failures)
    t_serial = fastest(() -> call(CB.SerialBackend(), x, u))
    t_thread = fastest(() -> call(CB.ThreadedBackend(), x, u))
    t_dist = fastest(() -> call(CB.DistributedBackend(), x, u))

    speedup_thread = t_serial / t_thread
    speedup_dist = t_serial / t_dist
    floor_t = threaded_floor(Threads.nthreads())

    ok_thread = speedup_thread >= floor_t
    ok_thread_slow = t_thread <= NOT_SLOWER_THAN_SERIAL * t_serial
    ok_dist_slow = t_dist <= NOT_SLOWER_THAN_SERIAL * t_serial
    ok = ok_thread && ok_thread_slow && ok_dist_slow

    ok || push!(failures, name)
    @printf("%-34s serial %8.4f  threaded %8.4f (%5.2f×)  distributed %8.4f (%5.2f×)  %s\n",
        name, t_serial, t_thread, speedup_thread, t_dist, speedup_dist, ok ? "ok" : "FAILED")
    ok_thread || @printf("    threaded speed-up %.2f× is under the %.2f× floor\n",
        speedup_thread, floor_t)
    ok_dist_slow || @printf("    distributed is %.2f× SLOWER than serial\n", t_dist / t_serial)
    return ok
end

function main()
    Random.seed!(20260918)
    println("threads=", Threads.nthreads(), "  workers=", nworkers(),
            "  threaded floor=", round(threaded_floor(Threads.nthreads()); digits = 2), "×")
    failures = String[]

    n = 5000
    x = rand(2, n)
    u = rand(2, n)
    check_route("point 1D", sf1d, x, u; failures)
    check_route("point joint 2D", joint2d, x, u; failures)
    check_route("point single-pass 1D", single_pass, x, u; failures)

    # The auxiliary-axis routes are the ones a slice-wise split silently ruins: the batch-leading
    # kernels amortise a pair's geometry over the slices, so any decomposition that splits slices
    # instead of the outer index pays it B times over.
    for nslices in (4, 8)
        ub = rand(2, n, nslices)
        check_route("auxiliary-axis 1D (B=$nslices)", sf1d, x, ub; failures)
        check_route("auxiliary-axis joint 2D (B=$nslices)", joint2d, x, ub; failures)
    end

    println()
    if isempty(failures)
        println("PARALLEL EFFICIENCY OK")
        return 0
    end
    println("PARALLEL EFFICIENCY FAILED on $(length(failures)) route(s): ", join(failures, ", "))
    return 1
end

exit(main())
