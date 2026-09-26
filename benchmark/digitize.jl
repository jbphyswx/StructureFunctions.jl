#!/usr/bin/env julia
# Digitize cost. Each typed lookup is timed against the binary search over `collect(bins)` that
# defines it, on the same queries, and must return the same indices; the squared plans likewise
# against `digitize(sqrt(r²))`. Then the per-call plan builds, and one S2 call per bin type.
#
#   julia -t 1 --project=benchmark benchmark/digitize.jl
#   SF_DIGITIZE_N=5000 julia -t 1 --project=benchmark benchmark/digitize.jl

using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    BinEdges, LinearBinEdges, LogBinEdges, InfPaddedBinEdges
using Printf: @printf
using Random: Random

const N_POINTS = parse(Int, get(ENV, "SF_DIGITIZE_N", "20000"))
const N_QUERIES = 1 << 16
const REPS = 7

function best_seconds(f)
    f()
    best = Inf
    for _ in 1:REPS
        t = time_ns()
        f()
        best = min(best, (time_ns() - t) / 1e9)
    end
    return best
end

function lookup_sum(bins, xs)
    s = 0
    @inbounds for x in xs
        s += searchsortedfirst(bins, x)
    end
    return s
end

function squared_sum(plan, r2s)
    s = 0
    @inbounds for r2 in r2s
        s += SF.squared_digitize(plan, r2)
    end
    return s
end

function sqrt_lookup_sum(edges, r2s)
    s = 0
    @inbounds for r2 in r2s
        s += searchsortedfirst(edges, sqrt(r2)) - 1
    end
    return s
end

function bin_cases(::Type{T}, n::Int) where {T}
    rng = Random.Xoshiro(n)
    arbitrary = sort!(T(1.5) .* rand(rng, T, n))
    return (
        ("linear", LinearBinEdges(T(0), T(1.5), n)),
        ("log", LogBinEdges(T(0.01), T(1.5), n)),
        ("log plan", SF.digitize_plan(LogBinEdges(T(0.01), T(1.5), n))),
        ("vector", BinEdges(arbitrary)),
        ("padded", InfPaddedBinEdges(LinearBinEdges(T(0), T(1.5), n))),
    )
end

function queries(::Type{T}, ref) where {T}
    rng = Random.Xoshiro(1)
    xs = T(1.6) .* rand(rng, T, N_QUERIES)
    for e in ref
        isfinite(e) && e >= 0 && append!(xs, (prevfloat(e), e, nextfloat(e)))
    end
    return xs
end

function lookup_rows()
    println("lookup: ns per query, typed vs binary search over collect(bins); squared: ns per r², plan vs sqrt + search")
    @printf("%-8s %-9s %5s  %8s %8s  %8s %8s  %10s %12s\n",
            "T", "bins", "n", "typed", "search", "squared", "sqrt+srch", "plan µs", "sq plan µs")
    for T in (Float32, Float64), n in (17, 65, 1025), (name, bins) in bin_cases(T, n)
        ref = collect(bins)
        xs = queries(T, ref)
        r2s = xs .* xs
        map(x -> searchsortedfirst(bins, x), xs) == map(x -> searchsortedfirst(ref, x), xs) ||
            error("$name $T n=$n: typed lookup disagrees with the binary search")
        plan = SF.squared_digitize_plan(bins)
        map(r2 -> SF.squared_digitize(plan, r2), r2s) == map(r2 -> searchsortedfirst(ref, sqrt(r2)) - 1, r2s) ||
            error("$name $T n=$n: squared plan disagrees with digitize(sqrt(r²))")
        nq = length(xs)
        t_typed = best_seconds(() -> lookup_sum(bins, xs)) / nq * 1e9
        t_ref = best_seconds(() -> lookup_sum(ref, xs)) / nq * 1e9
        t_sq = best_seconds(() -> squared_sum(plan, r2s)) / nq * 1e9
        t_sqref = best_seconds(() -> sqrt_lookup_sum(ref, r2s)) / nq * 1e9
        t_plan = best_seconds(() -> SF.digitize_plan(bins)) * 1e6
        t_sqplan = best_seconds(() -> SF.squared_digitize_plan(bins)) * 1e6
        @printf("%-8s %-9s %5d  %8.2f %8.2f  %8.2f %8.2f  %10.2f %12.2f\n",
                T, name, n, t_typed, t_ref, t_sq, t_sqref, t_plan, t_sqplan)
        flush(stdout)
    end
end

function s2_rows()
    @printf("\nS2, serial, %d thread(s), N = %d: ns per pair\n", Threads.nthreads(), N_POINTS)
    rng = Random.Xoshiro(2)
    x = rand(rng, 2, N_POINTS)
    u = randn(rng, 2, N_POINTS)
    pairs = N_POINTS * (N_POINTS - 1) / 2
    sf = SFT.SecondOrderStructureFunctionType()
    for (name, bins) in bin_cases(Float64, 65)
        name == "log plan" && continue
        t = best_seconds(() -> SFC.calculate_structure_function(sf, x, u, bins,
            SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend()))
        @printf("%-9s %8.3f\n", name, t / pairs * 1e9)
        flush(stdout)
    end
end

lookup_rows()
s2_rows()
println("DONE")
