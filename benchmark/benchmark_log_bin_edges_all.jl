# LogBinEdges digitize benchmark — see benchmark/LOG_BIN_EDGES_BENCHMARK.md
#
# Run:
#   julia --project=. benchmark/benchmark_log_bin_edges_all.jl
# Log:
#   test/debug/log_bin_edges_all_benchmark.log
using Printf: @printf
using StructureFunctions: StructureFunctions as SF
using Random: Random

function bench_fixed(f, container, queries; reps::Int=5)
    f(container, queries[1])
    best = typemax(Float64)
    for _ in 1:reps
        t0 = time_ns()
        s = 0
        for q in queries
            s += f(container, q)
        end
        t1 = time_ns()
        best = min(best, (t1 - t0) / length(queries))
        s == 0 && error("dead code elimination?")
    end
    return best
end

function bench_scalar(f, queries; reps::Int=5)
    f(queries[1])
    best = typemax(Float64)
    for _ in 1:reps
        t0 = time_ns()
        s = 0
        for q in queries
            s += f(q)
        end
        t1 = time_ns()
        best = min(best, (t1 - t0) / length(queries))
        s == 0 && error("dead code elimination?")
    end
    return best
end

"""Shipped floor(t)+1 + ULP correction on log-space coordinate lx."""
function search_lin_floor_corr(lin::SF.LinearBinEdges, lx)
    T = eltype(lin.edges)
    f = lin.first_edge
    lx <= f && return 1
    l = lin.last_edge
    n = length(lin.edges)
    lx > l && return n + 1
    t = muladd(lx, lin.inv_step, -f * lin.inv_step)
    idx = clamp(floor(Int, t) + 1, 1, n)
    u = muladd(T(idx - 1), lin.step_val, f)
    return u < lx ? idx + 1 : idx
end

"""P6d: floor(t)+1 guess (shipped discrete map), no ULP correction."""
function search_lin_floor_nocheck(lin::SF.LinearBinEdges, x)
    lx = log(x)
    f = lin.first_edge
    lx <= f && return 1
    l = lin.last_edge
    n = length(lin.edges)
    lx > l && return n + 1
    t = muladd(lx, lin.inv_step, -f * lin.inv_step)
    return clamp(floor(Int, t) + 1, 1, n)
end

count_errors(f, container, ref_f, queries) =
    sum(q -> f(container, q) != ref_f(q) ? 1 : 0, queries)

function run_for_type(::Type{T}, N::Int) where {T}
    println("=" ^ 72)
    @printf("TYPE %s   N_edges = %d   (time: 10k queries × 5 reps; errors: verify set)\n", T, N)
    println("=" ^ 72)

    Random.seed!(42)
    queries = rand(T, 10_000) .* T(999) .+ T(1)

    log_range = range(log(T(1)), log(T(1000)); length=N)
    log_vec = T[exp(log_range[i]) for i in 1:N]
    lin_on_log = SF.LinearBinEdges(log_range)

    verify = copy(queries)
    for i in 1:N
        push!(verify, log_vec[i], log_vec[i] - eps(log_vec[i]), log_vec[i] + eps(log_vec[i]))
    end
    push!(verify, T(0.5), T(1005))

    ref_phys(q) = searchsortedfirst(log_vec, q)
    ref_lin(q) = searchsortedfirst(lin_on_log, log(q))

    be_phys = SF.LogBinEdges(log_vec)
    be_log = SF.LogBinEdges_from_log_edges(log_range)

    methods = [
        ("P1: searchsortedfirst(physical Vector)", (v, x) -> searchsortedfirst(log_vec, x), log_vec),
        ("P2: LogBinEdges (unified log+FMA)", (v, x) -> searchsortedfirst(v, x), be_phys),
        ("P2b: LogBinEdges_from_log_edges", (v, x) -> searchsortedfirst(v, x), be_log),
        ("P5: log(x) + LinearBinEdges floor+corr [ref]", (v, x) -> searchsortedfirst(v, log(x)), lin_on_log),
        ("P6d: floor(t)+1 NO correction [regression]", (v, x) -> search_lin_floor_nocheck(v, x), lin_on_log),
    ]

    println("  err_phys = vs searchsortedfirst(exp.(log_edges), q)")
    println("  err_lin  = vs searchsortedfirst(LinearBinEdges(log_edges), log(q))  [package spec]")
    for (name, f, container) in methods
        t = bench_fixed(f, container, queries)
        e_phys = count_errors(f, container, ref_phys, verify)
        e_lin = count_errors(f, container, ref_lin, verify)
        @printf("  %-50s %7.2f ns   phys=%d  lin=%d\n", name, t, e_phys, e_lin)
    end
    println()
end

function run_timing_breakdown(::Type{T}, N::Int) where {T}
    println("=" ^ 72)
    @printf("TIMING BREAKDOWN %s N=%d  (10k queries × 5 reps; container built once)\n", T, N)
    println("=" ^ 72)

    Random.seed!(42)
    queries = rand(T, 10_000) .* T(999) .+ T(1)
    log_range = range(log(T(1)), log(T(1000)); length=N)
    lin = SF.LinearBinEdges(log_range)
    lx_vec = log.(queries)

    ref_lin(q) = searchsortedfirst(lin, log(q))

    rows = [
        ("log(q) only", queries, q -> log(q)),
        ("FMA floor+corr on precomputed lx", lx_vec, lx -> search_lin_floor_corr(lin, lx)),
        ("log(q) + floor+corr (P5 path)", queries, q -> search_lin_floor_corr(lin, log(q))),
        ("log(q) + searchsortedfirst(lin) [dispatch]", queries, q -> searchsortedfirst(lin, log(q))),
        ("LogBinEdges_from_log_edges [shipped]", queries, q -> searchsortedfirst(SF.LogBinEdges_from_log_edges(log_range), q)),
    ]

    for (name, data, f) in rows
        t = bench_scalar(f, data)
        @printf("  %-45s %7.2f ns\n", name, t)
    end

    e_floor = sum(q -> search_lin_floor_corr(lin, log(q)) != ref_lin(q) ? 1 : 0, queries)
    @printf("  floor+corr err_lin=%d\n", e_floor)
    println()
end

for T in (Float64, Float32)
    for N in (51, 1000)
        run_for_type(T, N)
    end
    run_timing_breakdown(T, 1000)
end
println("done. See benchmark/LOG_BIN_EDGES_BENCHMARK.md")
