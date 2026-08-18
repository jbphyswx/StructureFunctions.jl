#!/usr/bin/env julia
# Stage 0 measurement harness: one row per (entry x shape x D x eltype x bin type x N x B x backend).
#
# Every later performance claim is a before/after row from this table. Output is both an aligned
# human table and a CSV under benchmark/benchmark_results/ so runs can be diffed.
#
# The plan's full cross-product is ~2.5e4 cells and N=1e5 alone is 5e9 pairs, so the matrix is
# expressed as SWEEPS: each sweep varies one axis against a fixed base point. Scale with env knobs.
#
# Usage:
#   julia -t 32 --project=benchmark benchmark/regimes.jl
#   SF_BENCH_SCALE=full julia -t 32 --project=benchmark benchmark/regimes.jl
#   SF_BENCH_ONLY=dist_bins julia --project=benchmark benchmark/regimes.jl

using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, LinearBinEdges, LogBinEdges, InfPaddedBinEdges
using OhMyThreads: OhMyThreads              # threaded extension
using KernelAbstractions: KernelAbstractions as KA   # GPU(KA.CPU) parity backend
using Printf, Random, Dates

const SCALE = get(ENV, "SF_BENCH_SCALE", "fast")
const ONLY = get(ENV, "SF_BENCH_ONLY", "")
const REPS = parse(Int, get(ENV, "SF_BENCH_REPS", "3"))
const OUTDIR = joinpath(@__DIR__, "benchmark_results")

# ---------------------------------------------------------------------------------------------
# Timing. min-of-k after a warmup; allocations from a separate warmed call so the timing loop is
# not perturbed by @allocated.
# ---------------------------------------------------------------------------------------------
function measure(f, n_pairs::Real)
    f()                                   # warm (compile)
    GC.gc()
    best = Inf
    for _ in 1:REPS
        best = min(best, @elapsed f())
    end
    bytes = @allocated f()
    return (; seconds = best, bytes = bytes,
            ns_per_pair = best * 1e9 / max(n_pairs, 1), pairs = n_pairs)
end

# ---------------------------------------------------------------------------------------------
# Roofline. Measured triad bandwidth is the ceiling for the streaming half of the pair loop; the
# FLOP ceiling is measured with a dependency-free FMA chain. Reported so "speed of light" is a
# number and not a mood.
# ---------------------------------------------------------------------------------------------
function measured_triad_gbs(n = 1 << 24)
    a = rand(Float64, n); b = rand(Float64, n); c = similar(a)
    triad!() = (@inbounds @simd for i in eachindex(c); c[i] = a[i] + 3.0 * b[i]; end)
    triad!()
    t = Inf
    for _ in 1:3
        t = min(t, @elapsed triad!())
    end
    return 3 * n * sizeof(Float64) / t / 1e9      # 2 reads + 1 write
end

# No FLOP ceiling is reported. The pair loop is bound by the digitize dependency chain and the
# histogram scatter, not by arithmetic throughput (benchmark/profile/open_issues.md §3 measured
# `@fastmath` at exactly 0% twice, because the scatter blocks vectorization). A peak-FLOP number
# would be a real measurement of an irrelevant ceiling, so it is deliberately absent.

# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
_bin(::Val{:linear}, ::Type{T}, nb) where {T} = LinearBinEdges(range(T(0), T(2); length = nb + 1))
_bin(::Val{:log}, ::Type{T}, nb) where {T} = LogBinEdges(collect(T, 10 .^ range(T(-2), T(0.4); length = nb + 1)))
_bin(::Val{:infpad}, ::Type{T}, nb) where {T} = InfPaddedBinEdges(LinearBinEdges(range(T(0), T(2); length = nb + 1)))
_bin(::Val{:vector}, ::Type{T}, nb) where {T} = collect(T, range(T(0), T(2); length = nb + 1))
_valbins(::Type{T}, nv) where {T} = LinearBinEdges(range(T(-3), T(3); length = nv + 1))

# UInt32 counts saturate at N = 92682 (see `_assert_counts_representable`); widen past that so the
# harness measures the kernel instead of tripping the guard.
_count_eltype(N) = N > 92_682 ? UInt64 : UInt32

const BACKENDS = Dict(
    :serial => CB.SerialBackend(),
    :threaded => CB.ThreadedBackend(),
    :gpu_kacpu => CB.GPUBackend(KA.CPU()),
)

const SFTYPE = SFT.L2SFType()

# Each entry returns (closure, n_pairs). `B` is the auxiliary/batch extent (1 = point field).
function build_case(entry::Symbol, D::Int, ::Type{T}, binkind::Symbol, N::Int, B::Int, backend) where {T}
    Random.seed!(1234)
    nb, nv = 32, 16
    db = _bin(Val(binkind), T, nb)
    vb = _valbins(T, nv)
    ce = _count_eltype(N)
    x = B == 1 ? rand(T, D, N) : rand(T, D, N)          # shared positions when batched
    u = B == 1 ? rand(T, D, N) : rand(T, D, N, B)
    n_pairs = (N * (N - 1) ÷ 2) * B
    kw = (; verbose = false, show_progress = false)

    f = if entry === :sf1d
        () -> SFC.calculate_structure_function(SFTYPE, x, u, db, ce; backend = backend,
                  output_type = SF.StructureFunctionSumsAndCounts, kw...)
    elseif entry === :joint2d
        () -> SFC.calculate_structure_function(SFTYPE, x, u, db, vb; backend = backend,
                  count_eltype = ce, kw...)
    elseif entry === :sp1d
        () -> SFC.calculate_structure_functions_single_pass(x, u, db; backend = backend,
                  output_type = SF.StructureFunctionSumsAndCounts, count_eltype = ce)
    elseif entry === :sp2d
        () -> SFC.calculate_structure_functions_single_pass_2d(x, u, db, vb; backend = backend,
                  count_eltype = ce)
    elseif entry === :tensor
        () -> SFC.calculate_structure_function_tensor(Val(2), x, u, db; backend = backend, count_eltype = ce)
    else
        error("unknown entry $entry")
    end
    return f, n_pairs
end

# ---------------------------------------------------------------------------------------------
# Sweeps: each varies ONE axis against the base point.
# ---------------------------------------------------------------------------------------------
const BASE = (; entry = :sf1d, D = 2, T = Float64, bin = :log, N = 2000, B = 1, backend = :serial)

function sweeps()
    small = SCALE == "fast"
    Ns = small ? [500, 1000, 2000] : [1000, 3000, 10_000, 30_000]
    return [
        # The default bin type is LogBinEdges, so its digitize cost is on the default path.
        (:dist_bins, [merge(BASE, (; bin = b)) for b in (:linear, :log, :infpad, :vector)]),
        (:eltype, [merge(BASE, (; T = t)) for t in (Float64, Float32)]),
        (:dimension, [merge(BASE, (; D = d)) for d in (2, 3)]),
        (:entry, [merge(BASE, (; entry = e)) for e in (:sf1d, :sp1d, :joint2d, :sp2d, :tensor)]),
        (:scaling_N, [merge(BASE, (; N = n)) for n in Ns]),
        (:batch_B, [merge(BASE, (; N = 400, B = b)) for b in (1, 8, 64)]),
        (:backend, [merge(BASE, (; backend = k)) for k in (:serial, :threaded, :gpu_kacpu)]),
    ]
end

# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------
function main()
    mkpath(OUTDIR)
    stamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    csv = joinpath(OUTDIR, "regimes-$stamp.csv")

    gbs = measured_triad_gbs()
    @printf("%s\nStage 0 regimes | julia %s | nthreads=%d | scale=%s reps=%d\n",
            "="^118, VERSION, Threads.nthreads(), SCALE, REPS)
    @printf("roofline (measured): triad %.1f GB/s single-core stream bandwidth\n%s\n", gbs, "="^118)
    flush(stdout)

    open(csv, "w") do io
        println(io, "sweep,entry,D,eltype,bins,N,B,backend,pairs,seconds,ns_per_pair,bytes,status")
        for (name, cases) in sweeps()
            (!isempty(ONLY) && String(name) != ONLY) && continue
            @printf("\n--- sweep: %s ---\n", name)
            @printf("%-8s %-8s %-2s %-8s %-7s %7s %4s %-10s %12s %10s %9s\n",
                    "entry", "", "D", "eltype", "bins", "N", "B", "backend", "pairs", "ns/pair", "MiB")
            flush(stdout)
            for c in cases
                be = BACKENDS[c.backend]
                label = @sprintf("%-8s %-8s %-2d %-8s %-7s %7d %4d %-10s",
                                 c.entry, "", c.D, c.T, c.bin, c.N, c.B, c.backend)
                try
                    f, np = build_case(c.entry, c.D, c.T, c.bin, c.N, c.B, be)
                    m = measure(f, np)
                    @printf("%s %12d %10.2f %9.2f\n", label, m.pairs, m.ns_per_pair, m.bytes / 2^20)
                    println(io, join((name, c.entry, c.D, c.T, c.bin, c.N, c.B, c.backend,
                                      m.pairs, m.seconds, m.ns_per_pair, m.bytes, "ok"), ","))
                catch e
                    msg = first(split(sprint(showerror, e), '\n'))
                    @printf("%s %12s %10s %9s  SKIP: %s\n", label, "-", "-", "-", first(msg, 60))
                    println(io, join((name, c.entry, c.D, c.T, c.bin, c.N, c.B, c.backend,
                                      0, 0, 0, 0, "skip: " * replace(msg, "," => ";")), ","))
                end
                flush(stdout); flush(io)
            end
        end
    end
    @printf("\n%s\nwrote %s\nDONE\n", "="^118, csv)
end

main()
