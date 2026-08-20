#!/usr/bin/env julia
# CPU regime parity + timing harness (Phase 0 deliverable for the CPU perf rework).
#
# Fast + bounded by design (see lessons learned):
#   * flush(stdout) after every line so progress/hangs are visible when redirected to a log
#   * tiny default sizes (runtime ~ms; compilation TTFX ~60-80s dominates a fresh run)
#   * BOTH float types in ONE process to amortize compilation
#   * the threaded-BATCH path is currently BROKEN (threadid bug, can deadlock) — it is
#     skipped unless THREADED_BATCH=1, so a baseline never hangs on it.
#
# Usage:
#   julia -t 32 --project=benchmark benchmark/cpu_regimes.jl
#   N=300 B=256 NPF=3000 THREADED_BATCH=1 julia -t 32 --project=benchmark benchmark/cpu_regimes.jl
#
# Backends: SerialBackend, ThreadedBackend (OhMyThreads ext).

using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using OhMyThreads: OhMyThreads          # load threaded extension
using Printf, Random

# NOTE: defaults are deliberately TINY because the CURRENT batch path is O(1e3) ns/op
# (runtime-SVector type instability → ~2e9 allocations). After the Phase-2 rewrite, bump
# these (e.g. N=400 B=512) to exercise the fast path.
_envi(k, d) = parse(Int, get(ENV, k, string(d)))
const N   = _envi("N", 150)        # points for batch geometry
const B   = _envi("B", 64)         # batch / auxiliary size
const NB  = _envi("NB", 20)        # distance bins
const NV  = _envi("NV", 32)        # value bins (2D)
const NPF = _envi("NPF", 2000)     # point-field N (no batch)
const DO_THREADED_BATCH = get(ENV, "THREADED_BATCH", "0") == "1"

const SFTYPE = SFT.LongitudinalSecondOrderStructureFunctionType()

_bins(::Type{T}) where {T} = collect(T, range(T(0), T(1.5), length = NB + 1))
_vbins(::Type{T}) where {T} = collect(T, range(T(-3), T(3), length = NV + 1))

# min-of-k timing with one warmup
function timeit(f; k = 3)
    f(); GC.gc()
    best = Inf
    for _ in 1:k
        best = min(best, @elapsed f())
    end
    return best
end

_par(rS, rT) = (rS.counts == rT.counts) && isapprox(rS.sums, rT.sums; rtol = 1e-4)

# Measure serial always; threaded protected (can be skipped/broken). Flush every line.
function report(name, pairs, fS, fT; do_threaded = true)
    print("  running $name (serial)…"); flush(stdout)
    rS = fS(); tS = timeit(fS)
    spS = pairs / tS / 1e6
    if !do_threaded
        @printf("\r%-26s serial %8.2f ms  (%7.1f Mpair/s)%-20s\n", name, tS*1e3, spS, ""); flush(stdout)
        return
    end
    print("\r  running $name (threaded)…   "); flush(stdout)
    try
        rT = fT(); tT = timeit(fT); par = _par(rS, rT); spT = pairs / tT / 1e6
        @printf("\r%-26s serial %8.2f ms  thr %8.2f ms  %5.2fx  | %7.1f→%7.1f Mpair/s  parity=%s\n",
                name, tS*1e3, tT*1e3, tS/tT, spS, spT, par); flush(stdout)
    catch e
        @printf("\r%-26s serial %8.2f ms (%7.1f Mpair/s)  | threaded: BROKEN: %s\n",
                name, tS*1e3, spS, first(sprint(showerror, e), 50)); flush(stdout)
    end
end

@printf("%s\nCPU regimes | nthreads=%d  N=%d B=%d NB=%d NV=%d NPF=%d  threaded_batch=%s\n%s\n",
        "="^104, Threads.nthreads(), N, B, NB, NV, NPF, DO_THREADED_BATCH, "="^104); flush(stdout)

for T in (Float64, Float32)
    @printf("--- %s ---\n", T); flush(stdout)
    Random.seed!(42)

    # Regime 1: point-field 1D single SF (O(N^2) headline path); threaded = OMT (works)
    let x = rand(T, 3, NPF), u = rand(T, 3, NPF), bins = _bins(T)
        pairs = NPF*(NPF-1)÷2
        fS() = SFC.calculate_structure_function(SFTYPE, x, u, bins; backend=CB.SerialBackend(),
                  verbose=false, show_progress=false, output_type=SF.StructureFunctionSumsAndCounts)
        fT() = SFC.calculate_structure_function(SFTYPE, x, u, bins; backend=CB.ThreadedBackend(),
                  verbose=false, show_progress=false, output_type=SF.StructureFunctionSumsAndCounts)
        report("point-field 1D", pairs, fS, fT)
    end

    # Regime 2: batched shared-positions 1D ("same surface")
    let x = rand(T, 3, N), u = rand(T, 3, N, B), bins = _bins(T)
        pairs = N*(N-1)÷2 * B
        fS() = SFC.calculate_structure_function(SFTYPE, x, u, bins; backend=CB.SerialBackend(),
                  verbose=false, show_progress=false, output_type=SF.StructureFunctionSumsAndCounts)
        fT() = SFC.calculate_structure_function(SFTYPE, x, u, bins; backend=CB.ThreadedBackend(),
                  verbose=false, show_progress=false, output_type=SF.StructureFunctionSumsAndCounts)
        report("batch shared-pos 1D", pairs, fS, fT; do_threaded = DO_THREADED_BATCH)
    end

    # Regime 3: batched 2D joint
    let x = rand(T, 3, N), u = rand(T, 3, N, B), bins = _bins(T), vb = _vbins(T)
        pairs = N*(N-1)÷2 * B
        fS() = SFC.calculate_structure_function(SFTYPE, x, u, bins, vb; backend=CB.SerialBackend(),
                  verbose=false, show_progress=false, output_type=SF.StructureFunctionSumsAndCounts)
        fT() = SFC.calculate_structure_function(SFTYPE, x, u, bins, vb; backend=CB.ThreadedBackend(),
                  verbose=false, show_progress=false, output_type=SF.StructureFunctionSumsAndCounts)
        report("batch 2D joint", pairs, fS, fT; do_threaded = DO_THREADED_BATCH)
    end
end
@printf("%s\nDONE\n", "="^104); flush(stdout)
