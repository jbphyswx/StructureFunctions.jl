#!/usr/bin/env julia
# CPU regime parity and timing: serial against threaded for the point-field 1-D, shared-position batch
# 1-D and batch joint entries, both float types in one process.
#
# Usage:
#   julia -t 32 --project=benchmark benchmark/cpu_regimes.jl
#   N=300 B=256 NPF=3000 julia -t 32 --project=benchmark benchmark/cpu_regimes.jl

using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using OhMyThreads: OhMyThreads          # load threaded extension
using Printf: @printf
using Random: Random

_envi(k, d) = parse(Int, get(ENV, k, string(d)))
const N   = _envi("N", 400)        # points for batch geometry
const B   = _envi("B", 512)        # batch / auxiliary size
const NB  = _envi("NB", 20)        # distance bins
const NV  = _envi("NV", 32)        # value bins (2D)
const NPF = _envi("NPF", 6000)     # point-field N (no batch)

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

function report(name, pairs, fS, fT)
    print("  running $name (serial)…"); flush(stdout)
    rS = fS(); tS = timeit(fS)
    spS = pairs / tS / 1e6
    print("\r  running $name (threaded)…   "); flush(stdout)
    rT = fT(); tT = timeit(fT); par = _par(rS, rT); spT = pairs / tT / 1e6
    @printf("\r%-26s serial %8.2f ms  thr %8.2f ms  %5.2fx  | %7.1f→%7.1f Mpair/s  parity=%s\n",
            name, tS*1e3, tT*1e3, tS/tT, spS, spT, par); flush(stdout)
end

@printf("%s\nCPU regimes | nthreads=%d  N=%d B=%d NB=%d NV=%d NPF=%d\n%s\n",
        "="^104, Threads.nthreads(), N, B, NB, NV, NPF, "="^104); flush(stdout)

for T in (Float64, Float32)
    @printf("--- %s ---\n", T); flush(stdout)
    Random.seed!(42)

    let x = rand(T, 3, NPF), u = rand(T, 3, NPF), bins = _bins(T)
        pairs = NPF*(NPF-1)÷2
        fS() = SFC.calculate_structure_function(SFTYPE, x, u, bins, SF.StructureFunctionSumsAndCounts;
                  backend=CB.SerialBackend())
        fT() = SFC.calculate_structure_function(SFTYPE, x, u, bins, SF.StructureFunctionSumsAndCounts;
                  backend=CB.ThreadedBackend())
        report("point-field 1D", pairs, fS, fT)
    end

    let x = rand(T, 3, N), u = rand(T, 3, N, B), bins = _bins(T)
        pairs = N*(N-1)÷2 * B
        fS() = SFC.calculate_structure_function(SFTYPE, x, u, bins, SF.StructureFunctionSumsAndCounts;
                  backend=CB.SerialBackend())
        fT() = SFC.calculate_structure_function(SFTYPE, x, u, bins, SF.StructureFunctionSumsAndCounts;
                  backend=CB.ThreadedBackend())
        report("batch shared-pos 1D", pairs, fS, fT)
    end

    let x = rand(T, 3, N), u = rand(T, 3, N, B), bins = _bins(T), vb = _vbins(T)
        pairs = N*(N-1)÷2 * B
        fS() = SFC.calculate_structure_function(SFTYPE, x, u, bins, vb, SF.StructureFunction2DSumsAndCounts;
                  backend=CB.SerialBackend())
        fT() = SFC.calculate_structure_function(SFTYPE, x, u, bins, vb, SF.StructureFunction2DSumsAndCounts;
                  backend=CB.ThreadedBackend())
        report("batch 2D joint", pairs, fS, fT)
    end
end
@printf("%s\nDONE\n", "="^104); flush(stdout)
