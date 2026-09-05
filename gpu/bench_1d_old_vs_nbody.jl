# =============================================================================
# Head-to-head for the ONE 1D regime where the routing choice is still live:
# fixed-x individual SF, where the warp-replica strip kernel beats the CUDA
# N-body kernel (tiny-histogram, contention-bound). Verifies equal histograms.
#   julia --project=gpu gpu/bench_1d_old_vs_nbody.jl
#
# The other regimes were measured head-to-head and are no longer a choice — the
# losing launchers were unreachable from production and have been deleted:
#   ind varying  61 vs 115 bapps, SP1D fixed 30 vs 43, SP1D varying 28 vs 43,
#   SP2D fixed 5 vs 25, SP2D varying 1 vs 25 (all: N-body/unified wins).
# See gpu/OPTIMAL_KERNEL_DESIGN.md.
# =============================================================================
using StructureFunctions
import KernelAbstractions as KA
using CUDA, Printf
using Statistics: median
const SF = StructureFunctions
const SFC = SF.Calculations
const GE = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
const SFT = SF.StructureFunctionTypes
using StructureFunctions: LinearBinEdges
const FT = Float32
const sf2 = SFT.L2SFType()
const N = parse(Int, get(ENV, "SF_N", "20000"))
const B = parse(Int, get(ENV, "SF_B", "64"))
const NB = 50
const D = 2
const BE = CUDA.CUDABackend()
const GEOM = SF.HelperFunctions.FlatGeometry{D}()
tput(t) = (N * (N - 1) / 2) * B / t / 1e9
bench(f) = (f(); f(); ts = Float64[]; for _ in 1:5
    t = time_ns(); f(); push!(ts, (time_ns() - t) / 1e9)
end; median(ts))

lbe = LinearBinEdges(collect(range(0.05f0, 2.0f0; length = NB + 1)))
dig = GE._sf_batch_dist_digitizer(BE, lbe)
x_fix = rand(FT, D, N); u = randn(FT, D, N, B)

println("warp-replica strip vs N-body, 1D individual fixed-x, N=$N B=$B NB=$NB\n")
println("| regime | strip bapps (s@8064) | nbody bapps (s@8064) | winner | equal? |")

let
    xo, uo = GE._stage_batch_device(BE, x_fix, u; fixed_x = true)
    so = CUDA.zeros(FT, NB, B); co = CUDA.zeros(UInt32, NB, B)
    fo() = (CUDA.fill!(so, 0f0); CUDA.fill!(co, UInt32(0));
        GE._launch_batch_fixed_x_sf!(BE, so, co, xo, uo, sf2, N, B, lbe, GEOM); CUDA.synchronize())
    to = bench(fo)
    xn = CuArray(x_fix); un = CuArray(u)
    sn = CUDA.zeros(FT, 1, NB, B); cn = CUDA.zeros(UInt32, 1, NB, B)
    fn() = (CUDA.fill!(sn, 0f0); CUDA.fill!(cn, UInt32(0));
        SFC.gpu_fast_launch_1d_batch!(BE, sn, cn, xn, un, sf2, dig, N, NB, B, D, 1, true, GEOM, nothing);
        CUDA.synchronize())
    tn = bench(fn)
    eq = Array(co) == reshape(Array(cn), NB, B)
    @printf("| ind fixed | %.0f (%.1f) | %.0f (%.1f) | %s | %s |\n",
        tput(to), to * 8064 / B, tput(tn), tn * 8064 / B,
        tput(to) > tput(tn) ? "strip" : "nbody", eq)
end
println("\nDONE_BENCH")
