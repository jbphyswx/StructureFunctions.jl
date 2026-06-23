# =============================================================================
# Head-to-head: OLD batch launchers (global warp-replica + W-strip + merge) vs
# NEW CUDA N-body kernels, for the 1D regimes, at the SAME config (N=20000).
# Decides per-regime routing for "optimal everywhere". Also verifies the two
# produce equal histograms.
#   julia --project=gpu gpu/bench_1d_old_vs_nbody.jl
# =============================================================================
using StructureFunctions
import KernelAbstractions as KA
using CUDA, Printf
using Statistics: median
const SF = StructureFunctions
const SFC = SF.Calculations
const GE = Base.get_extension(SF, :StructureFunctionsGPUExt)
const SFT = SF.StructureFunctionTypes
using StructureFunctions: LinearBinEdges
const FT = Float32
const sf2 = SFT.L2SFType()
const N = parse(Int, get(ENV, "SF_N", "20000"))
const B = parse(Int, get(ENV, "SF_B", "64"))
const NB = 50
const D = 2
const BE = CUDA.CUDABackend()
tput(t) = (N*(N-1)/2)*B/t/1e9
bench(f) = (f(); f(); ts = Float64[]; for _ in 1:5; t = time_ns(); f(); push!(ts, (time_ns()-t)/1e9); end; median(ts))

lbe = LinearBinEdges(collect(range(0.05f0, 2.0f0; length = NB + 1)))
dig = GE._sf_batch_dist_digitizer(BE, lbe)
x_fix = rand(FT, D, N); u = randn(FT, D, N, B); x_var = rand(FT, D, N, B)

println("OLD vs N-body, 1D, N=$N B=$B NB=$NB\n")
println("| regime | old bapps (s@8064) | nbody bapps (s@8064) | winner | equal? |")

# ---- individual fixed (NMOM=1) ----
let
    xo, uo = GE._stage_batch_device(BE, x_fix, u; fixed_x = true)
    so = CUDA.zeros(FT, NB, B); co = CUDA.zeros(UInt32, NB, B)
    fo() = (CUDA.fill!(so,0f0); CUDA.fill!(co,UInt32(0)); GE._launch_batch_fixed_x_sf!(BE, so, co, xo, uo, sf2, N, B, lbe); CUDA.synchronize())
    to = bench(fo)
    xn = CuArray(x_fix); un = CuArray(u); sn = CUDA.zeros(FT,1,NB,B); cn = CUDA.zeros(UInt32,1,NB,B)
    fn() = (CUDA.fill!(sn,0f0); CUDA.fill!(cn,UInt32(0)); SFC.gpu_fast_launch_1d_batch!(BE, sn, cn, xn, un, sf2, dig, N, NB, B, D, 1, true); CUDA.synchronize())
    tn = bench(fn)
    eq = Array(co) == reshape(Array(cn), NB, B)
    @printf("| ind fixed | %.0f (%.1f) | %.0f (%.1f) | %s | %s |\n", tput(to), to*8064/B, tput(tn), tn*8064/B, tput(to)>tput(tn) ? "OLD" : "nbody", eq)
end

# ---- individual varying (NMOM=1) ----
let
    xo, uo = GE._stage_batch_device(BE, x_var, u; fixed_x = false)
    so = CUDA.zeros(FT, NB, B); co = CUDA.zeros(UInt32, NB, B)
    fo() = (CUDA.fill!(so,0f0); CUDA.fill!(co,UInt32(0)); GE._launch_batch_varying_x_sf!(BE, so, co, xo, uo, sf2, N, B, lbe); CUDA.synchronize())
    to = bench(fo)
    xn = CuArray(x_var); un = CuArray(u); sn = CUDA.zeros(FT,1,NB,B); cn = CUDA.zeros(UInt32,1,NB,B)
    fn() = (CUDA.fill!(sn,0f0); CUDA.fill!(cn,UInt32(0)); SFC.gpu_fast_launch_1d_batch!(BE, sn, cn, xn, un, sf2, dig, N, NB, B, D, 1, false); CUDA.synchronize())
    tn = bench(fn)
    eq = Array(co) == reshape(Array(cn), NB, B)
    @printf("| ind varying | %.0f (%.1f) | %.0f (%.1f) | %s | %s |\n", tput(to), to*8064/B, tput(tn), tn*8064/B, tput(to)>tput(tn) ? "OLD" : "nbody", eq)
end

# ---- SP1D fixed (NMOM=6) ----
let
    xo, uo = GE._stage_batch_device(BE, x_fix, u; fixed_x = true)
    so = CUDA.zeros(FT, 6, NB, B); co = CUDA.zeros(UInt32, 6, NB, B)
    fo() = (CUDA.fill!(so,0f0); CUDA.fill!(co,UInt32(0)); GE._launch_batch_fixed_x_sp1d!(BE, so, co, xo, uo, N, B, lbe); CUDA.synchronize())
    to = bench(fo)
    xn = CuArray(x_fix); un = CuArray(u); sn = CUDA.zeros(FT,6,NB,B); cn = CUDA.zeros(UInt32,6,NB,B)
    fn() = (CUDA.fill!(sn,0f0); CUDA.fill!(cn,UInt32(0)); SFC.gpu_fast_launch_1d_batch!(BE, sn, cn, xn, un, sf2, dig, N, NB, B, D, 6, true); CUDA.synchronize())
    tn = bench(fn)
    eq = Array(co) == Array(cn)
    @printf("| SP1D fixed | %.0f (%.1f) | %.0f (%.1f) | %s | %s |\n", tput(to), to*8064/B, tput(tn), tn*8064/B, tput(to)>tput(tn) ? "OLD" : "nbody", eq)
end
println("\nDONE_BENCH")
