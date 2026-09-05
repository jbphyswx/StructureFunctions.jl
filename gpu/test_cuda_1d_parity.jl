# =============================================================================
# Parity + timing for the CUDA fast 1D kernel (N-body broadcast, static-shared
# privatized histogram, TILE=256) vs the KA unified kernel on KA.CPU(). Also
# re-checks the joint2d-varying GPU↔CPU count diff is FP-boundary (few pairs),
# not systematic.
#   julia --project=gpu gpu/test_cuda_1d_parity.jl
# =============================================================================
using StructureFunctions
import KernelAbstractions as KA
using CUDA, StaticArrays, Printf
using Statistics: median
const SF = StructureFunctions
const SFC = SF.Calculations
const GE = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
const SFT = SF.StructureFunctionTypes
const FT = Float32
const N = parse(Int, get(ENV, "SF_T_N", "3000"))
const B = parse(Int, get(ENV, "SF_T_B", "8"))
const D = 2
const sf2 = SFT.L2SFType()
const GEOM = SF.HelperFunctions.FlatGeometry{D}()

function ref_1d(x, u, dig, N, NB, B, NMOM, fixed_x)
    out = zeros(FT, NMOM, NB, B); cnt = zeros(UInt32, NMOM, NB, B)
    GE._sf_launch_1d_batch!(KA.CPU(), out, cnt, x, u, sf2, dig, N, NB, B, D, Val(NMOM), fixed_x, GEOM)
    KA.synchronize(KA.CPU())
    return out, cnt
end
function cuda_1d(xd, ud, dig, N, NB, B, NMOM, fixed_x)
    out = CUDA.zeros(FT, NMOM, NB, B); cnt = CUDA.zeros(UInt32, NMOM, NB, B)
    h = SFC.gpu_fast_launch_1d_batch!(CUDA.CUDABackend(), out, cnt, xd, ud, sf2, dig, N, NB, B, D, NMOM, fixed_x, GEOM, nothing)
    CUDA.synchronize()
    return Array(out), Array(cnt), h
end

println("CUDA 1D fast-kernel parity vs KA.CPU() — N=$N B=$B D=$D\n")
println("| NMOM | fixed_x | NB | handled | max relΔ | max |Δcount| |")
for NMOM in (1, 6), fixed_x in (true, false), NB in (16, 50, 128)
    x_h = fixed_x ? rand(FT, D, N) : rand(FT, D, N, B)
    u_h = randn(FT, D, N, B)
    dist_bins = collect(FT, range(0.05f0, 2.0f0, length = NB + 1))
    dig_c = GE._sf_batch_dist_digitizer(KA.CPU(), dist_bins)
    dig_g = GE._sf_batch_dist_digitizer(CUDA.CUDABackend(), dist_bins)
    o_ref, c_ref = ref_1d(x_h, u_h, dig_c, N, NB, B, NMOM, fixed_x)
    o_cu, c_cu, h = cuda_1d(CuArray(x_h), CuArray(u_h), dig_g, N, NB, B, NMOM, fixed_x)
    rel = maximum(abs.(o_cu .- o_ref) ./ max.(abs.(o_ref), 1f-3))
    dcnt = maximum(abs.(Int.(c_cu) .- Int.(c_ref)))
    @printf("| %d | %s | %d | %s | %.2e | %d |\n", NMOM, fixed_x, NB, h, rel, dcnt)
end

# ---- timing: SP1D NB=50 & individual NB=50, fixed-x, real N=20000 ----
println("\n--- timing 1D fixed-x N=20000 (kernel, extrapolate to B=8064) ---")
let Nt = 20000, Bt = 64, NB = 50
    x_h = rand(FT, D, Nt); u_h = randn(FT, D, Nt, Bt)
    dist_bins = collect(FT, range(0.05f0, 2.0f0, length = NB + 1))
    dig = GE._sf_batch_dist_digitizer(CUDA.CUDABackend(), dist_bins)
    xd = CuArray(x_h); ud = CuArray(u_h)
    for NMOM in (1, 6)
        out = CUDA.zeros(FT, NMOM, NB, Bt); cnt = CUDA.zeros(UInt32, NMOM, NB, Bt)
        f() = (CUDA.fill!(out, 0f0); CUDA.fill!(cnt, UInt32(0));
               SFC.gpu_fast_launch_1d_batch!(CUDA.CUDABackend(), out, cnt, xd, ud, sf2, dig, Nt, NB, Bt, D, NMOM, true, GEOM, nothing);
               CUDA.synchronize())
        f(); f(); ts = Float64[]; for _ in 1:5; t = time_ns(); f(); push!(ts, (time_ns()-t)/1e9); end
        t = median(ts); bapps = (Nt*(Nt-1)/2)*Bt/t/1e9
        @printf("  NMOM=%d NB=%d: %.4f s  (%.1f bapps)  → B=8064 ≈ %.1f s\n", NMOM, NB, t, bapps, t*8064/Bt)
    end
end

# ---- joint2d-varying FP-boundary recheck: CUDA vs KA-CPU 2D, NMOM=1 varying ----
println("\n--- joint2d-varying CUDA vs KA-CPU count diff (expect few pairs) ---")
let Nj = 3000, Bj = 6, nd = 20, nv = 20
    for seed_shift in (0.0f0, 0.137f0)
        x_h = rand(FT, D, Nj, Bj) .+ seed_shift; u_h = randn(FT, D, Nj, Bj)
        db = collect(FT, range(0.05f0, 2.0f0, length = nd + 1))
        vb = collect(FT, range(-5f0, 5f0, length = nv + 1))
        ddig_c = GE._sf_batch_dist_digitizer(KA.CPU(), db); vpc = GE._gpu_build_value_digitize_plan(KA.CPU(), vb)
        ddig_g = GE._sf_batch_dist_digitizer(CUDA.CUDABackend(), db); vpg = GE._gpu_build_value_digitize_plan(CUDA.CUDABackend(), vb)
        oc = zeros(FT, 1, nd, nv, Bj); cc = zeros(UInt32, 1, nd, nv, Bj)
        GE._sf_launch_2d_batch!(KA.CPU(), oc, cc, x_h, u_h, sf2, ddig_c, vpc, Nj, nd, nv, Bj, D, Val(1), false, GEOM)
        KA.synchronize(KA.CPU())
        og = CUDA.zeros(FT, 1, nd, nv, Bj); cg = CUDA.zeros(UInt32, 1, nd, nv, Bj)
        GE._sf_launch_2d_batch!(CUDA.CUDABackend(), og, cg, CuArray(x_h), CuArray(u_h), sf2, ddig_g, vpg, Nj, nd, nv, Bj, D, Val(1), false, GEOM)
        CUDA.synchronize()
        dcnt = maximum(abs.(Int.(Array(cg)) .- Int.(cc)))
        tot = sum(cc)
        @printf("  seed_shift=%.3f: max|Δcount|=%d  total_pairs=%d  (%.1e fraction)\n", seed_shift, dcnt, tot, dcnt/tot)
    end
end
println("\nDONE_1D")
