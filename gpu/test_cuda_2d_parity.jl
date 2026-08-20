# =============================================================================
# Parity + timing for the CUDA fast 2D kernel (StructureFunctionsCUDAExt) vs the
# portable KA unified kernel run on KA.CPU() (the validated reference). Exercises
# the kernel through the same chokepoint the public API uses (_sf_launch_2d_batch!),
# for NMOM ∈ {1 (joint), 6 (single-pass)} × fixed/varying × bin sizes.
#   julia --project=gpu gpu/test_cuda_2d_parity.jl
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
const sf2 = SFT.SecondOrderStructureFunctionType()

# Reference: force the KA path on CPU (default hook returns false → KA unified).
function ka_cpu_2d(x, u, ddig_cpu, vplan_cpu, N, n_dist, n_val, B, NMOM, fixed_x)
    out = zeros(FT, NMOM, n_dist, n_val, B)
    cnt = zeros(UInt32, NMOM, n_dist, n_val, B)
    GE._sf_launch_2d_batch!(KA.CPU(), out, cnt, x, u, sf2, ddig_cpu, vplan_cpu,
                            N, n_dist, n_val, B, D, Val(NMOM), fixed_x)
    KA.synchronize(KA.CPU())
    return out, cnt
end

function cuda_2d(xd, ud, ddig, vplan, N, n_dist, n_val, B, NMOM, fixed_x)
    out = CUDA.zeros(FT, NMOM, n_dist, n_val, B)
    cnt = CUDA.zeros(UInt32, NMOM, n_dist, n_val, B)
    handled = SFC.gpu_fast_launch_2d_batch!(CUDA.CUDABackend(), out, cnt, xd, ud, sf2, ddig, vplan,
                                            N, n_dist, n_val, B, D, NMOM, fixed_x)
    CUDA.synchronize()
    return out, cnt, handled
end

println("CUDA 2D fast-kernel parity vs KA.CPU() reference — N=$N B=$B D=$D\n")
println("| NMOM | fixed_x | bins | handled | max relΔ sums | counts exact | TILE-fit |")
for NMOM in (1, 6)
    for fixed_x in (true, false)
        for (n_dist, n_val) in ((16, 8), (20, 20), (50, 50))
            x_h = fixed_x ? rand(FT, D, N) : rand(FT, D, N, B)
            u_h = randn(FT, D, N, B)
            dist_bins = collect(FT, range(0.05f0, 2.0f0, length = n_dist + 1))
            vb = collect(FT, range(-5.0f0, 5.0f0, length = n_val + 1))
            ddig_cpu = GE._sf_batch_dist_digitizer(KA.CPU(), dist_bins)
            vplan_cpu = GE._gpu_build_value_digitize_plan(KA.CPU(), vb)
            ddig = GE._sf_batch_dist_digitizer(CUDA.CUDABackend(), dist_bins)
            vplan = GE._gpu_build_value_digitize_plan(CUDA.CUDABackend(), vb)
            xd = fixed_x ? CuArray(x_h) : CuArray(x_h)
            ud = CuArray(u_h)

            o_ref, c_ref = ka_cpu_2d(x_h, u_h, ddig_cpu, vplan_cpu, N, n_dist, n_val, B, NMOM, fixed_x)
            o_cu, c_cu, handled = cuda_2d(xd, ud, ddig, vplan, N, n_dist, n_val, B, NMOM, fixed_x)
            oc = Array(o_cu); cc = Array(c_cu)
            rel = maximum(abs.(oc .- o_ref) ./ max.(abs.(o_ref), 1f-3))
            cexact = cc == c_ref
            @printf("| %d | %s | %dx%d | %s | %.2e | %s | %s |\n",
                    NMOM, fixed_x, n_dist, n_val, handled, rel, cexact, handled)
        end
    end
end

# ---- timing the user's headline case: SP2D 50x50, fixed-x, at this B ----
println("\n--- timing SP2D 50x50 fixed-x (extrapolate to B=8064) ---")
let n_dist = 50, n_val = 50, NMOM = 6, fixed_x = true
    x_h = rand(FT, D, N); u_h = randn(FT, D, N, B)
    dist_bins = collect(FT, range(0.05f0, 2.0f0, length = n_dist + 1))
    vb = collect(FT, range(-5.0f0, 5.0f0, length = n_val + 1))
    ddig = GE._sf_batch_dist_digitizer(CUDA.CUDABackend(), dist_bins)
    vplan = GE._gpu_build_value_digitize_plan(CUDA.CUDABackend(), vb)
    xd = CuArray(x_h); ud = CuArray(u_h)
    out = CUDA.zeros(FT, NMOM, n_dist, n_val, B); cnt = CUDA.zeros(UInt32, NMOM, n_dist, n_val, B)
    f() = (CUDA.fill!(out, 0f0); CUDA.fill!(cnt, UInt32(0));
           GE._sf_launch_2d_batch!(CUDA.CUDABackend(), out, cnt, xd, ud, sf2, ddig, vplan,
                                   N, n_dist, n_val, B, D, Val(6), true); CUDA.synchronize())
    f(); f(); ts = Float64[]; for _ in 1:5; t = time_ns(); f(); push!(ts, (time_ns()-t)/1e9); end
    t = median(ts); bapps = (N*(N-1)/2)*B/t/1e9
    @printf("  N=%d B=%d: %.3f s  (%.2f bapps)  → B=8064 ≈ %.1f s\n", N, B, t, bapps, t*8064/B)
end
println("\nDONE_PARITY")
