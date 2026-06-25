# =============================================================================
# End-to-end validation of the 2D CUDA fast path THROUGH THE PUBLIC API on a
# real CUDABackend: SP2D (fixed + varying) and joint 2D (fixed + varying) vs the
# serial CPU reference, plus headline real-N (20000) SP2D 50x50 wall-clock.
#   julia --project=gpu gpu/test_e2e_2d_cuda.jl
# =============================================================================
using StructureFunctions
import KernelAbstractions as KA
using CUDA, Printf
using Statistics: median
const SF = StructureFunctions
const SFC = SF.Calculations
const SFT = SF.StructureFunctionTypes
using StructureFunctions: LinearBinEdges
using StructureFunctions.Calculations:
    serial_calculate_structure_functions_single_pass_2d!, auxiliary_joint2d!
const FT = Float32
const GPU_BE = SFC.GPUBackend(CUDA.CUDABackend())
const SF_TYPE = SFT.L2SFType()
const SP2D_INV = (:S2, :L2, :T2, :S3, :L3, :L1T2)

maxrel(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1f-3))

# Max relative error of a keyed SP2D result `g` against a stacked (6, ...) ref `cs`,
# and whether all per-invariant counts match the corresponding ref slice `cc`.
function _sp2d_maxrel_counts(g, cs, cc)
    mr = 0.0
    counts_ok = true
    for (t, k) in enumerate(SP2D_INV)
        mr = max(mr, maxrel(g[k].sums, cs[t, :, :, :]))
        counts_ok &= g[k].counts == cc[t, :, :, :]
    end
    return mr, counts_ok
end

println("End-to-end 2D CUDA public-API parity vs serial CPU\n")
println("| case | bins | max relΔ | counts exact |")

# ---- SP2D fixed-x ----
let N = 1500, B = 4
    for (nd, nv) in ((16, 8), (50, 50))
        x = rand(FT, 2, N); u = randn(FT, 2, N, B)
        lbe = LinearBinEdges(collect(range(0.0f0, 1.5f0; length = nd + 1)))
        ve = LinearBinEdges(collect(range(-1.0f0, 1.0f0; length = nv + 1)))
        cs = zeros(FT, 6, nd, nv, B); cc = zeros(UInt32, 6, nd, nv, B)
        serial_calculate_structure_functions_single_pass_2d!(cs, cc, x, u, lbe, ve)
        g = SFC.calculate_structure_functions_single_pass_2d(x, u, lbe, ve; backend = GPU_BE)
        mr, counts_ok = _sp2d_maxrel_counts(g, cs, cc)
        @printf("| SP2D fixed | %dx%d | %.2e | %s |\n", nd, nv, mr, counts_ok)
    end
end
# ---- SP2D varying-x ----
let N = 1500, B = 4
    for (nd, nv) in ((20, 20),)
        x = rand(FT, 2, N, B); u = randn(FT, 2, N, B)
        lbe = LinearBinEdges(collect(range(0.0f0, 1.5f0; length = nd + 1)))
        ve = LinearBinEdges(collect(range(-1.0f0, 1.0f0; length = nv + 1)))
        cs = zeros(FT, 6, nd, nv, B); cc = zeros(UInt32, 6, nd, nv, B)
        serial_calculate_structure_functions_single_pass_2d!(cs, cc, x, u, lbe, ve)
        g = SFC.calculate_structure_functions_single_pass_2d(x, u, lbe, ve; backend = GPU_BE)
        mr, counts_ok = _sp2d_maxrel_counts(g, cs, cc)
        @printf("| SP2D varying | %dx%d | %.2e | %s |\n", nd, nv, mr, counts_ok)
    end
end
# ---- joint 2D fixed-x and varying-x ----
let N = 1500, B = 4
    x = rand(FT, 2, N); u = randn(FT, 2, N, B)
    lbe = LinearBinEdges(collect(range(0.0f0, 1.5f0; length = 21)))
    ve = LinearBinEdges(collect(range(-0.5f0, 1.5f0; length = 21)))
    cs = zeros(FT, 20, 20, B); cc = zeros(UInt32, 20, 20, B)
    auxiliary_joint2d!(cs, cc, SF_TYPE, x, u, lbe, ve)
    g = SFC.calculate_structure_function(SF_TYPE, x, u, lbe, ve; backend = GPU_BE)
    @printf("| joint2d fixed | 20x20 | %.2e | %s |\n", maxrel(g.sums, cs), g.counts == cc)

    xv = rand(FT, 2, N, B)
    cs2 = zeros(FT, 20, 20, B); cc2 = zeros(UInt32, 20, 20, B)
    auxiliary_joint2d!(cs2, cc2, SF_TYPE, xv, u, lbe, ve)
    g2 = SFC.calculate_structure_function(SF_TYPE, xv, u, lbe, ve; backend = GPU_BE)
    @printf("| joint2d varying | 20x20 | %.2e | %s |\n", maxrel(g2.sums, cs2), g2.counts == cc2)
end

# ---- headline timing: SP2D 50x50 fixed-x at real N=20000 (public API wall-clock) ----
println("\n--- headline: SP2D 50x50 fixed-x, N=20000 (public API, wall-clock) ---")
let N = 20000, B = 64, nd = 50, nv = 50
    x = rand(FT, 2, N); u = randn(FT, 2, N, B)
    lbe = LinearBinEdges(collect(range(0.0f0, 1.5f0; length = nd + 1)))
    ve = LinearBinEdges(collect(range(-1.0f0, 1.0f0; length = nv + 1)))
    f() = SFC.calculate_structure_functions_single_pass_2d(x, u, lbe, ve; backend = GPU_BE)
    f(); f()
    ts = Float64[]; for _ in 1:3; t = time_ns(); f(); push!(ts, (time_ns()-t)/1e9); end
    t = median(ts); bapps = (N*(N-1)/2)*B/t/1e9
    @printf("  N=%d B=%d: %.3f s  (%.2f bapps)  → B=8064 ≈ %.0f s\n", N, B, t, bapps, t*8064/B)
end
println("\nDONE_E2E")
