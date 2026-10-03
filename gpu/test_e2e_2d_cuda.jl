# =============================================================================
# End-to-end validation of the 2D CUDA fast path THROUGH THE PUBLIC API on a
# real CUDABackend: SP2D (fixed + varying) and joint 2D (fixed + varying) vs the
# serial CPU reference, plus headline real-N (20000) SP2D 50x50 wall-clock.
#   julia --project=gpu gpu/test_e2e_2d_cuda.jl
# =============================================================================
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions
import KernelAbstractions as KA
using CUDA, Printf
using Statistics: median
using Random: Random
const SF = StructureFunctions
const SFC = SF.Calculations
const SFT = SF.StructureFunctionTypes
const SFH = SF.HelperFunctions
using StructureFunctions: LinearBinEdges
using StructureFunctions.Calculations:
    serial_calculate_structure_functions_single_pass_2d!, auxiliary_joint2d!
const FT = Float32
const GPU_BE = CB.GPUBackend(CUDA.CUDABackend())
const SF_TYPE = SFT.L2SFType()
const SP2D_INV = (:S2, :L2, :T2, :S3, :L3, :L1T2)

# A pair whose value rounds differently on the two backends lands in another value bin; the table reports those
# pairs and the largest sum difference against the largest sum.
scaled(a, b) = maximum(abs.(a .- b)) / maximum(abs, b)
moved(c, ref) = sum(abs.(Int64.(c) .- Int64.(ref))) ÷ 2
row(case, bins, s, m, total) = @printf("| %s | %s | %.2e | %d / %d = %.1e |\n", case, bins, s, m, total, m / max(total, 1))

# A keyed SP2D result `g` against a stacked (6, ...) reference `cs`, `cc`, over every invariant.
function _sp2d_report(case, bins, g, cs, cc)
    s = maximum(t -> scaled(g[SP2D_INV[t]].sums, cs[t, :, :, :]), 1:6)
    m = sum(t -> moved(g[SP2D_INV[t]].counts, cc[t, :, :, :]), 1:6)
    row(case, bins, s, m, Int(sum(Int64, cc)))
end

Random.seed!(20260926)
println("End-to-end 2D CUDA public-API parity vs serial CPU\n")
println("| case | bins | max |Δsum| / max |sum| | pairs in another value bin |")

# ---- SP2D fixed-x ----
let N = 1500, B = 4
    for (nd, nv) in ((16, 8), (50, 50))
        x = rand(FT, 2, N); u = randn(FT, 2, N, B)
        lbe = LinearBinEdges(0.0f0, 1.5f0, nd + 1)
        ve = LinearBinEdges(-1.0f0, 1.0f0, nv + 1)
        cs = zeros(FT, 6, nd, nv, B); cc = zeros(UInt32, 6, nd, nv, B)
        serial_calculate_structure_functions_single_pass_2d!(cs, cc, x, u, lbe, ve; geometry = SFH.FlatGeometry{2}())
        g = SF.to_host(SFC.calculate_structure_functions_single_pass_2d(x, u, lbe, ve; backend = GPU_BE))
        _sp2d_report("SP2D fixed", "$(nd)x$(nv)", g, cs, cc)
    end
end
# ---- SP2D varying-x ----
let N = 1500, B = 4
    for (nd, nv) in ((20, 20),)
        x = rand(FT, 2, N, B); u = randn(FT, 2, N, B)
        lbe = LinearBinEdges(0.0f0, 1.5f0, nd + 1)
        ve = LinearBinEdges(-1.0f0, 1.0f0, nv + 1)
        cs = zeros(FT, 6, nd, nv, B); cc = zeros(UInt32, 6, nd, nv, B)
        serial_calculate_structure_functions_single_pass_2d!(cs, cc, x, u, lbe, ve; geometry = SFH.FlatGeometry{2}())
        g = SF.to_host(SFC.calculate_structure_functions_single_pass_2d(x, u, lbe, ve; backend = GPU_BE))
        _sp2d_report("SP2D varying", "$(nd)x$(nv)", g, cs, cc)
    end
end
# ---- joint 2D fixed-x and varying-x ----
let N = 1500, B = 4
    x = rand(FT, 2, N); u = randn(FT, 2, N, B)
    lbe = LinearBinEdges(0.0f0, 1.5f0, 21)
    ve = LinearBinEdges(-0.5f0, 1.5f0, 21)
    cs = zeros(FT, 20, 20, B); cc = zeros(UInt32, 20, 20, B)
    auxiliary_joint2d!(cs, cc, SF_TYPE, x, u, lbe, ve; geometry = SFH.FlatGeometry{2}())
    g = SF.to_host(SFC.calculate_structure_function(SF_TYPE, x, u, lbe, ve; backend = GPU_BE))
    row("joint2d fixed", "20x20", scaled(g.sums, cs), moved(g.counts, cc), Int(sum(Int64, cc)))

    xv = rand(FT, 2, N, B)
    cs2 = zeros(FT, 20, 20, B); cc2 = zeros(UInt32, 20, 20, B)
    auxiliary_joint2d!(cs2, cc2, SF_TYPE, xv, u, lbe, ve; geometry = SFH.FlatGeometry{2}())
    g2 = SF.to_host(SFC.calculate_structure_function(SF_TYPE, xv, u, lbe, ve; backend = GPU_BE))
    row("joint2d varying", "20x20", scaled(g2.sums, cs2), moved(g2.counts, cc2), Int(sum(Int64, cc2)))
end

# ---- headline timing: SP2D 50x50 fixed-x at real N=20000 (public API wall-clock) ----
println("\n--- headline: SP2D 50x50 fixed-x, N=20000 (public API, wall-clock) ---")
let N = 20000, B = 64, nd = 50, nv = 50
    x = rand(FT, 2, N); u = randn(FT, 2, N, B)
    lbe = LinearBinEdges(0.0f0, 1.5f0, nd + 1)
    ve = LinearBinEdges(-1.0f0, 1.0f0, nv + 1)
    f() = CUDA.@sync SFC.calculate_structure_functions_single_pass_2d(x, u, lbe, ve; backend = GPU_BE)
    f(); f()
    ts = Float64[]; for _ in 1:3; t = time_ns(); f(); push!(ts, (time_ns()-t)/1e9); end
    t = median(ts); bapps = (N*(N-1)/2)*B/t/1e9
    @printf("  N=%d B=%d: %.3f s  (%.2f bapps)  → B=8064 ≈ %.0f s\n", N, B, t, bapps, t*8064/B)
end
println("\nDONE_E2E")
