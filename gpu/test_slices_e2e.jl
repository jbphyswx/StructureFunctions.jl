# =============================================================================
# End-to-end validation of the fused SLICES (time-series) paths on a real
# CUDABackend vs the serial CPU reference: 1D individual, SP1D, joint2d, SP2D
# slices over (D,N,T). Confirms the unified-N-body rewiring of the per-slice
# loops is correct on GPU, plus timing.
#   julia --project=gpu gpu/test_slices_e2e.jl
# =============================================================================
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH
import KernelAbstractions as KA
using CUDA: CUDA
using Printf: Printf
using Statistics: Statistics
using Random: Random
using StructureFunctions: LinearBinEdges
using StructureFunctions.Calculations:
    serial_calculate_structure_functions_single_pass!,
    serial_calculate_structure_functions_single_pass_2d!,
    auxiliary_varying_positions!, auxiliary_joint2d!
const FT = Float32
const GPU_BE = CB.GPUBackend(CUDA.CUDABackend())
const sf2 = SFT.L2SFType()
# A pair whose value rounds differently on the two backends lands in another value bin; the table reports those
# pairs and the largest sum difference against the largest sum.
scaled(a, b) = maximum(abs.(Array(a) .- b)) / maximum(abs, b)
moved(c, ref) = sum(abs.(Int64.(Array(c)) .- Int64.(ref))) ÷ 2
row(case, bins, s, m, total) =
    Printf.@printf("| %s | %s | %.2e | %d / %d = %.1e |\n", case, bins, s, m, total, m / max(total, 1))

Random.seed!(20260926)
println("Fused SLICES e2e on CUDA vs serial CPU\n")
println("| case | bins | max |Δsum| / max |sum| | pairs in another bin |")
let N = 1500, T = 4
    x = rand(FT, 2, N, T); u = randn(FT, 2, N, T)
    lbe = LinearBinEdges(0.05f0, 1.5f0, 17)   # NB=16
    NB = 16
    # 1D individual slices
    cs = zeros(FT, NB, T); cc = zeros(UInt32, NB, T)
    auxiliary_varying_positions!(cs, cc, x, u, sf2, lbe; geometry = SFH.FlatGeometry{2}())
    gs = CUDA.zeros(FT, NB, T); gc = CUDA.zeros(UInt32, NB, T)
    SFC.calculate_structure_function_batch!(gs, gc, sf2, x, u, lbe; backend = GPU_BE)
    row("ind slices", "NB=$NB", scaled(gs, cs), moved(gc, cc), Int(sum(Int64, cc)))
    # SP1D slices
    cs1 = zeros(FT, 6, NB, T); cc1 = zeros(UInt32, 6, NB, T)
    serial_calculate_structure_functions_single_pass!(cs1, cc1, x, u, lbe; geometry = SFH.FlatGeometry{2}())
    gs1 = CUDA.zeros(FT, 6, NB, T); gc1 = CUDA.zeros(UInt32, 6, NB, T)
    SFC.calculate_structure_functions_single_pass_batch!(gs1, gc1, x, u, lbe; backend = GPU_BE)
    row("SP1D slices", "NB=$NB", scaled(gs1, cs1), moved(gc1, cc1), Int(sum(Int64, cc1)))
    # joint2d slices
    ve = LinearBinEdges(-0.5f0, 1.5f0, 21)
    nd = 16; nv = 20
    cs2 = zeros(FT, nd, nv, T); cc2 = zeros(UInt32, nd, nv, T)
    for b in 1:T
        @views auxiliary_joint2d!(cs2[:, :, b], cc2[:, :, b], sf2, x[:, :, b], u[:, :, b], lbe, ve;
                                  geometry = SFH.FlatGeometry{2}())
    end
    gs2 = CUDA.zeros(FT, nd, nv, T); gc2 = CUDA.zeros(UInt32, nd, nv, T)
    SFC.calculate_structure_function_2d_batch!(gs2, gc2, sf2, x, u, lbe, ve; backend = GPU_BE)
    row("joint2d slices", "$(nd)x$(nv)", scaled(gs2, cs2), moved(gc2, cc2), Int(sum(Int64, cc2)))
    # SP2D slices
    nd2 = 16; nv2 = 20
    cs3 = zeros(FT, 6, nd2, nv2, T); cc3 = zeros(UInt32, 6, nd2, nv2, T)
    serial_calculate_structure_functions_single_pass_2d!(cs3, cc3, x, u, lbe, ve; geometry = SFH.FlatGeometry{2}())
    gs3 = CUDA.zeros(FT, 6, nd2, nv2, T); gc3 = CUDA.zeros(UInt32, 6, nd2, nv2, T)
    SFC.calculate_structure_functions_single_pass_2d_batch!(gs3, gc3, x, u, lbe, ve; backend = GPU_BE)
    row("SP2D slices", "$(nd2)x$(nv2)", scaled(gs3, cs3), moved(gc3, cc3), Int(sum(Int64, cc3)))
end

println("\n--- slices timing N=20000, T=64 (wall-clock, public API) ---")
let N = 20000, T = 64
    x = rand(FT, 2, N, T); u = randn(FT, 2, N, T)
    lbe = LinearBinEdges(0.05f0, 1.5f0, 51)   # NB=50
    ve = LinearBinEdges(-1.0f0, 1.0f0, 51)    # nv=50
    gs3 = CUDA.zeros(FT, 6, 50, 50, T); gc3 = CUDA.zeros(UInt32, 6, 50, 50, T)
    f() = CUDA.@sync SFC.calculate_structure_functions_single_pass_2d_batch!(gs3, gc3, x, u, lbe, ve; backend = GPU_BE)
    f(); f(); ts = Float64[]; for _ in 1:3; t = time_ns(); f(); push!(ts, (time_ns()-t)/1e9); end
    t = Statistics.median(ts)
    Printf.@printf("  SP2D 50x50 slices: %.3f s @ T=%d  → T=8064 ≈ %.0f s\n", t, T, t*8064/T)
end
println("\nDONE_SLICES")
