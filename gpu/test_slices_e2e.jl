# =============================================================================
# End-to-end validation of the fused SLICES (time-series) paths on a real
# CUDABackend vs the serial CPU reference: 1D individual, SP1D, joint2d, SP2D
# slices over (D,N,T). Confirms the unified-N-body rewiring of the per-slice
# loops is correct on GPU, plus timing.
#   julia --project=gpu gpu/test_slices_e2e.jl
# =============================================================================
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions
import KernelAbstractions as KA
using CUDA, Printf
using Statistics: median
const SF = StructureFunctions
const SFC = SF.Calculations
const SFT = SF.StructureFunctionTypes
using StructureFunctions: LinearBinEdges
using StructureFunctions.Calculations:
    serial_calculate_structure_functions_single_pass!,
    serial_calculate_structure_functions_single_pass_2d!,
    auxiliary_varying_positions!, auxiliary_joint2d!
const FT = Float32
const GPU_BE = CB.GPUBackend(CUDA.CUDABackend())
const sf2 = SFT.L2SFType()
maxrel(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1f-3))

println("Fused SLICES e2e on CUDA vs serial CPU\n")
println("| case | bins | max relΔ | counts exact |")
let N = 1500, T = 4
    x = rand(FT, 2, N, T); u = randn(FT, 2, N, T)
    lbe = LinearBinEdges(collect(range(0.05f0, 1.5f0; length = 17)))   # NB=16
    NB = 16
    # 1D individual slices
    cs = zeros(FT, NB, T); cc = zeros(UInt32, NB, T)
    auxiliary_varying_positions!(cs, cc, x, u, sf2, lbe)
    gs = zeros(FT, NB, T); gc = zeros(UInt32, NB, T)
    SFC.calculate_structure_function_batch!(gs, gc, sf2, x, u, lbe; backend = GPU_BE)
    @printf("| ind slices | NB=%d | %.2e | %s |\n", NB, maxrel(gs, cs), gc == cc)
    # SP1D slices
    cs1 = zeros(FT, 6, NB, T); cc1 = zeros(UInt32, 6, NB, T)
    serial_calculate_structure_functions_single_pass!(cs1, cc1, x, u, lbe)
    gs1 = zeros(FT, 6, NB, T); gc1 = zeros(UInt32, 6, NB, T)
    SFC.calculate_structure_functions_single_pass_batch!(gs1, gc1, x, u, lbe; backend = GPU_BE)
    @printf("| SP1D slices | NB=%d | %.2e | %s |\n", NB, maxrel(gs1, cs1), gc1 == cc1)
    # joint2d slices
    ve = LinearBinEdges(collect(range(-0.5f0, 1.5f0; length = 21)))
    nd = 16; nv = 20
    cs2 = zeros(FT, nd, nv, T); cc2 = zeros(UInt32, nd, nv, T)
    for b in 1:T
        @views auxiliary_joint2d!(cs2[:, :, b], cc2[:, :, b], sf2, x[:, :, b], u[:, :, b], lbe, ve)
    end
    gs2 = zeros(FT, nd, nv, T); gc2 = zeros(UInt32, nd, nv, T)
    SFC.calculate_structure_function_2d_batch!(gs2, gc2, sf2, x, u, lbe, ve; backend = GPU_BE)
    @printf("| joint2d slices | %dx%d | %.2e | %s |\n", nd, nv, maxrel(gs2, cs2), gc2 == cc2)
    # SP2D slices
    nd2 = 16; nv2 = 20
    cs3 = zeros(FT, 6, nd2, nv2, T); cc3 = zeros(UInt32, 6, nd2, nv2, T)
    serial_calculate_structure_functions_single_pass_2d!(cs3, cc3, x, u, lbe, ve)
    gs3 = zeros(FT, 6, nd2, nv2, T); gc3 = zeros(UInt32, 6, nd2, nv2, T)
    SFC.calculate_structure_functions_single_pass_2d_batch!(gs3, gc3, x, u, lbe, ve; backend = GPU_BE)
    @printf("| SP2D slices | %dx%d | %.2e | %s |\n", nd2, nv2, maxrel(gs3, cs3), gc3 == cc3)
end

println("\n--- slices timing N=20000, T=64 (wall-clock, public API) ---")
let N = 20000, T = 64
    x = rand(FT, 2, N, T); u = randn(FT, 2, N, T)
    lbe = LinearBinEdges(collect(range(0.05f0, 1.5f0; length = 51)))   # NB=50
    ve = LinearBinEdges(collect(range(-1.0f0, 1.0f0; length = 51)))    # nv=50
    gs3 = zeros(FT, 6, 50, 50, T); gc3 = zeros(UInt32, 6, 50, 50, T)
    f() = SFC.calculate_structure_functions_single_pass_2d_batch!(gs3, gc3, x, u, lbe, ve; backend = GPU_BE)
    f(); f(); ts = Float64[]; for _ in 1:3; t = time_ns(); f(); push!(ts, (time_ns()-t)/1e9); end
    t = median(ts)
    @printf("  SP2D 50x50 slices: %.3f s @ T=%d  → T=8064 ≈ %.0f s\n", t, T, t*8064/T)
end
println("\nDONE_SLICES")
