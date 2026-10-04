using Test: Test
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, HelperFunctions as SFH
using Random: Random

Random.seed!(42)

# Fresh buffers, a workspace, repeated accumulation and the slice batch each give the serial answer.
Test.@testset "CUDA GPUSFWorkspace & slices" begin
    Test.@test CUDA.functional()

    backend = CUDA.CUDABackend()
    N = 500
    T = 4
    FT = Float32
    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)
    bins = collect(FT, range(0.0f0, 1.4f0; length = 11))
    NB = length(bins) - 1
    sft = SFT.L2SFType()

    ref = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, bins, SFO.StructureFunctionSumsAndCounts,
    )

    res_fresh = SFC.gpu_calculate_structure_function(
        sft, backend, x_gpu, u_gpu, bins, UInt32; geometry = SFH.FlatGeometry{2}(),
    )
    CUDA.synchronize()
    Test.@test Array(res_fresh.counts) == ref.counts
    max_Δ_fresh = maximum(abs, Array(res_fresh.sums) .- ref.sums)
    Test.@test max_Δ_fresh < 0.05f0

    ws = SFC.GPUSFWorkspace(backend, bins)
    res_ws = SFC.gpu_calculate_structure_function(
        sft, backend, x_gpu, u_gpu, bins, UInt32; workspace = ws, geometry = SFH.FlatGeometry{2}(),
    )
    CUDA.synchronize()
    Test.@test Array(res_ws.counts) == ref.counts
    max_Δ_ws = maximum(abs, Array(res_ws.sums) .- ref.sums)
    Test.@test max_Δ_ws < 0.05f0

    sums_acc = CUDA.zeros(Float64, NB)
    counts_acc = CUDA.zeros(UInt32, NB)
    for _ in 1:3
        SFC.gpu_calculate_structure_function!(
            sums_acc, counts_acc, sft, backend, x_gpu, u_gpu, bins; workspace = ws, geometry = SFH.FlatGeometry{2}(),
        )
    end
    CUDA.synchronize()
    Test.@test Array(counts_acc) == 3 .* ref.counts
    max_Δ_acc = maximum(abs, Array(sums_acc) .- 3 .* ref.sums)
    Test.@test max_Δ_acc < 0.15f0

    x_batch_cpu = rand(FT, 2, N, T)
    u_batch_cpu = rand(FT, 2, N, T)
    x_batch = CUDA.cu(x_batch_cpu)
    u_batch = CUDA.cu(u_batch_cpu)

    sums_ref = zeros(Float64, NB, T)
    counts_ref = zeros(UInt32, NB, T)
    for t in 1:T
        ref_t = SFC.gpu_calculate_structure_function(
            sft, backend, x_batch_cpu[:, :, t], u_batch_cpu[:, :, t], bins, UInt32; geometry = SFH.FlatGeometry{2}(),
        )
        CUDA.synchronize()
        sums_ref[:, t] .= Array(ref_t.sums)
        counts_ref[:, t] .= Array(ref_t.counts)
    end

    sums_drv = CUDA.zeros(FT, NB, T)
    counts_drv = CUDA.zeros(UInt32, NB, T)
    ws_slice = SFC.GPUSFWorkspace(backend, bins)
    SFC.gpu_calculate_structure_function_batch!(
        sums_drv, counts_drv, sft, backend, x_batch, u_batch, bins;
        workspace = ws_slice, geometry = SFH.FlatGeometry{2}(),
    )
    CUDA.synchronize()
    max_Δ_slice = maximum(abs, Array(sums_drv) .- sums_ref)
    Test.@test max_Δ_slice < 0.05f0
    Test.@test Array(counts_drv) == counts_ref

    SFC.release!(ws)
    SFC.release!(ws_slice)
end
