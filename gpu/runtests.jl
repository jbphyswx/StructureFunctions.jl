"""
    runtests.jl

GPU test entry point. Skips cleanly when no functional CUDA device is present.

Run from the repository root:

    julia --project=gpu gpu/runtests.jl
"""

using CUDA: CUDA
using Test: Test

if !CUDA.functional()
    @warn "CUDA not functional — skipping GPU tests" CUDA_VISIBLE_DEVICES=get(ENV, "CUDA_VISIBLE_DEVICES", "unset")
    exit(0)
end

println("CUDA device: ", CUDA.name(CUDA.device()))

Test.@testset "StructureFunctions GPU" begin
    include("test_cuda_parity.jl")
end

println("GPU tests passed.")
