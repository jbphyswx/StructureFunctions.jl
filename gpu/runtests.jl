"""
    runtests.jl

Tier-2 CUDA tests — **not** part of default `Pkg.test()`. Skips cleanly when no
functional CUDA device is present.

Run from the repository root (GPU allocation / SLURM):

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
    include("test_workspace_cuda.jl")
end

println("GPU tests passed.")
