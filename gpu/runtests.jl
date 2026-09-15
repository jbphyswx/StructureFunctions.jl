"""
    runtests.jl

Tier-2 CUDA tests — **not** part of default `Pkg.test()`. Skips cleanly when no
functional CUDA device is present.

Run from the repository root (GPU allocation / SLURM):

    julia --project=gpu gpu/runtests.jl

The testset files run in this process; the script-style parity files each run in their own
process, so their top-level constants never collide, and pass when they exit cleanly.
"""

using CUDA: CUDA
using Test: Test

if !CUDA.functional()
    @warn "CUDA not functional — skipping GPU tests" CUDA_VISIBLE_DEVICES=get(ENV, "CUDA_VISIBLE_DEVICES", "unset")
    exit(0)
end

println("CUDA device: ", CUDA.name(CUDA.device()))

const GPU_DIR = @__DIR__
const SCRIPT_SUITES = (
    "test_cuda_1d_parity.jl",
    "test_cuda_2d_parity.jl",
    "test_e2e_2d_cuda.jl",
    "test_slices_e2e.jl",
    "test_cuda_gridded_parity.jl",
)

Test.@testset "StructureFunctions GPU" begin
    include("test_cuda_parity.jl")
    include("test_workspace_cuda.jl")
    Test.@testset "script suites" begin
        for file in SCRIPT_SUITES
            cmd = `$(Base.julia_cmd()) --project=$(GPU_DIR) --threads=$(Threads.nthreads()) $(joinpath(GPU_DIR, file))`
            Test.@testset "$file" begin
                Test.@test success(pipeline(cmd; stdout, stderr))
            end
        end
    end
end

println("GPU tests passed.")
