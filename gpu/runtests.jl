# Tier-2 CUDA tests, outside `Pkg.test()`: `julia --project=gpu gpu/runtests.jl` on a GPU allocation.
using CUDA: CUDA
using Test: Test

CUDA.functional() || error("CUDA is not functional (CUDA_VISIBLE_DEVICES = $(get(ENV, "CUDA_VISIBLE_DEVICES", "unset")))")

# One module per file; the shared-memory file runs first, as it reads the compiler's report of the kernels it compiles.
const FILES = (
    "test_cuda_smem_budget.jl",
    "test_cuda_parity.jl",
    "test_workspace_cuda.jl",
    "test_cuda_batch_contract.jl",
    "test_cuda_1d_parity.jl",
    "test_cuda_2d_parity.jl",
    "test_cuda_batch_widths.jl",
    "test_cuda_widths.jl",
    "test_cuda_pair_weights.jl",
    "test_cuda_second_axis.jl",
    "test_cuda_batch_culling.jl",
    "test_cuda_sorted_line.jl",
    "test_cuda_lag_sweep.jl",
    "test_cuda_gridded_parity.jl",
    "test_cuda_nufft.jl",
    "test_cuda_harmonic.jl",
    "test_cuda_postprocess.jl",
)

Test.@testset "StructureFunctions GPU" begin
    Test.@testset "$file" for file in FILES
        name, path = Symbol(first(splitext(file))), joinpath(@__DIR__, file)
        @eval Main module $name
            include($path)
        end
    end
end
