#!/usr/bin/env julia
"""
    test_batch_prototypes_cuda.jl

CUDA parity for Phase 0 batch prototypes (fixed-x and varying-x).
Run on a GPU node:

    julia --project=gpu gpu/test_batch_prototypes_cuda.jl
    N=512 BATCH=64 julia --project=gpu gpu/test_batch_prototypes_cuda.jl
"""

using CUDA: CUDA
using Test: @test, @testset
using StructureFunctions: StructureFunctionTypes as SFT, LinearBinEdges

include(joinpath(@__DIR__, "batch_prototypes", "BatchPrototypes.jl"))
using .BatchPrototypes: BatchPrototypes as BP

function main()
    CUDA.functional() || error("CUDA not functional")
    FT = Float32
    N = parse(Int, get(ENV, "N", "512"))
    batch_shape = BP.parse_batch_env((64,))
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))

    x_fix, u_fix = BP.make_random_batch_problem(FT, N, batch_shape; fixed_x = true, seed = 42)
    x_var, u_var = BP.make_random_batch_problem(FT, N, batch_shape; fixed_x = false, seed = 42)
    backend = CUDA.CUDABackend()

    println("prototype: ", BP.batch_prototype_variant())
    println("device: ", CUDA.name(CUDA.device()))
    println("N=$N  batch_shape=$batch_shape")

    @testset "cuda batch prototypes" begin
        results = BP.run_parity_suite(
            x_fix, u_fix, x_var, u_var, sft, bin_edges;
            backend = backend,
        )
        BP.print_parity_results(results)
        for r in results
            if !r.ok || !r.counts_equal
                # On failure, compare GPU vs cpu_batch (same one-pass reference).
                lp = BP.linear_bin_params(bin_edges)
                NB = lp.n_bins - 1
                bd = BP.batch_dims(u_var)
                if r.geometry === BP.VaryingX && occursin("gpu", r.name)
                    sums_cb = zeros(FT, NB, bd...)
                    counts_cb = zeros(UInt32, NB, bd...)
                    BP.cpu_batch_varying_x!(sums_cb, counts_cb, x_var, u_var, sft, bin_edges)
                    sums_g = zeros(FT, NB, bd...)
                    counts_g = zeros(UInt32, NB, bd...)
                    BP.gpu_batch_tiled_varying_x!(backend, sums_g, counts_g, x_var, u_var, sft, bin_edges)
                    ok2, d2, ce2 = BP.check_parity(sums_cb, counts_cb, sums_g, counts_g)
                    println("  gpu vs cpu_batch varying: ok=$ok2 max|Δ|=$d2 counts=$ce2")
                    println("  sum(counts) ref=$(sum(counts_cb)) gpu=$(sum(counts_g)) expected=$(N * (N - 1) ÷ 2)")
                end
            end
            @test r.ok
            @test r.counts_equal
        end
    end
    println("cuda batch prototype parity passed.")
    return nothing
end

main()
