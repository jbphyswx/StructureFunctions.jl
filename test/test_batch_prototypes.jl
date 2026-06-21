# Phase 0 batch prototype parity (CPU + optional KA.CPU GPU kernel path)

using Test: Test, @test, @testset
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctionTypes as SFT, LinearBinEdges

const BP_DIR = joinpath(@__DIR__, "..", "gpu", "batch_prototypes")
include(joinpath(BP_DIR, "BatchPrototypes.jl"))
using .BatchPrototypes: BatchPrototypes as BP

Test.@testset "batch prototypes Phase 0" begin
    FT = Float32
    N = 48
    batch_shape = (8, 2)
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(1.5); length = 17))
    lp = BP.linear_bin_params(bin_edges)
    NB = lp.n_bins - 1

    x_fix, u_fix = BP.make_random_batch_problem(FT, N, batch_shape; fixed_x = true, seed = 1)
    x_var, u_var = BP.make_random_batch_problem(FT, N, batch_shape; fixed_x = false, seed = 2)

    backend = KA.CPU()

    results = BP.run_parity_suite(
        x_fix, u_fix, x_var, u_var, sft, bin_edges;
        backend = backend,
    )
    BP.print_parity_results(results)

    for r in results
        @test r.ok
        @test r.counts_equal
    end

    est = BP.estimate_batch_priv_bytes(N, prod(batch_shape), NB, FT)
    @test est.n_priv > 0
    @test est.partial_bytes > 0
    slabs = BP.batch_slab_ranges(prod(batch_shape), est.partial_bytes ÷ 2, N, NB, FT)
    @test length(slabs) >= 1
    @test sum(length, slabs) == prod(batch_shape)

    geo = BP.classify_batch_geometry(x_fix, u_fix)
    @test geo === BP.FixedX
    geo2 = BP.classify_batch_geometry(x_var, u_var)
    @test geo2 === BP.VaryingX

    sums = zeros(FT, NB, batch_shape...)
    counts = zeros(UInt32, NB, batch_shape...)
    @test size(sums)[2:end] == batch_shape
end

println("batch prototype tests passed.")
