using Test
using Random: Random
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC, LinearBinEdges, LogBinEdges

# D = 3 single-pass 2D on the tiled shared-histogram path. Before this was supported, D = 3 fell
# through to the global-atomic kernel with the general binary-search value plan and ran ~38x slower
# (383 ms vs 10 ms at N=20000, 32x16, A100). The tiled kernel is dimension-generic because the six
# invariants need only du_L and |du_T|² = |du|² - du_L², which requires no transverse basis vector.
Test.@testset "GPU single-pass 2D, D = 3" begin
    Random.seed!(20260816)
    backend = KA.CPU()
    FT = Float32

    @testset "matches the CPU reference for D = $D, $(nd)x$(nv) bins" for D in (2, 3),
                                                                          (nd, nv) in ((16, 8), (24, 12))
        N = 256
        x = rand(FT, D, N)
        u = rand(FT, D, N)
        db = LinearBinEdges(range(FT(0), FT(1.5); length = nd + 1))
        vb = LinearBinEdges(range(FT(-1), FT(2); length = nv + 1))

        gs, gc = SFC.gpu_calculate_structure_functions_single_pass_2d(backend, x, u, db, vb)
        cs = zeros(FT, 6, nd, nv)
        cc = zeros(UInt32, 6, nd, nv)
        SFC._accumulate_single_pass_2d!(cs, cc, x, u, db, vb)

        # Total counts are conserved exactly — no pair may be dropped or double-counted. Individual
        # cells may differ by one where a pair sits within an ulp of a bin edge and GPU FMA rounds
        # the other way; a systematic difference would be a real bug, a few boundary pairs are not.
        gcm, ccm = Array(gc), cc
        @test sum(Int.(gcm)) == sum(Int.(ccm))
        @test maximum(abs.(Int.(gcm) .- Int.(ccm))) <= 1
        @test count(gcm .!= ccm) <= max(4, length(ccm) ÷ 100)
        @test isapprox(Array(gs), cs; rtol = 1e-4)
    end

    @testset "log distance bins, D = 3" begin
        N, nd, nv = 256, 16, 8
        x = rand(FT, 3, N) .+ FT(0.5)
        u = rand(FT, 3, N)
        db = LogBinEdges(FT.(10 .^ range(-1.5, 0.3; length = nd + 1)))
        vb = LinearBinEdges(range(FT(-1), FT(2); length = nv + 1))

        gs, gc = SFC.gpu_calculate_structure_functions_single_pass_2d(backend, x, u, db, vb)
        cs = zeros(FT, 6, nd, nv)
        cc = zeros(UInt32, 6, nd, nv)
        SFC._accumulate_single_pass_2d!(cs, cc, x, u, db, vb)
        gcm = Array(gc)
        @test sum(Int.(gcm)) == sum(Int.(cc))
        @test maximum(abs.(Int.(gcm) .- Int.(cc))) <= 1
        @test isapprox(Array(gs), cs; rtol = 1e-4)
    end

    @testset "workspace reuse is stable in 3D" begin
        N, nd, nv = 256, 16, 8
        x = rand(FT, 3, N)
        u = rand(FT, 3, N)
        db = LinearBinEdges(range(FT(0), FT(1.5); length = nd + 1))
        vb = LinearBinEdges(range(FT(-1), FT(2); length = nv + 1))
        ws = SFC.GPUSFWorkspace(backend, db, vb; kind = :single_pass_2d)
        ref_s, ref_c = SFC.gpu_calculate_structure_functions_single_pass_2d(backend, x, u, db, vb)
        for _ in 1:3
            gs, gc = SFC.gpu_calculate_structure_functions_single_pass_2d(
                backend, x, u, db, vb; workspace = ws)
            @test Array(gc) == Array(ref_c)
            @test Array(gs) ≈ Array(ref_s)
        end
    end
end
