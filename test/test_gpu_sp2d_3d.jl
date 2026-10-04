using Test: Test
using Random: Random
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC, HelperFunctions as SFH, LinearBinEdges, LogBinEdges

# The GPU single-pass 2D histogram for D = 2 and 3 matches the CPU accumulation.
Test.@testset "GPU single-pass 2D, D = 3" begin
    Random.seed!(20260816)
    backend = KA.CPU()
    FT = Float32

    Test.@testset "matches the CPU reference for D = $D, $(nd)x$(nv) bins" for D in (2, 3),
                                                                          (nd, nv) in ((16, 8), (24, 12))
        N = 256
        x = rand(FT, D, N)
        u = rand(FT, D, N)
        db = LinearBinEdges(range(FT(0), FT(1.5); length = nd + 1))
        vb = LinearBinEdges(range(FT(-1), FT(2); length = nv + 1))

        g = SFH.FlatGeometry{D}()
        gs, gc = SFC.gpu_calculate_structure_functions_single_pass_2d(backend, x, u, db, vb, UInt32; geometry = g)
        cs = zeros(FT, 6, nd, nv)
        cc = zeros(UInt32, 6, nd, nv)
        SFC._accumulate_single_pass_2d!(cs, cc, x, u, db, vb; geometry = g)

        # Total counts are conserved exactly — no pair may be dropped or double-counted. Individual
        # cells may differ by one where a pair sits within an ulp of a bin edge and GPU FMA rounds
        # the other way, so the bound is one pair per cell and the total is exact.
        gcm, ccm = Array(gc), cc
        Test.@test sum(Int.(gcm)) == sum(Int.(ccm))
        Test.@test maximum(abs.(Int.(gcm) .- Int.(ccm))) <= 1
        Test.@test count(gcm .!= ccm) <= max(4, length(ccm) ÷ 100)
        Test.@test isapprox(Array(gs), cs; rtol = 1e-4)
    end

    Test.@testset "log distance bins, D = 3" begin
        N, nd, nv = 256, 16, 8
        x = rand(FT, 3, N) .+ FT(0.5)
        u = rand(FT, 3, N)
        db = LogBinEdges(FT(10^-1.5), FT(10^0.3), nd + 1)
        vb = LinearBinEdges(range(FT(-1), FT(2); length = nv + 1))

        g = SFH.FlatGeometry{3}()
        gs, gc = SFC.gpu_calculate_structure_functions_single_pass_2d(backend, x, u, db, vb, UInt32; geometry = g)
        cs = zeros(FT, 6, nd, nv)
        cc = zeros(UInt32, 6, nd, nv)
        SFC._accumulate_single_pass_2d!(cs, cc, x, u, db, vb; geometry = g)
        gcm = Array(gc)
        Test.@test sum(Int.(gcm)) == sum(Int.(cc))
        Test.@test maximum(abs.(Int.(gcm) .- Int.(cc))) <= 1
        Test.@test isapprox(Array(gs), cs; rtol = 1e-4)
    end

    Test.@testset "workspace reuse is stable in 3D" begin
        N, nd, nv = 256, 16, 8
        x = rand(FT, 3, N)
        u = rand(FT, 3, N)
        db = LinearBinEdges(range(FT(0), FT(1.5); length = nd + 1))
        vb = LinearBinEdges(range(FT(-1), FT(2); length = nv + 1))
        ws = SFC.GPUSFWorkspace(backend, db, vb; kind = :single_pass_2d)
        g = SFH.FlatGeometry{3}()
        ref_s, ref_c = SFC.gpu_calculate_structure_functions_single_pass_2d(backend, x, u, db, vb, UInt32;
                                                                            geometry = g)
        for _ in 1:3
            gs, gc = SFC.gpu_calculate_structure_functions_single_pass_2d(
                backend, x, u, db, vb, UInt32; geometry = g, workspace = ws)
            Test.@test Array(gc) == Array(ref_c)
            Test.@test Array(gs) ≈ Array(ref_s)
        end
    end
end
