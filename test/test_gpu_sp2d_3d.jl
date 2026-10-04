using Test: Test
using Random: Random
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC, HelperFunctions as SFH, LinearBinEdges

const GPU_SP2D_3D_CASES = ((2, 24, 12), (3, 16, 8))

# The device single-pass 2D histogram over two point tiles matches the CPU accumulation up to Float32 edge rounding.
Test.@testset "GPU single-pass 2D over two point tiles" begin
    Random.seed!(20260816)
    backend = KA.CPU()
    FT = Float32

    Test.@testset "D = $D, $(nd)x$(nv) bins" for (D, nd, nv) in GPU_SP2D_3D_CASES
        N = 200
        x = rand(FT, D, N)
        u = rand(FT, D, N)
        db = LinearBinEdges(range(FT(0), FT(1.5); length = nd + 1))
        vb = LinearBinEdges(range(FT(-1), FT(2); length = nv + 1))

        g = SFH.FlatGeometry{D}()
        gs, gc = SFC.gpu_calculate_structure_functions_single_pass_2d(backend, x, u, db, vb, UInt32; geometry = g)
        cs = zeros(FT, 6, nd, nv)
        cc = zeros(UInt32, 6, nd, nv)
        SFC._accumulate_single_pass_2d!(cs, cc, x, u, db, vb; geometry = g)

        gcm, ccm = Array(gc), cc
        Test.@test sum(Int.(gcm)) == sum(Int.(ccm))
        Test.@test maximum(abs.(Int.(gcm) .- Int.(ccm))) <= 1
        Test.@test count(gcm .!= ccm) <= max(4, length(ccm) ÷ 100)
        Test.@test isapprox(Array(gs), cs; rtol = 1e-4)
    end
end
