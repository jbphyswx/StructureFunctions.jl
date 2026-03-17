using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using StaticArrays: StaticArrays as SA
using Random: Random



Test.@testset "E2E: Structure Function E2E Suite" begin
    # Base Settings
    N_side = 16
    L = 10.0
    xs, ys, _ = SyntheticData.generate_nonuniform_domain(N_side; R_mask=0.0, lat_range=(0.0, 5.0), lon_range=(0.0, 10.0))
    
    # Generate vector field (u, v)
    dk = 2π / L
    peaks_u = [(dk * 2, dk * 1, 1.0)]
    peaks_v = [(dk * 1, dk * 2, 0.5)]
    u = real.(SyntheticData.generate_spectral_field(xs, ys; peaks=peaks_u))
    v = real.(SyntheticData.generate_spectral_field(xs, ys; peaks=peaks_v))
    
    pos = (xs, ys)
    vals = (u, v)
    
    # Target binning
    r_bins = SA.SVector{5}([(0.1 + (i-1)*0.4, 0.1 + i*0.4) for i in 1:5])

    Test.@testset "Basic Execution" begin
        sf_type = SFT.SecondOrderStructureFunction
        Test.@test_nowarn SFC.calculate_structure_function(sf_type, pos, vals, r_bins; verbose=false, show_progress=false)
    end

    Test.@testset "Multi-field Execution" begin
        sf_type = SFT.SecondOrderStructureFunction
        # calculate_structure_function now returns a StructureFunction object
        res = SFC.calculate_structure_function(sf_type, pos, vals, r_bins; verbose=false, show_progress=false)
        Test.@test length(res) == 5
        Test.@test all(res .>= 0)
    end

    Test.@testset "Mixed Precision (Float32)" begin
        xs32 = Float32.(xs)
        ys32 = Float32.(ys)
        u32 = Float32.(u)
        v32 = Float32.(v)
        # Use exact bin types
        r32 = SA.SVector{5, Tuple{Float32, Float32}}([(Float32(b[1]), Float32(b[2])) for b in r_bins])
        
        sf64 = SFC.calculate_structure_function(SFT.SecondOrderStructureFunction, pos, vals, r_bins; verbose=false, show_progress=false)
        sf32 = SFC.calculate_structure_function(SFT.SecondOrderStructureFunction, (xs32, ys32), (u32, v32), r32; verbose=false, show_progress=false)
        
        Test.@test sf32 ≈ Float32.(sf64) rtol=1e-5
    end
end
