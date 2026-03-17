using Test
using StructureFunctions
using StructureFunctions.Calculations
using StructureFunctions.StructureFunctionTypes
using StaticArrays
using Random

include("test_synthetic_data.jl")
using .SyntheticData

@testset "Phase 6: Structure Function E2E Suite" begin
    # Base Settings
    N_side = 16
    L = 10.0
    xs, ys, _ = generate_nonuniform_domain(N_side; R_mask=0.0, lat_range=(0.0, 5.0), lon_range=(0.0, 10.0))
    
    # Generate vector field (u, v)
    dk = 2π / L
    peaks_u = [(dk * 2, dk * 1, 1.0)]
    peaks_v = [(dk * 1, dk * 2, 0.5)]
    u = real.(generate_spectral_field(xs, ys; peaks=peaks_u))
    v = real.(generate_spectral_field(xs, ys; peaks=peaks_v))
    
    pos = (xs, ys)
    vals = (u, v)
    
    # Target binning
    r_bins = SVector{5}([(0.1 + (i-1)*0.4, 0.1 + i*0.4) for i in 1:5])

    @testset "Basic Execution" begin
        sf_type = SecondOrderStructureFunction
        @test_nowarn calculate_structure_function(sf_type, pos, vals, r_bins; verbose=false, show_progress=false)
    end

    @testset "Multi-field Execution" begin
        sf_type = LongitudinalSecondOrderStructureFunction
        # calculate_structure_function handles (u, v) as a vector field
        res, _ = calculate_structure_function(sf_type, pos, vals, r_bins; verbose=false, show_progress=false)
        @test length(res) == 5
        @test all(res .>= 0)
    end

    @testset "Mixed Precision (Float32)" begin
        xs32 = Float32.(xs)
        ys32 = Float32.(ys)
        u32 = Float32.(u)
        v32 = Float32.(v)
        # Use exact bin types
        r32 = SVector{5, Tuple{Float32, Float32}}([(Float32(b[1]), Float32(b[2])) for b in r_bins])
        
        sf64, _ = calculate_structure_function(SecondOrderStructureFunction, pos, vals, r_bins; verbose=false, show_progress=false)
        sf32, _ = calculate_structure_function(SecondOrderStructureFunction, (xs32, ys32), (u32, v32), r32; verbose=false, show_progress=false)
        
        @test sf32 ≈ Float32.(sf64) rtol=1e-5
    end
end
