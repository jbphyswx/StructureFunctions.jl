using Test
using StructureFunctions
using StructureFunctions.HelperFunctions
using Random
using LinearAlgebra

# Extension-specific packages (loaded if available)
using JLD2
using NCDatasets
using CSV
using DataFrames
using Zarr
using HDF5
using DiskArrays

Random.seed!(42)

@testset "Real Data Extensions Verification" begin
    # 1. Setup synthetic data
    N = 200
    FT = Float64
    x = (rand(FT, N), rand(FT, N))
    u = (rand(FT, N), rand(FT, N))
    
    # Reference calculation (memory-based)
    bin_edges = range(0.0, stop=1.0, length=10)
    sft = LongitudinalSecondOrderStructureFunction
    ref_sf, _ = calculate_structure_function(x, u, bin_edges, sft)

    # Temporary directory for test files
    mktempdir() do tmpdir
        @testset "JLD2 Extension" begin
            fpath = joinpath(tmpdir, "test.jld2")
            jldsave(fpath; x=x, u=u)
            
            # Test direct loading
            sf, _ = calculate_structure_function(fpath, bin_edges, sft; x_key="x", u_key="u")
            @test sf ≈ ref_sf
        end

        @testset "CSV Extension" begin
            fpath = joinpath(tmpdir, "test.csv")
            df = DataFrame(x1=x[1], x2=x[2], u1=u[1], u2=u[2])
            CSV.write(fpath, df)
            
            sf, _ = calculate_structure_function(fpath, bin_edges, sft; 
                x_key=("x1", "x2"), u_key=("u1", "u2"))
            @test sf ≈ ref_sf
        end

        @testset "NetCDF Extension (Coordinate Expansion)" begin
            fpath = joinpath(tmpdir, "test.nc")
            # Create a 2D grid for NetCDF test
            nx, ny = 20, 10
            lon = collect(range(0.0, 1.0, length=nx))
            lat = collect(range(0.0, 1.0, length=ny))
            # u_grid: (nx, ny)
            u_grid = rand(nx, ny)
            v_grid = rand(nx, ny)
            
            Dataset(fpath, "c") do ds
                defDim(ds, "lon", nx)
                defDim(ds, "lat", ny)
                defVar(ds, "lon", lon, ("lon",))
                defVar(ds, "lat", lat, ("lat",))
                defVar(ds, "u", u_grid, ("lon", "lat"))
                defVar(ds, "v", v_grid, ("lon", "lat"))
            end

            # Reference calculation for the grid
            x_nc = (repeat(lon, 1, ny), repeat(lat', nx, 1))
            u_nc = (u_grid, v_grid)
            ref_sf_nc, _ = calculate_structure_function(flatten_data(x_nc), flatten_data(u_nc), bin_edges, sft)

            # Test extension with automatic expansion
            sf_nc, _ = calculate_structure_function(fpath, bin_edges, sft; 
                x_key=("lon", "lat"), u_key=("u", "v"))
            
            @test sf_nc ≈ ref_sf_nc
        end

        @testset "Zarr Extension (Group)" begin
            fpath = joinpath(tmpdir, "test.zarr")
            # Create a group and add arrays
            zg = zgroup(fpath)
            # Correct zcreate invocation for group: zcreate(type, group, name, dims...)
            z1 = zcreate(FT, zg, "x1", N)
            z1[:] = x[1]
            z2 = zcreate(FT, zg, "x2", N)
            z2[:] = x[2]
            zu1 = zcreate(FT, zg, "u1", N)
            zu1[:] = u[1]
            zu2 = zcreate(FT, zg, "u2", N)
            zu2[:] = u[2]
            
            sf, _ = calculate_structure_function(fpath, bin_edges, sft; 
                x_key=("x1", "x2"), u_key=("u1", "u2"))
            @test sf ≈ ref_sf
        end

        @testset "HDF5 Extension" begin
            fpath = joinpath(tmpdir, "test.h5")
            h5open(fpath, "w") do file
                file["x1"] = x[1]
                file["x2"] = x[2]
                file["u1"] = u[1]
                file["u2"] = u[2]
            end
            
            sf, _ = calculate_structure_function(fpath, bin_edges, sft; 
                x_key=("x1", "x2"), u_key=("u1", "u2"))
            @test sf ≈ ref_sf
        end

        @testset "NaN Handling" begin
            # Inject NaNs
            x_nan = (copy(x[1]), copy(x[2]))
            u_nan = (copy(u[1]), copy(u[2]))
            x_nan[1][1] = NaN
            u_nan[2][10] = NaN
            
            # Reference: manual removal
            x_mat = hcat(x_nan[1], x_nan[2])'
            u_mat = hcat(u_nan[1], u_nan[2])'
            x_c, u_c = remove_nans(x_mat, u_mat)
            ref_sf_nan, _ = calculate_structure_function(x_c, u_c, bin_edges, sft)
            
            # Verify JLD2 loading with NaN
            fpath = joinpath(tmpdir, "test_nan.jld2")
            jldsave(fpath; x=x_nan, u=u_nan)
            sf_nan, _ = calculate_structure_function(fpath, bin_edges, sft)
            
            @test sf_nan ≈ ref_sf_nan
            @test length(x_c[1, :]) == N - 2 # verifying 2 points dropped
        end
    end
end
