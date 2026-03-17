using Test: Test
using StructureFunctions: StructureFunctions as SF, HelperFunctions as SFH, StructureFunctionTypes as SFT, Calculations as SFC
using Random: Random
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA

# Extension-specific packages
using JLD2: JLD2
using NCDatasets: NCDatasets as NC
using CSV: CSV
using DataFrames: DataFrame
using Zarr: Zarr
using HDF5: HDF5

Random.seed!(42)

Test.@testset "Real Data Extensions Verification" begin
    # 1. Setup synthetic data
    N = 200
    FT = Float64
    x = (rand(FT, N), rand(FT, N))
    u = (rand(FT, N), rand(FT, N))
    
    # Reference calculation (memory-based)
    bin_edges = collect(range(0.0, stop=1.0, length=11))
    bin_tuples = SA.SVector{10, Tuple{FT, FT}}([(bin_edges[i], bin_edges[i+1]) for i in 1:length(bin_edges)-1]...)
    sft = SFT.L2SFType()
    
    ref_sf = SFC.calculate_structure_function(sft, x, u, bin_tuples; verbose=false, show_progress=false)

    # Temporary directory for test files
    mktempdir() do tmpdir
        Test.@testset "JLD2 Extension" begin
            fpath = joinpath(tmpdir, "test.jld2")
            JLD2.jldsave(fpath; x=x, u=u)
            
            # Test direct loading
            sf = SFC.calculate_structure_function(sft, fpath, bin_tuples; x_key="x", u_key="u", verbose=false, show_progress=false)
            Test.@test sf.values ≈ ref_sf.values
        end

        Test.@testset "CSV Extension" begin
            fpath = joinpath(tmpdir, "test.csv")
            df = DataFrame(x1=x[1], x2=x[2], u1=u[1], u2=u[2])
            CSV.write(fpath, df)
            
            sf = SFC.calculate_structure_function(sft, fpath, bin_tuples; 
                x_key=("x1", "x2"), u_key=("u1", "u2"), verbose=false, show_progress=false)
            Test.@test sf.values ≈ ref_sf.values
        end

        Test.@testset "NetCDF Extension" begin
            fpath = joinpath(tmpdir, "test.nc")
            # Create a 2D grid for NetCDF test
            nx, ny = 20, 10
            lon = collect(range(0.0, 1.0, length=nx))
            lat = collect(range(0.0, 1.0, length=ny))
            u_grid = rand(nx, ny)
            v_grid = rand(nx, ny)
            
            NC.Dataset(fpath, "c") do ds
                NC.defDim(ds, "lon", nx)
                NC.defDim(ds, "lat", ny)
                NC.defVar(ds, "lon", lon, ("lon",))
                NC.defVar(ds, "lat", lat, ("lat",))
                NC.defVar(ds, "u", u_grid, ("lon", "lat"))
                NC.defVar(ds, "v", v_grid, ("lon", "lat"))
            end

            # Reference calculation for the grid
            x_nc = (repeat(lon, 1, ny), repeat(lat', nx, 1))
            u_nc = (u_grid, v_grid)
            ref_sf_nc = SFC.calculate_structure_function(sft, SFH.flatten_data(x_nc), SFH.flatten_data(u_nc), bin_tuples; verbose=false, show_progress=false)

            # Test extension with automatic expansion
            sf_nc = SFC.calculate_structure_function(sft, fpath, bin_tuples; 
                x_key=("lon", "lat"), u_key=("u", "v"), verbose=false, show_progress=false)
            
            Test.@test sf_nc.values ≈ ref_sf_nc.values
        end

        Test.@testset "Zarr Extension (Group)" begin
            fpath = joinpath(tmpdir, "test.zarr")
            zg = Zarr.zgroup(fpath)
            z1 = Zarr.zcreate(FT, zg, "x1", N)
            z1[:] = x[1]
            z2 = Zarr.zcreate(FT, zg, "x2", N)
            z2[:] = x[2]
            zu1 = Zarr.zcreate(FT, zg, "u1", N)
            zu1[:] = u[1]
            zu2 = Zarr.zcreate(FT, zg, "u2", N)
            zu2[:] = u[2]
            
            sf = SFC.calculate_structure_function(sft, fpath, bin_tuples; 
                x_key=("x1", "x2"), u_key=("u1", "u2"), verbose=false, show_progress=false)
            Test.@test sf.values ≈ ref_sf.values
        end

        Test.@testset "HDF5 Extension" begin
            fpath = joinpath(tmpdir, "test.h5")
            HDF5.h5open(fpath, "w") do file
                file["x1"] = x[1]
                file["x2"] = x[2]
                file["u1"] = u[1]
                file["u2"] = u[2]
            end
            
            sf = SFC.calculate_structure_function(sft, fpath, bin_tuples; 
                x_key=("x1", "x2"), u_key=("u1", "u2"), verbose=false, show_progress=false)
            Test.@test sf.values ≈ ref_sf.values
        end

        Test.@testset "NaN Handling" begin
            x_nan = (copy(x[1]), copy(x[2]))
            u_nan = (copy(u[1]), copy(u[2]))
            x_nan[1][1] = NaN
            u_nan[2][10] = NaN
            
            # Use flattened vectors directly
            x_mat = hcat(x_nan[1], x_nan[2])'
            u_mat = hcat(u_nan[1], u_nan[2])'
            x_c, u_c = SFH.remove_nans(x_mat, u_mat)
            ref_sf_nan = SFC.calculate_structure_function(sft, x_c, u_c, bin_tuples; verbose=false, show_progress=false)
            
            # Verify JLD2 loading with NaN
            fpath = joinpath(tmpdir, "test_nan.jld2")
            JLD2.jldsave(fpath; x=x_nan, u=u_nan)
            sf_nan = SFC.calculate_structure_function(sft, fpath, bin_tuples; verbose=false, show_progress=false)
            
            Test.@test sf_nan.values ≈ ref_sf_nan.values
        end
    end
end
