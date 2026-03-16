using Test
using BenchmarkTools
using StructureFunctions
using StaticArrays: SVector
using Random: Random
using JSON

# Benchmark Metadata
const RESULTS_FILE = joinpath(@__DIR__, "bench_results.json")

function load_previous_results()
    if isfile(RESULTS_FILE)
        try
            return JSON.parsefile(RESULTS_FILE)
        catch
            @warn "Failed to parse existing results file, starting fresh."
            return Dict()
        end
    end
    return Dict()
end

function save_results(results)
    open(RESULTS_FILE, "w") do f
        JSON.print(f, results, 4)
    end
end

function run_sf_benchmark(x, u, bins, sf_type; threaded=false)
    # Cap benchmarks to 0.2s to keep the test suite under 10s
    if threaded
        return @benchmark calculate_structure_function($x, $u, $bins, $sf_type; verbose=false, show_progress=false) seconds=0.2 samples=50
    else
        return @benchmark calculate_structure_function($x, $u, $bins, $sf_type; verbose=false, show_progress=false) seconds=0.2 samples=50
    end
end

@testset "Phase 7 Performance Regression & Comprehensive Tracking" begin
    Random.seed!(42)
    N = 100
    
    # 1D Setup
    x1 = (randn(N),)
    u1 = (randn(N),)
    # 2D Setup
    x2 = (randn(N), randn(N))
    u2 = (randn(N), randn(N))
    # 3D Setup
    x3 = (randn(N), randn(N), randn(N))
    u3 = (randn(N), randn(N), randn(N))
    
    # Matrix version of 2D
    xm = randn(2, N)
    um = randn(2, N)
    
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    
    sf_2nd_long = LongitudinalSecondOrderStructureFunction()
    sf_2nd_trans = TransverseSecondOrderStructureFunction()
    sf_3rd_diag = DiagonalConsistentThirdOrderStructureFunction()
    
    previous_results = load_previous_results()
    current_results = Dict()
    
    @info "Running Comprehensive Benchmarks..."

    # Matrix of tests
    test_matrix = [
        ("sf_2d_tuple_2nd_long", x2, u2, sf_2nd_long),
        ("sf_1d_tuple_2nd_long", x1, u1, sf_2nd_long),
        ("sf_3d_tuple_2nd_long", x3, u3, sf_2nd_long),
        ("sf_2d_matrix_2nd_long", xm, um, sf_2nd_long),
        ("sf_2d_tuple_2nd_trans", x2, u2, sf_2nd_trans),
        ("sf_2d_tuple_3rd_diag", x2, u2, sf_3rd_diag),
    ]

    for (name, x_in, u_in, sft) in test_matrix
        @testset "$name" begin
            b = run_sf_benchmark(x_in, u_in, bins, sft)
            allocs = b.allocs
            time_min = minimum(b.times) / 1e3 # μs
            
            current_results[name] = Dict("allocs" => allocs, "time_μs" => time_min)
            @info "$name: $allocs allocs, $time_min μs"
            
            if haskey(previous_results, name)
                prev = previous_results[name]
                @test allocs <= prev["allocs"]
            else
                @info "  New baseline recorded for $name."
                # Initial sanity gates
                @test allocs < 100000 
            end
        end
    end

    @testset "Spectral Analysis API" begin
        ms = (16, 16)
        b_direct = @benchmark calculate_spectrum(DirectSumBackend(), $x2, $u2, $ms; domain_size=(10.0, 10.0)) seconds=0.2 samples=10
        
        allocs = b_direct.allocs
        time_min = minimum(b_direct.times) / 1e3 # μs
        
        current_results["direct_sum_spectral"] = Dict("allocs" => allocs, "time_μs" => time_min)
        @info "DirectSum Spectral: $allocs allocs, $time_min μs"

        if haskey(previous_results, "direct_sum_spectral")
            prev = previous_results["direct_sum_spectral"]
            @test allocs <= prev["allocs"]
        end
    end

    @testset "Static Constructor Efficiency" begin
        using StructureFunctions.HelperFunctions: n̂
        r_hat = SVector(1.0, 0.0)
        b_nhat = @benchmark n̂($r_hat)
        
        allocs = b_nhat.allocs
        current_results["nhat_helper"] = Dict("allocs" => allocs, "time_μs" => minimum(b_nhat.times)/1e3)
        @info "HelperFunctions.n̂: $allocs allocs"
        
        @test allocs == 0
    end

    # Save current results as the new baseline
    @info "Saving comprehensive benchmarks to $RESULTS_FILE"
    save_results(current_results)
end
