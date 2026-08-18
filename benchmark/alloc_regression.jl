using Test: Test
using BenchmarkTools: BenchmarkTools
using StructureFunctions: StructureFunctions as SF
using StaticArrays: StaticArrays as SA
using Random: Random
using JSON: JSON

const BASELINE_FILE = joinpath(@__DIR__, "benchmark_results", "alloc_baseline.json")

"""Set `SF_UPDATE_ALLOC_BASELINE=1` to overwrite the committed baseline with this run."""
const UPDATE_BASELINE = get(ENV, "SF_UPDATE_ALLOC_BASELINE", "0") == "1"

function load_baseline()
    isfile(BASELINE_FILE) || return Dict{String, Any}()
    try
        return JSON.parsefile(BASELINE_FILE)
    catch
        @warn "Unparseable baseline at $BASELINE_FILE; treating every case as new."
        return Dict{String, Any}()
    end
end

function save_baseline(results)
    mkpath(dirname(BASELINE_FILE))
    open(BASELINE_FILE, "w") do f
        JSON.print(f, results, 4)
    end
end

function run_sf_benchmark(x, u, bins, sf_type)
    return BenchmarkTools.@benchmark SF.calculate_structure_function(
        $sf_type,
        $x,
        $u,
        $bins;
        verbose = false,
        show_progress = false,
    ) seconds = 0.2 samples = 50
end

Test.@testset "Allocation regression" begin
    Random.seed!(42)
    N = 100

    x2, u2 = randn(2, N), randn(2, N)
    x3, u3 = randn(3, N), randn(3, N)
    bins = [0.0, 0.5, 1.0, 1.5, 2.0]

    baseline = load_baseline()
    current = Dict{String, Any}()

    test_matrix = [
        ("sf_2d_array_2nd_long", x2, u2, SF.LongitudinalSecondOrderStructureFunction),
        ("sf_3d_array_2nd_long", x3, u3, SF.LongitudinalSecondOrderStructureFunction),
        ("sf_2d_array_2nd_trans", x2, u2, SF.TransverseSecondOrderStructureFunction),
        ("sf_2d_array_3rd_diag", x2, u2, SF.DiagonalConsistentThirdOrderStructureFunction),
    ]

    for (name, x_in, u_in, sft) in test_matrix
        Test.@testset "$name" begin
            b = run_sf_benchmark(x_in, u_in, bins, sft)
            allocs = b.allocs
            current[name] = Dict("allocs" => allocs, "time_μs" => minimum(b.times) / 1e3)
            if haskey(baseline, name)
                Test.@test allocs <= baseline[name]["allocs"]
            else
                @info "$name: new case, recording $allocs allocs"
                Test.@test allocs < 100_000
            end
        end
    end

    Test.@testset "n̂ allocates nothing" begin
        r_hat = SA.SVector(1.0, 0.0)
        b = BenchmarkTools.@benchmark SF.HelperFunctions.n̂($r_hat)
        current["nhat_helper"] = Dict("allocs" => b.allocs, "time_μs" => minimum(b.times) / 1e3)
        Test.@test b.allocs == 0
    end

    # Overwriting unconditionally would bake a regression in as the new baseline.
    if UPDATE_BASELINE
        save_baseline(merge(baseline, current))
        @info "Baseline updated: $BASELINE_FILE"
    else
        @info "Baseline unchanged; set SF_UPDATE_ALLOC_BASELINE=1 to record this run."
    end
end
