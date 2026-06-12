using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionObjects as SFO,
    StructureFunctionTypes as SFT
using OhMyThreads: OhMyThreads  # load extension for ThreadedBackend / AutoBackend when nthreads() > 1
using KernelAbstractions: KernelAbstractions as KA
using Test
using Random

"""Wide synthetic value-bin edges for unit tests only."""
function _synthetic_value_bins(n_bins::Int; pad_infinite::Bool = true)
    edges = collect(range(-1.0, 2.0, length = n_bins + 1))
    return pad_infinite ? vcat(-Inf, edges, Inf) : edges
end

function _synthetic_value_bins_by_type(n_bins::Int; pad_infinite::Bool = true)
    template = _synthetic_value_bins(n_bins; pad_infinite = pad_infinite)
    return [copy(template) for _ in 1:8]
end

Test.@testset "Single-Pass 2D Core Correctness & Parity" begin
    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5

    distance_bins = exp.(range(log(1000.0), log(50000.0), length = 6))
    value_bins_by_type = _synthetic_value_bins_by_type(10; pad_infinite = true)
    n_val = length(value_bins_by_type[1]) - 1
    n_bins = length(distance_bins) - 1

    sums_2d = zeros(Float64, 8, n_bins, n_val)
    counts_2d = zeros(Int64, 8, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_2d, counts_2d, x, u, distance_bins, value_bins_by_type;
        backend = SFC.SerialBackend(),
    )

    Test.@test size(sums_2d) == (8, n_bins, n_val)
    Test.@test size(counts_2d) == (8, n_bins, n_val)

    sft_types = [
        SFT.SecondOrderStructureFunctionType(),
        SFT.LongitudinalSecondOrderStructureFunctionType(),
        SFT.TransverseSecondOrderStructureFunctionType(),
        SFT.ThirdOrderStructureFunctionType(),
        SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
        SFT.DiagonalInconsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalConsistentThirdOrderStructureFunctionType(),
    ]

    x_tuple = (x[1, :], x[2, :])
    u_tuple = (u[1, :], u[2, :])
    # Slot 4 is δu_L·|δu|²; ThirdOrderStructureFunctionType accumulates |δu|³ instead.
    per_type_indices = (1, 2, 3, 5, 6, 7, 8)

    for t in per_type_indices
        sf2d = SFC.calculate_structure_function(
            sft_types[t],
            x_tuple,
            u_tuple,
            distance_bins,
            value_bins_by_type[t];
            backend = SFC.SerialBackend(),
            verbose = false,
            show_progress = false,
        )
        Test.@test sf2d isa SFO.StructureFunction2D
        Test.@test sums_2d[t, :, :] ≈ sf2d.sums
        Test.@test counts_2d[t, :, :] ≈ sf2d.counts
    end

    sums_1d, counts_1d = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.SerialBackend(),
    )

    for t in 1:8
        marg_sums = vec(dropdims(sum(sums_2d[t:t, :, :], dims = 3), dims = 1))
        marg_counts = vec(dropdims(sum(counts_2d[t:t, :, :], dims = 3), dims = 1))
        Test.@test marg_sums ≈ vec(sums_1d[t, :])
        Test.@test marg_counts == vec(counts_1d[t, :])
    end

    sums_10, counts_10 = SFC.ten_type_from_eight_2d(sums_2d, counts_2d, distance_bins)
    Test.@test sums_10 ≈ sums_1d
    Test.@test counts_10 == counts_1d

    fill!(sums_2d, 0.0)
    fill!(counts_2d, 0)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_2d, counts_2d, x, u, distance_bins, value_bins_by_type;
        backend = SFC.AutoBackend(),
    )
    t_sums, t_counts = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins_by_type;
        backend = SFC.AutoBackend(),
    )
    Test.@test t_sums ≈ sums_2d
    Test.@test t_counts == counts_2d
end

Test.@testset "Single-Pass 2D with Custom Distance Metric (Cityblock)" begin
    using Distances: Distances as DI

    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5

    distance_bins = exp.(range(log(1000.0), log(50000.0), length = 6))
    value_bins_by_type = _synthetic_value_bins_by_type(10; pad_infinite = true)
    n_val = length(value_bins_by_type[1]) - 1
    n_bins = length(distance_bins) - 1
    metric = DI.Cityblock()

    sums_2d = zeros(Float64, 8, n_bins, n_val)
    counts_2d = zeros(Int64, 8, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_2d, counts_2d, x, u, distance_bins, value_bins_by_type;
        backend = SFC.SerialBackend(),
        distance_metric = metric
    )

    sft_types = [
        SFT.SecondOrderStructureFunctionType(),
        SFT.LongitudinalSecondOrderStructureFunctionType(),
        SFT.TransverseSecondOrderStructureFunctionType(),
        SFT.ThirdOrderStructureFunctionType(),
        SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
        SFT.DiagonalInconsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalConsistentThirdOrderStructureFunctionType(),
    ]

    x_tuple = (x[1, :], x[2, :])
    u_tuple = (u[1, :], u[2, :])
    # Slot 4 is δu_L·|δu|²; ThirdOrderStructureFunctionType accumulates |δu|³ instead.
    per_type_indices = (1, 2, 3, 5, 6, 7, 8)

    for t in per_type_indices
        sf2d = SFC.calculate_structure_function(
            sft_types[t],
            x_tuple,
            u_tuple,
            distance_bins,
            value_bins_by_type[t];
            backend = SFC.SerialBackend(),
            distance_metric = metric,
            verbose = false,
            show_progress = false,
        )
        Test.@test sums_2d[t, :, :] ≈ sf2d.sums
        Test.@test counts_2d[t, :, :] ≈ sf2d.counts
    end

    sums_1d, counts_1d = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.SerialBackend(),
        distance_metric = metric
    )

    for t in 1:8
        marg_sums = vec(dropdims(sum(sums_2d[t:t, :, :], dims = 3), dims = 1))
        marg_counts = vec(dropdims(sum(counts_2d[t:t, :, :], dims = 3), dims = 1))
        Test.@test marg_sums ≈ vec(sums_1d[t, :])
        Test.@test marg_counts == vec(counts_1d[t, :])
    end

    # Compare ThreadedBackend with SerialBackend
    t_sums, t_counts = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins_by_type;
        backend = SFC.AutoBackend(),
        distance_metric = metric
    )
    Test.@test t_sums ≈ sums_2d
    Test.@test t_counts == counts_2d
end

Test.@testset "Single-Pass 2D GPU (KA.CPU) parity vs Serial" begin
    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5
    distance_bins = exp.(range(log(1000.0), log(50000.0), length = 6))
    value_bins_by_type = _synthetic_value_bins_by_type(10; pad_infinite = true)
    n_val = length(value_bins_by_type[1]) - 1
    n_bins = length(distance_bins) - 1

    sums_ref = zeros(Float64, 8, n_bins, n_val)
    counts_ref = zeros(Int64, 8, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, counts_ref, x, u, distance_bins, value_bins_by_type;
        backend = SFC.SerialBackend(),
    )

    sums_gpu, counts_gpu = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins_by_type;
        backend = SF.GPUBackend(KA.CPU()),
    )
    Test.@test sums_gpu ≈ sums_ref
    Test.@test counts_gpu == counts_ref

    sums_gpu2 = zeros(Float64, 8, n_bins, n_val)
    counts_gpu2 = zeros(Int64, 8, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu2, counts_gpu2, x, u, distance_bins, value_bins_by_type;
        backend = SF.GPUBackend(KA.CPU()),
    )
    Test.@test sums_gpu2 ≈ sums_ref
    Test.@test counts_gpu2 == counts_ref
end
