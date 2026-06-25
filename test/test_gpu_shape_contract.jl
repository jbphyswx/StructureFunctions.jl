using Test
using Random
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    Calculations as SFC,
    StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO,
    batch_histograms_equal

Random.seed!(20260621)

const GPU_SHAPE_BE = SFC.GPUBackend(KA.CPU())
const GPU_SHAPE_CPU_BE = SFC.SerialBackend()

function _gpu_shape_pairwise(sf, x, u, bins)
    return SFC.calculate_structure_function(
        sf, x, u, bins;
        backend = GPU_SHAPE_BE, output_type = SFO.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false,
    )
end

function _cpu_shape_pairwise(sf, x, u, bins)
    return SFC.calculate_structure_function(
        sf, x, u, bins;
        backend = GPU_SHAPE_CPU_BE, output_type = SFO.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false,
    )
end

function _assert_sums_counts_equal(gpu, cpu; atol = 1f-4)
    @test batch_histograms_equal(gpu.sums, gpu.counts, cpu.sums, cpu.counts; atol)
end

@testset "GPU public shape contract (KA.CPU)" begin
    sf = SFT.L2SFType()
    bins = collect(Float32, range(0.0f0, 1.75f0; length = 10))
    value_bins = collect(Float32, range(-0.1f0, 1.5f0; length = 8))
    n_bins = length(bins) - 1

    @testset "point fields use axis 1 as D" begin
        x2 = rand(Float32, 2, 10)
        u2 = rand(Float32, 2, 10)
        x3 = rand(Float32, 3, 10)
        u3 = rand(Float32, 3, 10)

        _assert_sums_counts_equal(_gpu_shape_pairwise(sf, x2, u2, bins), _cpu_shape_pairwise(sf, x2, u2, bins))
        _assert_sums_counts_equal(_gpu_shape_pairwise(sf, x3, u3, bins), _cpu_shape_pairwise(sf, x3, u3, bins))
    end

    @testset "shared-position auxiliary axes match explicit slices" begin
        x = rand(Float32, 2, 11)
        u = rand(Float32, 2, 11, 3, 2)

        gpu = _gpu_shape_pairwise(sf, x, u, bins)
        @test size(gpu.sums) == (n_bins, 3, 2)
        @test size(gpu.counts) == (n_bins, 3, 2)

        ref_sums = zeros(Float32, n_bins, 3, 2)
        ref_counts = zeros(UInt32, n_bins, 3, 2)
        for idx in CartesianIndices((3, 2))
            t, m = Tuple(idx)
            rt = _cpu_shape_pairwise(sf, x, @view(u[:, :, t, m]), bins)
            ref_sums[:, t, m] .= rt.sums
            ref_counts[:, t, m] .= rt.counts
        end
        @test batch_histograms_equal(gpu.sums, gpu.counts, ref_sums, ref_counts; atol = 1f-4)
    end

    @testset "varying-position auxiliary axes match explicit slices" begin
        x = rand(Float32, 2, 11, 3)
        u = rand(Float32, 2, 11, 3)

        gpu = _gpu_shape_pairwise(sf, x, u, bins)
        @test size(gpu.sums) == (n_bins, 3)
        @test size(gpu.counts) == (n_bins, 3)

        ref_sums = zeros(Float32, n_bins, 3)
        ref_counts = zeros(UInt32, n_bins, 3)
        for t in 1:3
            rt = _cpu_shape_pairwise(sf, @view(x[:, :, t]), @view(u[:, :, t]), bins)
            ref_sums[:, t] .= rt.sums
            ref_counts[:, t] .= rt.counts
        end
        @test batch_histograms_equal(gpu.sums, gpu.counts, ref_sums, ref_counts; atol = 1f-4)
    end

    @testset "joint 2D shared and varying auxiliary axes" begin
        x_shared = rand(Float32, 2, 9)
        u_shared = rand(Float32, 2, 9, 2)
        shared = SFC.calculate_structure_function(
            sf, x_shared, u_shared, bins, value_bins; backend = GPU_SHAPE_BE,
        )
        @test size(shared.sums) == (n_bins, length(value_bins) - 1, 2)

        x_varying = rand(Float32, 2, 9, 2)
        u_varying = rand(Float32, 2, 9, 2)
        varying = SFC.calculate_structure_function(
            sf, x_varying, u_varying, bins, value_bins; backend = GPU_SHAPE_BE,
        )
        @test size(varying.sums) == (n_bins, length(value_bins) - 1, 2)
    end

    @testset "single-pass auxiliary axes preserve public shape" begin
        x = rand(Float32, 2, 10)
        u = rand(Float32, 2, 10, 2, 3)

        gpu = SFC.calculate_structure_functions_single_pass(x, u, bins; backend = GPU_SHAPE_BE)
        cpu = SFC.calculate_structure_functions_single_pass(x, u, bins; backend = GPU_SHAPE_CPU_BE)
        @test size(gpu.sums) == (6, n_bins, 2, 3)
        @test batch_histograms_equal(gpu.sums, gpu.counts, cpu.sums, cpu.counts; atol = 1f-4)

        gpu2d = SFC.calculate_structure_functions_single_pass_2d(x, u, bins, value_bins; backend = GPU_SHAPE_BE)
        cpu2d = SFC.calculate_structure_functions_single_pass_2d(x, u, bins, value_bins; backend = GPU_SHAPE_CPU_BE)
        @test size(gpu2d.sums) == (6, n_bins, length(value_bins) - 1, 2, 3)
        @test batch_histograms_equal(gpu2d.sums, gpu2d.counts, cpu2d.sums, cpu2d.counts; atol = 1f-4)
    end

    @testset "invalid shapes fail before GPU launch" begin
        @test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(Float32, 1, 5), rand(Float32, 1, 5), bins;
            backend = GPU_SHAPE_BE, verbose = false, show_progress = false,
        )
        @test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(Float32, 2, 5), rand(Float32, 3, 5), bins;
            backend = GPU_SHAPE_BE, verbose = false, show_progress = false,
        )
        @test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(Float32, 2, 5, 2), rand(Float32, 2, 5, 3), bins;
            backend = GPU_SHAPE_BE, verbose = false, show_progress = false,
        )
        @test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(Float32, 2, 5, 2), rand(Float32, 2, 5), bins;
            backend = GPU_SHAPE_BE, verbose = false, show_progress = false,
        )
    end
end
