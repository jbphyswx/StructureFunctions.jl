# Parity matrix for production batch fast paths (KA.CPU).
using Test
using Random
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LinearBinEdges,
    batch_histograms_equal, batch_max_abs_diff, pair_from_linear
using StructureFunctions.Calculations:
    auxiliary_shared_positions!, auxiliary_varying_positions!, serial_calculate_structure_functions_single_pass!,
    serial_calculate_structure_functions_single_pass_2d!, auxiliary_joint2d!

Random.seed!(2025)

const SF_TYPE = SFT.L2SFType()
const CPU_BE = SFC.SerialBackend()
const GPU_BE = SFC.GPUBackend(KA.CPU())

function _rand_batch_fixed(N::Int, B::Int)
    FT = Float32
    x = rand(FT, 2, N)
    u = rand(FT, 2, N, B)
    edges = LinearBinEdges(collect(range(0.0f0, 1.5f0; length = 11)))
    return x, u, edges
end

function _rand_batch_varying(N::Int, B::Int)
    FT = Float32
    x = rand(FT, 2, N, B)
    u = rand(FT, 2, N, B)
    edges = LinearBinEdges(collect(range(0.0f0, 1.8f0; length = 11)))
    return x, u, edges
end

Test.@testset "batch matrix parity (KA.CPU)" begin
    N, B = 24, 3

    @testset "pair_from_linear large N" begin
        Nbig = 20_000
        total = Nbig * (Nbig - 1) ÷ 2
        for k in (1, 2, total ÷ 2, total - 1, total)
            i, j = pair_from_linear(k, Nbig)
            @test 1 <= i < j <= Nbig
        end
    end

    @testset "row1 individual 1D fixed-x" begin
        x, u, lbe = _rand_batch_fixed(N, B)
        @test ndims(x) == 2
        NB = length(lbe.edges) - 1
        cpu_s = zeros(Float32, NB, B)
        cpu_c = zeros(UInt32, NB, B)
        auxiliary_shared_positions!(cpu_s, cpu_c, x, u, SF_TYPE, lbe)

        gpu_out = SFC.calculate_structure_function(
            SF_TYPE, x, u, lbe;
            backend = GPU_BE, output_type = SF.StructureFunctionSumsAndCounts, verbose = false,
        )
        @test batch_histograms_equal(gpu_out.sums, gpu_out.counts, cpu_s, cpu_c; atol = 1f-4)
    end

    @testset "row1b individual 1D fixed-x B>strip (regression)" begin
        Nb, Bb = 24, 17
        x, u, lbe = _rand_batch_fixed(Nb, Bb)
        NB = length(lbe.edges) - 1
        cpu_s = zeros(Float32, NB, Bb)
        cpu_c = zeros(UInt32, NB, Bb)
        auxiliary_shared_positions!(cpu_s, cpu_c, x, u, SF_TYPE, lbe)
        gpu_out = SFC.calculate_structure_function(
            SF_TYPE, x, u, lbe;
            backend = GPU_BE, output_type = SF.StructureFunctionSumsAndCounts, verbose = false,
        )
        @test batch_histograms_equal(gpu_out.sums, gpu_out.counts, cpu_s, cpu_c; atol = 1f-4)
    end

    @testset "row2 individual 1D varying-x" begin
        x, u, lbe = _rand_batch_varying(N, B)
        @test ndims(x) == 3
        NB = length(lbe.edges) - 1
        cpu_s = zeros(Float32, NB, B)
        cpu_c = zeros(UInt32, NB, B)
        auxiliary_varying_positions!(cpu_s, cpu_c, x, u, SF_TYPE, lbe)

        gpu_s = zeros(Float32, NB, B)
        gpu_c = zeros(UInt32, NB, B)
        SFC.calculate_structure_function_slices!(
            gpu_s, gpu_c, SF_TYPE, x, u, lbe; backend = GPU_BE,
        )
        @test batch_histograms_equal(gpu_s, gpu_c, cpu_s, cpu_c; atol = 1f-4)
    end

    @testset "row3 SP1D fixed-x" begin
        x, u, lbe = _rand_batch_fixed(N, B)
        n_bins = length(lbe.edges) - 1
        ref_s = zeros(Float32, 6, n_bins, B)
        ref_c = zeros(UInt32, 6, n_bins, B)
        for b in 1:B
            SFC.calculate_structure_functions_single_pass!(
                @view(ref_s[:, :, b]), @view(ref_c[:, :, b]),
                x, u[:, :, b], lbe.edges; backend = CPU_BE,
            )
        end
        cpu_s = zeros(Float32, 6, n_bins, B)
        cpu_c = zeros(UInt32, 6, n_bins, B)
        serial_calculate_structure_functions_single_pass!(cpu_s, cpu_c, x, u, lbe)
        @test batch_histograms_equal(cpu_s, cpu_c, ref_s, ref_c)

        inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        gpu_sp = SFC.calculate_structure_functions_single_pass(
            x, u, lbe; backend = GPU_BE, output_type = SF.StructureFunctionSumsAndCounts,
        )
        for (t, k) in enumerate(inv)
            @test batch_histograms_equal(
                gpu_sp[k].sums, gpu_sp[k].counts, cpu_s[t, :, :], cpu_c[t, :, :]; atol = 1f-4,
            )
        end
    end

    @testset "row4 SP1D varying-x slices" begin
        x, u, lbe = _rand_batch_varying(N, B)
        n_bins = length(lbe.edges) - 1
        cpu_s = zeros(Float32, 6, n_bins, B)
        cpu_c = zeros(UInt32, 6, n_bins, B)
        serial_calculate_structure_functions_single_pass!(cpu_s, cpu_c, x, u, lbe)

        gpu_s = zeros(Float32, 6, n_bins, B)
        gpu_c = zeros(UInt32, 6, n_bins, B)
        SFC.calculate_structure_functions_single_pass_slices!(
            gpu_s, gpu_c, x, u, lbe; backend = GPU_BE,
        )
        @test batch_histograms_equal(gpu_s, gpu_c, cpu_s, cpu_c; atol = 1f-4)
    end

    @testset "row5 SP2D fixed-x" begin
        x, u, lbe = _rand_batch_fixed(N, B)
        val_edges = LinearBinEdges(collect(range(-1.0f0, 1.0f0; length = 9)))
        n_bins = length(lbe.edges) - 1
        n_val = length(val_edges.edges) - 1
        cpu_s = zeros(Float32, 6, n_bins, n_val, B)
        cpu_c = zeros(UInt32, 6, n_bins, n_val, B)
        serial_calculate_structure_functions_single_pass_2d!(cpu_s, cpu_c, x, u, lbe, val_edges)

        inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        gpu_sp = SFC.calculate_structure_functions_single_pass_2d(
            x, u, lbe, val_edges; backend = GPU_BE,
        )
        for (t, k) in enumerate(inv)
            @test batch_histograms_equal(
                gpu_sp[k].sums, gpu_sp[k].counts, cpu_s[t, :, :, :], cpu_c[t, :, :, :]; atol = 1f-4,
            )
        end
    end

    @testset "row6 SP2D varying-x slices" begin
        x, u, lbe = _rand_batch_varying(N, B)
        val_edges = LinearBinEdges(collect(range(-1.0f0, 1.0f0; length = 9)))
        n_bins = length(lbe.edges) - 1
        n_val = length(val_edges.edges) - 1
        cpu_s = zeros(Float32, 6, n_bins, n_val, B)
        cpu_c = zeros(UInt32, 6, n_bins, n_val, B)
        serial_calculate_structure_functions_single_pass_2d!(cpu_s, cpu_c, x, u, lbe, val_edges)

        gpu_s = zeros(Float32, 6, n_bins, n_val, B)
        gpu_c = zeros(UInt32, 6, n_bins, n_val, B)
        SFC.calculate_structure_functions_single_pass_2d_slices!(
            gpu_s, gpu_c, x, u, lbe, val_edges; backend = GPU_BE,
        )
        @test batch_histograms_equal(gpu_s, gpu_c, cpu_s, cpu_c; atol = 1f-4)
    end

    @testset "row7 joint 2D fixed-x" begin
        x, u, lbe = _rand_batch_fixed(N, B)
        val_edges = LinearBinEdges(collect(range(-0.5f0, 1.5f0; length = 9)))
        n_bins = length(lbe.edges) - 1
        n_val = length(val_edges.edges) - 1
        cpu_s = zeros(Float32, n_bins, n_val, B)
        cpu_c = zeros(UInt32, n_bins, n_val, B)
        auxiliary_joint2d!(cpu_s, cpu_c, SF_TYPE, x, u, lbe, val_edges)

        gpu_out = SFC.calculate_structure_function(
            SF_TYPE, x, u, lbe, val_edges; backend = GPU_BE,
        )
        @test batch_histograms_equal(gpu_out.sums, gpu_out.counts, cpu_s, cpu_c; atol = 1f-4)
    end

    @testset "row8 SP2D fixed-x production 50x50 bin grid (smoke)" begin
        Np, Bp = 32, 2
        x, u, lbe = _rand_batch_fixed(Np, Bp)
        val_edges = LinearBinEdges(collect(range(-1.0f0, 1.0f0; length = 51)))
        @test length(lbe.edges) - 1 == 10
        @test length(val_edges.edges) - 1 == 50
        n_bins = length(lbe.edges) - 1
        n_val = length(val_edges.edges) - 1
        cpu_s = zeros(Float32, 6, n_bins, n_val, Bp)
        cpu_c = zeros(UInt32, 6, n_bins, n_val, Bp)
        serial_calculate_structure_functions_single_pass_2d!(cpu_s, cpu_c, x, u, lbe, val_edges)

        inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        gpu_sp = SFC.calculate_structure_functions_single_pass_2d(
            x, u, lbe, val_edges; backend = GPU_BE,
        )
        for (t, k) in enumerate(inv)
            @test batch_histograms_equal(
                gpu_sp[k].sums, gpu_sp[k].counts, cpu_s[t, :, :, :], cpu_c[t, :, :, :]; atol = 1f-4,
            )
        end
    end
end
