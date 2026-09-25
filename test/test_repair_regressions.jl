using Test
using Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Distances: Euclidean
using KernelAbstractions: KernelAbstractions as KA
using FFTW
using SpectralBackends: SpectralBackends as SB

@testset "Repair regressions" begin
    @testset "Nonuniform finite differences" begin
        for r in ([1.0, 2.0, 4.0], [0.0, 0.2, 1.0, 3.0, 7.0])
            @test SF.KHM.finite_difference(r, r .^ 2) ≈ 2 .* r atol=1e-14
            @test SF.KHM.finite_difference(r, 3 .* r .+ 7) ≈ fill(3.0, length(r))
        end
        @test SF.KHM.finite_difference([1, 3], [2, 8]) == [3.0, 3.0]
        @test_throws ArgumentError SF.KHM.finite_difference([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        @test_throws ArgumentError SF.KHM.finite_difference([1.0, Inf, 3.0], [1.0, 2.0, 3.0])
    end
    @testset "Batch count capacity" begin
        for N in (4, 24), varying in (false, true)
            rng = MersenneTwister(932)
            u = randn(rng, 2, N, 2)
            x = varying ? rand(rng, 2, N, 2) : rand(rng, 2, N)
            bins = SF.BinEdges([0.0, 2.0])
            value_bins = SF.BinEdges([-1e6, 1e6])
            initial = N == 4 ? UInt8(250) : UInt8(0)
            for kind in (:sf1d, :joint2d, :single_pass, :single_pass_2d)
                dims = kind == :sf1d ? (1, 2) : kind == :joint2d ? (1, 1, 2) :
                       kind == :single_pass ? (6, 1, 2) : (6, 1, 1, 2)
                sums, counts = zeros(dims), fill(initial, dims)
                @test_throws ArgumentError if kind == :sf1d
                    SFC._bl_run_1d!(sums, counts, SFT.S2SFType(), x, u, bins, Euclidean(), SFC._bl_serial_exec)
                elseif kind == :joint2d
                    SFC._bl_run_joint2d!(sums, counts, SFT.S2SFType(), x, u, bins, value_bins, Euclidean(), SFC._bl_serial_exec)
                elseif kind == :single_pass
                    SFC._bl_run_sp1d!(sums, counts, x, u, bins, Euclidean(), SFC._bl_serial_exec)
                else
                    SFC._bl_run_sp2d!(sums, counts, x, u, bins, value_bins, Euclidean(), SFC._bl_serial_exec)
                end
                @test all(iszero, sums)
                @test all(==(initial), counts)
            end
        end
    end
    @testset "GPU count staging retains mass and width" begin
        ext = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
        for incoming in ([0.25, 2.5], UInt64[UInt64(typemax(UInt32)) + 2, 7]), with_workspace in (false, true)
            sums, counts = zeros(2), zeros(eltype(incoming), 2)
            ws = with_workspace ? SFC.GPUSFWorkspace(KA.CPU(), [0.0, 1.0, 2.0]) : nothing
            @test begin
                ext._accumulate_gpu_sf_host!(sums, counts, [1.0, 2.0], incoming; workspace=ws)
                counts == incoming && sums == [1.0, 2.0]
            end
        end
    end
    @testset "Float32 zonal FFT output alignment" begin
        ext = Base.get_extension(SF, :StructureFunctionsAbstractFFTsExt)
        old_budget = ext.FORWARD_BATCH_BYTES[]
        try
            for (nlon, nlat) in ((36, 27), (60, 45)), chunk in (7, nlat)
                ext.FORWARD_BATCH_BYTES[] = chunk * (nlon * sizeof(Float32) + (nlon ÷ 2 + 1) * sizeof(ComplexF32))
                u = randn(MersenneTwister(14), Float32, 2, nlon, nlat)
                lats = collect(range(-0.6f0, 0.5f0; length=nlat))
                schedule = SFC.ZonalLagSchedule(lats, nlon, Float32(2π/nlon), 1.0f0, true)
                bins = Float32[0, 0.15, 0.35]
                reference, refcounts = zeros(Float32, 2), zeros(UInt64, 2)
                SFC.gridded_lag_sweep!(reference, refcounts, SFT.S2SFType(), u, schedule, bins, Val(2))
                for repetition in 1:2
                    sums, counts = zeros(Float32, 2), zeros(UInt64, 2)
                    @test begin
                        SFC.gridded_sweep!(sums, counts, SFT.S2SFType(), u, schedule, bins, Val(2), SB.FastFourierTransformSpectralBackend())
                        counts == refcounts && isapprox(sums, reference; rtol=5e-5)
                    end
                end
            end
        finally
            ext.FORWARD_BATCH_BYTES[] = old_budget
        end
    end
end

@testset "Count addition and mutation preflight" begin
    using ComputationalBackends: SerialBackend
    bins = [0.0, 2.0]
    a = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), bins, [1.0], UInt8[250])
    b = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), bins, [1.0], UInt8[6])
    @test_throws ArgumentError a + b
    @test a.counts == UInt8[250]
    x = [0.0 0.1 0.2 0.3; 0.0 0.0 0.0 0.0]
    u = [0.0 1.0 2.0 3.0; 0.0 1.0 2.0 3.0]
    s, c = [0.0], UInt8[250]
    @test_throws ArgumentError SFC.calculate_structure_function!(s, c, SFT.S2SFType(), x, u, bins; backend=SerialBackend())
    @test s == [0.0] && c == UInt8[250]
    ts, tc = zeros(2, 2, 1), UInt8[250]
    @test_throws ArgumentError SFC.calculate_structure_function_tensor!(ts, tc, Val(2), x, u, bins; backend=SerialBackend())
    @test all(iszero, ts) && tc == UInt8[250]
end

@testset "Dimensions above eight" begin
    using ComputationalBackends: SerialBackend
    for D in (9, 12)
        x = zeros(D, 3); x[1, :] = [0, 1, 2]
        u = zeros(D, 3); u[end, :] = [0, 2, 5]
        r = SFC.calculate_structure_function(SFT.S2SFType(), x, u, [0.0, 3.0]; backend=SerialBackend(),
            output_type=SF.StructureFunctionSumsAndCounts, verbose=false, show_progress=false)
        @test r.counts == [3]
        @test r.sums == [38.0]
    end
end
