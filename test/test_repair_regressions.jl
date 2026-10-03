using Test: Test
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB

Test.@testset "Repair regressions" begin
    Test.@testset "Nonuniform finite differences" begin
        for r in ([1.0, 2.0, 4.0], [0.0, 0.2, 1.0, 3.0, 7.0])
            Test.@test SF.KHM.finite_difference(r, r .^ 2) ≈ 2 .* r atol=1e-14
            Test.@test SF.KHM.finite_difference(r, 3 .* r .+ 7) ≈ fill(3.0, length(r))
        end
        Test.@test SF.KHM.finite_difference([1, 3], [2, 8]) == [3.0, 3.0]
        Test.@test_throws ArgumentError SF.KHM.finite_difference([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        Test.@test_throws ArgumentError SF.KHM.finite_difference([1.0, Inf, 3.0], [1.0, 2.0, 3.0])
    end
    Test.@testset "Batch count capacity" begin
        for N in (4, 24), varying in (false, true)
            rng = Random.MersenneTwister(932)
            u = randn(rng, 2, N, 2)
            x = varying ? rand(rng, 2, N, 2) : rand(rng, 2, N)
            bins = SF.BinEdges([0.0, 2.0])
            value_bins = SF.BinEdges([-1e6, 1e6])
            initial = N == 4 ? UInt8(250) : UInt8(0)
            for kind in (:sf1d, :joint2d, :single_pass, :single_pass_2d)
                dims = kind == :sf1d ? (1, 2) : kind == :joint2d ? (1, 1, 2) :
                       kind == :single_pass ? (6, 1, 2) : (6, 1, 1, 2)
                sums, counts = zeros(dims), fill(initial, dims)
                Test.@test_throws ArgumentError if kind == :sf1d
                    SFC.calculate_structure_function_batch!(sums, counts, SFT.S2SFType(), x, u, bins;
                        backend = CB.SerialBackend())
                elseif kind == :joint2d
                    SFC.calculate_structure_function_2d_batch!(sums, counts, SFT.S2SFType(), x, u, bins, value_bins;
                        backend = CB.SerialBackend())
                elseif kind == :single_pass
                    SFC.calculate_structure_functions_single_pass_batch!(sums, counts, x, u, bins;
                        backend = CB.SerialBackend())
                else
                    SFC.calculate_structure_functions_single_pass_2d_batch!(sums, counts, x, u, bins, value_bins;
                        backend = CB.SerialBackend())
                end
                Test.@test all(iszero, sums)
                Test.@test all(==(initial), counts)
            end
        end
    end
    Test.@testset "Float32 zonal FFT output alignment" begin
        ext = Base.get_extension(SF, :StructureFunctionsAbstractFFTsExt)
        old_budget = ext.FORWARD_BATCH_BYTES[]
        try
            nkeys = length(SFC._monomial_keys(Val(2), Val(2)))
            for (nlon, nlat) in ((36, 27), (60, 45)), slabs in (7, nlat)
                ext.FORWARD_BATCH_BYTES[] = slabs * nkeys * (nlon * sizeof(Float32) + (nlon ÷ 2 + 1) * sizeof(ComplexF32))
                u = randn(Random.MersenneTwister(14), Float32, 2, nlon, nlat)
                lats = collect(range(-0.6f0, 0.5f0; length=nlat))
                schedule = SFC.ZonalLagSchedule(lats, nlon, Float32(2π/nlon), 1.0f0, true)
                bins = Float32[0, 0.15, 0.35]
                reference, refcounts = zeros(Float32, 2), zeros(UInt64, 2)
                SFC.gridded_lag_sweep!(reference, refcounts, SFT.S2SFType(), u, schedule, bins, Val(2))
                for repetition in 1:2
                    sums, counts = zeros(Float32, 2), zeros(UInt64, 2)
                    Test.@test begin
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

Test.@testset "Count addition and mutation preflight" begin
    using ComputationalBackends: SerialBackend
    bins = [0.0, 2.0]
    a = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), bins, [1.0], UInt8[250])
    b = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), bins, [1.0], UInt8[6])
    Test.@test_throws ArgumentError a + b
    Test.@test a.counts == UInt8[250]
    x = [0.0 0.1 0.2 0.3; 0.0 0.0 0.0 0.0]
    u = [0.0 1.0 2.0 3.0; 0.0 1.0 2.0 3.0]
    s, c = [0.0], UInt8[250]
    Test.@test_throws ArgumentError SFC.calculate_structure_function!(s, c, SFT.S2SFType(), x, u, bins; backend=SerialBackend())
    Test.@test s == [0.0] && c == UInt8[250]
    ts, tc = zeros(2, 2, 1), UInt8[250]
    Test.@test_throws ArgumentError SFC.calculate_structure_function_tensor!(ts, tc, Val(2), x, u, bins; backend=SerialBackend())
    Test.@test all(iszero, ts) && tc == UInt8[250]
end

Test.@testset "Dimensions above eight" begin
    using ComputationalBackends: SerialBackend
    for D in (9, 12)
        x = zeros(D, 3); x[1, :] = [0, 1, 2]
        u = zeros(D, 3); u[end, :] = [0, 2, 5]
        r = SFC.calculate_structure_function(SFT.S2SFType(), x, u, [0.0, 3.0], SF.StructureFunctionSumsAndCounts;
            backend=SerialBackend())
        Test.@test r.counts == [3]
        Test.@test r.sums == [38.0]
    end
end
