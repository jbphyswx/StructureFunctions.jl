using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using Random: Random
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    Calculations as SFC,
    StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO,
    batch_histograms_equal
using Distances: Distances as DI

Random.seed!(20260621)

const GPU_SHAPE_BE = CB.GPUBackend(KA.CPU())
const GPU_SHAPE_CPU_BE = CB.SerialBackend()
const GPU_SHAPE_INV = (:S2, :L2, :T2, :S3, :L3, :L1T2)

function _gpu_shape_pairwise(sf, x, u, bins)
    return SFC.calculate_structure_function(
        sf, x, u, bins, SFO.StructureFunctionSumsAndCounts;
        backend = GPU_SHAPE_BE,
    )
end

function _cpu_shape_pairwise(sf, x, u, bins)
    return SFC.calculate_structure_function(
        sf, x, u, bins, SFO.StructureFunctionSumsAndCounts;
        backend = GPU_SHAPE_CPU_BE,
    )
end

"""Whether the device result `g` equals the CPU result `c`, every invariant but `:helmholtz` of a named tuple."""
_agrees(g::NamedTuple, c; rtol) =
    all(k -> k === :helmholtz || (g[k].counts == c[k].counts && isapprox(g[k].sums, c[k].sums; rtol)), keys(c))
_agrees(g, c; rtol) = g.counts == c.counts && isapprox(g.sums, c.sums; rtol)

Test.@testset "GPU public shape contract (KA.CPU)" begin
    sf = SFT.L2SFType()
    bins = collect(Float32, range(0.0f0, 1.75f0; length = 10))
    value_bins = collect(Float32, range(-0.1f0, 1.5f0; length = 8))
    n_bins = length(bins) - 1

    Test.@testset "shared-position auxiliary axes match explicit slices" begin
        x = rand(Float32, 2, 11)
        u = rand(Float32, 2, 11, 3, 2)

        gpu = _gpu_shape_pairwise(sf, x, u, bins)
        Test.@test size(gpu.sums) == (n_bins, 3, 2)
        Test.@test size(gpu.counts) == (n_bins, 3, 2)

        ref_sums = zeros(Float32, n_bins, 3, 2)
        ref_counts = zeros(UInt32, n_bins, 3, 2)
        for idx in CartesianIndices((3, 2))
            t, m = Tuple(idx)
            rt = _cpu_shape_pairwise(sf, x, u[:, :, t, m], bins)
            ref_sums[:, t, m] .= rt.sums
            ref_counts[:, t, m] .= rt.counts
        end
        Test.@test batch_histograms_equal(gpu.sums, gpu.counts, ref_sums, ref_counts; atol = 1f-4)
    end

    Test.@testset "varying-position auxiliary axes match explicit slices" begin
        x = rand(Float32, 2, 11, 3)
        u = rand(Float32, 2, 11, 3)

        gpu = _gpu_shape_pairwise(sf, x, u, bins)
        Test.@test size(gpu.sums) == (n_bins, 3)
        Test.@test size(gpu.counts) == (n_bins, 3)

        ref_sums = zeros(Float32, n_bins, 3)
        ref_counts = zeros(UInt32, n_bins, 3)
        for t in 1:3
            rt = _cpu_shape_pairwise(sf, x[:, :, t], u[:, :, t], bins)
            ref_sums[:, t] .= rt.sums
            ref_counts[:, t] .= rt.counts
        end
        Test.@test batch_histograms_equal(gpu.sums, gpu.counts, ref_sums, ref_counts; atol = 1f-4)
    end

    Test.@testset "joint 2D shared and varying auxiliary axes" begin
        x_shared = rand(Float32, 2, 9)
        u_shared = rand(Float32, 2, 9, 2)
        shared = SFC.calculate_structure_function(
            sf, x_shared, u_shared, bins, value_bins; backend = GPU_SHAPE_BE,
        )
        Test.@test size(shared.sums) == (n_bins, length(value_bins) - 1, 2)

        x_varying = rand(Float32, 2, 9, 2)
        u_varying = rand(Float32, 2, 9, 2)
        varying = SFC.calculate_structure_function(
            sf, x_varying, u_varying, bins, value_bins; backend = GPU_SHAPE_BE,
        )
        Test.@test size(varying.sums) == (n_bins, length(value_bins) - 1, 2)
    end

    Test.@testset "single-pass auxiliary axes preserve public shape" begin
        x = rand(Float32, 2, 10)
        u = rand(Float32, 2, 10, 2, 3)

        gpu = SFC.calculate_structure_functions_single_pass(
            x, u, bins, SFO.StructureFunctionSumsAndCounts; backend = GPU_SHAPE_BE,
        )
        gpu2d = SFC.calculate_structure_functions_single_pass_2d(x, u, bins, value_bins; backend = GPU_SHAPE_BE)
        Test.@test keys(gpu) == GPU_SHAPE_INV
        Test.@test keys(gpu2d) == GPU_SHAPE_INV
        Test.@test all(k -> size(gpu[k].sums) == (n_bins, 2, 3) &&
                            size(gpu2d[k].sums) == (n_bins, length(value_bins) - 1, 2, 3), GPU_SHAPE_INV)
        slices = Tuple.(CartesianIndices((2, 3)))
        cpu = [SFC.calculate_structure_functions_single_pass(
                   x, u[:, :, t, m], bins, SFO.StructureFunctionSumsAndCounts; backend = GPU_SHAPE_CPU_BE,
               ) for (t, m) in slices]
        cpu2d = [SFC.calculate_structure_functions_single_pass_2d(x, u[:, :, t, m], bins, value_bins;
                                                                  backend = GPU_SHAPE_CPU_BE) for (t, m) in slices]
        Test.@test all(((t, m),) -> all(k -> batch_histograms_equal(gpu[k].sums[:, t, m], gpu[k].counts[:, t, m],
                                                                     cpu[t, m][k].sums, cpu[t, m][k].counts;
                                                                     atol = 1f-4), GPU_SHAPE_INV), slices)
        Test.@test all(((t, m),) -> all(k -> batch_histograms_equal(gpu2d[k].sums[:, :, t, m],
                                                                     gpu2d[k].counts[:, :, t, m],
                                                                     cpu2d[t, m][k].sums, cpu2d[t, m][k].counts;
                                                                     atol = 1f-4), GPU_SHAPE_INV), slices)
    end

    # A one-wide field runs on the device and equals the CPU.
    Test.@testset "a one-dimensional field runs on the device" begin
        Random.seed!(3)
        x1 = rand(Float32, 1, 64)
        u1 = randn(Float32, 1, 64)
        b1 = collect(Float32, range(0.0f0, 0.9f0; length = 9))
        cpu1 = SFC.calculate_structure_function(
            sf, x1, u1, b1, SFO.StructureFunctionSumsAndCounts; backend = CB.SerialBackend())
        gpu1 = SFC.calculate_structure_function(
            sf, x1, u1, b1, SFO.StructureFunctionSumsAndCounts; backend = GPU_SHAPE_BE)
        Test.@test gpu1.counts == cpu1.counts
        Test.@test isapprox(gpu1.sums, cpu1.sums; rtol = 1f-5)
    end

    Test.@testset "invalid shapes fail before GPU launch" begin
        Test.@test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(Float32, 2, 5), rand(Float32, 3, 5), bins;
            backend = GPU_SHAPE_BE,
        )
        Test.@test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(Float32, 2, 5, 2), rand(Float32, 2, 5, 3), bins;
            backend = GPU_SHAPE_BE,
        )
        Test.@test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(Float32, 2, 5, 2), rand(Float32, 2, 5), bins;
            backend = GPU_SHAPE_BE,
        )
    end
end

# A device backend holding a configured KernelAbstractions backend gives the default backend's answer.
Test.@testset "a configured device backend takes the outputs its device holds" begin
    Random.seed!(11)
    configured = CB.GPUBackend(KA.CPU(; static = true))
    x, u = rand(Float32, 2, 64), randn(Float32, 2, 64)
    bins = collect(Float32, range(0.0f0, 0.8f0; length = 9))
    ref_s, ref_c = zeros(Float32, 8), zeros(UInt32, 8)
    SFC.calculate_structure_function!(ref_s, ref_c, SFT.L2SFType(), x, u, bins; backend = GPU_SHAPE_BE)
    s, c = zeros(Float32, 8), zeros(UInt32, 8)
    SFC.calculate_structure_function!(s, c, SFT.L2SFType(), x, u, bins; backend = configured)
    Test.@test c == ref_c
    Test.@test s ≈ ref_s
    grid = randn(Float32, 2, 12 * 8)
    schedule = SFC.UniformLagSchedule((12, 8), (1.0, 1.0), (true, false))
    lag_bins = collect(Float32, range(0.0f0, 6.0f0; length = 7))
    ref_s, ref_c = zeros(Float32, 6), zeros(Int, 6)
    SFC.gridded_lag_sweep!(ref_s, ref_c, SFT.S3SFType(), grid, schedule, lag_bins, Val(2); backend = GPU_SHAPE_BE)
    s, c = zeros(Float32, 6), zeros(Int, 6)
    SFC.gridded_lag_sweep!(s, c, SFT.S3SFType(), grid, schedule, lag_bins, Val(2); backend = configured)
    Test.@test c == ref_c
    Test.@test s ≈ ref_s
end

# Every point-field family agrees with the CPU under a spherical metric, which differs from the flat answer and needs a geometry.
Test.@testset "GPU point-field families honour a spherical metric" begin
    FT = Float64
    N = 64
    R = 6.371e6
    m = DI.Haversine(R)
    sft = SFT.L2SFType()
    lon = 300 .* rand(N) .- 150
    lat = 100 .* rand(N) .- 50
    x = permutedims(hcat(lon, lat))
    u = permutedims(hcat(randn(N), randn(N)))
    db = collect(FT, range(0.0, 9.0e6; length = 11))
    vb = collect(FT, range(-4.0, 4.0; length = 9))
    kw = (; distance_metric = m)

    for (name, call) in (
            ("sf1d", (be,) -> SFC.calculate_structure_function(
                sft, x, u, db, SFO.StructureFunctionSumsAndCounts; backend = be, kw...)),
            ("joint2d", (be,) -> SFC.calculate_structure_function(
                sft, x, u, db, vb; backend = be, kw...)),
            ("sp1d", (be,) -> SFC.calculate_structure_functions_single_pass(
                x, u, db, SFO.StructureFunctionSumsAndCounts; backend = be, kw...)),
            ("sp2d", (be,) -> SFC.calculate_structure_functions_single_pass_2d(
                x, u, db, vb; backend = be, kw...)),
        )
        Test.@test (name, _agrees(call(GPU_SHAPE_BE), call(GPU_SHAPE_CPU_BE); rtol = 1e-8)) == (name, true)
    end

    raw(mm) = SFC.calculate_structure_function(
        sft, x, u, db, SFO.StructureFunctionSumsAndCounts;
        backend = CB.SerialBackend(), distance_metric = mm,
    )
    Test.@test raw(DI.Euclidean()).counts != raw(m).counts

    Test.@test_throws ArgumentError raw(DI.Cityblock())
end

# Every batch family agrees with the CPU under a spherical metric.
Test.@testset "GPU batch families honour a spherical metric" begin
    FT = Float64
    N, B = 40, 3
    R = 6.371e6
    m = DI.Haversine(R)
    lon = 300 .* rand(N) .- 150
    lat = 100 .* rand(N) .- 50
    x = permutedims(hcat(lon, lat))
    u3 = reshape(randn(FT, 2, N, B), 2, N, B)
    db = collect(FT, range(0.0, 9.0e6; length = 11))
    vb = collect(FT, range(-4.0, 4.0; length = 9))
    sft = SFT.L2SFType()
    kw = (; distance_metric = m)

    for (name, call) in (
            ("sf1d batch", (be,) -> SFC.calculate_structure_function(
                sft, x, u3, db, SFO.StructureFunctionSumsAndCounts; backend = be, kw...)),
            ("sp1d batch", (be,) -> SFC.calculate_structure_functions_single_pass(
                x, u3, db, SFO.StructureFunctionSumsAndCounts; backend = be, kw...)),
            ("sp2d batch", (be,) -> SFC.calculate_structure_functions_single_pass_2d(
                x, u3, db, vb; backend = be, kw...)),
        )
        Test.@test (name, _agrees(call(GPU_SHAPE_BE), call(GPU_SHAPE_CPU_BE); rtol = 1e-8)) == (name, true)
    end
end
