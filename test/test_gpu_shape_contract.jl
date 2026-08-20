using ComputationalBackends: ComputationalBackends as CB
using Test
using Random
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
        inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        x = rand(Float32, 2, 10)
        u = rand(Float32, 2, 10, 2, 3)

        gpu = SFC.calculate_structure_functions_single_pass(
            x, u, bins; backend = GPU_SHAPE_BE, output_type = SFO.StructureFunctionSumsAndCounts,
        )
        cpu = SFC.calculate_structure_functions_single_pass(
            x, u, bins; backend = GPU_SHAPE_CPU_BE, output_type = SFO.StructureFunctionSumsAndCounts,
        )
        @test keys(gpu) == inv
        for k in inv
            @test size(gpu[k].sums) == (n_bins, 2, 3)
            @test batch_histograms_equal(gpu[k].sums, gpu[k].counts, cpu[k].sums, cpu[k].counts; atol = 1f-4)
        end

        gpu2d = SFC.calculate_structure_functions_single_pass_2d(x, u, bins, value_bins; backend = GPU_SHAPE_BE)
        cpu2d = SFC.calculate_structure_functions_single_pass_2d(x, u, bins, value_bins; backend = GPU_SHAPE_CPU_BE)
        @test keys(gpu2d) == inv
        for k in inv
            @test size(gpu2d[k].sums) == (n_bins, length(value_bins) - 1, 2, 3)
            @test batch_histograms_equal(gpu2d[k].sums, gpu2d[k].counts, cpu2d[k].sums, cpu2d[k].counts; atol = 1f-4)
        end
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

# Every GPU kernel family carries the geometry, so a non-Euclidean metric produces the transported
# answer on GPU exactly as on CPU — the two must agree, and neither may silently return a flat one.
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
    kw = (; verbose = false, show_progress = false, distance_metric = m)

    for (name, call) in (
            ("sf1d", (be,) -> SFC.calculate_structure_function(
                sft, x, u, db; backend = be, output_type = SFO.StructureFunctionSumsAndCounts, kw...)),
            ("joint2d", (be,) -> SFC.calculate_structure_function(
                sft, x, u, db, vb; backend = be, kw...)),
            ("sp1d", (be,) -> SFC.calculate_structure_functions_single_pass(
                x, u, db; backend = be, output_type = SFO.StructureFunctionSumsAndCounts,
                distance_metric = m)),
            ("sp2d", (be,) -> SFC.calculate_structure_functions_single_pass_2d(
                x, u, db, vb; backend = be, distance_metric = m)),
        )
        g = call(GPU_SHAPE_BE); c = call(GPU_SHAPE_CPU_BE)
        if g isa NamedTuple
            for k in keys(c)
                k === :helmholtz && continue
                Test.@test g[k].counts == c[k].counts
                Test.@test isapprox(g[k].sums, c[k].sums; rtol = 1e-8)
            end
        else
            Test.@test g.counts == c.counts
            Test.@test isapprox(g.sums, c.sums; rtol = 1e-8)
        end
    end

    # The metric genuinely changes the answer: the transported result is not the flat one.
    raw(mm) = SFC.calculate_structure_function(
        sft, x, u, db; backend = CB.SerialBackend(), verbose = false, show_progress = false,
        distance_metric = mm, output_type = SFO.StructureFunctionSumsAndCounts,
    )
    Test.@test raw(DI.Euclidean()).counts != raw(m).counts

    # And a metric with NO geometry is refused outright on every backend rather than being assumed
    # flat — a distance function does not define a separation direction or a transport rule.
    Test.@test_throws ArgumentError raw(DI.Cityblock())
end

# The auxiliary-axis (batch) families carry the geometry into their kernels, so they honour a
# spherical metric and must reproduce the CPU's transported answer rather than a flat one.
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
    kw = (; verbose = false, show_progress = false, distance_metric = m)

    for (name, call) in (
            ("sf1d batch", (be,) -> SFC.calculate_structure_function(
                sft, x, u3, db; backend = be, output_type = SFO.StructureFunctionSumsAndCounts, kw...)),
            ("sp1d batch", (be,) -> SFC.calculate_structure_functions_single_pass(
                x, u3, db; backend = be, output_type = SFO.StructureFunctionSumsAndCounts, kw...)),
            ("sp2d batch", (be,) -> SFC.calculate_structure_functions_single_pass_2d(
                x, u3, db, vb; backend = be, kw...)),
        )
        g = call(GPU_SHAPE_BE); c = call(GPU_SHAPE_CPU_BE)
        if g isa NamedTuple
            for k in keys(c)
                k === :helmholtz && continue
                Test.@test g[k].counts == c[k].counts
                Test.@test isapprox(g[k].sums, c[k].sums; rtol = 1e-8)
            end
        else
            Test.@test g.counts == c.counts
            Test.@test isapprox(g.sums, c.sums; rtol = 1e-8)
        end
    end
end

# The point-field single-pass 2D family carries the geometry into its kernels, so it honors a
# non-Euclidean metric rather than refusing it, and must produce the CPU's transported answer.
Test.@testset "GPU single-pass 2D honors a spherical metric" begin
    FT = Float64
    N = 96
    lon = 300 .* rand(N) .- 150
    lat = 100 .* rand(N) .- 50
    x = permutedims(hcat(lon, lat))
    u = permutedims(hcat(randn(N), randn(N)))
    db = collect(FT, range(0.0, 9.0e6; length = 11))
    vbn = collect(FT, range(-4.0, 4.0; length = 9))
    m = DI.Haversine(6.371e6)

    got = SFC.calculate_structure_functions_single_pass_2d(
        x, u, db, vbn; backend = GPU_SHAPE_BE, distance_metric = m,
    )
    ref = SFC.calculate_structure_functions_single_pass_2d(
        x, u, db, vbn; backend = GPU_SHAPE_CPU_BE, distance_metric = m,
    )
    for k in (:S2, :L2, :T2, :S3, :L3, :L1T2)
        Test.@test got[k].counts == ref[k].counts
        Test.@test isapprox(got[k].sums, ref[k].sums; rtol = 1e-10)
    end

    # The metric genuinely changes the result: the transported answer is not the flat one.
    flat = SFC.calculate_structure_functions_single_pass_2d(
        x, u, db, vbn; backend = GPU_SHAPE_BE, distance_metric = DI.Euclidean(),
    )
    Test.@test flat.L2.counts != got.L2.counts
end

# The single-pass 2D GPU boundary used to splat display-only kwargs into a core with no `kwargs...`
# sink, so the package's own standard `verbose`/`show_progress` pair was a MethodError.
Test.@testset "GPU single-pass 2D accepts the standard display kwargs" begin
    FT = Float64
    N = 16
    x, u = rand(FT, 2, N), rand(FT, 2, N)
    db = collect(FT, range(0.0, 2.0; length = 9))
    vb = collect(FT, range(-2.0, 2.0; length = 7))
    ref = SFC.calculate_structure_functions_single_pass_2d(x, u, db, vb; backend = GPU_SHAPE_BE)
    got = SFC.calculate_structure_functions_single_pass_2d(
        x, u, db, vb; backend = GPU_SHAPE_BE, verbose = false, show_progress = false,
    )
    Test.@test keys(got) == keys(ref)
    for k in keys(ref)
        Test.@test got[k].counts == ref[k].counts
        Test.@test got[k].sums ≈ ref[k].sums
    end
end
