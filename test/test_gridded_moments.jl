using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, HelperFunctions as SFH
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using Random: Random

const FFT_TAG = SB.FastFourierTransformSpectralBackend()

# One call of the lag sweep (`backend === nothing`) or the transform, on a bare array or a multi-field.
function _moments_run(sf, field, dims, spacing, periodic, bins; valid = SFC.AllValid(), backend = nothing)
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(bins))
    s = zeros(Float64, nb)
    c = zeros(Int, nb)
    sched = SFC.UniformLagSchedule(dims, spacing, periodic)
    if field isa Fields
        backend === nothing ? SFC.gridded_lag_sweep!(s, c, sf, field, sched, bins; valid) :
                              SFC.gridded_sweep!(s, c, sf, field, sched, bins, backend; valid)
    else
        D = size(field, 1)
        backend === nothing ? SFC.gridded_lag_sweep!(s, c, sf, field, sched, bins, Val(D); valid) :
                              SFC.gridded_sweep!(s, c, sf, field, sched, bins, Val(D), backend; valid)
    end
    return s, c
end

# Absolute floor at the field scale: an odd operator on a self-reverse lag is exactly zero only in the sweep.
_moments_close(got, ref) = isapprox(got, ref; rtol = 1e-9, atol = 1e-10 * max(1.0, maximum(abs, ref)))

# Orders 2, 3 and 4 on every topology, a norm power, signed-transverse operators on a 2-D grid and on 3-D grids with
# lags along ẑ (the axis rule in 3-D), and a line; test_operator_contract.jl checks each operator's contraction.
const MOMENT_OPERATOR_CASES = (
    (((9, 6), (0.1, 0.2), (false, false)), (SFT.L2SFType(), SFT.L3SFType())),
    (((8, 8), (0.25, 0.25), (true, true)), (SFT.L2T1SFType(), SFT.FullVectorStructureFunctionType{4}())),
    (((10, 7), (0.15, 0.15), (true, false)), (SFT.T2ComponentSFType(), SFT.ProjectedStructureFunctionType{4, 0}())),
    (((12,), (0.3,), (false,)), (SFT.L3SFType(),)),
    (((6, 5, 4), (0.2, 0.2, 0.3), (false, false, false)),
     (SFT.ProjectedStructureFunctionType{0, 3}(
          SFH.ReferenceAxisTransverseBasis(SA.SVector(1.0, sqrt(2.0), sqrt(3.0)))),)),
    (((6, 6, 4), (0.2, 0.2, 0.3), (true, false, true)), (SFT.T3SFType(),)),
)

Test.@testset "the transform equals the lag sweep for each order, topology and convention" begin
    counts_ok, sums_ok = Bool[], Bool[]
    for ((dims, spacing, periodic), ops) in MOMENT_OPERATOR_CASES
        Dg = length(dims)
        Random.seed!(9100 + prod(dims) + Dg)
        u = randn(Dg, dims...)
        r_max = 0.7 * sum(d -> spacing[d] * dims[d], 1:Dg)
        bins = collect(range(0.0, r_max; length = 9))
        for sf in ops
            ref_s, ref_c = _moments_run(sf, u, dims, spacing, periodic, bins)
            got_s, got_c = _moments_run(sf, u, dims, spacing, periodic, bins; backend = FFT_TAG)
            push!(counts_ok, got_c == ref_c && sum(ref_c) > 0)
            push!(sums_ok, _moments_close(got_s, ref_s))
        end
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
end

# Each order once on a masked field, on a bounded, a wrapping and a mixed grid.
const MOMENT_MASKED_CASES = (
    (((9, 6), (0.1, 0.2), (false, false)), SFT.S2SFType()),
    (((8, 8), (0.25, 0.25), (true, true)), SFT.T3SFType()),
    (((10, 7), (0.15, 0.15), (true, false)), SFT.ProjectedStructureFunctionType{4, 0}()),
)

Test.@testset "the masked transform equals the masked lag sweep at every order" begin
    counts_ok, sums_ok = Bool[], Bool[]
    for ((dims, spacing, periodic), sf) in MOMENT_MASKED_CASES
        N = prod(dims)
        Random.seed!(9200 + N)
        u = randn(2, dims...)
        uf = reshape(u, 2, N)
        for k in 1:N
            rand() < 0.3 && (uf[1, k] = NaN)
        end
        valid = SFC.field_validity(u)
        bins = collect(range(0.0, 1.3; length = 8))
        ref_s, ref_c = _moments_run(sf, u, dims, spacing, periodic, bins; valid)
        got_s, got_c = _moments_run(sf, u, dims, spacing, periodic, bins; valid, backend = FFT_TAG)
        push!(counts_ok, got_c == ref_c && sum(ref_c) > 0)
        push!(sums_ok, all(isfinite, got_s) && _moments_close(got_s, ref_s))
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
end

# Grid coordinates as a point list, for the unstructured oracle.
function _grid_points(dims, spacing)
    Dg = length(dims)
    N = prod(dims)
    x = zeros(Float64, Dg, N)
    for (k, I) in enumerate(CartesianIndices(dims)), d in 1:Dg
        x[d, k] = (I[d] - 1) * spacing[d]
    end
    return x
end

# Each field kind once, an odd moment of a vector with a scalar and of a scalar alone, wrapping and bounded; one per
# field against the point path.
const MOMENT_FIELD_CASES = (
    (:vs, SFT.MixedSFType{1, 0, 2}(), (false, false)), (:vs, SFT.MixedSFType{1, 0, 1}(), (true, false)),
    (:s, SFT.ScalarSFType{3}(2), (true, false)), (:vv, SFT.VectorDotSFType(1, 2), (true, false)),
)
const MOMENT_FIELD_ORACLE_CASES =
    ((:vs, SFT.MixedSFType{1, 0, 1}()), (:s, SFT.ScalarSFType{3}(2)), (:vv, SFT.VectorDotSFType(1, 2)))

Test.@testset "multi-fields on a grid: scalar, mixed and cross-field moments" begin
    dims = (9, 7)
    spacing = (0.1, 0.15)
    Random.seed!(9300)
    u = randn(2, dims...)
    θ = randn(dims...)
    a = randn(2, dims...)
    bins = [0.0; collect(range(0.1137, 0.93; length = 7))]
    fields = (vs = Fields(vectors = (u,), scalars = (θ,)), s = Fields(scalars = (θ, a[1, :, :])),
              vv = Fields(vectors = (u, a)))
    transform_ok = Bool[]
    for (key, sf, periodic) in MOMENT_FIELD_CASES
        f = fields[key]
        ref_s, ref_c = _moments_run(sf, f, dims, spacing, periodic, bins)
        got_s, got_c = _moments_run(sf, f, dims, spacing, periodic, bins; backend = FFT_TAG)
        push!(transform_ok, got_c == ref_c && sum(ref_c) > 0 && _moments_close(got_s, ref_s))
    end
    Test.@test all(transform_ok)

    x = _grid_points(dims, spacing)
    sweep_ok = Bool[]
    for (key, sf) in MOMENT_FIELD_ORACLE_CASES
        f = fields[key]
        ref = SFC.calculate_structure_function(sf, x, f, bins, SFO.StructureFunctionSumsAndCounts;
                                               backend = CB.SerialBackend())
        got_s, got_c = _moments_run(sf, f, dims, spacing, (false, false), bins)
        push!(sweep_ok, got_c == Int.(ref.counts) && isapprox(got_s, ref.sums; rtol = 1e-10, atol = 1e-12))
    end
    Test.@test all(sweep_ok)

    f_vs, f_s = fields.vs, fields.s
    Test.@test_throws ArgumentError _moments_run(SFT.VectorDotSFType(1, 2), f_vs, dims, spacing,
                                                 (false, false), bins)
    Test.@test_throws ArgumentError _moments_run(SFT.VectorDotSFType(1, 2), f_vs, dims, spacing,
                                                 (false, false), bins; backend = FFT_TAG)
    Test.@test_throws ArgumentError _moments_run(SFT.L2SFType(), f_s, dims, spacing, (false, false),
                                                 bins; backend = FFT_TAG)
end

Test.@testset "directional output on a grid agrees with the unstructured joint path and marginalises" begin
    dims = (10, 8)
    spacing = (0.1, 0.1)
    Random.seed!(9400)
    u = randn(2, dims...)
    bins = [0.0; collect(range(0.1137, 0.95; length = 7))]
    ax_bins = [prevfloat(0.0); collect(range(0.3011, π - 0.3; length = 5)); π + 1e-9]
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    nb = length(bins) - 1
    na = length(ax_bins) - 1
    sched = SFC.UniformLagSchedule(dims, spacing, (false, false))

    ts = zeros(nb, na); tc = zeros(Float64, nb, na)
    SFC.gridded_sweep!(ts, tc, SFT.L2SFType(), u, sched, bins, ax_bins, Val(2), FFT_TAG; second_axis = src)
    ss = zeros(nb, na); sc = zeros(Float64, nb, na)
    SFC.gridded_lag_sweep!(ss, sc, SFT.L2SFType(), u, sched, bins, ax_bins, Val(2); second_axis = src)

    x = _grid_points(dims, spacing)
    ref = SFC.calculate_structure_function(SFT.L2SFType(), x, reshape(u, 2, :), bins, ax_bins;
                                           backend = CB.SerialBackend(), second_axis = src)
    Test.@test sc == Float64.(ref.counts)
    Test.@test tc == Float64.(ref.counts)
    Test.@test isapprox(ss, ref.sums; rtol = 1e-10, atol = 1e-12)
    Test.@test isapprox(ts, ref.sums; rtol = 1e-9, atol = 1e-10)
    Test.@test sum(tc) > 0

    s1, c1 = _moments_run(SFT.L2SFType(), u, dims, spacing, (false, false), bins)
    Test.@test vec(sum(tc; dims = 2)) == Float64.(c1)
    Test.@test isapprox(vec(sum(ts; dims = 2)), s1; rtol = 1e-9, atol = 1e-10)

    # half-turn lags of an even periodic grid split their pairs: integer counts are refused, float ones marginalise
    per = SFC.UniformLagSchedule((8, 8), (0.25, 0.25), (true, true))
    u8 = randn(2, 8, 8)
    pbins = [0.0; collect(range(0.2611, 1.5; length = 6))]
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(
        zeros(6, na), zeros(Int, 6, na), SFT.L2SFType(), u8, per, pbins, ax_bins, Val(2); second_axis = src)
    ps = zeros(6, na); pc = zeros(Float64, 6, na)
    SFC.gridded_sweep!(ps, pc, SFT.L2SFType(), u8, per, pbins, ax_bins, Val(2), FFT_TAG; second_axis = src)
    ps1, pc1 = _moments_run(SFT.L2SFType(), u8, (8, 8), (0.25, 0.25), (true, true), pbins)
    Test.@test isapprox(vec(sum(pc; dims = 2)), Float64.(pc1); atol = 1e-9)
    Test.@test isapprox(vec(sum(ps; dims = 2)), ps1; rtol = 1e-9, atol = 1e-10)
end

Test.@testset "the forward transforms' block size changes nothing beyond rounding" begin
    # 9 slabs × 6 monomials of 400 B: one block; blocks of 1; groups of 2; 4 and a short 2; chunks of 4 slabs and a 1.
    ext = Base.get_extension(SF, :StructureFunctionsAbstractFFTsExt)
    Random.seed!(9550)
    dims = (24, 9)
    coords = cumsum(0.7 .+ 0.4 .* rand(dims[2]))
    s = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((dims[1],), (0.5,), (true,)), (coords,), (1, 2))
    u = randn(2, dims...)
    bins = collect(range(0.0, 4.0; length = 9))
    budget = ext.FORWARD_BATCH_BYTES[]
    agree = Bool[]
    try
        nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(bins))
        run() = (a = zeros(Float64, nb); c = zeros(Int, nb);
                 SFC.gridded_sweep!(a, c, SFT.L2SFType(), u, s, bins, Val(2), FFT_TAG); (a, c))
        ref_s, ref_c = run()
        for bytes in (1, 800, 1600, 9600)
            ext.FORWARD_BATCH_BYTES[] = bytes
            got_s, got_c = run()
            push!(agree, got_c == ref_c && sum(ref_c) > 0 && isapprox(got_s, ref_s; rtol = 1e-12))
        end
    finally
        ext.FORWARD_BATCH_BYTES[] = budget
    end
    Test.@test all(agree)
end

Test.@testset "a descending axis gives the ascending axis's spectrum and wavenumbers" begin
    dims = (16, 12)
    Random.seed!(9500)
    u = randn(2, dims...)
    kax, dens = SFC.gridded_spectrum(u, SFC.UniformLagSchedule(dims, (0.5, 0.5), (true, true)), Val(2), FFT_TAG)
    kax2, dens2 = SFC.gridded_spectrum(u, SFC.UniformLagSchedule(dims, (0.5, -0.5), (true, true)), Val(2), FFT_TAG)
    Test.@test dens2 == dens
    Test.@test kax2[2] == kax[2]
end

Test.@testset "the transform with bins shorter than the grid equals the lag sweep" begin
    dims = (12, 10)
    spacing = (0.1, 0.1)
    Random.seed!(9600)
    u = randn(2, dims...)
    tight = collect(range(0.0, 0.35; length = 6))
    agree = Bool[]
    for sf in (SFT.L2SFType(), SFT.L3SFType())
        ref_s, ref_c = _moments_run(sf, u, dims, spacing, (false, false), tight)
        got_s, got_c = _moments_run(sf, u, dims, spacing, (false, false), tight; backend = FFT_TAG)
        push!(agree, got_c == ref_c && sum(ref_c) > 0 && _moments_close(got_s, ref_s))
    end
    Test.@test all(agree)
end

Test.@testset "the grid entry takes a multi-field and a joint request" begin
    geo = FG.Geometry.CartesianGeometry()
    nx, ny = 9, 7
    grid = FG.Grids.StructuredGrid(geo, range(0.0, step = 0.2, length = nx),
                                   range(0.0, step = 0.2, length = ny))
    Random.seed!(9700)
    u = randn(2, nx, ny)
    θ = randn(nx, ny)
    f = Fields(vectors = (u,), scalars = (θ,))
    bins = [0.0; collect(range(0.2137, 1.4; length = 7))]
    yag = SFT.MixedSFType{1, 0, 2}()
    swept = SFC.calculate_structure_function(yag, grid, f, bins, SFO.StructureFunctionSumsAndCounts)
    transformed = SFC.calculate_structure_function(yag, grid, f, bins, FFT_TAG, UInt32,
        SFO.StructureFunctionSumsAndCounts)
    Test.@test transformed.counts == swept.counts
    Test.@test _moments_close(transformed.sums, swept.sums)
    Test.@test sum(swept.counts) > 0

    ax_bins = [prevfloat(0.0); collect(range(0.3011, π - 0.3; length = 5)); π + 1e-9]
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    joint = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins, ax_bins;
        second_axis = src)
    joint_t = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins, ax_bins, FFT_TAG;
        second_axis = src)
    plain = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins, SFO.StructureFunctionSumsAndCounts)
    Test.@test joint isa SFO.StructureFunction2DSumsAndCounts
    Test.@test vec(sum(joint.counts; dims = 2)) == Float64.(plain.counts)
    Test.@test isapprox(vec(sum(joint.sums; dims = 2)), plain.sums; rtol = 1e-10, atol = 1e-12)
    Test.@test joint_t.counts == joint.counts
    Test.@test _moments_close(joint_t.sums, joint.sums)
end
