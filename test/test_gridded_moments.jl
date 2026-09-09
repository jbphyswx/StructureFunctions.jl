using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, Fields
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using Random: Random

const FFT_TAG = SB.FastFourierTransformSpectralBackend()

# One call for either route, on a bare array or a channel bundle.
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

# An odd operator on a self-reverse lag is exactly zero in the sweep and round-off in the transform, so
# the comparison carries an absolute floor set by the field scale.
_moments_close(got, ref) = isapprox(got, ref; rtol = 1e-9, atol = 1e-10 * max(1.0, maximum(abs, ref)))

const MOMENT_GRIDS = (
    ((9, 6), (0.1, 0.2), (false, false)),
    ((8, 8), (0.25, 0.25), (true, true)),
    ((10, 7), (0.15, 0.15), (true, false)),
    ((12,), (0.3,), (false,)),
    ((6, 5, 4), (0.2, 0.2, 0.3), (false, false, false)),
    ((6, 6, 4), (0.2, 0.2, 0.3), (true, false, true)),
)

const VECTOR_OPS = (
    SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType(), SFT.T2ComponentSFType(),
    SFT.L3SFType(), SFT.S3SFType(), SFT.L1T2SFType(), SFT.L1T2ComponentSFType(),
    SFT.ProjectedStructureFunctionType{4, 0}(), SFT.ProjectedStructureFunctionType{2, 2}(),
    SFT.FullVectorStructureFunctionType{4}(), SFT.VectorDotSFType(1, 1),
)
# An oriented transverse direction exists at D = 2 everywhere, and at D = 3 not for lags along ẑ, so
# these run on two-dimensional grids only.
const ODD_TRANSVERSE_OPS = (SFT.L2T1SFType(), SFT.T3SFType(), SFT.ProjectedStructureFunctionType{0, 4}())

Test.@testset "the transform equals the lag sweep for every polynomial operator" begin
    for (dims, spacing, periodic) in MOMENT_GRIDS
        Dg = length(dims)
        Random.seed!(9100 + prod(dims) + Dg)
        u = randn(Dg, dims...)
        r_max = 0.7 * sum(d -> spacing[d] * dims[d], 1:Dg)
        bins = collect(range(0.0, r_max; length = 9))
        ops = Dg == 2 ? (VECTOR_OPS..., ODD_TRANSVERSE_OPS...) : VECTOR_OPS
        for sf in ops
            Dg == 1 && sf isa Union{SFT.T2ComponentSFType, SFT.L1T2ComponentSFType} && continue
            ref_s, ref_c = _moments_run(sf, u, dims, spacing, periodic, bins)
            got_s, got_c = _moments_run(sf, u, dims, spacing, periodic, bins; backend = FFT_TAG)
            Test.@test got_c == ref_c
            Test.@test _moments_close(got_s, ref_s)
            Test.@test sum(got_c) > 0
        end
    end
end

Test.@testset "the masked transform equals the masked lag sweep at every order" begin
    for (dims, spacing, periodic) in MOMENT_GRIDS[1:3]
        N = prod(dims)
        Random.seed!(9200 + N)
        u = randn(2, dims...)
        uf = reshape(u, 2, N)
        for k in 1:N
            rand() < 0.3 && (uf[1, k] = NaN)
        end
        valid = SFC.field_validity(u)
        Test.@test !(valid isa SFC.AllValid)
        bins = collect(range(0.0, 1.3; length = 8))
        for sf in (SFT.S2SFType(), SFT.L2SFType(), SFT.L3SFType(), SFT.S3SFType(), SFT.L1T2SFType(),
                   SFT.ProjectedStructureFunctionType{4, 0}(), SFT.T3SFType())
            ref_s, ref_c = _moments_run(sf, u, dims, spacing, periodic, bins; valid)
            got_s, got_c = _moments_run(sf, u, dims, spacing, periodic, bins; valid, backend = FFT_TAG)
            Test.@test got_c == ref_c
            Test.@test _moments_close(got_s, ref_s)
            Test.@test all(isfinite, got_s)
            Test.@test sum(got_c) > 0
        end
    end
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

Test.@testset "channel bundles on a grid: scalar, mixed and cross-channel moments" begin
    dims = (9, 7)
    spacing = (0.1, 0.15)
    Random.seed!(9300)
    u = randn(2, dims...)
    θ = randn(dims...)
    a = randn(2, dims...)
    # edges between the separations the lattice can produce, so a whole shell never sits on one
    bins = [0.0; collect(range(0.1137, 0.93; length = 7))]
    f_vs = Fields(vectors = (u,), scalars = (θ,))
    f_s = Fields(scalars = (θ, a[1, :, :]))
    f_vv = Fields(vectors = (u, a))
    cases = (
        (f_vs, (SFT.ScalarSFType{2}(), SFT.ScalarSFType{3}(), SFT.MixedSFType{1, 0, 2}(),
                SFT.MixedSFType{1, 0, 1}(), SFT.MixedSFType{0, 2, 1}(), SFT.L2SFType(), SFT.S3SFType())),
        (f_s, (SFT.ScalarSFType{2}(), SFT.ScalarSFType{3}(2), SFT.ScalarDotSFType(1, 2))),
        (f_vv, (SFT.VectorDotSFType(1, 2), SFT.VectorDotSFType(2, 2), SFT.L2SFType())),
    )
    for periodic in ((true, false), (false, false)), (f, ops) in cases, sf in ops
        ref_s, ref_c = _moments_run(sf, f, dims, spacing, periodic, bins)
        got_s, got_c = _moments_run(sf, f, dims, spacing, periodic, bins; backend = FFT_TAG)
        Test.@test got_c == ref_c
        Test.@test _moments_close(got_s, ref_s)
        Test.@test sum(got_c) > 0
    end

    # The lag sweep on a bundle against the unstructured channel path over the same points, which is
    # checked against brute force in test_channels.jl and shares no code with the lag enumeration.
    x = _grid_points(dims, spacing)
    for (f, ops) in cases, sf in ops
        ref = SFC.calculate_structure_function(sf, x, f, bins; backend = CB.SerialBackend(),
                                               output_type = SFO.StructureFunctionSumsAndCounts,
                                               verbose = false, show_progress = false)
        got_s, got_c = _moments_run(sf, f, dims, spacing, (false, false), bins)
        Test.@test got_c == Int.(ref.counts)
        Test.@test isapprox(got_s, ref.sums; rtol = 1e-10, atol = 1e-12)
    end

    # a channel the bundle does not carry is refused, on both routes
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
    # edges placed between the separations and angles a lattice can produce
    bins = [0.0; collect(range(0.1137, 0.95; length = 7))]
    # angle 0 is exact for lags along the reference axis; bins are (lo, hi], so the first edge sits below it
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
    ref = SFC.serial_calculate_structure_function(SFT.L2SFType(), x, reshape(u, 2, :), bins, ax_bins;
                                                  second_axis = src, verbose = false, show_progress = false)
    Test.@test sc == Float64.(ref.counts)
    Test.@test tc == Float64.(ref.counts)
    Test.@test isapprox(ss, ref.sums; rtol = 1e-10, atol = 1e-12)
    Test.@test isapprox(ts, ref.sums; rtol = 1e-9, atol = 1e-10)
    Test.@test sum(tc) > 0

    # summing over the angle gives the 1-D result, bin for bin
    s1, c1 = _moments_run(SFT.L2SFType(), u, dims, spacing, (false, false), bins)
    Test.@test vec(sum(tc; dims = 2)) == Float64.(c1)
    Test.@test isapprox(vec(sum(ts; dims = 2)), s1; rtol = 1e-9, atol = 1e-10)

    # on a periodic grid with an even cell count the half-turn lags split their pairs between two
    # directions: integer counts are refused, floating-point ones still marginalise exactly
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

Test.@testset "gridded_spectrum transforms each monomial once, and a descending axis gives positive dk" begin
    ext = Base.get_extension(SF, :StructureFunctionsFFTExt)
    Test.@test ext !== nothing
    dims = (16, 12)
    Random.seed!(9500)
    u = randn(2, dims...)
    s = SFC.UniformLagSchedule(dims, (0.5, 0.5), (true, true))
    mt = ext.MonomialTransforms(reshape(u, 2, :), s, dims, SFC.AllValid())
    ext._trace_lags(mt, Val(2))
    Test.@test mt.misses == 1 + 2 * 2                 # the mask, each component, each square
    Test.@test mt.hits >= 1
    Test.@test all(k -> k[2] == 0 || k[1] == k[2], keys(mt.cache))       # no off-diagonal monomial

    kax, dens = SFC.gridded_spectrum(u, s, Val(2), FFT_TAG)
    sdesc = SFC.UniformLagSchedule(dims, (0.5, -0.5), (true, true))
    kax2, dens2 = SFC.gridded_spectrum(u, sdesc, Val(2), FFT_TAG)
    Test.@test dens2 == dens
    Test.@test kax2[2] == kax[2]
    Test.@test sum(dens) > 0
end

Test.@testset "padding to n + h_max reproduces the 2n − 1 result at every lag within r_max" begin
    ext = Base.get_extension(SF, :StructureFunctionsFFTExt)
    dims = (24, 20)
    spacing = (0.1, 0.1)
    Random.seed!(9600)
    u = randn(2, dims...)
    s = SFC.UniformLagSchedule(dims, spacing, (false, false))
    tight = collect(range(0.0, 0.35; length = 6))          # h_max = 3 along each direction
    P_tight = ext._pad_dims(s, 0.35)
    P_full = ext._pad_dims(s, Inf)
    Test.@test all(P_tight .< P_full)
    Test.@test all(P_tight .>= dims .+ 3)
    for sf in (SFT.L2SFType(), SFT.L3SFType())
        ref_s, ref_c = _moments_run(sf, u, dims, spacing, (false, false), tight)
        got_s, got_c = _moments_run(sf, u, dims, spacing, (false, false), tight; backend = FFT_TAG)
        Test.@test got_c == ref_c
        Test.@test _moments_close(got_s, ref_s)
    end
end

Test.@testset "the grid entry takes a channel bundle and a joint request" begin
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
    swept = SFC.calculate_structure_function(yag, grid, f, bins;
        output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    transformed = SFC.calculate_structure_function(yag, grid, f, bins, UInt32, FFT_TAG;
        output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test transformed.counts == swept.counts
    Test.@test _moments_close(transformed.sums, swept.sums)
    Test.@test sum(swept.counts) > 0

    ax_bins = [prevfloat(0.0); collect(range(0.3011, π - 0.3; length = 5)); π + 1e-9]
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    joint = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins, ax_bins;
        second_axis = src, verbose = false, show_progress = false)
    joint_t = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins, ax_bins, FFT_TAG;
        second_axis = src, verbose = false, show_progress = false)
    plain = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins;
        output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test joint isa SFO.StructureFunction2DSumsAndCounts
    Test.@test vec(sum(joint.counts; dims = 2)) == Float64.(plain.counts)
    Test.@test isapprox(vec(sum(joint.sums; dims = 2)), plain.sums; rtol = 1e-10, atol = 1e-12)
    Test.@test joint_t.counts == joint.counts
    Test.@test _moments_close(joint_t.sums, joint.sums)
end
