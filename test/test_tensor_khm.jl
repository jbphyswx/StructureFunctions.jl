using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, HelperFunctions as SFH
using StructureFunctions: MultiFields as MF
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using Random: Random
using Test: Test

Test.@testset "Tensor Structure Functions" begin
    # Three points: the rank-2 sums are each bin's outer products, the default is their mean, a slice of 2u adds 4×.
    x = [0.0 1.0 0.0; 0.0 0.0 1.0]
    u = [0.0 2.0 3.0; 0.0 5.0 7.0]
    bins = [0.0, 1.1, 2.0]
    du12 = u[:, 2] - u[:, 1]
    du13 = u[:, 3] - u[:, 1]
    du23 = u[:, 3] - u[:, 2]
    expected = cat(du12 * du12' + du13 * du13', du23 * du23'; dims = 3)

    t2 = SFC.calculate_structure_function_tensor(
        Val(2), x, u, bins, SF.StructureFunctionTensorSumsAndCounts; backend = CB.SerialBackend(),
    )
    Test.@test t2.counts == UInt32[2, 1]
    Test.@test t2.sums ≈ expected

    t2_mean = SFC.calculate_structure_function_tensor(Val(2), x, u, bins; backend = CB.SerialBackend())
    Test.@test t2_mean.values ≈ expected ./ reshape([2, 1], 1, 1, 2)

    t2_aux = SFC.calculate_structure_function_tensor(
        Val(2), x, cat(u, 2u; dims = 3), bins, SF.StructureFunctionTensorSumsAndCounts; backend = CB.SerialBackend(),
    )
    Test.@test t2_aux.counts == UInt32[2 2; 1 1]
    Test.@test t2_aux.sums ≈ cat(expected, 4 .* expected; dims = 4)
end

Test.@testset "each law's residual vanishes on a field obeying it" begin
    r = collect(range(0.1, 2.0; length = 9))
    eps = 0.37
    r4 = [1.0, 2.0, 3.0, 4.0]
    Test.@test SF.KHM.transverse_incompressibility_residual(r4, r4 .^ 2, 2 .* r4 .^ 2; dimension = 3) ≈ zeros(4)
    Test.@test all(abs.(SF.KHM.four_fifths_residual(r, -(4 / 5) .* eps .* r, eps)) .< 1e-12)
    Test.@test all(abs.(SF.KHM.four_thirds_residual(r, -(4 / 3) .* eps .* r, eps)) .< 1e-12)
    Test.@test all(abs.(SF.KHM.yaglom_residual(r, -(4 / 3) .* eps .* r, eps)) .< 1e-12)
end

# --- Tensors from the transform, higher orders, and the joint tensor over angle ---

const RAW_T = SF.StructureFunctionTensorSumsAndCounts
const FFT_TAG = SB.FastFourierTransformSpectralBackend()
const TK_SERIAL, TK_THREADED, TK_DEVICE = CB.SerialBackend(), CB.ThreadedBackend(), CB.GPUBackend(KA.CPU())

# Grid coordinates as a point list, in the cell order the packed field uses.
function _tensor_grid_points(dims, spacing)
    Dg = length(dims)
    x = zeros(Float64, Dg, prod(dims))
    for (k, I) in enumerate(CartesianIndices(dims)), d in 1:Dg
        x[d, k] = (I[d] - 1) * spacing[d]
    end
    return x
end

# The rank-P moment tensor of a point set by brute force: Σ over pairs of δu^{⊗P}, per (lo, hi] bin,
# an odd rank read from the lower to the upper end along the first coordinate that separates the pair.
function _brute_tensor(P, x, u, bins)
    D = size(u, 1)
    nb = length(bins) - 1
    sums = zeros(ntuple(_ -> D, P)..., nb)
    counts = zeros(Int, nb)
    N = size(x, 2)
    for i in 1:(N - 1), j in (i + 1):N
        dx = x[:, j] .- x[:, i]
        r = LA.norm(dx)
        b = searchsortedfirst(bins, r) - 1
        1 <= b <= nb || continue
        du = u[:, j] .- u[:, i]
        if isodd(P)
            lead = dx[findfirst(!iszero, dx)]
            du = (lead > 0 ? 1 : -1) .* du
        end
        for I in CartesianIndices(ntuple(_ -> D, P))
            sums[I, b] += prod(du[I[k]] for k in 1:P)
        end
        counts[b] += 1
    end
    return sums, counts
end

# (dims, spacing) of each grid.
const TK_GRIDS = (((9, 6), (0.1, 0.2)), ((5, 4, 4), (0.2, 0.25, 0.3)))

# (grid, order, tag, backend, field): every grid, order, tag and field kind.
const TK_TRANSFORM_CASES = (
    (1, 4, SB.AutoSpectralBackend(), TK_SERIAL, :plain),
    (1, 3, FFT_TAG, TK_SERIAL, :masked),
    (2, 2, FFT_TAG, TK_SERIAL, :weighted),
)

Test.@testset "the tensor from the transform equals the point tensor on a grid's points" begin
    Random.seed!(4100)
    geo = FG.Geometry.CartesianGeometry()
    grid_of(dims, spacing) =
        FG.Grids.StructuredGrid(geo, ntuple(d -> range(0.0, step = spacing[d], length = dims[d]), length(dims))...)
    bins_of(dims, spacing) = collect(range(0.0, 0.7 * sum(d -> spacing[d] * dims[d], 1:length(dims)); length = 7)) .+ 1e-3
    for (gi, P, tag, backend, field) in TK_TRANSFORM_CASES
        dims, spacing = TK_GRIDS[gi]
        Dg = length(dims)
        N = prod(dims)
        grid = grid_of(dims, spacing)
        bins = bins_of(dims, spacing)
        u = randn(Dg, dims...)
        x = _tensor_grid_points(dims, spacing)
        case = (dims, P, field, backend)
        if field === :masked
            # a masked field: the point tensor over the held cells
            umf = reshape(u, Dg, N)
            held = rand(N) .< 0.75
            umf[1, .!held] .= NaN
            ref = SFC.calculate_structure_function_tensor(Val(P), x[:, held], umf[:, held], bins, RAW_T;
                                                          backend = TK_SERIAL)
            got = SFC.calculate_structure_function_tensor(Val(P), grid, u, bins, tag, RAW_T; backend)
            Test.@test (case, got.counts == ref.counts) == (case, true)
        else
            ref = SFC.calculate_structure_function_tensor(Val(P), x, reshape(u, Dg, N), bins, RAW_T; backend = TK_SERIAL)
            if field === :weighted
                # weights of one change nothing but the count type
                got = SFC.calculate_structure_function_tensor(Val(P), grid, u, bins, tag, Float64, RAW_T;
                                                              weights = ones(N), backend)
                Test.@test (case, got.counts ≈ ref.counts) == (case, true)
            else
                got = SFC.calculate_structure_function_tensor(Val(P), grid, u, bins, tag, RAW_T; backend)
                Test.@test (case, got.counts == ref.counts) == (case, true)
            end
        end
        Test.@test (case, isapprox(got.sums, ref.sums; rtol = 1e-9, atol = 1e-10 * maximum(abs, ref.sums))) ==
                   (case, true)
    end
    # the averaged tensor is the default representation
    dims, spacing = TK_GRIDS[1]
    Dg, N = length(dims), prod(dims)
    grid, bins = grid_of(dims, spacing), bins_of(dims, spacing)
    u = randn(Dg, dims...)
    mean = SFC.calculate_structure_function_tensor(Val(2), grid, u, bins, FFT_TAG)
    ref2 = SFC.calculate_structure_function_tensor(Val(2), _tensor_grid_points(dims, spacing), reshape(u, Dg, N), bins;
                                                   backend = TK_SERIAL)
    Test.@test isapprox(mean.values, ref2.values; rtol = 1e-9, atol = 1e-10, nans = true)
    # one algorithm: the direct sum is refused, a multi-field of fields is refused
    Test.@test_throws ArgumentError SFC.calculate_structure_function_tensor(Val(2), grid, u, bins,
                                                                            SB.DirectSumSpectralBackend())
    Test.@test_throws ArgumentError SFC.calculate_structure_function_tensor(
        Val(2), grid, MF.Fields(vectors = (u,), scalars = (randn(dims...),)), bins, FFT_TAG)
end

Test.@testset "the tensor on a lat-lon grid is the point tensor in the geodesic frame" begin
    Random.seed!(4110)
    n_lon, n_lat = 15, 7
    lam = range(0.0, step = 2π / n_lon, length = n_lon)
    phi = range(-1.1, step = 0.35, length = n_lat)
    grid = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), lam, phi)
    u = randn(2, n_lon, n_lat)
    coords = FG.Grids.materialize(grid)
    x = Matrix(hcat(coords[1], coords[2])')
    bins = collect(range(0.0, π; length = 7)) .+ 1e-3
    ref = SFC.calculate_structure_function_tensor(Val(3), x, reshape(u, 2, :), bins, RAW_T; backend = TK_SERIAL,
                                                  distance_metric = SFH.SphericalDistance(1.0))
    got = SFC.calculate_structure_function_tensor(Val(3), grid, u, bins, FFT_TAG, RAW_T; backend = TK_SERIAL)
    Test.@test got.counts == ref.counts
    Test.@test isapprox(got.sums, ref.sums; rtol = 1e-9, atol = 1e-10 * maximum(abs, ref.sums))
end

Test.@testset "orders 1, 4 and 5 on points match brute force" begin
    Random.seed!(4120)
    N = 40
    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.0, 1.2; length = 6))
    for P in (1, 4, 5)
        ref_s, ref_c = _brute_tensor(P, x, u, bins)
        got = SFC.calculate_structure_function_tensor(Val(P), x, u, bins, RAW_T; backend = TK_SERIAL)
        Test.@test (P, got.counts == ref_c) == (P, true)
        Test.@test (P, isapprox(got.sums, ref_s; rtol = 1e-11, atol = 1e-12)) == (P, true)
    end
    Test.@test_throws ArgumentError SFT.MomentTensorOperator{2}()(SA.SVector(1.0, 0.0), SA.SVector(1.0, 0.0))
end

Test.@testset "the device tensor over batches adds in place" begin
    # an odd order over slices sharing positions, integer counts
    Random.seed!(4125)
    N, B, P = 200, 3, 3
    bins = collect(range(0.0, 0.2; length = 6))
    x, u = rand(2, N), randn(2, N, B)
    ref = SFC.calculate_structure_function_tensor(Val(P), x, u, bins, Int, RAW_T; backend = TK_SERIAL)
    got = SFC.calculate_structure_function_tensor(Val(P), x, u, bins, Int, RAW_T; backend = TK_DEVICE)
    Test.@test got.counts == ref.counts
    Test.@test isapprox(got.sums, ref.sums; rtol = 1e-10, atol = 1e-12)
    s, c = zeros(size(ref.sums)), zeros(Int, size(ref.counts))
    for _ in 1:2
        SFC.calculate_structure_function_tensor!(s, c, Val(P), x, u, bins; backend = TK_DEVICE)
    end
    Test.@test c == 2 .* ref.counts
    Test.@test isapprox(s, 2 .* ref.sums; rtol = 1e-10, atol = 1e-12)
end

Test.@testset "the joint tensor over angle marginalises to the tensor and to the joint histogram" begin
    Random.seed!(4130)
    N = 60
    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.0, 1.0; length = 6))
    θbins = collect(range(prevfloat(0.0), π; length = 5))
    axis = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    t1 = SFC.calculate_structure_function_tensor(Val(2), x, u, bins, RAW_T; backend = TK_SERIAL)
    # the trace of the joint tensor is the joint histogram of S2
    s2 = SFC.calculate_structure_function(SFT.S2SFType(), x, u, bins, θbins; backend = TK_SERIAL, second_axis = axis)
    for backend in (TK_SERIAL, TK_THREADED)
        joint = SFC.calculate_structure_function_tensor(Val(2), x, u, bins, θbins; second_axis = axis, backend)
        Test.@test size(joint.sums) == (2, 2, 5, 4)
        Test.@test dropdims(sum(joint.counts; dims = 2); dims = 2) == t1.counts
        Test.@test isapprox(dropdims(sum(joint.sums; dims = 4); dims = 4), t1.sums; rtol = 1e-11, atol = 1e-12)
        Test.@test joint.counts == s2.counts
        Test.@test isapprox(joint.sums[1, 1, :, :] .+ joint.sums[2, 2, :, :], s2.sums; rtol = 1e-11, atol = 1e-12)
    end
    Test.@test_throws ArgumentError SFC.calculate_structure_function_tensor(Val(2), x, cat(u, 2u; dims = 3), bins, θbins;
                                                                            second_axis = axis)
    # on a grid, from the transform
    dims, spacing = (9, 6), (0.1, 0.2)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, step = 0.1, length = 9),
                                   range(0.0, step = 0.2, length = 6))
    ug = randn(2, dims...)
    xg = _tensor_grid_points(dims, spacing)
    gbins = collect(range(0.0, 1.1; length = 6)) .+ 1e-3
    # angle edges placed between the angles a lattice can produce (its diagonals sit exactly on π/4 and 3π/4)
    gθbins = [prevfloat(0.0); collect(range(0.3011, π - 0.3; length = 5)); π + 1e-9]
    refj = SFC.calculate_structure_function_tensor(Val(2), xg, reshape(ug, 2, :), gbins, gθbins; second_axis = axis,
                                                   backend = TK_SERIAL)
    gotj = SFC.calculate_structure_function_tensor(Val(2), grid, ug, gbins, gθbins, FFT_TAG; second_axis = axis,
                                                   backend = TK_SERIAL)
    Test.@test gotj.counts ≈ refj.counts
    Test.@test isapprox(gotj.sums, refj.sums; rtol = 1e-9, atol = 1e-10 * maximum(abs, refj.sums))
end
