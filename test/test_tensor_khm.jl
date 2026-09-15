using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
using Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using Random: Random
using Test: Test

Test.@testset "Tensor Structure Functions" begin
    x = [0.0 1.0 0.0; 0.0 0.0 1.0]
    u = [0.0 2.0 3.0; 0.0 5.0 7.0]
    bins = [0.0, 1.1, 2.0]

    t2 = SFC.calculate_structure_function_tensor(
        Val(2), x, u, bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts,
    )
    Test.@test t2 isa SF.StructureFunctionTensorSumsAndCounts{2}
    Test.@test size(t2.sums) == (2, 2, 2)
    Test.@test size(t2.counts) == (2,)
    Test.@test t2.counts == UInt32[2, 1]

    du12 = u[:, 2] - u[:, 1]
    du13 = u[:, 3] - u[:, 1]
    du23 = u[:, 3] - u[:, 2]
    expected_bin1 = du12 * du12' + du13 * du13'
    expected_bin2 = du23 * du23'
    Test.@test t2.sums[:, :, 1] ≈ expected_bin1
    Test.@test t2.sums[:, :, 2] ≈ expected_bin2

    # Default output is the averaged mean tensor D_ij(r) = sums ./ counts (per distance bin).
    t2_mean = SFC.calculate_structure_function_tensor(Val(2), x, u, bins; backend = CB.SerialBackend())
    Test.@test t2_mean isa SF.StructureFunctionTensor{2}
    Test.@test t2_mean.values[:, :, 1] ≈ expected_bin1 ./ 2
    Test.@test t2_mean.values[:, :, 2] ≈ expected_bin2 ./ 1

    s2 = SFC.calculate_structure_function(
        SFT.S2SF, x, u, bins; backend = CB.SerialBackend(), output_type = SF.StructureFunctionSumsAndCounts
    )
    trace_sums = [sum(t2.sums[a, a, bin] for a in 1:2) for bin in axes(t2.sums, 3)]
    Test.@test trace_sums ≈ s2.sums
    Test.@test t2.counts == s2.counts

    t3 = SFC.calculate_structure_function_tensor(
        Val(3), x, u, bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts,
    )
    Test.@test t3 isa SF.StructureFunctionTensorSumsAndCounts{3}
    Test.@test size(t3.sums) == (2, 2, 2, 2)
    Test.@test t3.counts == t2.counts
    Test.@test t3.sums[1, 2, 2, 1] ≈ du12[1] * du12[2] * du12[2] +
        du13[1] * du13[2] * du13[2]

    u_aux = cat(u, 2u; dims = 3)
    t2_aux = SFC.calculate_structure_function_tensor(
        Val(2), x, u_aux, bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts,
    )
    Test.@test size(t2_aux.sums) == (2, 2, 2, 2)
    Test.@test size(t2_aux.counts) == (2, 2)
    Test.@test t2_aux.sums[:, :, :, 1] ≈ t2.sums
    Test.@test t2_aux.sums[:, :, :, 2] ≈ 4 .* t2.sums
end

Test.@testset "KHM Diagnostics" begin
    r = [1.0, 2.0, 3.0, 4.0]
    DLL = r .^ 2
    DTT = DLL .+ r .* SF.KHM.finite_difference(r, DLL) ./ 2
    Test.@test SF.KHM.transverse_incompressibility_residual(r, DLL, DTT; dimension = 3) ≈ zeros(4)

    ε = 0.2
    S3 = .-(4 / 5) .* ε .* r
    Test.@test SF.KHM.epsilon_from_four_fifths(r, S3) ≈ fill(ε, length(r))
    Test.@test SF.KHM.four_fifths_residual(r, S3, ε) ≈ zeros(length(r))
end

Test.@testset "each inertial-range law takes the quantity it is stated for" begin
    # §9.4: the four-fifths law is for ⟨δu_L³⟩ (L3SF) and the four-thirds law for ⟨δu_L‖δu‖²⟩
    # (S3SF). They differ by 5/3, so handing one function the other's quantity returns a wrong ε
    # that looks entirely plausible. These pin that the two are distinct and each inverts its own law.
    r = collect(range(0.1, 2.0; length = 9))
    eps = 0.37

    # a field obeying the four-fifths law exactly
    L3 = -(4 / 5) .* eps .* r
    Test.@test SF.KHM.epsilon_from_four_fifths(r, L3) ≈ fill(eps, length(r))
    Test.@test all(abs.(SF.KHM.four_fifths_residual(r, L3, eps)) .< 1e-12)

    # a field obeying the four-thirds law exactly
    S3 = -(4 / 3) .* eps .* r
    Test.@test SF.KHM.epsilon_from_four_thirds(r, S3) ≈ fill(eps, length(r))
    Test.@test all(abs.(SF.KHM.four_thirds_residual(r, S3, eps)) .< 1e-12)

    # the two are NOT interchangeable: feeding one quantity to the other's inverse is off by 5/3
    wrong = SF.KHM.epsilon_from_four_fifths(r, S3)
    Test.@test all(wrong ./ eps .≈ 5 / 3)

    # Yaglom returns the scalar dissipation, on its own law
    eps_th = 0.21
    LS2 = -(4 / 3) .* eps_th .* r
    Test.@test SF.KHM.epsilon_theta_from_yaglom(r, LS2) ≈ fill(eps_th, length(r))
    Test.@test all(abs.(SF.KHM.yaglom_residual(r, LS2, eps_th)) .< 1e-12)
end

Test.@testset "every backend computes the same tensor" begin
    # The pair set is the same upper triangle on every backend, and a histogram is order-independent,
    # so counts must be exactly equal and sums must agree to summation order.
    Random.seed!(4)
    N = 200
    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.0, 1.2; length = 9))

    ref = SFC.calculate_structure_function_tensor(
        Val(2), x, u, bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts)

    for be in (CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
        got = SFC.calculate_structure_function_tensor(
            Val(2), x, u, bins; backend = be,
            output_type = SF.StructureFunctionTensorSumsAndCounts)
        Test.@test got.counts == ref.counts
        Test.@test isapprox(got.sums, ref.sums; rtol = 1e-10, atol = 1e-12)
    end

    # order 3 as well, since the accumulator rank is a type parameter
    r3 = SFC.calculate_structure_function_tensor(
        Val(3), x, u, bins; backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts)
    t3 = SFC.calculate_structure_function_tensor(
        Val(3), x, u, bins; backend = CB.ThreadedBackend(),
        output_type = SF.StructureFunctionTensorSumsAndCounts)
    Test.@test t3.counts == r3.counts
    Test.@test isapprox(t3.sums, r3.sums; rtol = 1e-10, atol = 1e-12)

    # and on a sphere, where the increments are transported rather than differenced
    xs = vcat(reshape(2π .* rand(N), 1, N), reshape((rand(N) .- 0.5) .* 1.4, 1, N))
    sbins = collect(range(0.0, 2.4; length = 6))
    rs = SFC.calculate_structure_function_tensor(
        Val(2), xs, u, sbins; backend = CB.SerialBackend(),
        distance_metric = DI.SphericalAngle(),
        output_type = SF.StructureFunctionTensorSumsAndCounts)
    ts = SFC.calculate_structure_function_tensor(
        Val(2), xs, u, sbins; backend = CB.ThreadedBackend(),
        distance_metric = DI.SphericalAngle(),
        output_type = SF.StructureFunctionTensorSumsAndCounts)
    Test.@test sum(ts.counts) > 0
    Test.@test ts.counts == rs.counts
    Test.@test isapprox(ts.sums, rs.sums; rtol = 1e-10, atol = 1e-12)
end

# --- Tensors from the transform, higher orders, and the joint tensor over angle ---

const RAW_T = SF.StructureFunctionTensorSumsAndCounts
const RAW_T2 = SF.StructureFunctionTensor2DSumsAndCounts
const FFT_TAG = SB.FastFourierTransformSpectralBackend()

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

Test.@testset "the tensor from the transform equals the point tensor on a grid's points" begin
    Random.seed!(4100)
    geo = FG.Geometry.CartesianGeometry()
    for (dims, spacing, orders) in (((9, 6), (0.1, 0.2), (2, 3, 4)), ((5, 4, 4), (0.2, 0.25, 0.3), (2, 3)))
        Dg = length(dims)
        N = prod(dims)
        grid = FG.Grids.StructuredGrid(geo, ntuple(d -> range(0.0, step = spacing[d], length = dims[d]), Dg)...)
        u = randn(Dg, dims...)
        x = _tensor_grid_points(dims, spacing)
        r_max = 0.7 * sum(d -> spacing[d] * dims[d], 1:Dg)
        bins = collect(range(0.0, r_max; length = 7)) .+ 1e-3
        for P in orders
            ref = SFC.calculate_structure_function_tensor(Val(P), x, reshape(u, Dg, N), bins; backend = CB.SerialBackend(),
                                                          output_type = RAW_T)
            for tag in (FFT_TAG, SB.AutoSpectralBackend()), backend in (CB.SerialBackend(), CB.ThreadedBackend())
                got = SFC.calculate_structure_function_tensor(Val(P), grid, u, bins, tag; backend, verbose = false,
                                                              output_type = RAW_T)
                Test.@test got.counts == ref.counts
                Test.@test isapprox(got.sums, ref.sums; rtol = 1e-9, atol = 1e-10 * maximum(abs, ref.sums))
            end
            # a masked field: the point tensor over the held cells
            um = copy(u)
            umf = reshape(um, Dg, N)
            held = rand(N) .< 0.75
            umf[1, .!held] .= NaN
            refm = SFC.calculate_structure_function_tensor(Val(P), x[:, held], umf[:, held], bins;
                                                           backend = CB.SerialBackend(), output_type = RAW_T)
            gotm = SFC.calculate_structure_function_tensor(Val(P), grid, um, bins, FFT_TAG; verbose = false,
                                                           output_type = RAW_T)
            Test.@test gotm.counts == refm.counts
            Test.@test isapprox(gotm.sums, refm.sums; rtol = 1e-9, atol = 1e-10 * maximum(abs, refm.sums))
            # weights of one change nothing but the count type
            gotw = SFC.calculate_structure_function_tensor(Val(P), grid, u, bins, FFT_TAG; weights = ones(N),
                                                           count_eltype = Float64, verbose = false, output_type = RAW_T)
            Test.@test gotw.counts ≈ ref.counts
            Test.@test isapprox(gotw.sums, ref.sums; rtol = 1e-9, atol = 1e-10 * maximum(abs, ref.sums))
        end
        # the averaged tensor is the default representation
        mean = SFC.calculate_structure_function_tensor(Val(2), grid, u, bins, FFT_TAG; verbose = false)
        Test.@test mean isa SF.StructureFunctionTensor{2}
        ref2 = SFC.calculate_structure_function_tensor(Val(2), x, reshape(u, Dg, N), bins; backend = CB.SerialBackend())
        Test.@test isapprox(mean.values, ref2.values; rtol = 1e-9, atol = 1e-10, nans = true)
        # one algorithm: the direct sum is refused, a bundle of channels is refused
        Test.@test_throws ArgumentError SFC.calculate_structure_function_tensor(Val(2), grid, u, bins,
                                                                                SB.DirectSumSpectralBackend(); verbose = false)
        Test.@test_throws ArgumentError SFC.calculate_structure_function_tensor(
            Val(2), grid, SF.Fields(vectors = (u,), scalars = (randn(dims...),)), bins, FFT_TAG; verbose = false)
    end
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
    for P in (2, 3)
        ref = SFC.calculate_structure_function_tensor(Val(P), x, reshape(u, 2, :), bins; backend = CB.SerialBackend(),
                                                      distance_metric = SF.SphericalDistance(1.0), output_type = RAW_T)
        got = SFC.calculate_structure_function_tensor(Val(P), grid, u, bins, FFT_TAG; verbose = false, output_type = RAW_T)
        Test.@test got.counts == ref.counts
        Test.@test isapprox(got.sums, ref.sums; rtol = 1e-9, atol = 1e-10 * maximum(abs, ref.sums))
        Test.@test sum(ref.counts) > 0
    end
end

Test.@testset "higher orders on points match brute force on every CPU backend" begin
    Random.seed!(4120)
    N = 40
    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.0, 1.2; length = 6))
    for P in (1, 4, 5)
        ref_s, ref_c = _brute_tensor(P, x, u, bins)
        for backend in (CB.SerialBackend(), CB.ThreadedBackend())
            got = SFC.calculate_structure_function_tensor(Val(P), x, u, bins; backend, output_type = RAW_T)
            Test.@test got.counts == ref_c
            Test.@test isapprox(got.sums, ref_s; rtol = 1e-11, atol = 1e-12)
        end
    end
    Test.@test_throws ArgumentError SFC.calculate_structure_function_tensor(Val(4), x, u, bins;
                                                                            backend = CB.GPUBackend(KA.CPU()))
    Test.@test_throws ArgumentError SFT.MomentTensorOperator{2}()(SA.SVector(1.0, 0.0), SA.SVector(1.0, 0.0))
end

Test.@testset "the joint tensor over angle marginalises to the tensor and to the joint histogram" begin
    Random.seed!(4130)
    N = 60
    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.0, 1.0; length = 6))
    θbins = collect(range(prevfloat(0.0), π; length = 5))
    axis = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    t1 = SFC.calculate_structure_function_tensor(Val(2), x, u, bins; backend = CB.SerialBackend(), output_type = RAW_T)
    for backend in (CB.SerialBackend(), CB.ThreadedBackend())
        joint = SFC.calculate_structure_function_tensor(Val(2), x, u, bins, θbins; second_axis = axis, backend)
        Test.@test joint isa RAW_T2
        Test.@test size(joint.sums) == (2, 2, 5, 4)
        Test.@test dropdims(sum(joint.counts; dims = 2); dims = 2) == t1.counts
        Test.@test isapprox(dropdims(sum(joint.sums; dims = 4); dims = 4), t1.sums; rtol = 1e-11, atol = 1e-12)
        # the trace of the joint tensor is the joint histogram of S2
        s2 = SFC.serial_calculate_structure_function(SFT.S2SFType(), x, u, bins, θbins; second_axis = axis,
                                                     verbose = false, show_progress = false)
        Test.@test joint.counts == s2.counts
        Test.@test isapprox(joint.sums[1, 1, :, :] .+ joint.sums[2, 2, :, :], s2.sums; rtol = 1e-11, atol = 1e-12)
    end
    Test.@test_throws ArgumentError SFC.calculate_structure_function_tensor(Val(2), x, u, bins, θbins; second_axis = axis,
                                                                            backend = CB.GPUBackend(KA.CPU()))
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
    for P in (2, 3)
        refj = SFC.calculate_structure_function_tensor(Val(P), xg, reshape(ug, 2, :), gbins, gθbins; second_axis = axis,
                                                       backend = CB.SerialBackend())
        gotj = SFC.calculate_structure_function_tensor(Val(P), grid, ug, gbins, gθbins, FFT_TAG; second_axis = axis,
                                                       verbose = false)
        Test.@test gotj isa RAW_T2
        Test.@test gotj.counts ≈ refj.counts
        Test.@test isapprox(gotj.sums, refj.sums; rtol = 1e-9, atol = 1e-10 * maximum(abs, refj.sums))
    end
end
