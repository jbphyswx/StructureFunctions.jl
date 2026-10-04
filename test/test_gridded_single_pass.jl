using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using FFTW: FFTW
using Random: Random

const SP = SFT.SinglePassInvariants()
const SP_KEYS = (:S2, :L2, :T2, :S3, :L3, :L1T2)
const SP_DIRECT, SP_FFT, SP_AUTO = SB.DirectSumSpectralBackend(), SB.FastFourierTransformSpectralBackend(),
                                   SB.AutoSpectralBackend()
const SP_SERIAL, SP_THREADED, SP_DEVICE = CB.SerialBackend(), CB.ThreadedBackend(), CB.GPUBackend(KA.CPU())

# Each schedule takes both algorithms and one field variant; each algorithm takes every backend, single and batched.
const SP_GRID_CASES = (
    (schedule = :uniform, variant = :masked,
     single = ((SP_DIRECT, SP_SERIAL), (SP_FFT, SP_THREADED), (SP_AUTO, SP_SERIAL)),
     batch = ((SP_DIRECT, SP_THREADED), (SP_FFT, SP_DEVICE))),
    (schedule = :rectilinear, variant = :weighted,
     single = ((SP_DIRECT, SP_THREADED), (SP_FFT, SP_DEVICE)),
     batch = ((SP_DIRECT, SP_DEVICE), (SP_FFT, SP_SERIAL))),
    (schedule = :zonal, variant = :plain,
     single = ((SP_DIRECT, SP_DEVICE), (SP_FFT, SP_SERIAL)),
     batch = ((SP_DIRECT, SP_SERIAL), (SP_FFT, SP_THREADED))),
)

_sp_close(a, b) = isapprox(a, b; rtol = 1e-9, atol = 1e-9 * max(1, maximum(abs, b)))

# Row q of the single-pass histogram is the grid sweep of the q-th single-pass operator.
function _sp_rows(u, s, bins, valid, w, ::Type{CT}) where {CT}
    nb = length(bins) - 1
    sums, counts = zeros(6, nb), zeros(CT, 6, nb)
    for (q, op) in enumerate(values(SFC.SINGLE_PASS_OPERATORS))
        s1, c1 = zeros(nb), zeros(CT, nb)
        SFC.gridded_lag_sweep!(s1, c1, op, u, s, bins, Val(2); valid, weights = w, backend = CB.SerialBackend())
        sums[q, :] .= s1
        counts[q, :] .= c1
    end
    return sums, counts
end

# Whether every invariant of the single-pass result `r` matches the point entry's `ref`.
_sp_matches(r, ref) = all(k -> _sp_close(r[k].sums, ref[k].sums) && r[k].counts == ref[k].counts, SP_KEYS)

Test.@testset "each single-pass row is its operator's grid sweep, on every algorithm and backend" begin
    rng = Random.Xoshiro(606)
    ys = cumsum(0.05 .+ 0.2 .* rand(rng, 7))
    lats = collect(range(-0.9, 0.9; length = 6))
    schedules = (
        uniform = (SFC.UniformLagSchedule((10, 8), (0.1, 0.12), (true, true)), (10, 8),
                   collect(range(0.0, 0.5; length = 6))),
        rectilinear = (SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), (ys,), (1, 2)),
                       (9, 7), collect(range(0.0, 0.9; length = 7))),
        zonal = (SFC.ZonalLagSchedule(lats, 12, 2π / 12, 1.0, true), (12, 6), collect(range(0.0, 1.6; length = 7))),
    )
    single_ok, batch_ok = Bool[], Bool[]
    for case in SP_GRID_CASES
        (; schedule, variant, single, batch) = case
        s, dims, edges = schedules[schedule]
        bins = edges .+ 0.0137
        nb = length(bins) - 1
        u = randn(rng, 2, dims...)
        variant === :masked && (u[:, 3, 2] .= NaN; u[:, 5, 4] .= NaN)
        valid = SFC.field_validity(u)
        w = variant === :weighted ? 0.5 .+ rand(rng, prod(dims)) : nothing
        CT = w === nothing ? Int : Float64
        rs, rc = _sp_rows(u, s, bins, valid, w, CT)
        for (tag, backend) in single
            gs, gc = zeros(6, nb), zeros(CT, 6, nb)
            SFC.gridded_sweep!(gs, gc, SP, u, s, bins, Val(2), tag; valid, weights = w, backend)
            push!(single_ok, sum(rc) > 0 && _sp_close(gc, rc) && _sp_close(gs, rs))
        end
        ub = randn(rng, 2, dims..., 2)
        variant === :masked && (ub[:, 3, 2, 1] .= NaN; ub[:, 5, 4, 2] .= NaN)
        bv = SFC.batch_validity(ub)
        rb, rcb = zeros(6, nb, 2), zeros(CT, 6, nb, 2)
        for t in 1:2
            ut = ub[:, :, :, t]
            r1, c1 = _sp_rows(ut, s, bins, SFC.field_validity(ut), w, CT)
            rb[:, :, t] .= r1
            rcb[:, :, t] .= c1
        end
        for (tag, backend) in batch
            gb, gcb = zeros(6, nb, 2), zeros(CT, 6, nb, 2)
            SFC.gridded_sweep_batch!(gb, gcb, SP, ub, s, bins, Val(2), tag; valid = bv, weights = w, backend)
            push!(batch_ok, _sp_close(gcb, rcb) && _sp_close(gb, rb))
        end
    end
    Test.@test all(single_ok)
    Test.@test all(batch_ok)
end

Test.@testset "the grid single-pass entry is the point entry over the cell centres" begin
    # A Cartesian grid on each algorithm, a sphere, no uniform axis, a batch; a multi-vector field is refused.
    rng = Random.Xoshiro(607)
    geo = FG.Geometry.CartesianGeometry()
    grid = FG.Grids.StructuredGrid(geo, range(0.0, step = 0.1, length = 9), range(0.0, step = 0.13, length = 7))
    x = reshape(Float64[(d == 1 ? (i - 1) * 0.1 : (j - 1) * 0.13) for d in 1:2, i in 1:9, j in 1:7], 2, :)
    u = randn(rng, 2, 9, 7)
    bins = collect(range(0.0, 0.8; length = 7)) .+ 0.0137
    ref = SFC.calculate_structure_functions_single_pass(x, reshape(u, 2, :), bins; backend = CB.SerialBackend())
    agree = Bool[]
    for (tag, backend) in ((SP_DIRECT, SP_SERIAL), (SP_FFT, SP_DEVICE), (SP_AUTO, SP_DEVICE))
        got = SFC.calculate_structure_functions_single_pass(grid, u, bins, tag; backend)
        push!(agree, haskey(got, :helmholtz) && _sp_matches(got, ref))
    end
    lam = range(0.0, step = 2π / 11, length = 11)
    phi = range(-1.0, step = 0.4, length = 6)
    sgrid = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), lam, phi)
    us = randn(rng, 2, 11, 6)
    coords = FG.Grids.materialize(sgrid)
    xs = Matrix(hcat(coords[1], coords[2])')
    sbins = collect(range(0.0, π; length = 7)) .+ 1e-3
    sref = SFC.calculate_structure_functions_single_pass(xs, reshape(us, 2, :), sbins; backend = CB.SerialBackend(),
                                                         distance_metric = SF.HelperFunctions.SphericalDistance(1.0))
    push!(agree, _sp_matches(SFC.calculate_structure_functions_single_pass(sgrid, us, sbins, SP_FFT), sref))
    xl = cumsum(0.05 .+ 0.1 .* rand(rng, 7))
    yl = cumsum(0.05 .+ 0.2 .* rand(rng, 7))
    free = FG.Grids.StructuredGrid(geo, xl, yl)
    uf = randn(rng, 2, 7, 7)
    xf = reshape(Float64[(d == 1 ? xl[i] : yl[j]) for d in 1:2, i in 1:7, j in 1:7], 2, :)
    rf = SFC.calculate_structure_functions_single_pass(xf, reshape(uf, 2, :), bins; backend = CB.SerialBackend())
    push!(agree, _sp_matches(SFC.calculate_structure_functions_single_pass(free, uf, bins), rf))
    ub = randn(rng, 2, 9, 7, 2)
    gb, gcb = zeros(6, 6, 2), zeros(Int, 6, 6, 2)
    SFC.calculate_structure_functions_single_pass_batch!(gb, gcb, grid, ub, bins, SB.FastFourierTransformSpectralBackend())
    for t in 1:2
        r = SFC.calculate_structure_functions_single_pass(x, reshape(ub[:, :, :, t], 2, :), bins;
                                                          backend = CB.SerialBackend())
        push!(agree, all(((q, k),) -> _sp_close(gb[q, :, t], r[k].sums) && gcb[q, :, t] == r[k].counts,
                         enumerate(SP_KEYS)))
    end
    Test.@test all(agree)
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(zeros(6, 6), zeros(Int, 6, 6), SP, randn(rng, 5, 63),
                                                           SFC.UniformLagSchedule((9, 7), (0.1, 0.13), (false, false)),
                                                           bins, Val(2), Val(2), Val(1))
end
