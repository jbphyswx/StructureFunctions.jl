# Single-Pass 1D and 2D Calculations

# --- Helmholtz Decomposition Derived Quantities ---

"""
    helmholtz_decompose_2d(distance_bins, sums, counts)

Run the 2D isotropic Helmholtz decomposition from native single-pass sums/counts.
Rows 2 and 3 must contain `L2SF` and `T2SF`.
"""
function helmholtz_decompose_2d(
    distance_bins::AbstractVector{FT3},
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
) where {OT, CT, FT3}
    return helmholtz_decompose_2d(
        distance_bins,
        @view(sums[2, :]),
        @view(counts[2, :]),
        @view(sums[3, :]),
        @view(counts[3, :]),
    )
end

"""
    helmholtz_decompose_2d(distance_bins, L2_sums, L2_counts, T2_sums, T2_counts)

Run the 2D isotropic Helmholtz decomposition using the trapezoidal rule over
binned longitudinal/transverse second-order structure functions. This implements
the cumulative integral equations described by Lindborg (JFM 2015) and
Bühler, Callies, and Ferrari (JFM 2014):

``D_rot(r) = D_TT(r) + I(r)``, ``D_div(r) = D_LL(r) - I(r)``, where
``I(r) = ∫_0^r [D_TT(s) - D_LL(s)]/s ds``.

The integral runs from zero separation, where the integrand of a differentiable field vanishes, by
the trapezoid rule over the bin midpoints.
"""
function helmholtz_decompose_2d(
    distance_bins::AbstractVector{FT3},
    L2_sums::AbstractVector{OT},
    L2_counts::AbstractVector{CT},
    T2_sums::AbstractVector,
    T2_counts::AbstractVector,
) where {OT, CT, FT3}
    length(L2_sums) == length(T2_sums) ||
        throw(DimensionMismatch("L2_sums and T2_sums must have the same length"))
    length(L2_counts) == length(L2_sums) ||
        throw(DimensionMismatch("L2_counts must match L2_sums length"))
    length(T2_counts) == length(T2_sums) ||
        throw(DimensionMismatch("T2_counts must match T2_sums length"))
    n_bins = length(L2_sums)
    length(distance_bins) == n_bins + 1 ||
        throw(DimensionMismatch("distance_bins must have length n_bins + 1"))

    D_LL = _bin_average(L2_sums, L2_counts)
    D_TT = _bin_average(T2_sums, T2_counts)
    bin_mids = similar(D_LL, OT, n_bins)
    copyto!(bin_mids, collect(OT, midpoints(distance_bins)))
    integrand = (D_TT .- D_LL) ./ bin_mids
    increments = similar(D_LL)
    n_bins >= 1 && @views(increments[1:1] .= integrand[1:1] .* bin_mids[1:1] ./ 2)
    if n_bins >= 2
        @views increments[2:end] .= (integrand[1:end-1] .+ integrand[2:end]) .*
            (bin_mids[2:end] .- bin_mids[1:end-1]) ./ 2
    end
    integral = cumsum(increments)
    rotational_counts = copy(T2_counts)
    divergent_counts = copy(L2_counts)
    rotational_sums = (D_TT .+ integral) .* rotational_counts
    divergent_sums = (D_LL .- integral) .* divergent_counts

    return SFO.HelmholtzDecomposition2D(
        distance_bins,
        rotational_sums,
        rotational_counts,
        divergent_sums,
        divergent_counts,
        D_LL,
        D_TT,
    )
end

"""
    append_helmholtz_rotational_divergent_rows(sums, counts, distance_bins)

Append Helmholtz-derived rotational/divergent rows to native six-row single-pass
sums and counts. Rows 1 through 6 are copied unchanged; rows 7 and 8 are
rotational and divergent second-order components.
"""
function append_helmholtz_rotational_divergent_rows(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    distance_bins::AbstractVector{FT3},
) where {OT, CT, FT3}
    n_bins = size(sums, 2)
    size(sums, 1) == SINGLE_PASS_N ||
        throw(DimensionMismatch("sums must have $SINGLE_PASS_N rows"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape"))
    decomposition = helmholtz_decompose_2d(distance_bins, sums, counts)
    final_sums = similar(sums, OT, SINGLE_PASS_WITH_HELMHOLTZ_N, n_bins)
    final_counts = similar(counts, CT, SINGLE_PASS_WITH_HELMHOLTZ_N, n_bins)

    final_sums[1:SINGLE_PASS_N, :] .= sums
    final_counts[1:SINGLE_PASS_N, :] .= counts
    final_sums[7, :] .= decomposition.rotational_sums
    final_counts[7, :] .= decomposition.rotational_counts
    final_sums[8, :] .= decomposition.divergent_sums
    final_counts[8, :] .= decomposition.divergent_counts

    return final_sums, final_counts
end

"""
    marginalize_sp2d_then_append_helmholtz_rows(sums_6, counts_6, distance_bins)

Marginalize six native invariant 2D joint histograms to 1D, then append
rotational/divergent rows via [`append_helmholtz_rotational_divergent_rows`](@ref).
Returns ``(sums_8, counts_8)``.
"""
function marginalize_sp2d_then_append_helmholtz_rows(
    sums_6::AbstractArray{OT, 3},
    counts_6::AbstractArray{CT, 3},
    distance_bins::AbstractVector{FT3},
) where {OT, CT, FT3}
    sums_1d = dropdims(sum(sums_6, dims = 3), dims = 3)
    counts_1d = dropdims(sum(counts_6, dims = 3), dims = 3)
    return append_helmholtz_rotational_divergent_rows(sums_1d, counts_1d, distance_bins)
end

# --- 1D Single Pass Functions ---

"""Serial pair-loop accumulation into native ``(6, n_bins)`` buffers."""
function _accumulate_single_pass_1d!(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    geometry,
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    n_points = size(x, 2)
    n_bins = length(distance_bins) - 1
    size(sums) == (SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("sums must have shape ($SINGLE_PASS_N, n_bins); got $(size(sums))"))
    size(counts) == (SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("counts must have shape ($SINGLE_PASS_N, n_bins); got $(size(counts))"))
    # Flat D ∈ {2,3}: SIMD compute/scatter split.
    vD = _simd_width(geometry)
    if vD !== nothing
        _sp_simd_run!(sums, counts, x, u, distance_bins, vD, culling, weights)
        return sums, counts
    end

    xk, uk = SFH.prepare_pair_inputs(geometry, x, u)
    be = digitize_plan(distance_bins)
    grid, xk, uk = cull_sorted_matrices(xk, uk, geometry, distance_bins, culling)
    wc = grid === nothing ? weights : _permuted_point_weights(weights, grid.perm)
    _sp1d_run_blocks!(sums, counts, xk, uk, be, geometry, n_bins, 1:n_points, n_points, grid, wc)
    return sums, counts
end

"""
    _sp1d_derive_rows!(sums, counts)

Fill the derived rows 3 and 6 and replicate the shared count of a `(SINGLE_PASS_N, n_bins)`
accumulator.

`T2 = S2 - L2` and `L1T2 = S3 - L3` hold for every pair, so the pair loops store only rows 1, 2, 4, 5
and the count in row 1. Assigns with `=`, so repeated calls on one buffer are idempotent.
"""
@inline function _sp1d_derive_rows!(sums::AbstractMatrix, counts::AbstractMatrix)
    @inbounds for b in axes(sums, 2)
        sums[3, b] = sums[1, b] - sums[2, b]
        sums[6, b] = sums[4, b] - sums[5, b]
        c = counts[1, b]
        for t in 2:SINGLE_PASS_N
            counts[t, b] = c
        end
    end
    return nothing
end

"""
    _pf_sp_simd_pairs!(sums, counts, xc, uc, plan, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, sel, window, blocks)

Single-pass (6 invariants) point-field SIMD compute/scatter kernel over the pairs `blocks` covers.
For each `i`: `@simd` over its `j` block computes distance, `du_L = du·r̂`, and `|du|²` into buffers
indexed by `window` ([`PairWindow`](@ref)), then a scalar loop scatters the 6 invariants of the in-range
pairs as [`_pf_simd_pairs!`](@ref) does. Consumes `(i-block, j-block)` pairs (see [`block_pairs`](@ref)).
"""
function _pf_sp_simd_pairs!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, ::Val{D},
    keybuf::AbstractVector, duLbuf::AbstractVector, dn2buf::AbstractVector,
    idxbuf::AbstractVector{Int32}, sel::AbstractVector{Int32}, window::PairWindow, blocks, weights = NoWeights(),
) where {OT, CT, D}
    nb = n_histogram_bins(plan)
    chooses = _chooses_compaction(blocks)
    FTx = eltype(xc[1])
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        _check_run_fits(window, duLbuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            wi = _point_weight(weights, i)
            Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
            Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
            @simd for j in jlo:j_last
                dx = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D))) - Xi
                r2 = SFH.fma_dot(dx, dx)
                du = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D))) - Ui
                keybuf[j - o] = digitize_key(plan, r2)
                duLbuf[j - o], dn2buf[j - o] = SFH.increment_invariants(SFH.FlatGeometry{D}(), dx, sqrt(r2), du)
                if has_vector_index(plan)
                    idxbuf[j - o] = squared_approx_index(plan, r2)
                end
            end
            ks = (jlo - o):(j_last - o)
            if chooses && _compacts(_sample_in_range(plan, keybuf, ks)...)
                for m in 1:_compact_in_range!(sel, plan, keybuf, ks)
                    k = Int(sel[m])
                    _sp1d_accumulate!(sums, counts, duLbuf, dn2buf, weights, wi, o, k,
                                      squared_bin_select(plan, keybuf[k], idxbuf[k]))
                end
            else
                for k in ks
                    bin = chooses ? squared_bin_select(plan, keybuf[k], idxbuf[k]) : squared_bin(plan, keybuf[k], idxbuf[k])
                    if 1 <= bin <= nb
                        _sp1d_accumulate!(sums, counts, duLbuf, dn2buf, weights, wi, o, k, bin)
                    end
                end
            end
        end
    end
    _sp1d_derive_rows!(sums, counts)
    return nothing
end

"""Add slot `k`'s single-pass sums (rows 1, 2, 4, 5) and count into bin `bin`, weighted by the outer point's
weight `wi` times that of point `k + o`."""
@inline function _sp1d_accumulate!(sums, counts::AbstractMatrix{CT}, duLbuf, dn2buf, weights, wi, o, k, bin) where {CT}
    @inbounds begin
        duL = duLbuf[k]
        dn2 = dn2buf[k]
        duL2 = duL * duL
        w = wi * _point_weight(weights, k + o)
        sums[1, bin] += w * dn2
        sums[2, bin] += w * duL2
        sums[4, bin] += w * duL * dn2
        sums[5, bin] += w * duL * duL2
        counts[1, bin] += CT(w)
    end
    return nothing
end

"""
    _sp_run_blocks!(sums, counts, xc, uc, plan, ::Val{D}, bufs..., ilist, N, grid)

Single-pass analogue of [`_pf_run_blocks!`](@ref): dispatch on `grid` so the kernel receives one
concretely typed schedule.
"""
@inline _sp_run_blocks!(
    sums, counts, xc, uc, plan, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, sel, ilist, N, ::Nothing,
    weights = NoWeights(),
) where {D} = _pf_sp_simd_pairs!(sums, counts, xc, uc, plan, Val(D), keybuf, duLbuf, dn2buf,
    idxbuf, sel, _pair_window(N), pair_blocks(N, ilist), weights)

@inline _sp_run_blocks!(
    sums, counts, xc, uc, plan, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, sel, ilist, N,
    grid::CellGrid, weights = NoWeights(),
) where {D} = _pf_sp_simd_pairs!(sums, counts, xc, uc, plan, Val(D), keybuf, duLbuf, dn2buf,
    idxbuf, sel, _pair_window(N), pair_blocks(N, ilist; grid = grid), weights)

function _sp_simd_run!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, ::Val{D},
    culling::CullingPolicy = AutoCulling(), weights = NoWeights(),
) where {OT, CT, D}
    return _sp_simd_partial!(sums, counts, x, u, dist_be, Val(D), nothing, culling, weights)
end

"""
    _partial_single_pass_1d(inner, x, u, distance_bins, share, CT; geometry, culling, weights)

Six-invariant partial sums/counts over share `share = (w, k)` of the outer indices ([`_share_indices`](@ref)), for
one distributed worker or MPI rank, computed on its local backend `inner`; this method runs serially, and the
OhMyThreads extension threads a threaded `inner`. Euclidean `D ∈ {2,3}` takes the SIMD kernel; other metrics use the
scalar loop. The sum element type comes from the inputs.
"""
function _partial_single_pass_1d(
    ::CB.AbstractExecutionBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    share::NTuple{2, Int},
    ::Type{CT};
    geometry,
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    nb = n_histogram_bins(distance_bins)
    sums = zeros(OT, SINGLE_PASS_N, nb)
    counts = zeros(CT, SINGLE_PASS_N, nb)

    vD = _simd_width(geometry)
    if vD !== nothing
        _sp_simd_partial!(sums, counts, x, u, distance_bins, vD, share, culling, weights)
        return sums, counts
    end

    xk, uk = SFH.prepare_pair_inputs(geometry, x, u)
    dist_be = digitize_plan(distance_bins)
    grid, xk, uk = cull_sorted_matrices(xk, uk, geometry, distance_bins, culling)
    wc = grid === nothing ? weights : _permuted_point_weights(weights, grid.perm)
    N = size(xk, 2)
    _sp1d_run_blocks!(sums, counts, xk, uk, dist_be, geometry, nb, _share_indices(grid, N - 1, share), N, grid, wc)
    return sums, counts
end

"""
    _sp1d_pairs!(sums, counts, x, u, dist_be, geom, n_bins, blocks, weights)

Six-invariant scalar pair loop over the pairs `blocks` covers, for geometries with no SIMD fast path.
"""
function _sp1d_pairs!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1}, u::AbstractMatrix{FT2},
    dist_be, geom, n_bins::Int, blocks, weights = NoWeights(),
) where {OT, CT, FT1, FT2}
    vW = SFH.coordinate_width(geom)
    vD = SFH.field_width(geom)
    W = _val_int(vW)
    D = _val_int(vD)
    @inbounds for (ir, jr) in blocks
      j_first, j_last = first(jr), last(jr)
      for i in ir
        jlo = max(i + 1, j_first)
        jlo > j_last && continue
        wi = _point_weight(weights, i)
        x_i = SA.SVector{W, FT1}(ntuple(d -> x[d, i], vW))
        u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))
        for j in jlo:j_last
            x_j = SA.SVector{W, FT1}(ntuple(d -> x[d, j], vW))
            ok, r, frame = SFH.pair_frame(geom, x_i, x_j)
            bin = SFH.digitize(r, dist_be)
            if ok && 1 <= bin <= n_bins
                u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                du = SFH.pair_delta(geom, frame, x_i, x_j, u_i, u_j)
                duL, dn2 = SFH.increment_invariants(geom, frame, r, du)
                duL2 = duL * duL
                w = wi * _point_weight(weights, j)
                sums[1, bin] += w * dn2
                sums[2, bin] += w * duL2
                sums[4, bin] += w * duL * dn2
                sums[5, bin] += w * duL * duL2
                counts[1, bin] += CT(w)
            end
        end
      end
    end
    _sp1d_derive_rows!(sums, counts)
    return nothing
end

"""
    _sp1d_run_blocks!(sums, counts, x, u, dist_be, geom, n_bins, ilist, N, grid)

Curved-geometry single-pass analogue of [`_pf_run_blocks!`](@ref): dispatch on `grid` so the kernel
receives one concretely typed schedule.
"""
@inline _sp1d_run_blocks!(sums, counts, x, u, dist_be, geom, n_bins, ilist, N, ::Nothing,
                          weights = NoWeights()) =
    _sp1d_pairs!(sums, counts, x, u, dist_be, geom, n_bins, pair_blocks(N, ilist), weights)

@inline _sp1d_run_blocks!(sums, counts, x, u, dist_be, geom, n_bins, ilist, N, grid::CellGrid,
                          weights = NoWeights()) =
    _sp1d_pairs!(sums, counts, x, u, dist_be, geom, n_bins, pair_blocks(N, ilist; grid = grid),
                 weights)

"""Run [`_pf_sp_simd_pairs!`](@ref) over the outer indices `share` selects ([`_share_indices`](@ref)), with this
call's buffers."""
function _sp_simd_partial!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, ::Val{D}, share,
    culling::CullingPolicy = AutoCulling(), weights = NoWeights(),
) where {OT, CT, D}
    x_raw = ntuple(d -> collect(view(x, d, :)), Val(D))
    u_raw = ntuple(d -> collect(view(u, d, :)), Val(D))
    N = length(x_raw[1])
    FTx = eltype(x_raw[1])
    L = _pair_scratch_length(N)
    keybuf = Vector{FTx}(undef, L)
    duLbuf = Vector{OT}(undef, L)
    dn2buf = Vector{OT}(undef, L)
    idxbuf = Vector{Int32}(undef, L)
    sel = Vector{Int32}(undef, L)
    plan = squared_digitize_plan(dist_be)
    grid = culling isa NoCulling ? nothing :
           cull_grid_for(x_raw, SFH.FlatGeometry{D}(), dist_be, culling)
    xc, uc = isnothing(grid) ? (x_raw, u_raw) :
             (apply_perm(x_raw, grid.perm), apply_perm(u_raw, grid.perm))
    wc = isnothing(grid) ? weights : _permuted_point_weights(weights, grid.perm)
    _sp_run_blocks!(sums, counts, xc, uc, plan, Val(D), keybuf, duLbuf, dn2buf, idxbuf, sel,
        _share_indices(grid, N - 1, share), N, grid, wc)
    return nothing
end

"""Gather pair weights through a cull permutation; `NoWeights()` is a no-op."""
@inline _permuted_point_weights(w::NoWeights, _) = w
@inline _permuted_point_weights(w::AbstractVector, perm) = w[perm]

"""
    _partial_single_pass_2d(inner, x, u, distance_bins, value_bins, share, CT; geometry, culling, weights)

Six-invariant 2D joint partial sums/counts over share `share = (w, k)` of the outer indices, for one distributed
worker or MPI rank, computed on its local backend `inner` as [`_partial_single_pass_1d`](@ref) is.
The sum element type comes from the inputs.
"""
function _partial_single_pass_2d(
    ::CB.AbstractExecutionBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins,
    share::NTuple{2, Int},
    ::Type{CT};
    geometry,
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = n_histogram_bins(distance_bins)
    n_val = length(_sp2d_value_bin_at(value_bins, 1)) - 1
    _validate_value_bins!(value_bins, n_val)
    sums = zeros(OT, SINGLE_PASS_N, n_bins, n_val)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, n_val)
    _sp2d_accumulate_range!(sums, counts, x, u, distance_bins, digitize_plan(value_bins),
        geometry, n_bins, n_val, share, culling, weights)
    return sums, counts
end

function _dispatch_single_pass end
function _dispatch_single_pass! end

"""
    calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; backend=CB.SerialBackend(), kwargs...)

Accumulate into pre-allocated ``(6, n_bins)`` buffers using the requested execution backend.
"""
function calculate_structure_functions_single_pass!(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _require_backend(backend)
    _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), OT)
    _assert_counts_can_accumulate(counts, size(x, 2), w)
    _shaped(PointField, size(u, 1), distance_metric) do _, geometry
        _dispatch_single_pass!(backend, sums, counts, x, u, distance_bins; geometry, weights = w, kwargs...)
    end
    return sums, counts
end

function _dispatch_single_pass!(
    ::CB.AbstractSerialBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    return _accumulate_single_pass_1d!(sums, counts, x, u, distance_bins; kwargs...)
end

_dispatch_single_pass!(::CB.AbstractAutoBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix,
                       u::AbstractMatrix, distance_bins::AbstractVector; kwargs...) =
    _dispatch_single_pass!(resolve_auto_backend(), sums, counts, x, u, distance_bins; kwargs...)

function _dispatch_single_pass(
    ::CB.AbstractSerialBackend,
    ::PointField,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = n_histogram_bins(distance_bins)
    sums = zeros(OT, SINGLE_PASS_N, n_bins)
    counts = zeros(CT, SINGLE_PASS_N, n_bins)
    _accumulate_single_pass_1d!(sums, counts, x, u, distance_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass(
    ::CB.AbstractSerialBackend,
    ::Union{SharedPositionField, VaryingPositionField},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = size(u)[3:end]
    sums = zeros(OT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass(
    backend::CB.AbstractThreadedBackend,
    ::PointField,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    return _dispatch_single_pass(backend, x, u, distance_bins, CT; kwargs...)
end

function _dispatch_single_pass(
    ::CB.AbstractThreadedBackend,
    ::Union{SharedPositionField, VaryingPositionField},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = size(u)[3:end]
    sums = zeros(OT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    threaded_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass(
    backend::CB.AbstractGPUBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    ::Type{CT};
    kwargs...
) where {CT}
    return _dispatch_single_pass(backend, x, u, distance_bins, CT; kwargs...)
end

function _dispatch_single_pass(
    backend::CB.AbstractDistributedBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    ::Type{CT};
    kwargs...
) where {CT}
    return _dispatch_single_pass(backend, x, u, distance_bins, CT; kwargs...)
end


_dispatch_single_pass(::CB.AbstractAutoBackend, shape::AbstractFieldShape, x::AbstractArray, u::AbstractArray,
                      distance_bins::AbstractVector, ::Type{CT}; kwargs...) where {CT} =
    _dispatch_single_pass(resolve_auto_backend(), shape, x, u, distance_bins, CT; kwargs...)

# --- Single-pass result collections (keyed by invariant) ---

"""
    SINGLE_PASS_OPERATORS

The six native single-pass invariants in stacked-row order, keyed by short name. It labels each row
of the single-pass `(sums, counts)` accumulator when the result collection is built.
"""
const SINGLE_PASS_OPERATORS = (
    S2   = SFT.SecondOrderStructureFunctionType(),
    L2   = SFT.LongitudinalSecondOrderStructureFunctionType(),
    T2   = SFT.TransverseSecondOrderStructureFunctionType(),
    S3   = SFT.ThirdOrderStructureFunctionType(),
    L3   = SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
    L1T2 = SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
)

# View of stacked-row `t` across all trailing axes (n_bins, auxiliary...) — zero-copy.
@inline _sp_rowview(A::AbstractArray, t::Int) = view(A, t, ntuple(_ -> Colon(), ndims(A) - 1)...)

"""
    _single_pass_collection_1d(sums, counts, distance_bins, ::Type{OT})

Wrap the stacked 1D single-pass `(sums, counts)` into a `NamedTuple` keyed by invariant
(`S2, L2, T2, S3, L3, L1T2`), each value a single-operator result of representation `OT`.
Entries are zero-copy views into the stacked accumulator and share the (identical-by-construction)
counts row. For point-field input (stacked with the Helmholtz rows) a
`:helmholtz => HelmholtzDecomposition2D` entry is appended.
"""
function _single_pass_collection_1d(
    sums::AbstractArray, counts::AbstractArray, distance_bins, ::Type{OT},
) where {OT}
    cc = _sp_rowview(counts, 1)   # the six invariants share one (identical) counts row
    base = (
        S2   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.S2, distance_bins, _sp_rowview(sums, 1), cc), OT),
        L2   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.L2, distance_bins, _sp_rowview(sums, 2), cc), OT),
        T2   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.T2, distance_bins, _sp_rowview(sums, 3), cc), OT),
        S3   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.S3, distance_bins, _sp_rowview(sums, 4), cc), OT),
        L3   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.L3, distance_bins, _sp_rowview(sums, 5), cc), OT),
        L1T2 = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.L1T2, distance_bins, _sp_rowview(sums, 6), cc), OT),
    )
    # Point-field sums are a matrix, batched ones have ndims ≥ 3; branching on `ndims` keeps the return type concrete.
    if ndims(sums) == 2
        return merge(base, (; helmholtz = helmholtz_decompose_2d(distance_bins, sums, counts)))
    end
    return base
end

"""
    calculate_structure_functions_single_pass(x, u, distance_bins[, CT][, OT]; backend, distance_metric, weights, kwargs...)

Compute the six native invariant structure functions (S2, L2, T2, S3, L3, L1T2) in one pair
pass, returned as a `NamedTuple` keyed by invariant. `CT` is the count element type (default
`$(DEFAULT_COUNT_TYPE)`). Each entry is a single-operator result of representation `OT`, by default the
raw `StructureFunctionSumsAndCounts` that the 2D sibling also returns; pass `StructureFunction` for the
bin averages. For point-field input a `:helmholtz` entry (a
[`HelmholtzDecomposition2D`](@ref StructureFunctions.StructureFunctionObjects.HelmholtzDecomposition2D)) is included.

!!! note "Excluded invariants"
    The directional third-order invariants `L2T1` (`DiagonalInconsistentThirdOrderStructureFunction`)
    and `T3` (`OffDiagonalConsistentThirdOrderStructureFunction`) depend on a basis direction for the
    transverse component. They are available as operator types for [`calculate_structure_function`](@ref).
"""
function calculate_structure_functions_single_pass(
    x::AbstractArray{FT1},
    u::AbstractArray{FT2, M},
    distance_bins::AbstractVector{FT3},
    ::Type{CT},
    ::Type{OT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, M, CT <: Real, OT <: SFO.AbstractStructureFunction}
    _require_backend(backend)
    _validate_array_shape(x, u, distance_metric)
    OTv = promote_type(float(FT1), float(FT2))
    w = _pair_weights(weights, size(x, 2), OTv)
    _assert_count_type(CT, size(x, 2), w)
    raw = _shaped(_shape_kind(x, u), size(u, 1), distance_metric) do shape, geometry
        _dispatch_single_pass(backend, shape, x, u, distance_bins, CT; geometry, weights = w, kwargs...)
    end
    # The sum element type is `OTv`, the count element type `CT`, and the stacked accumulator's rank
    # `ndims(u)`: point-field `(6, n_bins)` is rank 2 and batched `(6, n_bins, aux...)` rank `M`.
    sums = raw.sums::AbstractArray{OTv, M}
    counts = raw.counts::AbstractArray{CT, M}
    return _single_pass_collection_1d(sums, counts, distance_bins, OT)
end

calculate_structure_functions_single_pass(x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector; kwargs...) =
    calculate_structure_functions_single_pass(x, u, distance_bins, DEFAULT_COUNT_TYPE,
                                              SFO.StructureFunctionSumsAndCounts; kwargs...)
calculate_structure_functions_single_pass(x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                          ::Type{CT}; kwargs...) where {CT <: Real} =
    calculate_structure_functions_single_pass(x, u, distance_bins, CT, SFO.StructureFunctionSumsAndCounts; kwargs...)
calculate_structure_functions_single_pass(x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                          ::Type{OT}; kwargs...) where {OT <: SFO.AbstractStructureFunction} =
    calculate_structure_functions_single_pass(x, u, distance_bins, DEFAULT_COUNT_TYPE, OT; kwargs...)


# --- 2D Single Pass Functions ---

"""
    calculate_structure_functions_single_pass_2d!(sums_3d, counts_3d, x, u, distance_bins, value_bins; backend, distance_metric, weights, kwargs...)

Accumulate the six invariants' joint distance × value histograms of a point list into `sums_3d`
and `counts_3d`, `(6, n_bins, n_val)` each, on `backend`; the in-place form of
[`calculate_structure_functions_single_pass_2d`](@ref). `value_bins` is one edge vector for every
invariant or a tuple of six.
"""
function calculate_structure_functions_single_pass_2d!(
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _require_backend(backend)
    _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), OT)
    _assert_counts_can_accumulate(counts_3d, size(x, 2), w)
    _shaped(PointField, size(u, 1), distance_metric) do _, geometry
        _dispatch_single_pass_2d!(backend, sums_3d, counts_3d, x, u, distance_bins, value_bins; geometry,
                                  weights = w, kwargs...)
    end
    return sums_3d, counts_3d
end

"""Serial pair-loop accumulation into native ``(6, n_bins, n_val)`` buffers."""
function _accumulate_single_pass_2d!(
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    geometry,
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    n_bins = length(distance_bins) - 1
    n_val = size(sums_3d, 3)
    size(sums_3d, 1) == SINGLE_PASS_N && size(sums_3d, 2) == n_bins ||
        throw(DimensionMismatch("sums must have shape ($SINGLE_PASS_N, n_bins, n_val); got $(size(sums_3d))"))
    size(counts_3d) == size(sums_3d) ||
        throw(DimensionMismatch("counts and sums must have the same shape"))
    _validate_value_bins!(value_bins, n_val)

    _sp2d_accumulate_range!(sums_3d, counts_3d, x, u, distance_bins, digitize_plan(value_bins),
        geometry, n_bins, n_val, nothing, culling, weights)
    return sums_3d, counts_3d
end

"""A histogram cell's sum, of `OT`, and the count of the pairs in it, of `CT`, adjacent in memory."""
struct SumCount{OT, CT}
    sum::OT
    count::CT
end

Base.zero(::Type{SumCount{OT, CT}}) where {OT, CT} = SumCount{OT, CT}(zero(OT), zero(CT))
Base.:+(a::SumCount{OT, CT}, b::SumCount{OT, CT}) where {OT, CT} =
    SumCount{OT, CT}(a.sum + b.sum, a.count + b.count)

"""Accumulate single-pass 2D pairs for the outer indices `share` selects ([`_share_indices`](@ref)) into the caller's
sums/counts."""
function _sp2d_accumulate_range!(
    sums_3d::AbstractArray{OT, 3}, counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix, u::AbstractMatrix, distance_bins, value_bins, geometry,
    n_bins::Int, n_val::Int, share, culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {OT, CT}
    h = _sp2d_histogram(OT, CT, n_bins, n_val)
    grid, x, u = cull_sorted_inputs(x, u, geometry, distance_bins, culling)
    wc = grid === nothing ? weights : _permuted_point_weights(weights, grid.perm)
    _sp2d_fill!(h, x, u, distance_bins, value_bins, geometry, n_bins, n_val,
                _share_indices(grid, size(x, 2) - 1, share), grid, wc)
    return _sp2d_unpack!(sums_3d, counts_3d, h, n_bins, n_val)
end

"""
Fill the interleaved accumulator from pairs whose outer index is in `ilist`: a flat geometry of width 2 or 3 takes the
SIMD compute/scatter split, every other geometry the scalar loop.
"""
function _sp2d_fill!(
    h::AbstractArray{SumCount{OT, CT}, 3},
    x::AbstractMatrix, u::AbstractMatrix, distance_bins, value_bins, geometry,
    n_bins::Int, n_val::Int, ilist, grid::Union{Nothing, CellGrid} = nothing, weights = NoWeights(),
) where {OT, CT}
    N = size(x, 2)
    vD = _simd_width(geometry)
    if vD !== nothing
        xc = ntuple(d -> collect(view(x, d, :)), vD)
        uc = ntuple(d -> collect(view(u, d, :)), vD)
        L = _pair_scratch_length(N)
        Lc = _sp2d_has_columns(value_bins, OT) ? L : 0
        _sp2d_run_blocks!(h, xc, uc, squared_digitize_plan(distance_bins), value_bins, vD,
            Vector{eltype(xc[1])}(undef, L), Vector{OT}(undef, L), Vector{OT}(undef, L), Vector{Int32}(undef, L),
            ntuple(_ -> Vector{Int32}(undef, Lc), Val(SINGLE_PASS_N)), Vector{Int32}(undef, L), n_val, ilist, N, grid,
            weights)
        return nothing
    end
    xk, uk = SFH.prepare_pair_inputs(geometry, x, u)
    _sp2d_curved_run_blocks!(h, xk, uk, digitize_plan(distance_bins), value_bins, geometry, n_bins, n_val,
        ilist, N, grid, weights)
    return nothing
end

"""
    _sp2d_histogram(OT, CT, n_bins, n_val) -> Array{SumCount{OT, CT}, 3}

The single-pass 2D accumulator, laid out `(invariant, value column, distance_bin)`, each cell a sum of
`OT` beside its count of `CT`: value column `c + 1` holds value bin `c`, and columns `1` and `n_val + 2`
the values below and above the value edges.
"""
@inline _sp2d_histogram(::Type{OT}, ::Type{CT}, n_bins::Int, n_val::Int) where {OT, CT} =
    zeros(SumCount{OT, CT}, SINGLE_PASS_N, n_val + 2, n_bins)

"""Add the interleaved accumulator's value bins into the caller's `(6, n_bins, n_val)` sums/counts."""
function _sp2d_unpack!(
    sums_3d::AbstractArray{OT, 3}, counts_3d::AbstractArray{CT, 3},
    h::AbstractArray{<:SumCount, 3}, n_bins::Int, n_val::Int,
) where {OT, CT}
    @inbounds for d in 1:n_bins, v in 1:n_val, t in 1:SINGLE_PASS_N
        c = h[t, v + 1, d]
        sums_3d[t, d, v] += c.sum
        counts_3d[t, d, v] += c.count
    end
    return nothing
end

"""
    _sp2d_pairs!(h, x, u, dist_be, value_bins, geom, n_bins, n_val, blocks)

Single-pass 2D scalar pair loop over the pairs `blocks` covers, for non-Euclidean metrics.
"""
function _sp2d_pairs!(
    h::AbstractArray{<:SumCount, 3},
    x::AbstractMatrix{FT1}, u::AbstractMatrix{FT2},
    dist_be, value_bins, geom, n_bins::Int, n_val::Int, blocks, weights = NoWeights(),
) where {FT1, FT2}
    vW = SFH.coordinate_width(geom)
    vD = SFH.field_width(geom)
    W = _val_int(vW)
    D = _val_int(vD)
    @inbounds for (ir, jr) in blocks
      j_first, j_last = first(jr), last(jr)
      for i in ir
        jlo = max(i + 1, j_first)
        jlo > j_last && continue
        wi = _point_weight(weights, i)
        x_i = SA.SVector{W, FT1}(ntuple(d -> x[d, i], vW))
        u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))
        for j in jlo:j_last
            x_j = SA.SVector{W, FT1}(ntuple(d -> x[d, j], vW))
            ok, r, frame = SFH.pair_frame(geom, x_i, x_j)
            bin_idx = SFH.digitize(r, dist_be)
            if ok && 1 <= bin_idx <= n_bins
                u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                du = SFH.pair_delta(geom, frame, x_i, x_j, u_i, u_j)
                vals = single_pass_invariants(SFH.increment_invariants(geom, frame, r, du)...)
                _sp2d_scatter!(h, bin_idx, vals, value_bins, n_val,
                               wi * _point_weight(weights, j))
            end
        end
      end
    end
    return nothing
end

"""
    _sp2d_curved_run_blocks!(h, x, u, dist_be, value_bins, geom, n_bins, n_val, ilist, N, grid)

Curved-geometry single-pass 2D analogue of [`_pf_run_blocks!`](@ref): dispatch on `grid` so the
kernel receives one concretely typed schedule.
"""
@inline _sp2d_curved_run_blocks!(
    h, x, u, dist_be, value_bins, geom, n_bins, n_val, ilist, N, ::Nothing,
    weights = NoWeights(),
) = _sp2d_pairs!(h, x, u, dist_be, value_bins, geom, n_bins, n_val,
    pair_blocks(N, ilist), weights)

@inline _sp2d_curved_run_blocks!(
    h, x, u, dist_be, value_bins, geom, n_bins, n_val, ilist, N, grid::CellGrid,
    weights = NoWeights(),
) = _sp2d_pairs!(h, x, u, dist_be, value_bins, geom, n_bins, n_val,
    pair_blocks(N, ilist; grid = grid), weights)

"""Scatter the six invariants of one pair into their value columns of the interleaved accumulator."""
@inline function _sp2d_scatter!(
    h::AbstractArray{SumCount{OT, CT}, 3}, dbin::Int, vals::NTuple{SINGLE_PASS_N}, value_bins,
    n_val::Int, w = true,
) where {OT, CT}
    @sp2d_each_invariant value_bins t vb begin
        _sp2d_add!(h, t, _value_column(vb, vals[t], n_val), dbin, vals[t], w)
    end
    return nothing
end

"""Add value `v`, weighted by `w`, and the count `w` to cell `(t, c, dbin)` of the interleaved accumulator."""
@inline function _sp2d_add!(h::AbstractArray{SumCount{OT, CT}, 3}, t, c, dbin, v, w) where {OT, CT}
    @inbounds x = h[t, c, dbin]
    @inbounds h[t, c, dbin] = SumCount{OT, CT}(x.sum + w * v, x.count + CT(w))
    return nothing
end

"""Add slot `k`'s six invariants, formed from its stored `du_L` and `|du|²`, into their stored value columns
`C1 … C6` at distance bin `dbin`, weighted by `w`."""
@inline function _sp2d_add_columns!(h, k, dbin, duLbuf, dn2buf, C1, C2, C3, C4, C5, C6, w)
    @inbounds begin
        s = single_pass_invariants(duLbuf[k], dn2buf[k])
        _sp2d_add!(h, 1, Int(C1[k]), dbin, s[1], w)
        _sp2d_add!(h, 2, Int(C2[k]), dbin, s[2], w)
        _sp2d_add!(h, 3, Int(C3[k]), dbin, s[3], w)
        _sp2d_add!(h, 4, Int(C4[k]), dbin, s[4], w)
        _sp2d_add!(h, 5, Int(C5[k]), dbin, s[5], w)
        _sp2d_add!(h, 6, Int(C6[k]), dbin, s[6], w)
    end
    return nothing
end

"""Whether the vectorized half forms the value columns of a run: for sums of 4 bytes always on a `culled`
schedule, else when the previous run's sample had more than 1/8 of its `n` slots in range (`n_in`); for wider
sums when it had more than 1/2."""
@inline _sp2d_column_pass(::Type{OT}, culled::Bool, n_in::Int, n::Int) where {OT} =
    sizeof(OT) <= 4 ? culled || 8 * n_in > n : 2 * n_in > n

"""Whether value edges `value_bins` give each invariant a linear edge set of the sums' type `OT` of its own, whose
value columns the vectorized half forms in a pass per invariant."""
@inline _sp2d_invariant_linear(::NTuple{SINGLE_PASS_N, LinearBinEdges{OT}}, ::Type{OT}) where {OT} = true
@inline _sp2d_invariant_linear(_, ::Type) = false

"""Whether a pass per invariant forms the value columns of a run, from the run's own sample of `n` slots, `n_in` of
them in range: more than 3/16 in range for sums of 4 bytes, more than 3/8 for wider sums."""
@inline _sp2d_invariant_column_pass(::Type{OT}, n_in::Int, n::Int) where {OT} =
    sizeof(OT) <= 4 ? 16 * n_in > 3 * n : 8 * n_in > 3 * n

"""Slots `ks`' value column of each invariant, from their stored `du_L` and `|du|²`, into `C`: a pass per invariant
over its own linear edges."""
@inline function _sp2d_invariant_columns!(C, value_bins, duLbuf, dn2buf, ks, n_val::Int)
    C1, C2, C3, C4, C5, C6 = C
    vb1, vb2, vb3, vb4, vb5, vb6 = value_bins
    @inbounds begin
        @simd ivdep for k in ks
            C1[k] = _vector_value_column(vb1, single_pass_invariants(duLbuf[k], dn2buf[k])[1], n_val)
        end
        @simd ivdep for k in ks
            C2[k] = _vector_value_column(vb2, single_pass_invariants(duLbuf[k], dn2buf[k])[2], n_val)
        end
        @simd ivdep for k in ks
            C3[k] = _vector_value_column(vb3, single_pass_invariants(duLbuf[k], dn2buf[k])[3], n_val)
        end
        @simd ivdep for k in ks
            C4[k] = _vector_value_column(vb4, single_pass_invariants(duLbuf[k], dn2buf[k])[4], n_val)
        end
        @simd ivdep for k in ks
            C5[k] = _vector_value_column(vb5, single_pass_invariants(duLbuf[k], dn2buf[k])[5], n_val)
        end
        @simd ivdep for k in ks
            C6[k] = _vector_value_column(vb6, single_pass_invariants(duLbuf[k], dn2buf[k])[6], n_val)
        end
    end
    return nothing
end

"""Whether the six value columns of `value_bins` can be formed in the vectorized half, in one pass or one per
invariant."""
@inline _sp2d_has_columns(value_bins, ::Type{OT}) where {OT} =
    has_vector_digitize(value_bins, OT) || _sp2d_invariant_linear(value_bins, OT)

"""
    _sp2d_simd_pairs!(h, xc, uc, plan, value_bins, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, C, sel, window,
                      n_val, blocks, weights)

Single-pass 2D point-field SIMD compute/scatter kernel over the pairs `blocks` covers, the 2D analogue
of [`_pf_sp_simd_pairs!`](@ref). The `@simd` half computes the distance key, `du_L` and `|du|²` into buffers
indexed by `window`, and the six invariants' value columns into `C` where [`has_vector_digitize`](@ref) holds
for the value edges and [`_sp2d_column_pass`](@ref) for the run; with a linear edge set per invariant, a pass per
invariant forms them when [`_sp2d_invariant_column_pass`](@ref) holds for the run. The scalar half adds each
in-range pair's six invariants into their cells, over the compacted in-range slots ([`_compact_in_range!`](@ref))
when the schedule and the run's sample choose it, else under the range test.
"""
function _sp2d_simd_pairs!(
    h::AbstractArray{SumCount{OT, CT}, 3},
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, value_bins, ::Val{D},
    keybuf::AbstractVector, duLbuf::AbstractVector, dn2buf::AbstractVector, idxbuf::AbstractVector{Int32},
    C::NTuple{SINGLE_PASS_N, AbstractVector{Int32}}, sel::AbstractVector{Int32}, window::PairWindow, n_val::Int,
    blocks, weights = NoWeights(),
) where {OT, CT, D}
    nb = n_histogram_bins(plan)
    vector_columns = has_vector_digitize(value_bins, OT)
    invariant_columns = _sp2d_invariant_linear(value_bins, OT)
    chooses = _chooses_compaction(blocks)
    FTx = eltype(xc[1])
    C1, C2, C3, C4, C5, C6 = C
    s_in, s_n = 0, 0
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        _check_run_fits(window, duLbuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            wi = _point_weight(weights, i)
            Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
            Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
            fused = vector_columns && _sp2d_column_pass(OT, !chooses, s_in, s_n)
            if fused
                @simd ivdep for j in jlo:j_last
                    dx = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D))) - Xi
                    r2 = SFH.fma_dot(dx, dx)
                    du = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D))) - Ui
                    k = j - o
                    keybuf[k] = digitize_key(plan, r2)
                    if has_vector_index(plan)
                        idxbuf[k] = squared_approx_index(plan, r2)
                    end
                    duL, dn2 = SFH.increment_invariants(SFH.FlatGeometry{D}(), dx, sqrt(r2), du)
                    a, b = OT(duL), OT(dn2)
                    duLbuf[k], dn2buf[k] = a, b
                    s = single_pass_invariants(a, b)
                    C1[k] = _vector_value_column(value_bins, s[1], n_val)
                    C2[k] = _vector_value_column(value_bins, s[2], n_val)
                    C3[k] = _vector_value_column(value_bins, s[3], n_val)
                    C4[k] = _vector_value_column(value_bins, s[4], n_val)
                    C5[k] = _vector_value_column(value_bins, s[5], n_val)
                    C6[k] = _vector_value_column(value_bins, s[6], n_val)
                end
            else
                @simd for j in jlo:j_last
                    dx = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D))) - Xi
                    r2 = SFH.fma_dot(dx, dx)
                    du = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D))) - Ui
                    keybuf[j - o] = digitize_key(plan, r2)
                    duLbuf[j - o], dn2buf[j - o] = SFH.increment_invariants(SFH.FlatGeometry{D}(), dx, sqrt(r2), du)
                    if has_vector_index(plan)
                        idxbuf[j - o] = squared_approx_index(plan, r2)
                    end
                end
            end
            ks = (jlo - o):(j_last - o)
            s_in, s_n = _sample_in_range(plan, keybuf, ks)
            columns = fused || (invariant_columns && _sp2d_invariant_column_pass(OT, s_in, s_n))
            columns && invariant_columns && _sp2d_invariant_columns!(C, value_bins, duLbuf, dn2buf, ks, n_val)
            compact = chooses && _compacts(s_in, s_n)
            if columns && compact
                for m in 1:_compact_in_range!(sel, plan, keybuf, ks)
                    k = Int(sel[m])
                    _sp2d_add_columns!(h, k, squared_bin_select(plan, keybuf[k], idxbuf[k]), duLbuf, dn2buf,
                                       C1, C2, C3, C4, C5, C6, wi * _point_weight(weights, k + o))
                end
            elseif columns
                for k in ks
                    dbin = chooses ? squared_bin_select(plan, keybuf[k], idxbuf[k]) : squared_bin(plan, keybuf[k], idxbuf[k])
                    if 1 <= dbin <= nb
                        _sp2d_add_columns!(h, k, dbin, duLbuf, dn2buf, C1, C2, C3, C4, C5, C6,
                                           wi * _point_weight(weights, k + o))
                    end
                end
            elseif compact
                for m in 1:_compact_in_range!(sel, plan, keybuf, ks)
                    k = Int(sel[m])
                    _sp2d_scatter!(h, squared_bin_select(plan, keybuf[k], idxbuf[k]),
                                   single_pass_invariants(duLbuf[k], dn2buf[k]), value_bins, n_val,
                                   wi * _point_weight(weights, k + o))
                end
            else
                for k in ks
                    dbin = chooses ? squared_bin_select(plan, keybuf[k], idxbuf[k]) : squared_bin(plan, keybuf[k], idxbuf[k])
                    if 1 <= dbin <= nb
                        _sp2d_scatter!(h, dbin, single_pass_invariants(duLbuf[k], dn2buf[k]), value_bins, n_val,
                                       wi * _point_weight(weights, k + o))
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _sp2d_run_blocks!(h, xc, uc, plan, value_bins, ::Val{D}, bufs..., n_val, ilist, N, grid)

Single-pass 2D analogue of [`_pf_run_blocks!`](@ref): dispatch on `grid` so the kernel receives one
concretely typed schedule.
"""
@inline _sp2d_run_blocks!(
    h, xc, uc, plan, value_bins, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, C, sel, n_val,
    ilist, N, ::Nothing, weights = NoWeights(),
) where {D} = _sp2d_simd_pairs!(h, xc, uc, plan, value_bins, Val(D), keybuf, duLbuf, dn2buf,
    idxbuf, C, sel, _pair_window(N), n_val, pair_blocks(N, ilist), weights)

@inline _sp2d_run_blocks!(
    h, xc, uc, plan, value_bins, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, C, sel, n_val,
    ilist, N, grid::CellGrid, weights = NoWeights(),
) where {D} = _sp2d_simd_pairs!(h, xc, uc, plan, value_bins, Val(D), keybuf, duLbuf, dn2buf,
    idxbuf, C, sel, _pair_window(N), n_val, pair_blocks(N, ilist; grid = grid), weights)

function _dispatch_single_pass_2d!(
    ::CB.AbstractSerialBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    return _accumulate_single_pass_2d!(sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_single_pass_2d!(
    backend::CB.AbstractGPUBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    gpu_calculate_structure_functions_single_pass_2d!(sums_3d, counts_3d, backend.backend, x, u, distance_bins, value_bins; kwargs...)
    return sums_3d, counts_3d
end

_dispatch_single_pass_2d!(::CB.AbstractAutoBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix,
                          u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins;
                          kwargs...) =
    _dispatch_single_pass_2d!(resolve_auto_backend(), sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)

function _dispatch_single_pass_2d(
    ::CB.AbstractSerialBackend,
    ::PointField,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins,
    ::Type{CT};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = n_histogram_bins(distance_bins)
    n_val = length(_sp2d_value_bin_at(value_bins, 1)) - 1
    sums = zeros(OT, SINGLE_PASS_N, n_bins, n_val)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, n_val)
    _accumulate_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass_2d(
    ::CB.AbstractSerialBackend,
    ::Union{SharedPositionField, VaryingPositionField},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins,
    ::Type{CT};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = length(value_bins isa Tuple ? value_bins[1] : value_bins) - 1
    _validate_value_bins!(value_bins, n_val)
    auxiliary_dims = size(u)[3:end]
    sums = zeros(OT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass_2d(
    backend::CB.AbstractThreadedBackend,
    ::PointField,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins,
    ::Type{CT};
    kwargs...
) where {CT}
    return _dispatch_single_pass_2d(backend, x, u, distance_bins, value_bins, CT; kwargs...)
end

function _dispatch_single_pass_2d(
    ::CB.AbstractThreadedBackend,
    ::Union{SharedPositionField, VaryingPositionField},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins,
    ::Type{CT};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = length(value_bins isa Tuple ? value_bins[1] : value_bins) - 1
    _validate_value_bins!(value_bins, n_val)
    auxiliary_dims = size(u)[3:end]
    sums = zeros(OT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    threaded_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass_2d(
    backend::CB.AbstractDistributedBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins,
    ::Type{CT};
    kwargs...
) where {CT}
    return _dispatch_single_pass_2d(backend, x, u, distance_bins, value_bins, CT; kwargs...)
end

function _dispatch_single_pass_2d(backend::CB.AbstractGPUBackend, x::AbstractMatrix{<:Number},
                                  u::AbstractMatrix{<:Number}, distance_bins::AbstractVector{<:Number},
                                  value_bins::SinglePass2DValueBins, ::Type{CT}; kwargs...) where {CT}
    return gpu_calculate_structure_functions_single_pass_2d(backend.backend, x, u, distance_bins, value_bins, CT; kwargs...)
end

function _dispatch_single_pass_2d(
    backend::CB.AbstractGPUBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins,
    ::Type{CT};
    kwargs...
) where {CT}
    return _dispatch_single_pass_2d(backend, x, u, distance_bins, value_bins, CT; kwargs...)
end


_dispatch_single_pass_2d(::CB.AbstractAutoBackend, shape::AbstractFieldShape, x::AbstractArray, u::AbstractArray,
                         distance_bins::AbstractVector, value_bins::SinglePass2DValueBins, ::Type{CT};
                         kwargs...) where {CT} =
    _dispatch_single_pass_2d(resolve_auto_backend(), shape, x, u, distance_bins, value_bins, CT; kwargs...)

# Per-invariant value bins: a single vector is shared across invariants; a 6-tuple is per-invariant.
@inline _sp_valuebins(vb::AbstractVector, t::Int) = vb
@inline _sp_valuebins(vb::Tuple, t::Int) = vb[t]

"""
    _single_pass_collection_2d(sums, counts, distance_bins, value_bins, ::Type{OT})

Wrap the stacked 2D single-pass `(sums, counts)` (shape `(6, n_dist, n_val, aux...)`) into a
`NamedTuple` keyed by invariant, each value a `StructureFunction2DSumsAndCounts` view into the
stacked accumulator. Counts are per invariant, since each invariant's value lands in its own value bin.
`OT` must be `StructureFunction2DSumsAndCounts`.
"""
function _single_pass_collection_2d(
    sums::AbstractArray, counts::AbstractArray, distance_bins, value_bins, ::Type{OT},
) where {OT}
    return (
        S2   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.S2, distance_bins, _sp_valuebins(value_bins, 1), _sp_rowview(sums, 1), _sp_rowview(counts, 1), InvariantValueAxis()), OT),
        L2   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.L2, distance_bins, _sp_valuebins(value_bins, 2), _sp_rowview(sums, 2), _sp_rowview(counts, 2), InvariantValueAxis()), OT),
        T2   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.T2, distance_bins, _sp_valuebins(value_bins, 3), _sp_rowview(sums, 3), _sp_rowview(counts, 3), InvariantValueAxis()), OT),
        S3   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.S3, distance_bins, _sp_valuebins(value_bins, 4), _sp_rowview(sums, 4), _sp_rowview(counts, 4), InvariantValueAxis()), OT),
        L3   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.L3, distance_bins, _sp_valuebins(value_bins, 5), _sp_rowview(sums, 5), _sp_rowview(counts, 5), InvariantValueAxis()), OT),
        L1T2 = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.L1T2, distance_bins, _sp_valuebins(value_bins, 6), _sp_rowview(sums, 6), _sp_rowview(counts, 6), InvariantValueAxis()), OT),
    )
end

"""
    calculate_structure_functions_single_pass_2d(x, u, distance_bins, value_bins[, CT][, OT]; backend, distance_metric, weights, kwargs...)

Compute the six invariant 2D joint structure-function histograms in one pass, returned as a
`NamedTuple` keyed by invariant (`S2, L2, T2, S3, L3, L1T2`). `CT` is the count element type (default
`$(DEFAULT_COUNT_TYPE)`). Each entry is a
[`StructureFunction2DSumsAndCounts`](@ref StructureFunctions.StructureFunctionObjects.StructureFunction2DSumsAndCounts) view into the stacked accumulator (the 2D joint
histogram has no averaged form, so `OT` must be `StructureFunction2DSumsAndCounts`).
"""
function calculate_structure_functions_single_pass_2d(
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins,
    ::Type{CT},
    ::Type{OT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT <: Real, OT <: SFO.AbstractStructureFunction}
    _require_backend(backend)
    _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), promote_type(float(FT1), float(FT2)))
    _assert_count_type(CT, size(x, 2), w)
    raw = _shaped(_shape_kind(x, u), size(u, 1), distance_metric) do shape, geometry
        _dispatch_single_pass_2d(backend, shape, x, u, distance_bins, value_bins, CT; geometry, weights = w,
                                 kwargs...)
    end
    return _single_pass_collection_2d(raw[1], raw[2], distance_bins, value_bins, OT)
end

calculate_structure_functions_single_pass_2d(x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                             value_bins::SinglePass2DValueBins; kwargs...) =
    calculate_structure_functions_single_pass_2d(x, u, distance_bins, value_bins, DEFAULT_COUNT_TYPE,
                                                 SFO.StructureFunction2DSumsAndCounts; kwargs...)
calculate_structure_functions_single_pass_2d(x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                             value_bins::SinglePass2DValueBins, ::Type{CT}; kwargs...) where {CT <: Real} =
    calculate_structure_functions_single_pass_2d(x, u, distance_bins, value_bins, CT,
                                                 SFO.StructureFunction2DSumsAndCounts; kwargs...)
calculate_structure_functions_single_pass_2d(x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                             value_bins::SinglePass2DValueBins,
                                             ::Type{OT}; kwargs...) where {OT <: SFO.AbstractStructureFunction} =
    calculate_structure_functions_single_pass_2d(x, u, distance_bins, value_bins, DEFAULT_COUNT_TYPE, OT; kwargs...)
