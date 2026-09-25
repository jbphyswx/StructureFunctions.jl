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

The quadrature starts at the first bin midpoint, so ``I`` omits ``∫_0^{r_1}``. That segment is
`D_LL(r_1)` for an ``r^{2/3}`` inertial range, so the decomposition is quantitative only for
``r ≫ r_1``; choose a first bin well below the scales of interest.
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
    # Separation metadata is small; numerical integration follows the field backend.
    bin_mids = similar(D_LL, OT, n_bins)
    copyto!(bin_mids, collect(OT, midpoints(distance_bins)))
    integrand = (D_TT .- D_LL) ./ bin_mids
    increments = similar(D_LL)
    fill!(increments, zero(OT))
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

"""Serial pair-loop accumulation into native ``(6, n_bins)`` buffers (no allocation)."""
function _accumulate_single_pass_1d!(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    D = size(u, 1)
    n_points = size(x, 2)
    n_bins = length(distance_bins) - 1
    size(sums) == (SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("sums must have shape ($SINGLE_PASS_N, n_bins); got $(size(sums))"))
    size(counts) == (SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("counts must have shape ($SINGLE_PASS_N, n_bins); got $(size(counts))"))
    # Fast path: Euclidean + D ∈ (2,3) via the SIMD compute/scatter split (vectorizes the
    # per-pair du_L / |du|² compute over j; the 6-way histogram scatter stays scalar).
    geom = SFH.pair_geometry_for(distance_metric, Val(D))
    if geom isa SFH.FlatGeometry && (D == 2 || D == 3)
        _sp_simd_run!(sums, counts, x, u, distance_bins, D == 2 ? Val(2) : Val(3), culling, weights)
        return sums, counts
    end

    xk, uk = SFH.prepare_pair_inputs(geom, x, u)
    be = digitize_plan(distance_bins)
    grid, xk, uk = cull_sorted_matrices(xk, uk, geom, distance_bins, culling)
    wc = grid === nothing ? weights : _permuted_point_weights(weights, grid.perm)
    _sp1d_run_blocks!(sums, counts, xk, uk, be, geom, n_bins, 1:n_points, n_points, grid, wc)
    return sums, counts
end

"""
    _sp1d_derive_rows!(sums, counts)

Fill the two derived invariant rows and replicate the shared count of a `(SINGLE_PASS_N, n_bins)`
accumulator.

`T2 = S2 - L2` and `L1T2 = S3 - L3` hold for every pair, and a bin is a sum, so the pair loops store
only rows 1, 2, 4, 5 — four stores per pair — and these two are differenced once per call. The count
is identical across all six rows, so it is accumulated into row 1 and broadcast here.

Uses `=`, never `+=`, so it is **idempotent**: correct whether a kernel runs once or many times over
the same buffer, and correct under threaded/partial reduction because the derivation is linear
(`Σ(S2-L2) = ΣS2 - ΣL2`). So it runs in one place, the end of each pair kernel, where the callers'
assembly points number more than twenty.
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
    _pf_sp_simd_pairs!(sums, counts, xc, uc, plan, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, blocks)

Single-pass (6 invariants) point-field SIMD compute/scatter kernel over the pairs `blocks` covers.
For each `i`: `@simd` over its `j` block computes distance, `du_L = du·r̂`, and `|du|²` into buffers
(contiguous components ⇒ packed loads, no scatter ⇒ vectorizes), then a scalar loop digitizes
and scatters the 6 invariants. Like [`_pf_simd_pairs!`](@ref) it consumes `(i-block, j-block)`
pairs (see [`block_pairs`](@ref)), so it gets both cache blocking and culling, and the loop must
live in this one kernel (not a per-`i` helper) for the `@simd` to vectorize.
Shared by serial + threaded.
"""
function _pf_sp_simd_pairs!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, ::Val{D},
    keybuf::AbstractVector, duLbuf::AbstractVector, dn2buf::AbstractVector,
    idxbuf::AbstractVector{Int32}, blocks, weights = NoWeights(),
) where {OT, CT, D}
    nb = n_histogram_bins(plan)
    FTx = eltype(xc[1])
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            wi = _point_weight(weights, i)
            Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
            Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
            @simd for j in jlo:j_last
                Xj = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D)))
                dx = Xj - Xi
                r2 = SFH.fma_dot(dx, dx)
                du = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D))) - Ui
                # δu_L needs r, so one reciprocal-sqrt stays; it vectorizes.
                inv_r = inv(sqrt(r2))
                keybuf[j] = digitize_key(plan, r2)
                duLbuf[j] = SFH.fma_dot(du, dx) * inv_r
                dn2buf[j] = SFH.fma_dot(du, du)
                if has_vector_index(plan)
                    idxbuf[j] = squared_approx_index(plan, r2)
                end
            end
            for j in jlo:j_last
                bin = squared_bin(plan, keybuf[j], idxbuf[j])
                if 1 <= bin <= nb
                    duL = duLbuf[j]
                    dn2 = dn2buf[j]
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
    _sp_run_blocks!(sums, counts, xc, uc, plan, ::Val{D}, bufs..., ilist, N, grid)

Single-pass analogue of [`_pf_run_blocks!`](@ref): dispatch on `grid` so the kernel receives one
concretely typed schedule.
"""
@inline _sp_run_blocks!(
    sums, counts, xc, uc, plan, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, ilist, N, ::Nothing,
    weights = NoWeights(),
) where {D} = _pf_sp_simd_pairs!(sums, counts, xc, uc, plan, Val(D), keybuf, duLbuf, dn2buf,
    idxbuf, pair_blocks(N, ilist), weights)

@inline _sp_run_blocks!(
    sums, counts, xc, uc, plan, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, ilist, N,
    grid::CellGrid, weights = NoWeights(),
) where {D} = _pf_sp_simd_pairs!(sums, counts, xc, uc, plan, Val(D), keybuf, duLbuf, dn2buf,
    idxbuf, pair_blocks(N, ilist; grid = grid), weights)

# Serial driver: the full outer range through the same per-worker kernel.
function _sp_simd_run!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, ::Val{D},
    culling::CullingPolicy = AutoCulling(), weights = NoWeights(),
) where {OT, CT, D}
    N = size(x, 2)
    return _sp_simd_partial!(sums, counts, x, u, dist_be, Val(D), 1:(N - 1), culling, weights)
end

"""
    _partial_single_pass_1d(x, u, distance_bins, ilist, CT; distance_metric, culling, weights)

Six-invariant partial sums/counts over an explicit outer-index list, for one distributed worker or
MPI rank. Euclidean `D ∈ {2,3}` takes the SIMD kernel; other metrics use the scalar loop. The sum
element type comes from the inputs.
"""
function _partial_single_pass_1d(
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    ilist,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    D = size(u, 1)
    nb = n_histogram_bins(distance_bins)
    sums = zeros(OT, SINGLE_PASS_N, nb)
    counts = zeros(CT, SINGLE_PASS_N, nb)

    geom = SFH.pair_geometry_for(distance_metric, Val(D))
    if geom isa SFH.FlatGeometry && (D == 2 || D == 3)
        _sp_simd_partial!(sums, counts, x, u, distance_bins, D == 2 ? Val(2) : Val(3), ilist, culling, weights)
        return sums, counts
    end

    xk, uk = SFH.prepare_pair_inputs(geom, x, u)
    dist_be = digitize_plan(distance_bins)
    grid, xk, uk = cull_sorted_matrices(xk, uk, geom, distance_bins, culling)
    wc = grid === nothing ? weights : _permuted_point_weights(weights, grid.perm)
    _sp1d_run_blocks!(sums, counts, xk, uk, dist_be, geom, nb, ilist, size(xk, 2), grid, wc)
    return sums, counts
end

"""
    _sp1d_pairs!(sums, counts, x, u, dist_be, geom, n_bins, ilist)

Six-invariant scalar pair loop over the outer indices `ilist`, for geometries with no SIMD fast
path. The coordinate and field widths come from `geom` as `Val`s, so the `SVector`s stay concrete
inside the loop.
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
                du, rh = SFH.pair_increments(geom, frame, r, x_i, x_j, u_i, u_j)
                duL = SFH.fma_dot(du, rh)
                duL2 = duL * duL
                dn2 = SFH.fma_dot(du, du)
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

"""Run [`_pf_sp_simd_pairs!`](@ref) over an explicit outer-index list, with this worker's buffers."""
function _sp_simd_partial!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, ::Val{D}, ilist,
    culling::CullingPolicy = AutoCulling(), weights = NoWeights(),
) where {OT, CT, D}
    x_raw = ntuple(d -> collect(view(x, d, :)), Val(D))
    u_raw = ntuple(d -> collect(view(u, d, :)), Val(D))
    N = length(x_raw[1])
    FTx = eltype(x_raw[1])
    keybuf = Vector{FTx}(undef, N)
    duLbuf = Vector{OT}(undef, N)
    dn2buf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    plan = squared_digitize_plan(dist_be)
    grid = culling isa NoCulling ? nothing :
           cull_grid_for(x_raw, SFH.FlatGeometry{D}(), dist_be, culling)
    xc, uc = isnothing(grid) ? (x_raw, u_raw) :
             (apply_perm(x_raw, grid.perm), apply_perm(u_raw, grid.perm))
    # The cull reorders the points, so the weights are gathered the same way.
    wc = isnothing(grid) ? weights : _permuted_point_weights(weights, grid.perm)
    _sp_run_blocks!(sums, counts, xc, uc, plan, Val(D), keybuf, duLbuf, dn2buf, idxbuf,
        ilist, N, grid, wc)
    return nothing
end

"""Gather pair weights through a cull permutation; `NoWeights()` is a no-op."""
@inline _permuted_point_weights(w::NoWeights, _) = w
@inline _permuted_point_weights(w::AbstractVector, perm) = w[perm]

"""
    _partial_single_pass_2d(x, u, distance_bins, value_bins, ilist, CT; distance_metric, culling, weights)

Six-invariant 2D joint partial sums/counts over an explicit outer-index list, for one distributed
worker or MPI rank. The sum element type comes from the inputs.
"""
function _partial_single_pass_2d(
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins,
    ilist,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
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
        distance_metric, n_bins, n_val, ilist, culling, weights)
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
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), OT)
    _assert_counts_can_accumulate(counts, size(x, 2), w)
    _dispatch_single_pass!(backend, sums, counts, x, u, distance_bins; distance_metric, weights = w, kwargs...)
    return sums, counts
end

function _dispatch_single_pass!(
    ::CB.AbstractSerialBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    return _accumulate_single_pass_1d!(sums, counts, x, u, distance_bins; kwargs...)
end

function _dispatch_single_pass!(
    ::CB.AbstractThreadedBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("Threaded in-place single-pass is unavailable. Load OhMyThreads or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass!(::CB.AbstractDistributedBackend, args...; kwargs...)
    throw(ArgumentError("Distributed in-place single-pass is unavailable. Load Distributed (`using Distributed`) or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass!(::CB.AbstractMPIBackend, args...; kwargs...)
    throw(ArgumentError("MPI in-place single-pass is unavailable. Load MPI (`using MPI`) or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass!(
    ::CB.AbstractGPUBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("GPU in-place single-pass is unavailable. Load GPUExt or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass!(
    ::CB.AbstractAutoBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    if distributed_adds_hardware(Val(:distributed))
        return _dispatch_single_pass!(CB.DistributedBackend(), sums, counts, x, u, distance_bins; kwargs...)
    end
    return _dispatch_single_pass!(_auto_local_backend(), sums, counts, x, u, distance_bins; kwargs...)
end

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
    # The raw six-row accumulator; the public entry builds the Helmholtz entry from it.
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
    ::CB.AbstractThreadedBackend,
    ::PointField,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    return _dispatch_single_pass(CB.ThreadedBackend(), x, u, distance_bins, CT; kwargs...)
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
    _require_threading("the auxiliary-axis single-pass driver")
    sums = zeros(OT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    threaded_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass(::CB.AbstractThreadedBackend, args...; kwargs...)
    throw(ArgumentError("Threaded single-pass backend is unavailable. Load the OhMyThreads extension or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass(::CB.AbstractDistributedBackend, args...; kwargs...)
    throw(ArgumentError("Distributed single-pass backend is unavailable. Load Distributed (`using Distributed`) or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass(::CB.AbstractGPUBackend, args...; kwargs...)
    throw(ArgumentError("GPU single-pass backend is unavailable. Load the GPUExt extension or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass(::CB.AbstractMPIBackend, args...; kwargs...)
    throw(ArgumentError("MPI single-pass backend is unavailable. Load MPI (`using MPI`) or use backend=CB.SerialBackend()."))
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


function _dispatch_single_pass(::CB.AbstractAutoBackend, shape::AbstractFieldShape, x::AbstractArray, u::AbstractArray,
                               distance_bins::AbstractVector, ::Type{CT}; kwargs...) where {CT}
    backend = resolve_auto_backend(
        shape,
        _ohmythreads_loaded,
    )
    return _dispatch_single_pass(backend, shape, x, u, distance_bins, CT; kwargs...)
end

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
(`S2, L2, T2, S3, L3, L1T2`), each value a single-operator result of representation `OT`
(default the averaged `StructureFunction`; pass `StructureFunctionSumsAndCounts` for raw).
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
    # Helmholtz exists exactly for point-field input (a 2D stacked matrix); batched input is
    # ndims ≥ 3. Branch on `ndims(sums)` ALONE (compile-time) — not a runtime size check — so the
    # return type is a single concrete NamedTuple (type-stable), not a Union of 6/7-key tuples.
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

!!! note "Why only six invariants (no L2T1 / T3)"
    The single-pass set is the six **isotropic** invariants. The directional third-order
    invariants `L2T1` (`DiagonalInconsistentThirdOrderStructureFunction`) and `T3`
    (`OffDiagonalConsistentThirdOrderStructureFunction`) are intentionally excluded: they require
    choosing a basis direction for the transverse/normal component (the isotropy does not cancel),
    so they are not basis-independent and are uncommon in practice. They remain available as
    standalone operator types for an explicit [`calculate_structure_function`](@ref) call.
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
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, M, CT <: Real, OT <: SFO.AbstractStructureFunction}
    shape = _validate_array_shape(x, u, distance_metric)
    OTv = promote_type(float(FT1), float(FT2))
    w = _pair_weights(weights, size(x, 2), OTv)
    _assert_count_type(CT, size(x, 2), w)
    raw = _dispatch_single_pass(backend, shape, x, u, distance_bins, CT; distance_metric, weights = w, kwargs...)
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
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), OT)
    _assert_counts_can_accumulate(counts_3d, size(x, 2), w)
    _dispatch_single_pass_2d!(
        backend, sums_3d, counts_3d, x, u, distance_bins, value_bins; distance_metric, weights = w, kwargs...
    )
    return sums_3d, counts_3d
end

"""Serial pair-loop accumulation into native ``(6, n_bins, n_val)`` buffers (no allocation)."""
function _accumulate_single_pass_2d!(
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    n_points = size(x, 2)
    n_bins = length(distance_bins) - 1
    n_val = size(sums_3d, 3)
    size(sums_3d, 1) == SINGLE_PASS_N && size(sums_3d, 2) == n_bins ||
        throw(DimensionMismatch("sums must have shape ($SINGLE_PASS_N, n_bins, n_val); got $(size(sums_3d))"))
    size(counts_3d) == size(sums_3d) ||
        throw(DimensionMismatch("counts and sums must have the same shape"))
    _validate_value_bins!(value_bins, n_val)

    _sp2d_accumulate_range!(sums_3d, counts_3d, x, u, distance_bins, digitize_plan(value_bins),
        distance_metric, n_bins, n_val, 1:n_points, culling, weights)
    return sums_3d, counts_3d
end

"""Accumulate single-pass 2D pairs for outer indices `ilist` into the caller's sums/counts."""
function _sp2d_accumulate_range!(
    sums_3d::AbstractArray{OT, 3}, counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix, u::AbstractMatrix, distance_bins, value_bins, distance_metric,
    n_bins::Int, n_val::Int, ilist, culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {OT, CT}
    h = _sp2d_histogram(OT, n_bins, n_val)
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    grid, x, u = cull_sorted_inputs(x, u, geom, distance_bins, culling)
    wc = grid === nothing ? weights : _permuted_point_weights(weights, grid.perm)
    _sp2d_fill!(h, x, u, distance_bins, value_bins, distance_metric, n_bins, n_val, ilist, grid, wc)
    return _sp2d_unpack!(sums_3d, counts_3d, h, n_bins, n_val)
end

"""
Fill the interleaved accumulator from pairs whose outer index is in `ilist`. Euclidean `D ∈ {2,3}`
takes the SIMD compute/scatter split; other metrics or dimensions take the scalar loop. The `Val{D}`
branch is a function barrier: `D` must be a type parameter inside the loop, or `SVector{D}` builds
its type per point.
"""
function _sp2d_fill!(
    h::AbstractArray{OT, 4},
    x::AbstractMatrix, u::AbstractMatrix, distance_bins, value_bins, distance_metric,
    n_bins::Int, n_val::Int, ilist, grid = nothing, weights = NoWeights(),
) where {OT}
    D = size(u, 1)
    geom = SFH.pair_geometry_for(distance_metric, Val(D))
    N = size(x, 2)
    if geom isa SFH.FlatGeometry && (D == 2 || D == 3)
        vD = D == 2 ? Val(2) : Val(3)
        xc = ntuple(d -> collect(view(x, d, :)), vD)
        uc = ntuple(d -> collect(view(u, d, :)), vD)
        _sp2d_run_blocks!(h, xc, uc, squared_digitize_plan(distance_bins), value_bins, vD,
            Vector{eltype(xc[1])}(undef, N), Vector{OT}(undef, N), Vector{OT}(undef, N),
            Vector{Int32}(undef, N), n_val, ilist, N, grid, weights)
        return nothing
    end
    xk, uk = SFH.prepare_pair_inputs(geom, x, u)
    _sp2d_curved_run_blocks!(h, xk, uk, digitize_plan(distance_bins), value_bins, geom, n_bins, n_val,
        ilist, N, grid, weights)
    return nothing
end

"""
    _sp2d_histogram(OT, n_bins, n_val) -> Array{OT,4}

The single-pass 2D accumulator, laid out `(sum|count, invariant, value_bin, distance_bin)`.

Each pair writes all six invariants at ONE distance bin but six different value bins, so putting
the value axis inside the distance axis keeps a pair's six updates inside one distance slab, and
interleaving sum with count puts each invariant's two updates on one cache line, so a pair touches
6 lines and the cost stops scaling with histogram size.
"""
@inline _sp2d_histogram(::Type{OT}, n_bins::Int, n_val::Int) where {OT} =
    zeros(OT, 2, SINGLE_PASS_N, n_val, n_bins)

"""Add the interleaved accumulator into the caller's `(6, n_bins, n_val)` sums/counts."""
function _sp2d_unpack!(
    sums_3d::AbstractArray{OT, 3}, counts_3d::AbstractArray{CT, 3},
    h::AbstractArray, n_bins::Int, n_val::Int,
) where {OT, CT}
    @inbounds for d in 1:n_bins, v in 1:n_val, t in 1:SINGLE_PASS_N
        sums_3d[t, d, v] += h[1, t, v, d]
        counts_3d[t, d, v] += CT(h[2, t, v, d])
    end
    return nothing
end

"""
    _sp2d_pairs!(h, x, u, dist_be, value_bins, geom, n_bins, n_val, blocks)

Single-pass 2D scalar pair loop over the pairs `blocks` covers, for non-Euclidean metrics.
Specialized on the spatial dimension `D` so the `SVector`s are concrete.
"""
function _sp2d_pairs!(
    h::AbstractArray{OT, 4},
    x::AbstractMatrix{FT1}, u::AbstractMatrix{FT2},
    dist_be, value_bins, geom, n_bins::Int, n_val::Int, blocks, weights = NoWeights(),
) where {OT, FT1, FT2}
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
                du, rh = SFH.pair_increments(geom, frame, r, x_i, x_j, u_i, u_j)
                du_L = SFH.fma_dot(du, rh)
                vals = single_pass_invariants(du_L, SFH.fma_dot(du, du))
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

"""Scatter the six invariants of one pair into their cells of the interleaved accumulator."""
@inline function _sp2d_scatter!(
    h::AbstractArray{OT, 4}, dbin::Int, vals::NTuple{SINGLE_PASS_N}, value_bins, n_val::Int,
    w = true,
) where {OT}
    @sp2d_each_invariant value_bins t vb begin
        vbin = SFH.digitize(vals[t], vb)
        if 1 <= vbin <= (length(vb) - 1) && vbin <= n_val
            @inbounds h[1, t, vbin, dbin] += w * vals[t]
            @inbounds h[2, t, vbin, dbin] += OT(w)
        end
    end
    return nothing
end

"""
    _sp2d_simd_pairs!(h, xc, uc, plan, value_bins, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, n_val, irange)

Single-pass 2D point-field SIMD compute/scatter kernel over outer indices `irange`, the 2D analogue
of [`_pf_sp_simd_pairs!`](@ref). The `@simd` half computes distance, `du_L` and `|du|²` into buffers;
the scalar half derives the six invariants from those two scalars and scatters each into its own
`(distance, value)` cell. Shared by serial + threaded.
"""
function _sp2d_simd_pairs!(
    h::AbstractArray{OT, 4},
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, value_bins, ::Val{D},
    keybuf::AbstractVector, duLbuf::AbstractVector, dn2buf::AbstractVector,
    idxbuf::AbstractVector{Int32}, n_val::Int, blocks, weights = NoWeights(),
) where {OT, D}
    nb = n_histogram_bins(plan)
    FTx = eltype(xc[1])
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            wi = _point_weight(weights, i)
            Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
            Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
            @simd for j in jlo:j_last
                Xj = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D)))
                dx = Xj - Xi
                r2 = SFH.fma_dot(dx, dx)
                du = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D))) - Ui
                inv_r = inv(sqrt(r2))
                keybuf[j] = digitize_key(plan, r2)
                duLbuf[j] = SFH.fma_dot(du, dx) * inv_r
                dn2buf[j] = SFH.fma_dot(du, du)
                if has_vector_index(plan)
                    idxbuf[j] = squared_approx_index(plan, r2)
                end
            end
            for j in jlo:j_last
                dbin = squared_bin(plan, keybuf[j], idxbuf[j])
                if 1 <= dbin <= nb
                    duL = duLbuf[j]
                    dn2 = dn2buf[j]
                    duL2 = duL * duL
                    duT2 = dn2 - duL2
                    vals = (dn2, duL2, duT2, duL * dn2, duL * duL2, duL * duT2)
                    _sp2d_scatter!(h, dbin, vals, value_bins, n_val,
                                   wi * _point_weight(weights, j))
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
    h, xc, uc, plan, value_bins, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, n_val,
    ilist, N, ::Nothing, weights = NoWeights(),
) where {D} = _sp2d_simd_pairs!(h, xc, uc, plan, value_bins, Val(D), keybuf, duLbuf, dn2buf,
    idxbuf, n_val, pair_blocks(N, ilist), weights)

@inline _sp2d_run_blocks!(
    h, xc, uc, plan, value_bins, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, n_val,
    ilist, N, grid::CellGrid, weights = NoWeights(),
) where {D} = _sp2d_simd_pairs!(h, xc, uc, plan, value_bins, Val(D), keybuf, duLbuf, dn2buf,
    idxbuf, n_val, pair_blocks(N, ilist; grid = grid), weights)

"""
    _sp2d_simd_partial!(h, x, u, dist_be, value_bins, ::Val{D}, n_val, ilist)

Run [`_sp2d_simd_pairs!`](@ref) over an explicit outer-index list, with this worker's buffers.
"""
function _sp2d_simd_partial!(
    h::AbstractArray{OT, 4},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, value_bins, ::Val{D}, n_val::Int, ilist,
    culling::CullingPolicy = AutoCulling(), weights = NoWeights(),
) where {OT, D}
    x_raw = ntuple(d -> collect(view(x, d, :)), Val(D))
    u_raw = ntuple(d -> collect(view(u, d, :)), Val(D))
    N = length(x_raw[1])
    keybuf = Vector{eltype(x_raw[1])}(undef, N)
    duLbuf = Vector{OT}(undef, N)
    dn2buf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    plan = squared_digitize_plan(dist_be)
    grid = culling isa NoCulling ? nothing :
           cull_grid_for(x_raw, SFH.FlatGeometry{D}(), dist_be, culling)
    xc, uc = isnothing(grid) ? (x_raw, u_raw) :
             (apply_perm(x_raw, grid.perm), apply_perm(u_raw, grid.perm))
    wc = isnothing(grid) ? weights : _permuted_point_weights(weights, grid.perm)
    _sp2d_run_blocks!(h, xc, uc, plan, value_bins, Val(D),
        keybuf, duLbuf, dn2buf, idxbuf, n_val, ilist, N, grid, wc)
    return nothing
end

function _dispatch_single_pass_2d!(
    ::CB.AbstractSerialBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    return _accumulate_single_pass_2d!(sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_single_pass_2d!(
    ::CB.AbstractThreadedBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    throw(ArgumentError("Threaded in-place 2D single-pass is unavailable. Load OhMyThreads or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass_2d!(::CB.AbstractDistributedBackend, args...; kwargs...)
    throw(ArgumentError("Distributed in-place 2D single-pass is unavailable. Load Distributed (`using Distributed`) or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass_2d!(::CB.AbstractMPIBackend, args...; kwargs...)
    throw(ArgumentError("MPI in-place 2D single-pass is unavailable. Load MPI (`using MPI`) or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass_2d!(
    backend::CB.AbstractGPUBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    gpu_calculate_structure_functions_single_pass_2d!(sums_3d, counts_3d, backend.backend, x, u, distance_bins, value_bins; kwargs...)
    return sums_3d, counts_3d
end

function _dispatch_single_pass_2d!(
    ::CB.AbstractAutoBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    if distributed_adds_hardware(Val(:distributed))
        return _dispatch_single_pass_2d!(CB.DistributedBackend(), sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
    end
    return _dispatch_single_pass_2d!(_auto_local_backend(), sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
end

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

function _dispatch_single_pass_2d(::CB.AbstractThreadedBackend, args...; kwargs...)
    throw(ArgumentError("Threaded 2D single-pass backend is unavailable. Load the OhMyThreads extension or use backend=CB.SerialBackend()."))
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
    _require_threading("the auxiliary-axis 2D single-pass driver")
    sums = zeros(OT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    threaded_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass_2d(::CB.AbstractDistributedBackend, args...; kwargs...)
    throw(ArgumentError("Distributed 2D single-pass backend is unavailable. Load the Distributed extension or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass_2d(::CB.AbstractMPIBackend, args...; kwargs...)
    throw(ArgumentError("MPI 2D single-pass backend is unavailable. Load MPI (`using MPI`) or use backend=CB.SerialBackend()."))
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


function _dispatch_single_pass_2d(::CB.AbstractAutoBackend, shape::AbstractFieldShape, x::AbstractArray, u::AbstractArray,
                                  distance_bins::AbstractVector, value_bins::SinglePass2DValueBins, ::Type{CT};
                                  kwargs...) where {CT}
    backend = resolve_auto_backend(
        shape,
        _ohmythreads_loaded,
    )
    return _dispatch_single_pass_2d(backend, shape, x, u, distance_bins, value_bins, CT; kwargs...)
end

# Per-invariant value bins: a single vector is shared across invariants; a 6-tuple is per-invariant.
@inline _sp_valuebins(vb::AbstractVector, t::Int) = vb
@inline _sp_valuebins(vb::Tuple, t::Int) = vb[t]

"""
    _single_pass_collection_2d(sums, counts, distance_bins, value_bins, ::Type{OT})

Wrap the stacked 2D single-pass `(sums, counts)` (shape `(6, n_dist, n_val, aux...)`) into a
`NamedTuple` keyed by invariant, each value a `StructureFunction2DSumsAndCounts` view into the
stacked accumulator. In 2-D each invariant's value lands in a different value bin, so the per-cell
counts differ per invariant and are taken per invariant. The 2D joint
histogram has no averaged representation, so `OT` must be `StructureFunction2DSumsAndCounts`.
"""
function _single_pass_collection_2d(
    sums::AbstractArray, counts::AbstractArray, distance_bins, value_bins, ::Type{OT},
) where {OT}
    return (
        S2   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.S2, distance_bins, _sp_valuebins(value_bins, 1), _sp_rowview(sums, 1), _sp_rowview(counts, 1)), OT),
        L2   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.L2, distance_bins, _sp_valuebins(value_bins, 2), _sp_rowview(sums, 2), _sp_rowview(counts, 2)), OT),
        T2   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.T2, distance_bins, _sp_valuebins(value_bins, 3), _sp_rowview(sums, 3), _sp_rowview(counts, 3)), OT),
        S3   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.S3, distance_bins, _sp_valuebins(value_bins, 4), _sp_rowview(sums, 4), _sp_rowview(counts, 4)), OT),
        L3   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.L3, distance_bins, _sp_valuebins(value_bins, 5), _sp_rowview(sums, 5), _sp_rowview(counts, 5)), OT),
        L1T2 = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.L1T2, distance_bins, _sp_valuebins(value_bins, 6), _sp_rowview(sums, 6), _sp_rowview(counts, 6)), OT),
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
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT <: Real, OT <: SFO.AbstractStructureFunction}
    shape = _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), promote_type(float(FT1), float(FT2)))
    _assert_count_type(CT, size(x, 2), w)
    raw = _dispatch_single_pass_2d(backend, shape, x, u, distance_bins, value_bins, CT;
        distance_metric, weights = w, kwargs...)
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
