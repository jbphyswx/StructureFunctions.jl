function serial_calculate_structure_function!(
    output::AbstractVector{OT},
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector;
    geometry = default_geometry(u_vecs),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {OT, CT, T1, T2}
    # Polynomial operator on a line: sorted once, every bin is an index range.
    if _on_a_line(geometry, structure_function_type)
        return sorted_line_sweep!(output, counts, structure_function_type, x_vecs[1],
            reshape(collect(u_vecs[1]), 1, :), distance_bins, Val(1), Val(1), Val(0); weights)
    end
    # Flat D ∈ {2,3}: SIMD compute/scatter-split kernel; other geometries: scalar kernel through `pair_frame`.
    vD = _simd_width(geometry)
    vD === nothing || return _pf_simd_run!(output, counts, structure_function_type, x_vecs, u_vecs, distance_bins, vD;
                                           culling, weights)
    _pf_scalar_run!(output, counts, geometry, structure_function_type, distance_bins, nothing,
                    _cull_sorted(x_vecs, u_vecs, weights, geometry, distance_bins, culling))
    return nothing
end

"""
    _pf_scalar_run!(output, counts, geometry, sf, distance_bins, share, (grid, x_vecs, u_vecs, weights))

Run [`_pf_scalar_pairs!`](@ref) over the outer indices `share` selects ([`_share_indices`](@ref)) of a
[`_cull_sorted`](@ref) result.
"""
function _pf_scalar_run!(output, counts, geometry, sf, distance_bins, share, (grid, xc, uc, wc))
    N = length(xc[1])
    _pf_scalar_pairs!(output, counts, geometry, sf, xc, uc, digitize_plan(distance_bins),
                      pair_blocks(N, _share_indices(grid, N - 1, share); grid), wc)
    return nothing
end

"""
    _cull_sorted(x_vecs, u_vecs, weights, geometry, distance_bins, culling) -> (grid, x_vecs, u_vecs, weights)

The component tuples and point weights sorted into the cull grid over the kernel coordinates, or
unchanged with `grid === nothing` when `culling` declines.
"""
function _cull_sorted(x_vecs::Tuple, u_vecs::Tuple, weights, geometry, distance_bins, culling::CullingPolicy)
    _cull_enabled(culling) || return nothing, x_vecs, u_vecs, weights
    grid = cull_grid_for(x_vecs, geometry, distance_bins, culling)
    grid === nothing && return nothing, x_vecs, u_vecs, weights
    p = grid.perm
    return grid, apply_perm(x_vecs, p), apply_perm(u_vecs, p), _permuted_point_weights(weights, p)
end

"""
    _pf_simd_run!(output, counts, sf, x_vecs, u_vecs, dist_be, ::Val{D}; culling, weights) -> mutates buffers

Point-field 1D (Euclidean) via the SIMD compute/scatter split. Materializes contiguous
per-component vectors, then for each `i`: `@simd` over `j>i` computes distance + SF value into
buffers, and a scalar loop digitizes + scatters into the histogram.
"""
function _pf_simd_run!(
    output::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, ::Val{D};
    culling::CullingPolicy = AutoCulling(), weights = NoWeights(),
) where {OT, CT, D}
    return _pf_simd_partial!(output, counts, sf, x_vecs, u_vecs, dist_be, Val(D), nothing, culling; weights)
end

"""The weight a point carries into its pairs; `true` for an unweighted sweep, which the compiler folds away."""
@inline _point_weight(::NoWeights, i::Int) = true
@inline _point_weight(w::AbstractVector, i::Int) = @inbounds w[i]

"""Point `j` of the component tuple `c` as a static vector, unchecked: the kernels index within their blocks."""
@inline _component_point(c::NTuple{D}, j, ::Val{D}) where {D} = SA.SVector{D}(ntuple(d -> @inbounds(c[d][j]), Val(D)))

"""
    _pf_simd_pairs!(output, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, sel, window, blocks, weights)

Accumulate the pairs `(i, j>i)` covered by `blocks` into `output`/`counts`, each pair carrying
`weights[i] * weights[j]` in both.

`blocks` yields `(i-block, j-block)` index ranges (see [`block_pairs`](@ref)); each is worked to
completion, so the `j` block stays cache-resident across its whole `i` sweep.

Uniqueness is `j > i`, independent of whether a block pair lies on the diagonal.

The `@simd` half writes the digitize key, the SF value, and the approximate bin index to buffers; the
scalar half scatters the in-range pairs straight into `output`/`counts`, under a range branch or, on a
schedule that [`_chooses_compaction`](@ref) and a run whose sampled keys [`_compacts`](@ref), through the
in-range list [`_compact_in_range!`](@ref) builds in `sel`. The buffers are indexed by `window`
([`PairWindow`](@ref)).

The `i`-loop and the inner `@simd` must stay in this function body to vectorize.
"""
function _pf_simd_pairs!(
    output::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, ::Val{D},
    r2buf::AbstractVector, valbuf::AbstractVector, idxbuf::AbstractVector{Int32}, sel::AbstractVector{Int32},
    window::PairWindow, blocks, weights,
) where {OT, CT, D}
    nb = n_histogram_bins(plan)
    chooses = _chooses_compaction(blocks)
    FTx = eltype(xc[1])
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        _check_run_fits(window, valbuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
            Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
            @simd for j in jlo:j_last
                Xj = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D)))
                dx = Xj - Xi
                r2 = SFH.norm2(dx)
                Uj = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D)))
                r2buf[j - o] = digitize_key(plan, r2)
                valbuf[j - o] = SFT.flat_pair_value(sf, Uj - Ui, dx, r2)
                if has_vector_index(plan)      # constant-folded: depends only on the plan type
                    idxbuf[j - o] = squared_approx_index(plan, r2)
                end
            end
            wi = _point_weight(weights, i)
            ks = (jlo - o):(j_last - o)
            if chooses && _compacts(_sample_in_range(plan, r2buf, ks)...)
                for m in 1:_compact_in_range!(sel, plan, r2buf, ks)
                    k = Int(sel[m])
                    _pf_accumulate!(output, counts, valbuf, weights, wi, o, k,
                                    squared_bin_select(plan, r2buf[k], idxbuf[k]))
                end
            else
                for k in ks
                    b = chooses ? squared_bin_select(plan, r2buf[k], idxbuf[k]) : squared_bin(plan, r2buf[k], idxbuf[k])
                    if 1 <= b <= nb
                        _pf_accumulate!(output, counts, valbuf, weights, wi, o, k, b)
                    end
                end
            end
        end
    end
    return nothing
end

"""Add slot `k`'s buffered value into bin `b` of `output`/`counts`, weighted by the outer point's weight `wi`
times that of point `k + o`."""
@inline function _pf_accumulate!(output, counts::AbstractVector{CT}, valbuf, weights, wi, o, k, b) where {CT}
    w = wi * _point_weight(weights, k + o)
    @inbounds output[b] += w * valbuf[k]
    @inbounds counts[b] += CT(w)
    return nothing
end

"""The unordered pair count `n(n-1)÷2` of `n` points, in `UInt128`, which holds it for every `Int` `n`."""
function _pair_count_bound(n::Int)
    n >= 0 || throw(ArgumentError("point count must be nonnegative"))
    wide = UInt128(n)
    return n < 2 ? UInt128(0) : wide * (wide - 1) ÷ 2
end

"""
    _assert_counts_representable(CT, n_points)

Throw unless the worst-case pair count `n_points*(n_points-1)÷2`, all pairs in one bin, fits in `CT`.
"""
@inline function _assert_counts_representable(::Type{CT}, n_points::Int) where {CT <: Integer}
    n_pairs = _pair_count_bound(n_points)
    n_pairs <= typemax(CT) || throw(
        ArgumentError(
            "the count type $CT cannot represent the worst-case pair count $n_pairs for N=$n_points " *
            "(typemax($CT) = $(typemax(CT))); pass UInt64 or Int64 as the count type.",
        ),
    )
    return nothing
end

# A floating-point count (a joint histogram over angle splits pairs between bins) is exact up to
# `maxintfloat`.
@inline function _assert_counts_representable(::Type{CT}, n_points::Int) where {CT <: AbstractFloat}
    n_pairs = _pair_count_bound(n_points)
    n_pairs <= maxintfloat(CT) || throw(
        ArgumentError(
            "the count type $CT counts exactly only up to $(maxintfloat(CT)), below the worst-case pair " *
            "count $n_pairs for N=$n_points; pass Float64 as the count type.",
        ),
    )
    return nothing
end

"""
    _assert_mass_counts(CT)

The count check of a soft-binned entry, whose counts are a kernel-weighted pair mass: a real number,
negative in a kernel's sidelobes, which only a floating-point `CT` holds.
"""
_assert_mass_counts(::Type{CT}) where {CT} = CT <: AbstractFloat ? nothing : throw(ArgumentError(
    "a soft-binned count is a kernel-weighted pair mass, which the count type $CT cannot hold; pass a " *
    "floating-point count type"))

"""
    _assert_count_type(CT, n_points, weights)

The count check of an allocating entry: weighted pairs need a floating-point `CT`, and an unweighted
sweep's worst case must fit it.
"""
@inline function _assert_count_type(::Type{CT}, n_points::Int, weights) where {CT}
    _check_weighted_counts(weights, CT)
    weights isa NoWeights && _assert_counts_representable(CT, n_points)
    return nothing
end

"""The least of the first and the greatest of the second members of two pairs."""
@inline _min_max(a, b) = (min(a[1], b[1]), max(a[2], b[2]))

"""The least and the greatest of `counts`, in one pass; an unsigned count's least is zero."""
_count_extrema(counts::AbstractArray{CT}) where {CT <: Unsigned} = (zero(CT), maximum(counts))
_count_extrema(counts::AbstractArray{CT}) where {CT} =
    mapreduce(v -> (v, v), _min_max, counts; init = (typemax(CT), typemin(CT)))

"""
    _assert_counts_can_accumulate(counts, n_points, weights)

The count check of a mutating entry: [`_assert_count_type`](@ref), then room in `counts` for every
pair on top of what it holds.
"""
function _assert_counts_can_accumulate(counts::AbstractArray{CT}, n_points::Int, weights) where {CT}
    _assert_count_type(CT, n_points, weights)
    (weights isa NoWeights && !isempty(counts)) || return nothing
    least, current = _count_extrema(counts)
    limit = CT <: Integer ? typemax(CT) : maxintfloat(CT)
    n_pairs = _pair_count_bound(n_points)
    least >= 0 && current <= limit && n_pairs <= limit - current ||
        throw(ArgumentError(
            "count accumulator cannot represent existing counts plus $n_pairs possible pairs; use a wider " *
            "count type or reset the accumulator"))
    return nothing
end

"""
    _bin_average!(out, sums, counts)
    _bin_average(sums, counts)

Per-bin mean `sums ./ counts` with the empty-bin guard `count == 0 → NaN`, cast to `eltype(out)`.
Elementwise over `sums`/`counts` of a shared shape. The allocating form returns a fresh array of
`eltype(sums)`.
"""
function _bin_average!(out::AbstractArray{T}, sums::AbstractArray, counts::AbstractArray) where {T}
    axes(out) == axes(sums) == axes(counts) || throw(DimensionMismatch("mean buffers must have matching axes"))
    out .= ifelse.(iszero.(counts), T(NaN), sums ./ counts)
    return out
end

@inline _bin_average(sums::AbstractArray, counts::AbstractArray) =
    _bin_average!(similar(sums, eltype(sums)), sums, counts)

"""
    _tensor_bin_average(sums, counts, ::Val{P})

Tensor analogue of `_bin_average`: `counts` (indexed by `(bin, aux...)`) broadcasts over
the `P` leading component axes of `sums` (shape `(D×P..., n_bins, aux...)`). Same empty-bin guard
(`count == 0 → NaN`) and `eltype` preservation.
"""
function _tensor_bin_average(sums::AbstractArray, counts::AbstractArray, ::Val{P}) where {P}
    T = eltype(sums)
    out = similar(sums, T)
    expanded_counts = reshape(counts, ntuple(_ -> 1, Val(P))..., size(counts)...)
    out .= ifelse.(iszero.(expanded_counts), T(NaN), sums ./ expanded_counts)
    return out
end

function serial_calculate_structure_function!(
    sums::AbstractVector{OT},
    counts::AbstractVector,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector;
    geometry,
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number}
    x_tuple, u_tuple = _prepared_tuples(geometry, x_arr, u_arr)
    return serial_calculate_structure_function!(sums, counts, structure_function_type, x_tuple, u_tuple, distance_bins;
                                                geometry, kwargs...)
end

"""
    _pf_scalar_pairs!(output, counts, geom, sf, x_vecs, u_vecs, dist_be, blocks, weights)

Accumulate the pairs `(i, j>i)` covered by `blocks` for any geometry, one pair at a time through
`pair_frame`.
"""
function _pf_scalar_pairs!(
    output::AbstractVector{OT},
    counts::AbstractVector{CT},
    geom,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    dist_be,
    blocks,
    weights = NoWeights(),
) where {OT, CT, T1, T2}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    nb = n_histogram_bins(dist_be)
    vW = SFH.coordinate_width(geom)
    vF = SFH.field_width(geom)
    W = _val_int(vW)
    F = _val_int(vF)
    for (ir, jr) in blocks, i in ir
        X1 = SA.SVector{W, FT1}(ntuple(k -> @inbounds(x_vecs[k][i]), vW))
        U1 = SA.SVector{F, FT2}(ntuple(k -> @inbounds(u_vecs[k][i]), vF))
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            X2 = SA.SVector{W, FT1}(ntuple(k -> @inbounds(x_vecs[k][j]), vW))
            ok, distance, frame = SFH.pair_frame(geom, X1, X2)
            bin = SFH.digitize(distance, dist_be)
            if ok && 1 <= bin <= nb
                U2 = SA.SVector{F, FT2}(ntuple(k -> @inbounds(u_vecs[k][j]), vF))
                δu = SFH.pair_delta(geom, frame, X1, X2, U1, U2)
                w = wi * _point_weight(weights, j)
                @inbounds output[bin] += w * SFT.pair_value(structure_function_type, geom, frame, distance, δu)
                @inbounds counts[bin] += CT(w)
            end
        end
    end
    return nothing
end

"""
    _partial_sums_counts(inner, sf_type, x_vecs, u_vecs, distance_bins, share, CT; kwargs...)

Partial 1D sums/counts of a distributed worker or MPI rank: the pairs `(i, j > i)` whose outer index `i` is in share
`share = (w, k)`, resolved against the cull grid the call builds ([`_share_indices`](@ref)), so the shares of
`w = 1:k` partition the sweep. `inner` selects how the share is computed locally: serially here, threaded by the
OhMyThreads extension. Returns a `StructureFunctionSumsAndCounts`.
"""
function _partial_sums_counts(
    ::CB.AbstractExecutionBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    share::NTuple{2, Int},
    ::Type{CT};
    geometry = default_geometry(u_vecs),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {CT}
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    nb = n_histogram_bins(distance_bins)
    sums = zeros(OT, nb)
    counts = zeros(CT, nb)
    # Flat D ∈ {2,3}: SIMD kernel; other geometries: scalar kernel.
    vD = _simd_width(geometry)
    if vD !== nothing
        _pf_simd_partial!(sums, counts, structure_function_type, x_vecs, u_vecs, distance_bins, vD, share,
                          culling; geometry, weights)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    _pf_scalar_run!(sums, counts, geometry, structure_function_type, distance_bins, share,
                    _cull_sorted(x_vecs, u_vecs, weights, geometry, distance_bins, culling))
    return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
end

"""
    _pf_run_blocks!(sums, counts, sf, xc, uc, plan, ::Val{D}, bufs..., ilist, N, grid, weights)

Run the pair kernel with the schedule `grid` selects (`nothing` or a `CellGrid`), over buffers sized by
[`_pair_scratch_length`](@ref)`(N)`.
"""
@inline _pf_run_blocks!(
    sums, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, sel, ilist, N, ::Nothing, weights,
) where {D} = _pf_simd_pairs!(sums, counts, sf, xc, uc, plan, Val(D), r2buf, valbuf, idxbuf, sel,
    _pair_window(N), pair_blocks(N, ilist), weights)

@inline _pf_run_blocks!(
    sums, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, sel, ilist, N, grid::CellGrid, weights,
) where {D} = _pf_simd_pairs!(sums, counts, sf, xc, uc, plan, Val(D), r2buf, valbuf, idxbuf, sel,
    _pair_window(N), pair_blocks(N, ilist; grid = grid), weights)

"""
    _pf_simd_partial!(sums, counts, sf, x_vecs, u_vecs, dist_be, ::Val{D}, share; kwargs...)

Run [`_pf_simd_pairs!`](@ref) over the outer indices `share` selects ([`_share_indices`](@ref)), materializing the
contiguous component vectors and scratch buffers this call needs; the inputs may arrive as strided views.

When `culling` yields a cull grid the points are sorted into it first, and the share is taken of the sorted order.
`weights`, one per point, are sorted with them.
"""
function _pf_simd_partial!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, ::Val{D}, share, culling::CullingPolicy = AutoCulling();
    geometry = SFH.FlatGeometry{D}(), weights = NoWeights(),
) where {OT, CT, D}
    xc = ntuple(d -> collect(x_vecs[d]), Val(D))   # contiguous component vectors
    uc = ntuple(d -> collect(u_vecs[d]), Val(D))
    wc = weights isa NoWeights ? weights : collect(weights)
    N = length(xc[1])
    plan = squared_digitize_plan(dist_be)
    L = _pair_scratch_length(N)
    r2buf = Vector{eltype(xc[1])}(undef, L)
    valbuf = Vector{OT}(undef, L)
    idxbuf = Vector{Int32}(undef, L)
    sel = Vector{Int32}(undef, L)
    grid = (culling isa NoCulling) ? nothing : cull_grid_for(xc, geometry, dist_be, culling) # this is type unstable
    if !isnothing(grid)
        xc = apply_perm(xc, grid.perm)
        uc = apply_perm(uc, grid.perm)
        wc = wc isa NoWeights ? wc : wc[grid.perm]
    end
    _pf_run_blocks!(sums, counts, sf, xc, uc, plan, Val(D),
        r2buf, valbuf, idxbuf, sel, _share_indices(grid, N - 1, share), N, grid, wc)
    return nothing
end

"""
    _outer_share(grid, indices, w, k)

Share `w` of `k` of the outer indices of a sweep over `grid`'s schedule, a range: every `k`-th index from the
`w`-th without a cull grid; the `w`-th of `k` consecutive runs with a grid or per-slice grids.
"""
@inline _outer_share(::Nothing, indices::AbstractRange, w::Integer, k::Integer) = indices[w:k:end]
@inline function _outer_share(::Union{CellGrid, AbstractVector}, indices::AbstractRange, w::Integer, k::Integer)
    n = length(indices)
    return indices[(((w - 1) * n) ÷ k + 1):((w * n) ÷ k)]
end

"""The `max(1, k)` shares [`_outer_share`](@ref) cuts `indices` into."""
_outer_shares(grid, indices::AbstractRange, k::Integer) = [_outer_share(grid, indices, w, max(1, k)) for w in 1:max(1, k)]

"""The outer indices `1:n` a call sweeps over `grid`'s schedule: all of them, or with `share = (w, k)` share `w` of
`k` ([`_outer_share`](@ref)), resolved against the grid the call itself built."""
@inline _share_indices(grid, n::Int, ::Nothing) = 1:n
@inline _share_indices(grid, n::Int, (w, k)::NTuple{2, Int}) = _outer_share(grid, 1:n, w, k)
