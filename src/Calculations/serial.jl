# Serial 1D CPU Reduction Kernels

function serial_calculate_structure_function!(
    output::AbstractVector{OT},
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector;
    geometry = SFH.FlatGeometry{length(u_vecs)}(),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, T1, T2}
    if verbose
        @info("calculating structure function (serial reduction)")
    end

    D = length(u_vecs)
    # A polynomial operator on a line: sorted once, every bin is an index range (sorted_line.jl).
    if _on_a_line(geometry, structure_function_type)
        _cull_reject_unsupported(culling, "the sorted line route")
        return sorted_line_sweep!(output, counts, structure_function_type, x_vecs[1],
            reshape(collect(u_vecs[1]), 1, :), distance_bins, Val(1), Val(1), Val(0); weights)
    end
    # Fast path: flat D ∈ (2,3) uses the SIMD compute/scatter-split kernel (vectorizes the per-pair
    # compute over j; only the histogram scatter is scalar). Curved geometries take the scalar
    # per-i kernel, which forms the frame through `pair_frame`.
    if geometry isa SFH.FlatGeometry && D == 2
        return _pf_simd_run!(output, counts, structure_function_type, x_vecs, u_vecs,
            distance_bins, Val(2); culling, weights)
    elseif geometry isa SFH.FlatGeometry && D == 3
        return _pf_simd_run!(output, counts, structure_function_type, x_vecs, u_vecs,
            distance_bins, Val(3); culling, weights)
    end
    _cull_reject_unsupported(culling, "the scalar per-point kernel that this geometry uses")

    be = digitize_plan(distance_bins)
    PM.@showprogress enabled = show_progress for i in eachindex(x_vecs[1])
        calculate_structure_function_i!(
            output, counts, geometry, structure_function_type, i, x_vecs, u_vecs, be, weights,
        )
    end
    return nothing
end

"""
    _pf_simd_run!(output, counts, sf, x_vecs, u_vecs, dist_be, ::Val{D}; culling, weights) -> mutates buffers

Point-field 1D (Euclidean) via the SIMD compute/scatter split. Materializes contiguous
per-component vectors (so consecutive `j` are unit-stride → packed loads), then for each `i`:
`@simd` over `j>i` computes distance + SF value into buffers (no scatter ⇒ vectorizes), and a
short scalar loop digitizes + scatters into the histogram. `Val{D}` keeps it type-stable.
"""
function _pf_simd_run!(
    output::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, ::Val{D};
    culling::CullingPolicy = AutoCulling(), weights = NoWeights(),
) where {OT, CT, D}
    N = length(x_vecs[1])
    return _pf_simd_partial!(output, counts, sf, x_vecs, u_vecs, dist_be, Val(D), 1:(N - 1), culling; weights)
end

"""The weight a point carries into its pairs; `true` for an unweighted sweep, which the compiler folds away."""
@inline _point_weight(::NoWeights, i::Int) = true
@inline _point_weight(w::AbstractVector, i::Int) = @inbounds w[i]

"""
Points per `j` block in the CPU pair loop. Sized so one block's coordinates, fields and the three
per-`j` buffers stay resident in a core's private cache while every `i` sweeps it.
"""
const SF_CPU_PAIR_TILE = 65536

"""
    _pf_simd_pairs!(output, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, blocks, weights)

Accumulate the pairs `(i, j>i)` covered by `blocks` into `output`/`counts`, each pair carrying
`weights[i] * weights[j]` in both.

`blocks` yields `(i-block, j-block)` index ranges (see [`block_pairs`](@ref)); each is worked to
completion, so the `j` block stays cache-resident across its whole `i` sweep. Under multi-core load
that is what keeps the loop off the memory bus: with one block spanning the array, per-core
throughput falls 83% once the arrays exceed L2.

Uniqueness is `j > i`, so a block pair never needs to know whether it lies on the diagonal, and a
culled schedule enumerating only nearby cells is exact for the same reason.

The `@simd` half writes `r²`, the SF value, and the approximate bin index to buffers; the scalar
half corrects the index and scatters straight into `output`/`counts`, skipping out-of-range bins.

The `i`-loop and the inner `@simd` must stay in this function body; factoring the inner loop into a
per-`i` helper stops it vectorizing.
"""
function _pf_simd_pairs!(
    output::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, ::Val{D},
    r2buf::AbstractVector, valbuf::AbstractVector, idxbuf::AbstractVector{Int32},
    blocks, weights,
) where {OT, CT, D}
    nb = n_histogram_bins(plan)
    FTx = eltype(xc[1])
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
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
                r2buf[j] = digitize_key(plan, r2)
                valbuf[j] = SFT._sf_raw(sf, Uj - Ui, dx, r2)
                if has_vector_index(plan)      # constant-folded: depends only on the plan type
                    idxbuf[j] = squared_approx_index(plan, r2)
                end
            end
            wi = _point_weight(weights, i)
            for j in jlo:j_last
                b = squared_bin(plan, r2buf[j], idxbuf[j])
                if 1 <= b <= nb
                    w = wi * _point_weight(weights, j)
                    output[b] += w * valbuf[j]
                    counts[b] += CT(w)
                end
            end
        end
    end
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

Throw unless the worst-case pair count `n_points*(n_points-1)÷2` fits in `CT`.

Every pair can land in one bin, so that product is the only safe bound. `UInt32` saturates at
`N = 92682`, past which the counter wraps silently.
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

"""
    _assert_counts_can_accumulate(counts, n_points, weights)

The count check of a mutating entry: [`_assert_count_type`](@ref), then room in `counts` for every
pair on top of what it holds.
"""
function _assert_counts_can_accumulate(counts::AbstractArray{CT}, n_points::Int, weights) where {CT}
    _assert_count_type(CT, n_points, weights)
    (weights isa NoWeights && !isempty(counts)) || return nothing
    current = maximum(counts)
    limit = CT <: Integer ? typemax(CT) : maxintfloat(CT)
    n_pairs = _pair_count_bound(n_points)
    (CT <: Unsigned || minimum(counts) >= 0) && current <= limit && n_pairs <= limit - current ||
        throw(ArgumentError(
            "count accumulator cannot represent existing counts plus $n_pairs possible pairs; use a wider " *
            "count type or reset the accumulator"))
    return nothing
end

"""
    _bin_average!(out, sums, counts)
    _bin_average(sums, counts)

Per-bin mean `sums ./ counts` with the empty-bin guard `count == 0 → NaN`. The cast uses
`eltype(out)` (so Float32 stays Float32, Float64 stays Float64). Elementwise: works for 1D
vectors, 2D matrices, and batched `(n_bins, batch...)` arrays whose `sums`/`counts` share a
shape. The allocating form returns a fresh array of `eltype(sums)`. This is the single
canonical averaging used by `_finalize`.
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
(`count == 0 → NaN`) and `eltype` preservation. Used by `_finalize` to average a tensor result.
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
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number}
    # `size(u_arr, 1)` is the velocity dimension here, before any conversion, so this is the one
    # place the geometry can be built. Everything downstream receives it.
    #
    # An array's axis length is a value, not a type parameter, so `Val` of it cannot be inferred and
    # a geometry built straight from it would type the whole sweep as `Any`. Branching on the two
    # common widths hands each branch a concretely typed geometry; the branch is chosen at runtime,
    # everything inside it is not.
    D = size(u_arr, 1)
    if D == 2
        return _serial_sf_with_geometry!(sums, counts, structure_function_type, x_arr, u_arr,
                                         distance_bins,
                                         SFH.pair_geometry_for(distance_metric, Val(2)); kwargs...)
    elseif D == 3
        return _serial_sf_with_geometry!(sums, counts, structure_function_type, x_arr, u_arr,
                                         distance_bins,
                                         SFH.pair_geometry_for(distance_metric, Val(3)); kwargs...)
    end
    return _serial_sf_with_geometry!(sums, counts, structure_function_type, x_arr, u_arr,
                                     distance_bins,
                                     SFH.pair_geometry_for(distance_metric, Val(D)); kwargs...)
end

function _serial_sf_with_geometry!(
    sums, counts, structure_function_type, x_arr, u_arr, distance_bins, geom; kwargs...,
)
    xk, uk = SFH.prepare_pair_inputs(geom, x_arr, u_arr)
    x_tuple = _component_vector_views(xk, SFH.coordinate_width(geom))
    u_tuple = _component_vector_views(uk, SFH.field_width(geom))
    return serial_calculate_structure_function!(
        sums,
        counts,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins;
        geometry = geom,
        kwargs...,
    )
end

function calculate_structure_function_i!(
    output::AbstractVector{OT},
    counts::AbstractVector,
    geom,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector,
    weights = NoWeights(),
) where {OT, T1, T2}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins)

    vW = SFH.coordinate_width(geom)
    vF = SFH.field_width(geom)
    W = _val_int(vW)
    F = _val_int(vF)
    X1 = SA.SVector{W, FT1}(ntuple(k -> @inbounds(x_vecs[k][i]), vW))
    U1 = SA.SVector{F, FT2}(ntuple(k -> @inbounds(u_vecs[k][i]), vF))
    wi = _point_weight(weights, i)

    iter_inds = eachindex(x_vecs[1])
    # @inbounds: x_vecs[k] are strided views; the bounds checks on every component access
    # were a large per-pair overhead. U2 is built only for in-range pairs.
    @inbounds for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{W, FT1}(ntuple(k -> x_vecs[k][j], vW))

        ok, distance, frame = SFH.pair_frame(geom, X1, X2)
        bin = SFH.digitize(distance, distance_bins)
        if ok && 1 <= bin < N3
            U2 = SA.SVector{F, FT2}(ntuple(k -> u_vecs[k][j], vF))
            δu, rh = SFH.pair_increments(geom, frame, distance, X1, X2, U1, U2)
            w = wi * _point_weight(weights, j)
            output[bin] += w * structure_function_type(δu, rh)
            counts[bin] += w
        end
    end
    return nothing
end

"""
    _partial_sums_counts(inner, sf_type, x_vecs, u_vecs, distance_bins, ilist, CT; kwargs...)

Partial 1D sums/counts over an explicit outer-index list `ilist` (each `i` contributes pairs
`(i, j>i)`). Used by the distributed driver to give each worker a balanced share; `inner`
selects how the worker computes its share locally. This generic method runs SERIALLY for any
backend; the OhMyThreads extension adds a `::CB.AbstractThreadedBackend` method that threads over `ilist`
(enabling hybrid distributed+threaded). Returns a `StructureFunctionSumsAndCounts`.
"""
function _partial_sums_counts(
    ::CB.AbstractExecutionBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    ilist,
    ::Type{CT};
    geometry = SFH.FlatGeometry{length(u_vecs)}(),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {CT}
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    nb = n_histogram_bins(distance_bins)
    sums = zeros(OT, nb)
    counts = zeros(CT, nb)
    D = length(u_vecs)
    # Flat D ∈ {2,3} takes the SIMD compute/scatter kernel, the same one the serial and threaded
    # drivers use; `_pf_simd_pairs!` accepts an arbitrary `irange`. Curved geometries take the
    # scalar per-`i` kernel.
    if geometry isa SFH.FlatGeometry && (D == 2 || D == 3)
        vD = D == 2 ? Val(2) : Val(3)
        _pf_simd_partial!(sums, counts, structure_function_type, x_vecs, u_vecs, distance_bins, vD, ilist,
                          culling; geometry = geometry, weights = weights)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    _cull_reject_unsupported(culling, "the scalar per-point kernel that this geometry uses")

    be = digitize_plan(distance_bins)
    for i in ilist
        calculate_structure_function_i!(
            sums, counts, geometry, structure_function_type, i, x_vecs, u_vecs, be, weights,
        )
    end
    return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
end

"""
    _pf_run_blocks!(sums, counts, sf, xc, uc, plan, ::Val{D}, bufs..., ilist, N, grid, weights)

Run the pair kernel with the schedule `grid` selects. Whether a grid exists is decided from the
data, so it arrives here as a `Union`; dispatching on it resolves that into one concretely typed
schedule per method, which is what keeps the kernel statically specialized.
"""
@inline _pf_run_blocks!(
    sums, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, ilist, N, ::Nothing, weights,
) where {D} = _pf_simd_pairs!(sums, counts, sf, xc, uc, plan, Val(D), r2buf, valbuf, idxbuf,
    pair_blocks(N, ilist), weights)

@inline _pf_run_blocks!(
    sums, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, ilist, N, grid::CellGrid, weights,
) where {D} = _pf_simd_pairs!(sums, counts, sf, xc, uc, plan, Val(D), r2buf, valbuf, idxbuf,
    pair_blocks(N, ilist; grid = grid), weights)

"""
    _pf_simd_partial!(sums, counts, sf, x_vecs, u_vecs, dist_be, ::Val{D}, ilist; kwargs...)

Run [`_pf_simd_pairs!`](@ref) over an explicit outer-index list, materializing the contiguous
component vectors and scratch buffers this worker needs. Shared by the distributed, MPI and
hybrid drivers, whose inputs arrive as strided views.

When `culling` yields a cull grid the points are sorted into it first; `ilist` then selects
positions in the sorted order, which leaves the union over workers unchanged. `weights`, one per
point, are sorted with them.
"""
function _pf_simd_partial!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, ::Val{D}, ilist, culling::CullingPolicy = AutoCulling();
    geometry = SFH.FlatGeometry{D}(), weights = NoWeights(),
) where {OT, CT, D}
    xc = ntuple(d -> collect(x_vecs[d]), Val(D))   # contiguous component vectors
    uc = ntuple(d -> collect(u_vecs[d]), Val(D))
    wc = weights isa NoWeights ? weights : collect(weights)
    N = length(xc[1])
    plan = squared_digitize_plan(dist_be)
    r2buf = Vector{eltype(xc[1])}(undef, N)
    valbuf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    grid = (culling isa NoCulling) ? nothing : cull_grid_for(xc, geometry, dist_be, culling) # this is type unstable
    if !isnothing(grid)
        xc = apply_perm(xc, grid.perm)
        uc = apply_perm(uc, grid.perm)
        wc = wc isa NoWeights ? wc : wc[grid.perm]
    end
    _pf_run_blocks!(sums, counts, sf, xc, uc, plan, Val(D),
        r2buf, valbuf, idxbuf, ilist, N, grid, wc)
    return nothing
end

"""
    _balanced_index_chunks(N, k) -> Vector of k index-lists

Split `1:N` into `k` balanced outer-index lists for the triangular pair loop (work ∝ N-i).
Round-robin assignment (`i ≡ w (mod k)`) gives each chunk a mix of cheap/expensive indices.
"""
function _balanced_index_chunks(N::Integer, k::Integer)
    k = max(1, k)
    # Ranges, not materialized vectors: `_partial_sums_counts` only iterates them, and the
    # distributed driver serializes one per worker.
    return [w:k:N for w in 1:k]
end
