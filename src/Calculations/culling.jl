# Pair culling: when the largest finite distance bin bounds the separation, only pairs inside that
# bound can land in a histogram bin, so the full upper triangle need not be enumerated.

"""
    cull_cutoff(geometry, r_max)

Euclidean cutoff, in the coordinate space the kernels see, that bounds a true separation of
`r_max`. Culling compares squared Euclidean distances against this, so it must never be smaller
than the true bound or in-range pairs would be dropped.

`nothing` means the geometry has declared no such bound, and culling then declines rather than
risk dropping pairs. A user-defined geometry is therefore correct by default and opts in by adding
a method — it is never required to have one.
"""
@inline cull_cutoff(::Any, r_max) = nothing

@inline cull_cutoff(::SFH.FlatGeometry, r_max) = r_max

# Kernel coordinates on a shell are the ambient unit position, so the Euclidean distance between
# two of them is the unit chord `2sin(σ/2)` for a central angle `σ = r/R`. Monotone for `r ≤ πR`;
# beyond that every pair is in range and the chord bound saturates at the diameter.
@inline function cull_cutoff(g::SFH.SphericalGeometry, r_max)
    half = r_max / (2 * g.radius)
    half >= π / 2 && return oftype(float(half), 2)
    return 2 * sin(half)
end

"""
    CellGrid{D, FT, IV, OV}

Uniform cell decomposition of a point set, with points sorted by cell id, so every cell is one
contiguous run: `cell_run(grid, c)` in the permuted order.

Only **occupied** cells are stored, as the sorted ids `cell_ids` and their run boundaries
`run_starts`. A dense array indexed by cell id would be `prod(dims)+1` long, which is unbounded:
`dims` grows as `(extent/cutoff)^D`, so a flat 3-D set at `r_max/L = 0.003` wants 2.3 GB and a
64 km cutoff on a shell wants 0.5 GB — while occupancy never exceeds the point count. Storing the
occupied cells makes both the memory and the block enumeration `O(N)` at any resolution.

The index and offset arrays are type parameters, not `Vector`s, so one type serves a host build and
a device-resident copy.
"""
struct CellGrid{D, FT, IV, OV}
    origin::NTuple{D, FT}
    inv_h::FT
    dims::NTuple{D, Int}
    cell_ids::IV      # sorted ids of the occupied cells
    run_starts::IV    # length(cell_ids)+1; cell_ids[k] occupies run_starts[k]:run_starts[k+1]-1
    perm::IV          # permuted index -> caller's original index
    offsets::OV       # row offsets over dims 2:D, shared by every point
    cutoff::FT
    span::Int
end

"""Occupied cells, which bounds every per-cell loop; `prod(dims)` is only the id space."""
@inline n_occupied_cells(g::CellGrid) = length(g.cell_ids)

"""Point range of cell `c`, empty when `c` holds no points."""
@inline function cell_run(g::CellGrid, c::Integer)
    k = searchsortedfirst(g.cell_ids, c)
    (k > length(g.cell_ids) || @inbounds(g.cell_ids[k]) != c) && return 1:0
    return @inbounds g.run_starts[k]:(g.run_starts[k + 1] - 1)
end

"""Point range of the `k`-th occupied cell, paired with its id."""
@inline function occupied_cell(g::CellGrid, k::Integer)
    return @inbounds(g.cell_ids[k]), @inbounds(g.run_starts[k]:(g.run_starts[k + 1] - 1))
end

"""
    cell_id_span_run(grid, c_lo, c_hi)

Point range covering every occupied cell with an id in `c_lo:c_hi`, empty when there is none.
Points are sorted by cell id, so a contiguous span of ids is a contiguous range of points — which
is what lets a whole stencil row be swept as one run.
"""
@inline function cell_id_span_run(g::CellGrid, c_lo::Integer, c_hi::Integer)
    k_lo = searchsortedfirst(g.cell_ids, c_lo)
    k_hi = searchsortedlast(g.cell_ids, c_hi)
    k_lo > k_hi && return 1:0
    return @inbounds g.run_starts[k_lo]:(g.run_starts[k_hi + 1] - 1)
end

"""Column-major cell id from a 1-based multi-index; dimension 1 varies fastest."""
@inline function cull_linear_index(dims::NTuple{D, Int}, c::NTuple{D, Int}) where {D}
    lin = c[D] - 1
    @inbounds for d in (D - 1):-1:1
        lin = lin * dims[d] + (c[d] - 1)
    end
    return lin + 1
end

"""1-based cell multi-index of point `i` of `xc`, clamped into the grid."""
@inline function cull_cell_multi_index(
    origin::NTuple{D, FT}, inv_h, dims::NTuple{D, Int}, xc::NTuple{D}, i::Integer,
) where {D, FT}
    return ntuple(Val(D)) do d
        c = 1 + floor(Int, (@inbounds(xc[d][i]) - origin[d]) * inv_h)
        min(max(c, 1), dims[d])
    end
end

"""
    cull_row_offsets(span, ::Val{D})

Each surviving stencil row as `(offset over dimensions 2:D, half-extent along dimension 1)`.
Dimension 1 is carried as an extent rather than an offset because cells adjacent along it are
contiguous in the sorted order, so a whole row is swept as one run; the extent is the widest that
still satisfies the corner test, so the row is no larger than the true stencil.
"""
function cull_row_offsets(span::Int, ::Val{D}) where {D}
    keep = Tuple{NTuple{D - 1, Int}, Int}[]
    for c in Iterators.product(ntuple(_ -> (-span):span, Val(D - 1))...)
        near2 = 0                                   # nearest-corner gap, in cell-side units
        for d in 1:(D - 1)
            g = max(abs(c[d]) - 1, 0)
            near2 += g * g
        end
        rem = span * span - near2
        rem < 0 && continue
        # widest |offset| along dimension 1 that still satisfies the corner test for this row
        push!(keep, (c, min(span, isqrt(rem) + 1)))
    end
    return keep
end

"""Cell-id space, in multiples of the point count, up to which a counting sort may be used."""
const SF_CULL_COUNTING_SORT_CELLS_PER_POINT = 8

"""
    _cull_sortperm(cell_raw, n_cells, N)

Permutation sorting the points by cell id.

A counting sort is faster but needs scratch proportional to the **cell-id space**, which is
unbounded in the cutoff. So it is used only while that space stays within a small multiple of the
point count; `sortperm` covers the rest, where the cells are mostly empty anyway.
"""
function _cull_sortperm(cell_raw::Vector{Int}, n_cells::Int, N::Int)
    n_cells <= SF_CULL_COUNTING_SORT_CELLS_PER_POINT * N || return sortperm(cell_raw)
    cursor = zeros(Int, n_cells + 1)
    @inbounds for i in 1:N
        cursor[cell_raw[i] + 1] += 1
    end
    @inbounds for c in 1:n_cells
        cursor[c + 1] += cursor[c]
    end
    perm = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        c = cell_raw[i]
        cursor[c] += 1
        perm[cursor[c]] = i
    end
    return perm
end

"""
    build_cell_grid(xc, cutoff, cells_per_cutoff)

Sort the points of `xc` into cells of side `cutoff / cells_per_cutoff` and return the
[`CellGrid`](@ref).

`O(N)` memory regardless of the cell-id space, which is unbounded in `cutoff`; the sort is
`O(N + n_cells)` while that space stays small and `O(N log N)` beyond it (see
[`_cull_sortperm`](@ref)).
"""
function build_cell_grid(
    xc::NTuple{D, <:AbstractVector{FT}}, cutoff::Real, cells_per_cutoff::Int,
) where {D, FT}
    N = length(xc[1])
    cutoff > 0 || throw(ArgumentError("cull cutoff must be positive (got $cutoff)"))
    cells_per_cutoff >= 1 ||
        throw(ArgumentError("cells_per_cutoff must be >= 1 (got $cells_per_cutoff)"))
    origin = ntuple(d -> FT(minimum(xc[d])), Val(D))
    hi = ntuple(d -> FT(maximum(xc[d])), Val(D))
    inv_h = inv(FT(cutoff) / cells_per_cutoff)
    dims = ntuple(d -> max(1, floor(Int, (hi[d] - origin[d]) * inv_h) + 1), Val(D))

    cell_raw = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        cell_raw[i] = cull_linear_index(dims, cull_cell_multi_index(origin, inv_h, dims, xc, i))
    end
    perm = _cull_sortperm(cell_raw, prod(dims), N)

    n_occ = 0
    @inbounds for p in 1:N
        (p == 1 || cell_raw[perm[p]] != cell_raw[perm[p - 1]]) && (n_occ += 1)
    end
    cell_ids = Vector{Int}(undef, n_occ)
    run_starts = Vector{Int}(undef, n_occ + 1)
    k = 0
    @inbounds for p in 1:N
        if p == 1 || cell_raw[perm[p]] != cell_raw[perm[p - 1]]
            k += 1
            cell_ids[k] = cell_raw[perm[p]]
            run_starts[k] = p
        end
    end
    @inbounds run_starts[n_occ + 1] = N + 1

    offsets = cull_row_offsets(cells_per_cutoff, Val(D))
    return CellGrid{D, FT, Vector{Int}, typeof(offsets)}(
        origin, inv_h, dims, cell_ids, run_starts, perm, offsets, FT(cutoff), cells_per_cutoff,
    )
end

"""
    apply_perm(vecs, perm)

Gather each component vector through `perm`, giving the cell-sorted layout the culled kernel needs.
"""
@inline apply_perm(
    vecs::NTuple{D, <:AbstractVector}, perm::AbstractVector{<:Integer},
) where {D} = ntuple(d -> vecs[d][perm], Val(D))


"""
    CullingPolicy

Whether a calculation may skip pairs that cannot land in a finite bin. Culling never changes the
result, so this selects only whether sorting the points is worth paying for.

- [`AutoCulling`](@ref) — cull when it removes pairs (the default)
- [`AlwaysCulling`](@ref) — cull whenever pairs can be skipped, and error when they cannot
- [`NoCulling`](@ref) — sweep every pair
"""
abstract type CullingPolicy end

"""Cull only when the cutoff is small enough to remove pairs; see [`CullingPolicy`](@ref)."""
struct AutoCulling <: CullingPolicy end

"""Cull whenever pairs can be skipped, and error when they cannot; see [`CullingPolicy`](@ref)."""
struct AlwaysCulling <: CullingPolicy end

"""Never cull; see [`CullingPolicy`](@ref)."""
struct NoCulling <: CullingPolicy end

"""Cells per cutoff in the cull grid; the stencil then reaches this many cells in each direction."""
const SF_CULL_CELLS_PER_CUTOFF = 2

"""Whether the last bin is unbounded, so every pair lands in a reported bin."""
@inline _cull_is_unbounded(::InfPaddedBinEdges) = true
@inline _cull_is_unbounded(be::BinEdges) = _cull_is_unbounded(be.edges)
@inline _cull_is_unbounded(_) = false

# The overflow bin is reported, and its sum needs each far pair's value, which needs that pair's
# displacement. So no pair can be skipped: only the digitize is knowable in advance, and that is
# not where the time goes.
_cull_on_unbounded(::AutoCulling) = nothing
_cull_on_unbounded(::AlwaysCulling) = throw(ArgumentError(
    "culling cannot skip any pair with InfPaddedBinEdges: the overflow bin is reported, and its " *
    "sum needs every far pair's value, which needs that pair's displacement. Only the digitize " *
    "could be skipped, which is not the cost. Pass culling = NoCulling(), or finite bin edges.",
))

# A geometry that declares no Euclidean bound (`cull_cutoff` returning `nothing`) cannot be culled
# safely: nothing relates its separations to the coordinates the cells are built from.
_cull_on_unknown_geometry(::AutoCulling, _) = nothing
_cull_on_unknown_geometry(::AlwaysCulling, geometry) = throw(ArgumentError(
    "culling cannot be used with $(typeof(geometry)): it declares no Euclidean bound relating " *
    "its separations to the kernel coordinates. Add a `cull_cutoff` method for it, or pass " *
    "culling = NoCulling().",
))

# `AlwaysCulling` sorts regardless; `AutoCulling` declines when the stencil already spans the grid,
# where culling removes no pairs. The test runs off the bounding box, before the sort is paid for.
_cull_is_worthwhile(::AlwaysCulling, _, _) = true
_cull_is_worthwhile(::AutoCulling, dims::NTuple{D, Int}, span::Int) where {D} =
    any(d -> dims[d] > 2 * span + 1, 1:D)

_cull_enabled(::NoCulling) = false
_cull_enabled(::CullingPolicy) = true

"""
    _cull_reject_unsupported(policy, what)

Refuse an explicit culling request on a path that cannot cull. `AutoCulling` means "cull where it
is supported and worthwhile", so it is a no-op here; `AlwaysCulling` is a request, and silently
ignoring it would hide that the pairs were never skipped.
"""
_cull_reject_unsupported(::AutoCulling, _) = nothing
_cull_reject_unsupported(::NoCulling, _) = nothing
_cull_reject_unsupported(::AlwaysCulling, what::AbstractString) = throw(ArgumentError(
    "culling = AlwaysCulling() is not supported by $what. Pass culling = AutoCulling() to cull " *
    "only where it is implemented, or culling = NoCulling().",
))

"""
    cull_cutoff_for(geometry, distance_bins, policy) -> cutoff or nothing

The Euclidean cutoff a cull grid would be built with, or `nothing` when culling does not apply:
the policy is [`NoCulling`](@ref), the last bin is unbounded, or the geometry declares no bound.
"""
function cull_cutoff_for(geometry, distance_bins, policy::CullingPolicy)
    _cull_enabled(policy) || return nothing
    _cull_is_unbounded(distance_bins) && return _cull_on_unbounded(policy)
    r_max = float(last(distance_bins))
    (isfinite(r_max) && r_max > 0) || return nothing
    cutoff = cull_cutoff(geometry, r_max)
    cutoff === nothing && return _cull_on_unknown_geometry(policy, geometry)
    (isfinite(cutoff) && cutoff > 0) || return nothing
    return cutoff
end

"""
    cull_grid_for(xc, geometry, distance_bins, policy) -> CellGrid or nothing

The cell grid to cull against, or `nothing` to sweep every pair.
"""
function cull_grid_for(
    xc::NTuple{D, <:AbstractVector{FT}}, geometry, distance_bins, policy::CullingPolicy,
) where {D, FT}
    cutoff = cull_cutoff_for(geometry, distance_bins, policy)
    cutoff === nothing && return nothing
    span = SF_CULL_CELLS_PER_CUTOFF
    inv_h = inv(FT(cutoff) / span)
    dims = ntuple(d -> max(1, floor(Int, (maximum(xc[d]) - minimum(xc[d])) * inv_h) + 1), Val(D))
    _cull_is_worthwhile(policy, dims, span) || return nothing
    return build_cell_grid(xc, cutoff, span)
end

"""
    cull_multi_index(dims, c)

1-based cell multi-index of linear cell id `c`, the inverse of [`cull_linear_index`](@ref).
Host-side: called once per cell when the work list is built, never per pair.
"""
@inline function cull_multi_index(dims::NTuple{D, Int}, c::Integer) where {D}
    r = c - 1
    return ntuple(Val(D)) do d
        stride = 1
        for k in 1:(d - 1)
            stride *= dims[k]
        end
        (r ÷ stride) % dims[d] + 1
    end
end

"""Linear-id shift of a stencil row's offset over dimensions `2:D`; dimension 1 varies fastest."""
@inline function cull_row_id_shift(dims::NTuple{D, Int}, off::NTuple{Dm1, Int}) where {D, Dm1}
    shift = 0
    stride = dims[1]
    @inbounds for d in 2:D
        shift += off[d - 1] * stride
        stride *= dims[d]
    end
    return shift
end

"""
    cull_sorted_matrices(x, u, geometry, distance_bins, policy) -> (grid, x, u)

Cull grid for the kernel coordinates held column-wise in `x`, with `x`/`u` reordered to match, or
`(nothing, x, u)` when no culling applies.

The grid is built in whatever space the kernels already see, which on a shell is the ambient unit
position, so a curved geometry needs no special case beyond its [`cull_cutoff`](@ref).
"""
function cull_sorted_matrices(
    x::AbstractMatrix, u::AbstractMatrix, geometry, distance_bins, policy::CullingPolicy,
)
    _cull_enabled(policy) || return nothing, x, u
    W = _val_int(SFH.coordinate_width(geometry))
    xc = ntuple(d -> view(x, d, :), W)
    grid = cull_grid_for(xc, geometry, distance_bins, policy)
    grid === nothing && return nothing, x, u
    return grid, x[:, grid.perm], u[:, grid.perm]
end

"""
    cull_sorted_inputs(x, u, geometry, distance_bins, policy) -> (grid, x, u)

Like [`cull_sorted_matrices`](@ref) but for the caller's own coordinates: the grid is built from
the kernel coordinates `geometry` derives from `x`, while `x`/`u` are returned reordered as given.

`prepare_pair_inputs` is per point, so it commutes with the reordering — a consumer may convert
after permuting and still agree with the grid.
"""
function cull_sorted_inputs(
    x::AbstractMatrix, u::AbstractMatrix, geometry, distance_bins, policy::CullingPolicy,
)
    _cull_enabled(policy) || return nothing, x, u
    xk, _ = SFH.prepare_pair_inputs(geometry, x, u)
    W = _val_int(SFH.coordinate_width(geometry))
    grid = cull_grid_for(ntuple(d -> view(xk, d, :), W), geometry, distance_bins, policy)
    grid === nothing && return nothing, x, u
    return grid, x[:, grid.perm], u[:, grid.perm]
end
