# Cull preparation on the device of any KernelAbstractions backend: the cell grid and the tile-pair work
# list are built by kernels, sorts and scans where the coordinates live.

"""Pairs of one slice a cull grid over `d` coordinates must serve before `AutoCulling` prepares one for a single call,
indexed by `d`."""
const GPU_CULL_COLD_MIN_PAIRS = (15 * 10^7, 15 * 10^7, 25 * 10^7)

"""
    _cull_pays(policy, workspace, n_points, geom) -> Bool

Whether to prepare a cull grid for `n_points` points in the coordinates of `geom`: always for `AlwaysCulling`, and for
`AutoCulling` when a workspace keeps the grid for later calls or one slice's pairs reach `GPU_CULL_COLD_MIN_PAIRS` at the
grid's coordinates; the slices sharing the grid do not count.
"""
_cull_pays(::SFC.AlwaysCulling, workspace, n_points::Int, geom) = true
_cull_pays(::SFC.AutoCulling, ::SFC.GPUSFWorkspace, n_points::Int, geom) = true
_cull_pays(::SFC.AutoCulling, ::Nothing, n_points::Int, geom) =
    n_points * (n_points - 1) ÷ 2 >=
    GPU_CULL_COLD_MIN_PAIRS[min(SFC._val_int(SFH.coordinate_width(geom)), SFC.SF_CULL_GRID_DIMS)]

"""Whether any slice of a batch whose slices carry positions of their own, `n_points` each, may prepare a cull grid:
the policy culls, `distance_bins` bound a cutoff, and one slice's grid pays ([`_cull_pays`](@ref))."""
_slices_may_cull(culling, workspace, geom, distance_bins, n_points::Int) =
    SFC._cull_enabled(culling) && SFC.cull_cutoff_for(geom, distance_bins, culling) !== nothing &&
    _cull_pays(culling, workspace, n_points, geom)

"""Index of the first element of the sorted integer vector `v` that is `≥ x`, or `length(v) + 1`."""
@inline function _first_at_least(v, x)
    lo, hi = 0, length(v) + 1
    while hi - lo > 1
        m = (lo + hi) >>> 1
        if @inbounds(v[m]) < x
            lo = m
        else
            hi = m
        end
    end
    return hi
end

"""`keys[i]` is the cell id of point `i` less one."""
KA.@kernel function _cull_cell_keys!(keys, xc, origin, inv_h, dims)
    i = @index(Global)
    K = eltype(keys)
    @inbounds keys[i] = K(SFC.cull_linear_index(dims, SFC.cull_cell_multi_index(origin, inv_h, dims, xc, i)) - 1)
end

"""`flags[p]` is whether `sorted[p]` begins a run of equal values."""
KA.@kernel function _run_flags!(flags, @Const(sorted))
    p = @index(Global)
    @inbounds flags[p] = p == 1 || sorted[p] != sorted[p - 1]
end

"""Each run of the `n` sorted cell keys that `flags` marks and `ends` counts (runs begun up to each key): its cell id
into `cell_ids` and its first position into `run_starts`, and `n + 1` after the last."""
KA.@kernel function _run_write!(cell_ids, run_starts, @Const(sorted), @Const(flags), @Const(ends), n)
    p = @index(Global)
    @inbounds if flags[p] != 0
        k = ends[p]
        cell_ids[k] = Int(sorted[p]) + 1
        run_starts[k] = p
    end
    @inbounds p == n && (run_starts[ends[n] + 1] = n + 1)
end

"""The occupied cells of the sorted zero-based cell keys `sorted`, as one-based ids, and where each one's points
begin, `length(sorted) + 1` appended."""
function _cell_runs(backend, sorted::AbstractVector)
    n = length(sorted)
    flags = KA.allocate(backend, _radix_index_type(n), n)
    _run_flags!(backend)(flags, sorted; ndrange = n)
    ends = cumsum(flags)
    n_occ = Int(only(Array(view(ends, n:n))))
    cell_ids = KA.allocate(backend, Int, n_occ)
    run_starts = KA.allocate(backend, Int, n_occ + 1)
    _run_write!(backend)(cell_ids, run_starts, sorted, flags, ends, n; ndrange = n)
    return cell_ids, run_starts
end

"""The least and the greatest of each of the `W` coordinates (rows) of `x`, as two host tuples."""
function _extents(x::AbstractMatrix{FT}, ::Val{W}) where {FT, W}
    e = Array(mapreduce(v -> (v, v), _min_max, x; dims = 2, init = (typemax(FT), typemin(FT))))
    return ntuple(d -> e[d][1], Val(W)), ntuple(d -> e[d][2], Val(W))
end

"""
    _gpu_cull_grid(backend, x, geom, cutoff, policy) -> CellGrid or nothing

The cull grid of the `(W, N)` kernel coordinates `x` on `backend`, over the coordinates the host rule
chooses ([`SFC._cull_grid_axes`](@ref)); `nothing` when `policy` declines it or there is no pair. The cell
ids, run starts and permutation stay on the device.
"""
function _gpu_cull_grid(backend, x::AbstractMatrix{FT}, geom, cutoff, policy) where {FT}
    n = size(x, 2)
    n >= 2 || return nothing
    lo, hi = _extents(x, SFH.coordinate_width(geom))
    axes = SFC._cull_grid_axes(map(-, hi, lo))
    span = SFC.SF_CULL_CELLS_PER_CUTOFF
    inv_h = inv(FT(cutoff) / span)
    origin = map(a -> lo[a], axes)
    dims = SFC._cull_dims(origin, map(a -> hi[a], axes), inv_h)
    dims === nothing && return SFC._cull_grid_too_fine(policy)
    SFC._cull_is_worthwhile(policy, dims, span) || return nothing
    n_cells = prod(dims)
    keys = KA.allocate(backend, n_cells <= typemax(UInt32) ? UInt32 : UInt64, n)
    _cull_cell_keys!(backend)(keys, map(a -> view(x, a, :), axes), origin, inv_h, dims; ndrange = n)
    sorted, perm = _radix_sortperm(keys, 8 * sizeof(Int) - leading_zeros(n_cells - 1))
    cell_ids, run_starts = _cell_runs(backend, sorted)
    return SFC.CellGrid(origin, inv_h, dims, cell_ids, run_starts, perm,
                        SFC.cull_row_offsets(span, Val(length(axes))), FT(cutoff), span)
end

"""The tiles `lo:hi` that stencil row `row = (id shift, half-extent along dimension 1)` reaches from tile
`ti`, empty when it reaches no occupied cell."""
@inline function _row_tiles(cell_ids, run_starts, row, n_cells, n_points, tile, ti)
    shift, e1 = row
    k_lo = _first_at_least(run_starts, (ti - 1) * tile + 2) - 1
    k_hi = _first_at_least(run_starts, min(ti * tile, n_points) + 1) - 1
    q_lo = _first_at_least(cell_ids, max(1, @inbounds(cell_ids[k_lo]) + shift - e1))
    q_hi = _first_at_least(cell_ids, min(n_cells, @inbounds(cell_ids[k_hi]) + shift + e1) + 1) - 1
    q_lo > q_hi && return 1, 0
    return cld(@inbounds(run_starts[q_lo]), tile), cld(@inbounds(run_starts[q_hi + 1]) - 1, tile)
end

"""Stencil rows of a cull grid over the most coordinates a grid is built on."""
const CULL_MAX_ROWS = (2 * SFC.SF_CULL_CELLS_PER_CUTOFF + 1)^(SFC.SF_CULL_GRID_DIMS - 1)

"""`iv[:, r, t]`: the first and last partner tiles `≤ t` that stencil row `rows[r]` reaches from tile `t`, the first
past the last when it reaches none."""
KA.@kernel function _worklist_rows!(iv, @Const(cell_ids), @Const(run_starts), @Const(rows), n_cells, n_points,
                                    tile)
    r, t = @index(Global, NTuple)
    a, b = _row_tiles(cell_ids, run_starts, @inbounds(rows[r]), n_cells, n_points, tile, t)
    @inbounds iv[1, r, t] = a
    @inbounds iv[2, r, t] = min(b, t)
end

"""Tile `t`'s row intervals `iv[:, :, t]` rewritten in place as the disjoint intervals of their union, in increasing
order, their count into `n_iv[t]` and their tiles into `counts[t]`."""
KA.@kernel function _worklist_merge!(counts, n_iv, iv, n_rows)
    t = @index(Global)
    lo = @private Int (CULL_MAX_ROWS,)
    hi = @private Int (CULL_MAX_ROWS,)
    m = 0
    for r in 1:n_rows
        a, b = Int(@inbounds(iv[1, r, t])), Int(@inbounds(iv[2, r, t]))
        a > b && continue
        k = m
        while k >= 1 && @inbounds(lo[k]) > a
            @inbounds lo[k + 1], hi[k + 1] = lo[k], hi[k]
            k -= 1
        end
        @inbounds lo[k + 1], hi[k + 1] = a, b
        m += 1
    end
    w, c = 0, 0
    for q in 1:m
        a, b = @inbounds(lo[q]), @inbounds(hi[q])
        if w >= 1 && a <= @inbounds(iv[2, w, t]) + 1
            e = Int(@inbounds(iv[2, w, t]))
            if b > e
                c += b - e
                @inbounds iv[2, w, t] = b
            end
        else
            w += 1
            @inbounds iv[1, w, t] = a
            @inbounds iv[2, w, t] = b
            c += b - a + 1
        end
    end
    @inbounds n_iv[t] = w
    @inbounds counts[t] = c
end

"""The packed pairs `(t′, t)` of tile `t`'s partners `t′ ≤ t` in increasing order, from its merged intervals, ending
at `ends[t]`."""
KA.@kernel function _worklist_write!(packed, @Const(ends), @Const(n_iv), @Const(iv), n_tiles)
    t = @index(Global)
    k = t == 1 ? 0 : Int(@inbounds(ends[t - 1]))
    I = eltype(packed)
    for q in 1:Int(@inbounds(n_iv[t]))
        for tp in Int(@inbounds(iv[1, q, t])):Int(@inbounds(iv[2, q, t]))
            k += 1
            @inbounds packed[k] = SFC.pack_tile_pair(I(tp), I(t), I(n_tiles))
        end
    end
end

function SFC.gpu_tile_worklist(grid::SFC.CellGrid, n_points::Int, tile::Int)
    n_tiles = cld(n_points, tile)
    n_tiles * n_tiles <= typemax(Int32) && return _tile_worklist(Int32, grid, n_points, tile, n_tiles)
    return _tile_worklist(Int64, grid, n_points, tile, n_tiles)
end

"""The tile pairs of `grid` in increasing packed order, each once: tile `t` lists its partners `t′ ≤ t`, which the
stencil's symmetry makes every pair's larger tile find."""
function _tile_worklist(::Type{I}, grid::SFC.CellGrid, n_points::Int, tile::Int, n_tiles::Int) where {I}
    length(grid.offsets) <= CULL_MAX_ROWS ||
        throw(ArgumentError("a cull grid with $(length(grid.offsets)) stencil rows; at most $CULL_MAX_ROWS"))
    backend = KA.get_backend(grid.cell_ids)
    rows = KA.adapt(backend, [(SFC.cull_row_id_shift(grid.dims, off), e1) for (off, e1) in grid.offsets])
    n_rows = length(rows)
    iv = KA.allocate(backend, I, 2, n_rows, n_tiles)
    _worklist_rows!(backend)(iv, grid.cell_ids, grid.run_starts, rows, prod(grid.dims), n_points, tile;
                             ndrange = (n_rows, n_tiles))
    counts, n_iv = KA.allocate(backend, Int, n_tiles), KA.allocate(backend, Int32, n_tiles)
    _worklist_merge!(backend)(counts, n_iv, iv, n_rows; ndrange = n_tiles)
    ends = cumsum(counts)
    packed = KA.allocate(backend, I, only(Array(view(ends, n_tiles:n_tiles))))
    _worklist_write!(backend)(packed, ends, n_iv, iv, n_tiles; ndrange = n_tiles)
    return SFC.TilePairWorkList(packed, Int32(n_tiles))
end
