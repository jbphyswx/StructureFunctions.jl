# How pair blocks are enumerated. One concept: a schedule lists the `(i-block, j-block)` pairs the
# kernel must sweep, outermost, so each block pair is worked to completion while it is cache-warm.

"""
    PairBlockSchedule

Enumeration of the `(i-block, j-block)` pairs covering the pairs a calculation needs. Blocks are
contiguous index ranges, and uniqueness is always `j > i`, so a block pair never needs to know
whether it is on the diagonal.
"""
abstract type PairBlockSchedule end

"""
    TiledUpperTriangle(n_points, tile)

Every pair, in blocks of `tile` points. The `j` block is the outer loop, so one block of `j`
coordinates and fields stays resident while every `i` sweeps it — which is what keeps the loop off
the memory bus when many cores run it at once.
"""
struct TiledUpperTriangle <: PairBlockSchedule
    n_points::Int
    tile::Int
end

"""
    CulledCellPairs(grid)

Only the block pairs that can hold a pair inside `grid.cutoff`: for each cell, one block per
stencil row. Rows extend on both sides of the cell, and uniqueness stays `j > i`, so each pair is
counted once — in the block whose `i` side holds the lower index.
"""
struct CulledCellPairs{G} <: PairBlockSchedule
    grid::G
end

const PairBlock = Tuple{UnitRange{Int}, UnitRange{Int}}

"""
    FullUpperTriangle(n_tiles)

Every device tile pair `(ti, tj)` with `ti ≤ tj`, enumerated by [`tile_for`](@ref) from a linear
block id. Passed into GPU kernels in place of the tile count: one integer, so the same
kernel-argument bytes, and the concrete type is what specializes the kernel.
"""
struct FullUpperTriangle{I <: Integer} <: PairBlockSchedule
    n_tiles::I
end

"""
    TilePairWorkList(pairs, n_tiles)

Only the tile pairs in `pairs`, each packed by [`pack_tile_pair`](@ref), in a host or device
vector. The vector type is a parameter so one type serves both sides of the transfer.
"""
struct TilePairWorkList{V <: AbstractVector} <: PairBlockSchedule
    pairs::V
    n_tiles::Int32
end

"""Pack `(ti, tj)` for a [`TilePairWorkList`](@ref) over `n_tiles` tiles; dimension `ti` fastest."""
@inline pack_tile_pair(ti::Integer, tj::Integer, n_tiles::Integer) = ti + (tj - 1) * n_tiles

"""
    tile_for(schedule, k) -> (ti, tj)

Tile pair for linear block id `k`, `ti ≤ tj`, in `O(1)`. The full triangle is enumerated row by
row, `(1,1), (1,2), …, (1,n), (2,2), …`; a work list is one indexed load.
"""
@inline function tile_for(s::FullUpperTriangle, k)
    n = Int(s.n_tiles)
    # Counted from the last block, the rows have lengths 1, 2, …, n, so the reversed id `kp` lies in
    # reversed row `m` with `m(m-1)/2 < kp <= m(m+1)/2`; the square root is corrected by one step.
    kp = n * (n + 1) ÷ 2 - Int(k) + 1
    m = unsafe_trunc(Int, ceil((sqrt(8.0 * kp + 1.0) - 1.0) / 2.0))
    m -= (m - 1) * m ÷ 2 >= kp
    m += m * (m + 1) ÷ 2 < kp
    ti = n - m + 1
    tj = n + 1 - (kp - (m - 1) * m ÷ 2)
    return oftype(k, ti), oftype(k, tj)
end

@inline function tile_for(s::TilePairWorkList, k)
    p = @inbounds s.pairs[k]
    n = oftype(p, s.n_tiles)
    tj = (p - one(p)) ÷ n + one(p)
    ti = p - (tj - one(p)) * n
    return ti, tj
end

n_pair_blocks(s::FullUpperTriangle) = Int(s.n_tiles) * (Int(s.n_tiles) + 1) ÷ 2
n_pair_blocks(s::TilePairWorkList) = length(s.pairs)

"""
    TiledBlockPairs(n_points, tile)

Lazy `(i-block, j-block)` iterator for the full upper triangle. The blocks are pure arithmetic, so
the uncalled sweep enumerates them without allocating; only a grid-dependent schedule needs a
materialized list.
"""
struct TiledBlockPairs
    n_points::Int
    tile::Int
end

Base.eltype(::Type{TiledBlockPairs}) = PairBlock
Base.length(b::TiledBlockPairs) =
    let t = cld(b.n_points, b.tile)
        t * (t + 1) ÷ 2
    end

@inline function Base.iterate(b::TiledBlockPairs, st::Tuple{Int, Int} = (1, 1))
    jt, it = st
    jt > cld(b.n_points, b.tile) && return nothing
    j0 = (jt - 1) * b.tile + 1
    i0 = (it - 1) * b.tile + 1
    block = (i0:min(i0 + b.tile - 1, b.n_points), j0:min(j0 + b.tile - 1, b.n_points))
    return block, (it < jt ? (jt, it + 1) : (jt + 1, 1))
end

"""
    block_pairs(schedule)

The schedule's `(i-block, j-block)` pairs: lazy for [`TiledUpperTriangle`](@ref), a materialized
work list for a culled schedule (which depends on the grid and doubles as the list backends slice
to divide work).
"""
function block_pairs(s::TiledUpperTriangle)
    s.tile >= 1 || throw(ArgumentError("tile must be >= 1 (got $(s.tile))"))
    return TiledBlockPairs(s.n_points, s.tile)
end

"""
    CulledBlockPairs(grid, cells = 1:n_occupied_cells(grid))

Lazy `(i-block, j-block)` iterator over the stencil rows of the occupied cells `cells` of a
[`CellGrid`](@ref). Each block is a pure function of `(cell, row)`, so the culled sweep enumerates without
allocating, exactly like the full one; `collect` it when a materialized work list is wanted.
"""
struct CulledBlockPairs{D, G}
    grid::G
    cells::UnitRange{Int}
end

CulledBlockPairs(grid::CellGrid{D}, cells::UnitRange{Int} = 1:n_occupied_cells(grid)) where {D} =
    CulledBlockPairs{D, typeof(grid)}(grid, cells)

Base.IteratorSize(::Type{<:CulledBlockPairs}) = Base.SizeUnknown()
Base.eltype(::Type{<:CulledBlockPairs}) = PairBlock

# Cells adjacent along dimension 1 are contiguous in the sorted order, so a whole stencil row is one
# run and the inner loop is `2*span+1` cells long.
# The outer walk is over OCCUPIED cells, so it never scales with the cell-id space.
@inline Base.iterate(b::CulledBlockPairs) = iterate(b, (first(b.cells), 1))

@inline function Base.iterate(b::CulledBlockPairs{D}, st::Tuple{Int, Int}) where {D}
    grid = b.grid
    dims = grid.dims
    kc_last = last(b.cells)
    nrow = length(grid.offsets)
    kc, k = st
    while kc <= kc_last
        if k > nrow
            kc += 1
            k = 1
            continue
        end
        ci, ir = occupied_cell(grid, kc)
        off, e1 = @inbounds grid.offsets[k]
        cmi = cull_multi_index(dims, ci)
        base = ntuple(d -> d == 1 ? 1 : cmi[d] + off[d - 1], Val(D))
        ok = true
        @inbounds for d in 2:D
            ok &= (1 <= base[d] <= dims[d])
        end
        if ok
            lo1 = max(1, cmi[1] - e1)
            hi1 = min(dims[1], cmi[1] + e1)
            c_lo = cull_linear_index(dims, ntuple(d -> d == 1 ? lo1 : base[d], Val(D)))
            c_hi = cull_linear_index(dims, ntuple(d -> d == 1 ? hi1 : base[d], Val(D)))
            jr = cell_id_span_run(grid, c_lo, c_hi)
            isempty(jr) || return ((ir, jr), (kc, k + 1))
        end
        k += 1
    end
    return nothing
end

block_pairs(s::CulledCellPairs) = CulledBlockPairs(s.grid)

"""
    n_pair_blocks(schedule)

Block-pair count: `O(1)` for [`TiledUpperTriangle`](@ref); a culled schedule has to walk its
stencil, so the count costs one pass over the cells.
"""
n_pair_blocks(s::TiledUpperTriangle) = length(block_pairs(s))
n_pair_blocks(s::CulledCellPairs) = count(_ -> true, block_pairs(s))

"""
    BlocksForI(blocks, irange)

`blocks` with each `i`-block narrowed to `irange`, dropping the ones that become empty.

This is what lets a backend keep partitioning by outer index while the kernel consumes block pairs:
the intersection of two ranges is a range, so a contiguous chunk and a strided rank share both stay
allocation-free.
"""
struct BlocksForI{B, R}
    blocks::B
    irange::R
end

Base.IteratorSize(::Type{<:BlocksForI}) = Base.SizeUnknown()
Base.eltype(::Type{<:BlocksForI}) = Tuple{Any, UnitRange{Int}}

@inline function _blocks_for_i_advance(b::BlocksForI, r)
    while r !== nothing
        (ir, jr), st = r
        ii = intersect(ir, b.irange)
        isempty(ii) || return ((ii, jr), st)
        r = iterate(b.blocks, st)
    end
    return nothing
end

Base.iterate(b::BlocksForI) = _blocks_for_i_advance(b, iterate(b.blocks))
Base.iterate(b::BlocksForI, st) = _blocks_for_i_advance(b, iterate(b.blocks, st))

"""
    TileRuns(blocks, tile)

`blocks` with each `j`-block cut into runs of at most `tile` points.
"""
struct TileRuns{B}
    blocks::B
    tile::Int
end

Base.IteratorSize(::Type{<:TileRuns}) = Base.SizeUnknown()
Base.eltype(::Type{<:TileRuns}) = PairBlock

@inline function Base.iterate(t::TileRuns)
    r = iterate(t.blocks)
    r === nothing && return nothing
    return _next_tile_run(t, r[1], r[2], 0)
end

@inline function Base.iterate(t::TileRuns, (blk, inner, off))
    if off == 0
        r = iterate(t.blocks, inner)
        r === nothing && return nothing
        blk, inner = r
    end
    return _next_tile_run(t, blk, inner, off)
end

"""The run of block `(ir, jr)` starting `off` past `first(jr)`, and the state after it: the block, the
inner state and the offset of the next run, `0` once the block is spent."""
@inline function _next_tile_run(t::TileRuns, (ir, jr), inner, off::Int)
    lo = first(jr) + off
    hi = min(last(jr), lo + t.tile - 1)
    return (ir, lo:hi), ((ir, jr), inner, hi == last(jr) ? 0 : off + t.tile)
end

"""
    pair_blocks(n_points, irange; grid = nothing, tile = SF_CPU_PAIR_TILE)

Block pairs for the `i` values in `irange`: the culled schedule when a [`CellGrid`](@ref) is given,
the full tiled upper triangle otherwise. Every `j`-block holds at most `tile` points.
"""
@inline pair_blocks(n_points::Int, irange; grid::Union{Nothing, CellGrid} = nothing, tile::Int = SF_CPU_PAIR_TILE) =
    BlocksForI(_schedule_blocks(grid, n_points, tile, irange), irange)

@inline _schedule_blocks(::Nothing, n_points::Int, tile::Int, irange) = block_pairs(TiledUpperTriangle(n_points, tile))
@inline _schedule_blocks(grid::CellGrid, n_points::Int, tile::Int, irange) =
    TileRuns(CulledBlockPairs(grid, _cells_holding(grid, irange)), tile)

"""The occupied cells of `grid` from the one holding the lowest point index of `irange` to the one holding its
highest."""
@inline function _cells_holding(grid::CellGrid, irange)
    isempty(irange) && return 1:0
    lo, hi = extrema(irange)
    return searchsortedlast(grid.run_starts, lo):searchsortedlast(grid.run_starts, hi)
end

"""
Points per `j` block in the CPU pair loop. Sized so one block's coordinates, fields and the three
per-`j` buffers stay resident in a core's private cache while every `i` sweeps it.
"""
const SF_CPU_PAIR_TILE = 65536

"""Points up to which a per-task pair buffer spans the whole sweep; above it a buffer spans one tile."""
const SF_CPU_WHOLE_RUN_MAX = 4 * SF_CPU_PAIR_TILE

"""How a per-task pair buffer is indexed: [`WholeRun`](@ref) or [`BlockRun`](@ref)."""
abstract type PairWindow end

"""A pair buffer spanning every point of the sweep; pair `(i, j)` fills slot `j`."""
struct WholeRun <: PairWindow end

"""A pair buffer spanning one tile; pair `(i, j)` of block `jr` fills slot `j - first(jr) + 1`."""
struct BlockRun <: PairWindow end

"""The window of an `n`-point sweep's pair buffers."""
@inline _pair_window(n::Integer) = n <= SF_CPU_WHOLE_RUN_MAX ? WholeRun() : BlockRun()

"""Length of a per-task pair buffer for `n` points under its window."""
@inline _pair_scratch_length(n::Integer) = _pair_scratch_length(_pair_window(n), n)
@inline _pair_scratch_length(::WholeRun, n::Integer) = max(1, Int(n))
@inline _pair_scratch_length(::BlockRun, ::Integer) = SF_CPU_PAIR_TILE

"""The buffer slot of `j` in block `jr` is `j - _slot_offset(window, jr)`."""
@inline _slot_offset(::WholeRun, jr) = 0
@inline _slot_offset(::BlockRun, jr) = first(jr) - 1

"""Throw unless a per-`j` buffer holds the slots of block `jr` under `window`."""
@inline function _check_run_fits(window::PairWindow, buf, jr)
    last_slot = last(jr) - _slot_offset(window, jr)
    return last_slot <= length(buf) || _run_exceeds_scratch(length(buf), last_slot)
end
@noinline _run_exceeds_scratch(L, n) =
    throw(ArgumentError("a pair block reaching slot $n exceeds the $L-entry pair buffer"))

"""
    _compact_in_range!(sel, plan, keybuf, ks) -> Int

Gather into `sel`, in increasing order, the slots `k ∈ ks` whose key the squared `plan` puts in a bin, and
return how many there are.
"""
@inline function _compact_in_range!(sel, plan, keybuf, ks)
    n = 0
    @inbounds for k in ks
        sel[n + 1] = k % Int32
        n += squared_in_range(plan, keybuf[k])
    end
    return n
end

"""Whether a pair kernel on schedule `blocks` chooses run by run between the range branch and the list
[`_compact_in_range!`](@ref) builds: on the tiled schedule; a culled schedule, whose points are sorted into
cells, always takes the branch."""
@inline _chooses_compaction(::BlocksForI{TiledBlockPairs}) = true
@inline _chooses_compaction(_) = false

"""
    _sample_in_range(plan, keybuf, ks) -> (n_in, n)

How many of every 16th slot of the run `ks`, `n` of them, have a key in range of the squared `plan`.
"""
@inline function _sample_in_range(plan, keybuf, ks)
    n_in, n = 0, 0
    @inbounds for k in first(ks):16:last(ks)
        n_in += squared_in_range(plan, keybuf[k])
        n += 1
    end
    return n_in, n
end

"""Whether a run is scattered through the list [`_compact_in_range!`](@ref) builds: its sample has `n_in` of `n`
slots in range, at most 7/8 of them."""
@inline _compacts(n_in::Int, n::Int) = 8 * n_in <= 7 * n

"""Whether a sample with `n_in` of `n` slots in range has at most 1/8 of them in range."""
@inline _sparse(n_in::Int, n::Int) = 8 * n_in <= n
