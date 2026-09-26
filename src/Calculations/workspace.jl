# CPUSFWorkspace — reusable host scratch for the batch (auxiliary-axis) drivers.

"""
    _bl_partition(B, n_tasks, accum_bytes) -> (bchunks, n_ichunks)

The batch-axis chunks and the number of outer-index chunks each is split into. Shared by the
threaded executor and by [`CPUSFWorkspace`](@ref) so a workspace cannot be sized for a partition
the executor will not use.
"""
@inline function _bl_partition(B::Int, n_tasks::Int, accum_bytes::Int)
    n_bchunks = _bl_batch_chunk_count(accum_bytes, B, n_tasks)
    bchunks = _bl_batch_chunks(B, n_bchunks)
    return bchunks, max(1, n_tasks ÷ n_bchunks)
end

"""Accumulator axes after the leading batch axis, per workspace kind."""
@inline _bl_accum_tail(::Val{:sf1d}, n_bins::Int, ::Int) = (n_bins,)
@inline _bl_accum_tail(::Val{:joint2d}, n_bins::Int, n_val::Int) = (n_bins, n_val)
@inline _bl_accum_tail(::Val{:single_pass}, n_bins::Int, ::Int) = (SINGLE_PASS_N, n_bins)
@inline _bl_accum_tail(::Val{:single_pass_2d}, n_bins::Int, n_val::Int) =
    (SINGLE_PASS_N, n_bins, n_val)

"""
    CPUSFWorkspace{kind}(x, u, distance_bins[, value_bins][, CT]; backend = SerialBackend())

Reusable CPU scratch for the batch drivers: the batch-leading working copies of `x`/`u`, one
accumulator per task, and the full-width reduction accumulator. Pass to a batch entry point as
`workspace = ...` to make repeated calls allocation-free. `CT` is the count element type of the
calls it serves (default `$(DEFAULT_COUNT_TYPE)`).

Without it a batch call allocates one accumulator per task, tens of MiB at large `B`, on every
call; the GC pauses that follow land on some calls and widen the spread of call times.

`kind` matches [`GPUSFWorkspace`](@ref): `:sf1d`, `:joint2d`, `:single_pass` or `:single_pass_2d`.
It is a type parameter, so the accumulator rank is fixed at construction and every field is
concretely typed.

Built from the same arguments as the call it serves — including `backend`, which fixes how many
accumulators it holds — and checked against them on use: a workspace built for a different shape,
element type or backend is a hard error, never a silent reallocation. Composes with [`BatchLeading`](@ref) — when the input is already batch-leading the
transpose buffers have length zero and are never touched.

Not thread-safe: one workspace serves one call at a time, exactly like [`GPUSFWorkspace`](@ref).
"""
struct CPUSFWorkspace{kind, OT, CT, FTx, FTu, NA}
    xb::Array{FTx, 3}
    ub::Array{FTu, 3}
    accs::Vector{Tuple{Array{OT, NA}, Array{CT, NA}}}
    result::Tuple{Array{OT, NA}, Array{CT, NA}}
    widths::Vector{Int}
    B::Int
    N::Int
    D::Int
    W::Int
    n_tasks::Int
end

CPUSFWorkspace{kind}(x::BatchInput, u::BatchInput, distance_bins::AbstractVector; kwargs...) where {kind} =
    CPUSFWorkspace{kind}(x, u, distance_bins, nothing, DEFAULT_COUNT_TYPE; kwargs...)
CPUSFWorkspace{kind}(x::BatchInput, u::BatchInput, distance_bins::AbstractVector, ::Type{CT};
                     kwargs...) where {kind, CT <: Real} =
    CPUSFWorkspace{kind}(x, u, distance_bins, nothing, CT; kwargs...)
CPUSFWorkspace{kind}(x::BatchInput, u::BatchInput, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins;
                     kwargs...) where {kind} =
    CPUSFWorkspace{kind}(x, u, distance_bins, value_bins, DEFAULT_COUNT_TYPE; kwargs...)

function CPUSFWorkspace{kind}(
    x::BatchInput,
    u::BatchInput,
    distance_bins::AbstractVector,
    value_bins::Union{Nothing, SinglePass2DValueBins},
    ::Type{CT};
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {kind, CT <: Real}
    kind in (:sf1d, :joint2d, :single_pass, :single_pass_2d) || throw(ArgumentError(
        "CPUSFWorkspace kind must be :sf1d, :joint2d, :single_pass or :single_pass_2d; got :$kind"))
    x_raw, x_bl = _bl_unwrap(x)
    u_raw, u_bl = _bl_unwrap(u)
    fixed_x = ndims(x_raw) == 2
    FTx, FTu = eltype(x_raw), eltype(u_raw)
    OT = promote_type(float(FTx), float(FTu))

    # Coordinate width and velocity width are independent: on a shell `x` is (2, N) while `u` may
    # be (3, N), so the two transpose buffers are sized separately.
    W = x_bl ? size(x_raw, 2) : size(x_raw, 1)
    if u_bl
        B, D, N = size(u_raw)
    else
        D, N = size(u_raw, 1), size(u_raw, 2)
        B = prod(size(u_raw)[3:end])
    end
    n_bins = n_histogram_bins(distance_bins)
    n_val = value_bins === nothing ? 0 :
            n_histogram_bins(value_bins isa Tuple ? value_bins[1] : value_bins)
    tail = _bl_accum_tail(Val(kind), n_bins, n_val)

    # Length zero when the input is already batch-leading: there is nothing to transpose into.
    xb = Array{FTx, 3}(undef, (fixed_x || x_bl) ? (0, 0, 0) : (B, W, N))
    ub = Array{FTu, 3}(undef, u_bl ? (0, 0, 0) : (B, D, N))

    n_tasks = _bl_n_tasks(backend)
    accum_bytes = B * prod(tail) * (sizeof(OT) + sizeof(CT))
    bchunks, n_ichunks = _bl_partition(B, n_tasks, accum_bytes)
    widths = [length(bc) for bc in bchunks for _ in 1:n_ichunks]
    accs = [(zeros(OT, w, tail...), zeros(CT, w, tail...)) for w in widths]
    result = (zeros(OT, B, tail...), zeros(CT, B, tail...))

    return CPUSFWorkspace{kind, OT, CT, FTx, FTu, length(tail) + 1}(
        xb, ub, accs, result, widths, B, N, D, W, n_tasks,
    )
end

"""Accumulator axes after the batch axis, as the workspace was built for."""
@inline _ws_tail(ws::CPUSFWorkspace) = size(ws.result[1])[2:end]

# The kernels write through `@inbounds`, so a workspace built for other inputs must be rejected
# before it is used, never left to corrupt memory. The two checks run where their information first
# exists: the input shape inside `_bl_prepare`, the accumulator layout in the driver.

"""Throw unless `ws` was built for this input shape. Checked before the transpose buffers are used."""
@inline _validate_ws_shape(::Nothing, ::Int, ::Int, ::Int, ::Int) = nothing

function _validate_ws_shape(ws::CPUSFWorkspace, B::Int, N::Int, D::Int, W::Int)
    (ws.B, ws.N, ws.D, ws.W) == (B, N, D, W) || throw(ArgumentError(
        "CPUSFWorkspace built for (B, N, D, W) = $((ws.B, ws.N, ws.D, ws.W)); called with $((B, N, D, W))"))
    return nothing
end

"""Throw unless `ws` holds accumulators of this kind, shape and element types. Checked before they are used."""
@inline _validate_ws_layout(::Nothing, ::Symbol, ::Tuple, ::Type, ::Type) = nothing

function _validate_ws_layout(ws::CPUSFWorkspace{kind, WOT, WCT}, want_kind::Symbol, tail::Tuple,
                             ::Type{OT}, ::Type{CT}) where {kind, WOT, WCT, OT, CT}
    kind === want_kind ||
        throw(ArgumentError("CPUSFWorkspace kind :$kind incompatible with requested :$want_kind"))
    _ws_tail(ws) == tail || throw(ArgumentError(
        "CPUSFWorkspace built for accumulator axes $(_ws_tail(ws)); called with $tail"))
    (WOT, WCT) == (OT, CT) || throw(ArgumentError(
        "CPUSFWorkspace built for sums of $WOT and counts of $WCT; called with $OT and $CT"))
    return nothing
end

"""Zero every accumulator held by a [`CPUSFWorkspace`](@ref); the drivers do this per call."""
function reset_histogram!(ws::CPUSFWorkspace)
    for (s, c) in ws.accs
        fill!(s, zero(eltype(s)))
        fill!(c, zero(eltype(c)))
    end
    fill!(ws.result[1], zero(eltype(ws.result[1])))
    fill!(ws.result[2], zero(eltype(ws.result[2])))
    return ws
end

# --- Buffer providers for the batch drivers ---

@inline _ws_ub(ws::CPUSFWorkspace) = ws.ub
@inline _ws_xb(ws::CPUSFWorkspace) = ws.xb

function _bl_accum_pool(ws::CPUSFWorkspace, ::F, widths::AbstractVector{Int}) where {F}
    ws.widths == widths || throw(ArgumentError(
        "CPUSFWorkspace holds accumulators of batch widths $(ws.widths); this call needs $widths. \
         Rebuild it with the same inputs, backend and n_tasks as the call."))
    return ws.accs
end

function _bl_result_accum(ws::CPUSFWorkspace, ::F, ::Int) where {F}
    _bl_zero_accum!(ws.result)
    return ws.result
end

# GPU device-resident workspace. The type carries no device dependency — every buffer is a
# type parameter — so it lives here beside CPUSFWorkspace; the constructors are added by
# StructureFunctionsKernelAbstractionsExt.

"""
    GPUSFWorkspace(backend, distance_bins; kind=:sf1d)
    GPUSFWorkspace(backend, distance_bins, value_bins; kind=:joint2d)

Reusable device bin preparation: the digitizers of the edges, the staged inputs, the cull grid and
its tile-pair schedules, and the single-pass 2D partitions. Load `KernelAbstractions`, then construct a
workspace for the execution backend and edges. Pass it as `workspace=ws` to a compatible calculation.
Each workspace serves one call at a time.

`kind` selects the calculation the workspace serves: `:sf1d`, `:joint2d`, `:single_pass` or
`:single_pass_2d`. Backend, kind and bin definitions are checked before execution; construct another
workspace when these change. Field values may change between calls. Coordinates are preparation data:
call `refresh!(ws)` after mutating them in place. Passing a different coordinate array is detected by
identity and prepares a new schedule. An allocating calculation returns buffers of its own; a mutating
one adds into the caller's outputs.

`release!(ws)` drops the input-staging, culling and partition buffers.
"""
struct GPUSFWorkspace{kind, FT, BE, DB, VB, DD, VP, L}
    backend::BE
    dist_bins::DB
    val_bins::VB
    dist_digitizer::DD
    val_plan::VP
    NB::Int
    n_val::Int
    joint2d_compile_cells::Int
    lazy::L
end

"""
    GPUCullMemo

What one cull prologue produced for a set of kernel coordinates, kept on the workspace so a call on
the same coordinates, cutoff and policy reuses it: the cell grid (which owns the permutation), the
coordinates already permuted, and one device work list per tile size, built on first use by
[`schedule_for`](@ref). `source` identifies the caller's prepared coordinate array; an in-place
mutation therefore requires [`refresh!`](@ref). `x` is the workspace-owned coordinate snapshot and
`to_device` uploads a host vector to the workspace's device.
"""
abstract type AbstractGPUCullMemo end

struct GPUCullMemo{X <: AbstractMatrix, FT, PO <: CullingPolicy, G <: CellGrid, XS, TD} <:
       AbstractGPUCullMemo
    source::Any
    x::X
    cutoff::FT
    policy::PO
    grid::G
    x_sorted::XS
    to_device::TD
    schedules::Dict{Int, TilePairWorkList}
end


"""Cached decision that culling cannot remove work for this prepared input."""
struct GPUNoCullMemo{FT, PO <: CullingPolicy} <: AbstractGPUCullMemo
    source::Any
    cutoff::FT
    policy::PO
end

"""Whether `memo` was built from these coordinates under this cutoff and policy."""
@inline _cull_memo_hit(::Nothing, source, cutoff, policy) = false
@inline _cull_memo_hit(m::AbstractGPUCullMemo, source, cutoff, policy) =
    m.source === source && m.policy === policy && m.cutoff == cutoff

"""
    schedule_for(cull, n_points, tile) -> PairBlockSchedule

The tile-pair schedule a kernel with `tile`-point tiles enumerates: the full upper triangle when
`cull` is `nothing`, otherwise the memo's device work list for that tile size, built and uploaded on
first use and kept for later calls. Each kernel family picks its own tile, so the list is derived
from the grid at the size asked for, not fixed when the memo is built.
"""
schedule_for(::Nothing, n_points::Int, tile::Int) = FullUpperTriangle(cld(n_points, tile))

function schedule_for(memo::GPUCullMemo, n_points::Int, tile::Int)
    n_points == length(memo.grid.perm) || throw(ArgumentError(
        "cull memo holds $(length(memo.grid.perm)) points, asked to schedule $n_points"))
    return get!(memo.schedules, tile) do
        wl = tile_pair_worklist(memo.grid, n_points, tile)
        TilePairWorkList(memo.to_device(wl.pairs), wl.n_tiles)
    end
end

"""State a workspace carries between launches: the single-pass 2D partitions, the staged inputs, the
cull memo, and `active`, the memo this call culls with (set by the prologue on every call)."""
mutable struct GPUSFLazyBuffers
    partition_sums_dev
    partition_counts_dev
    x_dev_cache
    u_dev_cache
    active::Union{Nothing, GPUCullMemo}
    cull::Union{Nothing, AbstractGPUCullMemo}
end

GPUSFLazyBuffers() = GPUSFLazyBuffers(nothing, nothing, nothing, nothing, nothing, nothing)
