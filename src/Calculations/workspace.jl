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
    CPUSFWorkspace{kind}(x, u, distance_bins[, value_bins]; backend = SerialBackend(), count_eltype = UInt32)

Reusable CPU scratch for the batch drivers: the batch-leading working copies of `x`/`u`, one
accumulator per task, and the full-width reduction accumulator. Pass to a batch entry point as
`workspace = ...` to make repeated calls allocation-free.

Without it a batch call allocates one accumulator per task — tens of MiB at large `B` — on every
call. That does not slow the fastest call, but the resulting GC pauses land on *some* calls, so the
typical call is far slower than the best one and the spread is wide. Reusing the buffers removes
the pauses, not the peak.

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

function CPUSFWorkspace{kind}(
    x::BatchInput,
    u::BatchInput,
    distance_bins::AbstractVector,
    value_bins = nothing;
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    count_eltype::Type{CT} = UInt32,
) where {kind, CT}
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

"""Throw unless `ws` holds accumulators of this kind and shape. Checked before they are used."""
@inline _validate_ws_layout(::Nothing, ::Symbol, ::Tuple) = nothing

function _validate_ws_layout(ws::CPUSFWorkspace{kind}, want_kind::Symbol, tail::Tuple) where {kind}
    kind === want_kind ||
        throw(ArgumentError("CPUSFWorkspace kind :$kind incompatible with requested :$want_kind"))
    _ws_tail(ws) == tail || throw(ArgumentError(
        "CPUSFWorkspace built for accumulator axes $(_ws_tail(ws)); called with $tail"))
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

# --- Buffer providers for the batch drivers (the `::Nothing` fallbacks live in batch_leading.jl,
# which loads before this type exists) ---

@inline _ws_ub(ws::CPUSFWorkspace) = ws.ub
@inline _ws_xb(ws::CPUSFWorkspace) = ws.xb

function _bl_accum_pool(ws::CPUSFWorkspace, ::F, widths::Vector{Int}) where {F}
    ws.widths == widths || throw(ArgumentError(
        "CPUSFWorkspace holds accumulators of batch widths $(ws.widths); this call needs $widths. \
         Rebuild it with the same inputs, backend and n_tasks as the call."))
    return ws.accs
end

function _bl_result_accum(ws::CPUSFWorkspace, ::F, ::Int) where {F}
    _bl_zero_accum!(ws.result)
    return ws.result
end
