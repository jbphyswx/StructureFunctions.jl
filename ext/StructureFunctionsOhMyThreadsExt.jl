module StructureFunctionsOhMyThreadsExt

using Distances: Distances as DI
using OhMyThreads: OhMyThreads as OMT
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions:
    StructureFunctions as SF,
    Calculations as SFC,
    StructureFunctionObjects as SFO,
    StructureFunctionTypes as SFT,
    HelperFunctions as SFH,
    digitize_plan,
    n_histogram_bins

function __init__()
    SFC._OHMYTHREADS_LOADED[] = true
    return nothing
end

"""Outer-index chunks per thread that a threaded sweep deals out on demand."""
const CHUNKS_PER_THREAD = 64

"""The number of chunks `n_tasks` tasks of a threaded sweep split `indices` into."""
@inline _n_chunks(indices, n_tasks::Int = Threads.nthreads()) =
    clamp(CHUNKS_PER_THREAD * n_tasks, 1, max(1, length(indices)))

"""
    _triangle_outer_chunks(indices, n_tasks = Threads.nthreads())

Round-robin chunks of the outer indices of an O(N²) pair kernel, whose work for index `i` is `N - i`: every
chunk takes indices from the whole range, so chunks carry equal work.
"""
@inline _triangle_outer_chunks(indices, n_tasks::Int = Threads.nthreads()) =
    collect(OMT.chunks(indices; n = _n_chunks(indices, n_tasks), split = OMT.RoundRobin()))

"""
    _outer_chunks(grid, indices, n_tasks = Threads.nthreads())

Outer-index chunks suited to the schedule in play: round-robin without a cull grid; consecutive with one, or
with one per slice, whose consecutive indices share cells, so a chunk sweeps only its own cells' stencils.
"""
@inline _outer_chunks(::Nothing, indices, n_tasks::Int = Threads.nthreads()) = _triangle_outer_chunks(indices, n_tasks)
@inline _outer_chunks(::Union{SFC.CellGrid, AbstractVector}, indices, n_tasks::Int = Threads.nthreads()) =
    collect(OMT.chunks(indices; n = _n_chunks(indices, n_tasks), split = OMT.Consecutive()))

"""
    _greedy_reduce(op, init, run!, chunks)

Run `run!(acc, scratch, chunk)` over `chunks` on one task per thread. Each task makes its `(acc, scratch) =
init()` once and takes the next chunk from a shared counter until none is left; the tasks' accumulators are
combined with `op`.
"""
function _greedy_reduce(op, init, run!, chunks)
    next = Threads.Atomic{Int}(1)
    n = length(chunks)
    return OMT.tmapreduce(op, 1:clamp(n, 1, Threads.nthreads())) do _
        acc, scratch = init()
        k = Threads.atomic_add!(next, 1)
        while k <= n
            run!(acc, scratch, chunks[k])
            k = Threads.atomic_add!(next, 1)
        end
        acc
    end
end

@inline _hist_add(a, b) = (a[1] .+= b[1]; a[2] .+= b[2]; a)

# --- Multi-field ---

"""
    threaded_calculate_structure_function!(sums, counts, sf, x, fields, distance_bins; kwargs...)

Multi-field pair sweep across threads.

The setup — widening the fields, sorting into the cull grid — happens **once**, above the task
loop; each task then sweeps its own share of outer indices into private histograms, which are
reduced. A field of one vector field forwards to the array path.
"""
function SFC.threaded_calculate_structure_function!(
    sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::SFC.MF.Fields{D, 1, 0}, distance_bins; kwargs...,
) where {D}
    return SFC.threaded_calculate_structure_function!(sums, counts, sf, x, SFC.MF.packed(f),
                                                      distance_bins; kwargs...)
end

function SFC.threaded_calculate_structure_function!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::SFC.MF.Fields{D, V, K}, distance_bins;
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT, D, V, K}
    Np = size(SFC.MF.packed(f), 2)
    size(x, 2) == Np || throw(DimensionMismatch(
        "x covers $(size(x, 2)) points and the field $Np",
    ))
    if SFC._on_a_line(geometry, sf)
        return SFC.sorted_line_sweep!(sums, counts, sf, SFC._line_coordinates(x), SFC.MF.packed(f),
                                      distance_bins, Val(D), Val(V), Val(K); weights,
                                      backend = CB.ThreadedBackend())
    end
    _threaded_field_pairs!(sums, counts, sf, x, f, distance_bins, nothing; geometry, culling, weights)
end

SFC._field_into!(::CB.AbstractThreadedBackend, sums, counts, sf, x, f::SFC.MF.Fields, distance_bins, share; kwargs...) =
    _threaded_field_pairs!(sums, counts, sf, x, f, distance_bins, share; kwargs...)

"""The multi-field pairs whose lower index is in the outer indices `share` selects (`SFC._share_indices`) added into
`sums`/`counts` across threads."""
function _threaded_field_pairs!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT}, sf, x, f::SFC.MF.Fields{D, V, K}, distance_bins, share;
    geometry, culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT, D, V, K}
    Np = size(SFC.MF.packed(f), 2)
    # Bound once and never reassigned: the tasks close over these, and reassigning a captured
    # variable boxes it, which OhMyThreads rejects outright.
    geom, xk, data, vF, plan, grid, wk = SFC.field_setup(f, x, distance_bins, geometry, culling, weights)
    nb = n_histogram_bins(distance_bins)
    vW = SFH.coordinate_width(geom)
    ls, lc = _greedy_reduce(_hist_add, () -> ((zeros(OT, nb), zeros(CT, nb)), _field_scratch(geom, xk, OT)),
                            (a, scratch, chunk) -> _field_chunk!(a, scratch, sf, xk, data, geom, vF, Val(V), Val(K),
                                                                 plan, nb, vW, chunk, Np, grid, wk),
                            _outer_chunks(grid, SFC._share_indices(grid, Np - 1, share)))
    sums .+= ls
    counts .+= lc
    return nothing
end

"""A task's scratch for the flat multi-field kernel — its pair window and buffers — or `nothing` for a geometry
whose kernel takes none."""
function _field_scratch(::SFH.FlatGeometry, xk, ::Type{OT}) where {OT}
    window = SFC._pair_window(size(xk, 2))
    L = SFC._pair_scratch_length(window, size(xk, 2))
    return (window, Vector{eltype(xk)}(undef, L), Vector{OT}(undef, L), Vector{Int32}(undef, L), Vector{Int32}(undef, L))
end
_field_scratch(geom, xk, ::Type) = nothing

"""Accumulate the multi-field pairs of outer indices `chunk` into `acc`, with the task's `scratch`."""
_field_chunk!(acc, ::Nothing, sf, xk, data, geom, vF, vV, vK, plan, nb, vW, chunk, N, grid, wk) =
    SFC._field_run_blocks!(acc[1], acc[2], sf, xk, data, geom, vF, vV, vK, plan, nb, vW, chunk, N, grid, wk)
_field_chunk!(acc, (window, keybuf, valbuf, idxbuf, sel)::Tuple, sf, xk, data, geom, vF, vV, vK, plan, nb, vW, chunk,
              N, grid, wk) =
    SFC._field_pairs!(acc[1], acc[2], sf, xk, data, geom, vF, vV, vK, plan, nb, vW, SFC.pair_blocks(N, chunk; grid), wk,
                      window, keybuf, valbuf, idxbuf, sel)


# --- 1D Array thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector;
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT, FT1 <: Number, FT2 <: Number}
    if SFC._on_a_line(geometry, structure_function_type)
        return SFC.sorted_line_sweep!(output_sums, output_counts, structure_function_type,
                                      SFC._line_coordinates(x_arr), u_arr, distance_bins, Val(1), Val(1), Val(0);
                                      weights, backend = CB.ThreadedBackend())
    end
    x_vecs, u_vecs = SFC._prepared_tuples(geometry, x_arr, u_arr)
    vD = SFC._simd_width(geometry)
    vD === nothing || return _threaded_pf_simd!(output_sums, output_counts, structure_function_type, x_vecs, u_vecs,
                                                distance_bins, vD, nothing; geometry, culling, weights)
    result = _threaded_scalar_1d(structure_function_type, geometry, x_vecs, u_vecs, distance_bins, nothing, OT, CT,
                                 culling, weights)
    output_sums .+= result.sums
    output_counts .+= result.counts
    return nothing
end

# The scalar kernel over the outer indices `share` selects (`SFC._share_indices`), threaded; inputs sorted into the
# cull grid once and shared read-only by the tasks.
function _threaded_scalar_1d(sf, geom, x_vecs, u_vecs, distance_bins, share, ::Type{OT}, ::Type{CT},
                             culling, weights) where {OT, CT}
    return _threaded_scalar_1d_run(sf, geom, distance_bins, share, OT, CT,
                                   SFC._cull_sorted(x_vecs, u_vecs, weights, geom, distance_bins, culling))
end

function _threaded_scalar_1d_run(sf, geom, distance_bins, share, ::Type{OT}, ::Type{CT},
                                 (grid, xc, uc, wc)) where {OT, CT}
    be = digitize_plan(distance_bins)
    nb = n_histogram_bins(distance_bins)
    N = length(xc[1])
    ls, lc = _greedy_reduce(_hist_add, () -> ((zeros(OT, nb), zeros(CT, nb)), nothing),
                            (a, _, chunk) -> SFC._pf_scalar_pairs!(a[1], a[2], geom, sf, xc, uc, be,
                                                                   SFC.pair_blocks(N, chunk; grid), wc),
                            _outer_chunks(grid, SFC._share_indices(grid, N - 1, share)))
    return SFO.StructureFunctionSumsAndCounts(sf, distance_bins, ls, lc)
end

# Threaded point-field SIMD compute/scatter split over the outer indices `share` selects (every one when `nothing`):
# contiguous component vectors materialized once (shared, read-only), per-task histogram + buffers, i-chunks from
# `_outer_chunks`.
function _threaded_pf_simd!(
    output_sums::AbstractVector{OT}, output_counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType, x_vecs::Tuple, u_vecs::Tuple, dist_be, ::Val{D}, share;
    geometry = SFH.FlatGeometry{D}(), culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT, D}
    x_raw = ntuple(d -> collect(x_vecs[d]), Val(D))
    u_raw = ntuple(d -> collect(u_vecs[d]), Val(D))
    Np = length(x_raw[1])
    nb = n_histogram_bins(dist_be)
    FTx = eltype(x_raw[1])
    plan = SF.squared_digitize_plan(dist_be)     # built once; read-only, shared across tasks
    # Sorted once and shared read-only, so every task sweeps the cells in the same order. `xc`/`uc`/`wc`
    # are bound once and never reassigned: the tasks close over them, and reassigning a captured
    # variable boxes it, which OhMyThreads rejects outright.
    grid = (culling isa SFC.NoCulling) ? nothing : SFC.cull_grid_for(x_raw, geometry, dist_be, culling)
    xc, uc = isnothing(grid) ? (x_raw, u_raw) :
             (SFC.apply_perm(x_raw, grid.perm), SFC.apply_perm(u_raw, grid.perm))
    wc = isnothing(grid) ? weights : SFC._permuted_point_weights(weights, grid.perm)
    L = SFC._pair_scratch_length(Np)
    ls, lc = _greedy_reduce(_hist_add,
        () -> ((zeros(OT, nb), zeros(CT, nb)),
               (Vector{FTx}(undef, L), Vector{OT}(undef, L), Vector{Int32}(undef, L), Vector{Int32}(undef, L))),
        (a, (r2buf, valbuf, idxbuf, sel), chunk) -> SFC._pf_run_blocks!(a[1], a[2], sf, xc, uc, plan, Val(D),
            r2buf, valbuf, idxbuf, sel, chunk, Np, grid, wc),
        _outer_chunks(grid, SFC._share_indices(grid, Np - 1, share)))
    output_sums .+= ls
    output_counts .+= lc
    return nothing
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    ::Type{CT};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    output = zeros(OT, N3)
    counts = zeros(CT, N3)

    SFC.threaded_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins;
        kwargs...,
    )

    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type,
        distance_bins,
        output,
        counts,
    )
end

# --- 2D Tuple thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    geometry = SFC.default_geometry(u_vecs),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT}
    val_be = digitize_plan(value_bins)

    vD = SFC._simd_width(geometry)
    if vD !== nothing
        _threaded_2d_simd!(sums_2d, counts_2d, structure_function_type, x_vecs, u_vecs, distance_bins, val_be, vD,
                           culling, weights, second_axis)
        return nothing
    end

    SFC._require_value_axis(second_axis, geometry)
    result = _threaded_scalar_2d(structure_function_type, geometry, x_vecs, u_vecs, distance_bins, value_bins,
                                 val_be, OT, CT, culling, weights, second_axis)
    sums_2d .+= result.sums
    counts_2d .+= result.counts
    return nothing
end

# The joint scalar kernel over the outer indices `share` selects (every one when `nothing`), threaded; inputs sorted
# into the cull grid once and shared read-only.
function _threaded_scalar_2d(sf, geom, x_vecs, u_vecs, distance_bins, value_bins, val_be, ::Type{OT},
                             ::Type{CT}, culling, weights, second_axis, share = nothing) where {OT, CT}
    return _threaded_scalar_2d_run(sf, geom, distance_bins, value_bins, val_be, OT, CT, second_axis, share,
                                   SFC._cull_sorted(x_vecs, u_vecs, weights, geom, distance_bins, culling))
end

function _threaded_scalar_2d_run(sf, geom, distance_bins, value_bins, val_be, ::Type{OT}, ::Type{CT},
                                 second_axis, share, (grid, xc, uc, wc)) where {OT, CT}
    dist_be = digitize_plan(distance_bins)
    nd, nv = n_histogram_bins(distance_bins), n_histogram_bins(value_bins)
    N = length(xc[1])
    ls, lc = _greedy_reduce(_hist_add, () -> ((zeros(OT, nd, nv), zeros(CT, nd, nv)), nothing),
                            (a, _, chunk) -> SFC._pf_2d_scalar_pairs!(a[1], a[2], geom, sf, xc, uc, dist_be, val_be,
                                                                      SFC.pair_blocks(N, chunk; grid), wc, second_axis),
                            _outer_chunks(grid, SFC._share_indices(grid, N - 1, share)))
    return SFO.StructureFunction2DSumsAndCounts(sf, distance_bins, value_bins, ls, lc, second_axis)
end

# Threaded 2D-joint point-field SIMD over the outer indices `share` selects (every one when `nothing`): contiguous
# components shared, per-task buffers + local (n_dist, n_val + 2) padded accumulators, i-chunks reduced by +.
function _threaded_2d_simd!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType, x_vecs, u_vecs, dist_be, val_be, ::Val{D},
    culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(), share = nothing,
) where {OT, CT, D}
    x_raw = ntuple(d -> collect(x_vecs[d]), Val(D))
    u_raw = ntuple(d -> collect(u_vecs[d]), Val(D))
    Np = length(x_raw[1])
    n_dist = n_histogram_bins(dist_be)
    n_val = n_histogram_bins(val_be)
    FTx = eltype(x_raw[1])
    j2d_plan = SF.squared_digitize_plan(dist_be)   # built once; read-only, shared across tasks
    # Bound once, never reassigned: the tasks close over these, and reassigning a captured
    # variable boxes it, which OhMyThreads rejects.
    grid = culling isa SFC.NoCulling ? nothing :
           SFC.cull_grid_for(x_raw, SFH.FlatGeometry{D}(), dist_be, culling)
    xc, uc = isnothing(grid) ? (x_raw, u_raw) :
             (SFC.apply_perm(x_raw, grid.perm), SFC.apply_perm(u_raw, grid.perm))
    wc = (weights isa SFC.NoWeights || isnothing(grid)) ? weights : weights[grid.perm]
    L = SFC._pair_scratch_length(Np)
    ps, pc = _greedy_reduce(_hist_add,
        () -> begin
            valbuf = Vector{OT}(undef, L)
            ((zeros(OT, n_dist, n_val + 2), zeros(CT, n_dist, n_val + 2)),
             (Vector{FTx}(undef, L), valbuf, Vector{Int32}(undef, L), Vector{Int32}(undef, L), Vector{Int32}(undef, L),
              SFC.needs_axis_buffer(second_axis) ? Vector{OT}(undef, L) : valbuf))
        end,
        (a, (keybuf, valbuf, idxbuf, colbuf, sel, axbuf), chunk) -> SFC._pf_2d_run_blocks!(a[1], a[2], sf, xc, uc,
            j2d_plan, val_be, Val(D), keybuf, valbuf, idxbuf, colbuf, sel, chunk, Np, grid, second_axis, axbuf, wc),
        _outer_chunks(grid, SFC._share_indices(grid, Np - 1, share)))
    sums2d .+= view(ps, :, 2:(n_val + 1))
    counts2d .+= view(pc, :, 2:(n_val + 1))
    return nothing
end

# --- 2D Array thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    geometry,
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number}
    x_tuple, u_tuple = SFC._prepared_tuples(geometry, x_arr, u_arr)
    return SFC.threaded_calculate_structure_function!(sums_2d, counts_2d, structure_function_type, x_tuple, u_tuple,
                                                      distance_bins, value_bins; geometry, kwargs...)
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    ::Type{CT};
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    N4 = n_histogram_bins(value_bins)

    sums_2d = zeros(OT, N3, N4)
    counts_2d = zeros(CT, N3, N4)

    SFC.threaded_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins,
        value_bins;
        second_axis,
        kwargs...,
    )

    return SFO.StructureFunction2DSumsAndCounts(
        structure_function_type,
        distance_bins,
        value_bins,
        sums_2d,
        counts_2d,
        second_axis,
    )
end

# --- Threaded single pass ---

function SFC._dispatch_single_pass(
    ::CB.AbstractThreadedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = n_histogram_bins(distance_bins)
    sums, counts = zeros(OT, SFC.SINGLE_PASS_N, n_bins), zeros(CT, SFC.SINGLE_PASS_N, n_bins)
    _threaded_single_pass_1d!(sums, counts, x, u, distance_bins, nothing; geometry, culling, weights)
    return (sums = sums, counts = counts)  # raw 6-row; public wrapper adds Helmholtz once
end

function SFC._dispatch_single_pass!(
    ::CB.AbstractThreadedBackend,
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _threaded_single_pass_1d!(sums, counts, x, u, distance_bins, nothing; geometry, culling, weights)
    return sums, counts
end

function SFC._partial_single_pass_1d(
    ::CB.AbstractThreadedBackend, x::AbstractMatrix{FT1}, u::AbstractMatrix{FT2}, distance_bins::AbstractVector,
    share::NTuple{2, Int}, ::Type{CT}; geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    n_bins = n_histogram_bins(distance_bins)
    sums, counts = zeros(OT, SFC.SINGLE_PASS_N, n_bins), zeros(CT, SFC.SINGLE_PASS_N, n_bins)
    _threaded_single_pass_1d!(sums, counts, x, u, distance_bins, share; geometry, culling, weights)
    return sums, counts
end

# The point single pass threaded into `sums`/`counts` over the outer indices `share` selects (`SFC._share_indices`):
# flat D ∈ (2,3) through the SIMD compute/scatter kernel, other geometries through the scalar loop, inputs sorted and
# prepared once and shared read-only.
function _threaded_single_pass_1d!(sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, x, u, distance_bins, share;
                                   geometry, culling, weights) where {OT, CT}
    n_bins = n_histogram_bins(distance_bins)
    n_points = size(x, 2)
    vD = SFC._simd_width(geometry)
    if vD !== nothing
        _threaded_sp_simd!(sums, counts, x, u, distance_bins, vD, culling, weights, share)
        return nothing
    end
    geom = geometry
    xk0, uk0 = SFH.prepare_pair_inputs(geom, x, u)
    dist_be = digitize_plan(distance_bins)
    grid, xk, uk = SFC.cull_sorted_matrices(xk0, uk0, geom, distance_bins, culling)
    wk = grid === nothing ? weights : SFC._permuted_point_weights(weights, grid.perm)
    chunk_sums, chunk_counts = _greedy_reduce(_hist_add,
        () -> ((zeros(OT, SFC.SINGLE_PASS_N, n_bins), zeros(CT, SFC.SINGLE_PASS_N, n_bins)), nothing),
        (a, _, chunk) -> SFC._sp1d_run_blocks!(a[1], a[2], xk, uk, dist_be, geom, n_bins, chunk, n_points, grid, wk),
        _outer_chunks(grid, SFC._share_indices(grid, n_points - 1, share)))
    sums .+= chunk_sums
    counts .+= chunk_counts
    return nothing
end

# Threaded single-pass SIMD over the outer indices `share` selects (every one when `nothing`): contiguous components
# shared, per-task buffers + local (6,nb) accumulators, i-chunks reduced by +.
function _threaded_sp_simd!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, x, u, dist_be, ::Val{D},
    culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(), share = nothing,
) where {OT, CT, D}
    x_raw = ntuple(d -> collect(view(x, d, :)), Val(D))
    u_raw = ntuple(d -> collect(view(u, d, :)), Val(D))
    Np = size(x, 2)
    nb = n_histogram_bins(dist_be)
    FTx = eltype(x_raw[1])
    sp_plan = SF.squared_digitize_plan(dist_be)   # built once; read-only, shared across tasks
    # Bound once, never reassigned: the tasks close over these, and reassigning a captured
    # variable boxes it, which OhMyThreads rejects.
    grid = culling isa SFC.NoCulling ? nothing :
           SFC.cull_grid_for(x_raw, SFH.FlatGeometry{D}(), dist_be, culling)
    xc, uc = isnothing(grid) ? (x_raw, u_raw) :
             (SFC.apply_perm(x_raw, grid.perm), SFC.apply_perm(u_raw, grid.perm))
    wc = isnothing(grid) ? weights : SFC._permuted_point_weights(weights, grid.perm)
    L = SFC._pair_scratch_length(Np)
    cs, cc = _greedy_reduce(_hist_add,
        () -> ((zeros(OT, SFC.SINGLE_PASS_N, nb), zeros(CT, SFC.SINGLE_PASS_N, nb)),
               (Vector{FTx}(undef, L), Vector{OT}(undef, L), Vector{OT}(undef, L), Vector{Int32}(undef, L),
                Vector{Int32}(undef, L))),
        (a, (keybuf, duLbuf, dn2buf, idxbuf, sel), chunk) -> SFC._sp_run_blocks!(a[1], a[2], xc, uc, sp_plan, Val(D),
            keybuf, duLbuf, dn2buf, idxbuf, sel, chunk, Np, grid, wc),
        _outer_chunks(grid, SFC._share_indices(grid, Np - 1, share)))
    sums .+= cs
    counts .+= cc
    return nothing
end

function SFC._dispatch_single_pass_2d(
    ::CB.AbstractThreadedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins,
    ::Type{CT};
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = SFC.n_histogram_bins(distance_bins)
    n_val = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1
    sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins, n_val)
    counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins, n_val)
    _threaded_sp2d!(sums, counts, x, u, distance_bins, digitize_plan(value_bins),
        geometry, n_bins, n_val, culling, weights)
    return sums, counts
end

function SFC._dispatch_single_pass_2d!(
    ::CB.AbstractThreadedBackend,
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _threaded_sp2d!(sums_3d, counts_3d, x, u, distance_bins, digitize_plan(value_bins),
        geometry, SFC.n_histogram_bins(distance_bins), size(sums_3d, 3), culling, weights)
    return sums_3d, counts_3d
end

# Thread-local interleaved accumulators reduced in place, each task running the single-pass 2D kernel the serial
# and distributed drivers use over the chunks it takes; the unpack to (6, n_bins, n_val) runs once.
function _threaded_sp2d!(
    sums_3d::AbstractArray{OT, 3}, counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix, u::AbstractMatrix, distance_bins, value_bins, geometry,
    n_bins::Int, n_val::Int, culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(), share = nothing,
) where {OT, CT}
    vD = SFC._simd_width(geometry)
    h = if vD !== nothing
        _threaded_sp2d_simd(OT, CT, x, u, distance_bins, value_bins, vD, n_bins, n_val, culling, weights, share)
    else
        _threaded_sp2d_scalar(OT, CT, x, u, distance_bins, value_bins, geometry, n_bins, n_val, culling, weights,
                              share)
    end
    SFC._sp2d_unpack!(sums_3d, counts_3d, h, n_bins, n_val)
    return nothing
end

# The flat single-pass 2D SIMD kernel threaded over the outer indices `share` selects (every one when `nothing`):
# components, plan and cull sort staged once and shared read-only, buffers made once per task.
function _threaded_sp2d_simd(::Type{OT}, ::Type{CT}, x, u, distance_bins, value_bins, ::Val{D}, n_bins::Int,
                             n_val::Int, culling, weights, share) where {OT, CT, D}
    x_raw = ntuple(d -> collect(view(x, d, :)), Val(D))
    u_raw = ntuple(d -> collect(view(u, d, :)), Val(D))
    Np = size(x, 2)
    FTx = eltype(x_raw[1])
    plan = SF.squared_digitize_plan(distance_bins)
    grid = culling isa SFC.NoCulling ? nothing :
           SFC.cull_grid_for(x_raw, SFH.FlatGeometry{D}(), distance_bins, culling)
    xc, uc = isnothing(grid) ? (x_raw, u_raw) :
             (SFC.apply_perm(x_raw, grid.perm), SFC.apply_perm(u_raw, grid.perm))
    wc = isnothing(grid) ? weights : SFC._permuted_point_weights(weights, grid.perm)
    L = SFC._pair_scratch_length(Np)
    Lc = SFC._sp2d_has_columns(value_bins, OT) ? L : 0
    return _greedy_reduce((a, b) -> (a .+= b; a),
        () -> (SFC._sp2d_histogram(OT, CT, n_bins, n_val),
               (Vector{FTx}(undef, L), Vector{OT}(undef, L), Vector{OT}(undef, L), Vector{Int32}(undef, L),
                ntuple(_ -> Vector{Int32}(undef, Lc), Val(SFC.SINGLE_PASS_N)), Vector{Int32}(undef, L))),
        (h, (keybuf, duLbuf, dn2buf, idxbuf, C, sel), chunk) -> SFC._sp2d_run_blocks!(h, xc, uc, plan, value_bins,
            Val(D), keybuf, duLbuf, dn2buf, idxbuf, C, sel, n_val, chunk, Np, grid, wc),
        _outer_chunks(grid, SFC._share_indices(grid, Np - 1, share)))
end

# The single-pass 2D scalar kernel threaded, for other metrics and widths: inputs sorted and prepared once.
function _threaded_sp2d_scalar(::Type{OT}, ::Type{CT}, x, u, distance_bins, value_bins, geom, n_bins::Int,
                               n_val::Int, culling, weights, share) where {OT, CT}
    grid, xs, us = SFC.cull_sorted_inputs(x, u, geom, distance_bins, culling)
    wk = grid === nothing ? weights : SFC._permuted_point_weights(weights, grid.perm)
    xk, uk = SFH.prepare_pair_inputs(geom, xs, us)
    be = digitize_plan(distance_bins)
    N = size(xs, 2)
    return _greedy_reduce((a, b) -> (a .+= b; a), () -> (SFC._sp2d_histogram(OT, CT, n_bins, n_val), nothing),
        (h, _, chunk) -> SFC._sp2d_curved_run_blocks!(h, xk, uk, be, value_bins, geom, n_bins, n_val, chunk, N, grid,
                                                      wk),
        _outer_chunks(grid, SFC._share_indices(grid, N - 1, share)))
end

function SFC._partial_single_pass_2d(
    ::CB.AbstractThreadedBackend, x::AbstractMatrix{FT1}, u::AbstractMatrix{FT2}, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins, share::NTuple{2, Int}, ::Type{CT}; geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    n_bins = n_histogram_bins(distance_bins)
    n_val = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    sums, counts = zeros(OT, SFC.SINGLE_PASS_N, n_bins, n_val), zeros(CT, SFC.SINGLE_PASS_N, n_bins, n_val)
    _threaded_sp2d!(sums, counts, x, u, distance_bins, digitize_plan(value_bins), geometry, n_bins, n_val,
                    culling, weights, share)
    return sums, counts
end

# Batch paths: tasks split the outer pair index `i`, with thread-local accumulators reduced by elementwise +.

# Executor: partition the (i, b) index space. `b` is split only as far as the accumulator budget demands;
# each batch chunk's tasks take chunks of `i` from that chunk's counter and write only their own accumulator,
# and the slices are summed into the result afterwards.
function _bl_threaded_exec(make_accum, make_scratch, run_chunk!, ifull, grid, B, accum_bytes, ws)
    nt = Threads.nthreads()
    if nt <= 1 || length(ifull) <= 1
        acc = SFC._bl_zero_accum!(SFC._bl_accum_pool(ws, make_accum, [B])[1])
        scratch = make_scratch()
        run_chunk!(acc, scratch, ifull, 1:B)
        SFC._bl_flush!(acc, scratch, 1:B)
        return acc
    end

    bchunks, n_ichunks = SFC._bl_partition(B, nt, accum_bytes)
    ichunks = _outer_chunks(grid, ifull, n_ichunks)
    tasks = [bi for bi in eachindex(bchunks) for _ in 1:n_ichunks]
    pool = SFC._bl_accum_pool(ws, make_accum, [length(bchunks[bi]) for bi in tasks])
    length(pool) == length(tasks) || throw(ArgumentError(
        "CPUSFWorkspace holds $(length(pool)) accumulators; this call needs $(length(tasks))"))
    counters = [Threads.Atomic{Int}(1) for _ in bchunks]

    OMT.tforeach(eachindex(tasks)) do k
        bi = tasks[k]
        _bl_run_chunks(pool[k], make_scratch(), run_chunk!, ichunks, counters[bi], bchunks[bi])
    end

    result = SFC._bl_result_accum(ws, make_accum, B)
    for (k, bi) in enumerate(tasks)
        selectdim(result[1], 1, bchunks[bi]) .+= pool[k][1]
        selectdim(result[2], 1, bchunks[bi]) .+= pool[k][2]
    end
    return result
end

# Named function so the pooled accumulator is concretely typed inside the task (the pool is heterogeneous in batch
# width, so indexing it is a dynamic call — once per task, not per pair). The task takes chunks of `ichunks` from
# `next`, which the other tasks of its batch chunk share.
@inline function _bl_run_chunks(acc, scratch, run_chunk!, ichunks, next, brange)
    SFC._bl_zero_accum!(acc)
    k = Threads.atomic_add!(next, 1)
    while k <= length(ichunks)
        run_chunk!(acc, scratch, ichunks[k], brange)
        k = Threads.atomic_add!(next, 1)
    end
    SFC._bl_flush!(acc, scratch, brange)
    return acc
end

SFC._bl_executor(::CB.AbstractThreadedBackend) = _bl_threaded_exec

function SFC.auxiliary_structure_function_threaded!(
    sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins;
    workspace = nothing, geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_1d!(sums, counts, sf_type, x, u, distance_bins, geometry,
        _bl_threaded_exec, workspace; weights, culling)
end

function SFC.auxiliary_joint2d_threaded!(
    sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins;
    workspace = nothing, geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins,
        geometry, _bl_threaded_exec, workspace; weights, culling, second_axis)
end

function SFC.threaded_calculate_structure_functions_single_pass!(
    sums::AbstractArray, counts::AbstractArray, x, u, distance_bins;
    workspace = nothing, geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_sp1d!(sums, counts, x, u, distance_bins, geometry,
        _bl_threaded_exec, workspace; weights, culling)
end

function SFC.threaded_calculate_structure_functions_single_pass_2d!(
    sums::AbstractArray, counts::AbstractArray, x, u, distance_bins,
    value_bins::SFC.SinglePass2DValueBins;
    workspace = nothing, geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_sp2d!(sums, counts, x, u, distance_bins, value_bins,
        geometry, _bl_threaded_exec, workspace; weights, culling)
end

# Batched (ndims(u) >= 3) non-mutating joint-2D.
function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    ::Type{CT};
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_dist = n_histogram_bins(distance_bins)
    n_val = n_histogram_bins(value_bins)
    bdims = size(u_arr)[3:end]
    sums = zeros(OT, n_dist, n_val, bdims...)
    counts = zeros(CT, n_dist, n_val, bdims...)
    SFC.auxiliary_joint2d_threaded!(sums, counts, structure_function_type, x_arr, u_arr, distance_bins, value_bins;
                                    second_axis, kwargs...)
    return SFO.StructureFunction2DSumsAndCounts(structure_function_type, distance_bins, value_bins, sums, counts,
                                                second_axis)
end

# A worker's share `(w, k)` of the outer indices threaded (distributed or MPI over threads).
function SFC._partial_sums_counts(
    ::CB.AbstractThreadedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    share::NTuple{2, Int},
    ::Type{CT};
    geometry = SFC.default_geometry(u_vecs),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {CT}
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    nb = n_histogram_bins(distance_bins)
    vD = SFC._simd_width(geometry)
    if vD !== nothing
        sums, counts = zeros(OT, nb), zeros(CT, nb)
        _threaded_pf_simd!(sums, counts, structure_function_type, x_vecs, u_vecs, distance_bins, vD, share;
                           geometry, culling, weights)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    return _threaded_scalar_1d(structure_function_type, geometry, x_vecs, u_vecs, distance_bins, share, OT, CT,
                               culling, weights)
end

# A worker's share `(w, k)` of the joint outer indices threaded (distributed or MPI over threads).
function SFC._partial_2d_sums_counts(
    ::CB.AbstractThreadedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    share::NTuple{2, Int},
    ::Type{CT};
    geometry = SFC.default_geometry(u_vecs),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
) where {CT}
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    sums = zeros(OT, n_histogram_bins(distance_bins), n_histogram_bins(value_bins))
    counts = zeros(CT, size(sums))
    val_be = digitize_plan(value_bins)
    vD = SFC._simd_width(geometry)
    if vD !== nothing
        _threaded_2d_simd!(sums, counts, structure_function_type, x_vecs, u_vecs, distance_bins, val_be, vD, culling,
                           weights, second_axis, share)
        return sums, counts
    end
    SFC._require_value_axis(second_axis, geometry)
    r = _threaded_scalar_2d(structure_function_type, geometry, x_vecs, u_vecs, distance_bins, value_bins, val_be,
                            OT, CT, culling, weights, second_axis, share)
    return r.sums, r.counts
end


# --- Harmonic pseudo-coefficients ---
# Each task sums its share of the points into its own coefficient matrix; the partials add because
# the sum over points is a reduction.
function SFC._direct_coefficients(::CB.AbstractThreadedBackend, f, θ, φ, s, lmax)
    chunks = SFC._outer_shares(nothing, 1:length(f), Threads.nthreads())
    return OMT.tmapreduce(+, chunks) do ch
        SFC.direct_coefficients_partial(f, θ, φ, s, lmax, ch)
    end
end

# --- Gridded sweeps ---

# Each task sweeps the round-robin chunks of the items it takes into private histograms with its own scratch;
# the partials add because a histogram is order-independent.
function SFC.threaded_sweep_reduce!(
    sums::AbstractArray{OT}, counts::AbstractArray{CT}, items::AbstractVector, make_scratch, body!,
) where {OT, CT}
    isempty(items) && return nothing
    ls, lc = _greedy_reduce(_hist_add, () -> ((zeros(OT, size(sums)), zeros(CT, size(counts))), make_scratch()),
        (a, scratch, chunk) -> foreach(it -> body!(a[1], a[2], it, scratch), chunk),
        _triangle_outer_chunks(items))
    sums .+= ls
    counts .+= lc
    return nothing
end

function SFC.threaded_sweep_foreach(items::AbstractVector, make_scratch, body!)
    isempty(items) && return nothing
    _greedy_reduce((a, _) -> a, () -> (nothing, make_scratch()), (_, scratch, it) -> body!(it, scratch), items)
    return nothing
end

# --- Tensor structure functions ---

# Each task accumulates the round-robin chunks of the outer index it takes into its own buffers; the partials add
# because a histogram is order-independent. The setup (widening, bin edges) happens once, shared read-only by the
# tasks.
function SFC.threaded_calculate_structure_function_tensor!(
    sums::AbstractArray, counts::AbstractArray, order::Val{P},
    shape::SFC.AbstractFieldShape{D}, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    geometry, axis = nothing,
    culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(),
) where {P, D}
    _threaded_tensor_pairs!(sums, counts, order, shape, x, u, distance_bins, nothing; geometry, axis, culling,
                            weights)
    return sums, counts
end

SFC._tensor_into!(::CB.AbstractThreadedBackend, sums, counts, order::Val, shape, x, u, distance_bins, share; kwargs...) =
    _threaded_tensor_pairs!(sums, counts, order, shape, x, u, distance_bins, share; kwargs...)

"""The tensor pairs whose lower index is in the outer indices `share` selects (every one when `nothing`) added into
`sums`/`counts` across threads."""
function _threaded_tensor_pairs!(
    sums, counts, order::Val, shape, x, u, distance_bins, share; geometry,
    axis = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(),
)
    _threaded_tensor_run!(sums, counts, order,
                          SFC._tensor_setup(order, shape, sums, counts, x, u, distance_bins, geometry, axis, weights,
                                            culling), share)
    return nothing
end

function _threaded_tensor_run!(sums, counts, order::Val, s, share)
    ls, lc = _greedy_reduce(_hist_add,
        () -> begin
            a = (zeros(eltype(sums), size(sums)), zeros(eltype(counts), size(counts)))
            (a, SFC._tensor_flat(a[1], a[2], s))
        end,
        (_, (sf, cf), chunk) -> SFC._tensor_pairs!(sf, cf, order, s, chunk),
        _outer_chunks(s.grid, SFC._share_indices(s.grid, s.N - 1, share)))
    sums .+= ls
    counts .+= lc
    return nothing
end

end # module
