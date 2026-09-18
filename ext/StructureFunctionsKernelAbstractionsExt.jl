"""
GPU-accelerated structure function kernels using KernelAbstractions.jl.

This extension is loaded automatically when `KernelAbstractions` is loaded by the user.
The `gpu_calculate_structure_function` entry point accepts any KA-compatible backend:
  - `KernelAbstractions.CPU()` – for CPU-parallel testing / parity verification
  - `CUDABackend()` from CUDA.jl – for NVIDIA GPU acceleration
  - `ROCBackend()` from AMDGPU.jl – for AMD GPU acceleration

For `N_dims ∈ {2,3}` the fast path uses tiled128 pair blocks with block-local
`UInt32` histograms (linear, log, and general bin routes). Joint 2D SF
(`calculate_structure_function` with `value_bins`) uses the same tiled schedule
when `n_dist × n_val ≤ SF_GPU_MAX_2D_HIST`. Six-invariant-type single-pass 1D uses
tiled128 block-local `(6, NB)` histograms when `NB ≤ SF_GPU_MAX_BINS`; six-invariant-type
single-pass 2D uses tiled128 pair traversal with **HTP-EJ** when `n_dist ≤ SF_GPU_MAX_BINS`
and distance bins are typed (`LinearBinEdges` / `LogBinEdges`):

- **On-chip** (`:shared`, `:typeplane`): shared histogram during the pair loop,
  block-end `@atomic` flush into final output (the joint 2D pattern), with no private partition and
  no merge.
- **Direct** (`:direct`): block-private global atomics during one pair pass, then a merge kernel.

Pass `force_global_atomic=true` to bypass HTP-EJ and use the global-atomic path.

## Count types on GPU

Device histogram buffers (global output and tiled `@localmem shared_cnts`) are
always `UInt32` — required for GPU integer atomics and fixed shared-memory layout.
The keyword `count_eltype` (default `UInt32`) selects the **host** array element
type after download only; it does not change device kernel types. Pass
`count_eltype=Int64` (etc.) to get converted host counts without recompiling kernels.

Bin edge routing mirrors CPU `BinEdges.jl`: the **type** of the bins object selects the
GPU kernel (`LinearBinEdges`, `LogBinEdges`, or general `Vector`). Pass typed edges for
fast paths; plain `Vector` inputs use the general kernel with no layout inference.

!!! note "KernelAbstractions Macro Limitations"
    We explicitly import `@index`, `@atomic`, `@Const`, `@private`, `@uniform`, and `@localmem` from `KernelAbstractions` because
    these macros currently fail to resolve correctly when called as `KA.@index`, etc.
    `@Const` is only valid on **kernel** parameter lists, not on host `@inline` helpers.
"""
module StructureFunctionsKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @localmem, @private, @uniform, @synchronize
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH, StructureFunctionTypes as SFT,
    AbstractBinEdges, LinearBinEdges, LogBinEdges, InfPaddedBinEdges, BinEdges, physical_edges_vector
using StructureFunctions.Calculations: GPUSFWorkspace, GPUSFLazyBuffers,
    _partition_n_tile_blocks, _ws_float_type,
    FullUpperTriangle, TilePairWorkList, tile_for, n_pair_blocks, schedule_for

# ---------------------------------------------------------------------------
# GPU bin normalization (unwrap wrappers; type selects kernel via dispatch)
# ---------------------------------------------------------------------------

"""Unwrap `InfPaddedBinEdges` / `BinEdges` wrappers; promote `AbstractRange` to `LinearBinEdges`."""
function _gpu_normalize_bins(edges)
    edges isa InfPaddedBinEdges && return _gpu_normalize_bins(edges.edges)
    edges isa LinearBinEdges && return edges
    edges isa LogBinEdges && return edges
    edges isa BinEdges && return _gpu_normalize_bins(edges.edges)
    edges isa AbstractRange && return LinearBinEdges(edges)
    edges isa AbstractVector && return collect(edges)
    error("GPUExt: unsupported bins type $(typeof(edges))")
end

"""Edge-vector length for histogram sizing (includes implicit `InfPaddedBinEdges` endpoints)."""
function _gpu_n_edges(bins)
    bins isa InfPaddedBinEdges && return length(bins)
    bins isa LinearBinEdges && return length(bins.edges)
    bins isa LogBinEdges && return length(bins)
    bins isa AbstractVector && return length(bins)
    error("GPUExt: unsupported bins type $(typeof(bins))")
end

"""Host edge vector for device upload (general / value-axis histograms)."""
function _gpu_host_edge_vector(bins)
    bins isa InfPaddedBinEdges && return [bins[i] for i in 1:length(bins)]
    bins isa LinearBinEdges && return collect(bins.edges)
    bins isa LogBinEdges && return physical_edges_vector(bins)
    bins isa AbstractVector && return bins isa Vector ? bins : Vector(bins)
    error("GPUExt: unsupported bins type $(typeof(bins))")
end

# ---------------------------------------------------------------------------
# Device-side digitize (matches HelperFunctions.digitize / BinEdges.jl)
# ---------------------------------------------------------------------------

# `T` is the EDGE type and the query is untyped, exactly as CPU `searchsortedfirst(::LinearBinEdges{T}, x)`
# declares it: the reconstructed edge is built in the edge type, the comparison promotes.
@inline function _gpu_digitize_linear(
    x,
    first_edge::T,
    last_edge::T,
    inv_step::T,
    step_val::T,
    n_edges::Int,
) where {T}
    if x <= first_edge
        return 0
    end
    if x > last_edge
        return n_edges
    end
    t = muladd(x, inv_step, -first_edge * inv_step)
    idx = clamp(floor(Int, t) + 1, 1, n_edges)
    edge_val = muladd(step_val, T(idx - 1), first_edge)
    search_idx = edge_val < x ? idx + 1 : idx
    return search_idx - 1
end

"""Log-spaced bins: `log(x)` then O(1) FMA on the log grid (matches CPU `LogBinEdges`)."""
@inline function _gpu_digitize_log_spaced(
    x,
    first_edge::T,
    last_edge::T,
    inv_step::T,
    step_val::T,
    n_edges::Int,
) where {T}
    x <= zero(x) && return 0
    return _gpu_digitize_linear(
        log(x), first_edge, last_edge, inv_step, step_val, n_edges,
    )
end

@inline function _gpu_digitize_log_spaced_col(
    x,
    first_edge,
    last_edge,
    inv_step,
    step_val,
    col::Int,
    n_edges::Int,
)
    x <= zero(x) && return 0
    return _gpu_digitize_linear(
        log(x),
        @inbounds(first_edge[col]),
        @inbounds(last_edge[col]),
        @inbounds(inv_step[col]),
        @inbounds(step_val[col]),
        n_edges,
    )
end

"""FMA scalar fields from [`LogBinEdges`](@ref) `log_linear` for GPU digitize."""
@inline function _dist_log_linear_fields(lbe::LogBinEdges)
    lb = lbe.log_linear
    return lb.first_edge, lb.last_edge, lb.inv_step, lb.step_val
end

@inline function _gpu_digitize_general(x, edges, n_edges::Int)
    low = 1
    high = n_edges
    while low <= high
        mid = (low + high) >>> 1
        if edges[mid] < x
            low = mid + 1
        else
            high = mid - 1
        end
    end
    return low - 1
end

@inline function _gpu_digitize_general_col(
    x,
    edges,
    col::Int,
    n_edges::Int,
)
    low = 1
    high = n_edges
    while low <= high
        mid = (low + high) >>> 1
        @inbounds edge_mid = edges[mid, col]
        if edge_mid < x
            low = mid + 1
        else
            high = mid - 1
        end
    end
    return low - 1
end


"""
Device digitize for [`InfPaddedBinEdges`](@ref) with [`LinearBinEdges`](@ref) interior.
Matches CPU `digitize(x, InfPadded)` (underflow bin 1, interior FMA, overflow past inner last).
"""
@inline function _gpu_digitize_inf_padded_linear(
    x,
    first_edge::T,
    last_edge::T,
    inv_step::T,
    step_val::T,
    n_inner_edges::Int,
    inner_last::T,
) where {T}
    if x <= typemin(T)
        return 0
    end
    if x > inner_last
        return n_inner_edges + 1
    end
    inner_bin = _gpu_digitize_linear(
        x, first_edge, last_edge, inv_step, step_val, n_inner_edges,
    )
    return inner_bin + 1
end

# ---------------------------------------------------------------------------
# Tiled128 + block-local UInt32 histogram (2D/3D production fast path)
# ---------------------------------------------------------------------------

"""Tile size for CADISHI-style pair blocks (`@localmem` histogram width is `SF_GPU_MAX_BINS`)."""
const SF_GPU_TILE = 128

"""Workgroup size for tiled structure-function kernels."""
const SF_GPU_TILED_WS = 256

"""Maximum distance-bin count compiled into tiled `@localmem` histograms."""
const SF_GPU_MAX_BINS = 128

"""
Maximum flat joint histogram cells ``n_dist × n_val`` for tiled128 2D joint SF
(``@localmem`` sums + counts). Requires ``n_dist ≤ SF_GPU_MAX_BINS``,
``n_val ≤ SF_GPU_MAX_BINS``, and ``n_dist * n_val ≤ SF_GPU_MAX_2D_HIST``.
"""
const SF_GPU_MAX_2D_HIST = SF_GPU_MAX_BINS * SF_GPU_MAX_BINS

"""True when the 2D joint histogram fits the tiled128 block-local path."""
@inline function _gpu_joint_2d_tiled_eligible(n_dist::Int, n_val::Int)
    return n_dist <= SF_GPU_MAX_BINS &&
           n_val <= SF_GPU_MAX_BINS &&
           n_dist * n_val <= SF_GPU_MAX_2D_HIST
end

"""Map 1-based upper-triangle pair index within a tile to `(ia, jb)` with `ia < jb`."""
@inline function _pair_from_linear(k, N)
    term = Float32(4 * N * N - 4 * N + 1 - 8 * (k - 1))
    i_float = (Float32(2 * N - 1) - sqrt(max(0.0f0, term))) * 0.5f0
    i = floor(Int, i_float) + 1
    j = k - (i - 1) * N + (i - 1) * i ÷ 2 + i
    return i, j
end

const SF_GPU_SINGLE_PASS_N = 6

include(joinpath(@__DIR__, "gpu", "value_digitize_plans.jl"))
include(joinpath(@__DIR__, "gpu", "sp2d_accumulation_strategy.jl"))
include(joinpath(@__DIR__, "gpu", "joint2d_shared_memory.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_1d_single_pass.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d_single_pass.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d_value_axis.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d_direct.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_batch.jl"))
# Unified parametric kernel core (building blocks) and the two tiled kernels built on it.
include(joinpath(@__DIR__, "gpu", "sf_core.jl"))
include(joinpath(@__DIR__, "gpu", "sf_tiled.jl"))
include(joinpath(@__DIR__, "gpu", "workspace.jl"))
include(joinpath(@__DIR__, "gpu", "launch.jl"))
include(joinpath(@__DIR__, "gpu", "tensor.jl"))
include(joinpath(@__DIR__, "gpu", "harmonic.jl"))
include(joinpath(@__DIR__, "gpu", "gridded_sweep.jl"))
include(joinpath(@__DIR__, "gpu", "multifields.jl"))

# The kernels compute Euclidean geometry inline, so every GPU entry types its `distance_metric`
# keyword as `DI.Euclidean`: asking for another metric is a TypeError naming the keyword, and the
# constraint lives in the signature.

"""`true` when `a` is an array living on `backend`. Only a missing `KA.get_backend` method (i.e. `a`
is not a recognized device array) counts as "not on this backend"; every other failure is a real
fault and propagates."""
function _array_on_backend(a, backend::KA.Backend)
    a_backend = try
        KA.get_backend(a)
    catch e
        e isa MethodError || rethrow()
        return false
    end
    return a_backend == backend
end

"""Host-dense view of `a` for `copyto!` into a device array. A host `Array` is passed through — the
DMA accepts it directly, so materializing a copy first would double the host traffic."""
@inline _as_host_dense(a::Array) = a
@inline _as_host_dense(a) = Array(a)

"""
    _stage_sf_device_inputs(backend, x_mat, u_mat, W, F, N_points)

Upload `(W, N_points)` coordinates and `(F, N_points)` fields to `backend` without padding. The two
widths differ on a sphere, where a point is an ambient position and a field an ambient vector.
Reuses device arrays when already on `backend` with matching shape.
"""
function _stage_sf_device_inputs(
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    W::Int,
    F::Int,
    N_points::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    if _array_on_backend(x_mat, backend) && _array_on_backend(u_mat, backend)
        d_x, n_x = size(x_mat)
        d_u, n_u = size(u_mat)
        if d_x == W && d_u == F && n_x == N_points && n_u == N_points
            if workspace !== nothing
                workspace.lazy.x_dev_cache = x_mat
                workspace.lazy.u_dev_cache = u_mat
            end
            return x_mat, u_mat
        end
    end

    if workspace !== nothing &&
       workspace.lazy.x_dev_cache !== nothing &&
       workspace.lazy.u_dev_cache !== nothing
        xd = workspace.lazy.x_dev_cache
        ud = workspace.lazy.u_dev_cache
        if _array_on_backend(xd, backend) &&
           _array_on_backend(ud, backend) &&
           size(xd) == (W, N_points) &&
           size(ud) == (F, N_points) &&
           eltype(xd) == FT &&
           eltype(ud) == FT
            copyto!(xd, _as_host_dense(x_mat))
            copyto!(ud, _as_host_dense(u_mat))
            return xd, ud
        end
    end

    x_dev = KA.allocate(backend, FT, W, N_points)
    u_dev = KA.allocate(backend, FT, F, N_points)
    copyto!(x_dev, _as_host_dense(x_mat))
    copyto!(u_dev, _as_host_dense(u_mat))
    if workspace !== nothing
        workspace.lazy.x_dev_cache = x_dev
        workspace.lazy.u_dev_cache = u_dev
    end
    return x_dev, u_dev
end

"""
    _gpu_prepare_and_stage(backend, x, u, distance_metric, N_points; workspace, distance_bins, culling) -> (geom, x_dev, u_dev)

Host-side prologue for a GPU entry: fix the geometry from the pre-conversion velocity dimension,
convert the inputs into the form the kernels index, and upload them. `size(u, 1)` is the velocity
dimension only before the conversion, so this is the one place it can be read.
"""
function _gpu_prepare_and_stage(
    backend::KA.Backend, x, u, distance_metric, N_points::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_bins = nothing,
    culling::SFC.CullingPolicy = SFC.NoCulling(),
    weights = SFC.NoWeights(),
)
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    xk, uk = SFH.prepare_pair_inputs(geom, x, u)
    W = SFC._val_int(SFH.coordinate_width(geom))
    F = SFC._val_int(SFH.field_width(geom))
    xk, uk, perm = _gpu_cull_and_permute!(workspace, backend, xk, uk, geom, distance_bins, culling, N_points)
    x_dev, u_dev = _stage_sf_device_inputs(backend, xk, uk, W, F, N_points; workspace = workspace)
    # A cull reorders the points, so the weights travel with the coordinates they belong to.
    # `_pair_weights` is also where the caller's `nothing` becomes `NoWeights()`, so everything
    # below this line sees one of the two internal spellings.
    w = SFC._pair_weights(weights, N_points, promote_type(float(eltype(xk)), float(eltype(uk))))
    w_dev = _sf_weights_to_device(backend, _permuted_weights(w, perm))
    return geom, x_dev, u_dev, w_dev
end

"""Reorder pair weights by a cull permutation; `NoWeights()` and an unpermuted call are no-ops."""
@inline _permuted_weights(w::SFC.NoWeights, _) = w
@inline _permuted_weights(w::AbstractVector, ::Nothing) = w
@inline _permuted_weights(w::AbstractVector, perm::AbstractVector{<:Integer}) = w[perm]

"""
    _gpu_cull_and_permute!(workspace, backend, xk, uk, geom, distance_bins, culling, N_points)

Sort the kernel coordinates into a cull grid, publish the memo this call culls with as
`workspace.lazy.active`, and return the reordered `(xk, uk)` for staging. The slot is cleared first,
so a launcher never sees a previous call's grid; each launcher takes its tile-pair list from the
active memo through `schedule_for` at its own tile size. The memo is kept as `workspace.lazy.cull`
and reused while the coordinates, cutoff and policy match, so a repeated call on the same points pays
only the field gather. The grid is built on the host, so culling needs host-resident inputs and a
workspace to carry the memo; without either, `AutoCulling` sweeps every pair and `AlwaysCulling`
says why it cannot.
"""
function _gpu_cull_and_permute!(workspace, backend, xk, uk, geom, distance_bins, culling, N_points)
    workspace === nothing && return _gpu_cull_no_workspace(culling, xk, uk)
    workspace.lazy.active = nothing
    (SFC._cull_enabled(culling) && distance_bins !== nothing) || return xk, uk, nothing
    (xk isa Array && uk isa Array) || return _gpu_cull_device_inputs(culling, xk, uk)
    cutoff = SFC.cull_cutoff_for(geom, distance_bins, culling)
    cutoff === nothing && return xk, uk, nothing
    memo = workspace.lazy.cull
    if SFC._cull_memo_hit(memo, xk, cutoff, culling)
        workspace.lazy.active = memo
        return memo.x_sorted, uk[:, memo.grid.perm], memo.grid.perm
    end
    grid, xs, us = SFC.cull_sorted_matrices(xk, uk, geom, distance_bins, culling)
    grid === nothing && return xk, uk, nothing
    memo = SFC.GPUCullMemo(copy(xk), cutoff, culling, grid, xs, v -> KA.adapt(backend, v),
                           Dict{Int, SFC.TilePairWorkList}())
    workspace.lazy.cull = memo
    workspace.lazy.active = memo
    return xs, us, grid.perm
end

"""The memo the current call culls with, or `nothing` without a workspace."""
_active_cull(::Nothing) = nothing
_active_cull(workspace::GPUSFWorkspace) = workspace.lazy.active

_gpu_cull_no_workspace(::SFC.AutoCulling, xk, uk) = (xk, uk, nothing)
_gpu_cull_no_workspace(::SFC.NoCulling, xk, uk) = (xk, uk, nothing)
_gpu_cull_no_workspace(::SFC.AlwaysCulling, _, _) = throw(ArgumentError(
    "GPU culling needs a GPUSFWorkspace to carry the tile-pair work list. Pass " *
    "workspace = GPUSFWorkspace(...), or culling = AutoCulling() / NoCulling().",
))
_gpu_cull_device_inputs(::SFC.AutoCulling, xk, uk) = (xk, uk, nothing)
_gpu_cull_device_inputs(::SFC.AlwaysCulling, _, _) = throw(ArgumentError(
    "GPU culling builds its cell grid on the host, so it needs host-resident x and u. Pass host " *
    "arrays, or culling = AutoCulling() / NoCulling().",
))

# ---------------------------------------------------------------------------
# Public API – extends the stub declared in Calculations.jl
# ---------------------------------------------------------------------------

"""
    gpu_calculate_structure_function(backend, x_mat, u_mat, distance_bins, sf_type; workgroup_size=64)

Compute structure functions on `backend` (any KernelAbstractions backend).

# Arguments
- `backend`: e.g. `KernelAbstractions.CPU()`, `CUDA.CUDABackend()`, etc.
- `x_mat`: `(N_dims, N_points)` matrix of spatial positions.
- `u_mat`: `(N_dims, N_points)` matrix of velocity components.
- `distance_bins`: bin *edges* (`AbstractVector{FT}` with `FT === eltype(x_mat)` — same
  element type as `x_mat`/`u_mat`; use `collect(FT, edges)` when building from a
  different-precision template). Also accepts [`AbstractBinEdges`](@ref) subtypes; the **type**
  selects the GPU kernel (`LinearBinEdges`, `LogBinEdges`, or general `Vector` only).
- `sf_type`: any `AbstractPairwiseStructureFunctionType`.
- `count_eltype::Type=UInt32`: element type of host count arrays returned or
  accumulated into caller buffers. Device histograms remain `UInt32`; conversion
  happens at download (see module docstring).

# Returns
The raw `StructureFunctionSumsAndCounts` accumulator with `Vector{Float64}` sums and a count
vector of type `count_eltype` (default `UInt32`, length `N_bins - 1`). The public
`calculate_structure_function` boundary maps this to the requested `output_type` (e.g. the
binned-mean `StructureFunction`) via `_finalize`.
"""
function SFC.gpu_calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT};
    kwargs...,
) where {FT}
    return _gpu_calculate_structure_function_core(
        sf_type,
        backend,
        x_mat,
        u_mat,
        distance_bins;
        kwargs...,
    )
end

"""
    _gpu_calculate_structure_function_core(sf_type, backend, x_mat, u_mat, distance_bins; workgroup_size=64)

GPU kernel execution core for structure-function evaluation on dense matrix inputs.

Accepts host `Matrix` or device arrays on `backend`. Inputs are staged as
`(N_dims, N_points)` with no padding (same layout as CPU).

Keyword arguments intended for CPU backends (e.g. `verbose`, `show_progress`) are
accepted and ignored so `calculate_structure_function(...; backend=CB.GPUBackend(...))`
can use the same call surface as threaded/serial paths.
"""
_tiled_launch_params(N_points::Int) = _tiled_launch_params(N_points, nothing)

# The prologue sets `workspace.lazy.active` on every call, so the schedule is always this call's.
function _tiled_launch_params(N_points::Int, workspace::Union{GPUSFWorkspace, Nothing})
    sched = schedule_for(_active_cull(workspace), N_points, SF_GPU_TILE)
    n_tile_blocks = n_pair_blocks(sched)
    ws = SF_GPU_TILED_WS
    return sched, n_tile_blocks, ws, n_tile_blocks * ws
end

"""
    _launch_sf_kernel!(backend, workgroup_size, out_dev, cnt_dev, x_dev, u_dev, sf_type, bins,
                       N_points, N_dims, N_bins, geom; workspace, weights)

Run a point list's distance histogram as the unified tiled kernel over a single slice: `out_dev`
and `cnt_dev` are `(NB,)`, viewed as the `(NMOM, NB, B)` the kernel writes with `NMOM = B = 1`.
`_sf_batch_dist_digitizer` turns any bin type into the digitizer that bins it exactly.
"""
function _launch_sf_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_dev,
    cnt_dev,
    x_dev,
    u_dev,
    sf_type,
    bins,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    NB = N_bins - 1
    _sf_tiled_1d_check_nb(NB)
    dig = _sf_point_dist_digitizer(backend, bins, workspace)
    D = N_dims
    _launch_sf_tiled_1d_varying!(
        backend,
        reshape(out_dev, 1, NB, 1),
        reshape(cnt_dev, 1, NB, 1),
        reshape(x_dev, D, N_points, 1),
        reshape(u_dev, D, N_points, 1),
        sf_type, dig, N_points, NB, 1,
        D == 2 ? Val(2) : Val(3), Val(1), geom;
        weights = weights,
    )
    return nothing
end

"""Distance digitizer for a point-list launch, reusing the workspace's device edge buffer for
general edges so a repeated call allocates nothing."""
function _sf_point_dist_digitizer(backend, bins, workspace)
    bins isa LinearBinEdges && return _sf_digitizer(bins)
    bins isa LogBinEdges && return _sf_digitizer(bins)
    bins isa AbstractRange && return _sf_digitizer(LinearBinEdges(bins))
    edges = bins isa BinEdges ? bins.edges : bins
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    if gen_e !== nothing && length(gen_e) == length(edges)
        copyto!(gen_e, edges)
        return _sf_digitizer_general(gen_e)
    end
    return _sf_digitizer_general(KA.adapt(backend, collect(edges)))
end

"""
    _launch_gpu_structure_function_core!(sf_type, backend, x_dev, u_dev, dist_bins, N_points, N_dims, N_bins; out_dev, cnt_dev, workspace=nothing, workgroup_size=64, synchronize=true)

Launch the tiled GPU structure-function kernel into pre-allocated device buffers (no allocation).
"""
function _launch_gpu_structure_function_core!(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_dev,
    u_dev,
    dist_bins,
    N_points::Int,
    N_dims::Int,
    N_bins::Int;
    out_dev,
    cnt_dev,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    workgroup_size::Int = 64,
    synchronize::Bool = true,
    weights = SFC.NoWeights(),
    geom,
)
    _launch_sf_kernel!(
        backend, workgroup_size,
        out_dev, cnt_dev, x_dev, u_dev,
        sf_type, dist_bins, N_points, N_dims, N_bins, geom;
        workspace = workspace, weights = weights,
    )
    synchronize && KA.synchronize(backend)
    return out_dev, cnt_dev
end

"""
    _launch_gpu_structure_function!(sf_type, backend, x_mat, u_mat, distance_bins; workgroup_size=64, workspace=nothing, synchronize=true)

Run the tiled GPU structure-function kernel and return device-resident `(out_dev, cnt_dev)`.
`distance_bins` must be a host edge vector (same convention as CPU).
Pass `workspace` to reuse device histogram buffers; default allocates fresh buffers each call.
"""
function _launch_gpu_structure_function!(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    count_eltype::Type = UInt32,
    kwargs...,
) where {FT}
    N_dims, N_points = size(x_mat)
    dist_bins = _gpu_normalize_bins(distance_bins)
    N_bins = _gpu_n_edges(distance_bins)

    if N_dims ∉ (2, 3)
        error("GPUExt: GPU structure functions require N_dims ∈ {2, 3} (got N_dims=$N_dims)")
    end
    NB = N_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("GPUExt: at most $SF_GPU_MAX_BINS distance bins on GPU (got NB=$NB)")

    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x_mat, u_mat, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    # The tiled kernel variant is chosen by the coordinate width it will index,
    # which is the converted width, not the width the caller passed.
    N_dims = SFC._val_int(SFH.coordinate_width(geom))

    # The workspace's count buffer is `UInt32`; a weighted sweep accumulates a pair mass, so it
    # takes its own buffer while the rest of the workspace (staging, cull memo) is still reused.
    CNT = _sf_count_type(w_dev, count_eltype, _sf_worst_case_pairs(N_points))
    if isnothing(workspace)
        out_dev = KA.zeros(backend, FT, NB)
        cnt_dev = KA.zeros(backend, CNT, NB)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :sf1d, NB)
        SFC.reset_histogram!(workspace)
        out_dev = workspace.out_sums_dev
        cnt_dev = CNT === eltype(workspace.out_cnts_dev) ? workspace.out_cnts_dev :
                  KA.zeros(backend, CNT, NB)
        ws = workspace
    end

    _launch_gpu_structure_function_core!(
        sf_type, backend, x_dev, u_dev, dist_bins, N_points, N_dims, N_bins;
        out_dev = out_dev, cnt_dev = cnt_dev, workspace = ws,
        workgroup_size = workgroup_size, synchronize = synchronize,
        weights = w_dev, geom = geom,
    )
    edges_host = _gpu_host_edge_vector(distance_bins)
    return out_dev, cnt_dev, edges_host
end

"""Download device sums/counts and accumulate into caller-owned host buffers."""
function _accumulate_gpu_sf_host!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    out_dev,
    cnt_dev;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {OT, CT}
    NB = length(output_sums)
    length(output_counts) == NB ||
        throw(ArgumentError("GPUExt: output_counts length ($(length(output_counts))) must match output_sums ($NB)"))
    if !isnothing(workspace) && OT === _ws_float_type(workspace)
        copyto!(workspace.host_sums_scratch, out_dev)
        output_sums .+= workspace.host_sums_scratch
    elseif OT === eltype(out_dev) && _array_on_backend(out_dev, KA.CPU())
        @. output_sums += out_dev
    else
        tmp_s = Vector{OT}(undef, NB)
        copyto!(tmp_s, out_dev)
        output_sums .+= tmp_s
    end
    if workspace !== nothing
        copyto!(workspace.host_counts_scratch, cnt_dev)
        if CT === UInt32
            output_counts .+= workspace.host_counts_scratch
        else
            @inbounds for k in eachindex(output_counts)
                output_counts[k] += CT(workspace.host_counts_scratch[k])
            end
        end
    elseif CT === UInt32
        tmp_c = Vector{UInt32}(undef, NB)
        copyto!(tmp_c, cnt_dev)
        output_counts .+= tmp_c
    else
        tmp_c = Vector{UInt32}(undef, NB)
        copyto!(tmp_c, cnt_dev)
        @inbounds for k in eachindex(output_counts)
            output_counts[k] += CT(tmp_c[k])
        end
    end
    return nothing
end

"""Owning download for the allocating API (no caller buffers)."""
function _download_gpu_sf_results(out_dev, cnt_dev, ::Type{FT}, ::Type{CT}) where {FT, CT}
    NB = length(out_dev)
    output = Vector{FT}(undef, NB)
    copyto!(output, out_dev)
    return output, _download_gpu_counts(cnt_dev, CT)
end

"""Download a device count histogram to a host array of type `CT`.

The device element type is [`_sf_count_type`](@ref)'s: `UInt32` for an unweighted sweep whose pair
count fits it, and `CT` when weights make the count a pair mass or the pair count needs the wider
type. The staging vector therefore takes the device's own element type, and the conversion to `CT`
happens on the host.
"""
function _download_gpu_counts(cnt_dev, ::Type{CT}) where {CT}
    tmp_c = Array(cnt_dev)
    return eltype(tmp_c) === CT ? tmp_c : CT.(tmp_c)
end

"""Copy device ``UInt32`` counts into a pre-allocated host buffer (``count_eltype``)."""
function _copy_gpu_counts!(host_counts, cnt_dev, ::Type{CT}) where {CT}
    tmp_c = Array(cnt_dev)
    if CT === eltype(host_counts)
        copyto!(host_counts, tmp_c)
    else
        copyto!(host_counts, CT.(tmp_c))
    end
    return host_counts
end

"""
    _download_gpu_sf_time_slice!(sums_sl, counts_sl, out_sums_dev, out_cnts_dev, ws)

Download one time slice of device histograms into host `sums_sl` / `counts_sl`.

CUDA/GPUArrays do **not** support `copyto!(cpu_subarray, cuarray)` with scalar
indexing disabled — see [GPUArrays.jl#422](https://github.com/JuliaGPU/GPUArrays.jl/issues/422)
and [CUDA.jl#1634](https://github.com/JuliaGPU/CUDA.jl/issues/1634).  Workaround:
DMA into a dense host `Vector` (`copyto!(scratch, cuarray)` or `Array(cuarray)`),
then host→host into the destination column/slice (`sums_sl .= scratch`).
"""
function _download_gpu_sf_time_slice!(
    sums_sl::AbstractArray{OT},
    counts_sl::AbstractArray{CT},
    out_sums_dev,
    out_cnts_dev,
    ws::GPUSFWorkspace,
) where {OT, CT}
    # Step 1: CuArray → dense host Vector (CUDA memcpy; never into SubArray dest)
    if ndims(out_sums_dev) == 1 && length(out_sums_dev) == length(ws.host_sums_scratch)
        copyto!(ws.host_sums_scratch, out_sums_dev)
        tmp_s = ws.host_sums_scratch
    else
        tmp_s = Array(out_sums_dev)
    end
    if ndims(out_cnts_dev) == 1 && length(out_cnts_dev) == length(ws.host_counts_scratch)
        copyto!(ws.host_counts_scratch, out_cnts_dev)
        tmp_c = ws.host_counts_scratch
    else
        tmp_c = Array(out_cnts_dev)
    end
    # Step 2: host → host (including `sums[:, t]` SubArray columns)
    if OT === eltype(tmp_s)
        copyto!(sums_sl, reshape(tmp_s, size(sums_sl)))
    else
        copyto!(sums_sl, OT.(reshape(tmp_s, size(sums_sl))))
    end
    if CT === UInt32
        copyto!(counts_sl, reshape(tmp_c, size(counts_sl)))
    else
        copyto!(counts_sl, CT.(reshape(tmp_c, size(counts_sl))))
    end
    return nothing
end

function SFC.gpu_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    kwargs...,
) where {OT, CT, FT}
    out_dev, cnt_dev, _ = _launch_gpu_structure_function!(
        sf_type, backend, x_mat, u_mat, distance_bins;
        workgroup_size = workgroup_size,
        workspace = workspace,
        count_eltype = CT,
        kwargs...,
    )
    _accumulate_gpu_sf_host!(
        output_sums, output_counts, out_dev, cnt_dev;
        workspace = workspace,
    )
    return nothing
end

function _gpu_calculate_structure_function_core(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    count_eltype::Type{CT} = UInt32,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT, CT}
    # This entry downloads before returning, so it must synchronize.
    synchronize || throw(
        ArgumentError(
            "gpu_calculate_structure_function returns host arrays and must synchronize; " *
            "use the in-place gpu_calculate_structure_function! for synchronize=false.",
        ),
    )
    # The tiled kernels stage a fixed number of coordinate components per point, so they cover the
    # widths they were written for. A width outside that set goes through the width-generic pair
    # kernel instead, which reads its widths from `Val` parameters.
    if size(x_mat, 1) ∉ (2, 3)
        nb = SFC.n_histogram_bins(distance_bins)
        sums = zeros(promote_type(float(FT), Float64), nb)
        counts = zeros(CT, nb)
        SFC.gpu_calculate_structure_function_fields!(
            CB.GPUBackend(backend), sums, counts, sf_type, x_mat,
            SFC.MF.Fields(vectors = (u_mat,)), distance_bins;
            distance_metric, culling, verbose, show_progress,
        )
        return SF.StructureFunctionSumsAndCounts(sf_type, distance_bins, sums, counts)
    end
    w = SFC._pair_weights(weights, size(x_mat, 2), float(FT))
    SFC._check_weighted_counts(w, CT)
    out_dev, cnt_dev, edges_host = _launch_gpu_structure_function!(
        sf_type, backend, x_mat, u_mat, distance_bins;
        workgroup_size = workgroup_size,
        workspace = workspace,
        distance_metric = distance_metric, culling = culling,
        weights = w, count_eltype = CT,
    )
    output, counts = _download_gpu_sf_results(out_dev, cnt_dev, FT, CT)

    return SF.StructureFunctionSumsAndCounts(sf_type, edges_host, output, counts)
end




"""True when six-invariant-type 1D single-pass can use tiled128 block-local histograms."""
@inline _gpu_single_pass_tiled_eligible(n_bins::Int) = n_bins <= SF_GPU_MAX_BINS

"""True when six-invariant-type 2D single-pass can use HTP-EJ tiled128 (`n_dist ≤ 64`, typed dist bins)."""
@inline _gpu_single_pass_2d_tiled_eligible(n_dist::Int) = n_dist <= SF_GPU_MAX_BINS

"""
Route to HTP-EJ whenever the distance axis is tiled-eligible. Every distance-bin form (linear, log,
arbitrary edges via device binary search) and every value plan has a tiled variant, so the
global-atomic pair loops are reached only when `n_dist` exceeds the tiled bin cap.
"""
@inline _gpu_single_pass_2d_use_tiled(dist_bins, ::GPUValueDigitizePlan, n_dist::Int) =
    _gpu_single_pass_2d_tiled_eligible(n_dist)

function _launch_single_pass_tiled_kernel!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    lbe::LinearBinEdges,
    N_points::Int,
    n_edges::Int,
    NB::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points, workspace)
    FT = eltype(lbe.edges)
    kernel! = _sf6_single_pass_kernel_tiled128_linear_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_edges, NB,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.step_val,
        sched, n_tile_blocks, ws,
        _sf_weights_to_device(backend, weights), Val(eltype(out_cnts_dev)), geom;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_tiled_kernel!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    lbe::LogBinEdges,
    N_points::Int,
    n_edges::Int,
    NB::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points, workspace)
    lb = lbe.log_linear
    kernel! = _sf6_single_pass_kernel_tiled128_log_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        lb.first_edge, lb.last_edge, lb.inv_step, lb.step_val,
        N_points, n_edges, NB,
        sched, n_tile_blocks, ws,
        _sf_weights_to_device(backend, weights), Val(eltype(out_cnts_dev)), geom;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_tiled_kernel!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    edges::Vector{FT},
    N_points::Int,
    n_edges::Int,
    NB::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
) where {FT}
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points, workspace)
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    if isnothing(gen_e)
        bins_dev = KA.allocate(backend, FT, n_edges)
        copyto!(bins_dev, edges)
    else
        bins_dev = gen_e
    end
    kernel! = _sf6_single_pass_kernel_tiled128_general_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        edges[1], bins_dev, N_points, n_edges, NB,
        sched, n_tile_blocks, ws,
        _sf_weights_to_device(backend, weights), Val(eltype(out_cnts_dev)), geom;
        ndrange = ndrange,
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Single-Pass GPU Kernels (global-atomic path when NB > SF_GPU_MAX_BINS)
# ---------------------------------------------------------------------------

@inline _gpu_ld_col(m, k::Int, ::Val{W}, ::Type{FT}) where {W, FT} =
    SA.SVector{W, FT}(ntuple(d -> @inbounds(m[d, k]), Val(W)))

@inline function _gpu_single_pass_pair_invariants(
    x_mat,
    u_mat,
    i::Int,
    j::Int,
    ::Val{2},
    ::Type{FT},
    geom,
) where {FT}
    X1 = _gpu_ld_col(x_mat, i, SFH.coordinate_width(geom), FT)
    X2 = _gpu_ld_col(x_mat, j, SFH.coordinate_width(geom), FT)
    U1 = _gpu_ld_col(u_mat, i, Val(2), FT)
    U2 = _gpu_ld_col(u_mat, j, Val(2), FT)
    ok, dist, frame = SFH.pair_frame(geom, X1, X2)
    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
    return ok, dist, du_L, du_n2
end

@inline function _gpu_single_pass_pair_invariants(
    x_mat,
    u_mat,
    i::Int,
    j::Int,
    ::Val{3},
    ::Type{FT},
    geom,
) where {FT}
    X1 = _gpu_ld_col(x_mat, i, SFH.coordinate_width(geom), FT)
    X2 = _gpu_ld_col(x_mat, j, SFH.coordinate_width(geom), FT)
    U1 = _gpu_ld_col(u_mat, i, Val(3), FT)
    U2 = _gpu_ld_col(u_mat, j, Val(3), FT)
    ok, dist, frame = SFH.pair_frame(geom, X1, X2)
    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
    return ok, dist, du_L, du_n2
end

@inline function _gpu_accumulate_single_pass_global!(
    output_sums,
    output_counts,
    bin::Int,
    du_L,
    du_n2,
    w = true,
)
    vals = SFC.single_pass_invariants(du_L, du_n2)
    for t in 1:SF_GPU_SINGLE_PASS_N
        @atomic output_sums[t, bin] += w * vals[t]
        @atomic output_counts[t, bin] += convert(eltype(output_counts), w)
    end
    return nothing
end

KA.@kernel unsafe_indices=true function _sf_single_pass_kernel_linear!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    wts,
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    step_val::FT,
    geom,
) where {D, FT}
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]

    if i < j
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT, geom,
        )
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, step_val, N_bins,
        )

        if ok && 1 <= bin < N_bins
            _gpu_accumulate_single_pass_global!(output_sums, output_counts, bin, du_L, du_n2,
                SFC._point_weight(wts, i) * SFC._point_weight(wts, j))
        end
    end
end

KA.@kernel unsafe_indices=true function _sf_single_pass_kernel_log!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    wts,
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    step_val::FT,
    geom,
) where {D, FT}
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]

    if i < j
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT, geom,
        )
        bin = _gpu_digitize_log_spaced(dist, first_edge, last_edge, inv_step, step_val, N_bins)

        if ok && 1 <= bin < N_bins
            _gpu_accumulate_single_pass_global!(output_sums, output_counts, bin, du_L, du_n2,
                SFC._point_weight(wts, i) * SFC._point_weight(wts, j))
        end
    end
end

KA.@kernel unsafe_indices=true function _sf_single_pass_kernel!(
    output_sums,                 # Matrix{FT} of size (6, N_bins-1)
    output_counts,               # Matrix{FT} of size (6, N_bins-1)
    @Const(x_mat),               # Matrix{FT} of size (2, N_points)
    @Const(u_mat),               # Matrix{FT} of size (2, N_points)
    @Const(distance_bins),       # monotone bin edges, length N_bins
    wts,
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    geom,
) where {D}
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]
    
    if i < j
        FT = eltype(x_mat)
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT, geom,
        )
        bin = _gpu_digitize_general(dist, distance_bins, N_bins)
        
        if ok && 1 <= bin < N_bins
            _gpu_accumulate_single_pass_global!(output_sums, output_counts, bin, du_L, du_n2,
                SFC._point_weight(wts, i) * SFC._point_weight(wts, j))
        end
    end
end

"""
Offer a non-batch single-pass 1D launch to the batch dispatcher as `B=1`, which reaches the CUDA
N-body kernel; `false` means the hook declined and the caller stays on the tiled128 path.

The offer is made per regime, not globally: a fixed-x 1D-individual launch at `B=1` pays strip
staging and two merge kernels for a strip of one, so it stays on its own path.
"""
@inline function _sp1d_try_fast_batch!(
    backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, dist_bins,
    N_points::Int, N_dims::Int, NB::Int, geom, cull,
)
    return SFC.gpu_fast_launch_1d_batch!(
        backend,
        reshape(out_sums_dev, SF_GPU_SINGLE_PASS_N, NB, 1),
        reshape(out_cnts_dev, SF_GPU_SINGLE_PASS_N, NB, 1),
        x_dev, reshape(u_dev, N_dims, N_points, 1),
        nothing, _sf_batch_dist_digitizer(backend, dist_bins),
        N_points, NB, 1, N_dims, SF_GPU_SINGLE_PASS_N, true, geom, cull,
    )
end

function _launch_single_pass_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    lbe::LinearBinEdges,
    N_points::Int,
    N_dims::Int,
    n_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    NB = n_edges - 1
    weights isa SFC.NoWeights &&
        _sp1d_try_fast_batch!(backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
                          lbe, N_points, N_dims, NB, geom, _active_cull(workspace)) && return nothing
    if N_dims == 2 && _gpu_single_pass_tiled_eligible(NB)
        return _launch_single_pass_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            lbe, N_points, n_edges, NB, geom; workspace = workspace, weights = weights,
        )
    end
    kernel! = _sf_single_pass_kernel_linear!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, _sf_weights_to_device(backend, weights),
        N_points, Val(N_dims), n_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.step_val, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_single_pass_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    lbe::LogBinEdges,
    N_points::Int,
    N_dims::Int,
    n_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    NB = n_edges - 1
    weights isa SFC.NoWeights &&
        _sp1d_try_fast_batch!(backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
                          lbe, N_points, N_dims, NB, geom, _active_cull(workspace)) && return nothing
    if N_dims == 2 && _gpu_single_pass_tiled_eligible(NB)
        return _launch_single_pass_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            lbe, N_points, n_edges, NB, geom; workspace = workspace, weights = weights,
        )
    end
    lb = lbe.log_linear
    kernel! = _sf_single_pass_kernel_log!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, _sf_weights_to_device(backend, weights),
        N_points, Val(N_dims), n_edges,
        lb.first_edge, lb.last_edge, lb.inv_step, lb.step_val, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_single_pass_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    edges::Vector{FT},
    N_points::Int,
    N_dims::Int,
    n_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
) where {FT}
    NB = n_edges - 1
    weights isa SFC.NoWeights &&
        _sp1d_try_fast_batch!(backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
                          edges, N_points, N_dims, NB, geom, _active_cull(workspace)) && return nothing
    if N_dims == 2 && _gpu_single_pass_tiled_eligible(NB)
        return _launch_single_pass_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            edges, N_points, n_edges, NB, geom; workspace = workspace, weights = weights,
        )
    end
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    if isnothing(gen_e)
        bins_dev = KA.allocate(backend, FT, n_edges)
        copyto!(bins_dev, edges)
    else
        bins_dev = gen_e
    end
    kernel! = _sf_single_pass_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        bins_dev, _sf_weights_to_device(backend, weights), N_points, Val(N_dims), n_edges, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Joint 2D SF kernels (one sf_type, distance × value histogram)
# ---------------------------------------------------------------------------

KA.@kernel unsafe_indices=true function _sf_joint_2d_kernel_linear!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    wts,
    @Const(value_edges),
    sf_type,
    N_points::Int,
    N_dist_bins::Int,
    N_val_edges::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    step_val::FT,
    geom,
    second_axis,
    ::Val{W},
) where {FT, W}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        XT = eltype(x_mat)
        UT = eltype(u_mat)
        X1 = _gpu_ld_col(x_mat, i, Val(W), XT)
        X2 = _gpu_ld_col(x_mat, j, Val(W), XT)
        U1 = _gpu_ld_col(u_mat, i, Val(W), UT)
        U2 = _gpu_ld_col(u_mat, j, Val(W), UT)
        ok, dist, frame = SFH.pair_frame(geom, X1, X2)
        dbin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, step_val, N_dist_bins,
        )
        if ok && 1 <= dbin < N_dist_bins
            dU, r̂ = SFH.pair_increments(geom, frame, dist, X1, X2, U1, U2)
            val = sf_type(dU, r̂)
            akey = _gpu_axis_key(second_axis, val, X1, X2, dist)
            vbin = _gpu_digitize_general(akey, value_edges, N_val_edges)
            if 1 <= vbin < N_val_edges
                pw = SFC._point_weight(wts, i) * SFC._point_weight(wts, j)
                @atomic output_sums[dbin, vbin] += pw * val
                @atomic output_counts[dbin, vbin] += convert(eltype(output_counts), pw)
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf_joint_2d_kernel_log!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    wts,
    @Const(value_edges),
    sf_type,
    N_points::Int,
    N_dist_bins::Int,
    N_val_edges::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    step_val::FT,
    geom,
    second_axis,
    ::Val{W},
) where {FT, W}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        XT = eltype(x_mat)
        UT = eltype(u_mat)
        X1 = _gpu_ld_col(x_mat, i, Val(W), XT)
        X2 = _gpu_ld_col(x_mat, j, Val(W), XT)
        U1 = _gpu_ld_col(u_mat, i, Val(W), UT)
        U2 = _gpu_ld_col(u_mat, j, Val(W), UT)
        ok, dist, frame = SFH.pair_frame(geom, X1, X2)
        dbin = _gpu_digitize_log_spaced(dist, first_edge, last_edge, inv_step, step_val, N_dist_bins)
        if ok && 1 <= dbin < N_dist_bins
            dU, r̂ = SFH.pair_increments(geom, frame, dist, X1, X2, U1, U2)
            val = sf_type(dU, r̂)
            akey = _gpu_axis_key(second_axis, val, X1, X2, dist)
            vbin = _gpu_digitize_general(akey, value_edges, N_val_edges)
            if 1 <= vbin < N_val_edges
                pw = SFC._point_weight(wts, i) * SFC._point_weight(wts, j)
                @atomic output_sums[dbin, vbin] += pw * val
                @atomic output_counts[dbin, vbin] += convert(eltype(output_counts), pw)
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf_joint_2d_kernel!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    wts,
    @Const(distance_edges),
    @Const(value_edges),
    sf_type,
    N_points::Int,
    N_dist_bins::Int,
    N_val_edges::Int,
    geom,
    second_axis,
    ::Val{W},
) where {W}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        XT = eltype(x_mat)
        UT = eltype(u_mat)
        X1 = _gpu_ld_col(x_mat, i, Val(W), XT)
        X2 = _gpu_ld_col(x_mat, j, Val(W), XT)
        U1 = _gpu_ld_col(u_mat, i, Val(W), UT)
        U2 = _gpu_ld_col(u_mat, j, Val(W), UT)
        ok, dist, frame = SFH.pair_frame(geom, X1, X2)
        dbin = _gpu_digitize_general(dist, distance_edges, N_dist_bins)
        if ok && 1 <= dbin < N_dist_bins
            dU, r̂ = SFH.pair_increments(geom, frame, dist, X1, X2, U1, U2)
            val = sf_type(dU, r̂)
            akey = _gpu_axis_key(second_axis, val, X1, X2, dist)
            vbin = _gpu_digitize_general(akey, value_edges, N_val_edges)
            if 1 <= vbin < N_val_edges
                pw = SFC._point_weight(wts, i) * SFC._point_weight(wts, j)
                @atomic output_sums[dbin, vbin] += pw * val
                @atomic output_counts[dbin, vbin] += convert(eltype(output_counts), pw)
            end
        end
    end
end

"""
Offer a non-batch joint 2D launch to the batch dispatcher as `B=1` (`NMOM=1`, so `sf_type` is
carried through and does the per-pair math); `false` means the hook declined and the caller
continues unchanged. `(n_dist, n_val)` reshapes to `(1, n_dist, n_val, 1)` at no cost — column-major
layout is identical.
"""
@inline function _joint2d_try_fast_batch!(
    backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type, dist_bins, vp,
    N_points::Int, n_dist::Int, n_val::Int, geom, cull,
)
    isnothing(vp) && return false
    D = size(x_dev, 1)
    return SFC.gpu_fast_launch_2d_batch!(
        backend,
        reshape(out_sums_dev, 1, n_dist, n_val, 1),
        reshape(out_cnts_dev, 1, n_dist, n_val, 1),
        x_dev, reshape(u_dev, D, N_points, 1),
        sf_type, _sf_batch_dist_digitizer(backend, dist_bins), vp,
        N_points, n_dist, n_val, 1, D, 1, true, geom, cull,
    )
end

function _launch_joint_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    sf_type,
    dist_bins::LinearBinEdges,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    val_plan::Union{GPUValueDigitizePlan, Nothing} = nothing,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    vp = isnothing(val_plan) && !isnothing(workspace) ? workspace.val_plan : val_plan
    weights isa SFC.NoWeights && second_axis isa SFC.InvariantValueAxis &&
        _joint2d_try_fast_batch!(backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
                             dist_bins, vp, N_points, n_dist, n_val, geom,
                             _active_cull(workspace)) && return nothing
    if _gpu_joint_2d_tiled_eligible(n_dist, n_val)
        return _launch_joint_2d_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
            sf_type, dist_bins, vp, N_points, n_dist_edges, n_val_edges, n_dist, n_val, geom;
            workspace = workspace, weights = weights, second_axis = second_axis,
        )
    end
    return _launch_joint_2d_global_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        sf_type, dist_bins, N_points, n_dist_edges, n_val_edges, geom;
        workspace = workspace, weights = weights, second_axis = second_axis,
    )
end

function _launch_joint_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    sf_type,
    dist_bins::LogBinEdges,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    val_plan::Union{GPUValueDigitizePlan, Nothing} = nothing,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    vp = isnothing(val_plan) && !isnothing(workspace) ? workspace.val_plan : val_plan
    weights isa SFC.NoWeights && second_axis isa SFC.InvariantValueAxis &&
        _joint2d_try_fast_batch!(backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
                             dist_bins, vp, N_points, n_dist, n_val, geom,
                             _active_cull(workspace)) && return nothing
    if _gpu_joint_2d_tiled_eligible(n_dist, n_val)
        return _launch_joint_2d_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
            sf_type, dist_bins, vp, N_points, n_dist_edges, n_val_edges, n_dist, n_val, geom;
            workspace = workspace, weights = weights, second_axis = second_axis,
        )
    end
    return _launch_joint_2d_global_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        sf_type, dist_bins, N_points, n_dist_edges, n_val_edges, geom;
        workspace = workspace, weights = weights, second_axis = second_axis,
    )
end

function _launch_joint_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    sf_type,
    dist_bins::Vector{FT},
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    val_plan::Union{GPUValueDigitizePlan, Nothing} = nothing,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
) where {FT}
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    vp = isnothing(val_plan) && !isnothing(workspace) ? workspace.val_plan : val_plan
    weights isa SFC.NoWeights && second_axis isa SFC.InvariantValueAxis &&
        _joint2d_try_fast_batch!(backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
                             dist_bins, vp, N_points, n_dist, n_val, geom,
                             _active_cull(workspace)) && return nothing
    if _gpu_joint_2d_tiled_eligible(n_dist, n_val)
        return _launch_joint_2d_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
            sf_type, dist_bins, vp, N_points, n_dist_edges, n_val_edges, n_dist, n_val, geom;
            workspace = workspace, weights = weights, second_axis = second_axis,
        )
    end
    return _launch_joint_2d_global_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        sf_type, dist_bins, N_points, n_dist_edges, n_val_edges, geom;
        workspace = workspace, weights = weights, second_axis = second_axis,
    )
end

function _launch_joint_2d_global_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    sf_type,
    lbe::LinearBinEdges,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
)
    kernel! = _sf_joint_2d_kernel_linear!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _sf_weights_to_device(backend, weights), value_edges_dev, sf_type,
        N_points, n_dist_edges, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.step_val, geom, second_axis,
        Val(SFC._val_int(SFH.coordinate_width(geom)));
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_joint_2d_global_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    sf_type,
    lbe::LogBinEdges,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
)
    lb = lbe.log_linear
    kernel! = _sf_joint_2d_kernel_log!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _sf_weights_to_device(backend, weights), value_edges_dev, sf_type,
        N_points, n_dist_edges, n_val_edges,
        lb.first_edge, lb.last_edge, lb.inv_step, lb.step_val, geom, second_axis,
        Val(SFC._val_int(SFH.coordinate_width(geom)));
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_joint_2d_global_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    sf_type,
    edges::Vector{FT},
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
) where {FT}
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    if isnothing(gen_e)
        dist_dev = KA.allocate(backend, FT, n_dist_edges)
        copyto!(dist_dev, edges)
    else
        dist_dev = gen_e
    end
    kernel! = _sf_joint_2d_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _sf_weights_to_device(backend, weights), dist_dev, value_edges_dev, sf_type,
        N_points, n_dist_edges, n_val_edges, geom, second_axis,
        Val(SFC._val_int(SFH.coordinate_width(geom)));
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_gpu_joint2d!(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix,
    u_mat::AbstractMatrix,
    distance_bins,
    value_bins;
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    count_eltype::Type = UInt32,
    second_axis = SFC.InvariantValueAxis(),
)
    FT = promote_type(eltype(x_mat), eltype(u_mat), eltype(distance_bins), eltype(value_bins))
    N_dims, N_points = size(x_mat)
    # Only the point count has to agree: on a shell a point takes two coordinates while the velocity
    # may carry a third, radial, component.
    size(u_mat, 2) == N_points ||
        throw(DimensionMismatch(
            "x_mat and u_mat must share the point count; got $(size(x_mat)) and $(size(u_mat))",
        ))
    N_dims in (2, 3) ||
        error("GPUExt: 2D joint structure functions require N_dims ∈ (2, 3) (got N_dims=$N_dims)")

    dist_bins = _gpu_normalize_bins(distance_bins)
    val_bins = _gpu_normalize_bins(value_bins)
    n_dist_edges = _gpu_n_edges(distance_bins)
    n_val_edges = _gpu_n_edges(value_bins)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    NB = n_dist

    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x_mat, u_mat, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    # The tiled kernel variant is chosen by the coordinate width it will index,
    # which is the converted width, not the width the caller passed.
    N_dims = SFC._val_int(SFH.coordinate_width(geom))

    val_plan = if isnothing(workspace)
        _joint2d_build_val_plan(backend, value_bins)
    else
        workspace.val_plan
    end

    if isnothing(workspace)
        value_host = _gpu_host_edge_vector(value_bins)
        value_edges_dev = KA.allocate(backend, FT, n_val_edges)
        copyto!(value_edges_dev, value_host)
        out_sums_dev = KA.zeros(backend, FT, n_dist, n_val)
        out_cnts_dev = KA.zeros(backend,
            _sf_count_type(w_dev, count_eltype, _sf_worst_case_pairs(N_points)), n_dist, n_val)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :joint2d, NB)
        SFC.reset_histogram!(workspace)
        value_edges_dev = workspace.value_edges_dev
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = workspace.out_cnts_dev
        ws = workspace
    end

    _launch_joint_2d_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        sf_type, dist_bins, N_points, n_dist_edges, n_val_edges,
        geom;
        val_plan = val_plan, workspace = ws, weights = w_dev, second_axis = second_axis,
    )
    synchronize && KA.synchronize(backend)
    return out_sums_dev, out_cnts_dev
end

"""
    gpu_calculate_structure_function_2d(sf_type, backend, x_mat, u_mat, distance_bins, value_bins; kwargs...)

Compute one 2D joint histogram (distance × SF value) for `sf_type` on `backend`.
Returns [`StructureFunction2DSumsAndCounts`](@ref) with the same flat edge vectors passed in.

Uses tiled128 block-local histograms when ``n_dist × n_val ≤ SF_GPU_MAX_2D_HIST``
(with each axis ``≤ SF_GPU_MAX_BINS``); otherwise falls back to ``(N_points, N_points)``
global-atomic pair kernels. Default compile-time shared histogram width is exact
``n_dist × n_val``; override on [`GPUSFWorkspace`](@ref) via `joint2d_compile_cells`
(see [`joint2d_smem_max`](@ref), [`joint2d_smem_align256`](@ref)). Device count buffers
are `UInt32`; `count_eltype` selects the host count matrix type after download.
Requires `N_dims == 2` matrix input.
"""
function SFC.gpu_calculate_structure_function_2d(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::AbstractVector{FT4};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, FT4 <: Number}
    if ndims(u) >= 3
        return _gpu_calculate_structure_function_2d_batch(
            sf_type, backend, x, u, distance_bins, value_bins; kwargs...,
        )
    end
    return _gpu_calculate_structure_function_2d_snapshot(
        sf_type, backend, x, u, distance_bins, value_bins; kwargs...,
    )
end

function _gpu_calculate_structure_function_2d_snapshot(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT1},
    u_mat::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::AbstractVector{FT4};
    count_eltype::Type{CT} = UInt32,
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, FT4 <: Number, CT}
    SFC._check_weighted_counts(weights, CT)
    # The angle reads `X2 - X1`, which is the separation only on a flat metric; on a curved one
    # each pair's direction lives in its own frame, as the CPU kernels say.
    if !(second_axis isa SFC.InvariantValueAxis)
        axis_geom = SFH.pair_geometry_for(distance_metric, Val(size(u_mat, 1)))
        axis_geom isa SFH.FlatGeometry ||
            SFC._require_value_axis(second_axis, axis_geom, size(u_mat, 1))
    end
    out_sums_dev, out_cnts_dev = _launch_gpu_joint2d!(
        sf_type, backend, x_mat, u_mat, distance_bins, value_bins;
        workgroup_size = workgroup_size, workspace = workspace,
        distance_metric = distance_metric, culling = culling,
        weights = weights, count_eltype = CT, second_axis = second_axis,
    )
    sums = Array(out_sums_dev)
    counts = _download_gpu_counts(out_cnts_dev, CT)
    return SF.StructureFunction2DSumsAndCounts(sf_type, distance_bins, value_bins, sums, counts)
end

# ---------------------------------------------------------------------------
# Single-pass 2D GPU kernels (eight distance × value joint histograms)
# ---------------------------------------------------------------------------


@inline function _gpu_accumulate_single_pass_2d_pair!(
    output_sums,
    output_counts,
    value_edges,
    bin::Int,
    du_L,
    du_n2,
    N_val_edges::Int,
    w,
)
    vals = SA.SVector(SFC.single_pass_invariants(du_L, du_n2))
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_general_col(vals[t], value_edges, t, N_val_edges)
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += w * vals[t]
            @atomic output_counts[t, bin, vbin] += convert(eltype(output_counts), w)
        end
    end
    return nothing
end

KA.@kernel unsafe_indices=true function _sf_single_pass_2d_kernel_linear!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(value_edges),
    wts,
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    N_val_edges::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    step_val::FT,
    geom,
) where {D, FT}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT, geom,
        )
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, step_val, N_bins,
        )
        if ok && 1 <= bin < N_bins
            _gpu_accumulate_single_pass_2d_pair!(
                output_sums, output_counts, value_edges, bin,
                du_L, du_n2, N_val_edges,
                SFC._point_weight(wts, i) * SFC._point_weight(wts, j),
            )
        end
    end
end

KA.@kernel unsafe_indices=true function _sf_single_pass_2d_kernel_log!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(value_edges),
    wts,
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    N_val_edges::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    step_val::FT,
    geom,
) where {D, FT}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT, geom,
        )
        bin = _gpu_digitize_log_spaced(dist, first_edge, last_edge, inv_step, step_val, N_bins)
        if ok && 1 <= bin < N_bins
            _gpu_accumulate_single_pass_2d_pair!(
                output_sums, output_counts, value_edges, bin,
                du_L, du_n2, N_val_edges,
                SFC._point_weight(wts, i) * SFC._point_weight(wts, j),
            )
        end
    end
end

KA.@kernel unsafe_indices=true function _sf_single_pass_2d_kernel!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(distance_bins),
    @Const(value_edges),
    wts,
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    N_val_edges::Int,
    geom,
) where {D}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        FT = eltype(x_mat)
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT, geom,
        )
        bin = _gpu_digitize_general(dist, distance_bins, N_bins)
        if ok && 1 <= bin < N_bins
            _gpu_accumulate_single_pass_2d_pair!(
                output_sums, output_counts, value_edges, bin,
                du_L, du_n2, N_val_edges,
                SFC._point_weight(wts, i) * SFC._point_weight(wts, j),
            )
        end
    end
end

const _SinglePass2DValueBins = SFC.SinglePass2DValueBins

include(joinpath(@__DIR__, "gpu", "batch_dispatch.jl"))

function _gpu_run_single_pass_2d!(
    gpu_backend::CB.AbstractGPUBackend,
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::_SinglePass2DValueBins;
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    force_global_atomic::Bool = false,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    backend = gpu_backend.backend
    FT = _sp2d_value_eltype(value_bins, promote_type(float(FT1), float(FT2), float(FT3)))
    N_dims, N_points = size(x)
    N_dims in (2, 3) ||
        error("GPUExt: single-pass 2D calculation requires N_dims ∈ (2, 3) (got N_dims=$N_dims)")

    n_bins = _gpu_n_edges(distance_bins) - 1
    n_val = size(sums_3d, 3)
    size(sums_3d) == (SF_GPU_SINGLE_PASS_N, n_bins, n_val) ||
        throw(DimensionMismatch("sums must have shape ($SF_GPU_SINGLE_PASS_N, n_bins, n_val); got $(size(sums_3d))"))
    size(counts_3d) == size(sums_3d) ||
        throw(DimensionMismatch("counts and sums must have the same shape"))
    SFC._validate_value_bins!(value_bins, n_val)

    dist_bins = _gpu_normalize_bins(distance_bins)
    n_dist_edges = _gpu_n_edges(distance_bins)
    n_val_edges = _sp2d_n_val_edges(value_bins)

    SFC._check_weighted_counts(weights, CT)
    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x, u, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    # The tiled kernel variant is chosen by the coordinate width it will index,
    # which is the converted width, not the width the caller passed.
    N_dims = SFC._val_int(SFH.coordinate_width(geom))
    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))

    # The global-atomic kernel only has methods for `GPUValueVectorCols`, so the plan choice must
    # use the same predicate the launcher does — see `_sp2d_prefers_global_atomics`.
    go_global = force_global_atomic ||
        _sp2d_prefers_global_atomics(n_bins, n_val, FT, SFC.gpu_device_caps(backend))
    if isnothing(workspace)
        val_plan = go_global ?
            _gpu_build_value_vector_cols_plan(backend, value_bins) :
            _gpu_build_value_digitize_plan(backend, value_bins)
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
        out_cnts_dev = KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :single_pass_2d, n_bins; n_val = n_val)
        SFC.reset_histogram!(workspace)
        val_plan = go_global ?
            GPUValueVectorCols{FT}(workspace.value_edges_dev) :
            workspace.val_plan
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = CNT === eltype(workspace.out_cnts_dev) ? workspace.out_cnts_dev :
                       KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
        ws = workspace
    end

    _launch_single_pass_2d!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, val_plan, N_points, N_dims, n_dist_edges, n_val_edges, geom;
        workspace = ws, weights = w_dev,
        force_global_atomic = force_global_atomic,
    )
    synchronize && KA.synchronize(backend)

    copyto!(sums_3d, Array(out_sums_dev))
    _copy_gpu_counts!(counts_3d, out_cnts_dev, CT)
    return sums_3d, counts_3d
end

function _launch_single_pass_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    lbe::LinearBinEdges,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    kernel! = _sf_single_pass_2d_kernel_linear!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        N_points, Val(2), n_dist_edges, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.step_val, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_single_pass_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    lbe::LogBinEdges,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    d_f, d_l, d_inv, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf_single_pass_2d_kernel_log!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        N_points, Val(2), n_dist_edges, n_val_edges,
        d_f, d_l, d_inv, d_st, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_single_pass_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    edges::Vector{FT},
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    if isnothing(gen_e)
        bins_dev = KA.allocate(backend, FT, n_dist_edges)
        copyto!(bins_dev, edges)
    else
        bins_dev = gen_e
    end
    kernel! = _sf_single_pass_2d_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        bins_dev, value_edges_dev, N_points, Val(2), n_dist_edges, n_val_edges, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

"""
    SFC._dispatch_single_pass(::CB.AbstractGPUBackend, x, u, distance_bins; workgroup_size=64, kwargs...)

Calculates single-pass structure functions utilizing GPU-accelerated computing.
Device histograms are `UInt32`; `count_eltype` (default `UInt32`) selects the host
count matrix type returned by [`append_helmholtz_rotational_divergent_rows`](@ref).
"""
function SFC._dispatch_single_pass(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    workgroup_size::Int = 64,
    count_eltype::Type{CT} = UInt32,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    backend = gpu_backend.backend
    FT = promote_type(float(FT1), float(FT2))
    N_dims, N_points = size(x)
    dist_bins = _gpu_normalize_bins(distance_bins)
    n_edges = _gpu_n_edges(distance_bins)
    n_bins = n_edges - 1

    N_dims in (2, 3) ||
        error("GPUExt: single-pass calculation requires N_dims ∈ (2, 3) (got N_dims=$N_dims)")

    w = SFC._pair_weights(weights, N_points, FT)
    SFC._check_weighted_counts(w, CT)
    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x, u, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = w)
    # The tiled kernel variant is chosen by the coordinate width it will index,
    # which is the converted width, not the width the caller passed.
    N_dims = SFC._val_int(SFH.coordinate_width(geom))

    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    if isnothing(workspace)
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_bins)
        out_cnts_dev = KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :single_pass, n_bins)
        SFC.reset_histogram!(workspace)
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = CNT === eltype(workspace.out_cnts_dev) ? workspace.out_cnts_dev :
                       KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = workspace
    end

    _launch_single_pass_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, N_points, N_dims, n_edges,
        geom;
        workspace = ws, weights = w_dev,
    )
    KA.synchronize(backend)

    sums = Array(out_sums_dev)
    counts = _download_gpu_counts(out_cnts_dev, CT)

    return (sums = sums, counts = counts)  # raw 6-row; public wrapper adds Helmholtz once
end

"""
    SFC.gpu_calculate_structure_functions_single_pass_2d(backend, x, u, distance_bins, value_bins; ...)

Six invariant distance × value joint histograms in one GPU pair pass. Pass one shared
[`LinearBinEdges`](@ref) / [`LogBinEdges`](@ref) / [`InfPaddedBinEdges`](@ref), or
`NTuple{6,...}` when value columns may differ.
"""
function SFC.gpu_calculate_structure_functions_single_pass_2d(
    backend::KA.Backend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::_SinglePass2DValueBins;
    workgroup_size::Int = 64,
    count_eltype::Type{CT} = UInt32,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    force_global_atomic::Bool = false,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = _sp2d_n_val_edges(value_bins) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    sums = zeros(OT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
    counts = zeros(CT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
    return _gpu_run_single_pass_2d!(
        CB.GPUBackend(backend), sums, counts, x, u, distance_bins, value_bins;
        workgroup_size = workgroup_size, workspace = workspace,
        synchronize = synchronize, force_global_atomic = force_global_atomic,
        distance_metric = distance_metric, culling = culling, weights = weights,
    )
end

function SFC.gpu_calculate_structure_functions_single_pass_2d!(
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    backend::KA.Backend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::_SinglePass2DValueBins;
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    force_global_atomic::Bool = false,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    fill!(sums_3d, zero(OT))
    fill!(counts_3d, 0)
    n_bins = length(distance_bins) - 1
    n_val = size(sums_3d, 3)
    size(sums_3d, 1) == SFC.SINGLE_PASS_N && size(sums_3d, 2) == n_bins ||
        throw(DimensionMismatch("sums must have shape ($(SFC.SINGLE_PASS_N), $n_bins, n_val); got $(size(sums_3d))"))
    size(counts_3d) == size(sums_3d) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums_3d))"))
    _gpu_run_single_pass_2d!(
        CB.GPUBackend(backend), sums_3d, counts_3d, x, u, distance_bins, value_bins;
        workgroup_size = workgroup_size, workspace = workspace,
        synchronize = synchronize, force_global_atomic = force_global_atomic,
        distance_metric = distance_metric, culling = culling, weights = weights,
    )
    return sums_3d, counts_3d
end

"""
    gpu_calculate_structure_function_batch!(sums, counts, sf_type, backend, x, u, distance_bins; workspace=nothing, ...)

Batch 1D structure functions over the third dimension of `x`, `u` with layout
`(N_dims, N_points, T)`. Host outputs `sums`, `counts` have shape `(NB, T)`.
Uploads `x`, `u` once and loops over optimized point-field GPU kernels.
"""
function SFC.gpu_calculate_structure_function_batch!(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT, 3},
    u::AbstractArray{FT, 3},
    distance_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {OT, CT, FT}
    N_dims, N_points, T = size(x)
    # Any distance-bin type: the unified device path digitizes via
    # _sf_batch_dist_digitizer (linear, log, or general edges).
    NB = length(distance_bins) - 1
    size(sums) == (NB, T) ||
        throw(DimensionMismatch("sums must have shape ($NB, $T); got $(size(sums))"))
    size(counts) == (NB, T) ||
        throw(DimensionMismatch("counts must have shape ($NB, $T); got $(size(counts))"))
    N_dims == 2 || N_dims == 3 ||
        error("GPUExt: slice batch requires N_dims ∈ {2,3} (got N_dims=$N_dims)")
    # Fused varying-x batch (B = T) through the unified N-body path: one launch for every slice.
    # `size(u, 1)` is the velocity dimension only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    oh, ch = _gpu_1d_individual_device(backend, sf_type, x, u, distance_bins, NB, T, false, OT,
        geom)
    copy!(sums, reshape(oh, NB, T))
    copy!(counts, reshape(ch, NB, T))
    return nothing
end

"""
    gpu_calculate_structure_function_2d_batch!(sums, counts, sf_type, backend, x, u, distance_bins, value_bins; workspace=nothing, ...)

Batch 2D joint histograms over `(N_dims, N_points, T)`; outputs `(n_dist, n_val, T)`.
"""
function SFC.gpu_calculate_structure_function_2d_batch!(
    sums::AbstractArray{OT, 3},
    counts::AbstractArray{CT, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT, 3},
    u::AbstractArray{FT, 3},
    distance_bins::AbstractVector{FT},
    value_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {OT, CT, FT}
    N_dims, N_points, T = size(x)
    n_dist = length(distance_bins) - 1
    n_val = length(value_bins) - 1
    size(sums) == (n_dist, n_val, T) ||
        throw(DimensionMismatch("sums must have shape ($n_dist, $n_val, $T); got $(size(sums))"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums))"))

    # Fused varying-x batch (B = T) through the unified joint-2D path (one launch,
    # no per-slice loop; N-body + privatized histogram).
    # `size(u, 1)` is the velocity dimension only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    oh, ch = _gpu_2d_unified_device(backend, x, u, sf_type, distance_bins, value_bins,
                                    Val(1), n_dist, n_val, T, false, OT, geom)
    copy!(sums, reshape(oh, n_dist, n_val, T))
    copy!(counts, reshape(ch, n_dist, n_val, T))
    return nothing
end

"""
    gpu_calculate_structure_functions_single_pass_batch!(sums, counts, backend, x, u, distance_bins; workspace=nothing, ...)

Batch six invariant 1D distance histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, T)`.
"""
function SFC.gpu_calculate_structure_functions_single_pass_batch!(
    sums::AbstractArray{OT, 3},
    counts::AbstractArray{CT, 3},
    backend::KA.Backend,
    x::AbstractArray{FT, 3},
    u::AbstractArray{FT, 3},
    distance_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {OT, CT, FT}
    N_dims, N_points, T = size(x)
    # Any distance-bin type: the unified device path digitizes via
    # _sf_batch_dist_digitizer (linear, log, or general edges).
    n_bins = length(distance_bins) - 1
    size(sums) == (SFC.SINGLE_PASS_N, n_bins, T) ||
        throw(DimensionMismatch("sums must have shape ($(SFC.SINGLE_PASS_N), $n_bins, $T); got $(size(sums))"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums))"))
    N_dims == 2 || N_dims == 3 ||
        error("GPUExt: single-pass slice batch requires N_dims ∈ {2,3} (got N_dims=$N_dims)")
    # Fused varying-x batch (B = T) through the unified N-body single-pass path.
    # `size(u, 1)` is the velocity dimension only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    oh, ch = _gpu_1d_unified_device(backend, x, u, nothing, distance_bins,
                                    Val(SFC.SINGLE_PASS_N), n_bins, T, false, OT, geom)
    copy!(sums, reshape(oh, SFC.SINGLE_PASS_N, n_bins, T))
    copy!(counts, reshape(ch, SFC.SINGLE_PASS_N, n_bins, T))
    return nothing
end

"""
    gpu_calculate_structure_functions_single_pass_2d_batch!(sums, counts, backend, x, u, distance_bins, value_bins; workspace=nothing, ...)

Batch six invariant distance × value joint histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, n_val, T)`.
"""
function SFC.gpu_calculate_structure_functions_single_pass_2d_batch!(
    sums::AbstractArray{OT, 4},
    counts::AbstractArray{CT, 4},
    backend::KA.Backend,
    x::AbstractArray{FT, 3},
    u::AbstractArray{FT, 3},
    distance_bins::AbstractVector{FT},
    value_bins::_SinglePass2DValueBins;
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {OT, CT, FT}
    N_dims, N_points, T = size(x)
    # Any distance-bin type: the unified device path digitizes via
    # _sf_batch_dist_digitizer (linear, log, or general edges).
    n_bins = length(distance_bins) - 1
    n_val = size(sums, 3)
    SFC._validate_value_bins!(value_bins, n_val)
    size(sums) == (SFC.SINGLE_PASS_N, n_bins, n_val, T) ||
        throw(DimensionMismatch("sums must have shape ($(SFC.SINGLE_PASS_N), $n_bins, $n_val, $T); got $(size(sums))"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums))"))
    N_dims == 2 || N_dims == 3 ||
        error("GPUExt: SP2D slice batch requires N_dims ∈ {2,3} (got N_dims=$N_dims)")
    # Fused varying-x batch (B = T) through the unified single-pass-2D path: N-body with a
    # dynamic-shared privatized histogram, one launch for every slice.
    # `size(u, 1)` is the velocity dimension only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    oh, ch = _gpu_sp2d_unified_device(backend, x, u, distance_bins, value_bins,
                                      n_bins, n_val, T, false, OT, geom)
    copy!(sums, reshape(oh, SFC.SINGLE_PASS_N, n_bins, n_val, T))
    copy!(counts, reshape(ch, SFC.SINGLE_PASS_N, n_bins, n_val, T))
    return nothing
end

"""
    SFC._dispatch_single_pass!(::CB.AbstractGPUBackend, sums, counts, x, u, distance_bins; ...)

In-place six-invariant-type single-pass distance histograms on GPU (no Helmholtz row append).
"""
function SFC._dispatch_single_pass!(
    gpu_backend::CB.AbstractGPUBackend,
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    kwargs...,
) where {OT, CT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    backend = gpu_backend.backend
    FT = promote_type(float(FT1), float(FT2))
    N_dims, N_points = size(x)
    dist_bins = _gpu_normalize_bins(distance_bins)
    n_edges = _gpu_n_edges(distance_bins)
    n_bins = n_edges - 1
    size(sums) == (SFC.SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("sums must have shape ($(SFC.SINGLE_PASS_N), $n_bins); got $(size(sums))"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums))"))
    N_dims in (2, 3) ||
        error("GPUExt: single-pass calculation requires N_dims ∈ (2, 3) (got N_dims=$N_dims)")

    w = SFC._pair_weights(weights, N_points, FT)
    SFC._check_weighted_counts(w, CT)
    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x, u, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = w)
    # The tiled kernel variant is chosen by the coordinate width it will index,
    # which is the converted width, not the width the caller passed.
    N_dims = SFC._val_int(SFH.coordinate_width(geom))

    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    if isnothing(workspace)
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_bins)
        out_cnts_dev = KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :single_pass, n_bins)
        SFC.reset_histogram!(workspace)
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = CNT === eltype(workspace.out_cnts_dev) ? workspace.out_cnts_dev :
                       KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = workspace
    end

    _launch_single_pass_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, N_points, N_dims, n_edges,
        geom;
        workspace = ws, weights = w_dev,
    )
    KA.synchronize(backend)

    tmp_s = Array(out_sums_dev)
    copyto!(sums, OT === eltype(tmp_s) ? tmp_s : OT.(tmp_s))
    tmp_c = Array(out_cnts_dev)
    copyto!(counts, CT === eltype(tmp_c) ? tmp_c : CT.(tmp_c))
    return sums, counts
end

function SFC.gpu_calculate_structure_function_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT};
    kwargs...,
) where {FT}
    return _gpu_calculate_structure_function_batch(
        sf_type, backend, x, u, distance_bins; kwargs...,
    )
end

function SFC.gpu_calculate_structure_function_2d_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT},
    value_bins::AbstractVector{FT};
    kwargs...,
) where {FT}
    return _gpu_calculate_structure_function_2d_batch(
        sf_type, backend, x, u, distance_bins, value_bins; kwargs...,
    )
end

function SFC._dispatch_single_pass(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractMatrix{FT1},
    u::AbstractArray{FT2, N},
    distance_bins::AbstractVector{FT3};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, N}
    N >= 3 ||
        throw(ArgumentError("fixed-x batch single-pass expects ndims(u) >= 3 (got ndims=$N)"))
    return _gpu_dispatch_single_pass_batch(
        gpu_backend.backend, x, u, distance_bins; kwargs...,
    )
end

function SFC._dispatch_single_pass(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractArray{FT1, N},
    u::AbstractArray{FT2, N},
    distance_bins::AbstractVector{FT3};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, N}
    N >= 3 ||
        throw(ArgumentError("batched single-pass expects ndims >= 3 (got ndims=$N)"))
    return _gpu_dispatch_single_pass_batch(
        gpu_backend.backend, x, u, distance_bins; kwargs...,
    )
end

function SFC._dispatch_single_pass!(
    gpu_backend::CB.AbstractGPUBackend,
    sums::AbstractArray{OT},
    counts::AbstractArray{CT},
    x::AbstractArray{FT1, N},
    u::AbstractArray{FT2, N},
    distance_bins::AbstractVector{FT3};
    kwargs...,
) where {OT, CT, FT1 <: Number, FT2 <: Number, FT3 <: Number, N}
    N >= 3 ||
        throw(ArgumentError("batched in-place single-pass expects ndims >= 3 (got ndims=$N)"))
    return _gpu_dispatch_single_pass_batch!(
        sums, counts, gpu_backend.backend, x, u, distance_bins; kwargs...,
    )
end

function SFC._dispatch_single_pass_2d(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractArray{FT1, N},
    u::AbstractArray{FT2, N},
    distance_bins::AbstractVector{FT3},
    value_bins::_SinglePass2DValueBins;
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, N}
    N >= 3 ||
        throw(ArgumentError("batched SP2D expects ndims >= 3 (got ndims=$N)"))
    return _gpu_dispatch_single_pass_2d_batch(
        gpu_backend.backend, x, u, distance_bins, value_bins; kwargs...,
    )
end

function SFC._dispatch_single_pass_2d(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::_SinglePass2DValueBins;
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    return SFC.gpu_calculate_structure_functions_single_pass_2d(
        gpu_backend.backend, x, u, distance_bins, value_bins; kwargs...,
    )
end

function SFC._dispatch_single_pass_2d(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractMatrix{FT1},
    u::AbstractArray{FT2, N},
    distance_bins::AbstractVector{FT3},
    value_bins::_SinglePass2DValueBins;
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, N}
    N >= 3 ||
        throw(ArgumentError("fixed-x batch SP2D expects ndims(u) >= 3 (got ndims=$N)"))
    return _gpu_dispatch_single_pass_2d_batch(
        gpu_backend.backend, x, u, distance_bins, value_bins; kwargs...,
    )
end

end # module GPUExt
