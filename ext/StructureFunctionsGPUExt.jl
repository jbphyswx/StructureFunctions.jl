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
  block-end `@atomic` flush into final output (same pattern as joint 2D) — no private partition, no merge.
- **Direct** (`:direct`): block-private global atomics during one pair pass, then merge kernel.

See [`gpu/SP2D_HTP_EJ.md`](../gpu/SP2D_HTP_EJ.md) for strategy, routing, benchmarks, and known perf gaps.

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
    We explicitly import `@index`, `@atomic`, `@Const`, `@private`, and `@localmem` from `KernelAbstractions` because
    these macros currently fail to resolve correctly when called as `KA.@index`, etc.
    `@Const` is only valid on **kernel** parameter lists, not on host `@inline` helpers.
"""
module StructureFunctionsGPUExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @localmem, @private, @synchronize
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH, StructureFunctionTypes as SFT,
    AbstractBinEdges, LinearBinEdges, LogBinEdges, InfPaddedBinEdges, BinEdges, physical_edges_vector

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

@inline function _gpu_digitize_linear(
    x::T,
    first_edge::T,
    last_edge::T,
    inv_step::T,
    offset::T,
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
    x::T,
    first_edge::T,
    last_edge::T,
    inv_step::T,
    offset::T,
    step_val::T,
    n_edges::Int,
) where {T}
    x <= zero(T) && return 0
    return _gpu_digitize_linear(
        log(x), first_edge, last_edge, inv_step, offset, step_val, n_edges,
    )
end

@inline function _gpu_digitize_log_spaced_col(
    x::T,
    first_edge,
    last_edge,
    inv_step,
    offset,
    step_val,
    col::Int,
    n_edges::Int,
) where {T}
    x <= zero(T) && return 0
    return _gpu_digitize_linear(
        log(x),
        @inbounds(first_edge[col]),
        @inbounds(last_edge[col]),
        @inbounds(inv_step[col]),
        @inbounds(offset[col]),
        @inbounds(step_val[col]),
        n_edges,
    )
end

"""FMA scalar fields from [`LogBinEdges`](@ref) `log_linear` for GPU digitize."""
@inline function _dist_log_linear_fields(lbe::LogBinEdges)
    lb = lbe.log_linear
    return lb.first_edge, lb.last_edge, lb.inv_step, lb.offset, lb.step_val
end

@inline function _gpu_digitize_general(x::T, edges, n_edges::Int) where {T}
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

"""
Device digitize for [`InfPaddedBinEdges`](@ref) with [`LinearBinEdges`](@ref) interior.
Matches CPU `digitize(x, InfPadded)` (underflow bin 1, interior FMA+offset, overflow past inner last).
"""
@inline function _gpu_digitize_inf_padded_linear(
    x::T,
    first_edge::T,
    last_edge::T,
    inv_step::T,
    offset::T,
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
        x, first_edge, last_edge, inv_step, offset, step_val, n_inner_edges,
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
const SF_GPU_MAX_BINS = 64

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

"""Map linear tile block id `k` to upper-triangle `(ti, tj)` with `ti ≤ tj`."""
@inline function _tile_from_linear(k, n_tiles)
    ti = one(k)
    rleft = k - one(k)
    while ti < n_tiles && rleft >= n_tiles - ti + one(k)
        rleft -= n_tiles - ti + one(k)
        ti += one(k)
    end
    tj = ti + rleft
    return ti, tj
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
include(joinpath(@__DIR__, "gpu", "kernels_1d.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_1d_single_pass.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d_single_pass.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d_value_axis.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d_direct.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_batch.jl"))
include(joinpath(@__DIR__, "gpu", "workspace.jl"))
include(joinpath(@__DIR__, "gpu", "launch.jl"))

function _array_on_backend(a, backend::KA.Backend)
    return try
        KA.get_backend(a) == backend
    catch
        false
    end
end

"""
    _stage_sf_device_inputs(backend, x_mat, u_mat, N_dims, N_points)

Upload `(N_dims, N_points)` inputs to `backend` without padding (same layout as CPU).
Reuses device arrays when already on `backend` with matching shape.
"""
function _stage_sf_device_inputs(
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    N_dims::Int,
    N_points::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    if _array_on_backend(x_mat, backend) && _array_on_backend(u_mat, backend)
        d_x, n_x = size(x_mat)
        d_u, n_u = size(u_mat)
        if d_x == N_dims && d_u == N_dims && n_x == N_points && n_u == N_points
            if workspace !== nothing
                workspace.x_dev_cache = x_mat
                workspace.u_dev_cache = u_mat
            end
            return x_mat, u_mat
        end
    end

    if workspace !== nothing &&
       workspace.x_dev_cache !== nothing &&
       workspace.u_dev_cache !== nothing
        xd = workspace.x_dev_cache
        ud = workspace.u_dev_cache
        if _array_on_backend(xd, backend) &&
           _array_on_backend(ud, backend) &&
           size(xd) == (N_dims, N_points) &&
           size(ud) == (N_dims, N_points) &&
           eltype(xd) == FT &&
           eltype(ud) == FT
            copyto!(xd, Array(x_mat))
            copyto!(ud, Array(u_mat))
            return xd, ud
        end
    end

    x_dev = KA.allocate(backend, FT, N_dims, N_points)
    u_dev = KA.allocate(backend, FT, N_dims, N_points)
    copyto!(x_dev, Array(x_mat))
    copyto!(u_dev, Array(u_mat))
    if workspace !== nothing
        workspace.x_dev_cache = x_dev
        workspace.u_dev_cache = u_dev
    end
    return x_dev, u_dev
end

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
When `return_sums_and_counts=true`, a `StructureFunctionSumsAndCounts` with
`Vector{Float64}` sums and count vector of type `count_eltype` (default `UInt32`,
length `N_bins - 1`).
Otherwise a binned-mean `StructureFunction`.
"""
function SFC.gpu_calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT};
    return_sums_and_counts::Bool = false,
    kwargs...,
) where {FT}
    return SFC.gpu_calculate_structure_function(
        sf_type,
        backend,
        x_mat,
        u_mat,
        distance_bins,
        Val(return_sums_and_counts);
        kwargs...,
    )
end

function SFC.gpu_calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT},
    ::Val{RSAC};
    kwargs...,
) where {FT, RSAC}
    return _gpu_calculate_structure_function_core(
        sf_type,
        backend,
        x_mat,
        u_mat,
        distance_bins,
        Val(RSAC);
        kwargs...,
    )
end

"""
    _gpu_calculate_structure_function_core(sf_type, backend, x_mat, u_mat, distance_bins, ::Val{RSAC}; workgroup_size=64)

GPU kernel execution core for structure-function evaluation on dense matrix inputs.

Accepts host `Matrix` or device arrays on `backend`. Inputs are staged as
`(N_dims, N_points)` with no padding (same layout as CPU).

Keyword arguments intended for CPU backends (e.g. `verbose`, `show_progress`) are
accepted and ignored so `calculate_structure_function(...; backend=GPUBackend(...))`
can use the same call surface as threaded/serial paths.
"""
function _tiled_launch_params(N_points::Int)
    TILE = SF_GPU_TILE
    n_tiles = cld(N_points, TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    ws = SF_GPU_TILED_WS
    return n_tiles, n_tile_blocks, ws, n_tile_blocks * ws
end

function _launch_sf_tiled_kernel!(
    backend::KA.Backend,
    out_dev,
    cnt_dev,
    x_dev,
    u_dev,
    sf_type,
    lbe::LinearBinEdges,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    dim2 = N_dims == 2
    kernel! = dim2 ?
        _sf_kernel_tiled128_2d_linear_u32!(backend, ws) :
        _sf_kernel_tiled128_3d_linear_u32!(backend, ws)
    kernel!(
        out_dev, cnt_dev, x_dev, u_dev, sf_type,
        N_points, N_dims, N_bins, NB,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_sf_tiled_kernel!(
    backend::KA.Backend,
    out_dev,
    cnt_dev,
    x_dev,
    u_dev,
    sf_type,
    lbe::LogBinEdges,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    dim2 = N_dims == 2
    lb = lbe.log_linear
    kernel! = dim2 ?
        _sf_kernel_tiled128_2d_log_u32!(backend, ws) :
        _sf_kernel_tiled128_3d_log_u32!(backend, ws)
    kernel!(
        out_dev, cnt_dev, x_dev, u_dev, sf_type,
        N_points, N_dims, N_bins, NB,
        lb.first_edge, lb.last_edge, lb.inv_step, lb.offset, lb.step_val,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_sf_tiled_kernel!(
    backend::KA.Backend,
    out_dev,
    cnt_dev,
    x_dev,
    u_dev,
    sf_type,
    edges::Vector{FT},
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int;
    general_edges_dev = nothing,
) where {FT}
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    dim2 = N_dims == 2
    if general_edges_dev === nothing
        bins_dev = KA.allocate(backend, FT, N_bins)
        copyto!(bins_dev, edges)
    else
        bins_dev = general_edges_dev
    end
    kernel! = dim2 ?
        _sf_kernel_tiled128_2d_general_u32!(backend, ws) :
        _sf_kernel_tiled128_3d_general_u32!(backend, ws)
    kernel!(
        out_dev, cnt_dev, x_dev, u_dev, sf_type,
        N_points, N_dims, N_bins, NB, edges[1], bins_dev,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_sf_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_dev,
    cnt_dev,
    x_dev,
    u_dev,
    sf_type,
    bins::LogBinEdges,
    N_points::Int,
    N_dims::Int,
    N_bins::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    NB = N_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("GPUExt: tiled kernels support at most $SF_GPU_MAX_BINS bins (got NB=$NB)")
    return _launch_sf_tiled_kernel!(
        backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, bins,
        N_points, N_dims, N_bins, NB,
    )
end

function _launch_sf_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_dev,
    cnt_dev,
    x_dev,
    u_dev,
    sf_type,
    bins::LinearBinEdges,
    N_points::Int,
    N_dims::Int,
    N_bins::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    NB = N_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("GPUExt: tiled kernels support at most $SF_GPU_MAX_BINS bins (got NB=$NB)")
    return _launch_sf_tiled_kernel!(
        backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, bins,
        N_points, N_dims, N_bins, NB,
    )
end

function _launch_sf_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_dev,
    cnt_dev,
    x_dev,
    u_dev,
    sf_type,
    bins::Vector{FT},
    N_points::Int,
    N_dims::Int,
    N_bins::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    NB = N_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("GPUExt: tiled kernels support at most $SF_GPU_MAX_BINS bins (got NB=$NB)")
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    return _launch_sf_tiled_kernel!(
        backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, bins,
        N_points, N_dims, N_bins, NB;
        general_edges_dev = gen_e,
    )
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
)
    _launch_sf_kernel!(
        backend, workgroup_size,
        out_dev, cnt_dev, x_dev, u_dev,
        sf_type, dist_bins, N_points, N_dims, N_bins;
        workspace = workspace,
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

    x_dev, u_dev = _stage_sf_device_inputs(backend, x_mat, u_mat, N_dims, N_points)

    if workspace === nothing
        out_dev = KA.zeros(backend, FT, NB)
        cnt_dev = KA.zeros(backend, UInt32, NB)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :sf1d, NB)
        SFC.reset_histogram!(workspace)
        out_dev = workspace.out_dev
        cnt_dev = workspace.cnt_dev
        ws = workspace
    end

    _launch_gpu_structure_function_core!(
        sf_type, backend, x_dev, u_dev, dist_bins, N_points, N_dims, N_bins;
        out_dev = out_dev, cnt_dev = cnt_dev, workspace = ws,
        workgroup_size = workgroup_size, synchronize = synchronize,
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
    if workspace !== nothing && OT === workspace.FT
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
    if CT === UInt32
        counts = Vector{UInt32}(undef, NB)
        copyto!(counts, cnt_dev)
    else
        tmp_c = Vector{UInt32}(undef, NB)
        copyto!(tmp_c, cnt_dev)
        counts = Vector{CT}(tmp_c)
    end
    return output, counts
end

"""Download device ``UInt32`` histogram buffer to a host array of type ``CT``.

All GPU paths accumulate counts in device-resident ``UInt32`` buffers (including
tiled ``@localmem shared_cnts``). ``count_eltype`` / ``CT`` only affects the host
return type, not kernel arithmetic.
"""
function _download_gpu_counts(cnt_dev, ::Type{CT}) where {CT}
    tmp_c = Array(cnt_dev)
    return CT === UInt32 ? tmp_c : CT.(tmp_c)
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
    distance_bins::AbstractVector{FT},
    ::Val{RSAC};
    workgroup_size::Int = 64,
    count_eltype::Type{CT} = UInt32,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    kwargs...,
) where {FT, RSAC, CT}
    out_dev, cnt_dev, edges_host = _launch_gpu_structure_function!(
        sf_type, backend, x_mat, u_mat, distance_bins;
        workgroup_size = workgroup_size,
        workspace = workspace,
    )
    N_bins = length(edges_host)
    output, counts = _download_gpu_sf_results(out_dev, cnt_dev, FT, CT)

    if RSAC
        return SF.StructureFunctionSumsAndCounts(sf_type, edges_host, output, counts)
    else
        output_div = similar(output)
        for k in eachindex(output)
            c = counts[k]
            output_div[k] = c == 0 ? FT(NaN) : output[k] / c
        end
        return SF.StructureFunction(sf_type, edges_host, output_div)
    end
end




"""True when six-invariant-type 1D single-pass can use tiled128 block-local histograms."""
@inline _gpu_single_pass_tiled_eligible(n_bins::Int) = n_bins <= SF_GPU_MAX_BINS

"""True when six-invariant-type 2D single-pass can use HTP-EJ tiled128 (`n_dist ≤ 64`, typed dist bins)."""
@inline _gpu_single_pass_2d_tiled_eligible(n_dist::Int) = n_dist <= SF_GPU_MAX_BINS

"""Route to HTP-EJ when tiled-eligible; raw `Vector` distance edges use naive pair loops."""
@inline _gpu_single_pass_2d_use_tiled(dist_bins, ::GPUValueDigitizePlan, n_dist::Int) =
    _gpu_single_pass_2d_tiled_eligible(n_dist) && !(dist_bins isa Vector)
@inline _gpu_single_pass_2d_use_tiled(dist_bins, ::GPUValueVectorCols, n_dist::Int) = false

function _launch_single_pass_tiled_kernel!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    lbe::LinearBinEdges,
    N_points::Int,
    n_edges::Int,
    NB::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    FT = eltype(lbe.edges)
    kernel! = _sf6_single_pass_kernel_tiled128_linear_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_edges, NB,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        n_tiles, n_tile_blocks, ws;
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
    NB::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lb = lbe.log_linear
    kernel! = _sf6_single_pass_kernel_tiled128_log_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        lb.first_edge, lb.last_edge, lb.inv_step, lb.offset, lb.step_val,
        N_points, n_edges, NB,
        n_tiles, n_tile_blocks, ws;
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
    NB::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    if gen_e === nothing
        bins_dev = KA.allocate(backend, FT, n_edges)
        copyto!(bins_dev, edges)
    else
        bins_dev = gen_e
    end
    kernel! = _sf6_single_pass_kernel_tiled128_general_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        edges[1], bins_dev, N_points, n_edges, NB,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Single-Pass GPU Kernels (global-atomic path when NB > SF_GPU_MAX_BINS)
# ---------------------------------------------------------------------------

@inline function _gpu_single_pass_pair_invariants(
    x_mat,
    u_mat,
    i::Int,
    j::Int,
    ::Val{2},
    ::Type{FT},
) where {FT}
    dx1 = x_mat[1, j] - x_mat[1, i]
    dx2 = x_mat[2, j] - x_mat[2, i]
    du1 = u_mat[1, j] - u_mat[1, i]
    du2 = u_mat[2, j] - u_mat[2, i]
    dist = sqrt(dx1^2 + dx2^2)
    rx = dx1 / dist
    ry = dx2 / dist
    du_L = du1 * rx + du2 * ry
    du_L2 = du_L * du_L
    du_T2 = du1^2 + du2^2 - du_L2
    return dist, du_L, du_L2, du_T2
end

@inline function _gpu_single_pass_pair_invariants(
    x_mat,
    u_mat,
    i::Int,
    j::Int,
    ::Val{3},
    ::Type{FT},
) where {FT}
    dx1 = x_mat[1, j] - x_mat[1, i]
    dx2 = x_mat[2, j] - x_mat[2, i]
    dx3 = x_mat[3, j] - x_mat[3, i]
    du1 = u_mat[1, j] - u_mat[1, i]
    du2 = u_mat[2, j] - u_mat[2, i]
    du3 = u_mat[3, j] - u_mat[3, i]
    dist = sqrt(dx1^2 + dx2^2 + dx3^2)
    rx = dx1 / dist
    ry = dx2 / dist
    rz = dx3 / dist
    du_L = du1 * rx + du2 * ry + du3 * rz
    du_L2 = du_L * du_L
    du_T2 = du1^2 + du2^2 + du3^2 - du_L2
    return dist, du_L, du_L2, du_T2
end

@inline function _gpu_accumulate_single_pass_global!(
    output_sums,
    output_counts,
    bin::Int,
    du_L,
    du_L2,
    du_T2,
)
    @atomic output_sums[1, bin] += du_L2 + du_T2
    @atomic output_sums[2, bin] += du_L2
    @atomic output_sums[3, bin] += du_T2
    @atomic output_sums[4, bin] += du_L * (du_L2 + du_T2)
    @atomic output_sums[5, bin] += du_L * du_L2
    @atomic output_sums[6, bin] += du_L * du_T2

    for t in 1:SF_GPU_SINGLE_PASS_N
        @atomic output_counts[t, bin] += one(eltype(output_counts))
    end
    return nothing
end

KA.@kernel function _sf_single_pass_kernel_linear!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {D, FT}
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]

    if i < j
        dist, du_L, du_L2, du_T2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT,
        )
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )

        if 1 <= bin < N_bins
            _gpu_accumulate_single_pass_global!(output_sums, output_counts, bin, du_L, du_L2, du_T2)
        end
    end
end

KA.@kernel function _sf_single_pass_kernel_log!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {D, FT}
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]

    if i < j
        dist, du_L, du_L2, du_T2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT,
        )
        bin = _gpu_digitize_log_spaced(dist, first_edge, last_edge, inv_step, offset, step_val, N_bins)

        if 1 <= bin < N_bins
            _gpu_accumulate_single_pass_global!(output_sums, output_counts, bin, du_L, du_L2, du_T2)
        end
    end
end

KA.@kernel function _sf_single_pass_kernel!(
    output_sums,                 # Matrix{FT} of size (6, N_bins-1)
    output_counts,               # Matrix{FT} of size (6, N_bins-1)
    @Const(x_mat),               # Matrix{FT} of size (2, N_points)
    @Const(u_mat),               # Matrix{FT} of size (2, N_points)
    @Const(distance_bins),       # monotone bin edges, length N_bins
    N_points::Int,
    ::Val{D},
    N_bins::Int,
) where {D}
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]
    
    if i < j
        FT = eltype(x_mat)
        dist, du_L, du_L2, du_T2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT,
        )
        bin = _gpu_digitize_general(dist, distance_bins, N_bins)
        
        if 1 <= bin < N_bins
            _gpu_accumulate_single_pass_global!(output_sums, output_counts, bin, du_L, du_L2, du_T2)
        end
    end
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
    n_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    NB = n_edges - 1
    if N_dims == 2 && _gpu_single_pass_tiled_eligible(NB)
        return _launch_single_pass_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            lbe, N_points, n_edges, NB; workspace = workspace,
        )
    end
    kernel! = _sf_single_pass_kernel_linear!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, Val(N_dims), n_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val;
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
    n_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    NB = n_edges - 1
    if N_dims == 2 && _gpu_single_pass_tiled_eligible(NB)
        return _launch_single_pass_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            lbe, N_points, n_edges, NB; workspace = workspace,
        )
    end
    lb = lbe.log_linear
    kernel! = _sf_single_pass_kernel_log!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, Val(N_dims), n_edges,
        lb.first_edge, lb.last_edge, lb.inv_step, lb.offset, lb.step_val;
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
    n_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    NB = n_edges - 1
    if N_dims == 2 && _gpu_single_pass_tiled_eligible(NB)
        return _launch_single_pass_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            edges, N_points, n_edges, NB; workspace = workspace,
        )
    end
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    if gen_e === nothing
        bins_dev = KA.allocate(backend, FT, n_edges)
        copyto!(bins_dev, edges)
    else
        bins_dev = gen_e
    end
    kernel! = _sf_single_pass_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        bins_dev, N_points, Val(N_dims), n_edges;
        ndrange = (N_points, N_points),
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Joint 2D SF kernels (one sf_type, distance × value histogram)
# ---------------------------------------------------------------------------

KA.@kernel function _sf_joint_2d_kernel_linear!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(value_edges),
    sf_type,
    N_points::Int,
    N_dist_bins::Int,
    N_val_edges::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {FT}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        X1 = SA.SVector{2}(x_mat[1, i], x_mat[2, i])
        X2 = SA.SVector{2}(x_mat[1, j], x_mat[2, j])
        U1 = SA.SVector{2}(u_mat[1, i], u_mat[2, i])
        U2 = SA.SVector{2}(u_mat[1, j], u_mat[2, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2)
        dbin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_dist_bins,
        )
        if 1 <= dbin < N_dist_bins
            r̂ = dX / dist
            val = sf_type(U2 - U1, r̂)
            vbin = _gpu_digitize_general(val, value_edges, N_val_edges)
            if 1 <= vbin < N_val_edges
                @atomic output_sums[dbin, vbin] += val
                @atomic output_counts[dbin, vbin] += one(eltype(output_counts))
            end
        end
    end
end

KA.@kernel function _sf_joint_2d_kernel_log!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(value_edges),
    sf_type,
    N_points::Int,
    N_dist_bins::Int,
    N_val_edges::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {FT}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        X1 = SA.SVector{2}(x_mat[1, i], x_mat[2, i])
        X2 = SA.SVector{2}(x_mat[1, j], x_mat[2, j])
        U1 = SA.SVector{2}(u_mat[1, i], u_mat[2, i])
        U2 = SA.SVector{2}(u_mat[1, j], u_mat[2, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2)
        dbin = _gpu_digitize_log_spaced(dist, first_edge, last_edge, inv_step, offset, step_val, N_dist_bins)
        if 1 <= dbin < N_dist_bins
            dU = U2 - U1
            r̂ = dX / dist
            val = sf_type(dU, r̂)
            vbin = _gpu_digitize_general(val, value_edges, N_val_edges)
            if 1 <= vbin < N_val_edges
                @atomic output_sums[dbin, vbin] += val
                @atomic output_counts[dbin, vbin] += one(eltype(output_counts))
            end
        end
    end
end

KA.@kernel function _sf_joint_2d_kernel!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(distance_edges),
    @Const(value_edges),
    sf_type,
    N_points::Int,
    N_dist_bins::Int,
    N_val_edges::Int,
)
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        X1 = SA.SVector{2}(x_mat[1, i], x_mat[2, i])
        X2 = SA.SVector{2}(x_mat[1, j], x_mat[2, j])
        U1 = SA.SVector{2}(u_mat[1, i], u_mat[2, i])
        U2 = SA.SVector{2}(u_mat[1, j], u_mat[2, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2)
        dbin = _gpu_digitize_general(dist, distance_edges, N_dist_bins)
        if 1 <= dbin < N_dist_bins
            dU = U2 - U1
            r̂ = dX / dist
            val = sf_type(dU, r̂)
            vbin = _gpu_digitize_general(val, value_edges, N_val_edges)
            if 1 <= vbin < N_val_edges
                @atomic output_sums[dbin, vbin] += val
                @atomic output_counts[dbin, vbin] += one(eltype(output_counts))
            end
        end
    end
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
    n_val_edges::Int;
    val_plan::Union{GPUValueDigitizePlan, Nothing} = nothing,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    vp = val_plan === nothing && workspace !== nothing ? workspace.val_plan : val_plan
    if _gpu_joint_2d_tiled_eligible(n_dist, n_val)
        return _launch_joint_2d_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
            sf_type, dist_bins, vp, N_points, n_dist_edges, n_val_edges, n_dist, n_val;
            workspace = workspace,
        )
    end
    return _launch_joint_2d_global_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        sf_type, dist_bins, N_points, n_dist_edges, n_val_edges;
        workspace = workspace,
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
    n_val_edges::Int;
    val_plan::Union{GPUValueDigitizePlan, Nothing} = nothing,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    vp = val_plan === nothing && workspace !== nothing ? workspace.val_plan : val_plan
    if _gpu_joint_2d_tiled_eligible(n_dist, n_val)
        return _launch_joint_2d_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
            sf_type, dist_bins, vp, N_points, n_dist_edges, n_val_edges, n_dist, n_val;
            workspace = workspace,
        )
    end
    return _launch_joint_2d_global_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        sf_type, dist_bins, N_points, n_dist_edges, n_val_edges;
        workspace = workspace,
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
    n_val_edges::Int;
    val_plan::Union{GPUValueDigitizePlan, Nothing} = nothing,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    vp = val_plan === nothing && workspace !== nothing ? workspace.val_plan : val_plan
    if _gpu_joint_2d_tiled_eligible(n_dist, n_val)
        return _launch_joint_2d_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
            sf_type, dist_bins, vp, N_points, n_dist_edges, n_val_edges, n_dist, n_val;
            workspace = workspace,
        )
    end
    return _launch_joint_2d_global_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        sf_type, dist_bins, N_points, n_dist_edges, n_val_edges;
        workspace = workspace,
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
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    kernel! = _sf_joint_2d_kernel_linear!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev, sf_type,
        N_points, n_dist_edges, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val;
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
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    lb = lbe.log_linear
    kernel! = _sf_joint_2d_kernel_log!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev, sf_type,
        N_points, n_dist_edges, n_val_edges,
        lb.first_edge, lb.last_edge, lb.inv_step, lb.offset, lb.step_val;
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
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    if gen_e === nothing
        dist_dev = KA.allocate(backend, FT, n_dist_edges)
        copyto!(dist_dev, edges)
    else
        dist_dev = gen_e
    end
    kernel! = _sf_joint_2d_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_dev, value_edges_dev, sf_type,
        N_points, n_dist_edges, n_val_edges;
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
)
    FT = promote_type(eltype(x_mat), eltype(u_mat), eltype(distance_bins), eltype(value_bins))
    N_dims, N_points = size(x_mat)
    size(u_mat) == (N_dims, N_points) ||
        throw(DimensionMismatch("x_mat and u_mat must have the same shape"))
    N_dims == 2 ||
        error("GPUExt: 2D joint structure functions require N_dims == 2 (got N_dims=$N_dims)")

    dist_bins = _gpu_normalize_bins(distance_bins)
    val_bins = _gpu_normalize_bins(value_bins)
    n_dist_edges = _gpu_n_edges(distance_bins)
    n_val_edges = _gpu_n_edges(value_bins)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    NB = n_dist

    x_dev, u_dev = _stage_sf_device_inputs(backend, x_mat, u_mat, N_dims, N_points)

    val_plan = if workspace === nothing
        _joint2d_build_val_plan(backend, value_bins)
    else
        workspace.val_plan
    end

    if workspace === nothing
        value_host = _gpu_host_edge_vector(value_bins)
        value_edges_dev = KA.allocate(backend, FT, n_val_edges)
        copyto!(value_edges_dev, value_host)
        out_sums_dev = KA.zeros(backend, FT, n_dist, n_val)
        out_cnts_dev = KA.zeros(backend, UInt32, n_dist, n_val)
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
        sf_type, dist_bins, N_points, n_dist_edges, n_val_edges;
        val_plan = val_plan, workspace = ws,
    )
    synchronize && KA.synchronize(backend)
    return out_sums_dev, out_cnts_dev
end

"""
    gpu_calculate_structure_function_2d(sf_type, backend, x_mat, u_mat, distance_bins, value_bins; kwargs...)

Compute one 2D joint histogram (distance × SF value) for `sf_type` on `backend`.
Returns [`StructureFunction2D`](@ref) with the same flat edge vectors passed in.

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
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, FT4 <: Number, CT}
    out_sums_dev, out_cnts_dev = _launch_gpu_joint2d!(
        sf_type, backend, x_mat, u_mat, distance_bins, value_bins;
        workgroup_size = workgroup_size, workspace = workspace,
    )
    sums = Array(out_sums_dev)
    counts = _download_gpu_counts(out_cnts_dev, CT)
    return SF.StructureFunction2D(sf_type, distance_bins, value_bins, sums, counts)
end

# ---------------------------------------------------------------------------
# Single-pass 2D GPU kernels (eight distance × value joint histograms)
# ---------------------------------------------------------------------------

@inline function _gpu_digitize_general_col(
    x::T,
    edges,
    col::Int,
    n_edges::Int,
) where {T}
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

@inline function _gpu_accumulate_single_pass_2d_pair!(
    output_sums,
    output_counts,
    value_edges,
    bin::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    N_val_edges::Int,
)
    FT = eltype(output_sums)
    vals = SA.SVector(
        du_L2 + du_T2,
        du_L2,
        du_T2,
        du_L * (du_L2 + du_T2),
        du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_general_col(vals[t], value_edges, t, N_val_edges)
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += vals[t]
            @atomic output_counts[t, bin, vbin] += one(eltype(output_counts))
        end
    end
    return nothing
end

KA.@kernel function _sf_single_pass_2d_kernel_linear!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(value_edges),
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    N_val_edges::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {D, FT}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        dist, du_L, du_L2, du_T2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT,
        )
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            _gpu_accumulate_single_pass_2d_pair!(
                output_sums, output_counts, value_edges, bin,
                du_L, zero(du_L), du_L2, du_T2, N_val_edges,
            )
        end
    end
end

KA.@kernel function _sf_single_pass_2d_kernel_log!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(value_edges),
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    N_val_edges::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {D, FT}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        dist, du_L, du_L2, du_T2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT,
        )
        bin = _gpu_digitize_log_spaced(dist, first_edge, last_edge, inv_step, offset, step_val, N_bins)
        if 1 <= bin < N_bins
            _gpu_accumulate_single_pass_2d_pair!(
                output_sums, output_counts, value_edges, bin,
                du_L, zero(du_L), du_L2, du_T2, N_val_edges,
            )
        end
    end
end

KA.@kernel function _sf_single_pass_2d_kernel!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(distance_bins),
    @Const(value_edges),
    N_points::Int,
    ::Val{D},
    N_bins::Int,
    N_val_edges::Int,
) where {D}
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        FT = eltype(x_mat)
        dist, du_L, du_L2, du_T2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), FT,
        )
        bin = _gpu_digitize_general(dist, distance_bins, N_bins)
        if 1 <= bin < N_bins
            _gpu_accumulate_single_pass_2d_pair!(
                output_sums, output_counts, value_edges, bin,
                du_L, zero(du_L), du_L2, du_T2, N_val_edges,
            )
        end
    end
end

const _SinglePass2DValueBins = SFC.SinglePass2DValueBins

include(joinpath(@__DIR__, "gpu", "batch_dispatch.jl"))

function _gpu_run_single_pass_2d!(
    gpu_backend::SF.GPUBackend,
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

    x_dev, u_dev = _stage_sf_device_inputs(backend, x, u, N_dims, N_points; workspace = workspace)

    if workspace === nothing
        val_plan = force_global_atomic ? _gpu_build_value_vector_cols_plan(backend, value_bins) : N_dims == 3 ?
            _gpu_build_value_vector_cols_plan(backend, value_bins) :
            _gpu_build_value_digitize_plan(backend, value_bins)
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
        out_cnts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :single_pass_2d, n_bins; n_val = n_val)
        SFC.reset_histogram!(workspace)
        val_plan = force_global_atomic ? GPUValueVectorCols{FT}(workspace.value_edges_sp2d_dev) : N_dims == 3 ?
            _gpu_build_value_vector_cols_plan(backend, value_bins) :
            workspace.val_plan
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = workspace.out_cnts_dev
        ws = workspace
    end

    _launch_single_pass_2d!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, val_plan, N_points, N_dims, n_dist_edges, n_val_edges;
        workspace = ws,
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
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    kernel! = _sf_single_pass_2d_kernel_linear!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        N_points, Val(2), n_dist_edges, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val;
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
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf_single_pass_2d_kernel_log!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        N_points, Val(2), n_dist_edges, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st;
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
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    if gen_e === nothing
        bins_dev = KA.allocate(backend, FT, n_dist_edges)
        copyto!(bins_dev, edges)
    else
        bins_dev = gen_e
    end
    kernel! = _sf_single_pass_2d_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        bins_dev, value_edges_dev, N_points, Val(2), n_dist_edges, n_val_edges;
        ndrange = (N_points, N_points),
    )
    return nothing
end

"""
    SFC._dispatch_single_pass(::GPUBackend, x, u, distance_bins; workgroup_size=64, kwargs...)

Calculates single-pass structure functions utilizing GPU-accelerated computing.
Device histograms are `UInt32`; `count_eltype` (default `UInt32`) selects the host
count matrix type returned by [`append_helmholtz_rotational_divergent_rows`](@ref).
"""
function SFC._dispatch_single_pass(
    gpu_backend::SF.GPUBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    workgroup_size::Int = 64,
    count_eltype::Type{CT} = UInt32,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    backend = gpu_backend.backend
    FT = promote_type(float(FT1), float(FT2))
    N_dims, N_points = size(x)
    dist_bins = _gpu_normalize_bins(distance_bins)
    n_edges = _gpu_n_edges(distance_bins)
    n_bins = n_edges - 1

    N_dims in (2, 3) ||
        error("GPUExt: single-pass calculation requires N_dims ∈ (2, 3) (got N_dims=$N_dims)")

    x_dev, u_dev = _stage_sf_device_inputs(backend, x, u, N_dims, N_points; workspace = workspace)

    if workspace === nothing
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_bins)
        out_cnts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :single_pass, n_bins)
        SFC.reset_histogram!(workspace)
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = workspace.out_cnts_dev
        ws = workspace
    end

    _launch_single_pass_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, N_points, N_dims, n_edges;
        workspace = ws,
    )
    KA.synchronize(backend)

    sums = Array(out_sums_dev)
    counts = _download_gpu_counts(out_cnts_dev, CT)
    edges_host = _gpu_host_edge_vector(distance_bins)

    return SFC.append_helmholtz_rotational_divergent_rows(sums, counts, edges_host)
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
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = _sp2d_n_val_edges(value_bins) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    sums = zeros(OT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
    counts = zeros(CT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
    return _gpu_run_single_pass_2d!(
        SF.GPUBackend(backend), sums, counts, x, u, distance_bins, value_bins;
        workgroup_size = workgroup_size, workspace = workspace, kwargs...,
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
    kwargs...,
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
        SF.GPUBackend(backend), sums_3d, counts_3d, x, u, distance_bins, value_bins;
        workgroup_size = workgroup_size, workspace = workspace, kwargs...,
    )
    return sums_3d, counts_3d
end

"""
    gpu_calculate_structure_function_slices!(sums, counts, sf_type, backend, x, u, distance_bins; workspace=nothing, ...)

Batch 1D structure functions over the third dimension of `x`, `u` with layout
`(N_dims, N_points, T)`. Host outputs `sums`, `counts` have shape `(NB, T)`.
Uploads `x`, `u` once and synchronizes the backend once after the last launch.
"""
function SFC.gpu_calculate_structure_function_slices!(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT, 3},
    u::AbstractArray{FT, 3},
    distance_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    kwargs...,
) where {OT, CT, FT}
    N_dims, N_points, T = size(x)
    dist_bins = _gpu_batch_linear_distance_bins(distance_bins)
    lbe = dist_bins
    NB = length(lbe.edges) - 1
    size(sums) == (NB, T) ||
        throw(DimensionMismatch("sums must have shape ($NB, $T); got $(size(sums))"))
    size(counts) == (NB, T) ||
        throw(DimensionMismatch("counts must have shape ($NB, $T); got $(size(counts))"))
    N_dims == 2 ||
        error("GPUExt: slice batch requires N_dims=2 (got N_dims=$N_dims)")
    sums_dev = KA.adapt(backend, zeros(FT, NB, T))
    counts_dev = KA.adapt(backend, zeros(UInt32, NB, T))
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = false)
    _launch_batch_varying_x_sf!(
        backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N_points, T, lbe,
    )
    copy!(sums, Array(sums_dev))
    copy!(counts, Array(counts_dev))
    return nothing
end

"""
    gpu_calculate_structure_function_2d_slices!(sums, counts, sf_type, backend, x, u, distance_bins, value_bins; workspace=nothing, ...)

Batch 2D joint histograms over `(N_dims, N_points, T)`; outputs `(n_dist, n_val, T)`.
"""
function SFC.gpu_calculate_structure_function_2d_slices!(
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
    kwargs...,
) where {OT, CT, FT}
    N_dims, N_points, T = size(x)
    n_dist = length(distance_bins) - 1
    n_val = length(value_bins) - 1
    size(sums) == (n_dist, n_val, T) ||
        throw(DimensionMismatch("sums must have shape ($n_dist, $n_val, $T); got $(size(sums))"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums))"))

    sums_dev = KA.adapt(backend, zeros(FT, n_dist, n_val, T))
    counts_dev = KA.adapt(backend, zeros(UInt32, n_dist, n_val, T))
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = false)
    _launch_batch_varying_x_joint2d!(
        backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N_points, T,
        distance_bins, value_bins, n_dist, n_val; workspace = workspace,
    )
    copy!(sums, Array(sums_dev))
    copy!(counts, Array(counts_dev))
    return nothing
end

"""
    gpu_calculate_structure_functions_single_pass_slices!(sums, counts, backend, x, u, distance_bins; workspace=nothing, ...)

Batch six invariant 1D distance histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, T)`.
"""
function SFC.gpu_calculate_structure_functions_single_pass_slices!(
    sums::AbstractArray{OT, 3},
    counts::AbstractArray{CT, 3},
    backend::KA.Backend,
    x::AbstractArray{FT, 3},
    u::AbstractArray{FT, 3},
    distance_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    kwargs...,
) where {OT, CT, FT}
    N_dims, N_points, T = size(x)
    lbe = _gpu_batch_linear_distance_bins(distance_bins)
    n_bins = length(lbe.edges) - 1
    size(sums) == (SFC.SINGLE_PASS_N, n_bins, T) ||
        throw(DimensionMismatch("sums must have shape ($(SFC.SINGLE_PASS_N), $n_bins, $T); got $(size(sums))"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums))"))
    N_dims == 2 ||
        error("GPUExt: single-pass slice batch requires N_dims=2 (got N_dims=$N_dims)")
    sums_dev = KA.adapt(backend, zeros(FT, SF_GPU_SINGLE_PASS_N, n_bins, T))
    counts_dev = KA.adapt(backend, zeros(UInt32, SF_GPU_SINGLE_PASS_N, n_bins, T))
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = false)
    _launch_batch_varying_x_sp1d!(backend, sums_dev, counts_dev, x_dev, u_dev, N_points, T, lbe)
    copy!(sums, Array(sums_dev))
    copy!(counts, Array(counts_dev))
    return nothing
end

"""
    gpu_calculate_structure_functions_single_pass_2d_slices!(sums, counts, backend, x, u, distance_bins, value_bins; workspace=nothing, ...)

Batch six invariant distance × value joint histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, n_val, T)`.
"""
function SFC.gpu_calculate_structure_functions_single_pass_2d_slices!(
    sums::AbstractArray{OT, 4},
    counts::AbstractArray{CT, 4},
    backend::KA.Backend,
    x::AbstractArray{FT, 3},
    u::AbstractArray{FT, 3},
    distance_bins::AbstractVector{FT},
    value_bins::_SinglePass2DValueBins;
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    kwargs...,
) where {OT, CT, FT}
    N_dims, N_points, T = size(x)
    lbe = _gpu_batch_linear_distance_bins(distance_bins)
    n_bins = length(lbe.edges) - 1
    n_val = size(sums, 3)
    SFC._validate_value_bins!(value_bins, n_val)
    size(sums) == (SFC.SINGLE_PASS_N, n_bins, n_val, T) ||
        throw(DimensionMismatch("sums must have shape ($(SFC.SINGLE_PASS_N), $n_bins, $n_val, $T); got $(size(sums))"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums))"))
    N_dims == 2 ||
        error("GPUExt: SP2D slice batch requires N_dims=2 (got N_dims=$N_dims)")
    val_plan = _gpu_build_value_digitize_plan(backend, value_bins)
    sums_dev = KA.adapt(backend, zeros(FT, SF_GPU_SINGLE_PASS_N, n_bins, n_val, T))
    counts_dev = KA.adapt(backend, zeros(UInt32, SF_GPU_SINGLE_PASS_N, n_bins, n_val, T))
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = false)
    _launch_batch_varying_x_sp2d!(
        backend, sums_dev, counts_dev, x_dev, u_dev, N_points, T, lbe, val_plan, n_bins, n_val,
    )
    copy!(sums, Array(sums_dev))
    copy!(counts, Array(counts_dev))
    return nothing
end

"""
    SFC._dispatch_single_pass!(::GPUBackend, sums, counts, x, u, distance_bins; ...)

In-place six-invariant-type single-pass distance histograms on GPU (no Helmholtz row append).
"""
function SFC._dispatch_single_pass!(
    gpu_backend::SF.GPUBackend,
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
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

    x_dev, u_dev = _stage_sf_device_inputs(backend, x, u, N_dims, N_points; workspace = workspace)

    if workspace === nothing
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_bins)
        out_cnts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :single_pass, n_bins)
        SFC.reset_histogram!(workspace)
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = workspace.out_cnts_dev
        ws = workspace
    end

    _launch_single_pass_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, N_points, N_dims, n_edges;
        workspace = ws,
    )
    KA.synchronize(backend)

    tmp_s = Array(out_sums_dev)
    copyto!(sums, OT === eltype(tmp_s) ? tmp_s : OT.(tmp_s))
    tmp_c = Array(out_cnts_dev)
    copyto!(counts, CT === UInt32 ? tmp_c : CT.(tmp_c))
    return sums, counts
end

function SFC.gpu_calculate_structure_function_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT},
    ::Val{RSAC};
    kwargs...,
) where {FT, RSAC}
    return _gpu_calculate_structure_function_batch(
        sf_type, backend, x, u, distance_bins, Val(RSAC); kwargs...,
    )
end

function SFC.gpu_calculate_structure_function_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT};
    return_sums_and_counts::Bool = false,
    kwargs...,
) where {FT}
    return gpu_calculate_structure_function_batch(
        sf_type, backend, x, u, distance_bins, Val(return_sums_and_counts); kwargs...,
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
    gpu_backend::SF.GPUBackend,
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
    gpu_backend::SF.GPUBackend,
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
    gpu_backend::SF.GPUBackend,
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
    gpu_backend::SF.GPUBackend,
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
    gpu_backend::SF.GPUBackend,
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
    gpu_backend::SF.GPUBackend,
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
