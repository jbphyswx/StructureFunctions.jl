"""
GPU-accelerated structure function kernels using KernelAbstractions.jl.

This extension is loaded automatically when `KernelAbstractions` is loaded by the user.
The `gpu_calculate_structure_function` entry point accepts any KA-compatible backend:
  - `KernelAbstractions.CPU()` – for CPU-parallel testing / parity verification
  - `CUDABackend()` from CUDA.jl – for NVIDIA GPU acceleration
  - `ROCBackend()` from AMDGPU.jl – for AMD GPU acceleration

For `N_dims ∈ {2,3}` the fast path uses tiled128 pair blocks with block-local
`UInt32` histograms. Joint 2D SF
(`calculate_structure_function` with `value_bins`) uses the same tiled schedule
when `n_dist × n_val ≤ SF_GPU_MAX_2D_HIST`. Six-invariant-type single-pass 1D uses
tiled128 block-local `(6, NB)` histograms when `NB ≤ SF_GPU_MAX_BINS`; six-invariant-type
single-pass 2D uses tiled128 pair traversal with **HTP-EJ** when `n_dist ≤ SF_GPU_MAX_BINS`:

- **On-chip** (`:shared`, `:typeplane`): shared histogram during the pair loop,
  block-end `@atomic` flush into final output (the joint 2D pattern), with no private partition and
  no merge.
- **Direct** (`:direct`): block-private global atomics during one pair pass, then a merge kernel.

Pass `force_global_atomic=true` to bypass HTP-EJ and use the global-atomic path.

## Count types on GPU

A device count histogram accumulates in `UInt32` while an unweighted sweep's worst-case pair count
fits it, and in the requested count type `CT` otherwise; results carry `CT` and stay on the device.

Kernels digitize with the host's `digitize_plan` of the bins, so every bin type bins on a device
exactly as on the CPU.

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
    HelperFunctions as SFH, StructureFunctionTypes as SFT
using StructureFunctions.Calculations: GPUSFWorkspace, GPUSFLazyBuffers,
    _partition_n_tile_blocks, _ws_float_type,
    FullUpperTriangle, TilePairWorkList, tile_for, n_pair_blocks, schedule_for

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

"""True when the 2D joint histogram and the four coordinate tiles of width `D` fit the tiled128
block-local path. A width whose tiles do not fit takes the global-atomic kernel, which stages
nothing."""
@inline function _gpu_joint_2d_tiled_eligible(n_dist::Int, n_val::Int, D::Int,
                                              ::Type{FT}) where {FT}
    return n_dist <= SF_GPU_MAX_BINS &&
           n_val <= SF_GPU_MAX_BINS &&
           n_dist * n_val <= SF_GPU_MAX_2D_HIST &&
           4 * D * SF_GPU_TILE * sizeof(FT) <= 44 * 1024
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

include(joinpath(@__DIR__, "gpu", "sp2d_accumulation_strategy.jl"))
include(joinpath(@__DIR__, "gpu", "joint2d_shared_memory.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_1d_single_pass.jl"))
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
    xk, uk, perm = _gpu_cull_and_permute!(
        workspace, backend, xk, uk, geom, distance_bins, culling, N_points, x,
    )
    x_dev, u_dev = _stage_sf_device_inputs(backend, xk, uk, W, F, N_points; workspace = workspace)
    # A cull reorders the points, so the weights travel with the coordinates they belong to.
    w_dev = _permuted_weights(backend, weights, perm)
    return geom, x_dev, u_dev, w_dev
end

"""Reorder pair weights by a cull permutation; `NoWeights()` and an unpermuted call are no-ops."""
@inline _permuted_weights(backend, w::SFC.NoWeights, _) = w
@inline _permuted_weights(backend, w::AbstractVector, ::Nothing) = _sf_weights_to_device(backend, w)
function _permuted_weights(backend, w::AbstractVector, perm::AbstractVector{<:Integer})
    w_device = _sf_weights_to_device(backend, w)
    perm_device = _array_on_backend(perm, backend) ? perm : KA.adapt(backend, perm)
    return w_device[perm_device]
end

"""
    _gpu_cull_and_permute!(workspace, backend, xk, uk, geom, distance_bins, culling, N_points)

Sort the kernel coordinates into a cull grid, publish the memo this call culls with as
`workspace.lazy.active`, and return the reordered `(xk, uk)` for staging. The slot is cleared first,
so a launcher never sees a previous call's grid; each launcher takes its tile-pair list from the
active memo through `schedule_for` at its own tile size. The memo is kept as `workspace.lazy.cull`
and reused while the coordinate-array identity, cutoff and policy match, so a repeated call on the
same prepared points pays only the field gather. Call `refresh!(workspace)` after mutating coordinates
in place. Host coordinates use the shared CPU grid builder. A device backend may supply
`gpu_device_cull_grid`, which keeps coordinates and the permutation on the device and transfers only
compact occupied-cell metadata for tile schedule construction. Culling needs a workspace to retain
the prepared grid and schedules.
"""
function _gpu_cull_and_permute!(workspace, backend, xk, uk, geom, distance_bins, culling,
                                N_points, source=xk)
    workspace === nothing && return _gpu_cull_no_workspace(culling, xk, uk)
    workspace.lazy.active = nothing
    (SFC._cull_enabled(culling) && distance_bins !== nothing) || return xk, uk, nothing
    cutoff = SFC.cull_cutoff_for(geom, distance_bins, culling)
    cutoff === nothing && return xk, uk, nothing
    memo = workspace.lazy.cull
    if SFC._cull_memo_hit(memo, source, cutoff, culling)
        memo isa SFC.GPUNoCullMemo && return xk, uk, nothing
        workspace.lazy.active = memo
        return memo.x_sorted, uk[:, memo.grid.perm], memo.grid.perm
    end
    grid, xs, us = if xk isa Array && uk isa Array
        SFC.cull_sorted_matrices(xk, uk, geom, distance_bins, culling)
    else
        device_grid = SFC.gpu_device_cull_grid(backend, xk, cutoff, culling)
        device_grid === nothing ? _gpu_cull_device_inputs(culling, xk, uk) :
            (device_grid, xk[:, device_grid.perm], uk[:, device_grid.perm])
    end
    if grid === nothing
        workspace.lazy.cull = SFC.GPUNoCullMemo(source, cutoff, culling)
        return xk, uk, nothing
    end
    memo = SFC.GPUCullMemo(source, copy(xk), cutoff, culling, grid, xs,
                           v -> KA.adapt(backend, v), Dict{Int, SFC.TilePairWorkList}())
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
    "the selected GPU backend does not implement device-resident culling for these coordinates; " *
    "pass culling = AutoCulling() / NoCulling(), or use a backend with device sort and compaction.",
))

# ---------------------------------------------------------------------------
# Public API – extends the stub declared in Calculations.jl
# ---------------------------------------------------------------------------

"""
    gpu_calculate_structure_function(sf_type, backend, x_mat, u_mat, distance_bins, CT; workspace, distance_metric, culling, weights)

Compute structure functions on `backend` (any KernelAbstractions backend).

# Arguments
- `backend`: e.g. `KernelAbstractions.CPU()`, `CUDA.CUDABackend()`, etc.
- `x_mat`: `(N_dims, N_points)` matrix of spatial positions, on the host or on `backend`.
- `u_mat`: `(N_dims, N_points)` matrix of velocity components.
- `distance_bins`: bin *edges* (`AbstractVector{FT}` with `FT === eltype(x_mat)` — same
  element type as `x_mat`/`u_mat`; use `collect(FT, edges)` when building from a
  different-precision template), or any [`AbstractBinEdges`](@ref), binned exactly as on the CPU.
- `sf_type`: any `AbstractPairwiseStructureFunctionType`.
- `CT`: result count type; weighted counts require a floating type.

`verbose` and `show_progress` are accepted, as on the CPU backends, and have no effect here.

# Returns
A raw `StructureFunctionSumsAndCounts` accumulator on `backend`, with the input
floating-point sum type and count type `CT`. Workspace results own their
buffers. Use `to_host(result)` for a synchronized host copy.
"""
function SFC.gpu_calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT},
    ::Type{CT};
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT, CT}
    out_dev, cnt_dev, edges_host = _launch_gpu_structure_function!(
        sf_type, backend, x_mat, u_mat, distance_bins, CT;
        workspace, distance_metric, culling, weights, synchronize = false,
    )
    output, counts = _owned_gpu_results(out_dev, cnt_dev, CT; workspace)
    synchronize && KA.synchronize(backend)
    return SF.StructureFunctionSumsAndCounts(sf_type, edges_host, output, counts)
end

_tiled_launch_params(N_points::Int) = _tiled_launch_params(N_points, nothing)

# The prologue sets `workspace.lazy.active` on every call, so the schedule is always this call's.
function _tiled_launch_params(N_points::Int, workspace::Union{GPUSFWorkspace, Nothing})
    sched = schedule_for(_active_cull(workspace), N_points, SF_GPU_TILE)
    n_tile_blocks = n_pair_blocks(sched)
    ws = SF_GPU_TILED_WS
    return sched, n_tile_blocks, ws, n_tile_blocks * ws
end

"""
    _launch_sf_kernel!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dig,
                       N_points, N_dims, N_bins, geom; workspace, weights)

Run a point list's distance histogram as the unified tiled kernel over a single slice: `out_dev`
and `cnt_dev` are `(NB,)`, viewed as the `(NMOM, NB, B)` the kernel writes with `NMOM = B = 1`.
`dig` is the device digitizer of the distance bins.
"""
function _launch_sf_kernel!(
    backend::KA.Backend,
    out_dev,
    cnt_dev,
    x_dev,
    u_dev,
    sf_type,
    dig,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    NB = N_bins - 1
    D = N_dims
    if weights isa SFC.NoWeights
        handled = SFC.gpu_fast_launch_1d_batch!(
            backend,
            reshape(out_dev, 1, NB, 1),
            reshape(cnt_dev, 1, NB, 1),
            x_dev,
            reshape(u_dev, D, N_points, 1),
            sf_type,
            dig,
            N_points,
            NB,
            1,
            D,
            1,
            true,
            geom,
            _active_cull(workspace),
        )
        handled && return nothing
    end
    go = Dv -> _launch_sf_tiled_1d_varying!(
        backend,
        reshape(out_dev, 1, NB, 1),
        reshape(cnt_dev, 1, NB, 1),
        reshape(x_dev, D, N_points, 1),
        reshape(u_dev, D, N_points, 1),
        sf_type, dig, N_points, NB, 1,
        Dv, Val(1), geom;
        weights = weights, workspace = workspace,
    )
    # 2 and 3 are the widths the package compiles ahead of time; any other is one more kernel
    # instantiation, paid once on its first launch.
    D == 2 ? go(Val(2)) : D == 3 ? go(Val(3)) : go(Val(D))
    return nothing
end

"""
    _launch_gpu_structure_function_core!(sf_type, backend, x_dev, u_dev, dig, N_points, N_dims, N_bins; out_dev, cnt_dev, workspace=nothing, synchronize=true)

Launch the tiled GPU structure-function kernel into pre-allocated device buffers (no allocation).
"""
function _launch_gpu_structure_function_core!(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_dev,
    u_dev,
    dig,
    N_points::Int,
    N_dims::Int,
    N_bins::Int;
    out_dev,
    cnt_dev,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    weights = SFC.NoWeights(),
    geom,
)
    _launch_sf_kernel!(
        backend,
        out_dev, cnt_dev, x_dev, u_dev,
        sf_type, dig, N_points, N_dims, N_bins, geom;
        workspace = workspace, weights = weights,
    )
    synchronize && KA.synchronize(backend)
    return out_dev, cnt_dev
end

"""
    _launch_gpu_structure_function!(sf_type, backend, x_mat, u_mat, distance_bins, CT; workspace=nothing, synchronize=true, ...)

Run the tiled GPU structure-function kernel and return device-resident `(out_dev, cnt_dev)`.
`distance_bins` must be a host edge vector (same convention as CPU).
Pass `workspace` to reuse device histogram buffers; default allocates fresh buffers each call.
"""
function _launch_gpu_structure_function!(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT},
    ::Type{CT};
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    destination_sums = nothing,
    destination_counts = nothing,
) where {FT, CT}
    N_dims, N_points = size(x_mat)
    N_bins = length(distance_bins)
    NB = N_bins - 1

    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x_mat, u_mat, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    # The tiled kernel variant is chosen by the coordinate width it will index,
    # which is the converted width, not the width the caller passed.
    N_dims = SFC._val_int(SFH.coordinate_width(geom))

    # The workspace's count buffer is `UInt32`; a weighted sweep accumulates a pair mass, so it
    # takes its own buffer while the rest of the workspace (staging, cull memo) is still reused.
    n_pairs = _sf_worst_case_pairs(N_points)
    fast_count_type = SFC.gpu_fast_1d_count_type(
        backend, w_dev, CT, n_pairs, N_dims, NB, 1,
    )
    CNT = fast_count_type === nothing ? _sf_count_type(w_dev, CT, n_pairs) : fast_count_type
    use_destination = destination_sums !== nothing && destination_counts !== nothing &&
                      size(destination_sums) == (NB,) && size(destination_counts) == (NB,) &&
                      eltype(destination_sums) === FT && eltype(destination_counts) === CNT
    workspace === nothing ||
        _validate_gpu_workspace!(workspace, backend, :sf1d, NB; distance_bins, sum_type=FT)
    if use_destination
        out_dev = destination_sums
        cnt_dev = destination_counts
        ws = workspace
    elseif isnothing(workspace)
        out_dev = KA.zeros(backend, FT, NB)
        cnt_dev = KA.zeros(backend, CNT, NB)
        ws = nothing
    else
        SFC.reset_histogram!(workspace)
        out_dev = workspace.out_sums_dev
        cnt_dev = CNT === eltype(workspace.out_cnts_dev) ? workspace.out_cnts_dev :
                  _workspace_snapshot_counts!(workspace, backend, CNT, (NB,))
        ws = workspace
    end

    _launch_gpu_structure_function_core!(
        sf_type, backend, x_dev, u_dev, _dist_digitizer(workspace, backend, distance_bins, Val(:sf1d)),
        N_points, N_dims, N_bins;
        out_dev = out_dev, cnt_dev = cnt_dev, workspace = ws,
        synchronize = synchronize, weights = w_dev, geom = geom,
    )
    return out_dev, cnt_dev, distance_bins
end

"""Detach allocating results from reusable scratch and convert counts on-device."""
function _owned_gpu_results(sums, counts, ::Type{CT}; workspace=nothing) where {CT}
    owned_sums = workspace === nothing ? sums : copy(sums)
    owned_counts = eltype(counts) === CT ? (workspace === nothing ? counts : copy(counts)) : CT.(counts)
    return owned_sums, owned_counts
end

"""Throw unless mutating GPU outputs have `shape`, reside on `backend` and do not alias workspace scratch."""
function _check_gpu_outputs(sums, counts, backend, shape; workspace = nothing)
    size(sums) == size(counts) == shape || throw(DimensionMismatch("output buffers must have shape $shape"))
    _array_on_backend(sums, backend) && _array_on_backend(counts, backend) ||
        throw(ArgumentError("mutating GPU outputs must reside on the selected backend; use to_host after computation"))
    if workspace !== nothing
        Base.mightalias(sums, workspace.out_sums_dev) && throw(ArgumentError("output must not alias workspace scratch"))
        Base.mightalias(counts, workspace.out_cnts_dev) && throw(ArgumentError("counts must not alias workspace scratch"))
        cached = workspace.lazy.batch
        if cached !== nothing
            (Base.mightalias(sums, cached.sums) || Base.mightalias(counts, cached.counts)) &&
                throw(ArgumentError("outputs must not alias batch workspace scratch"))
        end
    end
    return nothing
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
    count_scratch = if workspace !== nothing && eltype(workspace.host_counts_scratch) === eltype(cnt_dev)
        copyto!(workspace.host_counts_scratch, cnt_dev)
    else
        Array(cnt_dev)
    end
    if CT <: Integer
        all(c -> c >= 0 && c <= typemax(CT), count_scratch) || throw(ArgumentError("device counts exceed the output count type"))
        all(eachindex(output_counts, count_scratch)) do k
            count_scratch[k] <= typemax(CT) - output_counts[k]
        end || throw(ArgumentError("accumulated counts exceed the output count type"))
    end
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
    output_counts .+= count_scratch
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
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, FT}
    _check_gpu_outputs(output_sums, output_counts, backend, (length(distance_bins)-1,); workspace)
    out_dev, cnt_dev, _ = _launch_gpu_structure_function!(
        sf_type, backend, x_mat, u_mat, distance_bins, CT;
        workspace, distance_metric, culling, weights,
        destination_sums = output_sums,
        destination_counts = output_counts,
    )
    out_dev === output_sums || (output_sums .+= out_dev)
    cnt_dev === output_counts || (output_counts .+= cnt_dev)
    KA.synchronize(backend)
    return nothing
end




"""True when six-invariant-type 1D single-pass can use tiled128 block-local histograms."""
@inline _gpu_single_pass_tiled_eligible(n_bins::Int) = n_bins <= SF_GPU_MAX_BINS

"""True when six-invariant-type 2D single-pass can use HTP-EJ tiled128 (`n_dist ≤ 64`, typed dist bins)."""
@inline _gpu_single_pass_2d_tiled_eligible(n_dist::Int) = n_dist <= SF_GPU_MAX_BINS

function _launch_single_pass_tiled_kernel!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dig,
    N_points::Int,
    n_edges::Int,
    NB::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points, workspace)
    kernel! = _sf6_single_pass_kernel_tiled128_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, dig,
        N_points, n_edges, NB,
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
    ::Val{W},
    ::Type{FT},
    geom,
) where {W, FT}
    X1 = _gpu_ld_col(x_mat, i, SFH.coordinate_width(geom), FT)
    X2 = _gpu_ld_col(x_mat, j, SFH.coordinate_width(geom), FT)
    U1 = _gpu_ld_col(u_mat, i, Val(W), FT)
    U2 = _gpu_ld_col(u_mat, j, Val(W), FT)
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

KA.@kernel unsafe_indices=true function _sf_single_pass_kernel!(
    output_sums,                 # (6, N_bins - 1)
    output_counts,               # (6, N_bins - 1)
    @Const(x_mat),
    @Const(u_mat),
    dig,
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
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), eltype(x_mat), geom,
        )
        bin = SFH.digitize(dist, dig)
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
    backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, dig,
    N_points::Int, N_dims::Int, NB::Int, geom, cull,
)
    return SFC.gpu_fast_launch_1d_batch!(
        backend,
        reshape(out_sums_dev, SF_GPU_SINGLE_PASS_N, NB, 1),
        reshape(out_cnts_dev, SF_GPU_SINGLE_PASS_N, NB, 1),
        x_dev, reshape(u_dev, N_dims, N_points, 1),
        nothing, dig,
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
    dig,
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
                          dig, N_points, N_dims, NB, geom, _active_cull(workspace)) && return nothing
    if N_dims == 2 && _gpu_single_pass_tiled_eligible(NB)
        return _launch_single_pass_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dig, N_points, n_edges, NB, geom; workspace = workspace, weights = weights,
        )
    end
    kernel! = _sf_single_pass_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, dig, _sf_weights_to_device(backend, weights),
        N_points, Val(N_dims), n_edges, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Joint 2D SF kernels (one sf_type, distance × value histogram)
# ---------------------------------------------------------------------------

KA.@kernel unsafe_indices=true function _sf_joint_2d_kernel!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    wts,
    ddig,
    vdig,
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
        dbin = SFH.digitize(dist, ddig)
        if ok && 1 <= dbin < N_dist_bins
            dU, r̂ = SFH.pair_increments(geom, frame, dist, X1, X2, U1, U2)
            val = sf_type(dU, r̂)
            akey = SFC.pair_axis_key(second_axis, val, X1, X2, dist)
            vbin = SFH.digitize(akey, vdig)
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
    backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type, ddig, vdig,
    N_points::Int, n_dist::Int, n_val::Int, geom, cull,
)
    D = size(x_dev, 1)
    return SFC.gpu_fast_launch_2d_batch!(
        backend,
        reshape(out_sums_dev, 1, n_dist, n_val, 1),
        reshape(out_cnts_dev, 1, n_dist, n_val, 1),
        x_dev, reshape(u_dev, D, N_points, 1),
        sf_type, ddig, vdig,
        N_points, n_dist, n_val, 1, D, 1, true, geom, cull,
    )
end

"""Launch one joint distance × value histogram with device digitizers `ddig` and `vdig`."""
function _launch_joint_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    sf_type,
    ddig,
    vdig,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    W = SFC._val_int(SFH.coordinate_width(geom))
    weights isa SFC.NoWeights && second_axis isa SFC.InvariantValueAxis &&
        _joint2d_try_fast_batch!(backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
                             ddig, vdig, N_points, n_dist, n_val, geom,
                             _active_cull(workspace)) && return nothing
    if _gpu_joint_2d_tiled_eligible(n_dist, n_val, W, eltype(out_sums_dev))
        return _launch_joint_2d_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            sf_type, ddig, vdig, N_points, n_dist_edges, n_val_edges, n_dist, n_val, geom;
            workspace = workspace, weights = weights, second_axis = second_axis,
        )
    end
    kernel! = _sf_joint_2d_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _sf_weights_to_device(backend, weights), ddig, vdig, sf_type,
        N_points, n_dist_edges, n_val_edges, geom, second_axis, Val(W);
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
    value_bins,
    ::Type{CT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
) where {CT}
    FT = promote_type(eltype(x_mat), eltype(u_mat), eltype(distance_bins), eltype(value_bins))
    N_dims, N_points = size(x_mat)
    # Only the point count has to agree: on a shell a point takes two coordinates while the velocity
    # may carry a third, radial, component.
    size(u_mat, 2) == N_points ||
        throw(DimensionMismatch(
            "x_mat and u_mat must share the point count; got $(size(x_mat)) and $(size(u_mat))",
        ))

    n_dist_edges = length(distance_bins)
    n_val_edges = length(value_bins)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    workspace === nothing || _validate_gpu_workspace!(
        workspace, backend, :joint2d, n_dist; n_val, distance_bins, value_bins, sum_type = FT)

    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x_mat, u_mat, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)

    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    if isnothing(workspace)
        out_sums_dev = KA.zeros(backend, FT, n_dist, n_val)
        out_cnts_dev = KA.zeros(backend, CNT, n_dist, n_val)
    else
        SFC.reset_histogram!(workspace)
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = CNT === eltype(workspace.out_cnts_dev) ? workspace.out_cnts_dev :
            KA.zeros(backend, CNT, n_dist, n_val)
    end

    _launch_joint_2d_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
        _dist_digitizer(workspace, backend, distance_bins, Val(:joint2d)),
        _value_digitizer(workspace, backend, value_bins),
        N_points, n_dist_edges, n_val_edges, geom;
        workspace = workspace, weights = w_dev, second_axis = second_axis,
    )
    synchronize && KA.synchronize(backend)
    return out_sums_dev, out_cnts_dev
end

"""
    gpu_calculate_structure_function_2d(sf_type, backend, x_mat, u_mat, distance_bins, value_bins, CT; kwargs...)

Compute one 2D joint histogram (distance × SF value) for `sf_type` on `backend`.
Returns [`StructureFunction2DSumsAndCounts`](@ref) with the same flat edge vectors passed in.

Uses tiled128 block-local histograms when ``n_dist × n_val ≤ SF_GPU_MAX_2D_HIST``
(with each axis ``≤ SF_GPU_MAX_BINS``); otherwise falls back to ``(N_points, N_points)``
global-atomic pair kernels. Default compile-time shared histogram width is exact
``n_dist × n_val``; override on [`GPUSFWorkspace`](@ref) via `joint2d_compile_cells`
(see [`joint2d_smem_max`](@ref), [`joint2d_smem_align256`](@ref)). Results remain
on the selected backend and own their buffers, with counts of type `CT`. Use `to_host` for host
conversion.
"""
function SFC.gpu_calculate_structure_function_2d(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::AbstractVector{FT4},
    ::Type{CT};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, FT4 <: Number, CT}
    if ndims(u) >= 3
        return _gpu_calculate_structure_function_2d_batch(
            sf_type, backend, x, u, distance_bins, value_bins, CT; kwargs...,
        )
    end
    return _gpu_calculate_structure_function_2d_snapshot(
        sf_type, backend, x, u, distance_bins, value_bins, CT; kwargs...,
    )
end

function _gpu_calculate_structure_function_2d_snapshot(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT1},
    u_mat::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::AbstractVector{FT4},
    ::Type{CT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, FT4 <: Number, CT}
    SFC._require_value_axis(
        second_axis, SFH.pair_geometry_for(distance_metric, Val(size(u_mat, 1))),
    )
    out_sums_dev, out_cnts_dev = _launch_gpu_joint2d!(
        sf_type, backend, x_mat, u_mat, distance_bins, value_bins, CT;
        workgroup_size, workspace, distance_metric, culling, weights, second_axis,
    )
    sums, counts = _owned_gpu_results(out_sums_dev, out_cnts_dev, CT; workspace)
    KA.synchronize(backend)
    return SF.StructureFunction2DSumsAndCounts(sf_type, distance_bins, value_bins, sums, counts)
end

function SFC.gpu_calculate_structure_function_2d!(sums, counts, sf, backend::KA.Backend,
        x::AbstractMatrix, u::AbstractMatrix, distance_bins, value_bins;
        workspace=nothing, weights=SFC.NoWeights(), verbose=false, show_progress=false, kwargs...)
    shape = (length(distance_bins)-1, length(value_bins)-1)
    _check_gpu_outputs(sums, counts, backend, shape; workspace)
    ds, dc = _launch_gpu_joint2d!(sf, backend, x, u, distance_bins, value_bins, eltype(counts);
        workspace, weights, kwargs...)
    sums .+= ds
    counts .+= dc
    KA.synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Single-pass 2D GPU kernels (eight distance × value joint histograms)
# ---------------------------------------------------------------------------


@inline function _gpu_accumulate_single_pass_2d_pair!(
    output_sums,
    output_counts,
    vplan,
    bin::Int,
    du_L,
    du_n2,
    N_val_edges::Int,
    w,
)
    vals = SA.SVector(SFC.single_pass_invariants(du_L, du_n2))
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _sf_value_bin(vplan, vals[t], t)
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += w * vals[t]
            @atomic output_counts[t, bin, vbin] += convert(eltype(output_counts), w)
        end
    end
    return nothing
end

KA.@kernel unsafe_indices=true function _sf_single_pass_2d_kernel!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    ddig,
    vplan,
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
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(
            x_mat, u_mat, i, j, Val(D), eltype(x_mat), geom,
        )
        bin = SFH.digitize(dist, ddig)
        if ok && 1 <= bin < N_bins
            _gpu_accumulate_single_pass_2d_pair!(
                output_sums, output_counts, vplan, bin,
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

    n_dist_edges = length(distance_bins)
    n_bins = n_dist_edges - 1
    n_val = size(sums_3d, 3)
    size(sums_3d) == (SF_GPU_SINGLE_PASS_N, n_bins, n_val) ||
        throw(DimensionMismatch("sums must have shape ($SF_GPU_SINGLE_PASS_N, n_bins, n_val); got $(size(sums_3d))"))
    size(counts_3d) == size(sums_3d) ||
        throw(DimensionMismatch("counts and sums must have the same shape"))
    _check_gpu_outputs(sums_3d, counts_3d, backend, (SF_GPU_SINGLE_PASS_N, n_bins, n_val); workspace)
    SFC._validate_value_bins!(value_bins, n_val)
    workspace === nothing || _validate_gpu_workspace!(
        workspace, backend, :single_pass_2d, n_bins; n_val, distance_bins, value_bins, sum_type = FT)

    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x, u, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    # The tiled kernel variant is chosen by the coordinate width it will index,
    # which is the converted width, not the width the caller passed.
    N_dims = SFC._val_int(SFH.coordinate_width(geom))
    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    if isnothing(workspace)
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
        out_cnts_dev = KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
    else
        SFC.reset_histogram!(workspace)
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = CNT === eltype(workspace.out_cnts_dev) ? workspace.out_cnts_dev :
                       KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
    end

    _launch_single_pass_2d!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _dist_digitizer(workspace, backend, distance_bins, Val(:single_pass_2d)),
        _value_digitizer(workspace, backend, value_bins),
        N_points, N_dims, n_dist_edges, _n_value_edges(value_bins), geom;
        workspace = workspace, weights = w_dev,
        force_global_atomic = force_global_atomic,
    )
    synchronize && KA.synchronize(backend)

    sums_3d .+= out_sums_dev
    counts_3d .+= out_cnts_dev
    synchronize && KA.synchronize(backend)
    return sums_3d, counts_3d
end


"""
    SFC._dispatch_single_pass(::CB.AbstractGPUBackend, x, u, distance_bins, CT; workgroup_size=64, kwargs...)

The six single-pass invariants of a point list on a device: the raw `(6, n_bins)` sums and counts of
type `CT`, device-resident.
"""
function SFC._dispatch_single_pass(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    backend = gpu_backend.backend
    FT = promote_type(float(FT1), float(FT2))
    N_dims, N_points = size(x)
    n_edges = length(distance_bins)
    n_bins = n_edges - 1

    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x, u, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    # The tiled kernel variant is chosen by the coordinate width it will index,
    # which is the converted width, not the width the caller passed.
    N_dims = SFC._val_int(SFH.coordinate_width(geom))

    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    if isnothing(workspace)
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_bins)
        out_cnts_dev = KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :single_pass, n_bins; distance_bins, sum_type=FT)
        SFC.reset_histogram!(workspace)
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = CNT === eltype(workspace.out_cnts_dev) ? workspace.out_cnts_dev :
                       KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = workspace
    end

    _launch_single_pass_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _dist_digitizer(ws, backend, distance_bins, Val(:single_pass)), N_points, N_dims, n_edges,
        geom;
        workspace = ws, weights = w_dev,
    )
    KA.synchronize(backend)

    sums, counts = _owned_gpu_results(out_sums_dev, out_cnts_dev, CT; workspace)

    return (sums = sums, counts = counts)  # raw 6-row; public wrapper adds Helmholtz once
end

"""
    SFC.gpu_calculate_structure_functions_single_pass_2d(backend, x, u, distance_bins, value_bins, CT; ...)

Six invariant distance × value joint histograms in one GPU pair pass, with counts of type `CT`. Pass
one shared [`LinearBinEdges`](@ref) / [`LogBinEdges`](@ref) / [`InfPaddedBinEdges`](@ref), or
`NTuple{6,...}` when value columns may differ.
"""
function SFC.gpu_calculate_structure_functions_single_pass_2d(
    backend::KA.Backend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::_SinglePass2DValueBins,
    ::Type{CT};
    workgroup_size::Int = 64,
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    synchronize::Bool = true,
    force_global_atomic::Bool = false,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = _n_value_edges(value_bins) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    sums = KA.zeros(backend, OT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
    counts = KA.zeros(backend, CT, SF_GPU_SINGLE_PASS_N, n_bins, n_val)
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
function SFC.gpu_calculate_structure_function_batch!(sums, counts, sf::SFT.AbstractPairwiseStructureFunctionType,
        backend::KA.Backend, x::AbstractArray, u::AbstractArray, distance_bins; kwargs...)
    return _gpu_calculate_structure_function_batch!(sums, counts, sf, backend, x, u, distance_bins; kwargs...)
end

"""
    gpu_calculate_structure_function_2d_batch!(sums, counts, sf_type, backend, x, u, distance_bins, value_bins; workspace=nothing, ...)

Batch 2D joint histograms over `(N_dims, N_points, T)`; outputs `(n_dist, n_val, T)`.
"""
function SFC.gpu_calculate_structure_function_2d_batch!(sums, counts, sf::SFT.AbstractPairwiseStructureFunctionType,
        backend::KA.Backend, x::AbstractArray, u::AbstractArray, distance_bins, value_bins; kwargs...)
    return _gpu_calculate_structure_function_2d_batch!(sums, counts, sf, backend, x, u, distance_bins, value_bins; kwargs...)
end

"""
    gpu_calculate_structure_functions_single_pass_batch!(sums, counts, backend, x, u, distance_bins; workspace=nothing, ...)

Batch six invariant 1D distance histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, T)`.
"""
function SFC.gpu_calculate_structure_functions_single_pass_batch!(sums, counts, backend::KA.Backend,
        x::AbstractArray, u::AbstractArray, distance_bins; kwargs...)
    return _gpu_dispatch_single_pass_batch!(sums, counts, backend, x, u, distance_bins; kwargs...)
end

"""
    gpu_calculate_structure_functions_single_pass_2d_batch!(sums, counts, backend, x, u, distance_bins, value_bins; workspace=nothing, ...)

Batch six invariant distance × value joint histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, n_val, T)`.
"""
function SFC.gpu_calculate_structure_functions_single_pass_2d_batch!(sums, counts, backend::KA.Backend,
        x::AbstractArray, u::AbstractArray, distance_bins, value_bins; kwargs...)
    return _gpu_dispatch_single_pass_2d_batch!(sums, counts, backend, x, u, distance_bins, value_bins; kwargs...)
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
) where {OT, CT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    backend = gpu_backend.backend
    FT = promote_type(float(FT1), float(FT2))
    N_dims, N_points = size(x)
    n_edges = length(distance_bins)
    n_bins = n_edges - 1
    size(sums) == (SFC.SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("sums must have shape ($(SFC.SINGLE_PASS_N), $n_bins); got $(size(sums))"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums))"))

    _check_gpu_outputs(sums, counts, backend, (SFC.SINGLE_PASS_N, n_bins); workspace)
    geom, x_dev, u_dev, w_dev = _gpu_prepare_and_stage(backend, x, u, distance_metric, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    # The tiled kernel variant is chosen by the coordinate width it will index,
    # which is the converted width, not the width the caller passed.
    N_dims = SFC._val_int(SFH.coordinate_width(geom))

    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    if isnothing(workspace)
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_bins)
        out_cnts_dev = KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = nothing
    else
        _validate_gpu_workspace!(workspace, backend, :single_pass, n_bins; distance_bins, sum_type=FT)
        SFC.reset_histogram!(workspace)
        out_sums_dev = workspace.out_sums_dev
        out_cnts_dev = CNT === eltype(workspace.out_cnts_dev) ? workspace.out_cnts_dev :
                       KA.zeros(backend, CNT, SF_GPU_SINGLE_PASS_N, n_bins)
        ws = workspace
    end

    _launch_single_pass_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _dist_digitizer(ws, backend, distance_bins, Val(:single_pass)), N_points, N_dims, n_edges,
        geom;
        workspace = ws, weights = w_dev,
    )
    KA.synchronize(backend)

    sums .+= out_sums_dev
    counts .+= out_cnts_dev
    KA.synchronize(backend)
    return sums, counts
end

function SFC.gpu_calculate_structure_function_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT},
    ::Type{CT};
    kwargs...,
) where {FT, CT}
    return _gpu_calculate_structure_function_batch(
        sf_type, backend, x, u, distance_bins, CT; kwargs...,
    )
end

function SFC.gpu_calculate_structure_function_2d_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT},
    value_bins::AbstractVector{FT},
    ::Type{CT};
    kwargs...,
) where {FT, CT}
    return _gpu_calculate_structure_function_2d_batch(
        sf_type, backend, x, u, distance_bins, value_bins, CT; kwargs...,
    )
end

function SFC._dispatch_single_pass(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractMatrix{FT1},
    u::AbstractArray{FT2, N},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, N, CT}
    N >= 3 ||
        throw(ArgumentError("fixed-x batch single-pass expects ndims(u) >= 3 (got ndims=$N)"))
    return _gpu_dispatch_single_pass_batch(
        gpu_backend.backend, x, u, distance_bins, CT; kwargs...,
    )
end

function SFC._dispatch_single_pass(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractArray{FT1, N},
    u::AbstractArray{FT2, N},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, N, CT}
    N >= 3 ||
        throw(ArgumentError("batched single-pass expects ndims >= 3 (got ndims=$N)"))
    return _gpu_dispatch_single_pass_batch(
        gpu_backend.backend, x, u, distance_bins, CT; kwargs...,
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
    value_bins::_SinglePass2DValueBins,
    ::Type{CT};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, N, CT}
    N >= 3 ||
        throw(ArgumentError("batched SP2D expects ndims >= 3 (got ndims=$N)"))
    return _gpu_dispatch_single_pass_2d_batch(
        gpu_backend.backend, x, u, distance_bins, value_bins, CT; kwargs...,
    )
end

function SFC._dispatch_single_pass_2d(
    gpu_backend::CB.AbstractGPUBackend,
    x::AbstractMatrix{FT1},
    u::AbstractArray{FT2, N},
    distance_bins::AbstractVector{FT3},
    value_bins::_SinglePass2DValueBins,
    ::Type{CT};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, N, CT}
    N >= 3 ||
        throw(ArgumentError("fixed-x batch SP2D expects ndims(u) >= 3 (got ndims=$N)"))
    return _gpu_dispatch_single_pass_2d_batch(
        gpu_backend.backend, x, u, distance_bins, value_bins, CT; kwargs...,
    )
end

function SFC._dispatch_single_pass!(b::CB.AbstractGPUBackend, sums::AbstractArray, counts::AbstractArray,
        x::AbstractMatrix, u::AbstractArray{T,N}, bins; kwargs...) where {T,N}
    N >= 3 || throw(ArgumentError("shared-position batch needs trailing field axes"))
    return _gpu_dispatch_single_pass_batch!(sums, counts, b.backend, x, u, bins; kwargs...)
end

SFC._dispatch_single_pass!(::CB.AbstractGPUBackend, sums::AbstractArray, ::AbstractArray,
        ::AbstractMatrix{<:Number}, ::AbstractMatrix{<:Number}, ::AbstractVector{<:Number}; kwargs...) =
    throw(DimensionMismatch("one field's single-pass sums are a (6, n_bins) matrix; got size $(size(sums))"))

function SFC._dispatch_single_pass_2d!(b::CB.AbstractGPUBackend, sums::AbstractArray, counts::AbstractArray,
        x::AbstractMatrix, u::AbstractArray{T,N}, bins, value_bins::_SinglePass2DValueBins; kwargs...) where {T,N}
    N >= 3 || throw(ArgumentError("shared-position batch needs trailing field axes"))
    return _gpu_dispatch_single_pass_2d_batch!(sums, counts, b.backend, x, u, bins, value_bins; kwargs...)
end

end # module
