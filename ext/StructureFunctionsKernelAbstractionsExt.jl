"""
Device kernels for every structure-function route, written with KernelAbstractions.jl.

Loaded with `KernelAbstractions`. An entry called with `backend = GPUBackend(b)` runs here for a
KernelAbstractions backend `b` (`KernelAbstractions.CPU()`, `CUDA.CUDABackend()`, …). On CUDA the
native kernels of the CUDA extension take the calls their launch plans admit, and these kernels the rest.

The tiled kernels use 128-point pair blocks with block-local `UInt32` histograms at every coordinate
and field width whose staged tiles fit the device's shared memory. Joint 2D SF
(`calculate_structure_function` with `value_bins`) uses the same tiled schedule
when its histogram fits the device's shared memory. Six-invariant-type single-pass 1D uses
tiled128 block-local `(6, NB)` histograms when `NB ≤ SF_GPU_MAX_BINS`; six-invariant-type
single-pass 2D uses tiled128 pair traversal with **HTP-EJ** when `n_dist ≤ SF_GPU_MAX_BINS` and its
histogram, or one type plane of it (`:shared`, `:typeplane`), fits the device's shared memory: the shared
histogram fills during the pair loop and flushes into the output with block-end `@atomic`s (the joint 2D
pattern). A larger histogram takes the global-atomic kernel.

## Count types on GPU

A device count histogram accumulates in `UInt32` while an unweighted sweep's worst-case pair count
fits it, and in the requested count type `CT` when it does not; results carry `CT` and stay on the device.

Kernels digitize with the host's `digitize_plan` of the bins, so every bin type bins on a device
exactly as on the CPU.

!!! note "KernelAbstractions Macro Limitations"
    `@index`, `@atomic`, `@Const`, `@private`, `@uniform`, and `@localmem` are imported from `KernelAbstractions`
    by name; they do not resolve as `KA.@index`, etc.
    `@Const` is only valid on **kernel** parameter lists, not on host `@inline` helpers.
"""
module StructureFunctionsKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @localmem, @private, @uniform, @synchronize
using StaticArrays: StaticArrays as SA
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH, StructureFunctionTypes as SFT
using StructureFunctions.Calculations: GPUSFWorkspace, GPUSFLazyBuffers,
    FullUpperTriangle, TilePairWorkList, tile_for, n_pair_blocks, schedule_for, SINGLE_PASS_N,
    TensorComponents, FieldValue, _sf_value_bin, _sf_field_width, _sf_nmom, _sf_pair_moments,
    _sf_pair_moments_along, _sf_accum_moments, _sf_flush_moment, _sf_workspace_kind, _min_max

function __init__()
    SFC._KERNELABSTRACTIONS_LOADED[] = true
    return nothing
end

"""Tile size for CADISHI-style pair blocks (`@localmem` histogram width is `SF_GPU_MAX_BINS`)."""
const SF_GPU_TILE = 128

"""Workgroup size for tiled structure-function kernels."""
const SF_GPU_TILED_WS = 256

"""Maximum distance-bin count compiled into tiled `@localmem` histograms."""
const SF_GPU_MAX_BINS = 128

"""Map 1-based upper-triangle pair index within a tile to `(ia, jb)` with `ia < jb`."""
@inline function _pair_from_linear(k, N)
    term = Float32(4 * N * N - 4 * N + 1 - 8 * (k - 1))
    i_float = (Float32(2 * N - 1) - sqrt(max(0.0f0, term))) * 0.5f0
    i = floor(Int, i_float) + 1
    j = k - (i - 1) * N + (i - 1) * i ÷ 2 + i
    return i, j
end

include(joinpath(@__DIR__, "gpu", "sp2d_accumulation_strategy.jl"))
include(joinpath(@__DIR__, "gpu", "joint2d_shared_memory.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_1d_single_pass.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_2d_single_pass.jl"))
include(joinpath(@__DIR__, "gpu", "kernels_batch.jl"))
# Unified parametric kernel core (building blocks) and the two tiled kernels built on it.
include(joinpath(@__DIR__, "gpu", "sf_core.jl"))
include(joinpath(@__DIR__, "gpu", "sf_tiled.jl"))
include(joinpath(@__DIR__, "gpu", "workspace.jl"))
include(joinpath(@__DIR__, "gpu", "radix_sort.jl"))
include(joinpath(@__DIR__, "gpu", "culling.jl"))
include(joinpath(@__DIR__, "gpu", "in_range.jl"))
include(joinpath(@__DIR__, "gpu", "sorted_line.jl"))
include(joinpath(@__DIR__, "gpu", "launch.jl"))
include(joinpath(@__DIR__, "gpu", "tensor.jl"))
include(joinpath(@__DIR__, "gpu", "harmonic.jl"))
include(joinpath(@__DIR__, "gpu", "gridded_sweep.jl"))
include(joinpath(@__DIR__, "gpu", "multifields.jl"))

"""`true` when `a` is an array in the memory of `backend`'s kind of device, however `backend` is configured.
Only a missing `KA.get_backend` method (i.e. `a` is not a recognized device array) counts as "not on this
backend"; every other failure is a real fault and propagates."""
function _array_on_backend(a, backend::KA.Backend)
    a_backend = try
        KA.get_backend(a)
    catch e
        e isa MethodError || rethrow()
        return false
    end
    return typeof(a_backend) == typeof(backend)
end

"""Host-dense view of `a` for `copyto!` into a device array; a host `Array` is passed through."""
@inline _as_host_dense(a::Array) = a
@inline _as_host_dense(a) = Array(a)

"""
    _stage_sf_device_inputs(backend, x_mat, u_mat, W, F, N_points)

Upload `(W, N_points)` coordinates and `(F, N_points)` fields to `backend` without padding. The two
widths differ on a sphere, where a point is an ambient position and a field an ambient vector.
Arrays already on `backend` with matching shape are used as they are; a workspace's staging buffers
hold only uploads, so a later upload never writes into the caller's arrays.
"""
function _stage_sf_device_inputs(
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    W::Int,
    F::Int,
    N_points::Int;
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    if _array_on_backend(x_mat, backend) && _array_on_backend(u_mat, backend) &&
       size(x_mat) == (W, N_points) && size(u_mat) == (F, N_points)
        return x_mat, u_mat
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
    _gpu_prepare_and_stage(backend, x, u, geometry, N_points; workspace, distance_bins, culling, weights)
        -> (x_dev, u_dev, w_dev, cull)

Prologue for a GPU entry: convert the inputs into the form the kernels of `geometry` index, cull
([`_gpu_cull_and_permute!`](@ref)) and upload them. `cull` is the memo the launchers schedule from, or `nothing`.
"""
function _gpu_prepare_and_stage(
    backend::KA.Backend, x, u, geometry, N_points::Int;
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    distance_bins,
    culling::SFC.CullingPolicy,
    weights = SFC.NoWeights(),
)
    xk, uk = SFH.prepare_pair_inputs(geometry, x, u)
    W = SFC._val_int(SFH.coordinate_width(geometry))
    F = SFC._val_int(SFH.field_width(geometry))
    xk, uk, perm, cull = _gpu_cull_and_permute!(workspace, backend, xk, uk, geometry, distance_bins, culling, x)
    x_dev, u_dev = _stage_sf_device_inputs(backend, xk, uk, W, F, N_points; workspace = workspace)
    # A cull reorders the points, so the weights travel with the coordinates they belong to.
    w_dev = _permuted_weights(backend, weights, perm)
    return x_dev, u_dev, w_dev, cull
end

"""Reorder pair weights by a cull permutation (a permutation of the points, so every index is in bounds);
`NoWeights()` and an unpermuted call are no-ops."""
@inline _permuted_weights(backend, w::SFC.NoWeights, _) = w
@inline _permuted_weights(backend, w::AbstractVector, ::Nothing) = _sf_weights_to_device(backend, w)
function _permuted_weights(backend, w::AbstractVector, perm::AbstractVector{<:Integer})
    w_device = _sf_weights_to_device(backend, w)
    perm_device = _array_on_backend(perm, backend) ? perm : KA.adapt(backend, perm)
    return @inbounds w_device[perm_device]
end

"""
    _gpu_cull_and_permute!(workspace, backend, xk, uk, geom, distance_bins, culling, source = xk)
        -> (xs, us, perm, cull)

Sort the kernel coordinates `xk` into a cull grid on `backend` ([`_gpu_cull_grid`](@ref)) and return
them and the fields `uk` in the grid's order on `backend`, the permutation, and the memo `cull` every
launcher of the call schedules from ([`SFC.schedule_for`](@ref)); `(xk, uk, nothing, nothing)` when the
call sweeps every pair. A workspace keeps the memo and reuses it while `source` (the caller's
coordinate array, by identity), the cutoff and the policy match, so a repeated call on the same points
pays only the field gather; call `refresh!(workspace)` after mutating coordinates in place. Without a
workspace `AutoCulling` prepares a grid only when one slice's pairs pay for it ([`_cull_pays`](@ref)).
"""
function _gpu_cull_and_permute!(workspace, backend, xk, uk, geom, distance_bins, culling, source = xk)
    SFC._cull_enabled(culling) || return xk, uk, nothing, nothing
    cutoff = SFC.cull_cutoff_for(geom, distance_bins, culling)
    cutoff === nothing && return xk, uk, nothing, nothing
    memo = _kept_cull(workspace)
    if SFC._cull_memo_hit(memo, source, cutoff, culling)
        memo isa SFC.GPUNoCullMemo && return xk, uk, nothing, nothing
        return memo.x_sorted, _permute_points(KA.adapt(backend, uk), memo.grid.perm), memo.grid.perm, memo
    end
    _cull_pays(culling, workspace, size(xk, 2), geom) || return xk, uk, nothing, nothing
    xd = KA.adapt(backend, xk)
    grid = _gpu_cull_grid(backend, xd, geom, cutoff, culling)
    if grid === nothing
        _keep_cull!(workspace, SFC.GPUNoCullMemo(source, cutoff, culling))
        return xk, uk, nothing, nothing
    end
    memo = SFC.GPUCullMemo(source, cutoff, culling, grid, @inbounds(xd[:, grid.perm]),
                           Dict{Int, SFC.TilePairWorkList}())
    _keep_cull!(workspace, memo)
    return memo.x_sorted, _permute_points(KA.adapt(backend, uk), grid.perm), grid.perm, memo
end

"""`u` with its points, axis 2, in the order `perm`, a permutation of them; every trailing slice follows."""
@inline _permute_points(u::AbstractArray, perm) = @inbounds u[:, perm, ntuple(_ -> Colon(), ndims(u) - 2)...]

"""The cull memo `workspace` keeps, or `nothing` without a workspace."""
_kept_cull(::Nothing) = nothing
_kept_cull(workspace::SFC.GPUSFWorkspace) = workspace.lazy.cull

"""Keep `memo` on `workspace`; without a workspace the memo lives for the call."""
_keep_cull!(::Nothing, _) = nothing
_keep_cull!(workspace::SFC.GPUSFWorkspace, memo) = (workspace.lazy.cull = memo; nothing)

"""
    gpu_calculate_structure_function(sf_type, backend, x_mat, u_mat, distance_bins, CT; workspace, geometry, culling, weights)

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

# Returns
A raw `StructureFunctionSumsAndCounts` accumulator on `backend` in buffers of its own, with the input
floating-point sum type and count type `CT`, filled in the order of `backend`'s stream; `to_host(result)` waits for it
and copies it to the host.
"""
function SFC.gpu_calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT},
    ::Type{CT};
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT, CT}
    out_dev, cnt_dev, _ = _launch_gpu_structure_function!(
        sf_type, backend, x_mat, u_mat, distance_bins, FT, CT, nothing, nothing;
        workspace, geometry, culling, weights,
    )
    return SF.StructureFunctionSumsAndCounts(sf_type, distance_bins, out_dev, _result_counts(cnt_dev, CT))
end

"""`(schedule, n_tile_blocks, workgroup, ndrange)` of a tiled launch over `N_points` points culled by the
memo `cull` (`nothing` sweeps every tile pair)."""
function _tiled_launch_params(N_points::Int, cull)
    sched = schedule_for(cull, N_points, SF_GPU_TILE)
    n_tile_blocks = n_pair_blocks(sched)
    ws = SF_GPU_TILED_WS
    return sched, n_tile_blocks, ws, n_tile_blocks * ws
end

"""
    _launch_sf_kernel!(backend, plan, out_dev, cnt_dev, x_dev, u_dev, sf_type, dig,
                       N_points, N_bins, geom; cull, weights)

Run a point list's distance histogram over a single slice: the native kernel with `plan`, from
[`SFC.gpu_native_1d_plan`](@ref), or the portable tiled kernel when `plan` is `nothing`. `out_dev` and
`cnt_dev` are `(NB,)`, viewed as the `(NMOM, NB, B)` a kernel writes with `NMOM = B = 1`. `dig` is the
device digitizer of the distance bins; `cull` the call's cull memo or `nothing`.
"""
function _launch_sf_kernel!(
    backend::KA.Backend,
    plan,
    out_dev,
    cnt_dev,
    x_dev,
    u_dev,
    sf_type,
    dig,
    N_points::Int,
    N_bins::Int,
    geom;
    cull = nothing,
    weights = SFC.NoWeights(),
)
    NB = N_bins - 1
    out3, cnt3 = reshape(out_dev, 1, NB, 1), reshape(cnt_dev, 1, NB, 1)
    if plan === nothing
        W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFH.field_width(geom))
        _launch_sf_tiled_1d_varying!(
            backend, out3, cnt3, reshape(x_dev, W, N_points, 1), reshape(u_dev, F, N_points, 1),
            sf_type, dig, N_points, NB, 1, geom;
            weights = weights, cull = cull,
        )
    else
        SFC.gpu_native_launch_1d!(plan, out3, cnt3, x_dev, u_dev, weights, sf_type, dig, N_points, NB, 1,
                                  true, geom, cull)
    end
    return nothing
end

"""
    _launch_gpu_structure_function!(sf_type, backend, x_mat, u_mat, distance_bins, OT, CT, sums, counts;
                                    workspace, geometry, culling, weights) -> (sums_dev, counts_dev, direct)

Launch the distance histogram of one point list with sums of `OT` and counts of `CT`, into the caller's
`sums`/`counts` when [`_accumulation_buffers`](@ref) can take them (`direct`), and into fresh device buffers
when it cannot. Asynchronous.
"""
function _launch_gpu_structure_function!(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix,
    u_mat::AbstractMatrix,
    distance_bins::AbstractVector,
    ::Type{OT},
    ::Type{CT},
    sums,
    counts;
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT}
    N_points = size(x_mat, 2)
    NB = length(distance_bins) - 1
    workspace === nothing || _validate_gpu_workspace!(workspace, backend, :sf1d, NB; distance_bins)
    if SFC._on_a_line(geometry, sf_type)
        out_dev, cnt_dev, direct = _accumulation_buffers(backend, OT, CT, (NB,), sums, counts)
        _gpu_sorted_line!(out_dev, cnt_dev, backend, sf_type, SFC._line_coordinates(x_mat), u_mat, distance_bins,
                          Val(1), Val(1), Val(0), weights)
        return out_dev, cnt_dev, direct
    end
    x_dev, u_dev, w_dev, cull = _gpu_prepare_and_stage(backend, x_mat, u_mat, geometry, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    # The native kernel counts straight into `CT`; the portable kernels count in `_sf_count_type`.
    plan = SFC.gpu_native_1d_plan(backend, eltype(x_dev), eltype(u_dev), OT, CT, w_dev, geometry, NB, sf_type)
    CNT = plan === nothing ? _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points)) : CT
    out_dev, cnt_dev, direct = _accumulation_buffers(backend, OT, CNT, (NB,), sums, counts)
    _launch_sf_kernel!(
        backend, plan, out_dev, cnt_dev, x_dev, u_dev, sf_type,
        _dist_digitizer(workspace, backend, distance_bins, Val(:sf1d)), N_points, NB + 1, geometry;
        cull, weights = w_dev,
    )
    return out_dev, cnt_dev, direct
end

"""Throw unless mutating GPU outputs have `shape` and reside on `backend`."""
function _check_gpu_outputs(sums, counts, backend, shape)
    size(sums) == size(counts) == shape || throw(DimensionMismatch("output buffers must have shape $shape"))
    return _check_gpu_residency(sums, counts, backend)
end

"""Throw unless mutating GPU outputs reside on `backend`."""
function _check_gpu_residency(sums, counts, backend)
    _array_on_backend(sums, backend) && _array_on_backend(counts, backend) ||
        throw(ArgumentError("mutating GPU outputs must reside on the selected backend; use to_host after computation"))
    return nothing
end

SFC._result_zeros(b::CB.AbstractGPUBackend, ::Type{T}, dims::Integer...) where {T} = KA.zeros(b.backend, T, dims...)
SFC._adaptor(b::CB.AbstractGPUBackend) = a -> KA.adapt(b.backend, a)
SFC._require_device_outputs(b::CB.AbstractGPUBackend, sums, counts) = _check_gpu_residency(sums, counts, b.backend)

function SFC.gpu_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT};
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT, FT}
    _check_gpu_outputs(output_sums, output_counts, backend, (length(distance_bins)-1,))
    out_dev, cnt_dev, direct = _launch_gpu_structure_function!(
        sf_type, backend, x_mat, u_mat, distance_bins, OT, CT, output_sums, output_counts;
        workspace, geometry, culling, weights,
    )
    _add_accumulated!(output_sums, output_counts, out_dev, cnt_dev, direct)
    return nothing
end


"""True when six-invariant-type 2D single-pass can use HTP-EJ tiled128 (`n_dist ≤ SF_GPU_MAX_BINS`)."""
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
    cull = nothing,
    weights = SFC.NoWeights(),
)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points, cull)
    kernel! = _sf6_single_pass_kernel_tiled128_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, dig,
        N_points, n_edges, NB,
        sched, n_tile_blocks, ws,
        _sf_weights_to_device(backend, weights), SFH.coordinate_width(geom), SFH.field_width(geom),
        Val(eltype(out_cnts_dev)), geom;
        ndrange = ndrange,
    )
    return nothing
end

@inline _gpu_ld_col(m, k::Int, ::Val{W}, ::Type{FT}) where {W, FT} =
    SA.SVector{W, FT}(ntuple(d -> @inbounds(m[d, k]), Val(W)))

"""The frame and single-pass invariants of pair `(i, j)`, loaded at the geometry's coordinate and
field widths."""
@inline function _gpu_single_pass_pair_invariants(x_mat, u_mat, i::Int, j::Int, geom)
    vW, vF = SFH.coordinate_width(geom), SFH.field_width(geom)
    XT, UT = eltype(x_mat), eltype(u_mat)
    X1 = _gpu_ld_col(x_mat, i, vW, XT)
    X2 = _gpu_ld_col(x_mat, j, vW, XT)
    U1 = _gpu_ld_col(u_mat, i, vF, UT)
    U2 = _gpu_ld_col(u_mat, j, vF, UT)
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
    for t in 1:SINGLE_PASS_N
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
    N_bins::Int,
    geom,
)
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]
    if i < j
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(x_mat, u_mat, i, j, geom)
        bin = SFH.digitize(dist, dig)
        if ok && 1 <= bin < N_bins
            _gpu_accumulate_single_pass_global!(output_sums, output_counts, bin, du_L, du_n2,
                SFC._point_weight(wts, i) * SFC._point_weight(wts, j))
        end
    end
end

"""Launch a point list's six single-pass invariant histograms `(6, NB)`: the native kernel with its plan
for the call ([`SFC.gpu_native_1d_plan`](@ref)), or [`_launch_single_pass_portable!`](@ref) when there
is none."""
function _launch_single_pass_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dig,
    N_points::Int,
    n_edges::Int,
    geom;
    cull = nothing,
    weights = SFC.NoWeights(),
)
    NB = n_edges - 1
    plan = SFC.gpu_native_1d_plan(backend, eltype(x_dev), eltype(u_dev), eltype(out_sums_dev),
                                  eltype(out_cnts_dev), weights, geom, NB, SFT.SinglePassInvariants())
    if plan === nothing
        _launch_single_pass_portable!(backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev,
                                      dig, N_points, n_edges, geom; cull, weights)
    else
        SFC.gpu_native_launch_1d!(plan, reshape(out_sums_dev, SINGLE_PASS_N, NB, 1),
                                  reshape(out_cnts_dev, SINGLE_PASS_N, NB, 1), x_dev, u_dev, weights,
                                  SFT.SinglePassInvariants(), dig, N_points, NB, 1, true, geom, cull)
    end
    return nothing
end

"""Launch a point list's six single-pass invariant histograms on the portable kernels: the tiled
kernel when its histogram fits, the global-atomic kernel when it does not."""
function _launch_single_pass_portable!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dig,
    N_points::Int,
    n_edges::Int,
    geom;
    cull = nothing,
    weights = SFC.NoWeights(),
)
    NB = n_edges - 1
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFH.field_width(geom))
    if _gpu_single_pass_tiled_eligible(SFC.gpu_device_caps(backend), NB, eltype(x_dev), eltype(out_cnts_dev), W, F)
        return _launch_single_pass_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dig, N_points, n_edges, NB, geom; cull = cull, weights = weights,
        )
    end
    kernel! = _sf_single_pass_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, dig, _sf_weights_to_device(backend, weights),
        N_points, n_edges, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

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
)
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        XT = eltype(x_mat)
        UT = eltype(u_mat)
        vW, vF = SFH.coordinate_width(geom), SFH.field_width(geom)
        X1 = _gpu_ld_col(x_mat, i, vW, XT)
        X2 = _gpu_ld_col(x_mat, j, vW, XT)
        U1 = _gpu_ld_col(u_mat, i, vF, UT)
        U2 = _gpu_ld_col(u_mat, j, vF, UT)
        ok, dist, frame = SFH.pair_frame(geom, X1, X2)
        dbin = SFH.digitize(dist, ddig)
        if ok && 1 <= dbin < N_dist_bins
            dU = SFH.pair_delta(geom, frame, X1, X2, U1, U2)
            val = SFT.pair_value(sf_type, geom, frame, dist, dU)
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

"""Launch one joint distance × value histogram `(n_dist, n_val)` with device digitizers `ddig` and
`vdig`: the backend's native plan for the call ([`SFC.gpu_native_2d_plan`](@ref)), handed the portable launch as a
candidate, or [`_launch_joint_2d_portable!`](@ref) when there is none."""
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
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    cull = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    plan = SFC.gpu_native_2d_plan(backend, eltype(x_dev), eltype(u_dev), eltype(out_sums_dev),
                                  eltype(out_cnts_dev), weights, geom, sf_type, n_dist, n_val, vdig)
    portable!(o, c) = _launch_joint_2d_portable!(backend, workgroup_size, reshape(o, n_dist, n_val),
                                                 reshape(c, n_dist, n_val), x_dev, u_dev, sf_type, ddig, vdig,
                                                 N_points, n_dist_edges, n_val_edges, geom;
                                                 workspace, cull, weights, second_axis)
    if plan === nothing
        portable!(out_sums_dev, out_cnts_dev)
    else
        SFC.gpu_native_launch_2d!(plan, reshape(out_sums_dev, 1, n_dist, n_val, 1),
                                  reshape(out_cnts_dev, 1, n_dist, n_val, 1), x_dev, u_dev, weights, sf_type,
                                  ddig, vdig, N_points, n_dist, n_val, 1, true, geom, second_axis, cull, portable!)
    end
    return nothing
end

"""Launch one joint distance × value histogram on the portable kernels: the tiled kernel when its
shared histogram at the compile width fits the device, the global-atomic kernel when it does not."""
function _launch_joint_2d_portable!(
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
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    cull = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFH.field_width(geom))
    hist = workspace === nothing ? n_dist * n_val : workspace.joint2d_compile_cells
    if _gpu_joint_2d_tiled_eligible(SFC.gpu_device_caps(backend), W, F, eltype(x_dev),
                                    eltype(out_sums_dev), eltype(out_cnts_dev), hist)
        return _launch_joint_2d_tiled_kernel!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            sf_type, ddig, vdig, N_points, n_dist_edges, n_val_edges, n_dist, n_val, hist, geom;
            cull = cull, weights = weights, second_axis = second_axis,
        )
    end
    kernel! = _sf_joint_2d_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _sf_weights_to_device(backend, weights), ddig, vdig, sf_type,
        N_points, n_dist_edges, n_val_edges, geom, second_axis;
        ndrange = (N_points, N_points),
    )
    return nothing
end

"""
    _launch_gpu_joint2d!(sf_type, backend, x_mat, u_mat, distance_bins, value_bins, OT, CT, sums, counts;
                         workgroup_size, workspace, geometry, culling, weights, second_axis)
        -> (sums_dev, counts_dev, direct)

Launch the joint histogram of one point list with sums of `OT` and counts of `CT`, into the caller's
`sums`/`counts` when [`_accumulation_buffers`](@ref) can take them (`direct`), and into fresh device buffers
when it cannot. Asynchronous.
"""
function _launch_gpu_joint2d!(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix,
    u_mat::AbstractMatrix,
    distance_bins,
    value_bins,
    ::Type{OT},
    ::Type{CT},
    sums,
    counts;
    workgroup_size::Int = 64,
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
) where {OT, CT}
    N_points = size(x_mat, 2)
    # Only the point count has to agree: on a shell a point takes two coordinates while the velocity
    # may carry a third, radial, component.
    size(u_mat, 2) == N_points ||
        throw(DimensionMismatch(
            "x_mat and u_mat must share the point count; got $(size(x_mat)) and $(size(u_mat))",
        ))
    SFC._require_value_axis(second_axis, geometry)
    n_dist_edges = length(distance_bins)
    n_val_edges = length(value_bins)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    workspace === nothing ||
        _validate_gpu_workspace!(workspace, backend, :joint2d, n_dist; n_val, distance_bins, value_bins)
    x_dev, u_dev, w_dev, cull = _gpu_prepare_and_stage(backend, x_mat, u_mat, geometry, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    out_sums_dev, out_cnts_dev, direct = _accumulation_buffers(backend, OT, CNT, (n_dist, n_val), sums, counts)
    _launch_joint_2d_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
        _dist_digitizer(workspace, backend, distance_bins, Val(:joint2d)),
        _value_digitizer(workspace, backend, value_bins),
        N_points, n_dist_edges, n_val_edges, geometry;
        workspace = workspace, cull = cull, weights = w_dev, second_axis = second_axis,
    )
    return out_sums_dev, out_cnts_dev, direct
end

"""
    gpu_calculate_structure_function_2d(sf_type, backend, x_mat, u_mat, distance_bins, value_bins, CT; kwargs...)

Compute one 2D joint histogram (distance × SF value) for `sf_type` on `backend`.
Returns [`StructureFunction2DSumsAndCounts`](@ref) with the same flat edge vectors passed in.

Uses tiled128 block-local histograms when the histogram fits the device's shared memory, and
``(N_points, N_points)`` global-atomic pair kernels when it does not. The shared histogram is compiled
``n_dist × n_val`` wide unless a [`SFC.GPUSFWorkspace`](@ref) sets `joint2d_compile_cells`
(see [`joint2d_smem_max`](@ref), [`joint2d_smem_align256`](@ref)). Results stay on the selected
backend in buffers of their own, with counts of type `CT`. Use `to_host` for host conversion.
"""
function SFC.gpu_calculate_structure_function_2d(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT1},
    u_mat::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::AbstractVector{FT4},
    ::Type{CT};
    workgroup_size::Int = 64,
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, FT4 <: Number, CT}
    FT = promote_type(FT1, FT2, FT3, FT4)
    out_sums_dev, out_cnts_dev, _ = _launch_gpu_joint2d!(
        sf_type, backend, x_mat, u_mat, distance_bins, value_bins, FT, CT, nothing, nothing;
        workgroup_size, workspace, geometry, culling, weights, second_axis,
    )
    counts = _result_counts(out_cnts_dev, CT)
    return SF.StructureFunction2DSumsAndCounts(sf_type, distance_bins, value_bins, out_sums_dev, counts, second_axis)
end

function SFC.gpu_calculate_structure_function_2d!(sums, counts, sf, backend::KA.Backend,
        x::AbstractMatrix, u::AbstractMatrix, distance_bins, value_bins; kwargs...)
    shape = (length(distance_bins)-1, length(value_bins)-1)
    _check_gpu_outputs(sums, counts, backend, shape)
    ds, dc, direct = _launch_gpu_joint2d!(sf, backend, x, u, distance_bins, value_bins, eltype(sums), eltype(counts),
                                          sums, counts; kwargs...)
    _add_accumulated!(sums, counts, ds, dc, direct)
    return nothing
end


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
    for t in 1:SINGLE_PASS_N
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
    N_bins::Int,
    N_val_edges::Int,
    geom,
)
    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    if i < j
        ok, dist, du_L, du_n2 = _gpu_single_pass_pair_invariants(x_mat, u_mat, i, j, geom)
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
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    backend = gpu_backend.backend
    N_points = size(x, 2)
    n_dist_edges = length(distance_bins)
    n_bins = n_dist_edges - 1
    n_val = size(sums_3d, 3)
    size(sums_3d) == (SINGLE_PASS_N, n_bins, n_val) ||
        throw(DimensionMismatch("sums must have shape ($SINGLE_PASS_N, n_bins, n_val); got $(size(sums_3d))"))
    size(counts_3d) == size(sums_3d) ||
        throw(DimensionMismatch("counts and sums must have the same shape"))
    _check_gpu_outputs(sums_3d, counts_3d, backend, (SINGLE_PASS_N, n_bins, n_val))
    SFC._validate_value_bins!(value_bins, n_val)
    workspace === nothing || _validate_gpu_workspace!(
        workspace, backend, :single_pass_2d, n_bins; n_val, distance_bins, value_bins)
    x_dev, u_dev, w_dev, cull = _gpu_prepare_and_stage(backend, x, u, geometry, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    out_sums_dev, out_cnts_dev, direct = _accumulation_buffers(
        backend, OT, CNT, (SINGLE_PASS_N, n_bins, n_val), sums_3d, counts_3d)
    _launch_single_pass_2d!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _dist_digitizer(workspace, backend, distance_bins, Val(:single_pass_2d)),
        _value_digitizer(workspace, backend, value_bins),
        N_points, n_dist_edges, _n_value_edges(value_bins), geometry;
        cull = cull, weights = w_dev,
    )
    _add_accumulated!(sums_3d, counts_3d, out_sums_dev, out_cnts_dev, direct)
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
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    backend = gpu_backend.backend
    FT = promote_type(float(FT1), float(FT2))
    N_points = size(x, 2)
    n_edges = length(distance_bins)
    n_bins = n_edges - 1
    workspace === nothing || _validate_gpu_workspace!(workspace, backend, :single_pass, n_bins; distance_bins)
    x_dev, u_dev, w_dev, cull = _gpu_prepare_and_stage(backend, x, u, geometry, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    out_sums_dev, out_cnts_dev, _ = _accumulation_buffers(
        backend, FT, CNT, (SINGLE_PASS_N, n_bins), nothing, nothing)
    _launch_single_pass_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _dist_digitizer(workspace, backend, distance_bins, Val(:single_pass)), N_points, n_edges,
        geometry;
        cull, weights = w_dev,
    )
    return (sums = out_sums_dev, counts = _result_counts(out_cnts_dev, CT))
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
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = _n_value_edges(value_bins) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    sums = KA.zeros(backend, OT, SINGLE_PASS_N, n_bins, n_val)
    counts = KA.zeros(backend, CT, SINGLE_PASS_N, n_bins, n_val)
    return _gpu_run_single_pass_2d!(
        CB.GPUBackend(backend), sums, counts, x, u, distance_bins, value_bins;
        workgroup_size = workgroup_size, workspace = workspace,
        geometry, culling = culling, weights = weights,
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
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
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
        geometry, culling = culling, weights = weights,
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
    workspace::Union{SFC.GPUSFWorkspace, Nothing} = nothing,
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {OT, CT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    backend = gpu_backend.backend
    N_points = size(x, 2)
    n_edges = length(distance_bins)
    n_bins = n_edges - 1
    size(sums) == (SFC.SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("sums must have shape ($(SFC.SINGLE_PASS_N), $n_bins); got $(size(sums))"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape $(size(sums))"))
    _check_gpu_outputs(sums, counts, backend, (SFC.SINGLE_PASS_N, n_bins))
    workspace === nothing || _validate_gpu_workspace!(workspace, backend, :single_pass, n_bins; distance_bins)
    x_dev, u_dev, w_dev, cull = _gpu_prepare_and_stage(backend, x, u, geometry, N_points;
        workspace = workspace, distance_bins = distance_bins, culling = culling, weights = weights)
    CNT = _sf_count_type(w_dev, CT, _sf_worst_case_pairs(N_points))
    out_sums_dev, out_cnts_dev, direct = _accumulation_buffers(
        backend, OT, CNT, (SINGLE_PASS_N, n_bins), sums, counts)
    _launch_single_pass_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        _dist_digitizer(workspace, backend, distance_bins, Val(:single_pass)), N_points, n_edges,
        geometry;
        cull, weights = w_dev,
    )
    _add_accumulated!(sums, counts, out_sums_dev, out_cnts_dev, direct)
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
