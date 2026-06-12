"""
GPU-accelerated structure function kernels using KernelAbstractions.jl.

This extension is loaded automatically when `KernelAbstractions` is loaded by the user.
The `gpu_calculate_structure_function` entry point accepts any KA-compatible backend:
  - `KernelAbstractions.CPU()` – for CPU-parallel testing / parity verification
  - `CUDABackend()` from CUDA.jl – for NVIDIA GPU acceleration
  - `ROCBackend()` from AMDGPU.jl – for AMD GPU acceleration

For `N_dims ∈ {2,3}` the fast path uses tiled128 pair blocks with block-local
`UInt32` histograms (linear, log, and general bin routes).

Bin edge routing matches CPU `BinEdges.jl`: `LinearBinEdges`, `LogBinEdges`,
`InfPaddedBinEdges`, uniform ranges, and general monotone edge vectors.

!!! note "KernelAbstractions Macro Limitations"
    We explicitly import `@index`, `@atomic`, and `@Const` from `KernelAbstractions` because
    these macros currently fail to resolve correctly when called as `KA.@index`, etc.
    `@Const` is only valid on **kernel** parameter lists, not on host `@inline` helpers.
"""
module StructureFunctionsGPUExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @localmem, @synchronize
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    SpectralAnalysis as SFSA, HelperFunctions as SFH, StructureFunctionTypes as SFT,
    AbstractBinEdges, LinearBinEdges, LogBinEdges, InfPaddedBinEdges, BinEdges

# ---------------------------------------------------------------------------
# GPU bin layout (mirrors CPU AbstractBinEdges / BinEdges.jl routing)
# ---------------------------------------------------------------------------

"""
    _GPUBinLayout

Host-side bin routing for GPU kernels. Exactly one of `linear`, `log`, or
`general_edges` is populated — dispatch uses those types, not symbols.
"""
struct _GPUBinLayout
    linear::Union{LinearBinEdges, Nothing}
    log::Union{LogBinEdges, Nothing}
    general_edges::Union{Vector, Nothing}
end

function _GPUBinLayout(;
    linear::Union{LinearBinEdges, Nothing} = nothing,
    log::Union{LogBinEdges, Nothing} = nothing,
    general_edges::Union{Vector, Nothing} = nothing,
)
    nset = count(!isnothing, (linear, log, general_edges))
    nset == 1 ||
        error("_GPUBinLayout: exactly one of linear, log, general_edges must be set (got $nset)")
    return _GPUBinLayout(linear, log, general_edges)
end

"""Active bin-edges object for multiple dispatch (`LinearBinEdges`, `LogBinEdges`, or `Vector`)."""
function _gpu_active_bins(layout::_GPUBinLayout)
    layout.linear !== nothing && return layout.linear
    layout.log !== nothing && return layout.log
    layout.general_edges !== nothing && return layout.general_edges
    error("_GPUBinLayout: empty layout")
end

function _unwrap_inf_padded(edges)
    if edges isa InfPaddedBinEdges
        return edges.edges
    end
    return edges
end

function _is_positive_monotone(edges::AbstractVector{FT}) where {FT}
    n = length(edges)
    n < 2 && return false
    @inbounds for k in 1:n
        if edges[k] <= zero(FT)
            return false
        end
        if k > 1 && edges[k] <= edges[k - 1]
            return false
        end
    end
    return true
end

function _try_uniform_linear(edges::AbstractVector{FT}) where {FT}
    n = length(edges)
    n < 2 && return nothing
    step_val = (edges[end] - edges[1]) / (n - 1)
    if !isfinite(step_val)
        return nothing
    end
    if step_val == zero(FT)
        return LinearBinEdges(range(edges[1], edges[1]; length = n))
    end
    tol = max(FT(10) * eps(FT), abs(step_val) * FT(1e-5))
    @inbounds for k in 2:n
        if abs((edges[k] - edges[k - 1]) - step_val) > tol
            return nothing
        end
    end
    return LinearBinEdges(range(edges[1], edges[end]; length = n))
end

"""
    _resolve_gpu_bin_layout(edges)

Select the same fast binning strategy the CPU path uses via `AbstractBinEdges`.
"""
function _resolve_gpu_bin_layout(edges)
    inner = _unwrap_inf_padded(edges)

    if inner isa LinearBinEdges
        return _GPUBinLayout(linear = inner)
    elseif inner isa LogBinEdges
        return _GPUBinLayout(log = inner)
    elseif inner isa AbstractRange
        return _GPUBinLayout(linear = LinearBinEdges(inner))
    elseif inner isa BinEdges
        return _resolve_gpu_bin_layout(inner.edges)
    elseif inner isa AbstractVector
        FT = eltype(inner)
        linear = _try_uniform_linear(inner)
        if linear !== nothing
            return _GPUBinLayout(linear = linear)
        elseif _is_positive_monotone(inner)
            return _GPUBinLayout(log = LogBinEdges(collect(inner)))
        else
            return _GPUBinLayout(general_edges = collect(inner))
        end
    else
        error("GPUExt: unsupported distance_bins type $(typeof(edges))")
    end
end

function _layout_edge_vector(layout::_GPUBinLayout)
    bins = _gpu_active_bins(layout)
    if bins isa LinearBinEdges
        return collect(bins.edges)
    elseif bins isa LogBinEdges
        return collect(bins.edges)
    else
        return bins
    end
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
    idx = round(Int, muladd(x, inv_step, offset))
    idx = clamp(idx, 1, n_edges)
    edge_val = muladd(step_val, T(idx - 1), first_edge)
    search_idx = edge_val < x ? idx + 1 : idx
    return search_idx - 1
end

# Log digitize on device matches CPU `Base.searchsortedfirst(::LogBinEdges)`:
# exponent LUT (V9 hybrid) from `BinEdges_timings_and_theory.md` §4 — not `log(x)` +
# full-vector binary search. GPU must stay bit-identical to CPU digitize for parity.
@inline function _gpu_digitize_log(
    x::T,
    edges,
    lut,
    e_min::Int,
    n_edges::Int,
) where {T}
    f = edges[1]
    if x <= f
        return 0
    end
    l = edges[n_edges]
    if x > l
        return n_edges
    end
    e = exponent(x)
    e_idx = clamp(e - e_min + 1, 1, length(lut))
    idx_start = lut[e_idx]
    idx_end = e_idx < length(lut) ? lut[e_idx + 1] : n_edges
    if idx_end - idx_start <= 8
        idx = idx_start
        while idx <= idx_end && edges[idx] < x
            idx += 1
        end
        return idx - 1
    else
        low = idx_start
        high = idx_end
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

# ---------------------------------------------------------------------------
# Tiled128 + block-local UInt32 histogram (2D/3D production fast path)
# ---------------------------------------------------------------------------

"""Tile size for CADISHI-style pair blocks (`@localmem` histogram width is `SF_GPU_MAX_BINS`)."""
const SF_GPU_TILE = 128

"""Workgroup size for tiled structure-function kernels."""
const SF_GPU_TILED_WS = 256

"""Maximum distance-bin count compiled into tiled `@localmem` histograms."""
const SF_GPU_MAX_BINS = 64

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
    lo = one(k)
    hi = N - one(k)
    while lo < hi
        mid = (lo + hi) >>> 1
        row_start = (mid - one(mid)) * N - (mid - one(mid)) * mid ÷ 2
        row_end = row_start + (N - mid)
        if k > row_end
            lo = mid + one(k)
        else
            hi = mid
        end
    end
    i = lo
    j = i + (k - ((i - one(i)) * N - (i - one(i)) * i ÷ 2))
    return i, j
end

include(joinpath(@__DIR__, "TiledStructureFunctionKernels.jl"))

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
    N_points::Int,
) where {FT}
    if _array_on_backend(x_mat, backend) && _array_on_backend(u_mat, backend)
        d_x, n_x = size(x_mat)
        d_u, n_u = size(u_mat)
        if d_x == N_dims && d_u == N_dims && n_x == N_points && n_u == N_points
            return x_mat, u_mat
        end
    end

    x_dev = KA.allocate(backend, FT, N_dims, N_points)
    u_dev = KA.allocate(backend, FT, N_dims, N_points)
    copyto!(x_dev, Array(x_mat))
    copyto!(u_dev, Array(u_mat))
    return x_dev, u_dev
end


# ---------------------------------------------------------------------------
# Spectral Analysis Kernel (Direct Sum)
# ---------------------------------------------------------------------------

KA.@kernel function _spectral_kernel!(
    coeffs,                # Out: (ms..., NU)
    @Const(x_dev),         # (3, N)  - padded to 3
    @Const(u_dev),         # (NU, N)
    @Const(ks_phys_dev),   # 3 vectors of varying lengths ms[d] - padded to 3
    iflag::Int,
    N::Int,
    NU::Int,
    D::Int,
    ms::NTuple, # Use NTuple (untapped) to avoid UndefVarError
)
    # One thread per wavenumber I
    idx = @index(Global, Cartesian)

    # Pre-fetch k_phys components for this wavenumber
    # We use SVector{3} and dot with padded x_pos
    k_phys = SA.SVector{3, eltype(x_dev)}(
        D >= 1 ? ks_phys_dev[1][idx[1]] : zero(eltype(x_dev)),
        D >= 2 ? ks_phys_dev[2][idx[2]] : zero(eltype(x_dev)),
        D >= 3 ? ks_phys_dev[3][idx[3]] : zero(eltype(x_dev)),
    )

    for u_idx in 1:NU
        sum_val = zero(eltype(coeffs))
        for j in 1:N
            x_pos = SA.SVector{3, eltype(x_dev)}(
                x_dev[1, j],
                x_dev[2, j],
                x_dev[3, j],
            )

            # Phase factor
            phi = -iflag * (SA.dot(k_phys, x_pos))
            W = complex(cos(phi), sin(phi))

            sum_val += u_dev[u_idx, j] * W
        end
        coeffs[idx, u_idx] = sum_val / N
    end
end


# ---------------------------------------------------------------------------
# N-dimensional variant: pads 1D/2D inputs to 3D for uniformity
# ---------------------------------------------------------------------------

"""
    _pad3(v::SVector)

Pad 1D/2D static vectors to 3D by appending zeros.
3D vectors are returned unchanged.
"""
function _pad3(v::SA.SVector{N, T}) where {N, T}
    if N == 1
        return SA.SVector{3, T}(v[1], zero(T), zero(T))
    elseif N == 2
        return SA.SVector{3, T}(v[1], v[2], zero(T))
    else
        return v
    end
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
- `distance_bins`: bin *edges* (`AbstractVector` or any `AbstractBinEdges` — same types
  as CPU: `LinearBinEdges`, `LogBinEdges`, `InfPaddedBinEdges`, uniform ranges, etc.).
  Routing mirrors `BinEdges.jl`: uniform → O(1) FMA, positive monotone → exponent LUT,
  otherwise localized binary search on device.
- `sf_type`: any `AbstractStructureFunctionType`.

# Returns
When `return_sums_and_counts=true`, a `StructureFunctionSumsAndCounts` with
`Vector{Float64}` sums and count vector of type `count_eltype` (default `UInt32`,
length `N_bins - 1`).
Otherwise a binned-mean `StructureFunction`.
"""
function SFC.gpu_calculate_structure_function(
    sf_type::SFT.AbstractStructureFunctionType,
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
    sf_type::SFT.AbstractStructureFunctionType,
    backend::KA.Backend,
    x_vecs::Tuple{T, Vararg{T}},
    u_vecs::Tuple{U, Vararg{U}},
    distance_bins::AbstractVector{<:Tuple};
    return_sums_and_counts::Bool = false,
    kwargs...,
) where {T <: AbstractVector, U <: AbstractVector}
    return SFC.gpu_calculate_structure_function(
        sf_type,
        backend,
        x_vecs,
        u_vecs,
        distance_bins,
        Val(return_sums_and_counts);
        kwargs...,
    )
end

function SFC.gpu_calculate_structure_function(
    sf_type::SFT.AbstractStructureFunctionType,
    backend::KA.Backend,
    x_vecs::Tuple{T, Vararg{T}},
    u_vecs::Tuple{U, Vararg{U}},
    distance_bins::AbstractVector{<:Tuple},
    ::Val{RSAC};
    kwargs...,
) where {T <: AbstractVector, U <: AbstractVector, RSAC}
    N_dims = length(x_vecs)
    N_points = length(x_vecs[1])
    TX = eltype(T)
    TU = eltype(U)
    FT = promote_type(float(TX), float(TU))

    x_mat = Matrix{FT}(undef, N_dims, N_points)
    u_mat = Matrix{FT}(undef, N_dims, N_points)
    for k in 1:N_dims
        @views x_mat[k, :] .= x_vecs[k]
        @views u_mat[k, :] .= u_vecs[k]
    end

    n_bins = length(distance_bins)
    bin_edges = Vector{FT}(undef, n_bins + 1)
    for k in 1:n_bins
        bin_edges[k] = distance_bins[k][1]
    end
    bin_edges[end] = distance_bins[end][2]

    return SFC.gpu_calculate_structure_function(
        sf_type,
        backend,
        x_mat,
        u_mat,
        bin_edges,
        Val(RSAC);
        kwargs...,
    )
end

function SFC.gpu_calculate_structure_function(
    sf_type::SFT.AbstractStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT1},
    u_mat::AbstractMatrix{FT2},
    distance_bins::AbstractVector{<:Tuple};
    return_sums_and_counts::Bool = false,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number}
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
    sf_type::SFT.AbstractStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT1},
    u_mat::AbstractMatrix{FT2},
    distance_bins::AbstractVector{<:Tuple},
    ::Val{RSAC};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, RSAC}
    FT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins)
    bin_edges = Vector{FT}(undef, n_bins + 1)
    for k in 1:n_bins
        bin_edges[k] = distance_bins[k][1]
    end
    bin_edges[end] = distance_bins[end][2]

    return SFC.gpu_calculate_structure_function(
        sf_type,
        backend,
        x_mat,
        u_mat,
        bin_edges,
        Val(RSAC);
        kwargs...,
    )
end

function SFC.gpu_calculate_structure_function(
    sf_type::SFT.AbstractStructureFunctionType,
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
    FT = eltype(lbe.edges)
    edges_dev = KA.allocate(backend, FT, N_bins)
    lut_dev = KA.allocate(backend, Int32, length(lbe.lut))
    copyto!(edges_dev, collect(lbe.edges))
    copyto!(lut_dev, Int32.(lbe.lut))
    kernel! = dim2 ?
        _sf_kernel_tiled128_2d_log_u32!(backend, ws) :
        _sf_kernel_tiled128_3d_log_u32!(backend, ws)
    kernel!(
        out_dev, cnt_dev, x_dev, u_dev, sf_type,
        N_points, N_dims, N_bins, NB,
        lbe.edges[1], edges_dev, lut_dev, lbe.e_min,
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
    NB::Int,
) where {FT}
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    dim2 = N_dims == 2
    bins_dev = KA.allocate(backend, FT, N_bins)
    copyto!(bins_dev, edges)
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
    layout::_GPUBinLayout,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
)
    NB = N_bins - 1
    bins = _gpu_active_bins(layout)
    NB > SF_GPU_MAX_BINS &&
        error("GPUExt: tiled kernels support at most $SF_GPU_MAX_BINS bins (got NB=$NB)")
    _launch_sf_tiled_kernel!(
        backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, bins,
        N_points, N_dims, N_bins, NB,
    )
    return nothing
end

"""
    _launch_gpu_structure_function!(sf_type, backend, x_mat, u_mat, distance_bins; workgroup_size=64)

Run the tiled GPU structure-function kernel and return device-resident `(out_dev, cnt_dev)`.
`distance_bins` must be a host edge vector (same convention as CPU).
"""
function _launch_gpu_structure_function!(
    sf_type::SFT.AbstractStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    kwargs...,
) where {FT}
    N_dims, N_points = size(x_mat)
    layout = _resolve_gpu_bin_layout(distance_bins)
    edges_host = _layout_edge_vector(layout)
    N_bins = length(edges_host)

    if N_dims ∉ (2, 3)
        error("GPUExt: GPU structure functions require N_dims ∈ {2, 3} (got N_dims=$N_dims)")
    end
    NB = N_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("GPUExt: at most $SF_GPU_MAX_BINS distance bins on GPU (got NB=$NB)")

    x_dev, u_dev = _stage_sf_device_inputs(backend, x_mat, u_mat, N_dims, N_points)
    out_dev = KA.zeros(backend, FT, NB)
    cnt_dev = KA.zeros(backend, UInt32, NB)

    _launch_sf_kernel!(
        backend, workgroup_size,
        out_dev, cnt_dev, x_dev, u_dev,
        sf_type, layout, N_points, N_dims, N_bins,
    )
    KA.synchronize(backend)
    return out_dev, cnt_dev, edges_host
end

"""Download device sums/counts and accumulate into caller-owned host buffers."""
function _accumulate_gpu_sf_host!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    out_dev,
    cnt_dev,
) where {OT, CT}
    NB = length(output_sums)
    length(output_counts) == NB ||
        throw(ArgumentError("GPUExt: output_counts length ($(length(output_counts))) must match output_sums ($NB)"))
    tmp_s = Vector{OT}(undef, NB)
    copyto!(tmp_s, out_dev)
    output_sums .+= tmp_s
    tmp_c = Vector{UInt32}(undef, NB)
    copyto!(tmp_c, cnt_dev)
    if CT === UInt32
        output_counts .+= tmp_c
    else
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
    tmp_c = Vector{UInt32}(undef, NB)
    copyto!(tmp_c, cnt_dev)
    counts = CT === UInt32 ? tmp_c : Vector{CT}(tmp_c)
    return output, counts
end

function SFC.gpu_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    sf_type::SFT.AbstractStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT};
    workgroup_size::Int = 64,
    kwargs...,
) where {OT, CT, FT}
    out_dev, cnt_dev, _ = _launch_gpu_structure_function!(
        sf_type, backend, x_mat, u_mat, distance_bins;
        workgroup_size = workgroup_size,
        kwargs...,
    )
    _accumulate_gpu_sf_host!(output_sums, output_counts, out_dev, cnt_dev)
    return nothing
end

function SFC.gpu_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    sf_type::SFT.AbstractStructureFunctionType,
    backend::KA.Backend,
    x_vecs::Tuple{T, Vararg{T}},
    u_vecs::Tuple{U, Vararg{U}},
    distance_bins::AbstractVector{<:Tuple};
    kwargs...,
) where {OT, CT, T <: AbstractVector, U <: AbstractVector}
    N_dims = length(x_vecs)
    N_points = length(x_vecs[1])
    TX = eltype(T)
    TU = eltype(U)
    FT = promote_type(float(TX), float(TU))

    x_mat = Matrix{FT}(undef, N_dims, N_points)
    u_mat = Matrix{FT}(undef, N_dims, N_points)
    for k in 1:N_dims
        @views x_mat[k, :] .= x_vecs[k]
        @views u_mat[k, :] .= u_vecs[k]
    end

    n_bins = length(distance_bins)
    bin_edges = Vector{FT}(undef, n_bins + 1)
    for k in 1:n_bins
        bin_edges[k] = distance_bins[k][1]
    end
    bin_edges[end] = distance_bins[end][2]

    return SFC.gpu_calculate_structure_function!(
        output_sums, output_counts, sf_type, backend, x_mat, u_mat, bin_edges;
        kwargs...,
    )
end

function SFC.gpu_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    sf_type::SFT.AbstractStructureFunctionType,
    backend::KA.Backend,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector{<:Tuple};
    kwargs...,
) where {OT, CT, FT1 <: Number, FT2 <: Number}
    FT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins)
    bin_edges = Vector{FT}(undef, n_bins + 1)
    for k in 1:n_bins
        bin_edges[k] = distance_bins[k][1]
    end
    bin_edges[end] = distance_bins[end][2]

    x_mat = FT === FT1 && x_arr isa Matrix{FT1} ? x_arr : Matrix{FT}(x_arr)
    u_mat = FT === FT2 && u_arr isa Matrix{FT2} ? u_arr : Matrix{FT}(u_arr)

    return SFC.gpu_calculate_structure_function!(
        output_sums, output_counts, sf_type, backend, x_mat, u_mat, bin_edges;
        kwargs...,
    )
end

function _gpu_calculate_structure_function_core(
    sf_type::SFT.AbstractStructureFunctionType,
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    distance_bins::AbstractVector{FT},
    ::Val{RSAC};
    workgroup_size::Int = 64,
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT, RSAC, CT}
    out_dev, cnt_dev, edges_host = _launch_gpu_structure_function!(
        sf_type, backend, x_mat, u_mat, distance_bins;
        workgroup_size = workgroup_size,
        kwargs...,
    )
    N_bins = length(edges_host)
    output, counts = _download_gpu_sf_results(out_dev, cnt_dev, FT, CT)

    # Convert distance_bins edges to tuples for SF objects
    bin_tuples = SA.SVector{N_bins - 1, Tuple{FT, FT}}(
        [(edges_host[i], edges_host[i + 1]) for i in 1:(N_bins - 1)]...,
    )

    if RSAC
        return SF.StructureFunctionSumsAndCounts(sf_type, bin_tuples, output, counts)
    else
        output_div = similar(output)
        for k in eachindex(output)
            c = counts[k]
            output_div[k] = c == 0 ? FT(NaN) : output[k] / c
        end
        return SF.StructureFunction(sf_type, bin_tuples, output_div)
    end
end


# ---------------------------------------------------------------------------
# Spectral Analysis API Extension
# ---------------------------------------------------------------------------

function SFSA.gpu_calculate_spectrum(
    backend::KA.Backend,
    x_vecs::Tuple,
    u_vecs::Tuple,
    ms::NTuple{D, Int};
    iflag::Int = 1,
    domain_size::Union{Nothing, Tuple} = nothing,
    workgroup_size::Int = 16,
) where {D}
    FT = eltype(x_vecs[1])
    N = length(x_vecs[1])
    NU = length(u_vecs)

    # 1. Coordinate ranges (replicated from SpectralAnalysis.jl)
    ranges = ntuple(Val(D)) do d
        if domain_size !== nothing
            return domain_size[d]
        else
            min_x, max_x = extrema(x_vecs[d])
            return max_x - min_x
        end
    end

    ks_phys = ntuple(
        d ->
            range(FT(-ms[d] ÷ 2), stop = FT((ms[d] - 1) ÷ 2), length = ms[d]) .*
            (FT(2π) / (ranges[d] == 0 ? one(FT) : ranges[d])),
        Val(D),
    )

    # 2. Allocate and transfer
    # Standardize to 3D padding for SVector compatibility in kernel
    x_mat = zeros(FT, 3, N)
    u_mat = zeros(FT, NU, N)
    for d in 1:D
        x_mat[d, :] .= x_vecs[d]
    end
    for u_idx in 1:NU
        u_mat[u_idx, :] .= u_vecs[u_idx]
    end

    x_dev = KA.allocate(backend, FT, 3, N)
    u_dev = KA.allocate(backend, FT, NU, N)
    copyto!(x_dev, x_mat)
    copyto!(u_dev, u_mat)

    # Transfer ks_phys as vectors, padded to 3
    ks_phys_dev = ntuple(d_ -> begin
            if d_ <= D
                v = KA.allocate(backend, FT, length(ks_phys[d_]))
                copyto!(v, collect(ks_phys[d_]))
                return v
            else
                # Dummy for padding
                return KA.allocate(backend, FT, 1)
            end
        end, Val(3))

    coeffs_dev = KA.zeros(backend, Complex{FT}, ms..., NU)

    # 3. Launch kernel
    # ndrange is the grid of wavenumbers ms...
    kernel! = _spectral_kernel!(backend, workgroup_size)
    kernel!(
        coeffs_dev,
        x_dev, u_dev,
        ks_phys_dev,
        iflag,
        N, NU, D,
        ms;
        ndrange = ms,
    )
    KA.synchronize(backend)

    return Array(coeffs_dev), ks_phys
end

# ---------------------------------------------------------------------------
# Single-Pass GPU Kernels
# ---------------------------------------------------------------------------

KA.@kernel function _sf_single_pass_kernel_linear!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    N_points::Int,
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {FT}
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]

    if i < j
        X1 = SA.SVector{2}(x_mat[1, i], x_mat[2, i])
        X2 = SA.SVector{2}(x_mat[1, j], x_mat[2, j])
        U1 = SA.SVector{2}(u_mat[1, i], u_mat[2, i])
        U2 = SA.SVector{2}(u_mat[1, j], u_mat[2, j])

        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2)

        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )

        if 1 <= bin < N_bins
            dU = U2 - U1
            r̂ = dX / dist
            n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])

            du_L = SA.dot(dU, r̂)
            du_T = SA.dot(dU, n̂)

            du_L2 = du_L * du_L
            du_T2 = du_T * du_T

            @atomic output_sums[1, bin] += du_L2 + du_T2
            @atomic output_sums[2, bin] += du_L2
            @atomic output_sums[3, bin] += du_T2
            @atomic output_sums[4, bin] += du_L * (du_L2 + du_T2)
            @atomic output_sums[5, bin] += du_L * du_L2
            @atomic output_sums[6, bin] += du_L2 * du_T
            @atomic output_sums[7, bin] += du_L * du_T2
            @atomic output_sums[8, bin] += du_T * du_T2

            for t in 1:8
                @atomic output_counts[t, bin] += one(eltype(output_counts))
            end
        end
    end
end

KA.@kernel function _sf_single_pass_kernel_log!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_mat),
    @Const(edges),
    @Const(lut),
    N_points::Int,
    N_bins::Int,
    e_min::Int,
)
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]

    if i < j
        X1 = SA.SVector{2}(x_mat[1, i], x_mat[2, i])
        X2 = SA.SVector{2}(x_mat[1, j], x_mat[2, j])
        U1 = SA.SVector{2}(u_mat[1, i], u_mat[2, i])
        U2 = SA.SVector{2}(u_mat[1, j], u_mat[2, j])

        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2)

        bin = _gpu_digitize_log(dist, edges, lut, e_min, N_bins)

        if 1 <= bin < N_bins
            dU = U2 - U1
            r̂ = dX / dist
            FT = eltype(x_mat)
            n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])

            du_L = SA.dot(dU, r̂)
            du_T = SA.dot(dU, n̂)
            du_L2 = du_L * du_L
            du_T2 = du_T * du_T

            @atomic output_sums[1, bin] += du_L2 + du_T2
            @atomic output_sums[2, bin] += du_L2
            @atomic output_sums[3, bin] += du_T2
            @atomic output_sums[4, bin] += du_L * (du_L2 + du_T2)
            @atomic output_sums[5, bin] += du_L * du_L2
            @atomic output_sums[6, bin] += du_L2 * du_T
            @atomic output_sums[7, bin] += du_L * du_T2
            @atomic output_sums[8, bin] += du_T * du_T2

            for t in 1:8
                @atomic output_counts[t, bin] += one(eltype(output_counts))
            end
        end
    end
end

KA.@kernel function _sf_single_pass_kernel!(
    output_sums,                 # Matrix{FT} of size (8, N_bins-1)
    output_counts,               # Matrix{FT} of size (8, N_bins-1)
    @Const(x_mat),               # Matrix{FT} of size (2, N_points)
    @Const(u_mat),               # Matrix{FT} of size (2, N_points)
    @Const(distance_bins),       # monotone bin edges, length N_bins
    N_points::Int,
    N_bins::Int,
)
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]
    
    if i < j
        # Static arrays on stack (using 2D coordinates)
        X1 = SA.SVector{2}(x_mat[1, i], x_mat[2, i])
        X2 = SA.SVector{2}(x_mat[1, j], x_mat[2, j])
        U1 = SA.SVector{2}(u_mat[1, i], u_mat[2, i])
        U2 = SA.SVector{2}(u_mat[1, j], u_mat[2, j])
        
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2)
        
        bin = _gpu_digitize_general(dist, distance_bins, N_bins)
        
        if 1 <= bin < N_bins
            dU = U2 - U1
            r̂ = dX / dist
            n̂ = SA.SVector{2, eltype(x_mat)}(r̂[2], -r̂[1])
            
            du_L = SA.dot(dU, r̂)
            du_T = SA.dot(dU, n̂)
            
            du_L2 = du_L * du_L
            du_T2 = du_T * du_T
            
            # Atomically accumulate the 8 structure functions
            @atomic output_sums[1, bin] += du_L2 + du_T2
            @atomic output_sums[2, bin] += du_L2
            @atomic output_sums[3, bin] += du_T2
            @atomic output_sums[4, bin] += du_L * (du_L2 + du_T2)
            @atomic output_sums[5, bin] += du_L * du_L2
            @atomic output_sums[6, bin] += du_L2 * du_T
            @atomic output_sums[7, bin] += du_L * du_T2
            @atomic output_sums[8, bin] += du_T * du_T2
            
            for t in 1:8
                @atomic output_counts[t, bin] += one(eltype(output_counts))
            end
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
    layout::_GPUBinLayout,
    N_points::Int,
    n_edges::Int,
)
    bins = _gpu_active_bins(layout)
    return _launch_single_pass_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        bins, N_points, n_edges,
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
    n_edges::Int,
)
    kernel! = _sf_single_pass_kernel_linear!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_edges,
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
    n_edges::Int,
)
    FT = eltype(lbe.edges)
    edges_dev = KA.allocate(backend, FT, n_edges)
    lut_dev = KA.allocate(backend, Int32, length(lbe.lut))
    copyto!(edges_dev, collect(lbe.edges))
    copyto!(lut_dev, Int32.(lbe.lut))
    kernel! = _sf_single_pass_kernel_log!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        edges_dev, lut_dev, N_points, n_edges, lbe.e_min;
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
    n_edges::Int,
) where {FT}
    bins_dev = KA.allocate(backend, FT, n_edges)
    copyto!(bins_dev, edges)
    kernel! = _sf_single_pass_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        bins_dev, N_points, n_edges;
        ndrange = (N_points, N_points),
    )
    return nothing
end

"""
    SFC._dispatch_single_pass(::GPUBackend, x, u, distance_bins; workgroup_size=64, kwargs...)

Calculates single-pass structure functions utilizing GPU-accelerated computing.
"""
function SFC._dispatch_single_pass(
    gpu_backend::SF.GPUBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    workgroup_size::Int = 64,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    backend = gpu_backend.backend
    FT = promote_type(float(FT1), float(FT2))
    N_dims, N_points = size(x)
    layout = _resolve_gpu_bin_layout(distance_bins)
    edges_host = _layout_edge_vector(layout)
    n_edges = length(edges_host)
    n_bins = n_edges - 1

    if N_dims != 2
        error("GPUExt: single-pass calculation only supports 2D coordinates (got N_dims=$N_dims)")
    end

    x_dev = KA.allocate(backend, FT, 2, N_points)
    u_dev = KA.allocate(backend, FT, 2, N_points)
    out_sums_dev = KA.zeros(backend, FT, 8, n_bins)
    out_cnts_dev = KA.zeros(backend, FT, 8, n_bins)

    copyto!(x_dev, collect(x))
    copyto!(u_dev, collect(u))

    _launch_single_pass_kernel!(
        backend, workgroup_size,
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        layout, N_points, n_edges,
    )
    KA.synchronize(backend)

    sums = Array(out_sums_dev)
    counts = Int64.(Array(out_cnts_dev))

    return SFC.postprocess_single_pass_results(sums, counts, edges_host)
end

end # module GPUExt
