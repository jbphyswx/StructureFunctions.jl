# GPU batch routing — fixed-x / varying-x fused launches.
# Included from StructureFunctionsGPUExt.jl after BatchLaunch.jl.

"""Linear distance bins required by batch tiled GPU kernels (FMA digitize only)."""
function _gpu_batch_linear_distance_bins(distance_bins)
    if distance_bins isa LinearBinEdges
        return distance_bins
    elseif distance_bins isa AbstractRange
        return LinearBinEdges(distance_bins)
    elseif distance_bins isa BinEdges
        return _gpu_batch_linear_distance_bins(distance_bins.edges)
    elseif distance_bins isa AbstractVector
        first_val = first(distance_bins)
        last_val = last(distance_bins)
        len = length(distance_bins)
        if len >= 2
            r = range(first_val, last_val; length = len)
            if all(i -> isapprox(distance_bins[i], r[i]; atol = 1e-12), 1:len)
                return LinearBinEdges(r)
            end
        end
    end
    throw(ArgumentError(
        "GPU batch tiled kernels require linear distance bins (got $(typeof(distance_bins))). " *
        "Use evenly spaced edges or LinearBinEdges; log/general batch GPU kernels are not implemented.",
    ))
end

function _gpu_batch_allocate_outputs(FT::Type, NB::Int, bdims::Dims)
    sums = zeros(FT, NB, bdims...)
    counts = zeros(UInt32, NB, bdims...)
    return sums, counts
end

function _gpu_batch_allocate_sp1d(FT::Type, NB::Int, bdims::Dims)
    sums = zeros(FT, SFC.SINGLE_PASS_N, NB, bdims...)
    counts = zeros(UInt32, SFC.SINGLE_PASS_N, NB, bdims...)
    return sums, counts
end

function _gpu_batch_allocate_sp2d(FT::Type, n_dist::Int, n_val::Int, bdims::Dims)
    sums = zeros(FT, SFC.SINGLE_PASS_N, n_dist, n_val, bdims...)
    counts = zeros(UInt32, SFC.SINGLE_PASS_N, n_dist, n_val, bdims...)
    return sums, counts
end

function _gpu_batch_download!(sums, counts, sums_dev, counts_dev)
    copy!(sums, Array(sums_dev))
    copy!(counts, Array(counts_dev))
    return nothing
end

"""Unified 1D batch device launch (individual `NMOM=1` or single-pass `NMOM=6`).
Routes through `_sf_launch_1d_batch!`, taking the CUDA fast path (N-body
broadcast + static-shared privatized histogram, TILE=256) when
`StructureFunctionsCUDAExt` is active, else the portable KA tiled kernel. Covers
fixed-x and varying-x, `D ∈ {2,3}`, any distance-bin type
(`_sf_batch_dist_digitizer`). Returns host `(sums, counts)` of shape
`(NMOM, NB, B)`. `u` is staged `(D,N,B)` with NO batch-major permute."""
function _gpu_1d_unified_device(
    backend, x, u, sf_type, distance_bins,
    ::Val{NMOM}, NB::Int, B::Int, fixed_x::Bool, ::Type{OT},
) where {NMOM, OT}
    D = size(x, 1)
    N = size(x, 2)
    dig = _sf_batch_dist_digitizer(backend, distance_bins)
    out_dev = KA.adapt(backend, zeros(OT, NMOM, NB, B))
    cnt_dev = KA.adapt(backend, zeros(UInt32, NMOM, NB, B))
    if fixed_x
        x_dev = KA.adapt(backend, x)                       # (D, N)
        u_dev = KA.adapt(backend, reshape(u, D, N, B))     # (D, N, B), no permute
    else
        x_dev = KA.adapt(backend, reshape(x, D, N, B))
        u_dev = KA.adapt(backend, reshape(u, D, N, B))
    end
    _sf_launch_1d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dig,
                         N, NB, B, D, Val(NMOM), fixed_x)
    KA.synchronize(backend)
    return Array(out_dev), Array(cnt_dev)
end

"""Linear distance bins or `nothing` (non-throwing variant of
`_gpu_batch_linear_distance_bins`)."""
function _try_linear_distance_bins(distance_bins)
    try
        return _gpu_batch_linear_distance_bins(distance_bins)
    catch
        return nothing
    end
end

"""Individual (NMOM=1) 1D batch device launch, routed to the measured-optimal
kernel per regime (job 239164, histograms verified equal):
- fixed-x + linear bins → OLD global warp-replica + W-strip kernel (tiny-histogram
  contention-bound regime: 147 vs 115 bapps for N-body).
- varying-x, or general (non-linear) fixed-x bins → N-body broadcast (115 vs 61
  bapps for the old kernel on varying-x; also handles general bins).
Returns host `(sums, counts)` of shape `(NB, B)`."""
function _gpu_1d_individual_device(backend, sf_type, x, u, distance_bins,
                                   NB::Int, B::Int, fixed_x::Bool, ::Type{OT}) where {OT}
    lbe = fixed_x ? _try_linear_distance_bins(distance_bins) : nothing
    if lbe !== nothing
        N = size(x, 2)
        sums_dev = KA.adapt(backend, zeros(OT, NB, B))
        counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
        x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = true)
        _launch_batch_fixed_x_sf!(backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B, lbe)
        return Array(sums_dev), Array(counts_dev)
    end
    oh, ch = _gpu_1d_unified_device(backend, x, u, sf_type, distance_bins, Val(1), NB, B, fixed_x, OT)
    return reshape(oh, NB, B), reshape(ch, NB, B)
end

"""
    _gpu_calculate_structure_function_batch(sf_type, backend, x, u, distance_bins; ...)

Fused GPU batch driver for individual 1D structure functions.
"""
function _gpu_calculate_structure_function_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT};
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT, CT}
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    out_host, cnt_host = _gpu_1d_individual_device(backend, sf_type, x, u, distance_bins, NB, B, fixed_x, FT)
    sums = reshape(out_host, NB, bdims...)
    counts = reshape(cnt_host, NB, bdims...)
    return SF.StructureFunctionSumsAndCounts(
        sf_type, distance_bins, sums, CT === UInt32 ? counts : CT.(counts),
    )
end

function _gpu_calculate_structure_function_batch!(
    output_sums,
    output_counts,
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT};
    kwargs...,
) where {FT}
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    out_host, cnt_host = _gpu_1d_individual_device(
        backend, sf_type, x, u, distance_bins, NB, B, fixed_x, eltype(output_sums))
    output_sums .+= reshape(out_host, size(output_sums)...)
    cflat = reshape(cnt_host, size(output_counts)...)
    if eltype(output_counts) === UInt32
        output_counts .+= cflat
    else
        output_counts .+= eltype(output_counts).(cflat)
    end
    return nothing
end

function _accumulate_batch_host!(output_sums, output_counts, sums_dev, counts_dev)
    tmp_s = Array(sums_dev)
    tmp_c = Array(counts_dev)
    output_sums .+= tmp_s
    if eltype(output_counts) === UInt32
        output_counts .+= tmp_c
    else
        @inbounds for k in eachindex(output_counts)
            output_counts[k] += eltype(output_counts)(tmp_c[k])
        end
    end
    return nothing
end

function _gpu_dispatch_single_pass_batch(
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3};
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1, FT2, FT3, CT}
    FT = promote_type(float(FT1), float(FT2))
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    out_host, cnt_host = _gpu_1d_unified_device(
        backend, x, u, nothing, distance_bins, Val(SFC.SINGLE_PASS_N), NB, B, fixed_x, FT)
    sums = reshape(out_host, SFC.SINGLE_PASS_N, NB, bdims...)
    counts_u32 = reshape(cnt_host, SFC.SINGLE_PASS_N, NB, bdims...)
    return (sums = sums, counts = CT === UInt32 ? counts_u32 : CT.(counts_u32))
end

function _gpu_dispatch_single_pass_batch!(
    sums::AbstractArray{OT},
    counts::AbstractArray{CT},
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3};
    kwargs...,
) where {OT, CT, FT1, FT2, FT3}
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    out_host, cnt_host = _gpu_1d_unified_device(
        backend, x, u, nothing, distance_bins, Val(SFC.SINGLE_PASS_N), NB, B, fixed_x, OT)
    sums .+= reshape(out_host, size(sums)...)
    cflat = reshape(cnt_host, size(counts)...)
    if CT === UInt32
        counts .+= cflat
    else
        counts .+= CT.(cflat)
    end
    return sums, counts
end

"""Unified single-pass 2D batch device launch. Routes through the same
`_sf_launch_2d_batch!` chokepoint as joint 2D, so it takes the CUDA fast path
(N-body broadcast + dynamic-shared privatized histogram, TILE=1024) when
`StructureFunctionsCUDAExt` is active, and the portable KA tiled kernel
otherwise. Covers fixed-x and varying-x, `D ∈ {2,3}`, and any distance-bin type
(`_sf_batch_dist_digitizer`). Returns host `(sums, counts)` of shape
`(6, n_dist, n_val, B)`. `u` is staged `(D,N,B)` with NO batch-major permute (the
unified kernels read `u[d, point, b]` directly)."""
function _gpu_2d_unified_device(
    backend, x, u, sf_type, distance_bins, value_bins, ::Val{NMOM},
    n_dist::Int, n_val::Int, B::Int, fixed_x::Bool, ::Type{OT},
) where {NMOM, OT}
    D = size(x, 1)
    N = size(x, 2)
    ddig = _sf_batch_dist_digitizer(backend, distance_bins)
    vplan = _gpu_build_value_digitize_plan(backend, value_bins)
    out_dev = KA.adapt(backend, zeros(OT, NMOM, n_dist, n_val, B))
    cnt_dev = KA.adapt(backend, zeros(UInt32, NMOM, n_dist, n_val, B))
    if fixed_x
        x_dev = KA.adapt(backend, x)                       # (D, N)
        u_dev = KA.adapt(backend, reshape(u, D, N, B))     # (D, N, B), no permute
    else
        x_dev = KA.adapt(backend, reshape(x, D, N, B))
        u_dev = KA.adapt(backend, reshape(u, D, N, B))
    end
    _sf_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                         N, n_dist, n_val, B, D, Val(NMOM), fixed_x)
    KA.synchronize(backend)
    return Array(out_dev), Array(cnt_dev)
end

"""Single-pass (NMOM=6) 2D batch device launch — thin wrapper over the unified
`_gpu_2d_unified_device`."""
_gpu_sp2d_unified_device(backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, ::Type{OT}) where {OT} =
    _gpu_2d_unified_device(backend, x, u, nothing, distance_bins, value_bins,
                           Val(SFC.SINGLE_PASS_N), n_dist, n_val, B, fixed_x, OT)

function _gpu_dispatch_single_pass_2d_batch(
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1, FT2, FT3, CT}
    FT = promote_type(float(FT1), float(FT2))
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = _sp2d_n_val_edges(value_bins) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    out_host, cnt_host = _gpu_sp2d_unified_device(
        backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, FT)
    sums = reshape(out_host, SFC.SINGLE_PASS_N, n_dist, n_val, bdims...)
    counts_u32 = reshape(cnt_host, SFC.SINGLE_PASS_N, n_dist, n_val, bdims...)
    return (sums = sums, counts = CT === UInt32 ? counts_u32 : CT.(counts_u32))
end

function _gpu_dispatch_single_pass_2d_batch!(
    sums::AbstractArray{OT},
    counts::AbstractArray{CT},
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    kwargs...,
) where {OT, CT, FT1, FT2, FT3}
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = size(sums, 3)
    B = SFC.batch_size(u)
    out_host, cnt_host = _gpu_sp2d_unified_device(
        backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, OT)
    sums .+= reshape(out_host, size(sums)...)
    if CT === UInt32
        counts .+= reshape(cnt_host, size(counts)...)
    else
        counts .+= CT.(reshape(cnt_host, size(counts)...))
    end
    return sums, counts
end

function _gpu_batch_allocate_joint2d(FT::Type, n_dist::Int, n_val::Int, bdims::Dims)
    sums = zeros(FT, n_dist, n_val, bdims...)
    counts = zeros(UInt32, n_dist, n_val, bdims...)
    return sums, counts
end

"""Fused GPU batch driver for single-type joint 2D structure functions.

Unified path: one fused tiled launch (`sf_tiled_2d_*`) per fixed/varying mode —
no host `for b` loop, no naive per-cell kernel, no per-iteration allocations.
This is the path that fixes the prior batch joint2d performance regression."""
function _gpu_calculate_structure_function_2d_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT},
    value_bins::AbstractVector{FT};
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT, CT}
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = length(value_bins) - 1
    D = size(x, 1)
    N = size(x, 2)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)

    ddig = _sf_batch_dist_digitizer(backend, distance_bins)
    vplan = _gpu_build_value_digitize_plan(backend, value_bins)

    out_dev = KA.adapt(backend, zeros(FT, 1, n_dist, n_val, B))
    cnt_dev = KA.adapt(backend, zeros(UInt32, 1, n_dist, n_val, B))

    if fixed_x
        x_dev = KA.adapt(backend, x)                       # (D, N)
        u_dev = KA.adapt(backend, reshape(u, D, N, B))     # (D, N, B) — no permute
    else
        x_dev = KA.adapt(backend, reshape(x, D, N, B))
        u_dev = KA.adapt(backend, reshape(u, D, N, B))
    end
    _sf_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                         N, n_dist, n_val, B, D, Val(1), fixed_x)
    KA.synchronize(backend)

    sums = reshape(Array(out_dev)[1, :, :, :], n_dist, n_val, bdims...)
    counts_u32 = reshape(Array(cnt_dev)[1, :, :, :], n_dist, n_val, bdims...)
    counts = CT === UInt32 ? counts_u32 : CT.(counts_u32)
    return SF.StructureFunction2DSumsAndCounts(sf_type, distance_bins, value_bins, sums, counts)
end
