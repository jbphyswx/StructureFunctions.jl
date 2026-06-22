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

"""
    _gpu_calculate_structure_function_batch(sf_type, backend, x, u, distance_bins, ::Val{RSAC}; ...)

Fused GPU batch driver for individual 1D structure functions.
"""
function _gpu_calculate_structure_function_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT},
    ::Val{RSAC};
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT, RSAC, CT}
    fixed_x = ndims(x) == 2
    lbe = _gpu_batch_linear_distance_bins(distance_bins)
    NB = length(lbe.edges) - 1
    N = size(x, 2)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    sums, counts = _gpu_batch_allocate_outputs(FT, NB, bdims)
    sums_dev = KA.adapt(backend, sums)
    counts_dev = KA.adapt(backend, counts)
    sums_dev_flat = reshape(sums_dev, NB, B)
    counts_dev_flat = reshape(counts_dev, NB, B)
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = fixed_x)
    if fixed_x
        _launch_batch_fixed_x_sf!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, sf_type, N, B, lbe)
    else
        _launch_batch_varying_x_sf!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, sf_type, N, B, lbe)
    end
    _gpu_batch_download!(sums, counts, sums_dev, counts_dev)
    if RSAC
        return (sums = sums, counts = CT === UInt32 ? counts : CT.(counts),
                operator = sf_type, distance_bins = distance_bins)
    else
        out_div = similar(sums)
        @inbounds for k in eachindex(sums)
            c = counts[k]
            out_div[k] = c == 0 ? FT(NaN) : sums[k] / c
        end
        return SF.StructureFunction(sf_type, distance_bins, out_div)
    end
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
    lbe = _gpu_batch_linear_distance_bins(distance_bins)
    NB = length(lbe.edges) - 1
    N = size(x, 2)
    B = SFC.batch_size(u)
    sums_dev = KA.adapt(backend, zeros(eltype(output_sums), NB, SFC.batch_dims(u)...))
    counts_dev = KA.adapt(backend, zeros(UInt32, NB, SFC.batch_dims(u)...))
    sums_dev_flat = reshape(sums_dev, NB, B)
    counts_dev_flat = reshape(counts_dev, NB, B)
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = fixed_x)
    if fixed_x
        _launch_batch_fixed_x_sf!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, sf_type, N, B, lbe)
    else
        _launch_batch_varying_x_sf!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, sf_type, N, B, lbe)
    end
    _accumulate_batch_host!(output_sums, output_counts, sums_dev, counts_dev)
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
    lbe = _gpu_batch_linear_distance_bins(distance_bins)
    NB = length(lbe.edges) - 1
    N = size(x, 2)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    sums, counts = _gpu_batch_allocate_sp1d(FT, NB, bdims)
    sums_dev = KA.adapt(backend, sums)
    counts_dev = KA.adapt(backend, counts)
    sums_dev_flat = reshape(sums_dev, SFC.SINGLE_PASS_N, NB, B)
    counts_dev_flat = reshape(counts_dev, SFC.SINGLE_PASS_N, NB, B)
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = fixed_x)
    if fixed_x
        _launch_batch_fixed_x_sp1d!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, N, B, lbe)
    else
        _launch_batch_varying_x_sp1d!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, N, B, lbe)
    end
    _gpu_batch_download!(sums, counts, sums_dev, counts_dev)
    return (sums = sums, counts = CT === UInt32 ? counts : CT.(counts))
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
    lbe = _gpu_batch_linear_distance_bins(distance_bins)
    N = size(x, 2)
    B = SFC.batch_size(u)
    sums_dev = KA.adapt(backend, zeros(OT, size(sums)...))
    counts_dev = KA.adapt(backend, zeros(UInt32, size(counts)...))
    sums_dev_flat = reshape(sums_dev, SFC.SINGLE_PASS_N, NB, B)
    counts_dev_flat = reshape(counts_dev, SFC.SINGLE_PASS_N, NB, B)
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = fixed_x)
    if fixed_x
        _launch_batch_fixed_x_sp1d!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, N, B, lbe)
    else
        _launch_batch_varying_x_sp1d!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, N, B, lbe)
    end
    tmp_s = Array(sums_dev)
    tmp_c = Array(counts_dev)
    sums .+= tmp_s
    if CT === UInt32
        counts .+= tmp_c
    else
        counts .+= CT.(tmp_c)
    end
    return sums, counts
end

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
    lbe = _gpu_batch_linear_distance_bins(distance_bins)
    n_dist = length(distance_bins) - 1
    n_val = _sp2d_n_val_edges(value_bins) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    val_plan = _gpu_build_value_digitize_plan(backend, value_bins)
    N = size(x, 2)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    sums, counts = _gpu_batch_allocate_sp2d(FT, n_dist, n_val, bdims)
    sums_dev = KA.adapt(backend, sums)
    counts_dev = KA.adapt(backend, counts)
    sums_dev_flat = reshape(sums_dev, SFC.SINGLE_PASS_N, n_dist, n_val, B)
    counts_dev_flat = reshape(counts_dev, SFC.SINGLE_PASS_N, n_dist, n_val, B)
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = fixed_x)
    if fixed_x
        _launch_batch_fixed_x_sp2d!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, N, B, lbe, val_plan, n_dist, n_val)
    else
        _launch_batch_varying_x_sp2d!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, N, B, lbe, val_plan, n_dist, n_val)
    end
    _gpu_batch_download!(sums, counts, sums_dev, counts_dev)
    return (sums = sums, counts = CT === UInt32 ? counts : CT.(counts))
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
    lbe = _gpu_batch_linear_distance_bins(distance_bins)
    n_dist = length(distance_bins) - 1
    n_val = size(sums, 3)
    val_plan = _gpu_build_value_digitize_plan(backend, value_bins)
    N = size(x, 2)
    B = SFC.batch_size(u)
    sums_dev = KA.adapt(backend, zeros(OT, size(sums)...))
    counts_dev = KA.adapt(backend, zeros(UInt32, size(counts)...))
    sums_dev_flat = reshape(sums_dev, SFC.SINGLE_PASS_N, n_dist, n_val, B)
    counts_dev_flat = reshape(counts_dev, SFC.SINGLE_PASS_N, n_dist, n_val, B)
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = fixed_x)
    if fixed_x
        _launch_batch_fixed_x_sp2d!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, N, B, lbe, val_plan, n_dist, n_val)
    else
        _launch_batch_varying_x_sp2d!(backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, N, B, lbe, val_plan, n_dist, n_val)
    end
    tmp_s = Array(sums_dev)
    tmp_c = Array(counts_dev)
    sums .+= tmp_s
    if CT === UInt32
        counts .+= tmp_c
    else
        counts .+= CT.(tmp_c)
    end
    return sums, counts
end

function _gpu_batch_allocate_joint2d(FT::Type, n_dist::Int, n_val::Int, bdims::Dims)
    sums = zeros(FT, n_dist, n_val, bdims...)
    counts = zeros(UInt32, n_dist, n_val, bdims...)
    return sums, counts
end

"""Fused GPU batch driver for single-type joint 2D structure functions."""
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
    N = size(x, 2)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    sums, counts = _gpu_batch_allocate_joint2d(FT, n_dist, n_val, bdims)
    sums_dev = KA.adapt(backend, sums)
    counts_dev = KA.adapt(backend, counts)
    sums_dev_flat = reshape(sums_dev, n_dist, n_val, B)
    counts_dev_flat = reshape(counts_dev, n_dist, n_val, B)
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = fixed_x)
    if fixed_x
        _launch_batch_fixed_x_joint2d!(
            backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, sf_type, N, B,
            distance_bins, value_bins, n_dist, n_val,
        )
    else
        _launch_batch_varying_x_joint2d!(
            backend, sums_dev_flat, counts_dev_flat, x_dev, u_dev, sf_type, N, B,
            distance_bins, value_bins, n_dist, n_val,
        )
    end
    _gpu_batch_download!(sums, counts, sums_dev, counts_dev)
    return SF.StructureFunction2D(
        sf_type, distance_bins, value_bins, sums, CT === UInt32 ? counts : CT.(counts),
    )
end
