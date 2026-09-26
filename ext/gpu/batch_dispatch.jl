# GPU batch routing — fixed-x / varying-x fused launches.

function _batch_buffers(backend, ::Type{FT}, ::Type{CT}, dims, workspace) where {FT, CT}
    if workspace === nothing
        return KA.zeros(backend, FT, dims...), KA.zeros(backend, CT, dims...)
    end
    _array_on_backend(workspace.out_sums_dev, backend) || throw(ArgumentError("workspace backend differs from execution backend"))
    _ws_float_type(workspace) === FT || throw(ArgumentError("workspace sum precision differs from batch precision"))
    cached = workspace.lazy.batch
    if cached === nothing || size(cached.sums) != dims || eltype(cached.counts) !== CT
        cached = SFC.GPUBatchBuffers(KA.zeros(backend, FT, dims...), KA.zeros(backend, CT, dims...))
        workspace.lazy.batch = cached
    else
        fill!(cached.sums, zero(FT))
        fill!(cached.counts, zero(CT))
    end
    return cached.sums, cached.counts
end

"""Unified 1D batch device launch (individual `NMOM=1` or single-pass `NMOM=6`) through
`_sf_launch_1d_batch!`: fixed-x and varying-x, any width, any distance-bin type. Returns device
`(sums, counts)` of shape `(NMOM, NB, B)`. `u` is staged `(D,N,B)` with no batch-major permute."""
function _gpu_1d_unified_device(
    backend, x, u, sf_type, distance_bins,
    ::Val{NMOM}, NB::Int, B::Int, fixed_x::Bool, ::Type{OT}, ::Type{CT}, geom;
    weights = SFC.NoWeights(), workspace = nothing,
) where {NMOM, OT, CT}
    _validate_batch_workspace!(workspace, backend, NMOM == 1 ? :sf1d : :single_pass, distance_bins, OT)
    D = size(x, 1)
    N = size(x, 2)
    dig = _dist_digitizer(workspace, backend, distance_bins, Val(NMOM == 1 ? :sf1d : :single_pass))
    wts = _sf_weights_to_device(backend, weights)
    CNT = _sf_count_type(wts, CT, _sf_worst_case_pairs(N))
    out_dev, cnt_dev = _batch_buffers(backend, OT, CNT, (NMOM, NB, B), workspace)
    B == 0 && return out_dev, cnt_dev
    if fixed_x
        x_dev = KA.adapt(backend, x)                       # (D, N)
        u_dev = KA.adapt(backend, reshape(u, D, N, B))     # (D, N, B), no permute
    else
        x_dev = KA.adapt(backend, reshape(x, D, N, B))
        u_dev = KA.adapt(backend, reshape(u, D, N, B))
    end
    _sf_launch_1d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dig,
                         N, NB, B, D, Val(NMOM), fixed_x, geom; weights = wts)
    KA.synchronize(backend)
    return out_dev, cnt_dev
end

"""Individual batch distance histograms using fixed-position or unified tiles."""
function _gpu_1d_individual_device(backend, sf_type, x, u, distance_bins,
                                   NB::Int, B::Int, fixed_x::Bool, ::Type{OT}, ::Type{CT}, geom;
                                   weights = SFC.NoWeights(), workspace = nothing) where {OT, CT}
    two_wide = SFC._val_int(SFH.coordinate_width(geom)) == 2 &&
               SFC._val_int(SFH.field_width(geom)) == 2
    _validate_batch_workspace!(workspace, backend, :sf1d, distance_bins, OT)
    if fixed_x && two_wide && weights isa SFC.NoWeights && NB <= SF_GPU_MAX_BINS &&
       _sf_worst_case_pairs(size(x, 2)) <= typemax(UInt32) &&
       _batch_usmem_strip_w(SFC.gpu_device_caps(backend), OT) > 0
        N = size(x,2)
        sums_dev, counts_dev = _batch_buffers(backend, OT,
            _sf_count_type(weights, CT, _sf_worst_case_pairs(N)), (NB,B), workspace)
        B == 0 && return sums_dev, counts_dev
        x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x=true)
        _launch_batch_fixed_x_sf!(backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B,
                                  _dist_digitizer(workspace, backend, distance_bins, Val(:sf1d)), NB, geom)
        return sums_dev, counts_dev
    end
    sums, counts = _gpu_1d_unified_device(backend, x, u, sf_type, distance_bins,
        Val(1), NB, B, fixed_x, OT, CT, geom; weights, workspace)
    return reshape(sums, NB, B), reshape(counts, NB, B)
end

"""
    _gpu_calculate_structure_function_batch(sf_type, backend, x, u, distance_bins, CT; ...)

Fused GPU batch driver for individual 1D structure functions.
"""
function _gpu_calculate_structure_function_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT},
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), workspace = nothing,
) where {FT, CT}
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    out_device, cnt_device = _gpu_1d_individual_device(backend, sf_type, x, u, distance_bins, NB, B, fixed_x, FT, CT,
        geom; weights, workspace)
    out_device, cnt_device = _owned_gpu_results(out_device, cnt_device, CT; workspace)
    sums = reshape(out_device, NB, bdims...)
    counts = reshape(cnt_device, NB, bdims...)
    return SF.StructureFunctionSumsAndCounts(
        sf_type, distance_bins, sums, eltype(counts) === CT ? counts : CT.(counts),
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
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), workspace = nothing,
) where {FT}
    _check_gpu_outputs(output_sums, output_counts, backend, (length(distance_bins)-1, SFC.batch_dims(u)...); workspace)
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    CT = eltype(output_counts)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    out_device, cnt_device = _gpu_1d_individual_device(
        backend, sf_type, x, u, distance_bins, NB, B, fixed_x, eltype(output_sums), CT,
        geom; weights, workspace)
    output_sums .+= reshape(out_device, size(output_sums)...)
    cflat = reshape(cnt_device, size(output_counts)...)
    if eltype(cflat) === CT
        output_counts .+= cflat
    else
        output_counts .+= CT.(cflat)
    end
    return nothing
end

function _gpu_dispatch_single_pass_batch(
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), workspace = nothing,
) where {FT1, FT2, FT3, CT}
    FT = promote_type(float(FT1), float(FT2))
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    out_device, cnt_device = _gpu_1d_unified_device(
        backend, x, u, nothing, distance_bins, Val(SFC.SINGLE_PASS_N), NB, B, fixed_x, FT, CT,
        geom; weights, workspace)
    out_device, cnt_device = _owned_gpu_results(out_device, cnt_device, CT; workspace)
    sums = reshape(out_device, SFC.SINGLE_PASS_N, NB, bdims...)
    raw = reshape(cnt_device, SFC.SINGLE_PASS_N, NB, bdims...)
    return (sums = sums, counts = eltype(raw) === CT ? raw : CT.(raw))
end

function _gpu_dispatch_single_pass_batch!(
    sums::AbstractArray{OT},
    counts::AbstractArray{CT},
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), workspace = nothing,
) where {OT, CT, FT1, FT2, FT3}
    _check_gpu_outputs(sums, counts, backend, (SFC.SINGLE_PASS_N, length(distance_bins)-1, SFC.batch_dims(u)...);
                       workspace)
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    out_device, cnt_device = _gpu_1d_unified_device(
        backend, x, u, nothing, distance_bins, Val(SFC.SINGLE_PASS_N), NB, B, fixed_x, OT, CT,
        geom; weights, workspace)
    sums .+= reshape(out_device, size(sums)...)
    cflat = reshape(cnt_device, size(counts)...)
    if eltype(cflat) === CT
        counts .+= cflat
    else
        counts .+= CT.(cflat)
    end
    return sums, counts
end

"""Unified 2D batch device launch (joint `NMOM=1` or single-pass `NMOM=6`) through
`_sf_launch_2d_batch!`: fixed-x and varying-x, and any distance- and value-bin type. Returns device
`(sums, counts)` of shape `(NMOM, n_dist, n_val, B)`. `u` is staged `(D,N,B)` with no batch-major
permute."""
function _gpu_2d_unified_device(
    backend, x, u, sf_type, distance_bins, value_bins, ::Val{NMOM},
    n_dist::Int, n_val::Int, B::Int, fixed_x::Bool, ::Type{OT}, ::Type{CT}, geom;
    weights = SFC.NoWeights(), workspace = nothing,
) where {NMOM, OT, CT}
    _validate_batch_workspace!(workspace, backend, NMOM == 1 ? :joint2d : :single_pass_2d, distance_bins, OT; value_bins)
    D = size(x, 1)
    N = size(x, 2)
    ddig = _dist_digitizer(workspace, backend, distance_bins, Val(NMOM == 1 ? :joint2d : :single_pass_2d))
    vplan = _value_digitizer(workspace, backend, value_bins)
    wts = _sf_weights_to_device(backend, weights)
    CNT = _sf_count_type(wts, CT, _sf_worst_case_pairs(N))
    out_dev, cnt_dev = _batch_buffers(backend, OT, CNT, (NMOM, n_dist, n_val, B), workspace)
    B == 0 && return out_dev, cnt_dev
    if fixed_x
        x_dev = KA.adapt(backend, x)                       # (D, N)
        u_dev = KA.adapt(backend, reshape(u, D, N, B))     # (D, N, B), no permute
    else
        x_dev = KA.adapt(backend, reshape(x, D, N, B))
        u_dev = KA.adapt(backend, reshape(u, D, N, B))
    end
    _sf_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                         N, n_dist, n_val, B, D, Val(NMOM), fixed_x, geom; weights = wts)
    KA.synchronize(backend)
    return out_dev, cnt_dev
end

"""Single-pass (NMOM=6) 2D batch device launch — thin wrapper over the unified
`_gpu_2d_unified_device`."""
_gpu_sp2d_unified_device(backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, ::Type{OT},
                         ::Type{CT}, geom; weights = SFC.NoWeights(), workspace = nothing) where {OT, CT} =
    _gpu_2d_unified_device(backend, x, u, nothing, distance_bins, value_bins,
                           Val(SFC.SINGLE_PASS_N), n_dist, n_val, B, fixed_x, OT, CT, geom;
                           weights = weights, workspace = workspace)

function _gpu_dispatch_single_pass_2d_batch(
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), workspace = nothing,
) where {FT1, FT2, FT3, CT}
    FT = promote_type(float(FT1), float(FT2))
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = _n_value_edges(value_bins) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    out_device, cnt_device = _gpu_sp2d_unified_device(
        backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, FT, CT,
        geom; weights, workspace)
    out_device, cnt_device = _owned_gpu_results(out_device, cnt_device, CT; workspace)
    sums = reshape(out_device, SFC.SINGLE_PASS_N, n_dist, n_val, bdims...)
    raw = reshape(cnt_device, SFC.SINGLE_PASS_N, n_dist, n_val, bdims...)
    return (sums = sums, counts = eltype(raw) === CT ? raw : CT.(raw))
end

function _gpu_dispatch_single_pass_2d_batch!(
    sums::AbstractArray{OT},
    counts::AbstractArray{CT},
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), workspace = nothing,
) where {OT, CT, FT1, FT2, FT3}
    _check_gpu_outputs(sums, counts, backend,
        (SFC.SINGLE_PASS_N, length(distance_bins)-1, _n_value_edges(value_bins)-1, SFC.batch_dims(u)...); workspace)
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = size(sums, 3)
    B = SFC.batch_size(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    out_device, cnt_device = _gpu_sp2d_unified_device(
        backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, OT, CT,
        geom; weights, workspace)
    sums .+= reshape(out_device, size(sums)...)
    cflat = reshape(cnt_device, size(counts)...)
    if eltype(cflat) === CT
        counts .+= cflat
    else
        counts .+= CT.(cflat)
    end
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
    value_bins::AbstractVector{FT},
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), workspace = nothing,
    second_axis = SFC.InvariantValueAxis(),
) where {FT, CT}
    second_axis isa SFC.InvariantValueAxis || throw(ArgumentError(
        "the device joint slice batch bins each pair's own value; $(typeof(second_axis)) runs on " *
        "the CPU backends, or one slice at a time on the device.",
    ))
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = length(value_bins) - 1
    # The velocity dimension is `size(u, 1)`, and only before the conversion. `W` and `F` are the
    # widths the kernels index; they differ from each other and from `D` on a sphere.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    W = SFC._val_int(SFH.coordinate_width(geom))
    F = SFC._val_int(SFH.field_width(geom))
    N = size(x, 2)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)

    _validate_batch_workspace!(workspace, backend, :joint2d, distance_bins, FT; value_bins)
    ddig = _dist_digitizer(workspace, backend, distance_bins, Val(:joint2d))
    vplan = _value_digitizer(workspace, backend, value_bins)

    wts = _sf_weights_to_device(backend, weights)
    CNT = _sf_count_type(wts, CT, _sf_worst_case_pairs(N))
    out_dev, cnt_dev = _batch_buffers(backend, FT, CNT, (1, n_dist, n_val, B), workspace)

    if fixed_x
        x_dev = KA.adapt(backend, x)                       # (W, N)
        u_dev = KA.adapt(backend, reshape(u, F, N, B))     # (F, N, B) — no permute
    else
        x_dev = KA.adapt(backend, reshape(x, W, N, B))
        u_dev = KA.adapt(backend, reshape(u, F, N, B))
    end
    B == 0 || _sf_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                         N, n_dist, n_val, B, F, Val(1), fixed_x, geom; weights = wts)
    KA.synchronize(backend)

    out_dev, cnt_dev = _owned_gpu_results(out_dev, cnt_dev, CT; workspace)
    sums = reshape(out_dev, n_dist, n_val, bdims...)
    raw = reshape(cnt_dev, n_dist, n_val, bdims...)
    counts = eltype(raw) === CT ? raw : CT.(raw)
    return SF.StructureFunction2DSumsAndCounts(sf_type, distance_bins, value_bins, sums, counts)
end

function _gpu_calculate_structure_function_2d_batch!(sums, counts, sf, backend, x, u, distance_bins, value_bins;
        weights=SFC.NoWeights(), workspace=nothing, kwargs...)
    shape = (length(distance_bins)-1, length(value_bins)-1, SFC.batch_dims(u)...)
    _check_gpu_outputs(sums, counts, backend, shape; workspace)
    result = _gpu_calculate_structure_function_2d_batch(sf, backend, x, u, distance_bins, value_bins, eltype(counts);
        weights, workspace, kwargs...)
    sums .+= result.sums
    counts .+= result.counts
    KA.synchronize(backend)
    return nothing
end
