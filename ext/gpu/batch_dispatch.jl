# GPU batch routing — fixed-x / varying-x fused launches.

"""
    _gpu_batch_launch!(launch!, backend, out, cnt, x, u, geom, distance_bins, culling, weights, workspace,
                       source, fixed_x)

Run `launch!(out, cnt, x, u, weights, fixed_x, cull)` over a batch of kernel inputs `x` — `(W, N)`
shared or `(W, N, B)` varying — and `u` `(F, N, B)`, culled by `culling`. Shared positions are sorted
once, every slice's field and the weights following, and the launchers schedule from that one memo.
Positions varying per slice are sorted per slice and each slice runs through a one-slice buffer with
its own memo; when no slice culls the batch is one launch. `source` identifies the caller's coordinates
to a workspace's memo.
"""
function _gpu_batch_launch!(launch!, backend, out, cnt, x, u, geom, distance_bins, culling, weights, workspace,
                            source, fixed_x::Bool)
    if fixed_x
        xs, us, perm, cull = _gpu_cull_and_permute!(workspace, backend, x, u, geom, distance_bins, culling, source)
        return launch!(out, cnt, xs, us, _permuted_weights(backend, weights, perm), true, cull)
    end
    _slices_may_cull(culling, workspace, geom, distance_bins, size(x, 2)) ||
        return launch!(out, cnt, x, u, weights, false, nothing)
    slices = map(1:size(u, 3)) do b
        _gpu_cull_and_permute!(workspace, backend, view(x, :, :, b), view(u, :, :, b), geom, distance_bins, culling,
                               (source, b))
    end
    all(s -> s[4] === nothing, slices) && return launch!(out, cnt, x, u, weights, false, nothing)
    out_b = similar(out, size(out)[1:(end - 1)]..., 1)
    cnt_b = similar(cnt, size(cnt)[1:(end - 1)]..., 1)
    for (b, (xs, us, perm, cull)) in enumerate(slices)
        fill!(out_b, zero(eltype(out_b)))
        fill!(cnt_b, zero(eltype(cnt_b)))
        launch!(out_b, cnt_b, reshape(xs, size(xs)..., 1), reshape(us, size(us)..., 1),
                _permuted_weights(backend, weights, perm), false, cull)
        selectdim(out, ndims(out), b) .+= selectdim(out_b, ndims(out_b), 1)
        selectdim(cnt, ndims(cnt), b) .+= selectdim(cnt_b, ndims(cnt_b), 1)
    end
    return nothing
end

"""Unified 1D batch device launch of the moment set `moments` through `_sf_launch_1d_batch!`: fixed-x
and varying-x, any width, any distance-bin type, culled by `culling`. Accumulates into the caller's
`sums`/`counts` when [`_accumulation_buffers`](@ref) admits them, else into fresh buffers; returns
`(sums, counts, direct)` of shape `(NMOM, NB, B)`. `u` is staged `(F,N,B)` with no batch-major permute."""
function _gpu_1d_unified_device(
    backend, x, u, moments, distance_bins, NB::Int, B::Int, fixed_x::Bool, ::Type{OT}, ::Type{CT}, geom;
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(), source = x,
    sums = nothing, counts = nothing,
) where {OT, CT}
    kind = _sf_workspace_kind(moments, Val(1))
    _validate_batch_workspace!(workspace, backend, kind, distance_bins)
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFH.field_width(geom))
    N = size(x, 2)
    dig = _dist_digitizer(workspace, backend, distance_bins, Val(kind))
    CNT = _sf_count_type(_sf_weights_to_device(backend, weights), CT, _sf_worst_case_pairs(N))
    out_dev, cnt_dev, direct = _accumulation_buffers(backend, OT, CNT, (_sf_nmom(moments), NB, B), sums, counts)
    B == 0 && return out_dev, cnt_dev, direct
    launch! = (o, c, xs, us, w, fx, cull) -> _sf_launch_1d_batch!(
        backend, o, c, KA.adapt(backend, xs), KA.adapt(backend, us), moments, dig, N, NB, size(o, 3),
        fx, geom; weights = w, cull)
    _gpu_batch_launch!(launch!, backend, out_dev, cnt_dev, fixed_x ? x : reshape(x, W, N, B),
                       reshape(u, F, N, B), geom, distance_bins, culling, weights, workspace, source, fixed_x)
    return out_dev, cnt_dev, direct
end

"""Individual batch distance histograms: the unified tiles, whose launch takes the native kernel when the backend has
a plan for the call, or else the fixed-position field-strip kernel where it applies; returns `(sums, counts, direct)`
of shape `(NB, B)` as `_gpu_1d_unified_device` does."""
function _gpu_1d_individual_device(backend, sf_type, x, u, distance_bins,
                                   NB::Int, B::Int, fixed_x::Bool, ::Type{OT}, ::Type{CT}, geom;
                                   weights = SFC.NoWeights(), workspace = nothing,
                                   culling::SFC.CullingPolicy = SFC.AutoCulling(), source = x,
                                   sums = nothing, counts = nothing) where {OT, CT}
    two_wide = SFC._val_int(SFH.coordinate_width(geom)) == 2 &&
               SFC._val_int(SFH.field_width(geom)) == 2
    native = SFC.gpu_native_1d_plan(backend, eltype(x), eltype(u), OT,
                                    _sf_count_type(weights, CT, _sf_worst_case_pairs(size(x, 2))), weights, geom, NB,
                                    sf_type)
    if native === nothing && fixed_x && two_wide && weights isa SFC.NoWeights && NB <= SF_GPU_MAX_BINS &&
       _sf_worst_case_pairs(size(x, 2)) <= typemax(UInt32) &&
       _batch_usmem_strip_w(SFC.gpu_device_caps(backend), OT) > 0
        _validate_batch_workspace!(workspace, backend, :sf1d, distance_bins)
        N = size(x, 2)
        sums_dev, counts_dev, direct = _accumulation_buffers(backend, OT,
            _sf_count_type(weights, CT, _sf_worst_case_pairs(N)), (NB, B), sums, counts)
        B == 0 && return sums_dev, counts_dev, direct
        xs, us, _, cull = _gpu_cull_and_permute!(workspace, backend, x, reshape(u, 2, N, B), geom, distance_bins,
                                                 culling, source)
        x_dev, u_dev = _stage_batch_device(backend, xs, us; fixed_x=true)
        _launch_batch_fixed_x_sf!(backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B,
                                  _dist_digitizer(workspace, backend, distance_bins, Val(:sf1d)), NB, geom;
                                  cull)
        return sums_dev, counts_dev, direct
    end
    sums_dev, counts_dev, direct = _gpu_1d_unified_device(backend, x, u, sf_type, distance_bins,
        NB, B, fixed_x, OT, CT, geom; weights, workspace, culling, source, sums, counts)
    return reshape(sums_dev, NB, B), reshape(counts_dev, NB, B), direct
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
    geometry,
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {FT, CT}
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    source = x
    x, u = SFH.prepare_pair_inputs(geometry, x, u)
    out_dev, cnt_dev, _ = _gpu_1d_individual_device(backend, sf_type, x, u, distance_bins, NB, B, fixed_x, FT, CT,
        geometry; weights, workspace, culling, source)
    return SF.StructureFunctionSumsAndCounts(
        sf_type, distance_bins, reshape(out_dev, NB, bdims...), reshape(_result_counts(cnt_dev, CT), NB, bdims...),
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
    geometry,
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {FT}
    _check_gpu_outputs(output_sums, output_counts, backend, (length(distance_bins)-1, SFC.batch_dims(u)...))
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    source = x
    x, u = SFH.prepare_pair_inputs(geometry, x, u)
    out_dev, cnt_dev, direct = _gpu_1d_individual_device(
        backend, sf_type, x, u, distance_bins, NB, B, fixed_x, eltype(output_sums), eltype(output_counts),
        geometry; weights, workspace, culling, source, sums = output_sums, counts = output_counts)
    _add_accumulated!(output_sums, output_counts, out_dev, cnt_dev, direct)
    return nothing
end

function _gpu_dispatch_single_pass_batch(
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    geometry,
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {FT1, FT2, FT3, CT}
    FT = promote_type(float(FT1), float(FT2))
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    source = x
    x, u = SFH.prepare_pair_inputs(geometry, x, u)
    out_dev, cnt_dev, _ = _gpu_1d_unified_device(
        backend, x, u, SFT.SinglePassInvariants(), distance_bins, NB, B, fixed_x, FT, CT,
        geometry; weights, workspace, culling, source)
    return (sums = reshape(out_dev, SFC.SINGLE_PASS_N, NB, bdims...),
            counts = reshape(_result_counts(cnt_dev, CT), SFC.SINGLE_PASS_N, NB, bdims...))
end

function _gpu_dispatch_single_pass_batch!(
    sums::AbstractArray{OT},
    counts::AbstractArray{CT},
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3};
    geometry,
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {OT, CT, FT1, FT2, FT3}
    _check_gpu_outputs(sums, counts, backend, (SFC.SINGLE_PASS_N, length(distance_bins)-1, SFC.batch_dims(u)...))
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    source = x
    x, u = SFH.prepare_pair_inputs(geometry, x, u)
    out_dev, cnt_dev, direct = _gpu_1d_unified_device(
        backend, x, u, SFT.SinglePassInvariants(), distance_bins, NB, B, fixed_x, OT, CT,
        geometry; weights, workspace, culling, source, sums, counts)
    _add_accumulated!(sums, counts, out_dev, cnt_dev, direct)
    return sums, counts
end

"""Unified 2D batch device launch of the moment set `moments` through `_sf_launch_2d_batch!`: fixed-x
and varying-x, any distance- and value-bin type, culled by `culling`. Accumulates into the caller's
`sums`/`counts` when [`_accumulation_buffers`](@ref) admits them, else into fresh buffers; returns
`(sums, counts, direct)` of shape `(NMOM, n_dist, n_val, B)`. `u` is staged `(F,N,B)` with no batch-major
permute."""
function _gpu_2d_unified_device(
    backend, x, u, moments, distance_bins, value_bins,
    n_dist::Int, n_val::Int, B::Int, fixed_x::Bool, ::Type{OT}, ::Type{CT}, geom;
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(), source = x,
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(), sums = nothing, counts = nothing,
) where {OT, CT}
    kind = _sf_workspace_kind(moments, Val(2))
    _validate_batch_workspace!(workspace, backend, kind, distance_bins; value_bins)
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFH.field_width(geom))
    N = size(x, 2)
    ddig = _dist_digitizer(workspace, backend, distance_bins, Val(kind))
    vplan = _value_digitizer(workspace, backend, value_bins)
    CNT = _sf_count_type(_sf_weights_to_device(backend, weights), CT, _sf_worst_case_pairs(N))
    out_dev, cnt_dev, direct = _accumulation_buffers(backend, OT, CNT, (_sf_nmom(moments), n_dist, n_val, B),
                                                     sums, counts)
    B == 0 && return out_dev, cnt_dev, direct
    launch! = (o, c, xs, us, w, fx, cull) -> _sf_launch_2d_batch!(
        backend, o, c, KA.adapt(backend, xs), KA.adapt(backend, us), moments, ddig, vplan, N, n_dist, n_val,
        size(o, 4), fx, geom, second_axis; weights = w, cull)
    _gpu_batch_launch!(launch!, backend, out_dev, cnt_dev, fixed_x ? x : reshape(x, W, N, B),
                       reshape(u, F, N, B), geom, distance_bins, culling, weights, workspace, source, fixed_x)
    return out_dev, cnt_dev, direct
end

"""Single-pass 2D batch device launch — `_gpu_2d_unified_device` of the single-pass invariants."""
_gpu_sp2d_unified_device(backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, ::Type{OT},
                         ::Type{CT}, geom; weights = SFC.NoWeights(), workspace = nothing,
                         culling::SFC.CullingPolicy = SFC.AutoCulling(), source = x,
                         sums = nothing, counts = nothing) where {OT, CT} =
    _gpu_2d_unified_device(backend, x, u, SFT.SinglePassInvariants(), distance_bins, value_bins,
                           n_dist, n_val, B, fixed_x, OT, CT, geom;
                           weights, workspace, culling, source, sums, counts)

function _gpu_dispatch_single_pass_2d_batch(
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins,
    ::Type{CT};
    geometry,
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {FT1, FT2, FT3, CT}
    FT = promote_type(float(FT1), float(FT2))
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = _n_value_edges(value_bins) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    source = x
    x, u = SFH.prepare_pair_inputs(geometry, x, u)
    out_dev, cnt_dev, _ = _gpu_sp2d_unified_device(
        backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, FT, CT,
        geometry; weights, workspace, culling, source)
    return (sums = reshape(out_dev, SFC.SINGLE_PASS_N, n_dist, n_val, bdims...),
            counts = reshape(_result_counts(cnt_dev, CT), SFC.SINGLE_PASS_N, n_dist, n_val, bdims...))
end

function _gpu_dispatch_single_pass_2d_batch!(
    sums::AbstractArray{OT},
    counts::AbstractArray{CT},
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    geometry,
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {OT, CT, FT1, FT2, FT3}
    _check_gpu_outputs(sums, counts, backend,
        (SFC.SINGLE_PASS_N, length(distance_bins)-1, _n_value_edges(value_bins)-1, SFC.batch_dims(u)...))
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = size(sums, 3)
    B = SFC.batch_size(u)
    source = x
    x, u = SFH.prepare_pair_inputs(geometry, x, u)
    out_dev, cnt_dev, direct = _gpu_sp2d_unified_device(
        backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, OT, CT,
        geometry; weights, workspace, culling, source, sums, counts)
    _add_accumulated!(sums, counts, out_dev, cnt_dev, direct)
    return sums, counts
end

"""Fused GPU batch driver for single-type joint 2D structure functions: one tiled launch over every
slice, each pair binned by its value or, with a [`SFC.SeparationAngleAxis`](@ref) `second_axis`, by the
angle of its separation."""
function _gpu_calculate_structure_function_2d_batch(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT},
    value_bins::AbstractVector{FT},
    ::Type{CT};
    geometry,
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
) where {FT, CT}
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = length(value_bins) - 1
    SFC._require_value_axis(second_axis, geometry)
    source = x
    x, u = SFH.prepare_pair_inputs(geometry, x, u)
    bdims = SFC.batch_dims(u)
    out_dev, cnt_dev, _ = _gpu_2d_unified_device(backend, x, u, sf_type, distance_bins, value_bins,
                                                 n_dist, n_val, SFC.batch_size(u), fixed_x, FT, CT, geometry;
                                                 weights, workspace, culling, source, second_axis)
    return SF.StructureFunction2DSumsAndCounts(sf_type, distance_bins, value_bins,
        reshape(out_dev, n_dist, n_val, bdims...), reshape(_result_counts(cnt_dev, CT), n_dist, n_val, bdims...))
end

function _gpu_calculate_structure_function_2d_batch!(
    sums,
    counts,
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    backend::KA.Backend,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins::AbstractVector{FT},
    value_bins::AbstractVector{FT};
    geometry,
    weights = SFC.NoWeights(), workspace = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
) where {FT}
    n_dist = length(distance_bins) - 1
    n_val = length(value_bins) - 1
    _check_gpu_outputs(sums, counts, backend, (n_dist, n_val, SFC.batch_dims(u)...))
    fixed_x = ndims(x) == 2
    SFC._require_value_axis(second_axis, geometry)
    source = x
    x, u = SFH.prepare_pair_inputs(geometry, x, u)
    out_dev, cnt_dev, direct = _gpu_2d_unified_device(backend, x, u, sf_type, distance_bins, value_bins,
        n_dist, n_val, SFC.batch_size(u), fixed_x, eltype(sums), eltype(counts), geometry;
        weights, workspace, culling, source, second_axis, sums, counts)
    _add_accumulated!(sums, counts, out_dev, cnt_dev, direct)
    return nothing
end
