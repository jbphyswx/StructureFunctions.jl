# GPU batch routing — fixed-x / varying-x fused launches.
# Included from StructureFunctionsKernelAbstractionsExt.jl after BatchLaunch.jl.

"""Typed fast-path selector for the fixed-x tiled FMA kernels.

Purely type-driven: `LinearBinEdges` and `LogBinEdges` both qualify — they share
the same 5-parameter O(1) FMA digitize, log just runs it in log space (see
`_batch_fma_dist_params` / `_batch_dist_bin`'s `Val{LOG}` specialization). An
`AbstractRange` is uniform by construction and wraps to `LinearBinEdges`. Raw
edge vectors return `nothing` and take the unified digitizer path
(`_sf_batch_dist_digitizer`), which bins them exactly by binary search. No
runtime uniformity sniffing, no try/catch control flow — bin membership must
never depend on an `isapprox` tolerance.
"""
_fma_distance_bins(distance_bins::LinearBinEdges) = distance_bins
_fma_distance_bins(distance_bins::LogBinEdges) = distance_bins
_fma_distance_bins(distance_bins::AbstractRange) = LinearBinEdges(distance_bins)
_fma_distance_bins(::Any) = nothing

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
    ::Val{NMOM}, NB::Int, B::Int, fixed_x::Bool, ::Type{OT}, geom;
    weights = SFC.NoWeights(), count_eltype::Type{CT} = UInt32,
) where {NMOM, OT, CT}
    D = size(x, 1)
    N = size(x, 2)
    dig = _sf_batch_dist_digitizer(backend, distance_bins)
    wts = _sf_weights_to_device(backend, weights)
    CNT = _sf_count_type(wts, CT, _sf_worst_case_pairs(N))
    out_dev = KA.adapt(backend, zeros(OT, NMOM, NB, B))
    cnt_dev = KA.adapt(backend, zeros(CNT, NMOM, NB, B))
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
    return Array(out_dev), Array(cnt_dev)
end

"""Individual (NMOM=1) 1D batch device launch, routed per regime:
- fixed-x with linear or log FMA bins → global warp-replica kernel over W-strips, which wins where
  the histogram is small enough to be contention-bound.
- varying-x, or general (raw-vector) bins → N-body broadcast.
Returns host `(sums, counts)` of shape `(NB, B)`."""
function _gpu_1d_individual_device(backend, sf_type, x, u, distance_bins,
                                   NB::Int, B::Int, fixed_x::Bool, ::Type{OT}, geom;
                                   weights = SFC.NoWeights(),
                                   count_eltype::Type{CT} = UInt32) where {OT, CT}
    # The warp-replica fixed-x kernel has no weighted form, so a weighted call takes the unified
    # kernel, which does. Both are exact; this is a route choice inside one backend.
    lbe = (fixed_x && weights isa SFC.NoWeights) ? _fma_distance_bins(distance_bins) : nothing
    if lbe !== nothing
        N = size(x, 2)
        sums_dev = KA.adapt(backend, zeros(OT, NB, B))
        counts_dev = KA.adapt(backend, zeros(_sf_count_type(weights, CT, _sf_worst_case_pairs(N)), NB, B))
        x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = true)
        _launch_batch_fixed_x_sf!(backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B, lbe, geom)
        return Array(sums_dev), Array(counts_dev)
    end
    oh, ch = _gpu_1d_unified_device(backend, x, u, sf_type, distance_bins, Val(1), NB, B, fixed_x, OT, geom;
                                    weights = weights, count_eltype = CT)
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
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT, CT}
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    w = SFC._pair_weights(weights, size(x, 2), FT)
    SFC._check_weighted_counts(w, CT)
    out_host, cnt_host = _gpu_1d_individual_device(backend, sf_type, x, u, distance_bins, NB, B, fixed_x, FT,
        geom; weights = w, count_eltype = CT)
    sums = reshape(out_host, NB, bdims...)
    counts = reshape(cnt_host, NB, bdims...)
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
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT}
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    CT = eltype(output_counts)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    w = SFC._pair_weights(weights, size(x, 2), eltype(output_sums))
    SFC._check_weighted_counts(w, CT)
    out_host, cnt_host = _gpu_1d_individual_device(
        backend, sf_type, x, u, distance_bins, NB, B, fixed_x, eltype(output_sums),
        geom; weights = w, count_eltype = CT)
    output_sums .+= reshape(out_host, size(output_sums)...)
    cflat = reshape(cnt_host, size(output_counts)...)
    if eltype(cflat) === CT
        output_counts .+= cflat
    else
        output_counts .+= CT.(cflat)
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
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT1, FT2, FT3, CT}
    FT = promote_type(float(FT1), float(FT2))
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    w = SFC._pair_weights(weights, size(x, 2), FT)
    SFC._check_weighted_counts(w, CT)
    out_host, cnt_host = _gpu_1d_unified_device(
        backend, x, u, nothing, distance_bins, Val(SFC.SINGLE_PASS_N), NB, B, fixed_x, FT,
        geom; weights = w, count_eltype = CT)
    sums = reshape(out_host, SFC.SINGLE_PASS_N, NB, bdims...)
    raw = reshape(cnt_host, SFC.SINGLE_PASS_N, NB, bdims...)
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
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, FT1, FT2, FT3}
    fixed_x = ndims(x) == 2
    NB = length(distance_bins) - 1
    B = SFC.batch_size(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    w = SFC._pair_weights(weights, size(x, 2), OT)
    SFC._check_weighted_counts(w, CT)
    out_host, cnt_host = _gpu_1d_unified_device(
        backend, x, u, nothing, distance_bins, Val(SFC.SINGLE_PASS_N), NB, B, fixed_x, OT,
        geom; weights = w, count_eltype = CT)
    sums .+= reshape(out_host, size(sums)...)
    cflat = reshape(cnt_host, size(counts)...)
    if eltype(cflat) === CT
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
    n_dist::Int, n_val::Int, B::Int, fixed_x::Bool, ::Type{OT}, geom;
    weights = SFC.NoWeights(), count_eltype::Type{CT} = UInt32,
) where {NMOM, OT, CT}
    D = size(x, 1)
    N = size(x, 2)
    ddig = _sf_batch_dist_digitizer(backend, distance_bins)
    vplan = _gpu_build_value_digitize_plan(backend, value_bins)
    wts = _sf_weights_to_device(backend, weights)
    CNT = _sf_count_type(wts, CT, _sf_worst_case_pairs(N))
    out_dev = KA.adapt(backend, zeros(OT, NMOM, n_dist, n_val, B))
    cnt_dev = KA.adapt(backend, zeros(CNT, NMOM, n_dist, n_val, B))
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
    return Array(out_dev), Array(cnt_dev)
end

"""Single-pass (NMOM=6) 2D batch device launch — thin wrapper over the unified
`_gpu_2d_unified_device`."""
_gpu_sp2d_unified_device(backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, ::Type{OT}, geom;
                         weights = SFC.NoWeights(), count_eltype::Type = UInt32) where {OT} =
    _gpu_2d_unified_device(backend, x, u, nothing, distance_bins, value_bins,
                           Val(SFC.SINGLE_PASS_N), n_dist, n_val, B, fixed_x, OT, geom;
                           weights = weights, count_eltype = count_eltype)

function _gpu_dispatch_single_pass_2d_batch(
    backend::KA.Backend,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    count_eltype::Type{CT} = UInt32,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT1, FT2, FT3, CT}
    FT = promote_type(float(FT1), float(FT2))
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = _sp2d_n_val_edges(value_bins) - 1
    SFC._validate_value_bins!(value_bins, n_val)
    B = SFC.batch_size(u)
    bdims = SFC.batch_dims(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    w = SFC._pair_weights(weights, size(x, 2), FT)
    SFC._check_weighted_counts(w, CT)
    out_host, cnt_host = _gpu_sp2d_unified_device(
        backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, FT,
        geom; weights = w, count_eltype = CT)
    sums = reshape(out_host, SFC.SINGLE_PASS_N, n_dist, n_val, bdims...)
    raw = reshape(cnt_host, SFC.SINGLE_PASS_N, n_dist, n_val, bdims...)
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
    weights = SFC.NoWeights(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, FT1, FT2, FT3}
    fixed_x = ndims(x) == 2
    n_dist = length(distance_bins) - 1
    n_val = size(sums, 3)
    B = SFC.batch_size(u)
    # The velocity dimension is `size(u, 1)`, and only before the conversion.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    x, u = SFH.prepare_pair_inputs(geom, x, u)
    w = SFC._pair_weights(weights, size(x, 2), OT)
    SFC._check_weighted_counts(w, CT)
    out_host, cnt_host = _gpu_sp2d_unified_device(
        backend, x, u, distance_bins, value_bins, n_dist, n_val, B, fixed_x, OT,
        geom; weights = w, count_eltype = CT)
    sums .+= reshape(out_host, size(sums)...)
    cflat = reshape(cnt_host, size(counts)...)
    if eltype(cflat) === CT
        counts .+= cflat
    else
        counts .+= CT.(cflat)
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
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
    verbose::Bool = true,
    show_progress::Bool = true,
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

    ddig = _sf_batch_dist_digitizer(backend, distance_bins)
    vplan = _gpu_build_value_digitize_plan(backend, value_bins)

    w = SFC._pair_weights(weights, N, FT)
    SFC._check_weighted_counts(w, CT)
    wts = _sf_weights_to_device(backend, w)
    CNT = _sf_count_type(wts, CT, _sf_worst_case_pairs(N))
    out_dev = KA.adapt(backend, zeros(FT, 1, n_dist, n_val, B))
    cnt_dev = KA.adapt(backend, zeros(CNT, 1, n_dist, n_val, B))

    if fixed_x
        x_dev = KA.adapt(backend, x)                       # (W, N)
        u_dev = KA.adapt(backend, reshape(u, F, N, B))     # (F, N, B) — no permute
    else
        x_dev = KA.adapt(backend, reshape(x, W, N, B))
        u_dev = KA.adapt(backend, reshape(u, F, N, B))
    end
    _sf_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                         N, n_dist, n_val, B, F, Val(1), fixed_x, geom; weights = wts)
    KA.synchronize(backend)

    sums = reshape(Array(out_dev)[1, :, :, :], n_dist, n_val, bdims...)
    raw = reshape(Array(cnt_dev)[1, :, :, :], n_dist, n_val, bdims...)
    counts = eltype(raw) === CT ? raw : CT.(raw)
    return SF.StructureFunction2DSumsAndCounts(sf_type, distance_bins, value_bins, sums, counts)
end
