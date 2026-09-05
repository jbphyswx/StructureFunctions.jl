# Serial 2D CPU Joint Reduction Kernels

function serial_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    geometry = SFH.FlatGeometry{length(u_vecs)}(),
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, T1, T2}
    distance_bins = BinEdges(distance_bins)
    value_bins = BinEdges(value_bins)

    if verbose
        @info("calculating 2D joint structure function (serial reduction)")
    end

    # Fast path: Euclidean + D ∈ (2,3) via the SIMD compute/scatter split (distance + SF value
    # vectorize over j; the 2D (dist,value) scatter stays scalar).
    D = length(u_vecs)
    if geometry isa SFH.FlatGeometry && (D == 2 || D == 3)
        _pf_2d_simd_run!(sums_2d, counts_2d, structure_function_type, x_vecs, u_vecs,
                         distance_bins, value_bins, D == 2 ? Val(2) : Val(3); second_axis)
        return nothing
    end

    # The scalar fallback carries no second-axis source: it serves the curved geometries, where an
    # angle to a fixed reference axis is not a property of the pair alone.
    second_axis isa InvariantValueAxis || throw(ArgumentError(
        "$(typeof(second_axis)) is supported on the flat D ∈ {2,3} path; this call has " *
        "$(nameof(typeof(geometry))) with D = $D, whose separation direction lives in each pair's " *
        "own frame rather than a shared one.",
    ))
    PM.@showprogress enabled = show_progress for i in eachindex(x_vecs[1])
        calculate_structure_function_2d_i!(
            sums_2d, counts_2d, geometry, structure_function_type, i, x_vecs, u_vecs,
            distance_bins, value_bins,
        )
    end
    return nothing
end

"""
    _pf_2d_simd_pairs!(sums2d, counts2d, sf, xc, uc, plan, val_be, ::Val{D}, keybuf, valbuf,
                       idxbuf, blocks)

2D-joint point-field SIMD compute/scatter kernel over the pairs `blocks` covers: `@simd` over each
`j` block computes distance + SF value into buffers (no scatter ⇒ vectorizes), then a scalar loop
digitizes both axes and scatters into the (dist, second-axis) cell. What the second axis bins is
`second_axis`; binning the operator's own value reads the buffer the kernel already filled, so it
costs no extra store. Takes `(i-block, j-block)` pairs (see
[`block_pairs`](@ref)), so it gets cache blocking and culling; the loop lives in this one kernel so
the `@simd` vectorizes. Shared by serial + threaded.
"""
function _pf_2d_simd_pairs!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, val_be, ::Val{D},
    keybuf::AbstractVector, valbuf::AbstractVector, idxbuf::AbstractVector{Int32}, blocks,
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
    axbuf::AbstractVector = valbuf,
) where {OT, CT, D}
    n_dist = n_histogram_bins(plan)
    n_val = n_histogram_bins(val_be)
    FTx = eltype(xc[1])
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
            Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
            @simd for j in jlo:j_last
                Xj = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D)))
                dx = Xj - Xi
                r2 = LA.dot(dx, dx)
                Uj = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D)))
                keybuf[j] = digitize_key(plan, r2)
                valbuf[j] = SFT._sf_raw(sf, Uj - Ui, dx, r2)
                if has_vector_index(plan)
                    idxbuf[j] = squared_approx_index(plan, r2)
                end
                if needs_axis_buffer(second_axis)   # constant-folded on the operator-value axis
                    axbuf[j] = axis_quantity(second_axis, dx, r2)
                end
            end
            for j in jlo:j_last
                dbin = squared_bin(plan, keybuf[j], idxbuf[j])
                if 1 <= dbin <= n_dist
                    vbin = SFH.digitize(axis_key(second_axis, valbuf, axbuf, j), val_be)
                    if 1 <= vbin <= n_val
                        sums2d[dbin, vbin] += valbuf[j]
                        counts2d[dbin, vbin] += one(CT)
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _pf_2d_run_blocks!(sums2d, counts2d, sf, xc, uc, plan, val_be, ::Val{D}, bufs..., ilist, N, grid)

2D-joint analogue of [`_pf_run_blocks!`](@ref): dispatch on `grid` so the kernel receives one
concretely typed schedule.
"""
@inline _pf_2d_run_blocks!(
    sums2d, counts2d, sf, xc, uc, plan, val_be, ::Val{D}, keybuf, valbuf, idxbuf,
    ilist, N, ::Nothing, second_axis = InvariantValueAxis(), axbuf = valbuf,
) where {D} = _pf_2d_simd_pairs!(sums2d, counts2d, sf, xc, uc, plan, val_be, Val(D),
    keybuf, valbuf, idxbuf, pair_blocks(N, ilist), second_axis, axbuf)

@inline _pf_2d_run_blocks!(
    sums2d, counts2d, sf, xc, uc, plan, val_be, ::Val{D}, keybuf, valbuf, idxbuf,
    ilist, N, grid::CellGrid, second_axis = InvariantValueAxis(), axbuf = valbuf,
) where {D} = _pf_2d_simd_pairs!(sums2d, counts2d, sf, xc, uc, plan, val_be, Val(D),
    keybuf, valbuf, idxbuf, pair_blocks(N, ilist; grid = grid), second_axis, axbuf)

function _pf_2d_simd_run!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, val_be, ::Val{D};
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
) where {OT, CT, D}
    N = length(x_vecs[1])
    return _pf_2d_simd_partial!(sums2d, counts2d, sf, x_vecs, u_vecs, dist_be, val_be, Val(D),
                                1:(N - 1), AutoCulling(); second_axis)
end

"""
    _pf_2d_simd_partial!(sums2d, counts2d, sf, x_vecs, u_vecs, dist_be, val_be, ::Val{D}, ilist)

Run [`_pf_2d_simd_pairs!`](@ref) over an explicit outer-index list, materializing the contiguous
component vectors and scratch buffers this worker needs. Shared by the serial, distributed and MPI
drivers, whose inputs arrive as strided views.
"""
function _pf_2d_simd_partial!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, val_be, ::Val{D}, ilist,
    culling::CullingPolicy = AutoCulling();
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
) where {OT, CT, D}
    x_raw = ntuple(d -> collect(x_vecs[d]), Val(D))
    u_raw = ntuple(d -> collect(u_vecs[d]), Val(D))
    N = length(x_raw[1])
    keybuf = Vector{eltype(x_raw[1])}(undef, N)
    valbuf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    axbuf = needs_axis_buffer(second_axis) ? Vector{OT}(undef, N) : valbuf
    plan = squared_digitize_plan(dist_be)
    grid = culling isa NoCulling ? nothing :
           cull_grid_for(x_raw, SFH.FlatGeometry{D}(), dist_be, culling)
    xc, uc = isnothing(grid) ? (x_raw, u_raw) :
             (apply_perm(x_raw, grid.perm), apply_perm(u_raw, grid.perm))
    _pf_2d_run_blocks!(sums2d, counts2d, sf, xc, uc, plan, val_be, Val(D),
        keybuf, valbuf, idxbuf, ilist, N, grid, second_axis, axbuf)
    return nothing
end

"""
    _partial_2d_sums_counts(inner, sf_type, x_vecs, u_vecs, distance_bins, value_bins, ilist; kwargs...)

Partial 2D-joint sums/counts over an explicit outer-index list `ilist`, the 2D analogue of
[`_partial_sums_counts`](@ref). Euclidean `D ∈ {2,3}` takes the SIMD compute/scatter kernel; other
metrics fall back to the scalar per-`i` kernel. Returns `(sums, counts)`.
"""
function _partial_2d_sums_counts(
    ::CB.AbstractExecutionBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    ilist;
    geometry = SFH.FlatGeometry{length(u_vecs)}(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    _assert_counts_representable(CT, length(x_vecs[1]))
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    nd = n_histogram_bins(distance_bins)
    nv = n_histogram_bins(value_bins)
    sums = zeros(OT, nd, nv)
    counts = zeros(CT, nd, nv)
    dist_be = BinEdges(distance_bins)
    val_be = BinEdges(value_bins)
    D = length(u_vecs)

    if geometry isa SFH.FlatGeometry && (D == 2 || D == 3)
        vD = D == 2 ? Val(2) : Val(3)
        _pf_2d_simd_partial!(sums, counts, structure_function_type, x_vecs, u_vecs, dist_be, val_be, vD, ilist)
        return sums, counts
    end

    for i in ilist
        calculate_structure_function_2d_i!(
            sums, counts, geometry, structure_function_type, i, x_vecs, u_vecs, dist_be, val_be,
        )
    end
    return sums, counts
end

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {T1, T2, CT}
    _assert_counts_representable(CT, length(x_vecs[1]))
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    N4 = n_histogram_bins(value_bins)

    sums_2d = zeros(OT, N3, N4)
    counts_2d = zeros(CT, N3, N4)

    serial_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        value_bins;
        kwargs...,
    )

    return SFO.StructureFunction2DSumsAndCounts(
        structure_function_type,
        distance_bins,
        value_bins,
        sums_2d,
        counts_2d,
    )
end

function serial_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number}
    # `size(u_arr, 1)` is the velocity dimension here, before conversion, so this is where the
    # geometry is fixed; everything downstream receives it.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u_arr, 1)))
    xk, uk = SFH.prepare_pair_inputs(geom, x_arr, u_arr)
    return serial_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        _component_vector_views(xk, SFH.coordinate_width(geom)),
        _component_vector_views(uk, SFH.field_width(geom)),
        distance_bins,
        value_bins;
        geometry = geom,
        kwargs...,
    )
end

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    if ndims(u_arr) >= 3
        FT = promote_type(float(FT1), float(FT2))
        n_dist = length(distance_bins) - 1
        n_val = length(value_bins) - 1
        bdims = batch_dims(u_arr)
        sums = zeros(FT, n_dist, n_val, bdims...)
        # Consumed here, where `counts` is allocated. `auxiliary_joint2d!` takes already-allocated
        # buffers, so forwarding `count_eltype` into it is a MethodError.
        counts = zeros(CT, n_dist, n_val, bdims...)
        auxiliary_joint2d!(sums, counts, structure_function_type, x_arr, u_arr, distance_bins, value_bins; kwargs...)
        return SFO.StructureFunction2DSumsAndCounts(structure_function_type, distance_bins, value_bins, sums, counts)
    end
    geom = SFH.pair_geometry_for(get(kwargs, :distance_metric, DI.Euclidean()), Val(size(u_arr, 1)))
    xk, uk = SFH.prepare_pair_inputs(geom, x_arr, u_arr)
    rest = Base.structdiff(NamedTuple(kwargs), NamedTuple{(:distance_metric,)})
    return serial_calculate_structure_function(
        structure_function_type,
        _component_vector_views(xk, SFH.coordinate_width(geom)),
        _component_vector_views(uk, SFH.field_width(geom)),
        distance_bins,
        value_bins;
        count_eltype = count_eltype,
        geometry = geom,
        rest...,
    )
end

function calculate_structure_function_2d_i!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix,
    geom,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
) where {OT, T1, T2}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins)
    N4 = length(value_bins)

    # The geometry carries the coordinate width, the field width and the velocity dimension; none of
    # them need equal another.
    vW = SFH.coordinate_width(geom)
    vF = SFH.field_width(geom)
    W = _val_int(vW)
    F = _val_int(vF)
    X1 = SA.SVector{W, FT1}(ntuple(k -> x_vecs[k][i], vW))
    U1 = SA.SVector{F, FT2}(ntuple(k -> u_vecs[k][i], vF))

    iter_inds = eachindex(x_vecs[1])
    for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{W, FT1}(ntuple(k -> x_vecs[k][j], vW))
        U2 = SA.SVector{F, FT2}(ntuple(k -> u_vecs[k][j], vF))

        ok, distance, frame = SFH.pair_frame(geom, X1, X2)
        dist_bin = SFH.digitize(distance, distance_bins)
        if ok && 1 <= dist_bin < N3
            δu, rh = SFH.pair_increments(geom, frame, distance, X1, X2, U1, U2)
            val = structure_function_type(δu, rh)
            val_bin = SFH.digitize(val, value_bins)
            
            if 1 <= val_bin < N4
                @inbounds sums_2d[dist_bin, val_bin] += val
                @inbounds counts_2d[dist_bin, val_bin] += 1
            end
        end
    end
    return nothing
end

function calculate_structure_function_2d_i(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    geometry = SFH.FlatGeometry{length(u_vecs)}(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    FT1 = eltype(x_vecs[1])
    FT2 = eltype(u_vecs[1])
    N3 = n_histogram_bins(distance_bins)
    N4 = n_histogram_bins(value_bins)
    OT = promote_type(float(FT1), float(FT2))
    local_sums = zeros(OT, N3, N4)
    local_counts = zeros(CT, N3, N4)

    calculate_structure_function_2d_i!(
        local_sums,
        local_counts,
        geometry,
        structure_function_type,
        i,
        x_vecs,
        u_vecs,
        BinEdges(distance_bins),
        BinEdges(value_bins),
    )

    return SFO.StructureFunction2DSumsAndCounts(
        structure_function_type,
        distance_bins,
        value_bins,
        local_sums,
        local_counts,
    )
end
