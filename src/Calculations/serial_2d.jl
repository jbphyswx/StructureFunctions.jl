# Serial 2D CPU Joint Reduction Kernels

"""
    _require_value_axis(second_axis, geometry)

Refuse a second axis a curved geometry cannot read. An angle to a fixed reference axis needs a
separation direction shared by every pair; on a curved metric that direction lives in each pair's
own geodesic frame, so only the pair's own value is available.
"""
@inline function _require_value_axis(second_axis, geometry)
    geometry isa SFH.FlatGeometry || second_axis isa InvariantValueAxis || throw(ArgumentError(
        "$(typeof(second_axis)) needs a separation direction shared by every pair; this call has " *
        "$(nameof(typeof(geometry))), whose separation direction lives in each pair's own geodesic " *
        "frame. Bin the pair's own value with InvariantValueAxis(), or use a flat metric.",
    ))
    return nothing
end

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
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {OT, CT, T1, T2}
    val_be = digitize_plan(value_bins)

    # Fast path: Euclidean + D ∈ (2,3) via the SIMD compute/scatter split (distance + SF value
    # vectorize over j; the 2D (dist,value) scatter stays scalar).
    D = length(u_vecs)
    if geometry isa SFH.FlatGeometry && (D == 2 || D == 3)
        _pf_2d_simd_run!(sums_2d, counts_2d, structure_function_type, x_vecs, u_vecs,
                         distance_bins, val_be, D == 2 ? Val(2) : Val(3);
                         second_axis, culling, weights)
        return nothing
    end

    _require_value_axis(second_axis, geometry)
    _pf_2d_scalar_run!(sums_2d, counts_2d, geometry, structure_function_type, distance_bins, val_be, second_axis,
                       nothing, _cull_sorted(x_vecs, u_vecs, weights, geometry, distance_bins, culling))
    return nothing
end

"""
    _pf_2d_scalar_run!(sums, counts, geometry, sf, distance_bins, val_be, second_axis, share,
                       (grid, x_vecs, u_vecs, weights))

Run [`_pf_2d_scalar_pairs!`](@ref) over the outer indices `share` selects ([`_share_indices`](@ref)) of a
[`_cull_sorted`](@ref) result.
"""
function _pf_2d_scalar_run!(sums, counts, geometry, sf, distance_bins, val_be, second_axis, share,
                            (grid, xc, uc, wc))
    N = length(xc[1])
    _pf_2d_scalar_pairs!(sums, counts, geometry, sf, xc, uc, digitize_plan(distance_bins), val_be,
                         pair_blocks(N, _share_indices(grid, N - 1, share); grid), wc, second_axis)
    return nothing
end

"""
    _pf_2d_simd_pairs!(sums2d, counts2d, sf, xc, uc, plan, val_be, ::Val{D}, keybuf, valbuf, idxbuf, colbuf,
                       sel, window, blocks, second_axis, axbuf, weights)

2D-joint point-field SIMD compute/scatter kernel over the pairs `blocks` covers, into `sums2d`/`counts2d` of
shape `(n_dist, n_val + 2)`: value column `c + 1` holds value bin `c`, and the first and last columns the
values below and above the value edges. `@simd` over each `j` block computes the distance key and the SF
value into buffers indexed by `window`, and the value column into `colbuf` where [`has_vector_digitize`](@ref)
holds for the value edges and the schedule is culled or the run before had more than 1/8 of its sampled pairs
in range ([`_sparse`](@ref)); a scalar loop then scatters the in-range pairs as [`_pf_simd_pairs!`](@ref) does,
forming the value column itself when the vectorized half did not. What the second axis bins is `second_axis`; binning the operator's own value reads
the buffer the kernel already filled. Takes `(i-block, j-block)` pairs (see [`block_pairs`](@ref)), so it gets
cache blocking and culling; the loop lives in this one kernel so the `@simd` vectorizes. Shared by serial +
threaded.
"""
function _pf_2d_simd_pairs!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, val_be, ::Val{D},
    keybuf::AbstractVector, valbuf::AbstractVector, idxbuf::AbstractVector{Int32},
    colbuf::AbstractVector{Int32}, sel::AbstractVector{Int32}, window::PairWindow, blocks,
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
    axbuf::AbstractVector = valbuf,
    weights = NoWeights(),
) where {OT, CT, D}
    n_dist = n_histogram_bins(plan)
    n_val = n_histogram_bins(val_be)
    vector_columns = has_vector_digitize(val_be, OT)
    chooses = _chooses_compaction(blocks)
    FTx = eltype(xc[1])
    s_in, s_n = 0, 0
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        _check_run_fits(window, valbuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            wi = _point_weight(weights, i)
            Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
            Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
            columns = vector_columns && (!chooses || !_sparse(s_in, s_n))
            if columns
                @simd for j in jlo:j_last
                    dx = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D))) - Xi
                    r2 = SFH.norm2(dx)
                    Uj = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D)))
                    keybuf[j - o] = digitize_key(plan, r2)
                    v = OT(SFT.flat_pair_value(sf, Uj - Ui, dx, r2))
                    valbuf[j - o] = v
                    if has_vector_index(plan)
                        idxbuf[j - o] = squared_approx_index(plan, r2)
                    end
                    q = needs_axis_buffer(second_axis) ? OT(axis_quantity(second_axis, dx, r2)) : v
                    if needs_axis_buffer(second_axis)
                        axbuf[j - o] = q
                    end
                    colbuf[j - o] = _vector_value_column(val_be, q, n_val)
                end
            else
                @simd for j in jlo:j_last
                    dx = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D))) - Xi
                    r2 = SFH.norm2(dx)
                    Uj = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D)))
                    keybuf[j - o] = digitize_key(plan, r2)
                    valbuf[j - o] = SFT.flat_pair_value(sf, Uj - Ui, dx, r2)
                    if has_vector_index(plan)
                        idxbuf[j - o] = squared_approx_index(plan, r2)
                    end
                    if needs_axis_buffer(second_axis)   # constant-folded on the operator-value axis
                        axbuf[j - o] = axis_quantity(second_axis, dx, r2)
                    end
                end
            end
            ks = (jlo - o):(j_last - o)
            s_in, s_n = _sample_in_range(plan, keybuf, ks)
            compact = chooses && _compacts(s_in, s_n)
            if columns && compact
                for m in 1:_compact_in_range!(sel, plan, keybuf, ks)
                    k = Int(sel[m])
                    _joint_accumulate!(sums2d, counts2d, valbuf, weights, wi, o, k,
                                       squared_bin_select(plan, keybuf[k], idxbuf[k]), Int(colbuf[k]))
                end
            elseif columns
                for k in ks
                    db = chooses ? squared_bin_select(plan, keybuf[k], idxbuf[k]) : squared_bin(plan, keybuf[k], idxbuf[k])
                    if 1 <= db <= n_dist
                        _joint_accumulate!(sums2d, counts2d, valbuf, weights, wi, o, k, db, Int(colbuf[k]))
                    end
                end
            elseif compact
                for m in 1:_compact_in_range!(sel, plan, keybuf, ks)
                    k = Int(sel[m])
                    _joint_accumulate!(sums2d, counts2d, valbuf, weights, wi, o, k,
                                       squared_bin_select(plan, keybuf[k], idxbuf[k]),
                                       _value_column(val_be, axis_key(second_axis, valbuf, axbuf, k), n_val))
                end
            else
                for k in ks
                    db = chooses ? squared_bin_select(plan, keybuf[k], idxbuf[k]) : squared_bin(plan, keybuf[k], idxbuf[k])
                    if 1 <= db <= n_dist
                        _joint_accumulate!(sums2d, counts2d, valbuf, weights, wi, o, k, db,
                                           _value_column(val_be, axis_key(second_axis, valbuf, axbuf, k), n_val))
                    end
                end
            end
        end
    end
    return nothing
end

"""The padded value column of `q`: its bin `digitize(q, val_be)` plus one, the first and last columns taking
the values below and above the edges."""
@inline _value_column(val_be, q, n_val) = clamp(SFH.digitize(q, val_be), 0, n_val + 1) + 1
@inline _vector_value_column(val_be, q, n_val) =
    clamp(vector_digitize(val_be, q), Int32(0), Int32(n_val + 1)) + Int32(1)

"""Add slot `k`'s buffered value into cell `(db, c)` of `sums2d`/`counts2d`, weighted by the outer point's weight
`wi` times that of point `k + o`."""
@inline function _joint_accumulate!(sums2d, counts2d::AbstractMatrix{CT}, valbuf, weights, wi, o, k, db, c) where {CT}
    w = wi * _point_weight(weights, k + o)
    @inbounds sums2d[db, c] += w * valbuf[k]
    @inbounds counts2d[db, c] += CT(w)
    return nothing
end

"""
    _pf_2d_run_blocks!(sums2d, counts2d, sf, xc, uc, plan, val_be, ::Val{D}, bufs..., ilist, N, grid)

2D-joint analogue of [`_pf_run_blocks!`](@ref): dispatch on `grid` so the kernel receives one
concretely typed schedule.
"""
@inline _pf_2d_run_blocks!(
    sums2d, counts2d, sf, xc, uc, plan, val_be, ::Val{D}, keybuf, valbuf, idxbuf, colbuf, sel,
    ilist, N, ::Nothing, second_axis = InvariantValueAxis(), axbuf = valbuf,
    weights = NoWeights(),
) where {D} = _pf_2d_simd_pairs!(sums2d, counts2d, sf, xc, uc, plan, val_be, Val(D),
    keybuf, valbuf, idxbuf, colbuf, sel, _pair_window(N), pair_blocks(N, ilist), second_axis, axbuf, weights)

@inline _pf_2d_run_blocks!(
    sums2d, counts2d, sf, xc, uc, plan, val_be, ::Val{D}, keybuf, valbuf, idxbuf, colbuf, sel,
    ilist, N, grid::CellGrid, second_axis = InvariantValueAxis(), axbuf = valbuf,
    weights = NoWeights(),
) where {D} = _pf_2d_simd_pairs!(sums2d, counts2d, sf, xc, uc, plan, val_be, Val(D),
    keybuf, valbuf, idxbuf, colbuf, sel, _pair_window(N), pair_blocks(N, ilist; grid = grid), second_axis, axbuf,
    weights)

function _pf_2d_simd_run!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, val_be, ::Val{D};
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {OT, CT, D}
    return _pf_2d_simd_partial!(sums2d, counts2d, sf, x_vecs, u_vecs, dist_be, val_be, Val(D),
                                nothing, culling; second_axis, weights)
end

"""
    _pf_2d_simd_partial!(sums2d, counts2d, sf, x_vecs, u_vecs, dist_be, val_be, ::Val{D}, share)

Run [`_pf_2d_simd_pairs!`](@ref) over the outer indices `share` selects ([`_share_indices`](@ref)), materializing
the contiguous component vectors, scratch buffers and padded accumulator this call needs, and add the accumulator's
value bins into `sums2d`/`counts2d`; the inputs may arrive as strided views.
"""
function _pf_2d_simd_partial!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, val_be, ::Val{D}, share,
    culling::CullingPolicy = AutoCulling();
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
    weights = NoWeights(),
) where {OT, CT, D}
    x_raw = ntuple(d -> collect(x_vecs[d]), Val(D))
    u_raw = ntuple(d -> collect(u_vecs[d]), Val(D))
    N = length(x_raw[1])
    L = _pair_scratch_length(N)
    keybuf = Vector{eltype(x_raw[1])}(undef, L)
    valbuf = Vector{OT}(undef, L)
    idxbuf = Vector{Int32}(undef, L)
    colbuf = Vector{Int32}(undef, L)
    sel = Vector{Int32}(undef, L)
    axbuf = needs_axis_buffer(second_axis) ? Vector{OT}(undef, L) : valbuf
    nd, nv = size(sums2d)
    sp, cp = zeros(OT, nd, nv + 2), zeros(CT, nd, nv + 2)
    plan = squared_digitize_plan(dist_be)
    grid = culling isa NoCulling ? nothing :
           cull_grid_for(x_raw, SFH.FlatGeometry{D}(), dist_be, culling)
    xc, uc = isnothing(grid) ? (x_raw, u_raw) :
             (apply_perm(x_raw, grid.perm), apply_perm(u_raw, grid.perm))
    wc = (weights isa NoWeights || isnothing(grid)) ? weights : weights[grid.perm]
    _pf_2d_run_blocks!(sp, cp, sf, xc, uc, plan, val_be, Val(D),
        keybuf, valbuf, idxbuf, colbuf, sel, _share_indices(grid, N - 1, share), N, grid, second_axis, axbuf, wc)
    sums2d .+= view(sp, :, 2:(nv + 1))
    counts2d .+= view(cp, :, 2:(nv + 1))
    return nothing
end

"""
    _partial_2d_sums_counts(inner, sf_type, x_vecs, u_vecs, distance_bins, value_bins, share, CT; kwargs...)

Partial 2D-joint sums/counts over share `share = (w, k)` of the outer indices, the 2D analogue of
[`_partial_sums_counts`](@ref): serially here, threaded by the OhMyThreads extension. Euclidean `D ∈ {2,3}` takes
the SIMD compute/scatter kernel; other metrics the scalar kernel. Returns `(sums, counts)`.
"""
function _partial_2d_sums_counts(
    ::CB.AbstractExecutionBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    share::NTuple{2, Int},
    ::Type{CT};
    geometry = SFH.FlatGeometry{length(u_vecs)}(),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
) where {CT}
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    nd = n_histogram_bins(distance_bins)
    nv = n_histogram_bins(value_bins)
    sums = zeros(OT, nd, nv)
    counts = zeros(CT, nd, nv)
    val_be = digitize_plan(value_bins)
    D = length(u_vecs)

    if geometry isa SFH.FlatGeometry && (D == 2 || D == 3)
        vD = D == 2 ? Val(2) : Val(3)
        _pf_2d_simd_partial!(sums, counts, structure_function_type, x_vecs, u_vecs, distance_bins, val_be,
                             vD, share, culling; weights = weights, second_axis = second_axis)
        return sums, counts
    end

    _require_value_axis(second_axis, geometry)
    _pf_2d_scalar_run!(sums, counts, geometry, structure_function_type, distance_bins, val_be, second_axis, share,
                       _cull_sorted(x_vecs, u_vecs, weights, geometry, distance_bins, culling))
    return sums, counts
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
    value_bins::AbstractVector,
    ::Type{CT};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    FT = promote_type(float(FT1), float(FT2))
    n_dist = n_histogram_bins(distance_bins)
    n_val = n_histogram_bins(value_bins)
    if ndims(u_arr) >= 3
        bdims = batch_dims(u_arr)
        sums = zeros(FT, n_dist, n_val, bdims...)
        counts = zeros(CT, n_dist, n_val, bdims...)
        auxiliary_joint2d!(sums, counts, structure_function_type, x_arr, u_arr, distance_bins, value_bins; kwargs...)
        return SFO.StructureFunction2DSumsAndCounts(structure_function_type, distance_bins, value_bins, sums, counts)
    end
    sums_2d = zeros(FT, n_dist, n_val)
    counts_2d = zeros(CT, n_dist, n_val)
    serial_calculate_structure_function!(sums_2d, counts_2d, structure_function_type, x_arr, u_arr, distance_bins,
                                         value_bins; kwargs...)
    return SFO.StructureFunction2DSumsAndCounts(structure_function_type, distance_bins, value_bins, sums_2d, counts_2d)
end

"""
    _pf_2d_scalar_pairs!(sums_2d, counts_2d, geom, sf, x_vecs, u_vecs, dist_be, val_be, blocks, weights,
                         second_axis)

Joint analogue of [`_pf_scalar_pairs!`](@ref): each pair of `blocks` binned by distance and by what
`second_axis` reads.
"""
function _pf_2d_scalar_pairs!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{CT},
    geom,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    dist_be,
    val_be,
    blocks,
    weights = NoWeights(),
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
) where {OT, CT, T1, T2}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    n_dist = n_histogram_bins(dist_be)
    n_val = n_histogram_bins(val_be)
    # The geometry carries the coordinate width, the field width and the velocity dimension; none of
    # them need equal another.
    vW = SFH.coordinate_width(geom)
    vF = SFH.field_width(geom)
    W = _val_int(vW)
    F = _val_int(vF)
    for (ir, jr) in blocks, i in ir
        X1 = SA.SVector{W, FT1}(ntuple(k -> @inbounds(x_vecs[k][i]), vW))
        U1 = SA.SVector{F, FT2}(ntuple(k -> @inbounds(u_vecs[k][i]), vF))
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            X2 = SA.SVector{W, FT1}(ntuple(k -> @inbounds(x_vecs[k][j]), vW))
            ok, distance, frame = SFH.pair_frame(geom, X1, X2)
            dist_bin = SFH.digitize(distance, dist_be)
            if ok && 1 <= dist_bin <= n_dist
                U2 = SA.SVector{F, FT2}(ntuple(k -> @inbounds(u_vecs[k][j]), vF))
                δu = SFH.pair_delta(geom, frame, X1, X2, U1, U2)
                val = SFT.pair_value(structure_function_type, geom, frame, distance, δu)
                val_bin = SFH.digitize(pair_axis_key(second_axis, val, X1, X2, distance), val_be)
                if 1 <= val_bin <= n_val
                    w = wi * _point_weight(weights, j)
                    @inbounds sums_2d[dist_bin, val_bin] += w * val
                    @inbounds counts_2d[dist_bin, val_bin] += CT(w)
                end
            end
        end
    end
    return nothing
end
