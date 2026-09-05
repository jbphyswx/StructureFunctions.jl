# Multi-channel pair sweep. A field carrying more than one channel forms its increment channel by
# channel — vectors transported, scalars differenced — and hands the whole bundle to the operator,
# which reads the channels it names.

"""
    channel_increment(::Fields{D,V,K}, data, geom, frame, r, i, j) -> ChannelIncrement

One pair's increment across every channel of a packed field.

Each vector channel is transported into the pair's common frame before differencing, exactly as a
single-channel field is; each scalar channel is differenced where it stands, because a scalar has
nothing to transport.
"""
@inline function channel_increment(
    ::Val{F}, ::Val{V}, ::Val{K}, data::AbstractMatrix{T}, geom, frame, i::Integer, j::Integer,
) where {F, V, K, T}
    vectors = ntuple(Val(V)) do c
        o = (c - 1) * F
        a = SA.SVector{F, T}(ntuple(d -> @inbounds(data[o + d, i]), Val(F)))
        b = SA.SVector{F, T}(ntuple(d -> @inbounds(data[o + d, j]), Val(F)))
        SFH.pair_delta(geom, frame, nothing, nothing, a, b)
    end
    scalars = ntuple(Val(K)) do c
        @inbounds data[V * F + c, j] - data[V * F + c, i]
    end
    Dp = V == 0 ? 0 : length(first(vectors))
    return CH.ChannelIncrement{Dp, V, K, T}(vectors, scalars)
end

"""
    _kernel_channels(fields, geom, x) -> (x_kernel, packed_kernel, Val{F})

The field in the form the kernels index: every vector channel widened by the geometry exactly as a
single-channel field is, scalars untouched.

On a sphere a velocity is carried as an ambient 3-vector, so a bundle's vector channels must be
widened too — the same `prepare_pair_inputs` the array path uses, once per channel, so there is no
second conversion to drift from it.
"""
function _kernel_channels(f::CH.Fields{D, V, K}, geom, x::AbstractMatrix) where {D, V, K}
    data = CH.packed(f)
    N = size(data, 2)
    F = SFC_val_int(SFH.field_width(geom))
    if F == D
        return SFH.prepare_coordinates(geom, x), data, Val(D)
    end
    T = float(eltype(data))
    out = Matrix{T}(undef, V * F + K, N)
    @inbounds for c in 1:V
        rows = ((c - 1) * D + 1):(c * D)
        _, vk = SFH.prepare_pair_inputs(geom, x, @view(data[rows, :]))
        out[((c - 1) * F + 1):(c * F), :] .= vk
    end
    @inbounds for c in 1:K
        out[V * F + c, :] .= @view(data[V * D + c, :])
    end
    return SFH.prepare_coordinates(geom, x), out, Val(F)
end

"""
    serial_calculate_structure_function!(sums, counts, sf, x, fields, distance_bins; kwargs...)

Accumulate a multi-channel field's pairs into the 1-D distance histogram.

A field of one vector channel and no scalars is the array path — it forwards there, so a bare `u` and
`Fields(vectors = (u,))` give the same answer through the same kernel. Anything carrying more than
one channel takes this sweep, which builds each pair's whole increment bundle and hands it to the
operator; it is a scalar loop, where the single-channel path is a SIMD compute/scatter split.
"""
function serial_calculate_structure_function!(
    sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::CH.Fields{D, 1, 0}, distance_bins;
    kwargs...,
) where {D}
    return serial_calculate_structure_function!(sums, counts, sf, x, CH.packed(f), distance_bins;
                                                kwargs...)
end

function serial_calculate_structure_function!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::CH.Fields{D, V, K}, distance_bins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::CullingPolicy = AutoCulling(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, D, V, K}
    N = size(CH.packed(f), 2)
    size(x, 2) == N || throw(DimensionMismatch(
        "x covers $(size(x, 2)) points and the field $N",
    ))
    geom, xk, data, vF, plan, grid = channel_setup(f, x, distance_bins, distance_metric, culling)
    _channel_run_blocks!(sums, counts, sf, xk, data, geom, vF, Val(V), Val(K), plan,
                         n_histogram_bins(plan), Val(SFC_val_int(SFH.coordinate_width(geom))),
                         1:(N - 1), N, grid)
    return nothing
end

"""
    channel_setup(fields, x, distance_bins, metric, culling) -> (geom, xk, data, vF, plan, grid)

Everything a multi-channel sweep needs before its first pair: the geometry, the widened coordinates
and channels, the digitize plan, and the cull grid with both arrays already permuted into it.

Shared by the serial and threaded drivers so the sort and the widening happen **once**, above any
task loop — doing them inside one would pay them per task.
"""
function channel_setup(f::CH.Fields{D, V, K}, x::AbstractMatrix, distance_bins,
                       distance_metric, culling::CullingPolicy) where {D, V, K}
    geom = SFH.pair_geometry_for(distance_metric, Val(max(D, 1)))
    xk, data, vF = _kernel_channels(f, geom, x)
    W = SFC_val_int(SFH.coordinate_width(geom))
    plan = squared_digitize_plan(distance_bins)
    xc = ntuple(d -> collect(view(xk, d, :)), Val(W))
    grid = culling isa NoCulling ? nothing : cull_grid_for(xc, geom, distance_bins, culling)
    if grid !== nothing
        xk = xk[:, grid.perm]
        data = data[:, grid.perm]
    end
    return geom, xk, data, vF, plan, grid
end

# Dispatch on the grid so the kernel receives one concretely typed schedule, as the single-channel
# path does.
@inline _channel_run_blocks!(sums, counts, sf, xk, data, geom, vF, vV, vK, plan, nb, vW, ilist, N,
                             ::Nothing) =
    _channel_pairs!(sums, counts, sf, xk, data, geom, vF, vV, vK, plan, nb, vW,
                    pair_blocks(N, ilist))

@inline _channel_run_blocks!(sums, counts, sf, xk, data, geom, vF, vV, vK, plan, nb, vW, ilist, N,
                             grid::CellGrid) =
    _channel_pairs!(sums, counts, sf, xk, data, geom, vF, vV, vK, plan, nb, vW,
                    pair_blocks(N, ilist; grid = grid))

"""
    _channel_pairs!(sums, counts, sf, xk, data, geom, ...) -> nothing

Accumulate the pairs `blocks` covers for a multi-channel field.

Each block pair is worked to completion, so its columns stay cache-resident across the `i` sweep —
the reason the single-channel kernel is blocked, and it applies here for the same reason.
"""
# Flat geometry: the separation IS the displacement, so the whole per-pair computation is arithmetic
# on stack values and the compute half vectorizes — the same compute/scatter split the single-channel
# kernel uses, and for the same reason (a scatter in the loop body stops it vectorizing).
function _channel_pairs!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xk::AbstractMatrix, data::AbstractMatrix, geom::SFH.FlatGeometry, ::Val{F}, ::Val{V}, ::Val{K},
    plan::AbstractSquaredDigitizePlan, nb::Int, ::Val{W}, blocks,
) where {OT, CT, F, V, K, W}
    FTx = eltype(xk)
    T = eltype(data)
    N = size(xk, 2)
    keybuf = Vector{FTx}(undef, N)
    valbuf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            Xi = SA.SVector{W, FTx}(ntuple(d -> xk[d, i], Val(W)))
            @simd for j in jlo:j_last
                Xj = SA.SVector{W, FTx}(ntuple(d -> xk[d, j], Val(W)))
                dx = Xj - Xi
                r2 = LA.dot(dx, dx)
                vectors = ntuple(Val(V)) do c
                    o = (c - 1) * F
                    SA.SVector{F, T}(ntuple(d -> data[o + d, j] - data[o + d, i], Val(F)))
                end
                scalars = ntuple(c -> data[V * F + c, j] - data[V * F + c, i], Val(K))
                inc = CH.ChannelIncrement{F, V, K, T}(vectors, scalars)
                keybuf[j] = digitize_key(plan, r2)
                valbuf[j] = OT(sf(inc, dx / sqrt(r2)))
                if has_vector_index(plan)
                    idxbuf[j] = squared_approx_index(plan, r2)
                end
            end
            for j in jlo:j_last
                b = squared_bin(plan, keybuf[j], idxbuf[j])
                if 1 <= b <= nb
                    sums[b] += valbuf[j]
                    counts[b] += one(CT)
                end
            end
        end
    end
    return nothing
end

function _channel_pairs!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xk::AbstractMatrix, data::AbstractMatrix, geom, ::Val{F}, ::Val{V}, ::Val{K},
    plan::AbstractSquaredDigitizePlan, nb::Int, ::Val{W}, blocks,
) where {OT, CT, F, V, K, W}
    FTx = eltype(xk)
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            Xi = SA.SVector{W, FTx}(ntuple(d -> xk[d, i], Val(W)))
            for j in jlo:j_last
                Xj = SA.SVector{W, FTx}(ntuple(d -> xk[d, j], Val(W)))
                ok, r, frame = SFH.pair_frame(geom, Xi, Xj)
                ok || continue
                b = squared_digitize(plan, r * r)
                1 <= b <= nb || continue
                inc = channel_increment(Val(F), Val(V), Val(K), data, geom, frame, i, j)
                sums[b] += OT(sf(inc, SFH.pair_direction(geom, frame, r)))
                counts[b] += one(CT)
            end
        end
    end
    return nothing
end

@inline SFC_val_int(::Val{W}) where {W} = W

"""
    calculate_structure_function(sf, x, fields, distance_bins[, count_eltype]; kwargs...)

Structure function of a multi-channel field: several quantities sampled at the same points, swept
together so a mixed moment costs one pass rather than one per channel.
"""
function calculate_structure_function(
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix,
    f::CH.Fields,
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32;
    output_type::Type{OT} = SFO.StructureFunction,
    kwargs...,
) where {OT, CT}
    N = size(CH.packed(f), 2)
    _assert_counts_representable(CT, N)
    nb = n_histogram_bins(squared_digitize_plan(distance_bins))
    sums = zeros(float(eltype(CH.packed(f))), nb)
    counts = zeros(CT, nb)
    serial_calculate_structure_function!(sums, counts, sf, x, f, distance_bins; kwargs...)
    raw = SFO.StructureFunctionSumsAndCounts(sf, distance_bins, sums, counts)
    return _finalize(raw, output_type)
end
