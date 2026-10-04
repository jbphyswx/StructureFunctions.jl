"""
    field_increment(Val(F), Val(V), Val(K), geom, frame, a, b) -> FieldIncrement

One pair's increment across every field of a packed column: `a` and `b` are the two points' columns,
`V` vector fields of width `F` followed by `K` scalars.

Each vector field is transported into the pair's common frame before differencing, as in the
single-field case; each scalar field is differenced directly.
"""
@inline function field_increment(::Val{F}, ::Val{V}, ::Val{K}, geom, frame, a::SA.SVector{L, T},
                                 b::SA.SVector{L, T}) where {F, V, K, L, T}
    vectors = ntuple(Val(V)) do c
        o = (c - 1) * F
        ac = SA.SVector{F, T}(ntuple(d -> @inbounds(a[o + d]), Val(F)))
        bc = SA.SVector{F, T}(ntuple(d -> @inbounds(b[o + d]), Val(F)))
        SFH.pair_delta(geom, frame, nothing, nothing, ac, bc)
    end
    scalars = ntuple(c -> @inbounds(b[V * F + c] - a[V * F + c]), Val(K))
    Dp = V == 0 ? 0 : length(first(vectors))
    return MF.FieldIncrement{Dp, V, K, T}(vectors, scalars)
end

"""The increment of the pair `(i, j)` of the packed field `data`."""
@inline function field_increment(vF::Val{F}, vV::Val{V}, vK::Val{K}, data::AbstractMatrix{T}, geom, frame,
                                 i::Integer, j::Integer) where {F, V, K, T}
    a = SA.SVector{V * F + K, T}(ntuple(r -> @inbounds(data[r, i]), Val(V * F + K)))
    b = SA.SVector{V * F + K, T}(ntuple(r -> @inbounds(data[r, j]), Val(V * F + K)))
    return field_increment(vF, vV, vK, geom, frame, a, b)
end

"""
    _field_value(sf, ::Val{F}, ::Val{V}, ::Val{K}, data, geom, frame, r, i, j)

One pair's operator value for a multi-field, with the pair read in its canonical orientation
(see `pair_orientation`): an operator odd in a scalar increment takes the orientation's sign.
"""
@inline function _field_value(sf, ::Val{F}, ::Val{V}, ::Val{K}, data, geom, frame, r, i, j) where {F, V, K}
    inc = field_increment(Val(F), Val(V), Val(K), data, geom, frame, i, j)
    v = SFT.pair_value(sf, geom, frame, r, inc)
    return SFT.is_odd_in_scalars(sf) ? SFH.pair_orientation(geom, frame) * v : v
end

"""
    _kernel_fields(fields, geom, x) -> (x_kernel, packed_kernel, Val{F})

The field in the form the kernels index: every vector field widened by the geometry as in the
single-field case, scalars untouched.

On a sphere a velocity is carried as an ambient 3-vector; each vector field is widened with
`prepare_pair_inputs`, the array path's conversion.
"""
function _kernel_fields(f::MF.Fields{D, V, K}, geom, x::AbstractMatrix) where {D, V, K}
    data = MF.packed(f)
    N = size(data, 2)
    F = _val_int(SFH.field_width(geom))
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

Accumulate a multi-field's pairs into the 1-D distance histogram.

A field of one vector field and no scalars forwards to the array path, so a bare `u` and
`Fields(vectors = (u,))` give the same answer. Anything carrying more than one field takes this
sweep, which builds each pair's whole increment and hands it to the operator: a SIMD
compute/scatter split on flat geometry, a scalar loop otherwise.
"""
function serial_calculate_structure_function!(
    sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::MF.Fields{D, 1, 0}, distance_bins;
    kwargs...,
) where {D}
    return serial_calculate_structure_function!(sums, counts, sf, x, MF.packed(f), distance_bins;
                                                kwargs...)
end

function serial_calculate_structure_function!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::MF.Fields{D, V, K}, distance_bins;
    geometry,
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {OT, CT, D, V, K}
    N = size(MF.packed(f), 2)
    size(x, 2) == N || throw(DimensionMismatch(
        "x covers $(size(x, 2)) points and the field $N",
    ))
    if _on_a_line(geometry, sf)
        return sorted_line_sweep!(sums, counts, sf, _line_coordinates(x), MF.packed(f), distance_bins,
                                  Val(D), Val(V), Val(K); weights)
    end
    geom, xk, data, vF, plan, grid, wk = field_setup(f, x, distance_bins, geometry, culling, weights)
    _field_run_blocks!(sums, counts, sf, xk, data, geom, vF, Val(V), Val(K), plan,
                         n_histogram_bins(plan), SFH.coordinate_width(geom),
                         1:(N - 1), N, grid, wk)
    return nothing
end

"""Whether the pairs lie along a line and the operator is a polynomial the sorted route sums."""
@inline _on_a_line(geom, sf) = geom isa SFH.FlatGeometry{1} && SFT.is_polynomial_operator(sf)

"""The one coordinate per point of a `(1, N)` position matrix."""
function _line_coordinates(x::AbstractMatrix)
    size(x, 1) == 1 || throw(DimensionMismatch(
        "points on a line carry one coordinate each; got $(size(x, 1)) rows of coordinates",
    ))
    return vec(x)
end

"""
    field_setup(fields, x, distance_bins, geometry, culling, weights = NoWeights())
        -> (geom, xk, data, vF, plan, grid, weights)

Everything a multi-field sweep on `geometry` needs before its first pair: the widened coordinates
and fields, the digitize plan, and the cull grid with both arrays and the weights already permuted
into it.

Shared by the serial and threaded drivers; the sort and the widening happen once, above any task loop.
"""
function field_setup(f::MF.Fields{D, V, K}, x::AbstractMatrix, distance_bins,
                       geom, culling::CullingPolicy, weights = NoWeights()) where {D, V, K}
    xk, data, vF = _kernel_fields(f, geom, x)
    W = _val_int(SFH.coordinate_width(geom))
    plan = squared_digitize_plan(distance_bins)
    xc = ntuple(d -> collect(view(xk, d, :)), Val(W))
    grid = culling isa NoCulling ? nothing : cull_grid_for(xc, geom, distance_bins, culling)
    if grid !== nothing
        xk = xk[:, grid.perm]
        data = data[:, grid.perm]
        weights = weights isa NoWeights ? weights : weights[grid.perm]
    end
    return geom, xk, data, vF, plan, grid, weights
end

"""`g(geometry)` with the pair geometry of `distance_metric` for a multi-field over the positions `x`: at the vector
fields' width, or for a field of scalars alone, which has none, at the coordinate count of `x` through
[`_shaped`](@ref)."""
@inline _with_field_geometry(g, ::MF.Fields{D, V}, x::AbstractMatrix, distance_metric) where {D, V} =
    V == 0 ? _shaped((_, geometry) -> g(geometry), PointField, size(x, 1), distance_metric) :
             g(SFH.pair_geometry_for(distance_metric, Val(D)))

@inline _field_run_blocks!(sums, counts, sf, xk, data, geom, vF, vV, vK, plan, nb, vW, ilist, N,
                             ::Nothing, weights) =
    _field_pairs!(sums, counts, sf, xk, data, geom, vF, vV, vK, plan, nb, vW,
                    pair_blocks(N, ilist), weights)

@inline _field_run_blocks!(sums, counts, sf, xk, data, geom, vF, vV, vK, plan, nb, vW, ilist, N,
                             grid::CellGrid, weights) =
    _field_pairs!(sums, counts, sf, xk, data, geom, vF, vV, vK, plan, nb, vW,
                    pair_blocks(N, ilist; grid = grid), weights)

"""
    _field_pairs!(sums, counts, sf, xk, data, geom, ..., blocks, weights) -> nothing

Accumulate the pairs `blocks` covers for a multi-field, each pair carrying
`weights[i] * weights[j]` in both sums and counts.

Each block pair is worked to completion, so its columns stay cache-resident across the `i` sweep.
"""
# Flat geometry: the separation is the displacement; the compute half runs under `@simd`, the scatter is scalar.
function _field_pairs!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xk::AbstractMatrix, data::AbstractMatrix, geom::SFH.FlatGeometry, vF::Val, vV::Val, vK::Val,
    plan::AbstractSquaredDigitizePlan, nb::Int, vW::Val, blocks, weights,
) where {OT, CT}
    window = _pair_window(size(xk, 2))
    L = _pair_scratch_length(window, size(xk, 2))
    return _field_pairs!(sums, counts, sf, xk, data, geom, vF, vV, vK, plan, nb, vW, blocks, weights, window,
                         Vector{eltype(xk)}(undef, L), Vector{OT}(undef, L), Vector{Int32}(undef, L),
                         Vector{Int32}(undef, L))
end

function _field_pairs!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xk::AbstractMatrix, data::AbstractMatrix, geom::SFH.FlatGeometry, ::Val{F}, ::Val{V}, ::Val{K},
    plan::AbstractSquaredDigitizePlan, nb::Int, ::Val{W}, blocks, weights, window::PairWindow,
    keybuf::AbstractVector, valbuf::AbstractVector, idxbuf::AbstractVector{Int32}, sel::AbstractVector{Int32},
) where {OT, CT, F, V, K, W}
    FTx = eltype(xk)
    T = eltype(data)
    chooses = _chooses_compaction(blocks)
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        _check_run_fits(window, valbuf, jr)
        off = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            Xi = SA.SVector{W, FTx}(ntuple(d -> xk[d, i], Val(W)))
            wi = _point_weight(weights, i)
            @simd for j in jlo:j_last
                Xj = SA.SVector{W, FTx}(ntuple(d -> xk[d, j], Val(W)))
                dx = Xj - Xi
                r2 = SFH.norm2(dx)
                vectors = ntuple(Val(V)) do c
                    o = (c - 1) * F
                    SA.SVector{F, T}(ntuple(d -> data[o + d, j] - data[o + d, i], Val(F)))
                end
                scalars = ntuple(c -> data[V * F + c, j] - data[V * F + c, i], Val(K))
                inc = MF.FieldIncrement{F, V, K, T}(vectors, scalars)
                sgn = SFT.is_odd_in_scalars(sf) ? SFH.pair_orientation(geom, dx) : 1
                keybuf[j - off] = digitize_key(plan, r2)
                valbuf[j - off] = OT(sgn * SFT.flat_pair_value(sf, inc, dx, r2))
                if has_vector_index(plan)
                    idxbuf[j - off] = squared_approx_index(plan, r2)
                end
            end
            ks = (jlo - off):(j_last - off)
            if chooses && _compacts(_sample_in_range(plan, keybuf, ks)...)
                for m in 1:_compact_in_range!(sel, plan, keybuf, ks)
                    k = Int(sel[m])
                    _pf_accumulate!(sums, counts, valbuf, weights, wi, off, k,
                                    squared_bin_select(plan, keybuf[k], idxbuf[k]))
                end
            else
                for k in ks
                    b = chooses ? squared_bin_select(plan, keybuf[k], idxbuf[k]) : squared_bin(plan, keybuf[k], idxbuf[k])
                    if 1 <= b <= nb
                        _pf_accumulate!(sums, counts, valbuf, weights, wi, off, k, b)
                    end
                end
            end
        end
    end
    return nothing
end

function _field_pairs!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xk::AbstractMatrix, data::AbstractMatrix, geom, ::Val{F}, ::Val{V}, ::Val{K},
    plan::AbstractSquaredDigitizePlan, nb::Int, ::Val{W}, blocks, weights,
) where {OT, CT, F, V, K, W}
    FTx = eltype(xk)
    @inbounds for (ir, jr) in blocks
        j_first, j_last = first(jr), last(jr)
        for i in ir
            jlo = max(i + 1, j_first)
            jlo > j_last && continue
            Xi = SA.SVector{W, FTx}(ntuple(d -> xk[d, i], Val(W)))
            wi = _point_weight(weights, i)
            for j in jlo:j_last
                Xj = SA.SVector{W, FTx}(ntuple(d -> xk[d, j], Val(W)))
                ok, r, frame = SFH.pair_frame(geom, Xi, Xj)
                ok || continue
                b = squared_digitize(plan, r * r)
                1 <= b <= nb || continue
                w = wi * _point_weight(weights, j)
                sums[b] += w * OT(_field_value(sf, Val(F), Val(V), Val(K), data, geom, frame, r, i, j))
                counts[b] += CT(w)
            end
        end
    end
    return nothing
end

"""
    required_fields(operator) -> (n_vector, n_scalar)

The highest vector and scalar field index an operator reads.

An operator with no field index reads vector field 1.
"""
@inline required_fields(::SFT.AbstractPairwiseStructureFunctionType) = (1, 0)
@inline required_fields(sf::SFT.ScalarStructureFunctionType) = (0, sf.field)
@inline required_fields(sf::SFT.MixedStructureFunctionType) =
    (sf.vector_field, sf.scalar_field)
@inline required_fields(sf::SFT.VectorDotStructureFunctionType) = (max(sf.a, sf.b), 0)
@inline required_fields(sf::SFT.ScalarDotStructureFunctionType) = (0, max(sf.a, sf.b))

"""
    validate_fields(operator, fields)

Refuse an operator that reads a field the multi-field does not carry.

Checked once at the entry, before any backend task opens.
"""
validate_fields(sf::SFT.AbstractPairwiseStructureFunctionType, ::MF.Fields{D, V, K}) where {D, V, K} =
    validate_fields(sf, Val(V), Val(K))

function validate_fields(sf::SFT.AbstractPairwiseStructureFunctionType, ::Val{V}, ::Val{K}) where {V, K}
    nv, ns = required_fields(sf)
    nv <= V || throw(ArgumentError(
        "$(nameof(typeof(sf))) reads vector field $nv, but this field carries $V. Build the " *
        "multi-field with that field, e.g. `Fields(vectors = (u, 𝓐u))`.",
    ))
    ns <= K || throw(ArgumentError(
        "$(nameof(typeof(sf))) reads scalar field $ns, but this field carries $K. Build the " *
        "multi-field with that field, e.g. `Fields(vectors = (u,), scalars = (θ,))`.",
    ))
    return nothing
end

function validate_fields(::SFT.SinglePassInvariants, ::Val{V}, ::Val{K}) where {V, K}
    (V == 1 && K == 0) || throw(ArgumentError(
        "the single-pass invariants are of one vector field; got $V vector and $K scalar field(s)",
    ))
    return nothing
end

"""
    field_partial(inner, sf, x, fields, distance_bins, share, CT; geometry, culling, weights) -> (sums, counts)

A worker's share of a multi-field sweep: the pairs whose lower index is in share `share = (w, k)` of the outer
indices, resolved against the cull grid the worker builds ([`_share_indices`](@ref)), in freshly allocated
accumulators, computed on the worker's local backend `inner`. The shares of `w = 1:k` partition the sweep, so the
partials add to it exactly.
"""
function field_partial(
    inner::CB.AbstractExecutionBackend, sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractMatrix,
    f::MF.Fields, distance_bins, share::NTuple{2, Int}, ::Type{CT}; kwargs...,
) where {CT}
    nb = n_histogram_bins(distance_bins)
    sums, counts = zeros(float(eltype(MF.packed(f))), nb), zeros(CT, nb)
    _field_into!(inner, sums, counts, sf, x, f, distance_bins, share; kwargs...)
    return sums, counts
end

"""The pairs of a multi-field sweep whose lower index is in share `share` of the outer indices added into
`sums`/`counts` on the backend `inner`: serially here, threaded by the OhMyThreads extension."""
function _field_into!(
    ::CB.AbstractExecutionBackend, sums, counts, sf, x, f::MF.Fields{D, V, K}, distance_bins, share;
    geometry, culling::CullingPolicy = AutoCulling(), weights = NoWeights(),
) where {D, V, K}
    geom, xk, data, vF, plan, grid, wk = field_setup(f, x, distance_bins, geometry, culling, weights)
    N = size(MF.packed(f), 2)
    _field_run_blocks!(sums, counts, sf, xk, data, geom, vF, Val(V), Val(K), plan, n_histogram_bins(plan),
                       SFH.coordinate_width(geom), _share_indices(grid, N - 1, share), N, grid, wk)
    return nothing
end

"""
    distributed_calculate_structure_function!(inner, sums, counts, sf, x, fields, bins; kwargs...)

Accumulate a multi-field sweep across worker processes, each on the local backend `inner`. Supplied by
the Distributed extension.
"""
function distributed_calculate_structure_function! end

"""
    gpu_calculate_structure_function_fields!(backend, sums, counts, sf, x, fields, bins; kwargs...)

Accumulate a multi-field sweep on a device. Supplied by the KernelAbstractions extension.
"""
function gpu_calculate_structure_function_fields! end

@inline _field_dispatch!(::CB.AbstractSerialBackend, sums, counts, sf, x, f, bins; kwargs...) =
    serial_calculate_structure_function!(sums, counts, sf, x, f, bins; kwargs...)

@inline _field_dispatch!(::CB.AbstractThreadedBackend, sums, counts, sf, x, f, bins; kwargs...) =
    threaded_calculate_structure_function!(sums, counts, sf, x, f, bins; kwargs...)

"""
    mpi_calculate_structure_function!(sums, counts, sf, x, fields, bins; kwargs...)

Accumulate a multi-field structure function across MPI ranks. Supplied by the MPI extension.
"""
function mpi_calculate_structure_function! end

@inline function _field_dispatch!(b::CB.AbstractMPIBackend, sums, counts, sf, x, f, bins; kwargs...)
    return mpi_calculate_structure_function!(sums, counts, sf, x, f, bins; backend = b, kwargs...)
end

@inline function _field_dispatch!(b::CB.AbstractDistributedBackend, sums, counts, sf, x, f, bins;
                                    kwargs...)
    return distributed_calculate_structure_function!(CB.local_backend(b), sums, counts, sf, x, f, bins; kwargs...)
end

@inline function _field_dispatch!(be::CB.AbstractGPUBackend, sums, counts, sf, x, f, bins; kwargs...)
    return gpu_calculate_structure_function_fields!(be, sums, counts, sf, x, f, bins; kwargs...)
end

_field_dispatch!(::CB.AbstractAutoBackend, sums, counts, sf, x, f, bins; kwargs...) =
    _field_dispatch!(resolve_auto_backend(), sums, counts, sf, x, f, bins; kwargs...)

"""
    calculate_structure_function!(sums, counts, sf, x, fields, distance_bins; backend, kwargs...)

Accumulate a multi-field's pairs into `sums`/`counts` on `backend`.
"""
function calculate_structure_function!(
    sums, counts, sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractMatrix, f::MF.Fields,
    distance_bins; backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(), weights = nothing, kwargs...,
)
    _require_backend(backend)
    w = _pair_weights(weights, size(MF.packed(f), 2), eltype(sums))
    _assert_counts_can_accumulate(counts, size(MF.packed(f), 2), w)
    validate_fields(sf, f)
    _with_field_geometry(f, x, distance_metric) do geometry
        _field_dispatch!(backend, sums, counts, sf, x, f, distance_bins; geometry, weights = w, kwargs...)
    end
    return nothing
end

"""
    calculate_structure_function(sf, x, fields, distance_bins[, CT][, OT]; backend, weights, kwargs...)

Structure function of a multi-field: several quantities sampled at the same points, swept
together in one pass, so a mixed moment reads every field at each pair.
"""
function calculate_structure_function(
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix,
    f::MF.Fields,
    distance_bins::AbstractVector,
    ::Type{CT},
    ::Type{OT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
    kwargs...,
) where {CT <: Real, OT <: SFO.AbstractStructureFunction}
    _require_backend(backend)
    N = size(MF.packed(f), 2)
    ST = float(eltype(MF.packed(f)))
    w = _pair_weights(weights, N, ST)
    _assert_count_type(CT, N, w)
    nb = n_histogram_bins(distance_bins)
    validate_fields(sf, f)
    sums = _result_zeros(backend, ST, nb)
    counts = _result_zeros(backend, CT, nb)
    _with_field_geometry(f, x, distance_metric) do geometry
        _field_dispatch!(backend, sums, counts, sf, x, f, distance_bins; geometry, weights = w, kwargs...)
    end
    raw = SFO.StructureFunctionSumsAndCounts(sf, distance_bins, sums, counts)
    return _finalize(raw, OT)
end

calculate_structure_function(sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractMatrix, f::MF.Fields,
                             distance_bins::AbstractVector; kwargs...) =
    calculate_structure_function(sf, x, f, distance_bins, DEFAULT_COUNT_TYPE, SFO.StructureFunction; kwargs...)
calculate_structure_function(sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractMatrix, f::MF.Fields,
                             distance_bins::AbstractVector, ::Type{CT}; kwargs...) where {CT <: Real} =
    calculate_structure_function(sf, x, f, distance_bins, CT, SFO.StructureFunction; kwargs...)
calculate_structure_function(sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractMatrix, f::MF.Fields,
                             distance_bins::AbstractVector, ::Type{OT}; kwargs...) where {OT <: SFO.AbstractStructureFunction} =
    calculate_structure_function(sf, x, f, distance_bins, DEFAULT_COUNT_TYPE, OT; kwargs...)
