# Multi-field pair sweep. A field carrying more than one field forms its increment field by
# field — vectors transported, scalars differenced — and hands the whole multi-field to the operator,
# which reads the fields it names.

"""
    field_increment(::Fields{D,V,K}, data, geom, frame, r, i, j) -> FieldIncrement

One pair's increment across every field of a packed field.

Each vector field is transported into the pair's common frame before differencing, exactly as a
single-field case is; each scalar field is differenced where it stands, because a scalar has
nothing to transport.
"""
@inline function field_increment(
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
    return MF.FieldIncrement{Dp, V, K, T}(vectors, scalars)
end

"""
    _field_value(sf, ::Val{F}, ::Val{V}, ::Val{K}, data, geom, frame, r, i, j)

One pair's operator value for a multi-field, with the pair read in its canonical orientation
(see `pair_orientation`): an operator odd in a scalar increment takes the orientation's sign.
"""
@inline function _field_value(sf, ::Val{F}, ::Val{V}, ::Val{K}, data, geom, frame, r, i, j) where {F, V, K}
    inc = field_increment(Val(F), Val(V), Val(K), data, geom, frame, i, j)
    v = sf(inc, SFH.pair_direction(geom, frame, r))
    return SFT.is_odd_in_scalars(sf) ? SFH.pair_orientation(geom, frame) * v : v
end

"""
    _kernel_fields(fields, geom, x) -> (x_kernel, packed_kernel, Val{F})

The field in the form the kernels index: every vector field widened by the geometry exactly as a
single-field case is, scalars untouched.

On a sphere a velocity is carried as an ambient 3-vector, so a multi-field's vector fields must be
widened too — the same `prepare_pair_inputs` the array path uses, once per field, so there is no
second conversion to drift from it.
"""
function _kernel_fields(f::MF.Fields{D, V, K}, geom, x::AbstractMatrix) where {D, V, K}
    data = MF.packed(f)
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

Accumulate a multi-field's pairs into the 1-D distance histogram.

A field of one vector field and no scalars is the array path — it forwards there, so a bare `u` and
`Fields(vectors = (u,))` give the same answer through the same kernel. Anything carrying more than
one field takes this sweep, which builds each pair's whole increment and hands it to the
operator; it is a scalar loop, where the single-field path is a SIMD compute/scatter split.
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
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::CullingPolicy = AutoCulling(),
    weights = nothing,
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, D, V, K}
    N = size(MF.packed(f), 2)
    size(x, 2) == N || throw(DimensionMismatch(
        "x covers $(size(x, 2)) points and the field $N",
    ))
    w = _pair_weights(weights, N, float(eltype(MF.packed(f))))
    _check_weighted_counts(w, CT)
    if _on_a_line(_field_geometry(distance_metric, Val(D), Val(V), x), sf)
        _cull_reject_unsupported(culling, "the sorted line route")
        return sorted_line_sweep!(sums, counts, sf, _line_coordinates(x), MF.packed(f), distance_bins,
                                  Val(D), Val(V), Val(K); weights = w)
    end
    geom, xk, data, vF, plan, grid, wk = field_setup(f, x, distance_bins, distance_metric, culling, w)
    _field_run_blocks!(sums, counts, sf, xk, data, geom, vF, Val(V), Val(K), plan,
                         n_histogram_bins(plan), Val(SFC_val_int(SFH.coordinate_width(geom))),
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
    field_setup(fields, x, distance_bins, metric, culling, weights = NoWeights())
        -> (geom, xk, data, vF, plan, grid, weights)

Everything a multi-field sweep needs before its first pair: the geometry, the widened coordinates
and fields, the digitize plan, and the cull grid with both arrays and the weights already permuted
into it.

Shared by the serial and threaded drivers so the sort and the widening happen **once**, above any
task loop — doing them inside one would pay them per task.
"""
function field_setup(f::MF.Fields{D, V, K}, x::AbstractMatrix, distance_bins,
                       distance_metric, culling::CullingPolicy, weights = NoWeights()) where {D, V, K}
    geom = _field_geometry(distance_metric, Val(D), Val(V), x)
    xk, data, vF = _kernel_fields(f, geom, x)
    W = SFC_val_int(SFH.coordinate_width(geom))
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

# The geometry's dimension is the velocity dimension where there is one; a field of scalars alone has
# none, and its points are located by however many coordinates they carry.
@inline _field_geometry(distance_metric, ::Val{D}, ::Val{V}, x::AbstractMatrix) where {D, V} =
    V == 0 ? SFH.pair_geometry_for(distance_metric, Val(size(x, 1))) :
             SFH.pair_geometry_for(distance_metric, Val(D))

# Dispatch on the grid so the kernel receives one concretely typed schedule, as the single-field
# path does.
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

Each block pair is worked to completion, so its columns stay cache-resident across the `i` sweep —
the reason the single-field kernel is blocked, and it applies here for the same reason.
"""
# Flat geometry: the separation IS the displacement, so the whole per-pair computation is arithmetic
# on stack values and the compute half vectorizes — the same compute/scatter split the single-field
# kernel uses, and for the same reason (a scatter in the loop body stops it vectorizing).
function _field_pairs!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xk::AbstractMatrix, data::AbstractMatrix, geom::SFH.FlatGeometry, ::Val{F}, ::Val{V}, ::Val{K},
    plan::AbstractSquaredDigitizePlan, nb::Int, ::Val{W}, blocks, weights,
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
                inc = MF.FieldIncrement{F, V, K, T}(vectors, scalars)
                sgn = SFT.is_odd_in_scalars(sf) ? SFH.pair_orientation(geom, dx) : 1
                keybuf[j] = digitize_key(plan, r2)
                valbuf[j] = OT(sgn * sf(inc, dx / sqrt(r2)))
                if has_vector_index(plan)
                    idxbuf[j] = squared_approx_index(plan, r2)
                end
            end
            wi = _point_weight(weights, i)
            for j in jlo:j_last
                b = squared_bin(plan, keybuf[j], idxbuf[j])
                if 1 <= b <= nb
                    w = wi * _point_weight(weights, j)
                    sums[b] += w * valbuf[j]
                    counts[b] += CT(w)
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

@inline SFC_val_int(::Val{W}) where {W} = W

"""
    required_fields(operator) -> (n_vector, n_scalar)

The highest vector and scalar field index an operator reads.

Every operator that predates the multi-field reads the field itself, which is field 1 of the
vector side — that is what keeps `Fields(vectors = (u,))` identical to a bare `u`.
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

Checked once at the entry, where a threaded backend has not yet opened a task: an error thrown
inside one surfaces wrapped in a `TaskFailedException`.
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

"""
    field_partial(sf, x, fields, distance_bins, outer; kwargs...) -> (sums, counts)

A worker's share of a multi-field sweep: the pairs whose lower index is in `outer`, in freshly
allocated accumulators.

The outer lists partition `1:(N-1)`, so the partials add to the whole sweep exactly. Note the
indices are into the **culled ordering** when culling is on, which is a permutation of the input and
therefore still a partition.
"""
function field_partial(
    sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractMatrix, f::MF.Fields{D, V, K},
    distance_bins, outer;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::CullingPolicy = AutoCulling(),
    count_eltype::Type{CT} = UInt32,
) where {D, V, K, CT}
    N = size(MF.packed(f), 2)
    nb = n_histogram_bins(squared_digitize_plan(distance_bins))
    sums = zeros(float(eltype(MF.packed(f))), nb)
    counts = zeros(CT, nb)
    geom, xk, data, vF, plan, grid, wk = field_setup(f, x, distance_bins, distance_metric, culling)
    _field_run_blocks!(sums, counts, sf, xk, data, geom, vF, Val(V), Val(K), plan,
                         n_histogram_bins(plan), Val(SFC_val_int(SFH.coordinate_width(geom))),
                         outer, N, grid, wk)
    return sums, counts
end

"""
    distributed_calculate_structure_function!(sums, counts, sf, x, fields, bins; kwargs...)

Accumulate a multi-field sweep across worker processes. Supplied by the Distributed extension.
"""
function distributed_calculate_structure_function!(sums, counts, sf, x, f::MF.Fields, bins;
                                                   kwargs...)
    throw(ArgumentError(
        "the distributed multi-field sweep needs Distributed: run `using Distributed` and add " *
        "workers.",
    ))
end

"""
    gpu_calculate_structure_function!(backend, sums, counts, sf, x, fields, bins; kwargs...)

Accumulate a multi-field sweep on a device. Supplied by the KernelAbstractions extension.
"""
function gpu_calculate_structure_function_fields!(backend, sums, counts, sf, x, f::MF.Fields,
                                                    bins; kwargs...)
    throw(ArgumentError(
        "the GPU multi-field sweep needs KernelAbstractions: run `using KernelAbstractions` and " *
        "a device backend such as CUDA.",
    ))
end

# Backend selection for a multi-field, mirroring the array path's: the concrete backends dispatch,
# and `Auto` takes the threaded one when there are threads to use and the extension supplying it is
# loaded.
@inline _field_dispatch!(::CB.AbstractSerialBackend, sums, counts, sf, x, f, bins; kwargs...) =
    serial_calculate_structure_function!(sums, counts, sf, x, f, bins; kwargs...)

@inline _field_dispatch!(::CB.AbstractThreadedBackend, sums, counts, sf, x, f, bins; kwargs...) =
    threaded_calculate_structure_function!(sums, counts, sf, x, f, bins; kwargs...)

@inline function _field_dispatch!(::CB.AbstractDistributedBackend, sums, counts, sf, x, f, bins;
                                    kwargs...)
    _refuse_weights(kwargs, "the distributed multi-field sweep")
    return distributed_calculate_structure_function!(sums, counts, sf, x, f, bins; kwargs...)
end

@inline function _field_dispatch!(be::CB.AbstractGPUBackend, sums, counts, sf, x, f, bins; kwargs...)
    _refuse_weights(kwargs, "the GPU multi-field sweep")
    return gpu_calculate_structure_function_fields!(be, sums, counts, sf, x, f, bins; kwargs...)
end

function _field_dispatch!(::CB.AbstractAutoBackend, sums, counts, sf, x, f, bins; kwargs...)
    if Threads.nthreads() > 1 && _ohmythreads_loaded()
        return threaded_calculate_structure_function!(sums, counts, sf, x, f, bins; kwargs...)
    end
    return serial_calculate_structure_function!(sums, counts, sf, x, f, bins; kwargs...)
end

"""
    calculate_structure_function!(sums, counts, sf, x, fields, distance_bins; backend, kwargs...)

Accumulate a multi-field's pairs into `sums`/`counts` on `backend`.
"""
function calculate_structure_function!(
    sums, counts, sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractMatrix, f::MF.Fields,
    distance_bins; backend::CB.AbstractExecutionBackend = CB.SerialBackend(), kwargs...,
)
    _assert_counts_representable(eltype(counts), size(MF.packed(f), 2))
    validate_fields(sf, f)
    _field_dispatch!(backend, sums, counts, sf, x, f, distance_bins; kwargs...)
    return nothing
end

"""
    calculate_structure_function(sf, x, fields, distance_bins[, count_eltype]; kwargs...)

Structure function of a multi-field: several quantities sampled at the same points, swept
together in one pass, so a mixed moment reads every field at each pair.
"""
function calculate_structure_function(
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix,
    f::MF.Fields,
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction,
    kwargs...,
) where {OT, CT}
    N = size(MF.packed(f), 2)
    _assert_counts_representable(CT, N)
    nb = n_histogram_bins(squared_digitize_plan(distance_bins))
    validate_fields(sf, f)
    sums = zeros(float(eltype(MF.packed(f))), nb)
    counts = zeros(CT, nb)
    _field_dispatch!(backend, sums, counts, sf, x, f, distance_bins; kwargs...)
    raw = SFO.StructureFunctionSumsAndCounts(sf, distance_bins, sums, counts)
    return _finalize(raw, output_type)
end
