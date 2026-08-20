# Serial 1D CPU Reduction Kernels

function serial_calculate_structure_function!(
    output::AbstractVector{OT},
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, T1, T2}
    distance_bins = BinEdges(distance_bins)

    if verbose
        @info("calculating structure function (serial reduction)")
    end

    # Fast path: Euclidean + D ∈ (2,3) uses the SIMD compute/scatter-split kernel (vectorizes
    # the per-pair compute over j; only the histogram scatter is scalar). Other metrics/D fall
    # back to the scalar per-i kernel.
    D = length(u_vecs)
    if distance_metric isa DI.Euclidean && D == 2
        return _pf_simd_run!(output, counts, structure_function_type, x_vecs, u_vecs, distance_bins, Val(2))
    elseif distance_metric isa DI.Euclidean && D == 3
        return _pf_simd_run!(output, counts, structure_function_type, x_vecs, u_vecs, distance_bins, Val(3))
    end

    vN = Val(D)
    PM.@showprogress enabled = show_progress for i in eachindex(x_vecs[1])
        calculate_structure_function_i!(
            output, counts, vN, structure_function_type, i, x_vecs, u_vecs, distance_bins;
            distance_metric = distance_metric,
        )
    end
    return nothing
end

"""
    _pf_simd_run!(output, counts, sf, x_vecs, u_vecs, dist_be, ::Val{D}) -> mutates buffers

Point-field 1D (Euclidean) via the SIMD compute/scatter split. Materializes contiguous
per-component vectors (so consecutive `j` are unit-stride → packed loads), then for each `i`:
`@simd` over `j>i` computes distance + SF value into buffers (no scatter ⇒ vectorizes), and a
short scalar loop digitizes + scatters into the histogram. `Val{D}` keeps it type-stable.
"""
function _pf_simd_run!(
    output::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, ::Val{D},
) where {OT, CT, D}
    N = length(x_vecs[1])
    return _pf_simd_partial!(output, counts, sf, x_vecs, u_vecs, dist_be, Val(D), 1:(N - 1))
end

"""
    _pf_simd_pairs!(output, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, irange)

Accumulate pairs `(i, j>i)` for `i in irange` into `output`/`counts`.

The `@simd` half writes `r²`, the SF value, and the approximate bin index to buffers; the scalar
half corrects the index and scatters straight into `output`/`counts`, skipping out-of-range bins.

The `i`-loop and the inner `@simd` must stay in this function body; factoring the inner loop into a
per-`i` helper stops it vectorizing.
"""
function _pf_simd_pairs!(
    output::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, ::Val{D},
    r2buf::AbstractVector, valbuf::AbstractVector, idxbuf::AbstractVector{Int32},
    irange,
) where {OT, CT, D}
    N = length(xc[1])
    nb = n_histogram_bins(plan)
    FTx = eltype(xc[1])
    @inbounds for i in irange
        Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
        Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
        @simd for j in (i + 1):N
            Xj = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D)))
            dx = Xj - Xi
            r2 = LA.dot(dx, dx)
            Uj = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D)))
            r2buf[j] = digitize_key(plan, r2)
            valbuf[j] = SFT._sf_raw(sf, Uj - Ui, dx, r2)
            if has_vector_index(plan)      # constant-folded: depends only on the plan type
                idxbuf[j] = squared_approx_index(plan, r2)
            end
        end
        for j in (i + 1):N
            b = squared_bin(plan, r2buf[j], idxbuf[j])
            if 1 <= b <= nb
                output[b] += valbuf[j]
                counts[b] += one(CT)
            end
        end
    end
    return nothing
end

"""
    _assert_counts_representable(CT, n_points)

Throw unless the worst-case pair count `n_points*(n_points-1)÷2` fits in `CT`.

Every pair can land in one bin, so that product is the only safe bound. `UInt32` saturates at
`N = 92682`, past which the counter wraps silently.
"""
@inline function _assert_counts_representable(::Type{CT}, n_points::Integer) where {CT <: Integer}
    n_pairs = (Int128(n_points) * (Int128(n_points) - 1)) ÷ 2
    n_pairs <= Int128(typemax(CT)) || throw(
        ArgumentError(
            "count_eltype=$CT cannot represent the worst-case pair count $n_pairs for N=$n_points " *
            "(typemax($CT) = $(typemax(CT))); pass count_eltype=UInt64 or Int64.",
        ),
    )
    return nothing
end

"""
    _bin_average!(out, sums, counts)
    _bin_average(sums, counts)

Per-bin mean `sums ./ counts` with the empty-bin guard `count == 0 → NaN`. The cast uses
`eltype(out)` (so Float32 stays Float32, Float64 stays Float64). Elementwise: works for 1D
vectors, 2D matrices, and batched `(n_bins, batch...)` arrays whose `sums`/`counts` share a
shape. The allocating form returns a fresh array of `eltype(sums)`. This is the single
canonical averaging used by `_finalize`.
"""
function _bin_average!(out::AbstractArray{T}, sums::AbstractArray, counts::AbstractArray) where {T}
    @inbounds for k in eachindex(out, sums, counts)
        c = counts[k]
        out[k] = iszero(c) ? T(NaN) : sums[k] / c
    end
    return out
end

@inline _bin_average(sums::AbstractArray, counts::AbstractArray) =
    _bin_average!(similar(sums, eltype(sums)), sums, counts)

"""
    _tensor_bin_average(sums, counts, ::Val{P})

Tensor analogue of [`_bin_average`](@ref): `counts` (indexed by `(bin, aux...)`) broadcasts over
the `P` leading component axes of `sums` (shape `(D×P..., n_bins, aux...)`). Same empty-bin guard
(`count == 0 → NaN`) and `eltype` preservation. Used by `_finalize` to average a tensor result.
"""
function _tensor_bin_average(sums::AbstractArray, counts::AbstractArray, ::Val{P}) where {P}
    T = eltype(sums)
    out = similar(sums, T)
    comp = CartesianIndices(ntuple(d -> axes(sums, d), Val(P)))   # D^P component indices
    rest = CartesianIndices(axes(sums)[(P + 1):end])              # (n_bins, aux...)
    @inbounds for r in rest
        c = counts[r]
        if iszero(c)
            for ci in comp
                out[ci, r] = T(NaN)
            end
        else
            for ci in comp
                out[ci, r] = sums[ci, r] / c
            end
        end
    end
    return out
end

# Non-mutating backends always return the raw accumulator (`StructureFunctionSumsAndCounts`);
# the public boundary picks the representation via `_finalize(raw, output_type)`.
function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {T1, T2, CT}
    _assert_counts_representable(CT, length(x_vecs[1]))
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    output = zeros(OT, N3)
    counts = zeros(CT, N3)

    serial_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        kwargs...,
    )

    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type,
        distance_bins,
        output,
        counts,
    )
end

function serial_calculate_structure_function!(
    sums::AbstractVector{OT},
    counts::AbstractVector,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector;
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number}
    x_tuple = ntuple(k -> view(x_arr, k, :), size(x_arr, 1))
    u_tuple = ntuple(k -> view(u_arr, k, :), size(u_arr, 1))
    return serial_calculate_structure_function!(
        sums,
        counts,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins;
        kwargs...,
    )
end

function calculate_structure_function_i!(
    output::AbstractVector{OT},
    counts::AbstractVector,
    ::Val{N},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {OT, N, T1, T2}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins)

    # `N` is the velocity dimension; the coordinate count is the x tuple's own (static) length,
    # which differs on a shell.
    W = length(x_vecs)
    vW = Val(W)
    X1 = SA.SVector{W, FT1}(ntuple(k -> @inbounds(x_vecs[k][i]), vW))
    U1 = SA.SVector{N, FT2}(ntuple(k -> @inbounds(u_vecs[k][i]), Val(N)))

    iter_inds = eachindex(x_vecs[1])
    geom = SFH.pair_geometry_for(distance_metric, Val(N))
    # @inbounds: x_vecs[k] are strided views; the bounds checks on every component access
    # were a large per-pair overhead. U2 is built only for in-range pairs.
    @inbounds for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{W, FT1}(ntuple(k -> x_vecs[k][j], vW))

        ok, distance, frame = SFH.pair_frame(geom, X1, X2)
        bin = SFH.digitize(distance, distance_bins)
        if ok && 1 <= bin < N3
            U2 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][j], Val(N)))
            δu, rh = SFH.pair_increments(geom, frame, distance, X1, X2, U1, U2)
            output[bin] += structure_function_type(δu, rh)
            counts[bin] += 1
        end
    end
    return nothing
end

function calculate_structure_function_i(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    _assert_counts_representable(CT, length(x_vecs[1]))
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    N3 = n_histogram_bins(distance_bins)
    local_output = zeros(OT, N3)
    local_counts = zeros(CT, N3)
    calculate_structure_function_i!(
        local_output, local_counts,
        Val(length(u_vecs)),
        structure_function_type, i, x_vecs, u_vecs, BinEdges(distance_bins);
        distance_metric = distance_metric,
    )
    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type,
        distance_bins,
        local_output,
        local_counts,
    )
end

"""
    _partial_sums_counts(inner, sf_type, x_vecs, u_vecs, distance_bins, ilist; kwargs...)

Partial 1D sums/counts over an explicit outer-index list `ilist` (each `i` contributes pairs
`(i, j>i)`). Used by the distributed driver to give each worker a balanced share; `inner`
selects how the worker computes its share locally. This generic method runs SERIALLY for any
backend; the OhMyThreads extension adds a `::CB.AbstractThreadedBackend` method that threads over `ilist`
(enabling hybrid distributed+threaded). Returns a `StructureFunctionSumsAndCounts`.
"""
function _partial_sums_counts(
    ::CB.AbstractExecutionBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    ilist;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    _assert_counts_representable(CT, length(x_vecs[1]))
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    nb = n_histogram_bins(distance_bins)
    sums = zeros(OT, nb)
    counts = zeros(CT, nb)
    be = BinEdges(distance_bins)
    D = length(u_vecs)
    # Euclidean D ∈ {2,3} takes the SIMD compute/scatter kernel, the same one the serial and
    # threaded drivers use; `_pf_simd_pairs!` accepts an arbitrary `irange`. Other metrics or
    # dimensions fall back to the scalar per-`i` kernel.
    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        vD = D == 2 ? Val(2) : Val(3)
        _pf_simd_partial!(sums, counts, structure_function_type, x_vecs, u_vecs, be, vD, ilist)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end

    vN = Val(D)
    for i in ilist
        calculate_structure_function_i!(
            sums, counts, vN, structure_function_type, i, x_vecs, u_vecs, be;
            distance_metric = distance_metric,
        )
    end
    return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
end

"""
    _pf_simd_partial!(sums, counts, sf, x_vecs, u_vecs, dist_be, ::Val{D}, ilist)

Run [`_pf_simd_pairs!`](@ref) over an explicit outer-index list, materializing the contiguous
component vectors and scratch buffers this worker needs. Shared by the distributed, MPI and
hybrid drivers, whose inputs arrive as strided views.
"""
function _pf_simd_partial!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, ::Val{D}, ilist,
) where {OT, CT, D}
    xc = ntuple(d -> collect(x_vecs[d]), Val(D))   # contiguous component vectors
    uc = ntuple(d -> collect(u_vecs[d]), Val(D))
    N = length(xc[1])
    plan = squared_digitize_plan(dist_be)
    nb = n_histogram_bins(plan)
    r2buf = Vector{eltype(xc[1])}(undef, N)
    valbuf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    _pf_simd_pairs!(sums, counts, sf, xc, uc, plan, Val(D), r2buf, valbuf, idxbuf, ilist)
    return nothing
end

"""
    _balanced_index_chunks(N, k) -> Vector of k index-lists

Split `1:N` into `k` balanced outer-index lists for the triangular pair loop (work ∝ N-i).
Round-robin assignment (`i ≡ w (mod k)`) gives each chunk a mix of cheap/expensive indices.
"""
function _balanced_index_chunks(N::Integer, k::Integer)
    k = max(1, k)
    # Ranges, not materialized vectors: `_partial_sums_counts` only iterates them, and the
    # distributed driver serializes one per worker.
    return [w:k:N for w in 1:k]
end
