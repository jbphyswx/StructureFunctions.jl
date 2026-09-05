# Serial 1D CPU Reduction Kernels

function serial_calculate_structure_function!(
    output::AbstractVector{OT},
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector;
    geometry = SFH.FlatGeometry{length(u_vecs)}(),
    culling::CullingPolicy = AutoCulling(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, T1, T2}
    distance_bins = BinEdges(distance_bins)

    if verbose
        @info("calculating structure function (serial reduction)")
    end

    # Fast path: flat D ∈ (2,3) uses the SIMD compute/scatter-split kernel (vectorizes the per-pair
    # compute over j; only the histogram scatter is scalar). Curved geometries take the scalar
    # per-i kernel, which forms the frame through `pair_frame`.
    D = length(u_vecs)
    if geometry isa SFH.FlatGeometry && D == 2
        return _pf_simd_run!(output, counts, structure_function_type, x_vecs, u_vecs,
            distance_bins, Val(2); culling = culling)
    elseif geometry isa SFH.FlatGeometry && D == 3
        return _pf_simd_run!(output, counts, structure_function_type, x_vecs, u_vecs,
            distance_bins, Val(3); culling = culling)
    end
    _cull_reject_unsupported(culling, "the scalar per-point kernel that this geometry uses")

    PM.@showprogress enabled = show_progress for i in eachindex(x_vecs[1])
        calculate_structure_function_i!(
            output, counts, geometry, structure_function_type, i, x_vecs, u_vecs, distance_bins,
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
    x_vecs::Tuple, u_vecs::Tuple, dist_be, ::Val{D};
    culling::CullingPolicy = AutoCulling(),
) where {OT, CT, D}
    N = length(x_vecs[1])
    return _pf_simd_partial!(output, counts, sf, x_vecs, u_vecs, dist_be, Val(D), 1:(N - 1), culling)
end

"""
Points per `j` block in the CPU pair loop. Sized so one block's coordinates, fields and the three
per-`j` buffers stay resident in a core's private cache while every `i` sweeps it.
"""
const SF_CPU_PAIR_TILE = 65536

"""
    _pf_simd_pairs!(output, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, blocks)

Accumulate the pairs `(i, j>i)` covered by `blocks` into `output`/`counts`.

`blocks` yields `(i-block, j-block)` index ranges (see [`block_pairs`](@ref)); each is worked to
completion, so the `j` block stays cache-resident across its whole `i` sweep. Under multi-core load
that is what keeps the loop off the memory bus: with one block spanning the array, per-core
throughput falls 83% once the arrays exceed L2.

Uniqueness is `j > i`, so a block pair never needs to know whether it lies on the diagonal, and a
culled schedule enumerating only nearby cells is exact for the same reason.

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
    blocks,
) where {OT, CT, D}
    nb = n_histogram_bins(plan)
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
                r2buf[j] = digitize_key(plan, r2)
                valbuf[j] = SFT._sf_raw(sf, Uj - Ui, dx, r2)
                if has_vector_index(plan)      # constant-folded: depends only on the plan type
                    idxbuf[j] = squared_approx_index(plan, r2)
                end
            end
            for j in jlo:j_last
                b = squared_bin(plan, r2buf[j], idxbuf[j])
                if 1 <= b <= nb
                    output[b] += valbuf[j]
                    counts[b] += one(CT)
                end
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
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number}
    # `size(u_arr, 1)` is the velocity dimension here, before any conversion, so this is the one
    # place the geometry can be built. Everything downstream receives it.
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u_arr, 1)))
    xk, uk = SFH.prepare_pair_inputs(geom, x_arr, u_arr)
    x_tuple = _component_vector_views(xk, SFH.coordinate_width(geom))
    u_tuple = _component_vector_views(uk, SFH.field_width(geom))
    return serial_calculate_structure_function!(
        sums,
        counts,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins;
        geometry = geom,
        kwargs...,
    )
end

function calculate_structure_function_i!(
    output::AbstractVector{OT},
    counts::AbstractVector,
    geom,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector,
) where {OT, T1, T2}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins)

    vW = SFH.coordinate_width(geom)
    vF = SFH.field_width(geom)
    W = _val_int(vW)
    F = _val_int(vF)
    X1 = SA.SVector{W, FT1}(ntuple(k -> @inbounds(x_vecs[k][i]), vW))
    U1 = SA.SVector{F, FT2}(ntuple(k -> @inbounds(u_vecs[k][i]), vF))

    iter_inds = eachindex(x_vecs[1])
    # @inbounds: x_vecs[k] are strided views; the bounds checks on every component access
    # were a large per-pair overhead. U2 is built only for in-range pairs.
    @inbounds for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{W, FT1}(ntuple(k -> x_vecs[k][j], vW))

        ok, distance, frame = SFH.pair_frame(geom, X1, X2)
        bin = SFH.digitize(distance, distance_bins)
        if ok && 1 <= bin < N3
            U2 = SA.SVector{F, FT2}(ntuple(k -> u_vecs[k][j], vF))
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
    geometry = SFH.FlatGeometry{length(u_vecs)}(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    _assert_counts_representable(CT, length(x_vecs[1]))
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    N3 = n_histogram_bins(distance_bins)
    local_output = zeros(OT, N3)
    local_counts = zeros(CT, N3)
    calculate_structure_function_i!(
        local_output, local_counts, geometry,
        structure_function_type, i, x_vecs, u_vecs, BinEdges(distance_bins),
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
    geometry = SFH.FlatGeometry{length(u_vecs)}(),
    culling::CullingPolicy = AutoCulling(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    _assert_counts_representable(CT, length(x_vecs[1]))
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    nb = n_histogram_bins(distance_bins)
    sums = zeros(OT, nb)
    counts = zeros(CT, nb)
    be = BinEdges(distance_bins)
    D = length(u_vecs)
    # Flat D ∈ {2,3} takes the SIMD compute/scatter kernel, the same one the serial and threaded
    # drivers use; `_pf_simd_pairs!` accepts an arbitrary `irange`. Curved geometries take the
    # scalar per-`i` kernel.
    if geometry isa SFH.FlatGeometry && (D == 2 || D == 3)
        vD = D == 2 ? Val(2) : Val(3)
        _pf_simd_partial!(sums, counts, structure_function_type, x_vecs, u_vecs, be, vD, ilist, culling)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    _cull_reject_unsupported(culling, "the scalar per-point kernel that this geometry uses")

    for i in ilist
        calculate_structure_function_i!(
            sums, counts, geometry, structure_function_type, i, x_vecs, u_vecs, be,
        )
    end
    return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
end

"""
    _pf_run_blocks!(sums, counts, sf, xc, uc, plan, ::Val{D}, bufs..., ilist, N, grid)

Run the pair kernel with the schedule `grid` selects. Whether a grid exists is decided from the
data, so it arrives here as a `Union`; dispatching on it resolves that into one concretely typed
schedule per method, which is what keeps the kernel statically specialized.
"""
@inline _pf_run_blocks!(
    sums, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, ilist, N, ::Nothing,
) where {D} = _pf_simd_pairs!(sums, counts, sf, xc, uc, plan, Val(D), r2buf, valbuf, idxbuf,
    pair_blocks(N, ilist))

@inline _pf_run_blocks!(
    sums, counts, sf, xc, uc, plan, ::Val{D}, r2buf, valbuf, idxbuf, ilist, N, grid::CellGrid,
) where {D} = _pf_simd_pairs!(sums, counts, sf, xc, uc, plan, Val(D), r2buf, valbuf, idxbuf,
    pair_blocks(N, ilist; grid = grid))

"""
    _pf_simd_partial!(sums, counts, sf, x_vecs, u_vecs, dist_be, ::Val{D}, ilist; kwargs...)

Run [`_pf_simd_pairs!`](@ref) over an explicit outer-index list, materializing the contiguous
component vectors and scratch buffers this worker needs. Shared by the distributed, MPI and
hybrid drivers, whose inputs arrive as strided views.

When `culling` yields a cull grid the points are sorted into it first; `ilist` then selects
positions in the sorted order, which leaves the union over workers unchanged.
"""
function _pf_simd_partial!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, ::Val{D}, ilist, culling::CullingPolicy = AutoCulling();
    geometry = SFH.FlatGeometry{D}(),
) where {OT, CT, D}
    xc = ntuple(d -> collect(x_vecs[d]), Val(D))   # contiguous component vectors
    uc = ntuple(d -> collect(u_vecs[d]), Val(D))
    N = length(xc[1])
    plan = squared_digitize_plan(dist_be)
    r2buf = Vector{eltype(xc[1])}(undef, N)
    valbuf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    grid = (culling isa NoCulling) ? nothing : cull_grid_for(xc, geometry, dist_be, culling) # this is type unstable
    if !isnothing(grid)
        xc = apply_perm(xc, grid.perm)
        uc = apply_perm(uc, grid.perm)
    end
    _pf_run_blocks!(sums, counts, sf, xc, uc, plan, Val(D),
        r2buf, valbuf, idxbuf, ilist, N, grid)
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
