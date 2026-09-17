module StructureFunctionsAbstractFFTsKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @localmem, @synchronize
using AbstractFFTs: AbstractFFTs 
using LinearAlgebra: LinearAlgebra as LA 
using StaticArrays: StaticArrays as SA
using SpectralBackends: SpectralBackends as SB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, HelperFunctions as SFH


SFC.gpu_free_memory(::KA.CPU) = Int(Sys.free_memory())

const WORKGROUP = 256

# The lag `lin` of the box `lo .+ (0:len-1)`, column-major with the given strides.
@inline _decode_lag(lin::Int, lo::NTuple{Dg, Int}, len::NTuple{Dg, Int}, strides::NTuple{Dg, Int}) where {Dg} =
    ntuple(d -> @inbounds(lo[d] + ((lin - 1) ÷ strides[d]) % len[d]), Val(Dg))

# The slab pair in `lo:hi` owning work item `item`, from `off`, the exclusive prefix sum of every
# pair's lag-box volume: pair `b` owns `off[b] + 1 : off[b + 1]`.
@inline function _pair_of_item(off, item::Int, lo::Int, hi::Int)
    while lo < hi
        mid = (lo + hi + 1) >>> 1
        if @inbounds(off[mid]) < item
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

# The slab pair and the lag that work item `gid` owns, as one value so neither is a variable the lag
# kernel's frame closure captures and reassigns. Under `UB` every pair carries the same lag box and
# the pair follows by division; otherwise `pair_off` indexes the boxes.
@inline function _item_pair_lag(
    ::Val{UB}, ::Val{Dg}, gid::Int, first_pair::Int, last_pair::Int, n_box::Int, box_lo, box_len, box_strides,
    pair_off, pair_lo, pair_len, pair_str,
) where {UB, Dg}
    if UB
        g = first_pair + (gid - 1) ÷ n_box
        return g, _decode_lag(gid - (g - first_pair) * n_box, box_lo, box_len, box_strides)
    end
    item = gid + @inbounds(pair_off[first_pair])
    g = _pair_of_item(pair_off, item, first_pair, last_pair)
    blo = @inbounds pair_lo[g]
    blen = @inbounds pair_len[g]
    bstr = @inbounds pair_str[g]
    return g, _decode_lag(item - @inbounds(pair_off[g]), ntuple(d -> Int(blo[d]), Val(Dg)),
                          ntuple(d -> Int(blen[d]), Val(Dg)), ntuple(d -> Int(bstr[d]), Val(Dg)))
end

# Column `c` of slab pair `base + b` of slice `t`: the signed products of the two slabs' forward
# transforms its terms name, written to the batch-local column `b` of that slice.
KA.@kernel unsafe_indices = true function _spec_kernel!(
    specf::AbstractArray{CT, 4}, @Const(F), @Const(terms), @Const(col_ptr), @Const(pairs), base::Int, n::Int,
    ncols::Int, n_batch::Int, total::Int, workgroup_size::Int,
) where {CT}
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    gid = (bid - 1) * workgroup_size + lid
    if gid <= total
        q = gid - 1
        lin = q % n + 1
        q = q ÷ n
        c = q % ncols + 1
        q = q ÷ ncols
        b = q % n_batch + 1
        t = q ÷ n_batch + 1
        @inbounds begin
            I = Int(pairs[base + b][1])
            J = Int(pairs[base + b][2])
            acc = zero(CT)
            for k in Int(col_ptr[c]):(Int(col_ptr[c + 1]) - 1)
                sign, ki, kj = terms[k]
                acc += sign * conj(F[lin, I, Int(ki), t]) * F[lin, J, Int(kj), t]
            end
            specf[lin, c, b, t] = acc
        end
    end
end

# One work item per (lag, slab pair) over the pairs `first_pair:last_pair`, with the `nt` slices as
# its innermost loop: the lag's geometry is read once and every slice's moment row is contracted
# against it. A distance histogram accumulates in shared memory when it fits and is flushed once per
# block; the joint histogram uses global atomics. Histogram cells are indexed linearly, so one lag's
# slices are `nb · na` apart and its angles `nb` apart within a slice.
KA.@kernel unsafe_indices = true function _lag_kernel!(
    sums::AbstractArray{OT}, counts::AbstractArray{CT}, @Const(out), @Const(pairs), sf, s, su, plan, second_axis,
    axis_edges, @Const(pair_off), @Const(pair_lo), @Const(pair_len), @Const(pair_str), first_pair::Int,
    last_pair::Int, n_box::Int, box_lo, box_len, box_strides,
    P, strides, masked::Bool, weighted::Val{WEIGHTED}, ncols::Int,
    nb::Int, na::Int, nt::Int, total::Int, workgroup_size::Int, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W},
    ::Val{Po}, ::Val{N}, ::Val{Dg}, ::Val{UB}, ::Val{NC}, ::Val{SHARED},
) where {OT, CT, WEIGHTED, D, V, K, W, Po, N, Dg, UB, NC, SHARED}
    shared_s = @localmem OT (NC,)
    shared_c = @localmem CT (NC,)
    lid = @index(Local, Linear)
    if SHARED
        k = lid
        while k <= NC
            @inbounds shared_s[k] = zero(OT)
            @inbounds shared_c[k] = zero(CT)
            k += workgroup_size
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    gid = (bid - 1) * workgroup_size + lid
    if gid <= total
        g, h = _item_pair_lag(Val(UB), Val(Dg), gid, first_pair, last_pair, n_box, box_lo, box_len, box_strides,
                              pair_off, pair_lo, pair_len, pair_str)
        b = g - first_pair + 1
        I = Int(@inbounds pairs[g][1])
        J = Int(@inbounds pairs[g][2])
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        if v !== nothing
            bin, r2, factor, geometry, self_reverse = v
            idx = SFC._lag_index(h, P, strides)
            T = eltype(su.spacing)
            scale = self_reverse ? T(0.5) : one(T)
            inv_r = inv(sqrt(r2))
            tr = SFC.lag_transport(s)
            # Every name the slice loops write is assigned only inside these closures, so each is
            # local to its closure; one also assigned outside would be captured and reassigned.
            if second_axis === nothing
                SFC.with_frames(tr, geometry) do frames
                    for t in 1:nt
                        outb = view(out, :, :, b, t)
                        n_pairs = SFC._named_pairs(weighted, masked, outb, idx, ncols, su, h, self_reverse)
                        val = SFC._lag_value(tr, sf, outb, idx, scale, frames, inv_r, Val(W), Val(Po), Val(N),
                                             Val(V), Val(K))
                        cell = bin + (t - 1) * nb
                        if SHARED
                            @atomic shared_s[cell] += OT(factor * val)
                            @atomic shared_c[cell] += CT(n_pairs)
                        else
                            @atomic sums[cell] += OT(factor * val)
                            @atomic counts[cell] += CT(n_pairs)
                        end
                    end
                end
            else
                SFC.with_frames(tr, geometry) do frames
                    Mi = length(frames)
                    for t in 1:nt
                        outb = view(out, :, :, b, t)
                        n_pairs = SFC._named_pairs(weighted, masked, outb, idx, ncols, su, h, self_reverse)
                        Mo = SFC._lag_moments(tr, outb, idx, scale, nothing, nothing, Val(W), Val(Po), Val(N))
                        for f in frames
                            bθ = SFH.digitize(SFC.axis_quantity(second_axis, f.dir, r2), axis_edges)
                            if 1 <= bθ <= na
                                joint_cell = bin + (bθ - 1) * nb + (t - 1) * nb * na
                                value = OT(factor * SFT.moment_contract(sf, Mo, f.dir * inv_r, Val(V), Val(K)) / Mi)
                                @atomic sums[joint_cell] += value
                                @atomic counts[joint_cell] += CT(n_pairs / Mi)
                            end
                        end
                    end
                end
            end
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    if SHARED
        k = lid
        while k <= NC
            @inbounds begin
                sv = shared_s[k]
                cv = shared_c[k]
                if !iszero(cv) || !iszero(sv)
                    @atomic sums[k] += sv
                    @atomic counts[k] += cv
                end
            end
            k += workgroup_size
        end
    end
end

# Every column's terms as one flat table with per-column offsets.
function _term_table(columns)
    terms = NTuple{3, Int32}[]
    col_ptr = Int32[1]
    for col in columns
        for (sign, ki, kj) in col
            push!(terms, (Int32(sign), Int32(ki), Int32(kj)))
        end
        push!(col_ptr, Int32(length(terms) + 1))
    end
    return terms, col_ptr
end

_axis_parts(::Nothing) = (nothing, nothing, 1)
_axis_parts(axis::Tuple) = (axis[1], SFC.SeparationAngleAxis(SA.SVector{length(axis[3].reference_axis)}(axis[3].reference_axis)), axis[2])

# Inverse scratch for `Bb` slab pairs of `nt` slices and the plan that fills it, in the transforms'
# array family.
function _batch_buffers(F1, P, ncols::Int, Bb::Int, nt::Int)
    CF = eltype(F1)
    spec = fill!(similar(F1, size(F1)..., ncols * Bb * nt), zero(CF))
    out = fill!(similar(F1, real(CF), P..., ncols * Bb * nt), zero(real(CF)))
    return (spec = spec, out = out, iplan = AbstractFFTs.plan_irfft(spec, P[1], 1:length(P)))
end

# A single-slice sweep is the batch with one slice: the field and its validity gain a trailing
# singleton axis here, and the histogram cells are the same under linear indexing.
SFC.device_transform_sweep!(
    sums, counts, backend::SFC.CB.AbstractGPUBackend, sf, data::AbstractMatrix, s, dist_be, plan, nb::Int,
    vD::Val, vV::Val, vK::Val, valid, weights, tag, axis,
) = SFC.device_transform_sweep_batch!(sums, counts, backend, sf,
                                      reshape(data, size(data, 1), size(data, 2), 1), s, dist_be, plan, nb,
                                      vD, vV, vK, _one_slice_valid(valid), weights, tag, axis)

_one_slice_valid(::SFC.AllValid) = SFC.AllValid()
_one_slice_valid(v::AbstractVector) = reshape(v, length(v), 1)

function SFC.device_transform_sweep_batch!(
    sums::AbstractArray{OT}, counts::AbstractArray{CT}, backend::SFC.CB.AbstractGPUBackend, sf,
    data::AbstractArray{<:Any, 3}, s, dist_be, plan, nb::Int, ::Val{D}, ::Val{V}, ::Val{K}, valid, weights, tag,
    axis,
) where {OT, CT, D, V, K}
    dev = backend.backend
    to = x -> KA.adapt(dev, x)
    nt = size(data, 3)
    engs = [SFC.transform_engine(sf, view(data, :, :, t), s, dist_be, Val(D), Val(V), Val(K),
                                 SFC._valid_slice(valid, t), weights, tag; to) for t in 1:nt]
    eng = engs[1]
    su, P, r_max = eng.su, eng.P, eng.r_max
    Dg = length(P)
    F1 = eng.fwd[1][1]
    CF = eltype(F1)
    FT = real(CF)
    nlin = length(F1)
    nkeys = length(eng.fwd[1])
    nslabs = length(eng.fwd)
    # The kernel reads every slice's spectra from one array, slab-fastest within a monomial. The FFT
    # stage already writes a slab's spectra in that layout, so a slice is copied in one piece;
    # transforms from a non-uniform FFT provider come separately and are gathered.
    F = similar(F1, nlin, nslabs, nkeys, nt)
    for t in 1:nt
        eng_t = engs[t]
        if eng_t.flat === nothing
            for I in 1:nslabs, k in 1:nkeys
                view(F, :, I, k, t) .= vec(eng_t.fwd[I][k])
            end
        else
            view(F, :, :, :, t) .= reshape(eng_t.flat, nlin, nslabs, nkeys)
        end
    end
    terms, col_ptr = _term_table(eng.columns)
    d_terms, d_ptr = to(terms), to(col_ptr)
    ncols = length(eng.columns)
    items = SFC.sweep_items(s, r_max, 1, false)
    pairs = [(Int32(it[1]), Int32(it[2])) for it in items]
    n_items = length(pairs)
    lims = SFC.lag_limits(s, r_max)
    box_lo = ntuple(d -> first(SFC.lag_range(su, d, lims[d])), Dg)
    box_len = ntuple(d -> length(SFC.lag_range(su, d, lims[d])), Dg)
    n_box = prod(box_len)
    box_strides = SFC._lag_strides(box_len)
    # A schedule whose pairs do not share one lag box gets a box each, the same one the host loop
    # takes from `_pair_lags`: a parallel spans less distance the nearer it lies to a pole, so the box
    # over all row pairs stays as wide as the equator's however small `r_max` is. The tables are built
    # and moved once per call and the kernel finds a work item's pair in the prefix sum of the volumes.
    UB = SFC.uniform_lag_box(s)
    pair_off, d_off, d_plo, d_plen, d_pstr = if UB
        nothing, nothing, nothing, nothing, nothing
    else
        boxes = [SFC.lag_limits(s, it[1], it[2], r_max) for it in items]
        lens = NTuple{Dg, Int}[ntuple(d -> length(SFC.lag_range(su, d, L[d])), Dg) for L in boxes]
        los = NTuple{Dg, Int32}[ntuple(d -> Int32(first(SFC.lag_range(su, d, L[d]))), Dg) for L in boxes]
        offs = Vector{Int}(undef, n_items + 1)
        offs[1] = 0
        for k in 1:n_items
            offs[k + 1] = offs[k] + prod(lens[k])
        end
        offs, to(offs), to(los),
        to(NTuple{Dg, Int32}[ntuple(d -> Int32(l[d]), Dg) for l in lens]),
        to(NTuple{Dg, Int32}[ntuple(d -> Int32(st[d]), Dg) for st in map(SFC._lag_strides, lens)])
    end
    strides = SFC._lag_strides(P)
    Pn = prod(P)
    axis_edges, second_axis, na = _axis_parts(axis)
    if axis !== nothing && CT <: Integer &&
       any(d -> su.periodic[d] && iseven(su.dims[d]) && lims[d] >= su.dims[d] ÷ 2, 1:Dg)
        throw(ArgumentError(
            "a lag that half-turns a periodic direction splits each pair between its two directions, so a " *
            "joint histogram over angle needs a floating-point count type; got $CT",
        ))
    end
    d_axis_edges = axis_edges === nothing ? nothing : to(axis_edges)
    s_dev = to(s)
    plan_dev = to(plan)
    cells = nb * na * nt
    shared = axis === nothing && cells * (sizeof(OT) + sizeof(CT)) <= SFC.GPU_SMEM_STATIC_MAX ÷ 2
    NC = shared ? cells : 1
    dsums = KA.zeros(dev, OT, size(sums)...)
    dcounts = KA.zeros(dev, CT, size(counts)...)
    per_pair = nt * (nlin * ncols * sizeof(CF) + Pn * ncols * sizeof(FT))
    # Equal batches sharing one buffer set: as many pairs as the device has room for, and never more
    # than there are pairs. The room is half of what the device reports free, because the inverse
    # transform holds working memory beside the buffer set it reads and writes. Both kernels are
    # launched over the pairs their batch holds, so a short last batch leaves the trailing columns
    # untouched.
    width = SFC.gpu_free_memory(dev) ÷ (2 * per_pair)
    n_batches = cld(n_items, clamp(width, 1, n_items))
    B = cld(n_items, n_batches)
    spec_k = _spec_kernel!(dev, WORKGROUP)
    lag_k = _lag_kernel!(dev, WORKGROUP)
    spec, out, iplan = _batch_buffers(F1, P, ncols, B, nt)
    specf = reshape(spec, nlin, ncols, B, nt)
    out4 = reshape(out, Pn, ncols, B, nt)
    d_pairs = to(pairs)
    for lo in 1:B:n_items
        hi = min(lo + B - 1, n_items)
        Bb = hi - lo + 1
        total_spec = nlin * ncols * Bb * nt
        spec_k(specf, F, d_terms, d_ptr, d_pairs, lo - 1, nlin, ncols, Bb, total_spec, WORKGROUP;
               ndrange = cld(total_spec, WORKGROUP) * WORKGROUP)
        KA.synchronize(dev)
        LA.mul!(out, iplan, spec)
        total_lag = UB ? n_box * Bb : pair_off[hi + 1] - pair_off[lo]
        lag_k(dsums, dcounts, out4, d_pairs, sf, s_dev, su, plan_dev, second_axis, d_axis_edges,
              d_off, d_plo, d_plen, d_pstr, lo, hi, n_box, box_lo, box_len, box_strides,
              P, strides, eng.masked, eng.weighted, ncols, nb, na, nt, total_lag, WORKGROUP,
              Val(D), Val(V), Val(K), eng.vW, eng.vP, eng.vN, Val(Dg), Val(UB), Val(NC), Val(shared);
              ndrange = cld(total_lag, WORKGROUP) * WORKGROUP)
        KA.synchronize(dev)
    end
    sums .+= Array(dsums)
    counts .+= Array(dcounts)
    return nothing
end

end # module
