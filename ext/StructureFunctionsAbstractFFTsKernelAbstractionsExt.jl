module StructureFunctionsAbstractFFTsKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @localmem, @synchronize
using AbstractFFTs: AbstractFFTs 
using LinearAlgebra: LinearAlgebra as LA 
using StaticArrays: StaticArrays as SA
using SpectralBackends: SpectralBackends as SB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, HelperFunctions as SFH


"""Bytes of inverse-transform scratch one batch of slab pairs may hold on the device."""
const DEVICE_BATCH_BYTES = 1 << 30

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

# Column `c` of slab pair `base + b`: the signed products of the two slabs' forward transforms its
# terms name, written to the batch-local column `b`.
KA.@kernel unsafe_indices = true function _spec_kernel!(
    specf::AbstractArray{CT}, @Const(F), @Const(terms), @Const(col_ptr), @Const(pairs), base::Int, n::Int,
    ncols::Int, total::Int, workgroup_size::Int,
) where {CT}
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    gid = (bid - 1) * workgroup_size + lid
    if gid <= total
        b = (gid - 1) ÷ (n * ncols) + 1
        rem = (gid - 1) - (b - 1) * n * ncols
        c = rem ÷ n + 1
        lin = rem - (c - 1) * n + 1
        @inbounds begin
            I = Int(pairs[base + b][1])
            J = Int(pairs[base + b][2])
            acc = zero(CT)
            for t in Int(col_ptr[c]):(Int(col_ptr[c + 1]) - 1)
                sign, ki, kj = terms[t]
                acc += sign * conj(F[lin, I, Int(ki)]) * F[lin, J, Int(kj)]
            end
            specf[lin, c, b] = acc
        end
    end
end

# One work item per (lag, slab pair) over the pairs `first_pair:last_pair`: the lag's moment row is
# read from the inverted columns, the operator contracted in the lag's frames and the value binned. A
# distance histogram accumulates in shared memory when it fits and is flushed once per block; the
# joint histogram uses global atomics. Under `UB` every pair carries the same lag box and a work item
# names its pair by division; otherwise the boxes are per pair and `pair_off` indexes them.
KA.@kernel unsafe_indices = true function _lag_kernel!(
    sums::AbstractArray{OT}, counts::AbstractArray{CT}, @Const(out), @Const(pairs), sf, s, su, plan, second_axis,
    axis_edges, @Const(pair_off), @Const(pair_lo), @Const(pair_len), @Const(pair_str), first_pair::Int,
    last_pair::Int, n_box::Int, box_lo, box_len, box_strides,
    P, strides, masked::Bool, weighted::Val{WEIGHTED}, ncols::Int,
    nb::Int, na::Int, total::Int, workgroup_size::Int, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N},
    ::Val{Dg}, ::Val{UB}, ::Val{NC}, ::Val{SHARED},
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
        if UB
            g = first_pair + (gid - 1) ÷ n_box
            lin = gid - (g - first_pair) * n_box
            h = _decode_lag(lin, box_lo, box_len, box_strides)
        else
            item = gid + @inbounds(pair_off[first_pair])
            g = _pair_of_item(pair_off, item, first_pair, last_pair)
            lin = item - @inbounds(pair_off[g])
            blo = @inbounds pair_lo[g]
            blen = @inbounds pair_len[g]
            bstr = @inbounds pair_str[g]
            h = _decode_lag(lin, ntuple(d -> Int(blo[d]), Val(Dg)), ntuple(d -> Int(blen[d]), Val(Dg)),
                            ntuple(d -> Int(bstr[d]), Val(Dg)))
        end
        b = g - first_pair + 1
        I = Int(@inbounds pairs[g][1])
        J = Int(@inbounds pairs[g][2])
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        if v !== nothing
            bin, r2, factor, geometry, self_reverse = v
            idx = SFC._lag_index(h, P, strides)
            outb = view(out, :, :, b)
            T = eltype(su.spacing)
            n_pairs = SFC._named_pairs(weighted, masked, outb, idx, ncols, su, h, self_reverse)
            scale = self_reverse ? T(0.5) : one(T)
            inv_r = inv(sqrt(r2))
            tr = SFC.lag_transport(s)
            if second_axis === nothing
                val = SFC.with_frames(tr, geometry) do frames
                    SFC._lag_value(tr, sf, outb, idx, scale, frames, inv_r, Val(W), Val(Po), Val(N), Val(V), Val(K))
                end
                if SHARED
                    @atomic shared_s[bin] += OT(factor * val)
                    @atomic shared_c[bin] += CT(n_pairs)
                else
                    @atomic sums[bin] += OT(factor * val)
                    @atomic counts[bin] += CT(n_pairs)
                end
            else
                Mo = SFC._lag_moments(tr, outb, idx, scale, nothing, nothing, Val(W), Val(Po), Val(N))
                SFC.with_frames(tr, geometry) do frames
                    Mi = length(frames)
                    for f in frames
                        bθ = SFH.digitize(SFC.axis_quantity(second_axis, f.dir, r2), axis_edges)
                        if 1 <= bθ <= na
                            cell = bin + (bθ - 1) * nb
                            value = OT(factor * SFT.moment_contract(sf, Mo, f.dir * inv_r, Val(V), Val(K)) / Mi)
                            @atomic sums[cell] += value
                            @atomic counts[cell] += CT(n_pairs / Mi)
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

# Inverse scratch for `Bb` slab pairs and the plan that fills it, in the transforms' array family.
function _batch_buffers(F1, P, ncols::Int, Bb::Int)
    CF = eltype(F1)
    spec = fill!(similar(F1, size(F1)..., ncols * Bb), zero(CF))
    out = fill!(similar(F1, real(CF), P..., ncols * Bb), zero(real(CF)))
    return (spec = spec, out = out, iplan = AbstractFFTs.plan_irfft(spec, P[1], 1:length(P)))
end

function SFC.device_transform_sweep!(
    sums::AbstractArray{OT}, counts::AbstractArray{CT}, backend::SFC.CB.AbstractGPUBackend, sf, data, s, dist_be, plan,
    nb::Int, ::Val{D}, ::Val{V}, ::Val{K}, valid, weights, tag, axis,
) where {OT, CT, D, V, K}
    dev = backend.backend
    to = x -> KA.adapt(dev, x)
    eng = SFC.transform_engine(sf, data, s, dist_be, Val(D), Val(V), Val(K), valid, weights, tag; to)
    su, P, r_max = eng.su, eng.P, eng.r_max
    Dg = length(P)
    F1 = eng.fwd[1][1]
    CF = eltype(F1)
    FT = real(CF)
    nlin = length(F1)
    nkeys = length(eng.fwd[1])
    nslabs = length(eng.fwd)
    # The forward stage holds every spectrum in one array, slab-fastest within a monomial, which is the
    # layout this kernel reads, so it is reshaped. Separate transforms from a provider are gathered.
    F = if eng.flat === nothing
        G = similar(F1, nlin, nslabs, nkeys)
        for I in 1:nslabs, k in 1:nkeys
            view(G, :, I, k) .= vec(eng.fwd[I][k])
        end
        G
    else
        reshape(eng.flat, nlin, nslabs, nkeys)
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
    cells = nb * na
    shared = axis === nothing && cells * (sizeof(OT) + sizeof(CT)) <= SFC.GPU_SMEM_STATIC_MAX ÷ 2
    NC = shared ? cells : 1
    dsums = KA.zeros(dev, OT, size(sums)...)
    dcounts = KA.zeros(dev, CT, size(counts)...)
    per_pair = nlin * ncols * sizeof(CF) + Pn * ncols * sizeof(FT)
    # Equal batches no larger than the budget, sharing one buffer set: both kernels are launched over the
    # pairs their batch holds, so a short last batch leaves the trailing columns untouched.
    n_batches = cld(n_items, clamp(DEVICE_BATCH_BYTES ÷ per_pair, 1, n_items))
    B = cld(n_items, n_batches)
    spec_k = _spec_kernel!(dev, WORKGROUP)
    lag_k = _lag_kernel!(dev, WORKGROUP)
    spec, out, iplan = _batch_buffers(F1, P, ncols, B)
    specf = reshape(spec, nlin, ncols, B)
    out3 = reshape(out, Pn, ncols, B)
    d_pairs = to(pairs)
    for lo in 1:B:n_items
        hi = min(lo + B - 1, n_items)
        Bb = hi - lo + 1
        total_spec = nlin * ncols * Bb
        spec_k(specf, F, d_terms, d_ptr, d_pairs, lo - 1, nlin, ncols, total_spec, WORKGROUP;
               ndrange = cld(total_spec, WORKGROUP) * WORKGROUP)
        KA.synchronize(dev)
        LA.mul!(out, iplan, spec)
        total_lag = UB ? n_box * Bb : pair_off[hi + 1] - pair_off[lo]
        lag_k(dsums, dcounts, out3, d_pairs, sf, s_dev, su, plan_dev, second_axis, d_axis_edges,
              d_off, d_plo, d_plen, d_pstr, lo, hi, n_box, box_lo, box_len, box_strides,
              P, strides, eng.masked, eng.weighted, ncols, nb, na, total_lag, WORKGROUP, Val(D),
              Val(V), Val(K), eng.vW, eng.vP, eng.vN, Val(Dg), Val(UB), Val(NC), Val(shared);
              ndrange = cld(total_lag, WORKGROUP) * WORKGROUP)
        KA.synchronize(dev)
    end
    sums .+= Array(dsums)
    counts .+= Array(dcounts)
    return nothing
end

end # module
