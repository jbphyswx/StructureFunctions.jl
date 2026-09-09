module StructureFunctionsFFTKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @localmem, @synchronize
using AbstractFFTs: plan_irfft
using LinearAlgebra: mul!
using StaticArrays: StaticArrays as SA
using SpectralBackends: SpectralBackends as SB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH

const CB = SFC.CB
const Adapt = KA.Adapt

# Schedules, digitize plans and axis edges reach the kernel with their vectors on the device.
Adapt.adapt_structure(to, s::SFC.RectilinearLagSchedule) =
    SFC.RectilinearLagSchedule(s.uniform, map(v -> Adapt.adapt(to, v), s.enumerated), s.axis_order)
Adapt.adapt_structure(to, s::SFC.ZonalLagSchedule) =
    SFC.ZonalLagSchedule(Adapt.adapt(to, s.lats), s.n_lon, s.dlon, s.radius, s.lon_periodic)
function Adapt.adapt_structure(to, p::SF.SquaredLogPlan{T}) where {T}
    sq = Adapt.adapt(to, p.sqedges)
    return SF.SquaredLogPlan{T, typeof(sq)}(p.a, p.b, p.n_bins, sq)
end
function Adapt.adapt_structure(to, p::SF.SquaredLinearPlan{T}) where {T}
    sq = Adapt.adapt(to, p.sqedges)
    return SF.SquaredLinearPlan{T, typeof(p.edges), typeof(sq)}(p.edges, p.n_bins, sq)
end
function Adapt.adapt_structure(to, p::SF.SquaredGeneralPlan{T}) where {T}
    sq = Adapt.adapt(to, p.sqedges)
    return SF.SquaredGeneralPlan{T, typeof(sq)}(p.n_bins, sq)
end
Adapt.adapt_structure(to, p::SF.SquaredInfPaddedPlan) = SF.SquaredInfPaddedPlan(Adapt.adapt(to, p.inner))
Adapt.adapt_structure(to, b::SF.BinEdges) = SF.BinEdges(Adapt.adapt(to, b.edges))
Adapt.adapt_structure(to, b::SF.InfPaddedBinEdges) = SF.InfPaddedBinEdges(Adapt.adapt(to, b.edges))

"""Bytes of inverse-transform scratch one batch of slab pairs may hold on the device."""
const DEVICE_BATCH_BYTES = 1 << 30

const WORKGROUP = 256

# The lag `lin` of the box `lo .+ (0:len-1)`, column-major with the given strides.
@inline _decode_lag(lin::Int, lo::NTuple{Dg, Int}, len::NTuple{Dg, Int}, strides::NTuple{Dg, Int}) where {Dg} =
    ntuple(d -> @inbounds(lo[d] + ((lin - 1) ÷ strides[d]) % len[d]), Val(Dg))

# Column `c` of slab pair `b`: the signed products of the two slabs' forward transforms its terms name.
KA.@kernel unsafe_indices = true function _spec_kernel!(
    specf::AbstractArray{CT}, @Const(F), @Const(terms), @Const(col_ptr), @Const(pairs), n::Int, ncols::Int,
    total::Int, workgroup_size::Int,
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
            I = Int(pairs[b][1])
            J = Int(pairs[b][2])
            acc = zero(CT)
            for t in Int(col_ptr[c]):(Int(col_ptr[c + 1]) - 1)
                sign, ki, kj = terms[t]
                acc += sign * conj(F[lin, Int(ki), I]) * F[lin, Int(kj), J]
            end
            specf[lin, c, b] = acc
        end
    end
end

# One work item per (lag, slab pair): the lag's moment row is read from the inverted columns, the
# operator contracted in the lag's frames and the value binned. A distance histogram accumulates in
# shared memory when it fits and is flushed once per block; the joint histogram uses global atomics.
KA.@kernel unsafe_indices = true function _lag_kernel!(
    sums::AbstractArray{OT}, counts::AbstractArray{CT}, @Const(out), @Const(pairs), sf, s, su, plan, second_axis,
    axis_edges, n_box::Int, box_lo, box_len, box_strides, P, strides, masked::Bool, ncols::Int, nb::Int, na::Int, total::Int,
    workgroup_size::Int, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N}, ::Val{NC}, ::Val{SHARED},
) where {OT, CT, D, V, K, W, Po, N, NC, SHARED}
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
        b = (gid - 1) ÷ n_box + 1
        lin = gid - (b - 1) * n_box
        h = _decode_lag(lin, box_lo, box_len, box_strides)
        I = Int(@inbounds pairs[b][1])
        J = Int(@inbounds pairs[b][2])
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        if v !== nothing
            bin, r2, factor, geometry, self_reverse = v
            idx = SFC._lag_index(h, P, strides)
            outb = view(out, :, :, b)
            T = eltype(su.spacing)
            named = masked ? round(Int, @inbounds(outb[idx, ncols])) : SFC._lag_pair_count(su, h)
            n_pairs = self_reverse ? named ÷ 2 : named
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
    return (spec = spec, out = out, iplan = plan_irfft(spec, P[1], 1:length(P)))
end

function SFC.device_transform_sweep!(
    sums::AbstractArray{OT}, counts::AbstractArray{CT}, backend::CB.AbstractGPUBackend, sf, data, s, dist_be, plan,
    nb::Int, ::Val{D}, ::Val{V}, ::Val{K}, valid, axis,
) where {OT, CT, D, V, K}
    dev = backend.backend
    to = x -> KA.adapt(dev, x)
    eng = SFC.transform_engine(sf, data, s, dist_be, Val(D), Val(V), Val(K), valid; to)
    su, P, r_max = eng.su, eng.P, eng.r_max
    Dg = length(P)
    F1 = eng.fwd[1][1]
    CF = eltype(F1)
    FT = real(CF)
    nlin = length(F1)
    nkeys = length(eng.fwd[1])
    nslabs = length(eng.fwd)
    F = similar(F1, nlin, nkeys, nslabs)
    for I in 1:nslabs, k in 1:nkeys
        view(F, :, k, I) .= vec(eng.fwd[I][k])
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
    B = clamp(DEVICE_BATCH_BYTES ÷ per_pair, 1, n_items)
    spec_k = _spec_kernel!(dev, WORKGROUP)
    lag_k = _lag_kernel!(dev, WORKGROUP)
    main = _batch_buffers(F1, P, ncols, B)
    tail_size = n_items - (n_items ÷ B) * B
    tail = tail_size == 0 ? main : _batch_buffers(F1, P, ncols, tail_size)
    for lo in 1:B:n_items
        hi = min(lo + B - 1, n_items)
        Bb = hi - lo + 1
        spec, out, iplan = Bb == B ? main : tail
        d_pairs = to(pairs[lo:hi])
        specf = reshape(spec, nlin, ncols, Bb)
        total_spec = nlin * ncols * Bb
        spec_k(specf, F, d_terms, d_ptr, d_pairs, nlin, ncols, total_spec, WORKGROUP;
               ndrange = cld(total_spec, WORKGROUP) * WORKGROUP)
        KA.synchronize(dev)
        mul!(out, iplan, spec)
        out3 = reshape(out, Pn, ncols, Bb)
        total_lag = n_box * Bb
        lag_k(dsums, dcounts, out3, d_pairs, sf, s_dev, su, plan_dev, second_axis, d_axis_edges, n_box, box_lo,
              box_len, box_strides, P, strides, eng.masked, ncols, nb, na, total_lag, WORKGROUP, Val(D), Val(V), Val(K),
              eng.vW, eng.vP, eng.vN, Val(NC), Val(shared); ndrange = cld(total_lag, WORKGROUP) * WORKGROUP)
        KA.synchronize(dev)
    end
    sums .+= Array(dsums)
    counts .+= Array(dcounts)
    return nothing
end

end # module
