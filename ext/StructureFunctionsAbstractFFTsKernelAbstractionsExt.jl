module StructureFunctionsAbstractFFTsKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @localmem, @synchronize
using AbstractFFTs: AbstractFFTs 
using LinearAlgebra: LinearAlgebra as LA 
using StaticArrays: StaticArrays as SA
using SpectralBackends: SpectralBackends as SB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, HelperFunctions as SFH


SFC.gpu_free_memory(::KA.CPU) = Int(Sys.free_memory())

const WORKGROUP = 256

# Column `c` of slab pair `base + b` of slice `t`: the signed products of the two slabs' forward
# transforms its terms name, written to the batch-local column `b` of that slice. `F` holds the spectra
# `(blk, nchunks, ngroups, nt)`; `slabs[g]` is `(offset, chunk, monomial stride)` of each slab of pair `g`
# within its block, and a term is `(sign, group, place in group)` of each of its two monomials.
KA.@kernel unsafe_indices = true function _spec_kernel!(
    specf::AbstractArray{CT, 4}, @Const(F), @Const(terms), @Const(col_ptr), @Const(slabs), base::Int, n::Int,
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
            oI, cI, sI, oJ, cJ, sJ = slabs[base + b]
            iI, iJ = Int(oI) + lin, Int(oJ) + lin
            acc = zero(CT)
            for k in Int(col_ptr[c]):(Int(col_ptr[c + 1]) - 1)
                sign, gi, qi, gj, qj = terms[k]
                acc += sign * conj(F[iI + Int(qi) * Int(sI), Int(cI), Int(gi), t]) *
                       F[iJ + Int(qj) * Int(sJ), Int(cJ), Int(gj), t]
            end
            specf[lin, c, b, t] = acc
        end
    end
end

# One work item per (lag, slab pair) over the pairs `first_pair:last_pair`, with the `nt` slices as
# its innermost loop: the lag's geometry is read once and every slice's moment row is contracted
# against it (one operator, or the six single-pass invariants), or for a tensor binned as its packed
# components. A distance histogram accumulates in shared memory when it fits and is flushed once per
# block; the joint histogram uses global atomics. Histogram cells are indexed linearly, so one lag's
# slices are `nb · na` apart and its angles `nb` apart within a slice; a cell's several entries are
# adjacent.
KA.@kernel unsafe_indices = true function _lag_kernel!(
    sums::AbstractArray{OT}, counts::AbstractArray{CT}, @Const(out), @Const(pairs), sf, s, su, plan, second_axis,
    axis_edges, boxes, first_pair::Int, last_pair::Int,
    P, strides, masked::Bool, weighted::Val{WEIGHTED}, ncols::Int,
    nb::Int, na::Int, nt::Int, total::Int, workgroup_size::Int, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W},
    ::Val{Po}, ::Val{N}, ::Val{Dg}, ::Val{UB}, ::Val{NS}, ::Val{NC}, ::Val{SHARED},
) where {OT, CT, WEIGHTED, D, V, K, W, Po, N, Dg, UB, NS, NC, SHARED}
    shared_s = @localmem OT (NS,)
    shared_c = @localmem CT (NC,)
    lid = @index(Local, Linear)
    if SHARED
        k = lid
        while k <= NS
            @inbounds shared_s[k] = zero(OT)
            k += workgroup_size
        end
        k = lid
        while k <= NC
            @inbounds shared_c[k] = zero(CT)
            k += workgroup_size
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    gid = (bid - 1) * workgroup_size + lid
    if gid <= total
        g, h = SFC._item_pair_lag(Val(UB), Val(Dg), gid, first_pair, last_pair, boxes)
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
            # Names the slice loops write are assigned only inside these closures, keeping each closure-local.
            if second_axis === nothing
                SFC.with_frames(@inline(frames -> begin
                    for t in 1:nt
                        outb = view(out, :, :, b, t)
                        n_pairs = SFC._named_pairs(weighted, masked, outb, idx, ncols, su, h, self_reverse)
                        cell = bin + (t - 1) * nb
                        entry = _lag_entry(tr, sf, outb, idx, scale, frames, inv_r, factor, Val(W), Val(Po), Val(N),
                                           Val(V), Val(K))
                        if SHARED
                            _cell_add!(shared_s, entry, cell)
                            _cell_add!(shared_c, _count_entry(sf, CT(n_pairs)), cell)
                        else
                            _cell_add!(sums, entry, cell)
                            _cell_add!(counts, _count_entry(sf, CT(n_pairs)), cell)
                        end
                    end
                end), tr, geometry)
            else
                SFC.with_frames(@inline(frames -> begin
                    Mi = length(frames)
                    for t in 1:nt
                        outb = view(out, :, :, b, t)
                        n_pairs = SFC._named_pairs(weighted, masked, outb, idx, ncols, su, h, self_reverse)
                        Mo = SFC._lag_moments(tr, outb, idx, scale, nothing, nothing, Val(W), Val(Po), Val(N))
                        for f in frames
                            bθ = SFH.digitize(SFC.axis_quantity(second_axis, f.dir, r2), axis_edges)
                            if 1 <= bθ <= na
                                joint_cell = bin + (bθ - 1) * nb + (t - 1) * nb * na
                                _cell_add!(sums, _frame_entry(sf, Mo, f.dir, inv_r, factor, Mi, Val(V), Val(K)),
                                           joint_cell)
                                @atomic counts[joint_cell] += CT(n_pairs / Mi)
                            end
                        end
                    end
                end), tr, geometry)
            end
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    if SHARED
        k = lid
        while k <= NS
            sv = @inbounds shared_s[k]
            iszero(sv) || @atomic sums[k] += sv
            k += workgroup_size
        end
        k = lid
        while k <= NC
            cv = @inbounds shared_c[k]
            iszero(cv) || @atomic counts[k] += cv
            k += workgroup_size
        end
    end
end

# What a lag adds to its histogram cell: the operator's value, the six single-pass invariants, or a
# tensor's packed components.
@inline _lag_entry(tr, sf, out, idx, scale, frames, inv_r, factor, vW, vPo, vN, vV, vK) =
    factor * SFC._lag_value(tr, sf, out, idx, scale, frames, inv_r, vW, vPo, vN, vV, vK)
@inline _lag_entry(tr, ::SFT.MomentTensorOperator, out, idx, scale, frames, inv_r, factor, vW, vPo, vN, vV, vK) =
    SFC._tensor_factor(tr, factor) * SFC._frames_tensor(tr, out, idx, scale, frames, vW, vPo, vN)

# One of `Mi` frames' share of a lag, as `_lag_entry` gives a lag.
@inline _frame_entry(sf, Mo, dir, inv_r, factor, Mi, vV, vK) =
    factor * SFT.moment_contract(sf, Mo, dir * inv_r, vV, vK) / Mi
@inline _frame_entry(::SFT.MomentTensorOperator, Mo, dir, inv_r, factor, Mi, vV, vK) = factor * Mo.data / Mi

# A lag's pair count, in a row per single-pass invariant.
@inline _count_entry(sf, n) = n
@inline _count_entry(::SFT.SinglePassInvariants, n) = SA.SVector(ntuple(_ -> n, Val(SFC.SINGLE_PASS_N)))

# Add `v` to cell `cell` of a histogram, a vector's entries at `(q, cell)` of its `(Q, cells)` layout.
@inline _cell_add!(S, v::Number, cell::Int) = (@atomic S[cell] += eltype(S)(v); nothing)
@inline function _cell_add!(S, v::SA.SVector{Q}, cell::Int) where {Q}
    base = (cell - 1) * Q
    for q in 1:Q
        @atomic S[base + q] += eltype(S)(v[q])
    end
    return nothing
end

"""Sums and counts per histogram cell: one value, a rank-`P` tensor's packed components (one count), or
the six single-pass invariants (a count each)."""
@inline _cell_entries(sf, ::Val{D}) where {D} = (1, 1)
@inline _cell_entries(::SFT.MomentTensorOperator{P}, ::Val{D}) where {P, D} = (length(SFT.symmetric_indices(Val(D), Val(P))), 1)
@inline _cell_entries(::SFT.SinglePassInvariants, ::Val{D}) where {D} = (SFC.SINGLE_PASS_N, SFC.SINGLE_PASS_N)

"""Static shared bytes of `_lag_kernel!` with `NS` sums `OT` and `NC` counts `CT`."""
@inline _lag_hist_smem_bytes(::Type{OT}, ::Type{CT}, NS::Int, NC::Int) where {OT, CT} =
    SFC.gpu_localmem_bytes(OT, NS) + SFC.gpu_localmem_bytes(CT, NC)

# Every column's terms as one flat table with per-column offsets, each monomial as its group and place in
# spectra of layout `lay`.
function _term_table(columns, lay)
    terms = NTuple{5, Int32}[]
    col_ptr = Int32[1]
    for col in columns
        for (sign, ki, kj) in col
            push!(terms, Int32.((sign, SFC._key_group(lay, ki)..., SFC._key_group(lay, kj)...)))
        end
        push!(col_ptr, Int32(length(terms) + 1))
    end
    return terms, col_ptr
end

_axis_parts(::Nothing) = (nothing, nothing, 1)
_axis_parts(axis::Tuple) = (axis[1], axis[3], axis[2])

# Inverse scratch for `Bb` slab pairs of `nt` slices and the plan that fills it, in the transforms'
# array family.
function _batch_buffers(F1, P, ncols::Int, Bb::Int, nt::Int)
    CF = eltype(F1)
    spec = fill!(similar(F1, size(F1)..., ncols * Bb * nt), zero(CF))
    out = fill!(similar(F1, real(CF), P..., ncols * Bb * nt), zero(real(CF)))
    return (spec = spec, out = out,
            iplan = AbstractFFTs.plan_irfft(spec, P[1], 1:length(P); SFC._fft_plan_options(spec, 1)...))
end

# A non-uniform FFT provider's spectra of every slice, gathered into the kernel's layout at one slab and every
# monomial per block, with that layout.
function _gathered_spectra(engs, F1, nlin::Int, nslabs::Int, nkeys::Int, nt::Int)
    lay = (; L = nlin, chunk = 1, nchunks = nslabs, last_nb = 1, group = nkeys)
    F = similar(F1, nkeys * nlin, nslabs, 1, nt)
    for t in 1:nt, I in 1:nslabs, k in 1:nkeys
        view(F, (k - 1) * nlin .+ (1:nlin), I, 1, t) .= vec(engs[t].fwd[I][k])
    end
    return F, lay
end

# A single-slice sweep is the batch with one slice: the field and its validity gain a trailing
# singleton axis here, and the histogram cells are the same under linear indexing.
SFC.device_transform_sweep!(
    sums, counts, backend::SFC.CB.AbstractGPUBackend, sf, data::AbstractMatrix, s, dist_be, plan, nb::Int,
    vD::Val, vV::Val, vK::Val, valid, weights, tag, axis, workspace,
) = SFC.device_transform_sweep_batch!(sums, counts, backend, sf, SFC._one_slice(data), s, dist_be, plan, nb,
                                      vD, vV, vK, SFC._one_slice_valid(valid), weights, tag, axis, workspace)

function SFC.device_transform_sweep_batch!(
    sums::AbstractArray{OT}, counts::AbstractArray{CT}, backend::SFC.CB.AbstractGPUBackend, sf,
    data::AbstractArray{<:Any, 3}, s, dist_be, plan, nb::Int, ::Val{D}, ::Val{V}, ::Val{K}, valid, weights, tag,
    axis, workspace,
) where {OT, CT, D, V, K}
    dev = backend.backend
    SFC._require_device_outputs(backend, sums, counts)
    to = x -> KA.adapt(dev, x)
    nt = size(data, 3)
    # Every slice's FFT spectra are one array, the spectral kernel's input as it stands.
    engs = [SFC.transform_engine(sf, view(data, :, :, t), s, dist_be, Val(D), Val(V), Val(K),
                                 SFC._valid_slice(valid, t), weights, tag; to, workspace, slice = (t, nt))
            for t in 1:nt]
    eng = engs[1]
    su, P, r_max = eng.su, eng.P, eng.r_max
    Dg = length(P)
    F1 = eng.fwd[1][1]
    CF = eltype(F1)
    FT = real(CF)
    nlin = length(F1)
    nkeys = length(eng.fwd[1])
    nslabs = length(eng.fwd)
    F, lay = eng.spectra === nothing ? _gathered_spectra(engs, F1, nlin, nslabs, nkeys, nt) : (eng.spectra, eng.layout)
    terms, col_ptr = _term_table(eng.columns, lay)
    d_terms, d_ptr = to(terms), to(col_ptr)
    ncols = length(eng.columns)
    items = SFC.sweep_items(s, r_max, 1, false)
    pairs = [(Int32(it[1]), Int32(it[2])) for it in items]
    n_items = length(pairs)
    UB, boxes, host_off = SFC._lag_boxes(s, su, items, r_max, to)
    strides = SFC._lag_strides(P)
    Pn = prod(P)
    axis_edges, second_axis, na = _axis_parts(axis)
    d_axis_edges, d_second_axis = to(axis_edges), to(second_axis)
    cells = nb * na * nt
    per_sum, per_count = _cell_entries(sf, Val(D))
    shared = axis === nothing &&
        SFC.gpu_static_smem_fits(SFC.gpu_device_caps(dev), _lag_hist_smem_bytes(OT, CT, per_sum * cells, per_count * cells))
    NS, NC = shared ? (per_sum * cells, per_count * cells) : (1, 1)
    per_pair = nt * (nlin * ncols * sizeof(CF) + Pn * ncols * sizeof(FT))
    # Equal batches sharing one buffer set: as many pairs as fit in half the device's free memory (the inverse
    # transform holds working memory beside the buffers), at most `n_items`. A workspace-kept set keeps its width.
    sizes = (:device_batch, typeof(F1), size(F1), P, ncols, n_items, nt)
    set = SFC._borrow!(workspace, sizes, () -> begin
        width = cld(n_items, cld(n_items, clamp(SFC.gpu_free_memory(dev) ÷ (2 * per_pair), 1, n_items)))
        (B = width, buffers = _batch_buffers(F1, P, ncols, width, nt))
    end)
    try
        _device_pairs!(sums, counts, set.B, set.buffers, F, lay, eng, sf, s, su, plan, nb, na, nt, n_items, pairs,
                       boxes, host_off, Val(UB), P, strides, ncols, nlin, d_terms, d_ptr, d_second_axis, d_axis_edges,
                       dev, to, Val(D), Val(V), Val(K), Val(Dg), Val(NS), Val(NC), Val(shared))
    finally
        SFC._give_back!(workspace, sizes, set)
    end
    return nothing
end

function _device_pairs!(
    sums, counts, B, buffers, F, lay, eng, sf, s, su, plan, nb, na, nt, n_items, pairs, boxes, host_off, ::Val{UB}, P,
    strides, ncols, nlin, d_terms, d_ptr, d_second_axis, d_axis_edges, dev, to, ::Val{D}, ::Val{V}, ::Val{K},
    ::Val{Dg}, ::Val{NS}, ::Val{NC}, ::Val{SHARED},
) where {UB, D, V, K, Dg, NS, NC, SHARED}
    spec, out, iplan = buffers
    Pn = prod(P)
    specf = reshape(spec, nlin, ncols, B, nt)
    out4 = reshape(out, Pn, ncols, B, nt)
    spec_k = _spec_kernel!(dev, WORKGROUP)
    lag_k = _lag_kernel!(dev, WORKGROUP)
    d_pairs = to(pairs)
    d_slabs = to([Int32.((SFC._slab_offsets(lay, Int(I))..., SFC._slab_offsets(lay, Int(J))...)) for (I, J) in pairs])
    s_dev, plan_dev = to(s), to(plan)
    for lo in 1:B:n_items
        hi = min(lo + B - 1, n_items)
        Bb = hi - lo + 1
        total_spec = nlin * ncols * Bb * nt
        spec_k(specf, F, d_terms, d_ptr, d_slabs, lo - 1, nlin, ncols, Bb, total_spec, WORKGROUP;
               ndrange = cld(total_spec, WORKGROUP) * WORKGROUP)
        LA.mul!(out, iplan, spec)
        total_lag = SFC._lag_items(boxes.n_box, host_off, lo, hi)
        lag_k(sums, counts, out4, d_pairs, sf, s_dev, su, plan_dev, d_second_axis, d_axis_edges, boxes, lo, hi,
              P, strides, eng.masked, eng.weighted, ncols, nb, na, nt, total_lag, WORKGROUP,
              Val(D), Val(V), Val(K), eng.vW, eng.vP, eng.vN, Val(Dg), Val(UB), Val(NS), Val(NC), Val(SHARED);
              ndrange = cld(total_lag, WORKGROUP) * WORKGROUP)
    end
    # The inverse plan is released on return when the call keeps no workspace.
    KA.synchronize(dev)
    return nothing
end

end # module
