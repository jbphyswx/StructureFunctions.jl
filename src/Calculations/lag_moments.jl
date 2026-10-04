# The lag algebra of the transform engine: which monomials a moment reads, how the inverted columns
# combine under the frames, and the operator on one lag. Pure functions of static sizes, shared by the
# host loop and the device kernel.

"""A monomial key of any degree up to `Pm`, zero-padded."""
@inline function _padded_key(::Val{Pm}, key::Tuple) where {Pm}
    length(key) <= Pm || throw(ArgumentError("a monomial of degree $(length(key)) exceeds the cache's degree $Pm"))
    return ntuple(i -> i <= length(key) ? Int(key[i]) : 0, Val(Pm))
end

"""Pairs a lag names between two complete slabs: every cell on a wrapping direction, the overlap on a bounded one."""
@inline _lag_pair_count(s::UniformLagSchedule{Dg}, h::NTuple{Dg, Int}) where {Dg} =
    prod(ntuple(d -> @inbounds(s.periodic[d]) ? @inbounds(s.dims[d]) :
                     @inbounds(s.dims[d]) - abs(h[d]), Val(Dg)))

"""
Pairs a lag names between two slabs: the count column of the inverted transforms when the field is masked
or weighted (a weighted pair mass in the latter case), else the slab overlap; halved on a lag equal
to its own reverse.
"""
@inline function _named_pairs(::Val{false}, masked::Bool, out, idx::Int, ncol::Int, su::UniformLagSchedule, h,
                              self_reverse::Bool)
    n = masked ? round(Int, @inbounds(out[idx, ncol])) : _lag_pair_count(su, h)
    return self_reverse ? n ÷ 2 : n
end

@inline function _named_pairs(::Val{true}, ::Bool, out, idx::Int, ncol::Int, ::UniformLagSchedule, h,
                              self_reverse::Bool)
    n = @inbounds out[idx, ncol]
    return self_reverse ? n / 2 : n
end

"""Whether a sweep carries pair weights, as a `Val` the kernels specialise on."""
@inline _weighted_val(::NoWeights) = Val(false)
@inline _weighted_val(::AbstractVector) = Val(true)

"""Column-major strides of a transform of size `P`."""
@inline _lag_strides(P::NTuple{Dg, Int}) where {Dg} =
    ntuple(d -> prod(k -> P[k], 1:(d - 1); init = 1), Val(Dg))

"""The lag at position `lin` of a box starting at `lo`, of extent `len` and strides `strides`."""
@inline _decode_lag(lin::Int, lo::NTuple{Dg, Int}, len::NTuple{Dg, Int},
                    strides::NTuple{Dg, Int}) where {Dg} =
    ntuple(d -> @inbounds(lo[d] + ((lin - 1) ÷ strides[d]) % len[d]), Val(Dg))

"""The slab pair in `lo:hi` owning work item `item`, from `off`, the exclusive prefix sum of every pair's
lag-box volume: pair `b` owns `off[b] + 1 : off[b + 1]`."""
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

"""
    _item_pair_lag(Val(UB), Val(Dg), gid, first_pair, last_pair, boxes) -> (pair, lag)

The slab pair and the lag that work item `gid` of the pairs `first_pair:last_pair` owns. Under `UB` every
pair carries the one box `(n_box, lo, len, strides)` and the pair follows by division; under `!UB`
`boxes.off` indexes the per-pair boxes `boxes.plo`, `boxes.plen`, `boxes.pstr`.
"""
@inline function _item_pair_lag(::Val{UB}, ::Val{Dg}, gid::Int, first_pair::Int, last_pair::Int, boxes) where {UB, Dg}
    if UB
        g = first_pair + (gid - 1) ÷ boxes.n_box
        return g, _decode_lag(gid - (g - first_pair) * boxes.n_box, boxes.lo, boxes.len, boxes.strides)
    end
    item = gid + @inbounds(boxes.off[first_pair])
    g = _pair_of_item(boxes.off, item, first_pair, last_pair)
    blo, blen, bstr = @inbounds(boxes.plo[g]), @inbounds(boxes.plen[g]), @inbounds(boxes.pstr[g])
    return g, _decode_lag(item - @inbounds(boxes.off[g]), ntuple(d -> Int(blo[d]), Val(Dg)),
                          ntuple(d -> Int(blen[d]), Val(Dg)), ntuple(d -> Int(bstr[d]), Val(Dg)))
end

"""
    _lag_boxes(schedule, uniform, items, r_max, to) -> (UB, boxes, host_off)

The lag boxes a device launch indexes its work items by, for the slab pairs `items` names: one box for
every pair when [`uniform_lag_box`](@ref) holds, else each pair's own box from `lag_limits` with the
exclusive prefix sum of their volumes. `boxes` is what a kernel reads, its per-pair tables moved by
`to`; `host_off` is the prefix sum on the host, `nothing` under one box.
"""
function _lag_boxes(s::AbstractSeparableSchedule, su::UniformLagSchedule{Dg}, items, r_max, to) where {Dg}
    lims = lag_limits(s, r_max)
    lo = ntuple(d -> first(lag_range(su, d, lims[d])), Val(Dg))
    len = ntuple(d -> length(lag_range(su, d, lims[d])), Val(Dg))
    n_box = prod(len)
    if uniform_lag_box(s)
        return true, (n_box = n_box, lo = lo, len = len, strides = _lag_strides(len), off = nothing,
                      plo = nothing, plen = nothing, pstr = nothing), nothing
    end
    boxes = [lag_limits(s, it[1], it[2], r_max) for it in items]
    lens = NTuple{Dg, Int}[ntuple(d -> length(lag_range(su, d, L[d])), Val(Dg)) for L in boxes]
    los = NTuple{Dg, Int32}[ntuple(d -> Int32(first(lag_range(su, d, L[d]))), Val(Dg)) for L in boxes]
    offs = Vector{Int}(undef, length(items) + 1)
    offs[1] = 0
    for k in eachindex(lens)
        offs[k + 1] = offs[k] + prod(lens[k])
    end
    return false, (n_box = n_box, lo = lo, len = len, strides = _lag_strides(len), off = to(offs), plo = to(los),
                   plen = to(NTuple{Dg, Int32}[ntuple(d -> Int32(l[d]), Val(Dg)) for l in lens]),
                   pstr = to(NTuple{Dg, Int32}[ntuple(d -> Int32(st[d]), Val(Dg)) for st in map(_lag_strides, lens)])),
           offs
end

"""Work items of the pairs `lo:hi` under the boxes [`_lag_boxes`](@ref) built."""
@inline _lag_items(n_box::Int, ::Nothing, lo::Int, hi::Int) = n_box * (hi - lo + 1)
@inline _lag_items(::Int, host_off::AbstractVector, lo::Int, hi::Int) = host_off[hi + 1] - host_off[lo]

"""Slab `I`'s place in forward spectra of layout `lay`, whose blocks hold a chunk of `lay.chunk` slabs (the last
`lay.last_nb`, of `lay.nchunks`) for each of a group of monomials, slab fastest: its offset in its block, the block's
chunk, and the offset between consecutive monomials of the block."""
@inline function _slab_offsets(lay, I::Int)
    c, i = divrem(I - 1, lay.chunk)
    nb = c + 1 == lay.nchunks ? lay.last_nb : lay.chunk
    return i * lay.L, c + 1, nb * lay.L
end

"""Monomial `k`'s group of `lay.group` monomials in forward spectra of layout `lay`, and its place in the group."""
@inline function _key_group(lay, k::Int)
    g, q = divrem(k - 1, lay.group)
    return g + 1, q
end

"""Linear position of lag `h` in a transform of size `P`, wrapping the negative offsets."""
@inline function _lag_index(h::NTuple{Dg, Int}, P::NTuple{Dg, Int}, strides::NTuple{Dg, Int}) where {Dg}
    lin = 1
    @inbounds for d in 1:Dg
        lin += mod(h[d], P[d]) * strides[d]
    end
    return lin
end


"""Every sorted multi-index of degree `0:P` over `W` components, zero-padded to `P`: the monomials a degree-`P` moment reads."""
@generated function _monomial_keys(::Val{W}, ::Val{P}) where {W, P}
    keys = NTuple{P, Int}[_padded_key(Val(P), k) for d in 0:P for k in SFT.symmetric_indices(Val(W), Val(d))]
    return :(copy($keys))
end

# Raw cross-moments under frame transport: for k = 0..P, every sorted multi-index over the first slab
# of degree P − k with every sorted one over the second of degree k, in that order.
function _raw_columns(::Val{W}, ::Val{P}) where {W, P}
    cols = Tuple{Tuple{Vararg{Int}}, Tuple{Vararg{Int}}}[]
    for k in 0:P, cI in SFT.symmetric_indices(Val(W), Val(P - k)), cJ in SFT.symmetric_indices(Val(W), Val(k))
        push!(cols, (cI, cJ))
    end
    return cols
end

_n_raw(W::Int, P::Int) = sum(binomial(W + P - k - 1, P - k) * binomial(W + k - 1, k) for k in 0:P)

# Each inverse column as the signed products `sign · conj(F_I[keyI]) · F_J[keyJ]` it sums; keys are
# positions in the monomial key list. Evaluated when `_columns` is generated.
function _column_terms(::Type{IdentityTransport}, ::Val{W}, ::Val{P}, key_index::Dict) where {W, P}
    cols = Vector{Vector{NTuple{3, Int}}}()
    for j in SFT.symmetric_indices(Val(W), Val(P))
        terms = NTuple{3, Int}[]
        for mask in 0:((1 << P) - 1)
            primed = Tuple(j[k] for k in 1:P if (mask >> (k - 1)) & 1 == 1)
            unprimed = Tuple(j[k] for k in 1:P if (mask >> (k - 1)) & 1 == 0)
            sign = isodd(P - length(primed)) ? -1 : 1
            push!(terms, (sign, key_index[_padded_key(Val(P), unprimed)], key_index[_padded_key(Val(P), primed)]))
        end
        push!(cols, terms)
    end
    return cols
end

_column_terms(::Type{FrameTransport}, ::Val{W}, ::Val{P}, key_index::Dict) where {W, P} =
    [[(1, key_index[_padded_key(Val(P), cI)], key_index[_padded_key(Val(P), cJ)])]
     for (cI, cJ) in _raw_columns(Val(W), Val(P))]

"""
    _columns(transport, Val(W), Val(P), Val(Pm)) -> Vector{Vector{NTuple{3, Int}}}

The inverse columns of the degree-`P` moments under `transport`, each the signed products `(sign, a, b)` it sums,
`a` and `b` positions in `_monomial_keys(Val(W), Val(Pm))`.
"""
@generated function _columns(::T, ::Val{W}, ::Val{P}, ::Val{Pm}) where {T <: AbstractLagTransport, W, P, Pm}
    key_index = Dict(k[1:P] => i for (i, k) in enumerate(_monomial_keys(Val(W), Val(Pm))) if all(iszero, k[(P + 1):end]))
    cols = _column_terms(T, Val(W), Val(P), key_index)
    return :(Vector{NTuple{3, Int}}[copy(c) for c in $cols])
end

_inverse_count(::IdentityTransport, W::Int, P::Int) = binomial(W + P - 1, P)
_inverse_count(::FrameTransport, W::Int, P::Int) = _n_raw(W, P)

"""The number of inverse columns of the degree-`P` moments under `transport`, as a `Val`."""
@generated _column_count(::T, ::Val{W}, ::Val{P}) where {T <: AbstractLagTransport, W, P} =
    :(Val($(_inverse_count(T(), W, P))))

"""
    _sf_columns(sf, transport, ::Val{W}, ::Val{Pm}) -> (columns, Val(N))

The inverse columns of the moments `sf` reads, over the monomials `_monomial_keys(Val(W), Val(Pm))`, and their
count: the degree-`order(sf)` moments, or for the single-pass invariants the degree-2 columns then the degree-3
ones, `N = (N2, N3)`.
"""
function _sf_columns(sf, tr::AbstractLagTransport, vW::Val, vPm::Val)
    vp = Val(SFT.order(sf))
    return _columns(tr, vW, vp, vPm), _column_count(tr, vW, vp)
end

function _sf_columns(::SFT.SinglePassInvariants, tr::AbstractLagTransport, vW::Val, vPm::Val)
    columns = vcat(_columns(tr, vW, Val(2), vPm), _columns(tr, vW, Val(3), vPm))
    return columns, Val((_val_int(_column_count(tr, vW, Val(2))), _val_int(_column_count(tr, vW, Val(3)))))
end

"""Inverse columns per slab pair of the moments `sf` reads."""
_sf_inverse_count(sf, tr::AbstractLagTransport, W::Int) = _inverse_count(tr, W, SFT.order(sf))
_sf_inverse_count(::SFT.SinglePassInvariants, tr::AbstractLagTransport, W::Int) =
    _inverse_count(tr, W, 2) + _inverse_count(tr, W, 3)

"""
    _frame_moments(A, B, raw, scale, Val(W), Val(P)) -> SymmetricMoments{W, P}

The symmetric increment moment tensor of one lag from its raw cross-moments under the frames `A`,
`B`: `Σ_S (−1)^{P−|S|} Π_{k∈S} B_{a_k c_k} Π_{k∉S} A_{a_k c_k} · X[m u_{c_{S^c}}, m u_{c_S}]`. For each
`k = |S|` the raw block is transformed mode by mode — `A` on the first `P − k` slots, `B` on the
rest — and read at every placement of `k` slots of the sorted multi-index.
"""
@generated function _frame_moments(
    A::SA.SMatrix{W, W, TA}, B::SA.SMatrix{W, W, TB}, raw::SA.SVector{NR, TR}, scale,
    ::Val{W}, ::Val{P},
) where {W, TA, TB, NR, TR, P}
    T = promote_type(TA, TB, TR)
    L = W^P
    cols = _raw_columns(Val(W), Val(P))
    col_of = Dict(c => n for (n, c) in enumerate(cols))
    sorted = SFT.symmetric_indices(Val(W), Val(P))
    N = length(sorted)
    digits_of(lin) = ntuple(i -> (lin ÷ W^(i - 1)) % W + 1, P)
    body = Expr[]
    for k in 0:P
        table = ntuple(L) do l
            d = digits_of(l - 1)
            col_of[(Tuple(sort(collect(d[1:(P - k)]))), Tuple(sort(collect(d[(P - k + 1):P]))))]
        end
        push!(body, :(for lin in 1:$L
            dense[lin] = raw[$(table)[lin]]
        end))
        for i in 1:P
            mat = i <= P - k ? :A : :B
            stride = W^(i - 1)
            push!(body, quote
                for lin in 0:$(L - 1)
                    d = (lin ÷ $stride) % $W
                    base = lin - d * $stride
                    acc = zero($T)
                    for b in 0:$(W - 1)
                        acc += $mat[d + 1, b + 1] * dense[base + b * $stride + 1]
                    end
                    tmp[lin + 1] = acc
                end
                dense, tmp = tmp, dense
            end)
        end
        sign = isodd(P - k) ? -1 : 1
        for (n, a) in enumerate(sorted)
            lins = Int[]
            for mask in 0:((1 << P) - 1)
                count_ones(mask) == k || continue
                S = [i for i in 1:P if (mask >> (i - 1)) & 1 == 1]
                Sc = [i for i in 1:P if (mask >> (i - 1)) & 1 == 0]
                lin = 0
                for (i, slot) in enumerate(Sc)
                    lin += (a[slot] - 1) * W^(i - 1)
                end
                for (j, slot) in enumerate(S)
                    lin += (a[slot] - 1) * W^(P - k + j - 1)
                end
                push!(lins, lin + 1)
            end
            push!(body, :(for lin in $(Tuple(lins))
                m[$n] += $sign * dense[lin]
            end))
        end
    end
    return quote
        dense = SA.MVector{$L, $T}(undef)
        tmp = SA.MVector{$L, $T}(undef)
        m = zero(SA.MVector{$N, $T})
        @inbounds begin
            $(body...)
        end
        return SFT.SymmetricMoments{$W, $P}(SA.SVector(m) * scale)
    end
end

# The moment tensor of one lag from the inverted columns; a self-reverse lag's columns hold each pair
# twice, with equal values, so `scale` halves them.
@inline function _lag_moments(
    ::IdentityTransport, out, idx::Int, scale::T, A, B, ::Val{W}, ::Val{P}, ::Val{N},
) where {T, W, P, N}
    return SFT.SymmetricMoments{W, P}(SA.SVector{N, T}(ntuple(n -> T(@inbounds(out[idx, n])) * scale, Val(N))))
end

@inline function _lag_moments(
    ::FrameTransport, out, idx::Int, scale::T, A, B, ::Val{W}, ::Val{P}, ::Val{N},
) where {T, W, P, N}
    raw = SA.SVector{N, T}(ntuple(n -> T(@inbounds(out[idx, n])), Val(N)))
    return _frame_moments(A, B, raw, scale, Val(W), Val(P))
end

# The symmetric moment store of one lag, averaged over the lag's frames where the transport gives it
# more than one.
@inline function _lag_tensor(
    tr::IdentityTransport, out, idx, scale::T, geometry, ::Val{W}, ::Val{P}, ::Val{N},
) where {T, W, P, N}
    return _lag_moments(tr, out, idx, scale, nothing, nothing, Val(W), Val(P), Val(N)).data
end

@inline _lag_tensor(tr::FrameTransport, out, idx, scale, geometry, ::Val{W}, ::Val{P}, ::Val{N}) where {W, P, N} =
    with_frames(frames -> _frames_tensor(tr, out, idx, scale, frames, Val(W), Val(P), Val(N)), tr, geometry)

"""The symmetric moment store of one lag under frames already built for it."""
@inline _frames_tensor(tr::IdentityTransport, out, idx, scale, frames, ::Val{W}, ::Val{P}, ::Val{N}) where {W, P, N} =
    _lag_tensor(tr, out, idx, scale, nothing, Val(W), Val(P), Val(N))

@inline function _frames_tensor(
    tr::FrameTransport, out, idx, scale::T, frames::NTuple{M, <:NamedTuple}, ::Val{W}, ::Val{P}, ::Val{N},
) where {T, M, W, P, N}
    acc = _lag_moments(tr, out, idx, scale, frames[1].A, frames[1].B, Val(W), Val(P), Val(N)).data
    for m in 2:M
        f = frames[m]
        acc += _lag_moments(tr, out, idx, scale, f.A, f.B, Val(W), Val(P), Val(N)).data
    end
    return acc / M
end

# The operator on one lag, averaged over the lag's frames.
@inline function _lag_value(
    tr::IdentityTransport, sf, out, idx, scale::T, frames::NTuple{M, <:NamedTuple}, inv_r,
    ::Val{W}, ::Val{P}, ::Val{N}, ::Val{V}, ::Val{K},
) where {T, M, W, P, N, V, K}
    Mo = _lag_moments(tr, out, idx, scale, nothing, nothing, Val(W), Val(P), Val(N))
    acc = zero(T)
    for f in frames
        acc += SFT.moment_contract(sf, Mo, f.dir * inv_r, Val(V), Val(K))
    end
    return acc / M
end

@inline function _lag_value(
    tr::FrameTransport, sf, out, idx, scale::T, frames::NTuple{M, <:NamedTuple}, inv_r,
    ::Val{W}, ::Val{P}, ::Val{N}, ::Val{V}, ::Val{K},
) where {T, M, W, P, N, V, K}
    acc = zero(T)
    for f in frames
        Mo = _lag_moments(tr, out, idx, scale, f.A, f.B, Val(W), Val(P), Val(N))
        acc += SFT.moment_contract(sf, Mo, f.dir, Val(V), Val(K))
    end
    return acc / M
end

# The six single-pass invariants of one lag, averaged over its frames: S2, L2 and T2 from the degree-2
# moments in columns `1:N2`, S3, L3 and L1T2 from the degree-3 ones in the next `N3`.
@inline function _lag_value(
    tr::IdentityTransport, ::SFT.SinglePassInvariants, out, idx, scale::T, frames::NTuple{M, <:NamedTuple}, inv_r,
    ::Val{W}, ::Val{P}, ::Val{NN}, ::Val{V}, ::Val{K},
) where {T, M, W, P, NN, V, K}
    N2, N3 = NN
    M2 = _lag_moments(tr, out, idx, scale, nothing, nothing, Val(W), Val(2), Val(N2))
    M3 = _lag_moments(tr, view(out, :, (N2 + 1):(N2 + N3)), idx, scale, nothing, nothing, Val(W), Val(3), Val(N3))
    acc = zero(SA.SVector{SINGLE_PASS_N, T})
    for f in frames
        acc += _single_pass_contract(M2, M3, f.dir * inv_r, Val(V), Val(K))
    end
    return acc / M
end

@inline function _lag_value(
    tr::FrameTransport, ::SFT.SinglePassInvariants, out, idx, scale::T, frames::NTuple{M, <:NamedTuple}, inv_r,
    ::Val{W}, ::Val{P}, ::Val{NN}, ::Val{V}, ::Val{K},
) where {T, M, W, P, NN, V, K}
    N2, N3 = NN
    out3 = view(out, :, (N2 + 1):(N2 + N3))
    acc = zero(SA.SVector{SINGLE_PASS_N, T})
    for f in frames
        M2 = _lag_moments(tr, out, idx, scale, f.A, f.B, Val(W), Val(2), Val(N2))
        M3 = _lag_moments(tr, out3, idx, scale, f.A, f.B, Val(W), Val(3), Val(N3))
        acc += _single_pass_contract(M2, M3, f.dir, Val(V), Val(K))
    end
    return acc / M
end

"""The single-pass invariants, in `SINGLE_PASS_OPERATORS` order, from a lag's degree-2 and degree-3 moments."""
@inline _single_pass_contract(M2, M3, r̂, vV::Val, vK::Val) = SA.SVector(
    SFT.moment_contract(SINGLE_PASS_OPERATORS.S2, M2, r̂, vV, vK),
    SFT.moment_contract(SINGLE_PASS_OPERATORS.L2, M2, r̂, vV, vK),
    SFT.moment_contract(SINGLE_PASS_OPERATORS.T2, M2, r̂, vV, vK),
    SFT.moment_contract(SINGLE_PASS_OPERATORS.S3, M3, r̂, vV, vK),
    SFT.moment_contract(SINGLE_PASS_OPERATORS.L3, M3, r̂, vV, vK),
    SFT.moment_contract(SINGLE_PASS_OPERATORS.L1T2, M3, r̂, vV, vK),
)

