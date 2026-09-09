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

"""Column-major strides of a transform of size `P`."""
@inline _lag_strides(P::NTuple{Dg, Int}) where {Dg} =
    ntuple(d -> prod(ntuple(k -> P[k], d - 1); init = 1), Val(Dg))

"""Linear position of lag `h` in a transform of size `P`, wrapping the negative offsets."""
@inline function _lag_index(h::NTuple{Dg, Int}, P::NTuple{Dg, Int}, strides::NTuple{Dg, Int}) where {Dg}
    lin = 1
    @inbounds for d in 1:Dg
        lin += mod(h[d], P[d]) * strides[d]
    end
    return lin
end


"""Every sorted multi-index of degree `0:P` over `W` components, zero-padded to `P`: the monomials a degree-`P` moment reads."""
function _monomial_keys(::Val{W}, ::Val{P}) where {W, P}
    keys = NTuple{P, Int}[]
    for d in 0:P, k in SFT.symmetric_indices(Val(W), Val(d))
        push!(keys, _padded_key(Val(P), k))
    end
    return keys
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
# positions in the monomial key list.
function _columns(::IdentityTransport, ::Val{W}, ::Val{P}, key_index::Dict) where {W, P}
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

_columns(::FrameTransport, ::Val{W}, ::Val{P}, key_index::Dict) where {W, P} =
    [[(1, key_index[_padded_key(Val(P), cI)], key_index[_padded_key(Val(P), cJ)])]
     for (cI, cJ) in _raw_columns(Val(W), Val(P))]

_inverse_count(::IdentityTransport, W::Int, P::Int) = binomial(W + P - 1, P)
_inverse_count(::FrameTransport, W::Int, P::Int) = _n_raw(W, P)

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

