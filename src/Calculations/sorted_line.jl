# Exact pair statistics of polynomial operators on points along a line: sorted once, every bin of every
# point is one index range, and the moments over a range are prefix-sum differences of the monomials.

"""
    sorted_line_sweep!(sums, counts, sf, x, data, distance_bins, ::Val{D}, ::Val{V}, ::Val{K}; weights, backend)

Accumulate `Σ w_i w_j sf(δu, r̂)` and `Σ w_i w_j` over the pairs of a one-dimensional point list into
the distance histogram `sums`/`counts`, exactly, in `O(N log N + N n_bins)`.

`x` holds the coordinate of each point, `data` the packed channels `(V·D + K, N)` — `D = 1` for a
vector channel — and `sf` a polynomial operator ([`SFT.is_polynomial_operator`](@ref)). Sorted by
coordinate, the partners of a point in one bin form an index range, found by pointers that advance
with the point and bin pairs with the `digitize` call and the `x_j − x_i` the pair loop uses, so the
counts equal the pair loop's exactly. The increment moments over a range are prefix-sum differences
of the weighted monomials, contracted by `moment_contract` with the pair read from the lower to the
upper coordinate, which is the canonical reading of an odd scalar increment. Threaded backends split
the points into consecutive blocks with private histograms.
"""
function sorted_line_sweep!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT}, sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractVector, data::AbstractMatrix, distance_bins, ::Val{D}, ::Val{V}, ::Val{K};
    weights = NoWeights(), backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {OT, CT, D, V, K}
    SFT.is_polynomial_operator(sf) || throw(ArgumentError(
        "$(typeof(sf)) is not a polynomial in the increment; the sorted line route sums increment moments " *
        "and the pair loop evaluates any operator",
    ))
    validate_channels(sf, Val(V), Val(K))
    N = length(x)
    size(data, 2) == N || throw(DimensionMismatch("x holds $N points and the field $(size(data, 2))"))
    W = V * D + K
    size(data, 1) == W || throw(DimensionMismatch(
        "field has $(size(data, 1)) components, declared $V vector channel(s) of width $D and $K scalar " *
        "channel(s), $W components",
    ))
    w = _pair_weights(weights, N, OT)
    _check_weighted_counts(w, CT)
    be = BinEdges(distance_bins)
    nb = n_histogram_bins(be)
    length(sums) == nb == length(counts) || throw(DimensionMismatch(
        "$nb bins, but sums has $(length(sums)) entries and counts $(length(counts))",
    ))
    N < 2 && return nothing
    xs, ds, ws = _sorted_line_inputs(x, data, w, OT)
    P = SFT.order(sf)
    return _sorted_line_run!(sums, counts, sf, xs, ds, ws, be, nb, Val(W), Val(P), Val(D), Val(V), Val(K), backend)
end

"""The points in coordinate order, the field's columns and the weights permuted with them."""
function _sorted_line_inputs(x::AbstractVector, data::AbstractMatrix, w, ::Type{OT}) where {OT}
    xs = collect(float(eltype(x)), x)
    ds = Matrix{OT}(data)
    issorted(xs) && return xs, ds, (w isa NoWeights ? w : collect(w))
    perm = sortperm(xs)
    return xs[perm], ds[:, perm], (w isa NoWeights ? w : w[perm])
end

function _sorted_line_run!(
    sums::AbstractVector{OT}, counts, sf, xs::AbstractVector, ds::AbstractMatrix{OT}, w, be, nb::Int,
    ::Val{W}, ::Val{P}, ::Val{D}, ::Val{V}, ::Val{K}, backend,
) where {OT, W, P, D, V, K}
    N = length(xs)
    S = _monomial_prefix_sums(ds, w, Val(W), Val(P))
    r̂ = _unit_line(_direction_width(Val(D), Val(V), Val(1)), OT)
    n_tasks = max(1, sweep_tasks(backend))
    items = _consecutive_chunks(N - 1, n_tasks)
    make_scratch = () -> Vector{Int}(undef, nb + 1)
    vNK = _monomial_count(Val(W), Val(P))
    body! = (ls, lc, chunk, q) -> _sorted_line_chunk!(ls, lc, sf, xs, ds, w, S, be, nb, q, chunk, r̂,
                                                     Val(W), Val(P), Val(V), Val(K), vNK)
    sweep_reduce!(sums, counts, backend, items, make_scratch, body!)
    return nothing
end

"""The number of monomial keys of degree `≤ P` over `W` components, as a `Val`."""
@generated _monomial_count(::Val{W}, ::Val{P}) where {W, P} = :(Val($(length(_monomial_keys(Val(W), Val(P))))))

"""`1:n` cut into at most `k` consecutive ranges of near-equal length."""
function _consecutive_chunks(n::Int, k::Int)
    n < 1 && return UnitRange{Int}[]
    k = clamp(k, 1, n)
    return [((t - 1) * n ÷ k + 1):(t * n ÷ k) for t in 1:k]
end

@inline _unit_line(::Val{Dr}, ::Type{T}) where {Dr, T} =
    SA.SVector{Dr, T}(ntuple(d -> d == 1 ? one(T) : zero(T), Val(Dr)))

"""
Every monomial of degree `≤ P` of one point's channels, in the order of `_monomial_keys`; the first
entry is the degree-zero monomial `1`.
"""
@generated function _line_monomials(u::SA.SVector{W, T}, ::Val{W}, ::Val{P}) where {W, T, P}
    keys = _monomial_keys(Val(W), Val(P))
    terms = map(keys) do key
        factors = [:(u[$c]) for c in key if c != 0]
        isempty(factors) ? :(one($T)) : Expr(:call, :*, factors...)
    end
    return :(SA.SVector{$(length(keys)), $T}($(terms...)))
end

"""
    _monomial_prefix_sums(data, weights, Val(W), Val(P)) -> Matrix (n_keys, N + 1)

`S[k, n + 1] = Σ_{j ≤ n} w_j μ_k(j)` for every monomial key `k` of degree `≤ P`, `S[:, 1] = 0`.
"""
_monomial_prefix_sums(ds::AbstractMatrix, w, ::Val{W}, ::Val{P}) where {W, P} =
    _monomial_prefix_sums(ds, w, Val(W), Val(P), _monomial_count(Val(W), Val(P)))

function _monomial_prefix_sums(ds::AbstractMatrix{OT}, w, ::Val{W}, ::Val{P}, ::Val{NK}) where {OT, W, P, NK}
    N = size(ds, 2)
    S = Matrix{OT}(undef, NK, N + 1)
    acc = zero(SA.SVector{NK, OT})
    @inbounds S[:, 1] .= zero(OT)
    @inbounds for j in 1:N
        u = SA.SVector{W, OT}(ntuple(c -> ds[c, j], Val(W)))
        acc += _line_monomials(u, Val(W), Val(P)) * OT(_point_weight(w, j))
        for k in 1:NK
            S[k, j + 1] = acc[k]
        end
    end
    return S
end

"""
    _line_moments(μi, Δ, Val(W), Val(P)) -> SymmetricMoments{W, P}

The symmetric increment moment tensor of one point against a range of partners: `μi` the point's
monomials, `Δ` the range's summed monomials, combined as `Σ_S (−1)^{P−|S|} μ_{c_{Sᶜ}}(i) Δ[c_S]`.
"""
@generated function _line_moments(μi::SA.SVector{NK, T}, Δ::SA.SVector{NK, T}, ::Val{W}, ::Val{P}) where {NK, T, W, P}
    keys = _monomial_keys(Val(W), Val(P))
    key_index = Dict(k => n for (n, k) in enumerate(keys))
    cols = _columns(IdentityTransport(), Val(W), Val(P), key_index)
    entries = map(cols) do terms
        Expr(:call, :+, [:($sign * μi[$a] * Δ[$b]) for (sign, a, b) in terms]...)
    end
    return :(SFT.SymmetricMoments{$W, $P}(SA.SVector{$(length(entries)), $T}($(entries...))))
end

"""The pair mass of a partner range: its length, or its summed weights."""
@inline _range_mass(::NoWeights, S, lo::Int, hi::Int) = hi - lo
@inline _range_mass(::AbstractVector, S, lo::Int, hi::Int) = @inbounds S[1, hi + 1] - S[1, lo + 1]

# The pairs whose lower coordinate lies in `chunk`. `q[b + 1]` is the last index `j ≥ i − 1` whose
# separation from `i` digitizes to a bin `≤ b`; the predicate is monotone in `j` and in `i`, so the
# pointers only advance. Bin `b` of point `i` is the range `(max(q[b], i), q[b + 1]]`.
function _sorted_line_chunk!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT}, sf, xs::AbstractVector, ds::AbstractMatrix{OT},
    w, S::AbstractMatrix{OT}, be, nb::Int, q::Vector{Int}, chunk::UnitRange{Int}, r̂,
    ::Val{W}, ::Val{P}, ::Val{V}, ::Val{K}, ::Val{NK},
) where {OT, CT, W, P, V, K, NK}
    N = length(xs)
    i0 = first(chunk)
    @inbounds for b in 0:nb
        q[b + 1] = _line_partner_bound(xs, be, i0, b, N)
    end
    @inbounds for i in chunk
        xi = xs[i]
        prev = i - 1
        for b in 0:nb
            p = max(q[b + 1], prev)
            while p < N && SFH.digitize(xs[p + 1] - xi, be) <= b
                p += 1
            end
            q[b + 1] = p
            prev = p
        end
        ui = SA.SVector{W, OT}(ntuple(c -> ds[c, i], Val(W)))
        μi = _line_monomials(ui, Val(W), Val(P))
        wi = OT(_point_weight(w, i))
        for b in 1:nb
            lo = max(q[b], i)
            hi = q[b + 1]
            hi > lo || continue
            Δ = SA.SVector{NK, OT}(ntuple(k -> S[k, hi + 1] - S[k, lo + 1], Val(NK)))
            M = _line_moments(μi, Δ, Val(W), Val(P))
            sums[b] += wi * SFT.moment_contract(sf, M, r̂, Val(V), Val(K))
            counts[b] += CT(wi * _range_mass(w, S, lo, hi))
        end
    end
    return nothing
end

"""The last index `j ∈ [i − 1, N]` whose separation from point `i` digitizes to a bin `≤ b`, by bisection."""
function _line_partner_bound(xs::AbstractVector, be, i::Int, b::Int, N::Int)
    lo, hi = i - 1, N
    xi = @inbounds xs[i]
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if SFH.digitize(@inbounds(xs[mid]) - xi, be) <= b
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end
