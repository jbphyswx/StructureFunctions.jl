# Kernel-binned pair statistics on a sphere from spherical-harmonic pseudo-coefficients: the
# truncated Legendre / Wigner-d series of the masked field's pseudo-spectra is, exactly, the pair sum
# with a soft kernel of width ~π/lmax in place of a hard bin.

# ---------------------------------------------------------------------------------------------------
# Wigner small-d functions: d^l_{mn}(β) = ⟨l m| exp(−iβ J_y) |l n⟩, Condon–Shortley phase
# ---------------------------------------------------------------------------------------------------

"""`log k!` for `k = 0:N` at `t[k + 1]`."""
function _log_factorials(N::Integer)
    t = zeros(Float64, N + 1)
    for k in 1:N
        t[k + 1] = t[k] + log(k)
    end
    return t
end

# d^l_{mn}(β) at l = max(|m|, |n|), where one index is extreme and the sum over k has one term.
function _wigner_seed(l::Int, m::Int, n::Int, β::Real, lf::AbstractVector)
    l == 0 && return 1.0
    c, s = cos(β / 2), sin(β / 2)
    if abs(m) == l
        A = exp((lf[2l + 1] - lf[l + n + 1] - lf[l - n + 1]) / 2)
        return m > 0 ? A * c^(l + n) * (-s)^(l - n) : A * c^(l - n) * s^(l + n)
    end
    A = exp((lf[2l + 1] - lf[l + m + 1] - lf[l - m + 1]) / 2)
    return n > 0 ? A * c^(l + m) * s^(l - m) : A * c^(l - m) * (-s)^(l + m)
end

"""
    wigner_d_column!(out, m, n, β, lmax, lf) -> out

`out[l + 1] = d^l_{mn}(β)` for `l = 0:lmax`, zero below `max(|m|, |n|)`, by the three-term recurrence
in `l` from the closed form at the lowest degree; `lf` is [`_log_factorials`](@ref)`(2lmax)` or longer.
The recurrence is stable in the direction of increasing `l`.
"""
function wigner_d_column!(out::AbstractVector{Float64}, m::Integer, n::Integer, β::Real, lmax::Integer, lf)
    fill!(out, 0.0)
    l0 = max(abs(m), abs(n))
    l0 > lmax && return out
    x = cos(β)
    @inbounds out[l0 + 1] = _wigner_seed(l0, m, n, β, lf)
    prev = 0.0
    @inbounds for l in l0:(lmax - 1)
        cur = out[l + 1]
        a = l == 0 ? 0.0 : (m * n) / (l * (l + 1))
        c1 = l == 0 ? 0.0 : sqrt(Float64((l^2 - m^2) * (l^2 - n^2))) / (l * (2l + 1))
        c2 = sqrt(Float64(((l + 1)^2 - m^2) * ((l + 1)^2 - n^2))) / ((l + 1) * (2l + 1))
        out[l + 2] = ((x - a) * cur - c1 * prev) / c2
        prev = cur
    end
    return out
end

"""
    wigner_d_column(m, n, β, lmax) -> Vector

[`wigner_d_column!`](@ref) into a fresh vector.
"""
wigner_d_column(m::Integer, n::Integer, β::Real, lmax::Integer) =
    wigner_d_column!(zeros(Float64, lmax + 1), m, n, β, lmax, _log_factorials(2lmax + 2))

# ---------------------------------------------------------------------------------------------------
# Pseudo-coefficients
# ---------------------------------------------------------------------------------------------------

"""
    pseudo_coefficients_direct(f, θ, φ, s, lmax) -> Matrix{ComplexF64}

`C[l + 1, m + lmax + 1] = Σ_i f_i conj(ₛY_lm(θ_i, φ_i))` with `ₛY_lm = √((2l+1)/4π) d^l_{m,−s}(θ) e^{imφ}`,
by direct summation over the points: `O(N lmax²)`, and the reference every faster provider is checked
against. `θ` is colatitude and `φ` longitude, both in radians.
"""
function pseudo_coefficients_direct(f::AbstractVector{<:Number}, θ::AbstractVector, φ::AbstractVector,
                                    s::Integer, lmax::Integer)
    N = length(f)
    length(θ) == length(φ) == N || throw(DimensionMismatch("f, θ and φ must have one entry per point"))
    out = zeros(ComplexF64, lmax + 1, 2lmax + 1)
    col = zeros(Float64, lmax + 1)
    lf = _log_factorials(2lmax + 2)
    norms = [sqrt((2l + 1) / (4π)) for l in 0:lmax]
    @inbounds for i in 1:N
        fi = ComplexF64(f[i])
        iszero(fi) && continue
        for m in -lmax:lmax
            wigner_d_column!(col, m, -s, θ[i], lmax, lf)
            ph = fi * cis(-m * φ[i])
            for l in max(abs(m), abs(s)):lmax
                out[l + 1, m + lmax + 1] += ph * norms[l + 1] * col[l + 1]
            end
        end
    end
    return out
end

"""
    direct_sum_provider(θ, φ, lmax) -> (f, s) -> coefficients

The pseudo-coefficient provider of the direct sum, in the form every provider takes: a callable of a
complex point field and a spin.
"""
direct_sum_provider(θ::AbstractVector, φ::AbstractVector, lmax::Integer) =
    (f, s) -> pseudo_coefficients_direct(f, θ, φ, s, lmax)

# Spin-(−s) pseudo-coefficients of `conj(f)` from the spin-`s` ones of `f`:
# conj(ₛY_lm) = (−1)^{m+s} ₋ₛY_{l,−m}, so ₋ₛ[conj f]_lm = (−1)^{m+s} conj(ₛf_{l,−m}).
function _conjugate_field_coefficients(C::AbstractMatrix, s::Integer, lmax::Integer)
    out = similar(C)
    @inbounds for l in 0:lmax, m in -lmax:lmax
        out[l + 1, m + lmax + 1] = (isodd(m + s) ? -1 : 1) * conj(C[l + 1, -m + lmax + 1])
    end
    return out
end

"""`X_l = (1/(2l+1)) Σ_m F_lm conj(G_lm)`, the cross pseudo-spectrum of two coefficient arrays."""
function _cross_spectrum(F::AbstractMatrix, G::AbstractMatrix, lmax::Integer)
    X = zeros(ComplexF64, lmax + 1)
    @inbounds for l in 0:lmax
        acc = zero(ComplexF64)
        for m in -l:l
            acc += F[l + 1, m + lmax + 1] * conj(G[l + 1, m + lmax + 1])
        end
        X[l + 1] = acc / (2l + 1)
    end
    return X
end

# (2l+1)/(4π) b_l d^l_{ss′}(β_k) for every degree and node, as (lmax + 1, n_nodes).
function _node_kernel(s::Integer, s′::Integer, nodes::HarmonicNodes, lf)
    L = nodes.lmax
    K = zeros(Float64, L + 1, length(nodes))
    col = zeros(Float64, L + 1)
    for (k, β) in enumerate(nodes.separations)
        wigner_d_column!(col, s, s′, β, L, lf)
        for l in 0:L
            K[l + 1, k] = (2l + 1) / (4π) * harmonic_taper(nodes.taper, l, L) * col[l + 1]
        end
    end
    return K
end

# ---------------------------------------------------------------------------------------------------
# An operator as a polynomial in the two ends' spin quantities
# ---------------------------------------------------------------------------------------------------
#
# In the pair's geodesic frame `u_L + i u_T = Ū`, where `Ū = U e^{−iψ}` is the spin-1 quantity
# `U = u_θ + i u_φ` rotated by the bearing `ψ` of the geodesic toward the other point. At the second
# point the frame continues the geodesic, so its tangent has bearing `ψ + π` and the frame components
# there are `−Ū`. Hence `δu_L + i δu_T = −(Ū_i + Ū_j)`; a radial component and a scalar difference as
# `w_j − w_i`, `θ_j − θ_i`. Every polynomial operator is a polynomial in these, and every monomial
# splits into a product at `i` and a product at `j`, each a spin-weighted monomial of the field.

# A monomial of the point quantities at one end: powers of U_v, conj(U_v), the radial component of
# channel v, and the scalar channels; its spin is Σ_v (a_v − b_v).
struct SiteMonomial{V, K}
    a::NTuple{V, Int}
    b::NTuple{V, Int}
    r::NTuple{V, Int}
    c::NTuple{K, Int}
end

_spin(p::SiteMonomial) = sum(p.a; init = 0) - sum(p.b; init = 0)
_conjugate(p::SiteMonomial) = SiteMonomial(p.b, p.a, p.r, p.c)

# Site variables of a pair, as positions in an exponent tuple: for channel v the four
# (Ū_i, conj Ū_i, Ū_j, conj Ū_j), then (w_i, w_j) per channel on a shell, then (θ_i, θ_j) per scalar.
struct SiteLayout{V, K, R} end
@inline _n_vars(::SiteLayout{V, K, R}) where {V, K, R} = Val(4V + (R ? 2V : 0) + 2K)
_exponent_type(::SiteLayout{V, K, R}) where {V, K, R} = NTuple{4V + (R ? 2V : 0) + 2K, Int}
_ubar(::SiteLayout, v, end_) = 4(v - 1) + (end_ == 1 ? 1 : 3)
_cubar(::SiteLayout, v, end_) = 4(v - 1) + (end_ == 1 ? 2 : 4)
_rad(::SiteLayout{V}, v, end_) where {V} = 4V + 2(v - 1) + end_
_sca(::SiteLayout{V, K, R}, k, end_) where {V, K, R} = 4V + (R ? 2V : 0) + 2(k - 1) + end_

_unit_exponent(l::SiteLayout, i) = ntuple(j -> Int(j == i), _n_vars(l))
_zero_exponent(l::SiteLayout) = ntuple(_ -> 0, _n_vars(l))

# The increment of packed component `comp` as a polynomial over the site variables.
function _increment_polynomial(l::SiteLayout{V}, comp::Int, D::Int) where {V}
    p = Dict{_exponent_type(l), ComplexF64}()
    if comp <= V * D
        v = (comp - 1) ÷ D + 1
        d = (comp - 1) % D + 1
        if d == 1
            for e in (_ubar(l, v, 1), _ubar(l, v, 2), _cubar(l, v, 1), _cubar(l, v, 2))
                p[_unit_exponent(l, e)] = -0.5
            end
        elseif d == 2
            for e in (_ubar(l, v, 1), _ubar(l, v, 2))
                p[_unit_exponent(l, e)] = 0.5im
            end
            for e in (_cubar(l, v, 1), _cubar(l, v, 2))
                p[_unit_exponent(l, e)] = -0.5im
            end
        else
            p[_unit_exponent(l, _rad(l, v, 2))] = 1.0
            p[_unit_exponent(l, _rad(l, v, 1))] = -1.0
        end
    else
        k = comp - V * D
        p[_unit_exponent(l, _sca(l, k, 2))] = 1.0
        p[_unit_exponent(l, _sca(l, k, 1))] = -1.0
    end
    return p
end

function _poly_mul(p::Dict{E, ComplexF64}, q::Dict{E, ComplexF64}) where {E}
    out = Dict{E, ComplexF64}()
    for (e1, c1) in p, (e2, c2) in q
        e = e1 .+ e2
        out[e] = get(out, e, zero(ComplexF64)) + c1 * c2
    end
    return out
end

# The polynomial of `sf` in the site variables: Σ_a c_a Π_k δu_{a_k}, with `c_a` the coefficient of
# each monomial of the packed increment read off the operator's contraction of a basis moment tensor.
function _operator_polynomial(sf, ::Val{D}, ::Val{V}, ::Val{K}) where {D, V, K}
    P = SFT.order(sf)
    W = V * D + K
    lay = SiteLayout{V, K, D == 3}()
    ê₁ = SA.SVector{D, Float64}(ntuple(i -> i == 1 ? 1.0 : 0.0, Val(D)))
    idx = SFT.symmetric_indices(Val(W), Val(P))
    N = length(idx)
    incr = [_increment_polynomial(lay, comp, D) for comp in 1:W]
    total = Dict{_exponent_type(lay), ComplexF64}()
    for (n, a) in enumerate(idx)
        E = SFT.SymmetricMoments{W, P}(SA.SVector{N, Float64}(ntuple(i -> i == n ? 1.0 : 0.0, Val(N))))
        c = SFT.moment_contract(sf, E, ê₁, Val(V), Val(K))
        iszero(c) && continue
        mono = Dict{_exponent_type(lay), ComplexF64}(_zero_exponent(lay) => 1.0)
        for comp in a
            mono = _poly_mul(mono, incr[comp])
        end
        for (e, coef) in mono
            total[e] = get(total, e, zero(ComplexF64)) + c * coef
        end
    end
    return total, lay
end

# Split each monomial into the product at `i`, a spin-weighted monomial `F` of the field, and the
# product at `j`, which is the conjugate of a spin-weighted monomial `G`. Returns (F, G) => coefficient.
function _operator_terms(sf, vD::Val{D}, vV::Val{V}, vK::Val{K}) where {D, V, K}
    poly, lay = _operator_polynomial(sf, vD, vV, vK)
    radial = D == 3
    terms = Dict{Tuple{SiteMonomial{V, K}, SiteMonomial{V, K}}, ComplexF64}()
    for (e, coef) in poly
        F = SiteMonomial(ntuple(v -> e[_ubar(lay, v, 1)], Val(V)), ntuple(v -> e[_cubar(lay, v, 1)], Val(V)),
                         ntuple(v -> radial ? e[_rad(lay, v, 1)] : 0, Val(V)), ntuple(k -> e[_sca(lay, k, 1)], Val(K)))
        G = SiteMonomial(ntuple(v -> e[_cubar(lay, v, 2)], Val(V)), ntuple(v -> e[_ubar(lay, v, 2)], Val(V)),
                         ntuple(v -> radial ? e[_rad(lay, v, 2)] : 0, Val(V)), ntuple(k -> e[_sca(lay, k, 2)], Val(K)))
        key = (F, G)
        terms[key] = get(terms, key, zero(ComplexF64)) + coef
    end
    return terms
end

# ---------------------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------------------

# Point-wise values of a site monomial, weighted and masked: zero where the point holds nothing.
function _monomial_values(p::SiteMonomial, U::AbstractMatrix{ComplexF64}, R::AbstractMatrix,
                          Θ::AbstractMatrix, wm::AbstractVector)
    N = length(wm)
    out = Vector{ComplexF64}(undef, N)
    @inbounds for i in 1:N
        if iszero(wm[i])
            out[i] = 0
            continue
        end
        v = ComplexF64(wm[i])
        for c in eachindex(p.a)
            v *= U[c, i]^p.a[c] * conj(U[c, i])^p.b[c] * R[c, i]^p.r[c]
        end
        for k in eachindex(p.c)
            v *= Θ[k, i]^p.c[k]
        end
        out[i] = v
    end
    return out
end

# Pseudo-coefficients of every site monomial the operator needs, one transform per distinct monomial
# of non-negative spin; a negative spin is the conjugate of a monomial already transformed. The mask's
# transform, taken at construction, fixes the coefficient array type.
mutable struct HarmonicTransforms{P, M <: SiteMonomial, AT <: AbstractMatrix{ComplexF64}}
    provider::P
    lmax::Int
    cache::Dict{M, AT}
    transforms::Int
end

function HarmonicTransforms(provider, lmax::Int, mask::SiteMonomial, U, R, Θ, wm)
    C = provider(_monomial_values(mask, U, R, Θ, wm), 0)
    return HarmonicTransforms(provider, lmax, Dict{typeof(mask), typeof(C)}(mask => C), 1)
end

function _coefficients!(ht::HarmonicTransforms, p::SiteMonomial, U, R, Θ, wm)
    cached = get(ht.cache, p, nothing)
    cached === nothing || return cached
    s = _spin(p)
    if s < 0
        C = _conjugate_field_coefficients(_coefficients!(ht, _conjugate(p), U, R, Θ, wm), -s, ht.lmax)
    else
        C = ht.provider(_monomial_values(p, U, R, Θ, wm), s)
        ht.transforms += 1
    end
    ht.cache[p] = C
    return C
end

@inline _spin_sign(s::Integer) = isodd(s) ? -1 : 1

"""
    _sphere_angles(geometry, x) -> (θ, φ)

Colatitude and longitude in radians of each point of `x`, given as `(lon, lat)` in the geometry's
metric's own angle unit.
"""
function _sphere_angles(g::SFH.SphericalGeometry, x::AbstractMatrix)
    size(x, 1) == 2 || throw(DimensionMismatch("a point on a sphere is (lon, lat); got $(size(x, 1)) coordinates"))
    N = size(x, 2)
    θ = Vector{Float64}(undef, N)
    φ = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        p = SFH.unit_position(g.metric, x[1, i], x[2, i])
        θ[i] = acos(clamp(p[3], -1.0, 1.0))
        φ[i] = mod(atan(p[2], p[1]), 2π)
    end
    return θ, φ
end

"""
    harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes, ::Val{D}, ::Val{V}, ::Val{K}, spectral_backend; valid)

Accumulate the kernel-binned pair statistic of `sf` at the nodes' separations into `sums`, and the
kernel-weighted pair count into `counts`, from spherical-harmonic pseudo-coefficients of the masked,
weighted field. `x` is `(lon, lat)` per point in the metric's angle unit; `data` is the packed field
with vector channels in `(east, north[, radial])` components; `weights` one weight per point; `valid`
which points hold a datum. `spectral_backend` names what computes the pseudo-coefficients: the
direct sum, `O(N lmax²)`, or a fast spherical harmonic transform from an extension.

For a spin-`s` monomial `F` at one end and a spin-`s′` monomial `G` at the other, rotated into the
pair's geodesic frame,

```
Σ_{ij} w_i w_j F̄_i conj(Ḡ_j) K^{ss′}(γ_ij, β) = (−1)^s Σ_l (2l+1)/(4π) b_l X^{FG}_l d^l_{ss′}(β),
K^{ss′}(γ, β) = (1/16π²) Σ_l (2l+1) b_l d^l_{ss′}(γ) d^l_{ss′}(β),
```

with `X^{FG}_l = (1/(2l+1)) Σ_m F̃_lm conj(G̃_lm)` the cross pseudo-spectrum, so every polynomial
operator is a finite sum of such series and the counts are the mask's own Legendre series. The
kernel tends to `(1/8π²) δ(cos γ − cos β)` as `lmax` grows, so the ratio `sums ./ counts` tends to
the hard-binned pair average.

An operator odd in a scalar increment is refused: the kernel sum runs over both readings of every
pair, so such a moment is identically zero here.
"""
harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes::HarmonicNodes, ::Val{D}, ::Val{V}, ::Val{K},
                ::SB.AbstractDirectSumSpectralBackend; valid = AllValid()) where {D, V, K} =
    _harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes, Val(D), Val(V), Val(K), valid,
                     direct_sum_provider)

harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes::HarmonicNodes, ::Val{D}, ::Val{V}, ::Val{K},
                spectral_backend; valid = AllValid()) where {D, V, K} = _no_harmonic_provider(spectral_backend)

_no_harmonic_provider(spectral_backend) = throw(ArgumentError(
    "no method computes spherical harmonic pseudo-coefficients with $(typeof(spectral_backend)). " *
    "DirectSumSpectralBackend() is the direct sum; `using NUFSHT` supplies the fast transform " *
    "(NUFSHTSpectralBackend()).",
))

function _harmonic_sweep!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT}, sf::SFT.AbstractPairwiseStructureFunctionType,
    g::SFH.SphericalGeometry, x::AbstractMatrix, weights::AbstractVector, data::AbstractMatrix,
    nodes::HarmonicNodes, ::Val{D}, ::Val{V}, ::Val{K}, valid, make_provider,
) where {OT, CT <: AbstractFloat, D, V, K}
    SFT.is_polynomial_operator(sf) || throw(ArgumentError(
        "$(typeof(sf)) is not a polynomial in the increment, so no harmonic series produces it; " *
        "the pair loop evaluates any pairwise operator.",
    ))
    SFT.is_odd_in_scalars(sf) && throw(ArgumentError(
        "$(typeof(sf)) is odd in a scalar increment, and a harmonic kernel sums both readings of " *
        "every pair, so the moment is identically zero on this route. Use the pair loop, which reads " *
        "each pair once by its convention.",
    ))
    validate_channels(sf, Val(V), Val(K))
    (V == 0 || D == 2 || D == 3) || throw(ArgumentError(
        "a vector channel on a sphere has 2 components (east, north) or 3 (with radial); got $D",
    ))
    N = size(x, 2)
    W = V * D + K
    size(data) == (W, N) || throw(DimensionMismatch(
        "field is $(size(data)); expected ($W, $N) for $N points",
    ))
    length(weights) == N || throw(DimensionMismatch("$(length(weights)) weights for $N points"))
    nb = length(nodes)
    length(sums) == nb && length(counts) == nb || throw(DimensionMismatch(
        "sums and counts must have one entry per node, $nb; got $(length(sums)) and $(length(counts))",
    ))

    θ, φ = _sphere_angles(g, x)
    # the spin-1 quantity u_θ + i u_φ of each vector channel: θ̂ is south, φ̂ is east
    U = Matrix{ComplexF64}(undef, V, N)
    R = Matrix{Float64}(undef, V, N)
    Θ = Matrix{Float64}(undef, K, N)
    wm = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        ok = valid[i]
        wm[i] = ok ? Float64(weights[i]) : 0.0
        for v in 1:V
            o = (v - 1) * D
            U[v, i] = ok ? ComplexF64(-data[o + 2, i], data[o + 1, i]) : 0
            R[v, i] = (ok && D == 3) ? data[o + 3, i] : 0.0
        end
        for k in 1:K
            Θ[k, i] = ok ? data[V * D + k, i] : 0.0
        end
    end

    L = nodes.lmax
    lf = _log_factorials(2L + 2)
    mask = SiteMonomial(ntuple(_ -> 0, Val(V)), ntuple(_ -> 0, Val(V)), ntuple(_ -> 0, Val(V)), ntuple(_ -> 0, Val(K)))
    ht = HarmonicTransforms(make_provider(θ, φ, L), L, mask, U, R, Θ, wm)
    Cm = _coefficients!(ht, mask, U, R, Θ, wm)
    kernels = Dict{Tuple{Int, Int}, Matrix{Float64}}()
    kernel(s, s′) = get!(kernels, (s, s′)) do
        _node_kernel(s, s′, nodes, lf)
    end
    Xmm = _cross_spectrum(Cm, Cm, L)
    K00 = kernel(0, 0)
    @inbounds for k in 1:nb
        acc = zero(ComplexF64)
        for l in 0:L
            acc += Xmm[l + 1] * K00[l + 1, k]
        end
        counts[k] += CT(real(acc))
    end

    acc = zeros(ComplexF64, nb)
    for ((F, G), coef) in _operator_terms(sf, Val(D), Val(V), Val(K))
        s, s′ = _spin(F), _spin(G)
        X = _cross_spectrum(_coefficients!(ht, F, U, R, Θ, wm), _coefficients!(ht, G, U, R, Θ, wm), L)
        Kd = kernel(s, s′)
        sign = _spin_sign(s)
        @inbounds for k in 1:nb
            series = zero(ComplexF64)
            for l in 0:L
                series += X[l + 1] * Kd[l + 1, k]
            end
            acc[k] += coef * sign * series
        end
    end
    @inbounds for k in 1:nb
        sums[k] += OT(real(acc[k]))
    end
    return sums, counts
end

"""
    calculate_structure_function(sf, x, u, nodes::HarmonicNodes, spectral_backend; distance_metric, weights, valid, output_type)

The kernel-binned structure function of `u` sampled at the points `x` of a sphere, at the nodes'
separations, by spherical harmonic pseudo-coefficients (see [`harmonic_sweep!`](@ref)). `x` is
`(lon, lat)` in the angle unit of `distance_metric`, which must be spherical; `u` is a `(D, cells...)`
field of `(east, north[, radial])` components, a `(1, cells...)` scalar, or a `Fields` bundle, with
`prod(cells) == size(x, 2)`. `weights` default to
one per point; on a grid the cell measure makes the statistic an area average.

The result's `distance` is the `HarmonicNodes` object itself, one value per node, and its counts are
the kernel-weighted pair counts, floating point.
"""
function calculate_structure_function(
    sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractMatrix, u::Union{AbstractArray, CH.Fields},
    nodes::HarmonicNodes, spectral_backend;
    distance_metric::DI.PreMetric = DI.SphericalAngle(), weights = nothing, valid = nothing,
    output_type::Type{OT} = SFO.StructureFunction, verbose::Bool = true, show_progress::Bool = true,
) where {OT}
    data, vD, vV, vK = _packed(u)
    D = SFC_val_int(vD)
    V = SFC_val_int(vV)
    geometry = SFH.pair_geometry_for(distance_metric, Val(V == 0 ? 2 : D))
    geometry isa SFH.SphericalGeometry || throw(ArgumentError(
        "a harmonic structure function lives on a sphere; $(typeof(distance_metric)) describes none. " *
        "Pass SphericalAngle(), SphericalDistance(R) or Haversine(R).",
    ))
    N = size(x, 2)
    w = weights === nothing ? ones(Float64, N) : weights
    v = valid === nothing ? field_validity(data) : valid
    nb = length(nodes)
    sums = zeros(float(eltype(data)), nb)
    counts = zeros(Float64, nb)
    verbose && @info "harmonic structure function: $(nb) nodes, lmax = $(nodes.lmax), $(nameof(typeof(spectral_backend)))"
    harmonic_sweep!(sums, counts, sf, geometry, x, w, data, nodes, vD, vV, vK, spectral_backend; valid = v)
    return _finalize(SFO.StructureFunctionSumsAndCounts(sf, nodes, sums, counts), OT)
end

# ---------------------------------------------------------------------------------------------------
# Pseudo-spectra
# ---------------------------------------------------------------------------------------------------

"""
    harmonic_spectra(x, u, lmax, spectral_backend; distance_metric, weights, valid) -> (l, C) or (l, EE, BB, EB)

Pseudo-spectra of the masked, weighted field: `C̃_l = (1/(2l+1)) Σ_m |ũ_lm|²` for a `(1, N)` scalar;
for a `(2, N)` tangent vector in `(east, north)`, the gradient (`E`) and curl (`B`) spectra and their
cross spectrum from the spin-1 coefficients `₁U_lm` of `U = u_θ + i u_φ` and the spin-(−1) coefficients
`₋₁Ū_lm` of its conjugate, which follow from the first as `₋₁Ū_lm = (−1)^{m+1} conj(₁U_{l,−m})`:

```
Φ_lm = (₁U_lm − ₋₁Ū_lm) / (2√(l(l+1))),   Ψ_lm = −i (₁U_lm + ₋₁Ū_lm) / (2√(l(l+1))),
C^E_l = l(l+1) (1/(2l+1)) Σ_m |Φ_lm|²,   C^B_l likewise from Ψ,   C^{EB}_l from Φ conj(Ψ),
```

so that `Σ_l (2l+1)/(4π) (C^E_l + C^B_l)` is the mean square of `u` on a complete sphere with exact
quadrature weights. On a masked or unevenly sampled sphere these are the pseudo-spectra of the
window and the field together, the input to the kernel-binned statistics.
"""
harmonic_spectra(x::AbstractMatrix, u::AbstractMatrix, lmax::Integer, ::SB.AbstractDirectSumSpectralBackend;
                 distance_metric::DI.PreMetric = DI.SphericalAngle(), weights = nothing, valid = nothing) =
    _harmonic_spectra(x, u, lmax, distance_metric, weights, valid, direct_sum_provider)

harmonic_spectra(x::AbstractMatrix, u::AbstractMatrix, lmax::Integer, spectral_backend;
                 distance_metric::DI.PreMetric = DI.SphericalAngle(), weights = nothing, valid = nothing) =
    _no_harmonic_provider(spectral_backend)

function _harmonic_spectra(x::AbstractMatrix, u::AbstractMatrix, lmax::Integer, distance_metric, weights,
                           valid, make_provider)
    D = size(u, 1)
    D in (1, 2) || throw(ArgumentError(
        "harmonic spectra are defined for a scalar (1, N) or a tangent vector (2, N) in (east, north); got $D rows",
    ))
    N = size(x, 2)
    size(u, 2) == N || throw(DimensionMismatch("$N points and $(size(u, 2)) samples"))
    g = SFH.pair_geometry_for(distance_metric, Val(2))
    g isa SFH.SphericalGeometry || throw(ArgumentError(
        "harmonic spectra live on a sphere; $(typeof(distance_metric)) describes none.",
    ))
    θ, φ = _sphere_angles(g, x)
    w = weights === nothing ? ones(Float64, N) : weights
    v = valid === nothing ? field_validity(u) : valid
    wm = [v[i] ? Float64(w[i]) : 0.0 for i in 1:N]
    provider = make_provider(θ, φ, lmax)
    l = 0:lmax
    if D == 1
        f = [iszero(wm[i]) ? 0.0 : wm[i] * u[1, i] for i in 1:N]
        C = provider(f, 0)
        return (l = l, C = real.(_cross_spectrum(C, C, lmax)))
    end
    U = [iszero(wm[i]) ? zero(ComplexF64) : wm[i] * ComplexF64(-u[2, i], u[1, i]) for i in 1:N]
    Cp = provider(U, 1)                                    # ₁U, the spin-1 coefficients of u_θ + i u_φ
    Cm = _conjugate_field_coefficients(Cp, 1, lmax)         # ₋₁[conj U], those of u_θ − i u_φ
    Φ = zeros(ComplexF64, lmax + 1, 2lmax + 1)
    Ψ = zeros(ComplexF64, lmax + 1, 2lmax + 1)
    @inbounds for ll in 1:lmax, m in -ll:ll
        f = 1 / (2 * sqrt(ll * (ll + 1)))
        Φ[ll + 1, m + lmax + 1] = (Cp[ll + 1, m + lmax + 1] - Cm[ll + 1, m + lmax + 1]) * f
        Ψ[ll + 1, m + lmax + 1] = -im * (Cp[ll + 1, m + lmax + 1] + Cm[ll + 1, m + lmax + 1]) * f
    end
    weight = [Float64(ll * (ll + 1)) for ll in l]
    EE = real.(_cross_spectrum(Φ, Φ, lmax)) .* weight
    BB = real.(_cross_spectrum(Ψ, Ψ, lmax)) .* weight
    EB = real.(_cross_spectrum(Φ, Ψ, lmax)) .* weight
    return (l = l, EE = EE, BB = BB, EB = EB)
end
