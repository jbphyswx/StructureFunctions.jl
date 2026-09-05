module StructureFunctionsFFTExt

using AbstractFFTs: AbstractFFTs, plan_rfft, irfft
using StaticArrays: StaticArrays as SA
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using SpectralBackends: SpectralBackends as SB

"""
    _pad_dims(schedule) -> NTuple{Dg, Int}

Transform length per direction. A periodic direction needs none — the circular correlation is the
sum wanted. A bounded one is padded past `2n - 1` so the circular correlation equals the linear one,
rounded up to a length the transform factors well.
"""
@inline _pad_dims(s::SFC.UniformLagSchedule{Dg}) where {Dg} =
    ntuple(d -> @inbounds(s.periodic[d]) ? @inbounds(s.dims[d]) :
                nextprod((2, 3, 5), 2 * @inbounds(s.dims[d]) - 1), Val(Dg))

"""Embed `src` at the origin of a zero array of size `P`."""
function _embed(src::AbstractArray{FT, Dg}, P::NTuple{Dg, Int}) where {FT, Dg}
    out = zeros(FT, P)
    out[map(n -> 1:n, size(src))...] = src
    return out
end

"""Column-major index of the pair `(a, b)`, `a ≤ b`, among the `D(D+1)/2` distinct components."""
@inline function _tri_index(D::Int, a::Int, b::Int)
    i, j = minmax(a, b)
    return (i - 1) * D - ((i - 1) * i) ÷ 2 + j
end

"""
    _increment_tensor_spectra(u, schedule, P) -> (tensor_arrays, D)

`T[a,b]` for every lag at once, as one array per distinct `(a, b)`.

`Σ_x δu_a δu_b` expands into four correlations of the padded field — `⟨m, u_a u_b⟩`, its reverse,
and the two `⟨u_a, u_b⟩` — each a product in the transform domain, so the whole tensor for all lags
costs `1 + D + D(D+1)/2` forward transforms and `D(D+1)/2` inverse ones. The mask is what makes the
padding inert: outside the real region it is zero, so no padded cell enters a sum.
"""
function _increment_tensor_spectra(
    u::AbstractArray{FT}, s::SFC.UniformLagSchedule{Dg}, P::NTuple{Dg, Int}, valid,
) where {FT, Dg}
    D = size(u, 1)
    cells = ntuple(d -> s.dims[d], Val(Dg))
    n = prod(cells)
    uf = reshape(u, D, n)
    F = plan_rfft(zeros(FT, P))

    # An empty cell contributes to nothing, so its field is replaced by zero rather than multiplied
    # by the mask: it may hold NaN, and NaN * 0 is NaN, which one transform would spread everywhere.
    w = reshape([valid[k] ? one(FT) : zero(FT) for k in 1:n], cells)
    held(a) = reshape([valid[k] ? uf[a, k] : zero(FT) for k in 1:n], cells)

    m̂ = F * _embed(w, P)
    p̂ = ntuple(a -> F * _embed(held(a), P), D)
    q̂ = [F * _embed(held(a) .* held(b), P) for a in 1:D for b in a:D]

    tensors = Vector{Array{FT, Dg}}(undef, D * (D + 1) ÷ 2)
    for a in 1:D, b in a:D
        k = _tri_index(D, a, b)
        spec = @. conj(m̂) * q̂[k] + conj(q̂[k]) * m̂ - conj(p̂[a]) * p̂[b] - conj(p̂[b]) * p̂[a]
        tensors[k] = irfft(spec, P[1])
    end
    # Pairs with both ends held, for every lag at once. Only needed where cells are missing: with a
    # complete field the count is exact arithmetic, and a transform would need rounding.
    pair_counts = valid isa SFC.AllValid ? nothing : irfft(@.(conj(m̂) * m̂), P[1])
    return tensors, pair_counts, D
end

"""Pairs a lag names, exactly: every cell on a wrapping direction, the overlap on a bounded one."""
@inline _lag_pair_count(s::SFC.UniformLagSchedule{Dg}, h::NTuple{Dg, Int}) where {Dg} =
    prod(ntuple(d -> @inbounds(s.periodic[d]) ? @inbounds(s.dims[d]) :
                     @inbounds(s.dims[d]) - abs(h[d]), Val(Dg)))

"""Position of lag `h` in a transform of size `P`, wrapping the negative offsets."""
@inline _lag_index(h::NTuple{Dg, Int}, P::NTuple{Dg, Int}) where {Dg} =
    CartesianIndex(ntuple(d -> mod(h[d], P[d]) + 1, Val(Dg)))

function SFC.gridded_sweep!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    u::AbstractArray, s::SFC.UniformLagSchedule{Dg, T}, dist_be, ::Val{D},
    ::SB.AbstractFastFourierTransformSpectralBackend; valid = SFC.AllValid(),
) where {OT, CT, Dg, T, D}
    SFC.is_quadratic_operator(sf) || throw(ArgumentError(
        "$(typeof(sf)) is not a quadratic form in δu, so it is not a contraction of the " *
        "second-order increment tensor the transform produces. Omit the spectral backend for " *
        "the lag sweep, which evaluates any pairwise operator exactly.",
    ))
    size(u)[2:end] == s.dims || throw(DimensionMismatch(
        "field cells $(size(u)[2:end]) do not match the schedule's dims $(s.dims)",
    ))

    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    P = _pad_dims(s)
    tensors, pair_counts, Du = _increment_tensor_spectra(u, s, P, valid)
    Du == D || throw(DimensionMismatch("field has $Du components, declared $D"))

    r_max = SFC._cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    lags = ntuple(d -> SFC.lag_range(s, d, r_max), Val(Dg))

    for H in CartesianIndices(lags)
        h = Tuple(H)
        all(iszero, h) && continue
        hn = ntuple(d -> SFC._lag_negate(s, d, h[d]), Val(Dg))
        h < hn && continue
        dx = SA.SVector{D, T}(ntuple(d -> d <= Dg ? T(h[d]) * s.spacing[d] : zero(T), Val(D)))
        r2 = sum(abs2, dx)
        b = SFC.squared_digitize(plan, r2)
        1 <= b <= nb || continue

        idx = _lag_index(h, P)
        Tm = SA.SMatrix{D, D, T}(
            (a <= D && bb <= D ? tensors[_tri_index(D, a, bb)][idx] : zero(T)
             for a in 1:D, bb in 1:D)...,
        )
        n_pairs = pair_counts === nothing ? _lag_pair_count(s, h) :
                  round(Int, pair_counts[idx])
        # A self-reverse lag's tensor sums each pair twice; `δu` flips between the two visits and the
        # tensor is even in it, so the sum is exactly doubled.
        if h == hn
            Tm = Tm / 2
            n_pairs ÷= 2
        end

        amb = SFC._ambiguous_dirs(s, h)
        val = _mean_over_images(sf, Tm, dx, r2, amb, Val(Dg))
        sums[b] += OT(val)
        counts[b] += CT(n_pairs)
    end
    return sums, counts
end

# The separation direction is not unique where a lag half-turns a periodic direction, so the
# contraction is averaged over the equal-length images, exactly as the lag sweep does.
@inline function _mean_over_images(sf, Tm, dx, r2, amb::NTuple{Dg, Bool}, ::Val{Dg}) where {Dg}
    go(vk) = begin
        images = SFC._lag_images(dx, amb, vk)
        acc = SFC.increment_tensor_value(sf, Tm, images[1] / sqrt(r2))
        for m in 2:length(images)
            acc += SFC.increment_tensor_value(sf, Tm, images[m] / sqrt(r2))
        end
        acc / length(images)
    end
    K = count(amb)
    K == 0 && return go(Val(0))
    K == 1 && return go(Val(1))
    K == 2 && return go(Val(2))
    K == 3 && return go(Val(3))
    return go(Val(K))
end

# `Auto` is answered here because only this extension knows what a transform would cost: the sweep
# visits `n_lags` lags over `M` cells, the transform pays `O(M_padded log M_padded)` per component
# pair however few lags are wanted. Dispatching on the concrete tag out-specialises the core method.
function SFC.gridded_sweep!(
    sums, counts, sf, u, s::SFC.UniformLagSchedule{Dg, T}, dist_be, ::Val{D},
    ::SB.AutoSpectralBackend; valid = SFC.AllValid(),
) where {Dg, T, D}
    SFC.is_quadratic_operator(sf) ||
        return SFC.gridded_lag_sweep!(sums, counts, sf, u, s, dist_be, Val(D); valid)
    r_max = SFC._cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    n_lags = prod(ntuple(d -> length(SFC.lag_range(s, d, r_max)), Val(Dg)))
    P = _pad_dims(s)
    transforms = 1 + D + D * (D + 1)          # forward and inverse together
    sweep_cost = n_lags * prod(s.dims)
    fft_cost = transforms * prod(P) * log2(max(2, prod(P)))
    return fft_cost < sweep_cost ?
        SFC.gridded_sweep!(sums, counts, sf, u, s, dist_be, Val(D),
                           SB.FastFourierTransformSpectralBackend(); valid) :
        SFC.gridded_lag_sweep!(sums, counts, sf, u, s, dist_be, Val(D); valid)
end

"""
    _lag_space_second_order(u, schedule, P, valid) -> Array

`⟨‖δu‖²⟩` at every lag, as an average over the pairs that lag actually holds.

The tensor arrays hold sums, so each is divided by its own pair count: with cells missing that count
is the mask autocorrelation and varies from lag to lag.
"""
function _lag_space_second_order(u, s::SFC.UniformLagSchedule{Dg, T}, P, valid) where {Dg, T}
    tensors, pair_counts, D = _increment_tensor_spectra(u, s, P, valid)
    trace = zeros(T, size(first(tensors)))
    for a in 1:D
        trace .+= tensors[_tri_index(D, a, a)]
    end
    out = similar(trace)
    @inbounds for I in CartesianIndices(trace)
        h = ntuple(d -> let m = Tuple(I)[d] - 1
                m > P[d] ÷ 2 ? m - P[d] : m
            end, Val(Dg))
        n = pair_counts === nothing ? _lag_pair_count(s, h) : round(Int, pair_counts[I])
        n <= 0 && throw(ArgumentError(
            "lag $h is named by no pair, so its structure function is undefined; the mask is too " *
            "sparse for a lag-space transform.",
        ))
        out[I] = trace[I] / n
    end
    return out
end

function SFC.gridded_spectrum(
    u::AbstractArray, s::SFC.UniformLagSchedule{Dg, T}, ::Val{D},
    ::SB.AbstractFastFourierTransformSpectralBackend; valid = SFC.AllValid(),
) where {Dg, T, D}
    all(s.periodic) || throw(ArgumentError(
        "a lag-space transform needs every direction periodic; a bounded direction has no natural " *
        "Fourier basis and its spectrum depends on a windowing choice this does not make.",
    ))
    size(u)[2:end] == s.dims || throw(DimensionMismatch(
        "field cells $(size(u)[2:end]) do not match the schedule's dims $(s.dims)",
    ))
    P = _pad_dims(s)                       # every direction periodic, so this is `dims`
    s2 = _lag_space_second_order(u, s, P, valid)

    # C(h) = C(0) - S₂(h)/2, and a constant transforms to k = 0 alone, so every other wavenumber
    # follows from S₂ by itself.
    ncells = prod(ntuple(d -> s.dims[d], Val(Dg)))
    modes = real.(AbstractFFTs.fft(s2)) .* (-inv(2 * ncells))

    dk = ntuple(d -> T(2π) / (s.dims[d] * s.spacing[d]), Val(Dg))
    density = modes ./ prod(dk)
    # k = 0 carries the mean and the variance, neither of which a structure function knows
    density[ntuple(_ -> 1, Val(Dg))...] = zero(T)

    wavenumbers = ntuple(d -> T(2π) .* AbstractFFTs.fftfreq(s.dims[d], 1 / s.spacing[d]), Val(Dg))
    return wavenumbers, density
end

end # module
