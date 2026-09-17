module StructureFunctionsNUFSHTExt

using NUFSHT: NUFSHT
using SpectralBackends: SpectralBackends as SB
using StructureFunctions: Calculations as SFC

# NUFSHT's spin-weighted analysis is the exact adjoint of its synthesis in the same convention the
# core's direct sum uses: `ₛY_lm = √((2l+1)/4π) d^l_{m,−s}(θ) e^{imφ}`, coefficients dense as
# `sf[l+1, m+lmax+1]`. So a provider is one spin plan per spin the operator's monomials need, built on
# the points once and reused for every monomial of that spin.
struct NUFSHTProvider{T, AV <: AbstractVector{T}, P}
    θ::AV
    φ::AV
    lmax::Int
    plans::Dict{Int, P}
end

function (p::NUFSHTProvider{T, AV, P})(f::AbstractVector, s::Integer) where {T, AV, P}
    plan = get!(p.plans, Int(s)) do
        NUFSHT.make_spin_plan(ComplexF64, p.θ, p.φ, p.lmax, Int(s))::P
    end
    out = zeros(ComplexF64, p.lmax + 1, 2p.lmax + 1)
    NUFSHT.nusht_type1_spin!(out, ComplexF64.(f), plan)
    return out
end

# The plan's type does not depend on the spin, and spin 0 is needed by every call (the mask), so it
# is built first and fixes the type of the rest.
function _nufsht_provider(θ, φ, lmax)
    θf, φf = convert(AbstractVector{Float64}, θ), convert(AbstractVector{Float64}, φ)
    p0 = NUFSHT.make_spin_plan(ComplexF64, θf, φf, Int(lmax), 0)
    return NUFSHTProvider(θf, φf, Int(lmax), Dict{Int, typeof(p0)}(0 => p0))
end

SFC.harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes::SFC.HarmonicNodes, ::Val{D}, ::Val{V},
                    ::Val{K}, ::SB.AbstractNonUniformFastSphericalHarmonicsTransformSpectralBackend;
                    valid = SFC.AllValid()) where {D, V, K} =
    SFC._harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes, Val(D), Val(V), Val(K), valid,
                         _nufsht_provider)

SFC.harmonic_spectra(x::AbstractMatrix, u::AbstractMatrix, lmax::Integer,
                     ::SB.AbstractNonUniformFastSphericalHarmonicsTransformSpectralBackend;
                     distance_metric = SFC.DI.SphericalAngle(), weights = nothing, valid = nothing) =
    SFC._harmonic_spectra(x, u, lmax, distance_metric, weights, valid, _nufsht_provider)

end # module
