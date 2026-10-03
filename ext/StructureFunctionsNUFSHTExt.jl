module StructureFunctionsNUFSHTExt

using NUFSHT: NUFSHT
using SpectralBackends: SpectralBackends as SB
using StructureFunctions: Calculations as SFC

# NUFSHT's spin-weighted analysis is the exact adjoint of its synthesis in the same convention the
# core's direct sum uses: `ₛY_lm = √((2l+1)/4π) d^l_{m,−s}(θ) e^{imφ}`, coefficients dense as
# `sf[l+1, m+lmax+1]`. So a provider is one spin plan per spin the operator's monomials need, built on
# the points once and reused for every monomial of that spin. A plan built on device nodes runs on that
# device, with the coefficients there.
struct NUFSHTProvider{AV <: AbstractVector{Float64}, P}
    θ::AV
    φ::AV
    lmax::Int
    plans::Dict{Int, P}
end

function (p::NUFSHTProvider{AV, P})(f::AbstractVector, s::Integer) where {AV, P}
    plan = get!(p.plans, Int(s)) do
        NUFSHT.make_spin_plan(ComplexF64, p.θ, p.φ, p.lmax, Int(s))::P
    end
    out = similar(p.θ, ComplexF64, p.lmax + 1, 2p.lmax + 1)
    NUFSHT.nusht_type1_spin!(out, eltype(f) === ComplexF64 ? f : ComplexF64.(f), plan)
    return out
end

# The plan's type does not depend on the spin, and spin 0 is needed by every call (the mask), so it
# is built first and fixes the type of the rest.
function _nufsht_provider(θ::AbstractVector{Float64}, φ::AbstractVector{Float64}, lmax)
    p0 = NUFSHT.make_spin_plan(ComplexF64, θ, φ, Int(lmax), 0)
    return NUFSHTProvider(θ, φ, Int(lmax), Dict{Int, typeof(p0)}(0 => p0))
end

function SFC.harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes::SFC.HarmonicNodes, ::Val{D},
                             ::Val{V}, ::Val{K}, ::SB.AbstractNonUniformFastSphericalHarmonicsTransformSpectralBackend;
                             valid = SFC.AllValid(),
                             backend::SFC.CB.AbstractExecutionBackend = SFC.CB.AutoBackend()) where {D, V, K}
    SFC._require_backend(backend)
    SFC._require_device_outputs(backend, sums, counts)
    return SFC._harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes, Val(D), Val(V), Val(K), valid,
                                SFC._adaptor(backend), _nufsht_provider)
end

function SFC.harmonic_spectra(x::AbstractMatrix, u::AbstractMatrix, lmax::Integer,
                              ::SB.AbstractNonUniformFastSphericalHarmonicsTransformSpectralBackend;
                              distance_metric = SFC.DI.SphericalAngle(), weights = nothing, valid = nothing,
                              backend::SFC.CB.AbstractExecutionBackend = SFC.CB.AutoBackend())
    SFC._require_backend(backend)
    return SFC._harmonic_spectra(x, u, lmax, distance_metric, weights, valid, SFC._adaptor(backend),
                                 _nufsht_provider)
end

end # module
