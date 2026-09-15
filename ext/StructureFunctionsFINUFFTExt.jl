module StructureFunctionsFINUFFTExt

using FINUFFT: FINUFFT
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC

SFC._nufft_loaded(::Type{SFC.FINUFFTSpectralBackend}) = true

# FINUFFT's type-1 transform of complex strengths with `iflag = -1` is `Σ_j c_j e^{-i k·θ_j}` on the full
# mode grid, in FFT order under `modeord = 1`. Two real monomials ride in one complex transform,
# `c = μ₁ + i μ₂`, and their half spectra `k₁ ∈ 0:M₁÷2` are read off it as `F₁(k) = (F(k) + conj F(−k))/2`
# and `F₂(k) = (F(k) − conj F(−k))/2i`; every transform of the point set is one batched plan.
function SFC.nufft_monomial_transforms(
    tag::SFC.FINUFFTSpectralBackend, s::SFC.ScatteredModesSchedule{Dg, T},
    data::AbstractMatrix, valid, weights, keys, ::Val{Pm}; to = identity,
) where {Dg, T, Pm}
    FT = float(eltype(data))
    θ = ntuple(d -> to(FT.(SFC.mode_coordinates(s, d))), Val(Dg))
    N = size(data, 2)
    nkeys = length(keys)
    ntrans = cld(nkeys, 2)
    strengths = similar(θ[1], Complex{FT}, N, ntrans)
    for j in 1:ntrans
        a = SFC._held_monomial_vector(data, valid, weights, keys[2j - 1], FT)
        if 2j <= nkeys
            b = SFC._held_monomial_vector(data, valid, weights, keys[2j], FT)
            view(strengths, :, j) .= complex.(a, b)
        else
            view(strengths, :, j) .= complex.(a)
        end
    end
    full = similar(θ[1], Complex{FT}, s.modes..., ntrans)
    _type1!(full, strengths, θ, s.modes, tag.tolerance, KA.get_backend(θ[1]))
    H = s.modes[1] ÷ 2 + 1
    half = ntuple(d -> 1:(d == 1 ? H : s.modes[d]), Val(Dg))
    neg = ntuple(d -> to([i == 1 ? 1 : s.modes[d] - i + 2 for i in half[d]]), Val(Dg))
    taper = SFC.mode_taper_weights(s, FT, (H, Base.tail(s.modes)...), to)
    return map(1:nkeys) do n
        F = view(full, ntuple(_ -> Colon(), Val(Dg))..., cld(n, 2))
        Fneg = conj.(F[neg...])
        û = isodd(n) ? (F[half...] .+ Fneg) ./ 2 : (F[half...] .- Fneg) ./ (2im)
        taper === nothing || (û .*= taper)
        û
    end
end

function _type1!(full, strengths, θ, modes, tolerance, ::KA.CPU)
    plan = FINUFFT.finufft_makeplan(1, collect(Int64, modes), -1, size(strengths, 2), tolerance;
                                    dtype = eltype(θ[1]), modeord = 1)
    try
        FINUFFT.finufft_setpts!(plan, θ...)
        FINUFFT.finufft_exec!(plan, strengths, full)
    finally
        FINUFFT.finufft_destroy!(plan)
    end
    return full
end

# cuFINUFFT's names exist once `using CUDA` has loaded FINUFFT's device interface, so they are looked up
# here and never named in a signature.
function _type1!(full, strengths, θ, modes, tolerance, backend::KA.GPU)
    isdefined(FINUFFT, :cufinufft_makeplan) || throw(ArgumentError(
        "FINUFFT's device transform is cuFINUFFT, loaded by `using CUDA`; the points live on " *
        "$(nameof(typeof(backend))).",
    ))
    plan = FINUFFT.cufinufft_makeplan(1, collect(Int64, modes), -1, size(strengths, 2), tolerance;
                                      dtype = eltype(θ[1]), modeord = 1)
    try
        FINUFFT.cufinufft_setpts!(plan, θ...)
        FINUFFT.cufinufft_exec!(plan, strengths, full)
    finally
        FINUFFT.cufinufft_destroy!(plan)
    end
    return full
end

end # module
