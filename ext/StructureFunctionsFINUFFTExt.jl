module StructureFunctionsFINUFFTExt

using FINUFFT: FINUFFT
using StructureFunctions: StructureFunctions as SF, Calculations as SFC

# FINUFFT's type-1 transform of complex strengths with `iflag = -1` is `Σ_j c_j e^{-i k·θ_j}` on the
# mode grid, in FFT order under `modeord = 1`. Two real monomials ride in one complex transform,
# `c = μ₁ + i μ₂`, and their half spectra are read off it as `F₁(k) = (F(k) + conj F(−k))/2` and
# `F₂(k) = (F(k) − conj F(−k))/2i`; every transform of the point set is one batched plan.
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
    # Unpacking a mode needs its negation, so the transform is taken on the mode set closed under
    # negation, `−⌊M/2⌋ … ⌊M/2⌋`: the schedule's own set when a direction holds an odd number of modes,
    # one mode more when it holds an even number, whose `−M/2` the schedule's set lacks.
    grid = ntuple(d -> 2 * (s.modes[d] ÷ 2) + 1, Val(Dg))
    full = similar(θ[1], Complex{FT}, grid..., ntrans)
    SFC.nufft_type1!(tag, full, strengths, θ, grid)
    half = ntuple(d -> d == 1 ? s.modes[1] ÷ 2 + 1 : s.modes[d], Val(Dg))
    pos = ntuple(d -> to([_grid_index(SFC._mode_integer(i, s.modes[d], d == 1), grid[d]) for i in 1:half[d]]),
                 Val(Dg))
    neg = ntuple(d -> to([_grid_index(-SFC._mode_integer(i, s.modes[d], d == 1), grid[d]) for i in 1:half[d]]),
                 Val(Dg))
    taper = SFC.mode_taper_weights(s, FT, half, to)
    return map(1:nkeys) do n
        F = view(full, ntuple(_ -> Colon(), Val(Dg))..., cld(n, 2))
        Fp = F[pos...]
        Fm = conj.(F[neg...])
        û = isodd(n) ? (Fp .+ Fm) ./ 2 : (Fp .- Fm) ./ (2im)
        taper === nothing || (û .*= taper)
        û
    end
end

"""Position of mode `k` in a grid of `n` modes in FFT order."""
@inline _grid_index(k::Int, n::Int) = k >= 0 ? k + 1 : k + n + 1

function SFC.nufft_type1!(tag::SFC.FINUFFTSpectralBackend, full::Array, strengths::Array, θ::Tuple{Vararg{Array}}, modes)
    plan = FINUFFT.finufft_makeplan(1, collect(Int64, modes), -1, size(strengths, 2), tag.tolerance;
                                    dtype = eltype(θ[1]), modeord = 1)
    try
        FINUFFT.finufft_setpts!(plan, θ...)
        FINUFFT.finufft_exec!(plan, strengths, full)
    finally
        FINUFFT.finufft_destroy!(plan)
    end
    return full
end

end # module
