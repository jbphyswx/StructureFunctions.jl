module StructureFunctionsFINUFFTExt

using FINUFFT: FINUFFT
using StructureFunctions: StructureFunctions as SF, Calculations as SFC

# FINUFFT's type-1 transform of complex strengths with `iflag = -1` is `Σ_j c_j e^{-i k·θ_j}` on the
# mode grid, in FFT order under `modeord = 1`. Two real monomials ride in one complex transform,
# `c = μ₁ + i μ₂`, and their half spectra are read off it as `F₁(k) = (F(k) + conj F(−k))/2` and
# `F₂(k) = (F(k) − conj F(−k))/2i`; every transform of the point set is one batched plan, borrowed from the
# workspace for every field on the same schedule.
function SFC.nufft_monomial_transforms(
    tag::SFC.FINUFFTSpectralBackend, s::SFC.ScatteredModesSchedule{Dg, T},
    data::AbstractMatrix, valid, weights, keys, ::Val{Pm}; to = identity, workspace = nothing,
) where {Dg, T, Pm}
    FT = float(eltype(data))
    N = size(data, 2)
    nkeys = length(keys)
    ntrans = cld(nkeys, 2)
    sizes = (:finufft, tag, s, FT, typeof(to(FT[])), ntrans)
    set = SFC._borrow!(workspace, sizes, () -> _points_plan(tag, s, FT, ntrans, to))
    try
        strengths = similar(set.θ[1], Complex{FT}, N, ntrans)
        for j in 1:ntrans
            a = SFC._held_monomial_vector(data, valid, weights, keys[2j - 1], FT)
            if 2j <= nkeys
                b = SFC._held_monomial_vector(data, valid, weights, keys[2j], FT)
                view(strengths, :, j) .= complex.(a, b)
            else
                view(strengths, :, j) .= complex.(a)
            end
        end
        full = similar(set.θ[1], Complex{FT}, set.grid..., ntrans)
        SFC.nufft_type1_exec!(set.plan, strengths, full)
        return map(1:nkeys) do n
            F = view(full, ntuple(_ -> Colon(), Val(Dg))..., cld(n, 2))
            Fp = view(F, set.pos...)
            Fm = view(F, set.neg...)
            taper = set.taper
            if isodd(n)
                return taper === nothing ? @.((Fp + conj(Fm)) / 2) : @.((Fp + conj(Fm)) * taper / 2)
            end
            return taper === nothing ? @.((Fp - conj(Fm)) / (2im)) : @.((Fp - conj(Fm)) * taper / (2im))
        end
    finally
        workspace === nothing ? SFC._release_plan!(set) : SFC._give_back!(workspace, sizes, set)
    end
end

# The transform is taken on the mode set closed under negation, `−⌊M/2⌋ … ⌊M/2⌋` per direction.
"""The schedule's batched plan of `ntrans` transforms in precision `FT` with its points set, the points, the
closed mode grid, the positions of each half-spectrum mode and of its negation in it, and the taper."""
function _points_plan(tag, s::SFC.ScatteredModesSchedule{Dg}, ::Type{FT}, ntrans::Int, to) where {Dg, FT}
    θ = ntuple(d -> to(FT.(SFC.mode_coordinates(s, d))), Val(Dg))
    grid = ntuple(d -> 2 * (s.modes[d] ÷ 2) + 1, Val(Dg))
    half = ntuple(d -> d == 1 ? s.modes[1] ÷ 2 + 1 : s.modes[d], Val(Dg))
    pos = ntuple(d -> to([_grid_index(SFC._mode_integer(i, s.modes[d], d == 1), grid[d]) for i in 1:half[d]]),
                 Val(Dg))
    neg = ntuple(d -> to([_grid_index(-SFC._mode_integer(i, s.modes[d], d == 1), grid[d]) for i in 1:half[d]]),
                 Val(Dg))
    return (; plan = SFC.nufft_type1_plan(tag, θ, grid, ntrans), θ, grid, pos, neg,
            taper = SFC.mode_taper_weights(s, FT, half, to))
end

"""Position of mode `k` in a grid of `n` modes in FFT order."""
@inline _grid_index(k::Int, n::Int) = k >= 0 ? k + 1 : k + n + 1

function SFC.nufft_type1_plan(tag::SFC.FINUFFTSpectralBackend, θ::Tuple{Vararg{Array}}, modes, ntrans::Int)
    plan = FINUFFT.finufft_makeplan(1, collect(Int64, modes), -1, ntrans, tag.tolerance;
                                    dtype = eltype(θ[1]), modeord = 1)
    try
        FINUFFT.finufft_setpts!(plan, θ...)
    catch
        FINUFFT.finufft_destroy!(plan)
        rethrow()
    end
    return plan
end

SFC.nufft_type1_exec!(plan::FINUFFT.finufft_plan, strengths, full) = (FINUFFT.finufft_exec!(plan, strengths, full); full)
SFC._release_plan!(plan::FINUFFT.finufft_plan) = (FINUFFT.finufft_destroy!(plan); nothing)

end # module
