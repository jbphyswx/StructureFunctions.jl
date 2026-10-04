module StructureFunctionsAbstractFFTsExt

using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, HelperFunctions as SFH
using SpectralBackends: SpectralBackends as SB

const CB = SFC.CB

function __init__()
    SFC._ABSTRACTFFTS_LOADED[] = true
    return nothing
end

"""Bytes of monomial scratch the forward stage holds while it transforms a batch of them."""
const FORWARD_BATCH_BYTES = Ref(1 << 25)

"""
    _pad_dims(schedule, r_max) -> NTuple{Du, Int}

Transform length per uniform direction. A periodic direction needs none — the circular correlation
is the sum wanted. A bounded one is padded to at least `n + h_max`, with `h_max` the largest lag any
slab pair reads within `r_max`, so the circular correlation equals the linear one at every lag that
is read; rounded up to a length the transform factors well.
"""
_pad_dims(s::SFC.AbstractSeparableSchedule, r_max) = _pad_dims(SFC.uniform_axes(s), SFC.lag_limits(s, r_max))

@inline function _pad_dims(su::SFC.UniformLagSchedule{Dg}, lims::NTuple{Dg, Int}) where {Dg}
    return ntuple(Val(Dg)) do d
        n = @inbounds su.dims[d]
        @inbounds(su.periodic[d]) && return n
        nextprod((2, 3, 5), n + min(n - 1, lims[d]))
    end
end

"""Embed `src` at the origin of a zero array of size `P`, in `src`'s array family."""
function _embed(src::AbstractArray{FT, Dg}, P::NTuple{Dg, Int}) where {FT, Dg}
    out = similar(src, P)
    # Only the padding is zeroed; a fully periodic schedule pads nothing.
    size(src) == P && return copyto!(out, src)
    fill!(out, zero(FT))
    view(out, map(n -> 1:n, size(src))...) .= src
    return out
end

"""
    MonomialTransforms(data, valid, weights, uniform, P, plan, ::Val{Pm})
    MonomialTransforms(data, uniform, P, valid, weights = NoWeights())

Forward transforms of the masked, weighted monomials of one slab of a packed field, built on first use
and kept, in the array family of `data`. A monomial is named by the sorted components it multiplies,
zero-padded to `Pm` entries; the all-zero key is the mask (times the weights), transformed at
construction. `hits` and `misses` count the lookups that found and built a transform.
"""
mutable struct MonomialTransforms{FT, Dg, T, Pm, DT <: AbstractMatrix, VT, WT, PL, AT <: AbstractArray{Complex{FT}, Dg}}
    data::DT
    valid::VT
    weights::WT
    schedule::SFC.UniformLagSchedule{Dg, T}
    P::NTuple{Dg, Int}
    plan::PL
    cache::Dict{NTuple{Pm, Int}, AT}
    hits::Int
    misses::Int
end

function MonomialTransforms(
    data::AbstractMatrix, valid, weights, s::SFC.UniformLagSchedule{Dg, T}, P::NTuple{Dg, Int}, plan, ::Val{Pm},
) where {Dg, T, Pm}
    FT = float(eltype(data))
    mask = ntuple(_ -> 0, Val(Pm))
    F0 = plan * _embed(_held_monomial(data, valid, weights, s.dims, mask, FT), P)
    cache = Dict{NTuple{Pm, Int}, typeof(F0)}(mask => F0)
    return MonomialTransforms{FT, Dg, T, Pm, typeof(data), typeof(valid), typeof(weights), typeof(plan), typeof(F0)}(
        data, valid, weights, s, P, plan, cache, 0, 1,
    )
end

MonomialTransforms(data::AbstractMatrix, s::SFC.UniformLagSchedule, P::Tuple, valid, weights = SFC.NoWeights()) =
    MonomialTransforms(data, valid, weights, s, P, AbstractFFTs.plan_rfft(_zeros_like(data, P)), Val(2))

"""`AbstractFFTs.plan_rfft` and `plan_irfft` of a plan `threads` threads execute (`SFC._fft_plan_options`)."""
_plan_rfft(x, region, threads::Int = 1) = AbstractFFTs.plan_rfft(x, region; SFC._fft_plan_options(x, threads)...)
_plan_irfft(x, d::Int, region, threads::Int = 1) =
    AbstractFFTs.plan_irfft(x, d, region; SFC._fft_plan_options(x, threads)...)

"""A zero array of shape `P` in `data`'s array family, in its float type."""
_zeros_like(data::AbstractArray, P::Tuple) =
    fill!(similar(parent(data), float(eltype(data)), P), zero(float(eltype(data))))

# The masked, weighted monomial of one slab, shaped as its cells.
_held_monomial(data::AbstractMatrix, valid, weights, dims::NTuple{Dg, Int}, key::NTuple{Pm, Int},
               ::Type{FT}) where {Dg, Pm, FT} =
    reshape(SFC._held_monomial_vector(data, valid, weights, key, FT), dims)

"""The forward transform of the masked, weighted monomial `key`, from the cache or built once."""
function monomial_transform!(mt::MonomialTransforms{FT, Dg, T, Pm}, key::Tuple) where {FT, Dg, T, Pm}
    k = SFC._padded_key(Val(Pm), key)
    cached = get(mt.cache, k, nothing)
    if cached !== nothing
        mt.hits += 1
        return cached
    end
    mt.misses += 1
    F = mt.plan * _embed(_held_monomial(mt.data, mt.valid, mt.weights, mt.schedule.dims, k, FT), mt.P)
    mt.cache[k] = F
    return F
end

"""Pairs with both ends held, at every lag of the padded grid: the two masks' cross-correlation."""
pair_counts(mtI::MonomialTransforms, mtJ::MonomialTransforms) =
    AbstractFFTs.irfft(conj.(monomial_transform!(mtI, ())) .* monomial_transform!(mtJ, ()), mtI.P[1])

pair_counts(mt::MonomialTransforms) = pair_counts(mt, mt)

"""
    moment_component(mtI, mtJ, j::NTuple{P, Int}) -> Array

`Σ_x m(x) m(x+h) Π_k δu[j_k]` at every lag `h` of the padded grid, for the sorted multi-index `j`,
with `x` in the first slab and `x + h` in the second.

Expanding each factor `δu = u(x+h) − u(x)` gives `2^P` cross-correlations of masked monomials, one per
choice of which factors sit at `x + h`; each is a product in the transform domain, and the whole sum
is inverted once.
"""
function moment_component(
    mtI::MonomialTransforms{FT, Dg}, mtJ::MonomialTransforms{FT, Dg}, j::NTuple{P, Int},
) where {FT, Dg, P}
    spec = fill!(similar(monomial_transform!(mtI, ())), zero(Complex{FT}))
    for mask in 0:((1 << P) - 1)
        primed = Tuple(j[k] for k in 1:P if (mask >> (k - 1)) & 1 == 1)
        unprimed = Tuple(j[k] for k in 1:P if (mask >> (k - 1)) & 1 == 0)
        sign = isodd(P - length(primed)) ? -one(FT) : one(FT)
        Fu = monomial_transform!(mtI, unprimed)
        Fp = monomial_transform!(mtJ, primed)
        @. spec += sign * conj(Fu) * Fp
    end
    return AbstractFFTs.irfft(spec, mtI.P[1])
end

moment_component(mt::MonomialTransforms, j::NTuple) = moment_component(mt, mt, j)

# Fill the columns for slab pair (I, J) from the two slabs' forward transforms and invert them all at
# once; returns the `(lags, columns)` matrix of raw moments.
function _pair_inverse!(scratch, fwdI::AbstractVector, fwdJ::AbstractVector, columns::AbstractVector)
    _fill_columns!(scratch.specf, fwdI, fwdJ, columns, eachindex(columns))
    LA.mul!(scratch.out, scratch.iplan, scratch.spec)
    return scratch.outf
end

"""Columns `cols` of the spectrum products of slabs `I` and `J`: column `c` is `Σ sign · conj(F_I[ki]) F_J[kj]` over
its terms."""
function _fill_columns!(specf, fwdI::AbstractVector, fwdJ::AbstractVector, columns::AbstractVector, cols)
    n = size(specf, 1)
    @inbounds for c in cols
        for lin in 1:n
            specf[lin, c] = 0
        end
        for (sign, ki, kj) in columns[c]
            FI = fwdI[ki]
            FJ = fwdJ[kj]
            @simd for lin in 1:n
                specf[lin, c] += sign * conj(FI[lin]) * FJ[lin]
            end
        end
    end
    return nothing
end

"""The scratch of one slab pair's inverse shared by `k` tasks: [`_inverse_plan`](@ref)'s, its plan run by `k`
threads, and the `ranges` of columns each task fills."""
_split_inverse_plan(eng, ncols::Int, k::Int) =
    () -> (; _inverse_plan(eng, ncols, k)()..., ranges = collect(Iterators.partition(1:ncols, cld(ncols, k))))

"""Slab pair `(I, J)`'s `(lags, columns)` raw moments, the tasks of `backend` filling the column ranges and the
plan's own threads inverting them."""
function _pair_inverse_split!(sc, fwdI::AbstractVector, fwdJ::AbstractVector, columns::AbstractVector, backend)
    SFC.sweep_foreach(backend, sc.ranges, Returns(nothing),
                      (cols, _) -> _fill_columns!(sc.specf, fwdI, fwdJ, columns, cols))
    LA.mul!(sc.out, sc.iplan, sc.spec)
    return sc.outf
end

SFC.transform_engine(sf, data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, vD::Val, vV::Val, vK::Val,
                     valid, weights, tag; to = identity, workspace = nothing, slice::NTuple{2, Int} = (1, 1)) =
    _transform_prepare(sf, data, s, dist_be, vD, vV, vK, valid, weights, tag; to,
                       forward = _kept_forward(workspace, slice...), workspace)

"""
    _transform_prepare(sf, data, schedule, distance_bins, ::Val{D}, ::Val{V}, ::Val{K}, valid, weights, tag; to, forward, workspace, stage_backend)

The engine's state for one field. `forward(dp, FT, layout) -> (spectra, all)` supplies the buffer the field's
spectra are written into, `all` being the array `spectra` is a slice of; the forward transforms run on
`stage_backend`, whose tasks each borrow a forward stage from `workspace`, and a non-uniform FFT borrows its plan
there. `layout` is the spectra's ([`_forward_layout`](@ref)), `nothing` for a non-uniform FFT's.
"""
function _transform_prepare(
    sf, data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    valid, weights, tag; to = identity, forward = _kept_forward(nothing, 1, 1), workspace = nothing,
    stage_backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {D, V, K}
    SFT.is_polynomial_operator(sf) || throw(ArgumentError(
        "$(typeof(sf)) is not a polynomial in the increment, so the transform cannot produce it. " *
        "Omit the spectral backend for the lag sweep, which evaluates any pairwise operator exactly.",
    ))
    SFC._check_grid_field(sf, data, s, Val(D), Val(V), Val(K))
    su = SFC.uniform_axes(s)
    T = eltype(su.spacing)
    r_max = SFC._cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    P = _pad_dims(s, r_max)
    W = V * D + K
    p = SFT.order(sf)
    dp0, vp0, wp0 = SFC.separable_layout(s, data, valid, weights)
    dp = to(dp0)
    vp = vp0 isa SFC.AllValid ? vp0 : to(Vector{Bool}(vp0))
    wp = wp0 isa SFC.NoWeights ? wp0 : to(wp0)
    fwd, keys, spectra, layout = _slab_transforms(tag, s, dp, vp, wp, su, P, Val(W), Val(p); to, forward, workspace,
                                                  stage_backend)
    transport = SFC.lag_transport(s)
    columns, vN = SFC._sf_columns(sf, transport, Val(W), Val(p))
    # the count column: the two masks' correlation, the weighted pair mass, or a soft-binned kernel mass
    soft = SFC._soft_binned(s)
    masked = !(valid isa SFC.AllValid) || !(wp isa SFC.NoWeights) || soft
    masked && push!(columns, [(1, 1, 1)])
    weighted = (wp isa SFC.NoWeights && !soft) ? Val(false) : Val(true)
    return (; s, su, P, r_max, fwd, spectra, layout, columns, masked, weighted, transport, vW = Val(W), vP = Val(p), vN)
end

"""
    _kept_forward(workspace, t, nt)

The `forward` source of slice `t` of `nt`: the spectra of every slice as one `(blk, nchunks, ngroups, nt)` array,
kept in `workspace` — allocated for this call without one.
"""
_kept_forward(workspace, t::Int, nt::Int) = (dp, FT, lay) -> begin
    whole = SFC._kept!(workspace, :spectra, (_array_family(dp), FT, lay.blk, lay.nchunks, lay.ngroups, nt),
                       () -> similar(parent(dp), Complex{FT}, lay.blk, lay.nchunks, lay.ngroups, nt))
    (view(whole, :, :, :, t), whole)
end

"""The kind of array `similar(parent(dp), …)` builds, whatever wraps `dp`."""
_array_family(dp) = typeof(similar(parent(dp), Bool, 0))

"""
    _with_workspace(f, workspace)

`f(ws)` with the caller's workspace, or with one this call owns, whose transform plans are finalized when the
call returns.
"""
_with_workspace(f, workspace::SFC.TransformWorkspace) = f(workspace)
function _with_workspace(f, ::Nothing)
    ws = SFC.TransformWorkspace()
    try
        return f(ws)
    finally
        SFC._release_plans!(ws)
    end
end

SFC._release_plan!(p::AbstractFFTs.ScaledPlan) = SFC._release_plan!(p.p)
SFC._release_plan!(p::AbstractFFTs.Plan) = finalize(p)

"""
    _lent_forward(workspace, lent)

The `forward` source of an executor sweeping one slice at a time: one slice's spectra borrowed from `workspace`'s
pool, recorded in `lent[]` for [`_return_forward!`](@ref).
"""
_lent_forward(workspace, lent::Base.RefValue) = (dp, FT, lay) -> begin
    sizes = (:forward, _array_family(dp), FT, lay.blk, lay.nchunks, lay.ngroups)
    set = SFC._borrow!(workspace, sizes, () -> _kept_forward(nothing, 1, 1)(dp, FT, lay))
    lent[] = sizes => set
    set
end

_return_forward!(workspace, lent::Base.RefValue) =
    lent[] === nothing || SFC._give_back!(workspace, first(lent[]), last(lent[]))

"""Bytes each block of forward spectra starts on: the widest alignment an FFT library plans a transform for."""
const SPECTRA_ALIGNMENT = 64

"""
    _forward_layout(P, nslabs, nkeys, FT, tasks) -> (; half, L, chunk, nchunks, last_nb, group, ngroups, last_ng, blk)

The shape of a field's forward spectra: `nslabs` slabs of half-spectra `half` (`L` entries) for each of `nkeys`
monomials, in blocks of `blk` entries each holding a chunk of `chunk` slabs (the last of `nchunks` holding
`last_nb`) for each of a group of `group` monomials (the last of `ngroups` holding `last_ng`), slab fastest.
A block holds as many (slab, monomial) transforms as `FORWARD_BATCH_BYTES` admits and as leave a block for each of
`tasks` tasks: every monomial of a chunk of slabs, or one slab's monomials in groups when fewer fit together.
"""
function _forward_layout(P::NTuple, nslabs::Int, nkeys::Int, ::Type{FT}, tasks::Int) where {FT}
    half = (P[1] ÷ 2 + 1, Base.tail(P)...)
    L = prod(half)
    fit = FORWARD_BATCH_BYTES[] ÷ max(prod(P) * sizeof(FT) + L * sizeof(Complex{FT}), 1)
    units = clamp(min(fit, cld(nslabs * nkeys, tasks)), 1, nslabs * nkeys)
    chunk, group = units >= nkeys ? (min(nslabs, units ÷ nkeys), nkeys) : (1, units)
    nchunks, ngroups = cld(nslabs, chunk), cld(nkeys, group)
    blk = cld(chunk * group * L * sizeof(Complex{FT}), SPECTRA_ALIGNMENT) * SPECTRA_ALIGNMENT ÷ sizeof(Complex{FT})
    return (; half, L, chunk, nchunks, last_nb = nslabs - (nchunks - 1) * chunk, group, ngroups,
            last_ng = nkeys - (ngroups - 1) * group, blk)
end

# The forward stage's scratch: the padded monomials of one block, the plans of a full block and of the last
# block, which alone may hold fewer transforms, and the monomial keys. A plan applies to a view of `held` with
# the strides it was planned for.
function _forward_stage(dp, ::Type{FT}, P::NTuple{Dg}, lay, keys) where {FT, Dg}
    colons = ntuple(_ -> Colon(), Val(Dg))
    n, n_last = lay.chunk * lay.group, lay.last_nb * lay.last_ng
    held = similar(parent(dp), FT, P..., n)
    full = view(held, colons..., 1:n)
    last = view(held, colons..., 1:n_last)
    plan = _plan_rfft(full, 1:Dg)
    plan_last = n_last == n ? plan : _plan_rfft(last, 1:Dg)
    return (; held, full, last, plan, plan_last, keys = copyto!(similar(parent(dp), eltype(keys), length(keys)), keys))
end

# Every monomial of degree ≤ Pm of every slab, transformed; the first key is the mask. Each block of the
# spectra is built in one broadcast and transformed in one batch straight into its place, the blocks shared out
# over the tasks of `stage_backend`. A scattered schedule's single slab is transformed by the non-uniform FFT
# provider.
function _slab_transforms(
    ::SB.AbstractFastFourierTransformSpectralBackend, s::SFC.AbstractSeparableSchedule, dp, vp, wp, su, P,
    ::Val{W}, ::Val{Pm}; to, forward, workspace, stage_backend,
) where {W, Pm}
    keys = SFC._monomial_keys(Val(W), Val(Pm))
    FT = float(eltype(dp))
    nslabs = SFC.n_slabs(s)
    lay = _forward_layout(P, nslabs, length(keys), FT, SFC.sweep_tasks(stage_backend))
    spectra, whole = forward(dp, FT, lay)
    make_stage, done = SFC._executor_scratch(workspace, (:forward_stage, _array_family(dp), FT, P, lay.chunk,
                                                         lay.last_nb, lay.group, lay.last_ng, keys),
                                             () -> _forward_stage(dp, FT, P, lay, keys))
    try
        _fill_spectra!(spectra, make_stage, dp, vp, wp, su, P, lay, stage_backend)
    finally
        done()
    end
    fwd = [[_slab_spectrum(spectra, lay, I, k) for k in eachindex(keys)] for I in 1:nslabs]
    return fwd, keys, whole, lay
end

"""Slab `I`'s half-spectrum of monomial `k` in the forward spectra of layout `lay`."""
@inline function _slab_spectrum(spectra, lay, I::Int, k::Int)
    o, c, s = SFC._slab_offsets(lay, I)
    g, q = SFC._key_group(lay, k)
    return reshape(view(spectra, (o + q * s) .+ (1:lay.L), c, g), lay.half)
end

"""
The masked, weighted monomial at `I = (position in a padded slab..., slab, monomial)` of a block holding slabs from
`lo` for each monomial from `keys[k0]`: `w_k Π_c data[c, k]` in a cell, selected by the mask (an empty cell may hold
NaN), and zero in the padding.
"""
@inline function _block_monomial(I::CartesianIndex, dp, vp, wp, keys, dims::NTuple{Dg, Int}, lo::Int, k0::Int,
                                 ::Type{FT}) where {Dg, FT}
    t = Tuple(I)
    pos = ntuple(d -> t[d], Val(Dg))
    all(map(<=, pos, dims)) || return zero(FT)
    col = LinearIndices(dims)[pos...] + (lo + t[Dg + 1] - 2) * prod(dims)
    (vp isa SFC.AllValid || @inbounds(vp[col])) || return zero(FT)
    v = one(FT)
    for c in @inbounds(keys[k0 + t[Dg + 2] - 1])
        c == 0 || (v *= @inbounds(dp[c, col]))
    end
    return wp isa SFC.NoWeights ? v : v * @inbounds(wp[col])
end

function _fill_spectra!(spectra, make_stage, dp, vp, wp, su, P, lay, backend)
    blocks = vec([(c, g) for c in 1:lay.nchunks, g in 1:lay.ngroups])
    SFC.sweep_foreach(backend, blocks, make_stage,
                      (cg, stage) -> _fill_block!(spectra, stage, dp, vp, wp, su, P, lay, cg[1], cg[2]))
    return nothing
end

"""Block `(c, g)` of the forward spectra: chunk `c` of slabs for group `g` of monomials, built in `stage`."""
function _fill_block!(spectra, stage, dp, vp, wp, su, P, lay, c::Int, g::Int)
    FT = eltype(stage.held)
    nb = c == lay.nchunks ? lay.last_nb : lay.chunk
    ng = g == lay.ngroups ? lay.last_ng : lay.group
    n = nb * ng
    held = reshape(view(vec(stage.held), 1:(n * prod(P))), P..., nb, ng)
    held .= _block_monomial.(CartesianIndices(held), Ref(dp), Ref(vp), Ref(wp), Ref(stage.keys), Ref(su.dims),
                             (c - 1) * lay.chunk + 1, (g - 1) * lay.group + 1, FT)
    block = reshape(view(spectra, 1:(n * lay.L), c, g), lay.half..., n)
    n == lay.chunk * lay.group ? LA.mul!(block, stage.plan, stage.full) : LA.mul!(block, stage.plan_last, stage.last)
    return nothing
end

function _slab_transforms(
    tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend, s::SFC.ScatteredModesSchedule, dp, vp, wp, su, P,
    ::Val{W}, ::Val{Pm}; to, forward, workspace, stage_backend,
) where {W, Pm}
    keys = SFC._monomial_keys(Val(W), Val(Pm))
    return [SFC.nufft_monomial_transforms(tag, s, dp, vp, wp, keys, Val(Pm); to, workspace)], keys, nothing, nothing
end

# The per-executor scratch for `ncols` columns, in the transforms' array family: the buffers and the
# batched inverse plan, run by `threads` threads, that fills them. An FFTW plan holds a pointer into the process
# that created it, so it is built here, by whichever process runs the work, and never sent to another one.
function _inverse_plan(eng, ncols::Int, threads::Int = 1)
    F1 = eng.fwd[1][1]
    CT = eltype(F1)
    FT = real(CT)
    P = eng.P
    Ph = size(F1)
    # Neither buffer is zeroed: `_pair_inverse!` writes every element of `spec` before reading it,
    # and the inverse transform writes `out` whole.
    return () -> begin
        spec = similar(F1, Ph..., ncols)
        out = similar(F1, FT, P..., ncols)
        iplan = _plan_irfft(spec, P[1], 1:length(P), threads)
        (iplan = iplan, spec = spec, specf = reshape(spec, :, ncols),
         out = out, outf = reshape(out, :, ncols))
    end
end

"""The `(make, done)` scratch source of a sweep's executors for `ncols` columns (see `SFC._executor_scratch`)."""
_inverse_scratch(eng, ncols::Int, workspace, backend) =
    SFC._executor_scratch(workspace, backend, (:inverse, typeof(eng.fwd[1][1]), size(eng.fwd[1][1]), eng.P, ncols),
                          _inverse_plan(eng, ncols))

function _transform_item!(
    sums::AbstractArray, counts::AbstractArray, sf, eng, item::NTuple{4, Int}, scratch, plan,
    nb, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N},
) where {D, V, K, W, Po, N}
    out = _pair_inverse!(scratch, eng.fwd[item[1]], eng.fwd[item[2]], eng.columns)
    return _transform_lags!(sums, counts, sf, eng, out, item, plan, nb,
                            Val(D), Val(V), Val(K), Val(W), Val(Po), Val(N))
end

# The lag loop, fed either a whole pair (`part = n_parts = 1`) or the `part`-th share of that
# pair's lags. A share reads `out` and never writes it, so the pair's inverse is computed once and
# the shares run concurrently over it.
function _transform_lags!(
    sums::AbstractArray, counts::AbstractArray, sf, eng, out, item::NTuple{4, Int}, plan,
    nb, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N},
) where {D, V, K, W, Po, N}
    I, J, part, n_parts = item
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    lags = SFC._pair_lags(s, su, I, J, eng.r_max)
    @inbounds for li in part:n_parts:length(lags)
        h = Tuple(lags[li])
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, size(out, 2), su, h, self_reverse)
        scale = self_reverse ? T(0.5) : one(T)
        inv_r = inv(sqrt(r2))
        val = SFC.with_frames(tr, geometry) do frames
            SFC._lag_value(tr, sf, out, idx, scale, frames, inv_r, Val(W), Val(Po), Val(N), Val(V), Val(K))
        end
        SFC._bin_add!(SFC._plain_add!, sums, counts, factor * val, n_pairs, b)
    end
    return nothing
end

function _transform_item!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, sf, eng, item::NTuple{4, Int}, scratch, plan,
    nb, axis_edges, na, second_axis, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N},
) where {OT, CT, D, V, K, W, Po, N}
    out = _pair_inverse!(scratch, eng.fwd[item[1]], eng.fwd[item[2]], eng.columns)
    return _transform_lags!(sums, counts, sf, eng, out, item, plan, nb, axis_edges, na, second_axis,
                            Val(D), Val(V), Val(K), Val(W), Val(Po), Val(N))
end

function _transform_lags!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, sf, eng, out, item::NTuple{4, Int}, plan,
    nb, axis_edges, na, second_axis, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N},
) where {OT, CT, D, V, K, W, Po, N}
    I, J, part, n_parts = item
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    lags = SFC._pair_lags(s, su, I, J, eng.r_max)
    @inbounds for li in part:n_parts:length(lags)
        h = Tuple(lags[li])
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, size(out, 2), su, h, self_reverse)
        scale = self_reverse ? T(0.5) : one(T)
        inv_r = inv(sqrt(r2))
        Mo = SFC._lag_moments(tr, out, idx, scale, nothing, nothing, Val(W), Val(Po), Val(N))
        SFC.with_frames(tr, geometry) do frames
            Mi = length(frames)
            for f in frames
                bθ = SFH.digitize(SFC.axis_quantity(second_axis, f.dir, r2), axis_edges)
                1 <= bθ <= na || continue
                sums[b, bθ] += OT(factor * SFT.moment_contract(sf, Mo, f.dir * inv_r, Val(V), Val(K)) / Mi)
                counts[b, bθ] += CT(n_pairs / Mi)
            end
        end
    end
    return nothing
end

function SFC.gridded_sweep!(
    sums::AbstractArray, counts::AbstractArray, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::SB.AbstractFastFourierTransformSpectralBackend;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
) where {D, V, K}
    SFC._require_backend(backend)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    SFC._check_hist_shape(sums, counts, sf, nb)
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    SFC._assert_grid_counts(s, counts, size(data, 2), w)
    _with_workspace(workspace) do ws
        _transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid, w, tag,
                          nothing, ws)
    end
    return sums, counts
end

function SFC.gridded_sweep!(
    sums::AbstractMatrix, counts::AbstractMatrix, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::SB.AbstractFastFourierTransformSpectralBackend;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
    second_axis::SFC.AbstractSecondAxisSource,
) where {D, V, K}
    SFC._require_backend(backend)
    _transform_axis(second_axis)
    SFC._require_directional(s)
    SFC._check_half_turn_counts(eltype(counts), s, dist_be)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    axis_edges = SFC.digitize_plan(axis_be)
    na = SFC.n_histogram_bins(axis_edges)
    size(sums) == (nb, na) && size(counts) == (nb, na) || throw(DimensionMismatch(
        "sums and counts must be ($nb, $na); got $(size(sums)) and $(size(counts))",
    ))
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    SFC._assert_grid_counts(s, counts, size(data, 2), w)
    _with_workspace(workspace) do ws
        _transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid, w, tag,
                          (axis_edges, na, second_axis), ws)
    end
    return sums, counts
end

# The non-uniform FFT route: the engine on a ScatteredModesSchedule, its forward transforms from a NUFFT
# provider. Counts are a kernel-weighted pair mass and need a floating-point type.

function SFC.gridded_sweep!(
    sums::AbstractArray, counts::AbstractArray{CT}, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.ScatteredModesSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
) where {CT, D, V, K}
    SFC._require_backend(backend)
    SFC._assert_mass_counts(CT)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    SFC._check_hist_shape(sums, counts, sf, nb)
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    _with_workspace(workspace) do ws
        _transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid, w, tag,
                          nothing, ws)
    end
    return sums, counts
end

function SFC.gridded_sweep!(
    sums::AbstractMatrix, counts::AbstractMatrix{CT}, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.ScatteredModesSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
    second_axis::SFC.AbstractSecondAxisSource,
) where {CT, D, V, K}
    SFC._require_backend(backend)
    _transform_axis(second_axis)
    SFC._assert_mass_counts(CT)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    axis_edges = SFC.digitize_plan(axis_be)
    na = SFC.n_histogram_bins(axis_edges)
    size(sums) == (nb, na) && size(counts) == (nb, na) || throw(DimensionMismatch(
        "sums and counts must be ($nb, $na); got $(size(sums)) and $(size(counts))",
    ))
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    _with_workspace(workspace) do ws
        _transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid, w, tag,
                          (axis_edges, na, second_axis), ws)
    end
    return sums, counts
end

"""Throw unless a transform can bin by `second_axis`: it produces each lag's moment sum, which has one
direction but not one value."""
_transform_axis(::SFC.SeparationAngleAxis) = nothing
_transform_axis(::SFC.InvariantValueAxis) = throw(ArgumentError(
    "a transform produces the sum of each lag's increment moments, not the values of the lag's pairs, so it " *
    "cannot bin pairs by value; DirectSumSpectralBackend() or AutoSpectralBackend() sweeps the lags and bins " *
    "each pair's value.",
))

"""Whether `Auto` transforms a joint histogram over `second_axis`: never by value, which needs the pairs."""
_auto_joint_transform(::SFC.InvariantValueAxis, sf, s, dist_be, W, valid, weights, backend) = false
_auto_joint_transform(::SFC.SeparationAngleAxis, sf, s, dist_be, W, valid, weights, backend) =
    _auto_transform(sf, s, dist_be, W, valid, weights, backend)

_scattered_needs_nufft(tag) = throw(ArgumentError(
    "a ScatteredModesSchedule sums its pairs by non-uniform FFT, which $(nameof(typeof(tag))) does not name; pass " *
    "NonuniformFFTsSpectralBackend() or FINUFFTSpectralBackend(). Auto never selects the soft-binned route.",
))
_nufft_needs_scattered(s) = throw(ArgumentError(
    "a non-uniform FFT tag is for a ScatteredModesSchedule; a $(nameof(typeof(s))) transforms with " *
    "FastFourierTransformSpectralBackend().",
))

for tagT in (:(SB.AbstractFastFourierTransformSpectralBackend), :(SB.AutoSpectralBackend))
    @eval begin
        SFC.gridded_sweep!(::AbstractArray, ::AbstractArray, ::SFT.AbstractPairwiseStructureFunctionType,
                           ::AbstractMatrix, ::SFC.ScatteredModesSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
                           tag::$tagT; kwargs...) where {D, V, K} = _scattered_needs_nufft(tag)
        SFC.gridded_sweep!(::AbstractMatrix, ::AbstractMatrix, ::SFT.AbstractPairwiseStructureFunctionType,
                           ::AbstractMatrix, ::SFC.ScatteredModesSchedule, dist_be, axis_be, ::Val{D}, ::Val{V},
                           ::Val{K}, tag::$tagT; kwargs...) where {D, V, K} = _scattered_needs_nufft(tag)
    end
end

SFC.gridded_sweep!(::AbstractArray, ::AbstractArray, ::SFT.AbstractPairwiseStructureFunctionType, ::AbstractMatrix,
                   s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
                   ::SB.AbstractNonUniformFastFourierTransformSpectralBackend; kwargs...) where {D, V, K} =
    _nufft_needs_scattered(s)
SFC.gridded_sweep!(::AbstractMatrix, ::AbstractMatrix, ::SFT.AbstractPairwiseStructureFunctionType, ::AbstractMatrix,
                   s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K},
                   ::SB.AbstractNonUniformFastFourierTransformSpectralBackend; kwargs...) where {D, V, K} =
    _nufft_needs_scattered(s)

# Batches over a trailing slice axis: one engine per slice, one geometry pass, the slices as the
# innermost loop of every lag.

# The per-executor scratch for `nt` slices of `ncols` columns and the batched inverse plan that
# fills it; `outs[t]` is the `(lags, columns)` matrix of slice `t`. Both buffers are written whole
# before they are read — every column of `specf`, then all of `out` by the transform — so neither is
# zeroed. The plan is built here for the same reason as in `_inverse_plan`.
function _inverse_plan_batch(eng, ncols::Int, nt::Int)
    F1 = eng.fwd[1][1]
    FT = real(eltype(F1))
    P = eng.P
    Ph = size(F1)
    return () -> begin
        spec = similar(F1, Ph..., ncols * nt)
        out = similar(F1, FT, P..., ncols * nt)
        out3 = reshape(out, :, ncols, nt)
        (iplan = _plan_irfft(spec, P[1], 1:length(P)),
         spec = spec, specf = reshape(spec, :, ncols, nt), out = out,
         outs = [view(out3, :, :, t) for t in 1:nt])
    end
end

_inverse_scratch_batch(eng, ncols::Int, nt::Int, workspace, backend) =
    SFC._executor_scratch(workspace, backend,
                          (:inverse_batch, typeof(eng.fwd[1][1]), size(eng.fwd[1][1]), eng.P, ncols, nt),
                          _inverse_plan_batch(eng, ncols, nt))

# Fill every slice's columns for slab pair (I, J) and invert them all at once; returns one
# `(lags, columns)` matrix per slice.
function _pair_inverse_batch!(scratch, engs::AbstractVector, I::Int, J::Int, columns::AbstractVector)
    specf = scratch.specf
    n = size(specf, 1)
    @inbounds for t in eachindex(engs)
        fwdI = engs[t].fwd[I]
        fwdJ = engs[t].fwd[J]
        for (c, terms) in enumerate(columns)
            for lin in 1:n
                specf[lin, c, t] = 0
            end
            for (sign, ki, kj) in terms
                FI = fwdI[ki]
                FJ = fwdJ[kj]
                @simd for lin in 1:n
                    specf[lin, c, t] += sign * conj(FI[lin]) * FJ[lin]
                end
            end
        end
    end
    LA.mul!(scratch.out, scratch.iplan, scratch.spec)
    return scratch.outs
end

function _transform_item_batch!(
    sums::AbstractArray, counts::AbstractArray, sf, engs, item::NTuple{4, Int}, scratch, plan,
    nb, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N},
) where {D, V, K, W, Po, N}
    I, J = item[1], item[2]
    eng = engs[1]
    outs = _pair_inverse_batch!(scratch, engs, I, J, eng.columns)
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    ncol = length(eng.columns)
    @inbounds for H in SFC._pair_lags(s, su, I, J, eng.r_max)
        h = Tuple(H)
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        scale = self_reverse ? T(0.5) : one(T)
        inv_r = inv(sqrt(r2))
        SFC.with_frames(tr, geometry) do frames
            for t in eachindex(outs)
                out = outs[t]
                n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, ncol, su, h, self_reverse)
                val = SFC._lag_value(tr, sf, out, idx, scale, frames, inv_r, Val(W), Val(Po), Val(N), Val(V),
                                     Val(K))
                SFC._bin_add!(SFC._plain_add!, sums, counts, factor * val, n_pairs, b, t)
            end
        end
    end
    return nothing
end

function _transform_item_batch!(
    sums::AbstractArray{OT, 3}, counts::AbstractArray{CT, 3}, sf, engs, item::NTuple{4, Int}, scratch, plan,
    nb, axis_edges, na, second_axis, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N},
) where {OT, CT, D, V, K, W, Po, N}
    I, J = item[1], item[2]
    eng = engs[1]
    outs = _pair_inverse_batch!(scratch, engs, I, J, eng.columns)
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    ncol = length(eng.columns)
    @inbounds for H in SFC._pair_lags(s, su, I, J, eng.r_max)
        h = Tuple(H)
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        scale = self_reverse ? T(0.5) : one(T)
        inv_r = inv(sqrt(r2))
        SFC.with_frames(tr, geometry) do frames
            Mi = length(frames)
            bins = map(f -> SFH.digitize(SFC.axis_quantity(second_axis, f.dir, r2), axis_edges), frames)
            for t in eachindex(outs)
                out = outs[t]
                n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, ncol, su, h, self_reverse)
                Mo = SFC._lag_moments(tr, out, idx, scale, nothing, nothing, Val(W), Val(Po), Val(N))
                for m in 1:Mi
                    bθ = bins[m]
                    1 <= bθ <= na || continue
                    dir = frames[m].dir
                    sums[b, bθ, t] += OT(factor * SFT.moment_contract(sf, Mo, dir * inv_r, Val(V), Val(K)) / Mi)
                    counts[b, bθ, t] += CT(n_pairs / Mi)
                end
            end
        end
    end
    return nothing
end

# One engine per slice: the slices share the schedule, the padding and the column table, and differ
# only in the field the forward transforms are taken of; their spectra are the slices of one array kept in
# `workspace`.
function _slice_engines(sf, data::AbstractArray{<:Any, 3}, s, dist_be, vD::Val, vV::Val, vK::Val, valid, weights, tag,
                        workspace, stage_backend)
    nt = size(data, 3)
    return [_transform_prepare(sf, view(data, :, :, t), s, dist_be, vD, vV, vK, SFC._valid_slice(valid, t), weights,
                               tag; forward = _kept_forward(workspace, t, nt), workspace, stage_backend)
            for t in 1:nt]
end

"""The backend of this process that runs the engine's forward transforms and single-pair inverses."""
_stage_backend(b::CB.AbstractExecutionBackend) = b
_stage_backend(b::Union{CB.AbstractDistributedBackend, CB.AbstractMPIBackend}) = CB.local_backend(b)

function _transform_sweep_batch!(
    sums, counts, backend::CB.AbstractExecutionBackend, sf, data, s, dist_be, plan, nb, vD::Val, vV::Val,
    vK::Val, valid, weights, tag, axis, workspace,
)
    SFC.batch_shares_lag_geometry(s) ||
        return _transform_sweep_slices!(sums, counts, backend, sf, data, s, dist_be, plan, nb, vD, vV, vK,
                                        valid, weights, tag, axis, workspace)
    return _transform_sweep_fused!(sums, counts, backend, sf, data, s, dist_be, plan, nb, vD, vV, vK, valid,
                                   weights, tag, axis, workspace)
end

function _transform_sweep_fused!(
    sums, counts, backend::CB.AbstractExecutionBackend, sf, data, s, dist_be, plan, nb, ::Val{D}, ::Val{V},
    ::Val{K}, valid, weights, tag, axis, workspace,
) where {D, V, K}
    engs = _slice_engines(sf, data, s, dist_be, Val(D), Val(V), Val(K), valid, weights, tag, workspace,
                          _stage_backend(backend))
    eng = engs[1]
    make_scratch, done = _inverse_scratch_batch(eng, length(eng.columns), length(engs), workspace, backend)
    try
        items = SFC.sweep_items(s, eng.r_max, SFC.sweep_tasks(backend), false)
        body! = _item_body_batch(sf, engs, plan, nb, axis, Val(D), Val(V), Val(K), eng.vW, eng.vP, eng.vN)
        SFC.sweep_reduce!(sums, counts, backend, items, make_scratch, body!)
    finally
        done()
    end
    return nothing
end

# One slice per work item, each with the engine and inverse scratch of a single slice: the
# arrangement a schedule whose lag geometry is a displacement and a bin takes. Slices are
# independent and write disjoint output columns, so any backend can execute them, and each item
# borrows its buffers for the length of its slice. With fewer slices than tasks, the slices run one
# after another, each on every task.
function _transform_sweep_slices!(
    sums, counts, backend::CB.AbstractExecutionBackend, sf, data, s, dist_be, plan, nb, vD::Val, vV::Val,
    vK::Val, valid, weights, tag, axis, workspace,
)
    ws = SFC._local_workspace(workspace, backend)
    nt = size(data, 3)
    if nt < SFC.sweep_tasks(backend)
        for t in 1:nt
            _slice_sweep!(_slice_out(sums, t), _slice_out(counts, t), sf, view(data, :, :, t), s, dist_be, plan, nb,
                          vD, vV, vK, SFC._valid_slice(valid, t), weights, tag, axis, ws, backend)
        end
        return nothing
    end
    body! = (ls, lc, t, _) -> _slice_sweep!(_slice_out(ls, t), _slice_out(lc, t), sf, view(data, :, :, t), s,
                                            dist_be, plan, nb, vD, vV, vK, SFC._valid_slice(valid, t), weights,
                                            tag, axis, ws, CB.SerialBackend())
    SFC.sweep_reduce!(sums, counts, backend, 1:nt, () -> nothing, body!)
    return nothing
end

function _slice_sweep!(sums, counts, sf, data, s, dist_be, plan, nb, vD, vV, vK, valid, weights, tag, axis,
                       workspace, backend)
    lent = Ref{Any}(nothing)
    try
        _transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, vD, vV, vK, valid,
                          weights, tag, axis, workspace; forward = _lent_forward(workspace, lent))
    finally
        _return_forward!(workspace, lent)
    end
    return nothing
end

@inline _slice_out(a::AbstractMatrix, t::Int) = view(a, :, t)
@inline _slice_out(a::AbstractArray{<:Any, 3}, t::Int) = view(a, :, :, t)

_item_body_batch(sf, engs, plan, nb, ::Nothing, vD, vV, vK, vW, vP, vN) =
    (ls, lc, it, scratch) -> _transform_item_batch!(ls, lc, sf, engs, it, scratch, plan, nb, vD, vV, vK, vW,
                                                    vP, vN)

_item_body_batch(sf, engs, plan, nb, axis::Tuple, vD, vV, vK, vW, vP, vN) =
    (ls, lc, it, scratch) -> _transform_item_batch!(ls, lc, sf, engs, it, scratch, plan, nb, axis[1],
                                                    axis[2], axis[3], vD, vV, vK, vW, vP, vN)

_transform_sweep_batch!(
    sums, counts, backend::CB.AbstractGPUBackend, sf, data, s, dist_be, plan, nb, vD::Val, vV::Val, vK::Val,
    valid, weights, tag, axis, workspace,
) = SFC.device_transform_sweep_batch!(sums, counts, backend, sf, data, s, dist_be, plan, nb, vD, vV, vK, valid,
                                      weights, tag, axis, workspace)

const BatchTransformTag = Union{SB.AbstractFastFourierTransformSpectralBackend,
                                SB.AbstractNonUniformFastFourierTransformSpectralBackend}

function SFC.gridded_sweep_batch!(
    sums::AbstractArray, counts::AbstractArray, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractArray{<:Any, 3}, s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::BatchTransformTag;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
) where {D, V, K}
    SFC._require_backend(backend)
    _batch_tag_schedule(tag, s)
    nt = SFC._check_batch(sf, data, s, valid, Val(D), Val(V), Val(K))
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    SFC._check_hist_shape(sums, counts, sf, nb, nt)
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    SFC._assert_grid_counts(s, counts, size(data, 2), w)
    _with_workspace(workspace) do ws
        _transform_sweep_batch!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid,
                                w, tag, nothing, ws)
    end
    return sums, counts
end

function SFC.gridded_sweep_batch!(
    sums::AbstractArray{<:Any, 3}, counts::AbstractArray{CT, 3}, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractArray{<:Any, 3}, s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V},
    ::Val{K}, tag::BatchTransformTag;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
    second_axis::SFC.AbstractSecondAxisSource,
) where {CT, D, V, K}
    SFC._require_backend(backend)
    _transform_axis(second_axis)
    _batch_tag_schedule(tag, s)
    SFC._require_directional(s)
    SFC._check_half_turn_counts(CT, s, dist_be)
    nt = SFC._check_batch(sf, data, s, valid, Val(D), Val(V), Val(K))
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    axis_edges = SFC.digitize_plan(axis_be)
    na = SFC.n_histogram_bins(axis_edges)
    size(sums) == (nb, na, nt) && size(counts) == (nb, na, nt) || throw(DimensionMismatch(
        "sums and counts must be ($nb, $na, $nt); got $(size(sums)) and $(size(counts))",
    ))
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    SFC._assert_grid_counts(s, counts, size(data, 2), w)
    _with_workspace(workspace) do ws
        _transform_sweep_batch!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid,
                                w, tag, (axis_edges, na, second_axis), ws)
    end
    return sums, counts
end

# The transform a batch runs with: a grid's slabs take an FFT and a scattered mode set a non-uniform one.
_batch_tag_schedule(::SB.AbstractFastFourierTransformSpectralBackend, ::SFC.AbstractSeparableSchedule) = nothing
_batch_tag_schedule(tag::SB.AbstractFastFourierTransformSpectralBackend, ::SFC.ScatteredModesSchedule) =
    _scattered_needs_nufft(tag)
_batch_tag_schedule(::SB.AbstractNonUniformFastFourierTransformSpectralBackend, ::SFC.ScatteredModesSchedule) = nothing
_batch_tag_schedule(::SB.AbstractNonUniformFastFourierTransformSpectralBackend, s::SFC.AbstractSeparableSchedule) =
    _nufft_needs_scattered(s)

function SFC.gridded_sweep_batch!(
    sums::AbstractArray, counts::AbstractArray, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractArray{<:Any, 3}, s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::SB.AutoSpectralBackend; valid = SFC.AllValid(), weights = nothing,
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(), workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
) where {D, V, K}
    return _auto_transform(sf, s, dist_be, V * D + K, valid, weights, backend) ?
        SFC.gridded_sweep_batch!(sums, counts, sf, data, s, dist_be, Val(D), Val(V), Val(K),
                                 _batch_auto_tag(tag, s); valid, weights, backend, workspace) :
        SFC.gridded_lag_sweep_batch!(sums, counts, sf, data, s, dist_be, Val(D), Val(V), Val(K);
                                     valid, weights, backend)
end

function SFC.gridded_sweep_batch!(
    sums::AbstractArray{<:Any, 3}, counts::AbstractArray{<:Any, 3}, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractArray{<:Any, 3}, s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V},
    ::Val{K}, tag::SB.AutoSpectralBackend; valid = SFC.AllValid(), weights = nothing,
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(), second_axis::SFC.AbstractSecondAxisSource,
    workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
) where {D, V, K}
    return _auto_joint_transform(second_axis, sf, s, dist_be, V * D + K, valid, weights, backend) ?
        SFC.gridded_sweep_batch!(sums, counts, sf, data, s, dist_be, axis_be, Val(D), Val(V), Val(K),
                                 _batch_auto_tag(tag, s); valid, weights, backend, second_axis, workspace) :
        SFC.gridded_lag_sweep_batch!(sums, counts, sf, data, s, dist_be, axis_be, Val(D), Val(V), Val(K);
                                     valid, weights, backend, second_axis)
end

_batch_auto_tag(::SB.AbstractAutoSpectralBackend, ::SFC.AbstractSeparableSchedule) =
    SB.FastFourierTransformSpectralBackend()
_batch_auto_tag(tag::SB.AbstractAutoSpectralBackend, ::SFC.ScatteredModesSchedule) = _scattered_needs_nufft(tag)

# Tensors on grids: each lag's symmetric moment store is binned, and the dense tensor is assembled once
# at the end.

# The transform a tensor sweep runs with: `Auto` is the FFT on a grid, and a tag must match its schedule.
_tensor_tag(tag::SB.AbstractFastFourierTransformSpectralBackend, ::SFC.AbstractSeparableSchedule) = tag
_tensor_tag(tag::SB.AbstractFastFourierTransformSpectralBackend, ::SFC.ScatteredModesSchedule) = _scattered_needs_nufft(tag)
_tensor_tag(tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend, ::SFC.ScatteredModesSchedule) = tag
_tensor_tag(::SB.AbstractNonUniformFastFourierTransformSpectralBackend, s::SFC.AbstractSeparableSchedule) = _nufft_needs_scattered(s)
_tensor_tag(::SB.AbstractAutoSpectralBackend, ::SFC.AbstractSeparableSchedule) = SB.FastFourierTransformSpectralBackend()
_tensor_tag(tag::SB.AbstractAutoSpectralBackend, ::SFC.ScatteredModesSchedule) = _scattered_needs_nufft(tag)

const TensorTag = Union{SB.AbstractFastFourierTransformSpectralBackend, SB.AbstractNonUniformFastFourierTransformSpectralBackend,
                        SB.AbstractAutoSpectralBackend}

# The pair weights and the tag of a tensor sweep, after its boundary's count check.
function _tensor_boundary(data, s, counts, weights, tag)
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    SFC._assert_grid_counts(s, counts, size(data, 2), w)
    return w, _tensor_tag(tag, s)
end

# Every lag's rank-`P` symmetric store `(n_sym, size(counts)...)`, binned on `backend` and added into the
# dense `sums`; `axis` is `nothing` or `(axis_edges, n_axis, second_axis)`.
function _tensor_run!(sums, counts, ::Val{P}, data, s, dist_be, plan, nb, axis, ::Val{D}, valid, w, tag,
                      backend::CB.AbstractExecutionBackend, workspace) where {P, D}
    sf = SFT.MomentTensorOperator{P}()
    eng = _transform_prepare(sf, data, s, dist_be, Val(D), Val(1), Val(0), valid, w, tag;
                             forward = _kept_forward(workspace, 1, 1), workspace,
                             stage_backend = _stage_backend(backend))
    make_scratch, done = _inverse_scratch(eng, length(eng.columns), workspace, backend)
    try
        _tensor_pairs!(sums, counts, sf, eng, plan, nb, axis, Val(D), Val(P), backend, make_scratch, workspace)
    finally
        done()
    end
    return nothing
end

function _tensor_pairs!(sums, counts, sf, eng, plan, nb, axis, ::Val{D}, ::Val{P}, backend, make_scratch,
                        workspace) where {D, P}
    vNs = Val(length(SFT.symmetric_indices(Val(D), Val(P))))
    sym = zeros(eltype(sums), SFC._val_int(vNs), size(counts)...)
    item! = (ls, lc, it, scratch) -> _transform_tensor_item!(ls, lc, sf, eng, it, scratch, plan, nb, axis, Val(D),
                                                             eng.vW, eng.vP, eng.vN, vNs)
    lags = out -> (ls, lc, it, _) -> _transform_tensor_lags!(ls, lc, sf, eng, out, it, plan, nb, axis, Val(D),
                                                             eng.vW, eng.vP, eng.vN, vNs)
    _sweep_pairs!(sym, counts, backend, eng, make_scratch, workspace, item!, lags)
    SFC._expand_symmetric!(sums, sym, Val(D), Val(P))
    return nothing
end

function _tensor_run!(sums, counts, ::Val{P}, data, s, dist_be, plan, nb, axis, ::Val{D}, valid, w, tag,
                      backend::CB.AbstractGPUBackend, workspace) where {P, D}
    SFC._require_device_outputs(backend, sums, counts)
    sym = SFC._result_zeros(backend, eltype(sums), binomial(D + P - 1, P), size(counts)...)
    SFC.device_transform_sweep!(sym, counts, backend, SFT.MomentTensorOperator{P}(), data, s, dist_be, plan, nb,
                                Val(D), Val(1), Val(0), valid, w, tag, axis, workspace)
    SFC._expand_symmetric!(sums, sym, Val(D), Val(P))
    return nothing
end

function SFC.gridded_tensor_sweep!(
    sums::AbstractArray{OT}, counts::AbstractVector{CT}, order::Val{P}, data::AbstractMatrix,
    s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, tag::TensorTag;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
) where {OT, CT, P, D}
    SFC._require_backend(backend)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    size(sums) == (ntuple(_ -> D, P)..., nb) && length(counts) == nb || throw(DimensionMismatch(
        "sums must be $((ntuple(_ -> D, P)..., nb)) and counts of length $nb; got $(size(sums)) and $(length(counts))",
    ))
    w, run_tag = _tensor_boundary(data, s, counts, weights, tag)
    _with_workspace(workspace) do ws
        _tensor_run!(sums, counts, order, data, s, dist_be, plan, nb, nothing, Val(D), valid, w, run_tag, backend, ws)
    end
    return sums, counts
end

function SFC.gridded_tensor_sweep!(
    sums::AbstractArray{OT}, counts::AbstractMatrix{CT}, order::Val{P}, data::AbstractMatrix,
    s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, tag::TensorTag;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
    second_axis::SFC.SeparationAngleAxis,
) where {OT, CT, P, D}
    SFC._require_backend(backend)
    SFC._require_directional(s)
    SFC._check_half_turn_counts(CT, s, dist_be)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    axis_edges = SFC.digitize_plan(axis_be)
    na = SFC.n_histogram_bins(axis_edges)
    size(sums) == (ntuple(_ -> D, P)..., nb, na) && size(counts) == (nb, na) || throw(DimensionMismatch(
        "sums must be $((ntuple(_ -> D, P)..., nb, na)) and counts ($nb, $na); got $(size(sums)) and $(size(counts))",
    ))
    w, run_tag = _tensor_boundary(data, s, counts, weights, tag)
    _with_workspace(workspace) do ws
        _tensor_run!(sums, counts, order, data, s, dist_be, plan, nb, (axis_edges, na, second_axis), Val(D), valid, w,
                     run_tag, backend, ws)
    end
    return sums, counts
end

function _transform_tensor_item!(
    sym::AbstractArray, counts::AbstractArray, sf, eng, item::NTuple{4, Int}, scratch, plan, nb, axis, vD::Val,
    vW::Val, vPo::Val, vN::Val, vNs::Val,
)
    out = _pair_inverse!(scratch, eng.fwd[item[1]], eng.fwd[item[2]], eng.columns)
    return _transform_tensor_lags!(sym, counts, sf, eng, out, item, plan, nb, axis, vD, vW, vPo, vN, vNs)
end

# The tensor lag loop over a whole pair or the `part`-th share of its lags, as `_transform_lags!`.
function _transform_tensor_lags!(
    sym::AbstractMatrix{OT}, counts::AbstractVector{CT}, sf, eng, out, item::NTuple{4, Int}, plan, nb,
    ::Nothing, ::Val{D}, ::Val{W}, ::Val{Po}, ::Val{N}, ::Val{Ns},
) where {OT, CT, D, W, Po, N, Ns}
    I, J, part, n_parts = item
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    lags = SFC._pair_lags(s, su, I, J, eng.r_max)
    @inbounds for li in part:n_parts:length(lags)
        h = Tuple(lags[li])
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(1), Val(0))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, size(out, 2), su, h, self_reverse)
        scale = self_reverse ? T(0.5) : one(T)
        m = SFC._lag_tensor(tr, out, idx, scale, geometry, Val(W), Val(Po), Val(N))
        f = SFC._tensor_factor(tr, factor)
        for n in 1:Ns
            sym[n, b] += OT(f * m[n])
        end
        counts[b] += CT(n_pairs)
    end
    return nothing
end

function _transform_tensor_lags!(
    sym::AbstractArray{OT, 3}, counts::AbstractMatrix{CT}, sf, eng, out, item::NTuple{4, Int}, plan, nb,
    axis::Tuple, ::Val{D}, ::Val{W}, ::Val{Po}, ::Val{N}, ::Val{Ns},
) where {OT, CT, D, W, Po, N, Ns}
    axis_edges, na, second_axis = axis
    I, J, part, n_parts = item
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    lags = SFC._pair_lags(s, su, I, J, eng.r_max)
    @inbounds for li in part:n_parts:length(lags)
        h = Tuple(lags[li])
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(1), Val(0))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, size(out, 2), su, h, self_reverse)
        scale = self_reverse ? T(0.5) : one(T)
        m = SFC._lag_tensor(tr, out, idx, scale, geometry, Val(W), Val(Po), Val(N))
        fac = SFC._tensor_factor(tr, factor)
        SFC.with_frames(tr, geometry) do frames
            Mi = length(frames)
            for f in frames
                bθ = SFH.digitize(SFC.axis_quantity(second_axis, f.dir, r2), axis_edges)
                1 <= bθ <= na || continue
                for n in 1:Ns
                    sym[n, b, bθ] += OT(fac * m[n] / Mi)
                end
                counts[b, bθ] += CT(n_pairs / Mi)
            end
        end
    end
    return nothing
end

# The CPU engine: one inverse per slab pair, the lags of each read on the host. `forward` is where the field's
# spectra go (see `_transform_prepare`).
function _transform_sweep!(
    sums, counts, backend::CB.AbstractExecutionBackend, sf, data, s, dist_be, plan, nb, ::Val{D}, ::Val{V},
    ::Val{K}, valid, weights, tag, axis, workspace; forward = _kept_forward(workspace, 1, 1),
) where {D, V, K}
    eng = _transform_prepare(sf, data, s, dist_be, Val(D), Val(V), Val(K), valid, weights, tag; forward, workspace,
                             stage_backend = _stage_backend(backend))
    make_scratch, done = _inverse_scratch(eng, length(eng.columns), workspace, backend)
    try
        _transform_pairs!(sums, counts, backend, sf, eng, plan, nb, axis, make_scratch, workspace, Val(D), Val(V),
                          Val(K))
    finally
        done()
    end
    return nothing
end

function _transform_pairs!(sums, counts, backend, sf, eng, plan, nb, axis, make_scratch, workspace, ::Val{D},
                           ::Val{V}, ::Val{K}) where {D, V, K}
    item! = _item_body(sf, eng, plan, nb, axis, Val(D), Val(V), Val(K), eng.vW, eng.vP, eng.vN)
    lags = out -> _lag_body(sf, eng, out, plan, nb, axis, Val(D), Val(V), Val(K), eng.vW, eng.vP, eng.vN)
    _sweep_pairs!(sums, counts, backend, eng, make_scratch, workspace, item!, lags)
    return nothing
end

"""
    _sweep_pairs!(sums, counts, backend, engine, make_scratch, workspace, item!, lags)

Every slab pair of `engine` on `backend`. With at least as many pairs as tasks, a task takes whole pairs
(`item!`, its inverse scratch from `make_scratch()`). With fewer, a pair's lags cannot be read until its inverse
is done, so the pairs run one at a time in this process: the tasks of its backend fill the pair's columns, one plan
on as many threads inverts them, and the tasks share the pair's lags (`lags(out)`).
"""
function _sweep_pairs!(sums, counts, backend, eng, make_scratch, workspace, item!, lags)
    pairs = SFC.sweep_items(eng.s, eng.r_max, 1, false)
    if length(pairs) >= SFC.sweep_tasks(backend)
        SFC.sweep_reduce!(sums, counts, backend, pairs, make_scratch, item!)
        return nothing
    end
    here = _stage_backend(backend)
    k = SFC.sweep_tasks(here)
    ncols = length(eng.columns)
    F1 = eng.fwd[1][1]
    split = SFC._kept!(workspace, :inverse_split, (typeof(F1), size(F1), eng.P, ncols, k),
                       _split_inverse_plan(eng, ncols, k))
    for it in pairs
        out = _pair_inverse_split!(split, eng.fwd[it[1]], eng.fwd[it[2]], eng.columns, here)
        SFC.sweep_reduce!(sums, counts, here, [(it[1], it[2], p, k) for p in 1:k], () -> nothing, lags(out))
    end
    return nothing
end

_item_body(sf, eng, plan, nb, ::Nothing, vD, vV, vK, vW, vP, vN) =
    (ls, lc, it, scratch) -> _transform_item!(ls, lc, sf, eng, it, scratch, plan, nb, vD, vV, vK, vW, vP, vN)

_item_body(sf, eng, plan, nb, axis::Tuple, vD, vV, vK, vW, vP, vN) =
    (ls, lc, it, scratch) -> _transform_item!(ls, lc, sf, eng, it, scratch, plan, nb, axis[1], axis[2],
                                              axis[3], vD, vV, vK, vW, vP, vN)

_lag_body(sf, eng, out, plan, nb, ::Nothing, vD, vV, vK, vW, vP, vN) =
    (ls, lc, it, _) -> _transform_lags!(ls, lc, sf, eng, out, it, plan, nb, vD, vV, vK, vW, vP, vN)

_lag_body(sf, eng, out, plan, nb, axis::Tuple, vD, vV, vK, vW, vP, vN) =
    (ls, lc, it, _) -> _transform_lags!(ls, lc, sf, eng, out, it, plan, nb, axis[1], axis[2],
                                        axis[3], vD, vV, vK, vW, vP, vN)

# A device runs the engine through the KernelAbstractions extension.
_transform_sweep!(
    sums, counts, backend::CB.AbstractGPUBackend, sf, data, s, dist_be, plan, nb, vD::Val, vV::Val, vK::Val,
    valid, weights, tag, axis, workspace,
) = SFC.device_transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, vD, vV, vK, valid, weights,
                                tag, axis, workspace)

# The lag sweep visits every lag of every slab pair over the slab's cells; the transform pays one forward
# transform per monomial per slab and one inverse per raw moment per slab pair, however few lags are wanted.
_auto_transform(sf, s, dist_be, W::Int, valid, weights, ::CB.AbstractExecutionBackend) =
    _prefers_transform(sf, s, dist_be, W, valid, weights)

# A device runs the transform for a polynomial operator and the direct sweep for any other.
_auto_transform(sf, s, dist_be, W::Int, valid, weights, ::CB.AbstractGPUBackend) = SFT.is_polynomial_operator(sf)

function _prefers_transform(sf, s::SFC.AbstractSeparableSchedule, dist_be, W::Int, valid, weights)
    SFT.is_polynomial_operator(sf) || return false
    su = SFC.uniform_axes(s)
    Dg = SFC.grid_dimension(su)
    T = eltype(su.spacing)
    r_max = SFC._cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    lims = SFC.lag_limits(s, r_max)
    n_lags = prod(ntuple(d -> length(SFC.lag_range(su, d, lims[d])), Dg))
    n = prod(_pad_dims(s, r_max))
    p = SFT.order(sf)
    n_pairs = SFC.n_enumerated_pairs(s, r_max)
    count_column = (valid isa SFC.AllValid && weights === nothing) ? 0 : 1
    per_pair = SFC._sf_inverse_count(sf, SFC.lag_transport(s), W) + count_column
    transforms = SFC.n_slabs(s) * binomial(W + p, p) + n_pairs * per_pair
    return transforms * n * log2(max(2, n)) < n_pairs * n_lags * SFC.n_cells(su)
end

function SFC.gridded_sweep!(
    sums::AbstractArray, counts::AbstractArray, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    ::SB.AutoSpectralBackend; valid = SFC.AllValid(), weights = nothing,
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(), workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
) where {D, V, K}
    return _auto_transform(sf, s, dist_be, V * D + K, valid, weights, backend) ?
        SFC.gridded_sweep!(sums, counts, sf, data, s, dist_be, Val(D), Val(V), Val(K),
                           SB.FastFourierTransformSpectralBackend(); valid, weights, backend, workspace) :
        SFC.gridded_lag_sweep!(sums, counts, sf, data, s, dist_be, Val(D), Val(V), Val(K); valid, weights, backend)
end

function SFC.gridded_sweep!(
    sums::AbstractMatrix, counts::AbstractMatrix, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K},
    ::SB.AutoSpectralBackend; valid = SFC.AllValid(), weights = nothing,
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    second_axis::SFC.AbstractSecondAxisSource, workspace::Union{Nothing, SFC.TransformWorkspace} = nothing,
) where {D, V, K}
    return _auto_joint_transform(second_axis, sf, s, dist_be, V * D + K, valid, weights, backend) ?
        SFC.gridded_sweep!(sums, counts, sf, data, s, dist_be, axis_be, Val(D), Val(V), Val(K),
                           SB.FastFourierTransformSpectralBackend(); valid, weights, backend, second_axis, workspace) :
        SFC.gridded_lag_sweep!(sums, counts, sf, data, s, dist_be, axis_be, Val(D), Val(V), Val(K);
                               valid, weights, backend, second_axis)
end

"""
    _trace_lags(mt, ::Val{D}) -> (trace, counts)

`Σ_pairs ‖δu‖²` and the pair count at every lag of the padded grid, both weighted when the transforms
are. Only the diagonal moment components are formed, so the monomials transformed are the mask, each
component and each square; with every cell held and no weights the count is the box overlap and no
mask transform is taken.
"""
function _trace_lags(mt::MonomialTransforms{FT, Dg}, ::Val{D}) where {FT, Dg, D}
    trace = moment_component(mt, (1, 1))
    for d in 2:D
        trace .+= moment_component(mt, (d, d))
    end
    complete = mt.valid isa SFC.AllValid && mt.weights isa SFC.NoWeights
    pc = complete ? nothing : pair_counts(mt)
    counts = _count_array(trace, mt.weights)
    P = mt.P
    @inbounds for I in CartesianIndices(trace)
        h = _lag_of_index(Tuple(I), P)
        counts[I] = pc === nothing ? _padded_pair_count(mt.schedule, h) : _count_value(pc[I], mt.weights)
    end
    return trace, counts
end

_count_array(trace, ::SFC.NoWeights) = similar(trace, Int)
_count_array(trace, ::AbstractVector) = similar(trace)
_count_value(c, ::SFC.NoWeights) = round(Int, c)
_count_value(c, ::AbstractVector) = c

# The signed lag a position of the padded transform holds.
@inline _lag_of_index(I::NTuple{Dg, Int}, P::NTuple{Dg, Int}) where {Dg} =
    ntuple(d -> (m = I[d] - 1; m > P[d] ÷ 2 ? m - P[d] : m), Val(Dg))

# Pairs a lag of the padded grid names on a complete field: zero beyond the cells of a bounded direction.
@inline function _padded_pair_count(s::SFC.UniformLagSchedule{Dg}, h::NTuple{Dg, Int}) where {Dg}
    n = 1
    @inbounds for d in 1:Dg
        n *= s.periodic[d] ? s.dims[d] : max(0, s.dims[d] - abs(h[d]))
    end
    return n
end

"""Weighted variance of the held cells about their weighted mean, summed over the field's components."""
function _held_variance(data::AbstractMatrix, valid, weights)
    W, n = size(data)
    T = float(eltype(data))
    acc = zero(T)
    for c in 1:W
        m = zero(T)
        k = zero(T)
        @inbounds for i in 1:n
            valid[i] || continue
            w = _cell_weight(weights, i)
            m += w * data[c, i]
            k += w
        end
        k > 0 || throw(ArgumentError("no held cell carries weight"))
        m /= k
        @inbounds for i in 1:n
            valid[i] && (acc += _cell_weight(weights, i) * (data[c, i] - m)^2 / k)
        end
    end
    return acc
end

@inline _cell_weight(::SFC.NoWeights, i) = true
@inline _cell_weight(w::AbstractVector, i) = @inbounds w[i]

function SFC.gridded_spectrum(
    u::AbstractArray, s::SFC.UniformLagSchedule{Dg, T}, ::Val{D},
    ::SB.AbstractFastFourierTransformSpectralBackend; valid = SFC.AllValid(), weights = nothing,
    taper::SFC.AbstractTaper = SFC.NoTaper(),
    missing_lags::SFC.AbstractMissingLagPolicy = SFC.RefuseMissingLags(),
) where {Dg, T, D}
    size(u, 1) == D || throw(DimensionMismatch("field has $(size(u, 1)) components, declared $D"))
    data = reshape(u, D, :)
    SFC._check_grid_field(SFT.S2SFType(), data, s, Val(D), Val(1), Val(0))
    w = SFC._pair_weights(weights, size(data, 2), T)
    # every lag of a bounded direction, |h| < n, transformed linearly: P ≥ 2n − 1; a periodic one needs
    # no padding
    P = _pad_dims(s, T(Inf))
    mt = MonomialTransforms(data, s, P, valid, w)
    trace, counts = _trace_lags(mt, Val(D))
    σ2 = _held_variance(data, valid, w)
    r_max = sqrt(sum(d -> (s.periodic[d] ? (s.dims[d] ÷ 2) * s.spacing[d] : (s.dims[d] - 1) * s.spacing[d])^2,
                     1:Dg))

    # C(h) = C(0) − D(h)/2 with C(0) the held cells' variance, weighted by the taper; a lag no held pair
    # names is structurally absent beyond a bounded direction's cells and a missing datum within them
    cov = similar(trace)
    @inbounds for I in CartesianIndices(trace)
        h = _lag_of_index(Tuple(I), P)
        n = counts[I]
        if n > 0
            r = sqrt(sum(d -> (h[d] * s.spacing[d])^2, 1:Dg))
            cov[I] = SFC.taper_weight(taper, r, r_max) * (σ2 - trace[I] / (2n))
        elseif all(d -> s.periodic[d] || abs(h[d]) < s.dims[d], 1:Dg)
            cov[I] = _missing_lag_value(missing_lags, h, T)
        else
            cov[I] = zero(T)
        end
    end

    density = real.(AbstractFFTs.fft(cov)) .* (prod(abs, s.spacing) / T(2π)^Dg)
    wavenumbers = ntuple(d -> T(2π) .* AbstractFFTs.fftfreq(P[d], 1 / abs(s.spacing[d])), Val(Dg))
    return wavenumbers, density
end

_missing_lag_value(::SFC.RefuseMissingLags, h, ::Type{T}) where {T} = throw(ArgumentError(
    "lag $h is named by no pair of held cells, so its structure function is undefined; pass " *
    "`missing_lags = ZeroDeviationAtMissingLags()` to set its covariance to zero, or fill the field.",
))
_missing_lag_value(::SFC.ZeroDeviationAtMissingLags, h, ::Type{T}) where {T} = zero(T)

end # module
