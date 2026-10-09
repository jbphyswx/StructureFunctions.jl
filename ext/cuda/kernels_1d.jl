# The native 1-D kernel: an N-body broadcast over tile pairs into a block's shared distance histogram, replicated
# `R` times against atomic contention, for every moment set, shared or per-slice positions, weighted or not, at the
# geometry's coordinate and field widths. Output is `(NMOM, NB, B)`; counts are per bin, the same for every moment.

"""Compiled-in cap on distance bins for the static shared 1D histogram."""
const CU_MAX_BINS = 128

"""Launch plans `(TILE, R)` to try for the formula's `(TILE, R)`, in order: it, fewer replicas, then smaller tiles."""
_cuda_1d_steps(TILE::Int, R::Int) = ((TILE, R), (TILE, R ÷ 2), (TILE, R ÷ 4), (256, 8), (256, 2), (128, 1))

"""Static shared bytes of the native 1-D kernel for coordinates of `XT` at width `W`, fields of `UT` at width `F`,
sums of `FT` and counts of `CST`, `NMOM` moments, `TILE`-point tiles and a histogram of `H` bins at stride `S`."""
@inline _cuda_1d_smem_bytes(::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int, F::Int, NMOM::Int,
                            TILE::Int, S::Int, H::Int) where {XT, UT, FT, CST} =
    2 * SFC.gpu_localmem_bytes(XT, W * TILE) + 2 * SFC.gpu_localmem_bytes(UT, F * TILE) +
    SFC.gpu_localmem_bytes(FT, NMOM * H * S) + SFC.gpu_localmem_bytes(CST, H * S)

"""The first `(TILE, R)` of `plans` with at least one replica whose kernel, at histogram stride `R`, fits the device
`caps` describes, or `nothing`."""
function _cuda_1d_fitting_plan(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST},
                               W::Int, F::Int, NMOM::Int, H::Int, plans::Tuple) where {XT, UT, FT, CST}
    for (TILE, R) in plans
        R >= 1 || continue
        SFC.gpu_static_smem_fits(caps, _cuda_1d_smem_bytes(XT, UT, FT, CST, W, F, NMOM, TILE, R, H)) &&
            return (TILE, R)
    end
    return nothing
end

@inline function _cuda_1d_bin_capacity(NB::Int)
    NB <= 16 && return Val(16)
    NB <= 32 && return Val(32)
    NB <= 64 && return Val(64)
    return Val(128)
end

function _cuda_sf_1d_kernel!(
    output,                 # (NMOM, NB, B)
    counts,                 # (NMOM, NB, B)
    x,                      # (W, N, B) varying / (W, N, 1) fixed
    u,                      # (F, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    ddig,                   # device digitize plan of the distance bins
    N::Int, NB::Int,
    sched, ntb::Int,
    ::Val{W}, ::Val{F}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, ::Val{R}, ::Val{S}, ::Val{H}, ::Val{CST},
    geom,
) where {W, F, NMOM, FIXED_X, TILE, R, S, H, CST}
    FT = eltype(output)
    lid = Int(threadIdx().x)
    wg = Int(blockDim().x)
    lb = Int(blockIdx().x)
    bid = (lb - 1) % ntb + 1
    b = (lb - 1) ÷ ntb + 1

    sxi = CuStaticSharedArray(eltype(x), W * TILE)
    sxj = CuStaticSharedArray(eltype(x), W * TILE)
    sui = CuStaticSharedArray(eltype(u), F * TILE)
    suj = CuStaticSharedArray(eltype(u), F * TILE)
    ssum = CuStaticSharedArray(FT, NMOM * H * S)
    scnt = CuStaticSharedArray(CST, H * S)

    c = lid
    while c <= NMOM * NB * S
        @inbounds ssum[c] = zero(FT)
        c += wg
    end
    c = lid
    while c <= NB * S
        @inbounds scnt[c] = zero(CST)
        c += wg
    end

    ti, tj = SFC.tile_for(sched, bid)
    i0 = (ti - 1) * TILE + 1
    j0 = (tj - 1) * TILE + 1
    ni = min(TILE, N - i0 + 1)
    nj = min(TILE, N - j0 + 1)
    xb = FIXED_X ? 1 : b

    if lid <= ni
        @inbounds gi = i0 + lid - 1
        @inbounds for d in 1:W
            sxi[(d - 1) * TILE + lid] = x[d, gi, xb]
        end
        @inbounds for d in 1:F
            sui[(d - 1) * TILE + lid] = u[d, gi, b]
        end
    end
    if ti < tj && lid <= nj
        @inbounds gj = j0 + lid - 1
        @inbounds for d in 1:W
            sxj[(d - 1) * TILE + lid] = x[d, gj, xb]
        end
        @inbounds for d in 1:F
            suj[(d - 1) * TILE + lid] = u[d, gj, b]
        end
    end
    sync_threads()

    if lid <= ni
        lane = (lid - 1) % R + 1
        Xi = _cuda_ld(sxi, Val(W), Val(TILE), lid)
        Ui = _cuda_ld(sui, Val(F), Val(TILE), lid)
        wi = SFC._point_weight(wts, i0 + lid - 1)
        diag = !(ti < tj)
        jbase = diag ? i0 : j0
        jj = diag ? lid + 1 : 1
        jend = diag ? ni : nj
        while jj <= jend
            if diag
                Xj = _cuda_ld(sxi, Val(W), Val(TILE), jj)
                Uj = _cuda_ld(sui, Val(F), Val(TILE), jj)
            else
                Xj = _cuda_ld(sxj, Val(W), Val(TILE), jj)
                Uj = _cuda_ld(suj, Val(F), Val(TILE), jj)
            end
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            bin = SFH.digitize(dist, ddig)
            if ok && 1 <= bin <= NB
                moments = SFC._sf_pair_moments(sf_type, geom, frame, dist, Xi, Xj, Ui, Uj)
                pw = wi * SFC._point_weight(wts, jbase + jj - 1)
                @inbounds for m in SFC._sf_accum_moments(sf_type)
                    CUDA.@atomic ssum[((m - 1) * NB + bin - 1) * S + lane] += pw * moments[m]
                end
                @inbounds CUDA.@atomic scnt[(bin - 1) * S + lane] += CST(pw)
            end
            jj += 1
        end
    end
    sync_threads()

    cell = lid
    while cell <= NMOM * NB
        m = (cell - 1) ÷ NB + 1
        bin = (cell - 1) % NB + 1
        s = _cuda_flush_replicated(sf_type, ssum, 0, NB, m, bin, Val(R), Val(S))
        if s != zero(FT)
            CUDA.@atomic output[m, bin, b] += s
        end
        cell += wg
    end
    bcell = lid
    while bcell <= NB
        cnt = _cuda_sum_replicas(scnt, (bcell - 1) * S, Val(R))
        if cnt != zero(CST)
            @inbounds for m in 1:NMOM
                CUDA.@atomic counts[m, bcell, b] += cnt
            end
        end
        bcell += wg
    end
    return nothing
end

@inline function _cuda_sum_replicas(values, base::Int, ::Val{R}) where {R}
    total = zero(eltype(values))
    @inbounds for lane in 1:R
        total += values[base + lane]
    end
    return total
end

"""Moment `m` of the moment set `M` in bin `bin` of the histogram at offset `off` of `sums`, its `R` replicas summed;
the single-pass differences are formed as `_sf_flush_moment` forms them."""
@inline function _cuda_flush_replicated(M, sums, off::Int, NB::Int, m::Int, bin::Int, ::Val{R}, ::Val{S}) where {R, S}
    return _cuda_sum_replicas(sums, off + ((m - 1) * NB + bin - 1) * S, Val(R))
end

@inline function _cuda_flush_replicated(::SFT.SinglePassInvariants, sums, off::Int, NB::Int, m::Int, bin::Int,
                                        ::Val{R}, ::Val{S}) where {R, S}
    m1 = m == 3 ? 1 : m == 6 ? 4 : m
    value = _cuda_sum_replicas(sums, off + ((m1 - 1) * NB + bin - 1) * S, Val(R))
    if m == 3 || m == 6
        value -= _cuda_sum_replicas(sums, off + (m1 * NB + bin - 1) * S, Val(R))
    end
    return value
end

# A batch over shared positions: a block per tile pair and strip of `SW` slices forms each pair's geometry once, then
# adds the pair's moments of every slice of the strip into that slice's histogram (lanes (slice, replica)); a pair's
# count, the same for every slice, is added once and flushed to every slice.
function _cuda_sf_1d_strip_kernel!(
    output, counts,         # (NMOM, NB, B)
    x,                      # (W, N) shared positions
    u,                      # (F, N, B)
    wts, sf_type, ddig, N::Int, NB::Int, B::Int, sched, ntb::Int,
    ::Val{W}, ::Val{F}, ::Val{NMOM}, ::Val{TILE}, ::Val{SW}, ::Val{R}, ::Val{H}, ::Val{CST}, geom,
) where {W, F, NMOM, TILE, SW, R, H, CST}
    FT = eltype(output)
    lid = Int(threadIdx().x)
    wg = Int(blockDim().x)
    lb = Int(blockIdx().x)
    bid = (lb - 1) % ntb + 1
    b0 = ((lb - 1) ÷ ntb) * SW
    nsw = min(SW, B - b0)

    sxi = CuStaticSharedArray(eltype(x), W * TILE)
    sxj = CuStaticSharedArray(eltype(x), W * TILE)
    sui = CuStaticSharedArray(eltype(u), F * TILE * SW)
    suj = CuStaticSharedArray(eltype(u), F * TILE * SW)
    ssum = CuStaticSharedArray(FT, NMOM * H * SW * R)
    scnt = CuStaticSharedArray(CST, H * R)

    c = lid
    while c <= NMOM * NB * SW * R
        @inbounds ssum[c] = zero(FT)
        c += wg
    end
    c = lid
    while c <= NB * R
        @inbounds scnt[c] = zero(CST)
        c += wg
    end

    ti, tj = SFC.tile_for(sched, bid)
    i0 = (ti - 1) * TILE + 1
    j0 = (tj - 1) * TILE + 1
    ni = min(TILE, N - i0 + 1)
    nj = min(TILE, N - j0 + 1)
    if lid <= ni
        gi = i0 + lid - 1
        @inbounds for d in 1:W
            sxi[(d - 1) * TILE + lid] = x[d, gi]
        end
        @inbounds for s in 1:nsw, d in 1:F
            sui[(s - 1) * F * TILE + (d - 1) * TILE + lid] = u[d, gi, b0 + s]
        end
    end
    if ti < tj && lid <= nj
        gj = j0 + lid - 1
        @inbounds for d in 1:W
            sxj[(d - 1) * TILE + lid] = x[d, gj]
        end
        @inbounds for s in 1:nsw, d in 1:F
            suj[(s - 1) * F * TILE + (d - 1) * TILE + lid] = u[d, gj, b0 + s]
        end
    end
    sync_threads()

    if lid <= ni
        rl = (lid - 1) % R + 1
        Xi = _cuda_ld(sxi, Val(W), Val(TILE), lid)
        wi = SFC._point_weight(wts, i0 + lid - 1)
        diag = !(ti < tj)
        jbase = diag ? i0 : j0
        jj = diag ? lid + 1 : 1
        jend = diag ? ni : nj
        while jj <= jend
            Xj = diag ? _cuda_ld(sxi, Val(W), Val(TILE), jj) : _cuda_ld(sxj, Val(W), Val(TILE), jj)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            bin = SFH.digitize(dist, ddig)
            if ok && 1 <= bin <= NB
                pw = wi * SFC._point_weight(wts, jbase + jj - 1)
                @inbounds CUDA.@atomic scnt[(bin - 1) * R + rl] += CST(pw)
                for s in 1:nsw
                    base = (s - 1) * F * TILE
                    Ui = _cuda_ld_at(sui, Val(F), Val(TILE), base, lid)
                    Uj = diag ? _cuda_ld_at(sui, Val(F), Val(TILE), base, jj) : _cuda_ld_at(suj, Val(F), Val(TILE), base, jj)
                    moments = SFC._sf_pair_moments(sf_type, geom, frame, dist, Xi, Xj, Ui, Uj)
                    @inbounds for m in SFC._sf_accum_moments(sf_type)
                        CUDA.@atomic ssum[(((s - 1) * NMOM + m - 1) * NB + bin - 1) * R + rl] += pw * moments[m]
                    end
                end
            end
            jj += 1
        end
    end
    sync_threads()

    cell = lid
    while cell <= NMOM * NB * nsw
        m = (cell - 1) % NMOM + 1
        bin = ((cell - 1) ÷ NMOM) % NB + 1
        s = (cell - 1) ÷ (NMOM * NB) + 1
        v = _cuda_flush_replicated(sf_type, ssum, (s - 1) * NMOM * NB * R, NB, m, bin, Val(R), Val(R))
        if v != zero(FT)
            CUDA.@atomic output[m, bin, b0 + s] += v
        end
        cell += wg
    end
    bcell = lid
    while bcell <= NB
        cnt = _cuda_sum_replicas(scnt, (bcell - 1) * R, Val(R))
        if cnt != zero(CST)
            for s in 1:nsw, m in 1:NMOM
                @inbounds CUDA.@atomic counts[m, bcell, b0 + s] += cnt
            end
        end
        bcell += wg
    end
    return nothing
end

"""Load local point `k`'s `W`-vector from a tile buffer staged as `base + (d-1)*TILE + k`."""
@inline _cuda_ld_at(buf, ::Val{W}, ::Val{TILE}, base::Int, k::Integer) where {W, TILE} =
    SA.SVector{W}(ntuple(d -> @inbounds(buf[base + (d - 1) * TILE + k]), Val(W)))

"""Launch plan of the native 1-D kernel: coordinate width `W`, field width `F`, `NMOM` moments, `TILE`-point tiles,
`R` histogram replicas at stride `S`, bin capacity `H` and shared counts of `CST`."""
struct CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST}
    function CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST}() where {W, F, NMOM, TILE, R, S, H, CST}
        S >= R || throw(ArgumentError("a shared histogram stride of $S does not cover $R replicas"))
        return new{W, F, NMOM, TILE, R, S, H, CST}()
    end
end

"""The plan of the formula's `(TILE, R)` at its first fitting step ([`_cuda_1d_steps`](@ref)) on the device `caps`
describes, or `nothing`."""
function _cuda_1d_fit(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int, F::Int,
                      NMOM::Int, H::Int, (TILE, R)::NTuple{2, Int}) where {XT, UT, FT, CST}
    launch = _cuda_1d_fitting_plan(caps, XT, UT, FT, CST, W, F, NMOM, H, _cuda_1d_steps(TILE, R))
    return launch === nothing ? nothing : CUDA1DPlan{W, F, NMOM, launch[1], launch[2], launch[2], H, CST}()
end

"""Launch plan of `_cuda_sf_1d_strip_kernel!`: coordinate width `W`, field width `F`, `NMOM` moments, `TILE`-point
tiles, strips of `SW` slices, `R` histogram replicas per slice, bin capacity `H` and shared counts of `CST`."""
struct CUDA1DStripPlan{W, F, NMOM, TILE, SW, R, H, CST} end

"""Static shared bytes of `_cuda_sf_1d_strip_kernel!` for coordinates of `XT` at width `W`, fields of `UT` at width
`F`, sums of `FT` and counts of `CST`, `NMOM` moments, `TILE`-point tiles, strips of `SW` slices, `R` replicas and a
histogram of `H` bins."""
@inline _cuda_1d_strip_smem_bytes(::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int, F::Int, NMOM::Int,
                                  TILE::Int, SW::Int, R::Int, H::Int) where {XT, UT, FT, CST} =
    2 * SFC.gpu_localmem_bytes(XT, W * TILE) + 2 * SFC.gpu_localmem_bytes(UT, F * TILE * SW) +
    SFC.gpu_localmem_bytes(FT, NMOM * H * SW * R) + SFC.gpu_localmem_bytes(CST, H * R)

"""Strip plans `(TILE, SW, R)` to try for the formula's, in order: it, fewer replicas, narrower strips, then small
ones."""
_cuda_1d_strip_steps(TILE::Int, SW::Int, R::Int) =
    ((TILE, SW, R), (TILE, SW, R ÷ 2), (TILE, SW ÷ 2, R), (256, 4, 1), (256, 2, 1), (128, 2, 1))

"""The strip plan of the formula's `(:strip, TILE, SW, R)` at its first fitting step ([`_cuda_1d_strip_steps`](@ref))
on the device `caps` describes, else `nothing`."""
function _cuda_1d_fit(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int, F::Int,
                      NMOM::Int, H::Int, (_, TILE, SW, R)::Tuple{Symbol, Int, Int, Int}) where {XT, UT, FT, CST}
    for (t, s, r) in _cuda_1d_strip_steps(TILE, SW, R)
        (s >= 2 && r >= 1) || continue
        SFC.gpu_static_smem_fits(caps, _cuda_1d_strip_smem_bytes(XT, UT, FT, CST, W, F, NMOM, t, s, r, H)) &&
            return CUDA1DStripPlan{W, F, NMOM, t, s, r, H, CST}()
    end
    return nothing
end

"""Replica scale of the native 1-D histogram ([`_cuda_replicas`](@ref)), by moment count (one, or several)."""
const CU_1D_REPLICA_SCALE = (0.7, 2.8)

"""In-range share at most which a batch over shared positions also tries the strip kernel."""
const CU_STRIP_SHARE = 0.2

"""Pair evaluations from which a native 1-D batch over shared positions samples its in-range share and times its
candidates ([`CUDAChoice`](@ref)), by moment count (one, or several)."""
const CU_1D_CHOOSE_FROM = (6.0e7, 1.5e6)

"""Replicas of a shared histogram of `H` bins and `NMOM` moments for `TILE`-point tiles at in-range share `share`:
`c · TILE · √(share / (H · NMOM))` as a power of two in `1:32`, balancing the shared atomics a block's in-range pairs
contend on against the global atomics its flush makes."""
_cuda_replicas(c, TILE::Int, share::Real, H::Int, NMOM::Int) =
    clamp(prevpow(2, max(1, floor(Int, c * TILE * sqrt(share / (H * NMOM))))), 1, 32)

"""The native 1-D kernel's plan for coordinates of `XT` at width `W`, fields of `UT` at width `F`, sums of `FT`, shared
counts of `CST`, `NMOM` moments and bin capacity `H` on the device `caps` describes, for a launch over `N` points and
`B` slices at in-range share `share`: tiles of 384 (at a share of at least 0.6) or 256 points for one moment and 256
for several, the largest whose tile pairs reach 8 (one moment) or 2 blocks per multiprocessor, else 128; replicas by
[`_cuda_replicas`](@ref); stepped down until the kernel fits."""
function _cuda_1d_point_plan(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int, F::Int,
                             NMOM::Int, H::Int, N::Int, B::Int, share::Real) where {XT, UT, FT, CST}
    t = NMOM == 1 ? _cuda_tile_for(caps, N, B, 8, share >= 0.6 ? (384, 256) : (256,)) :
        _cuda_tile_for(caps, N, B, 2, (256,))
    c = CU_1D_REPLICA_SCALE[NMOM == 1 ? 1 : 2]
    return _cuda_1d_fit(caps, XT, UT, FT, CST, W, F, NMOM, H, (t, _cuda_replicas(c, t, share, H, NMOM)))
end

"""The candidate plans ([`CUDAChoice`](@ref)) of a batch over shared positions, for the types of
[`_cuda_1d_point_plan`](@ref): at an in-range share of at most [`CU_STRIP_SHARE`](@ref) and enough slices, the strip
kernel at tile 256 with strips of up to 8 slices and up to 4 replicas, then the point plan at that share; else the
point plan."""
function _cuda_1d_fixed_candidates(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int,
                                   F::Int, NMOM::Int, H::Int) where {XT, UT, FT, CST}
    c = CU_1D_REPLICA_SCALE[NMOM == 1 ? 1 : 2]
    return function (N::Int, B::Int, fixed::Bool, evaluations::Real, share::Real)
        point = _cuda_1d_point_plan(caps, XT, UT, FT, CST, W, F, NMOM, H, N, B, share)
        (share <= CU_STRIP_SHARE && B >= (NMOM == 1 ? 2 : 4)) || return Any[point]
        strip = _cuda_1d_fit(caps, XT, UT, FT, CST, W, F, NMOM, H,
                             (:strip, 256, min(prevpow(2, B), 8), min(_cuda_replicas(c, 256, share, H, NMOM), 4)))
        return unique(filter(!isnothing, Any[strip, point]))
    end
end

"""The native 1-D kernel's rule for one call's types on the device `caps` describes: coordinates of `XT` at width `W`,
fields of `UT` at width `F`, sums of `FT`, shared counts of `CST`, `NMOM` moments and bin capacity `H`.
[`_cuda_1d_launch_plan`](@ref) gives a call's plan; a batch over shared positions takes the plan `fixed`
([`CUDAChoice`](@ref)) measures."""
struct CUDA1DRule{XT, UT, FT, CST, W, F, NMOM, H, C}
    caps::SFC.GPUDeviceCaps
    fixed::C
end

"""The plan of a launch over `N` points and `B` slices, which samples no share ([`_cuda_1d_point_plan`](@ref) at a
share of one)."""
_cuda_1d_launch_plan(r::CUDA1DRule{XT, UT, FT, CST, W, F, NMOM, H}, N::Int,
                     B::Int) where {XT, UT, FT, CST, W, F, NMOM, H} =
    _cuda_1d_point_plan(r.caps, XT, UT, FT, CST, W, F, NMOM, H, N, B, 1.0)

"""The native 1-D rule ([`CUDA1DRule`](@ref)) for the moment set `M` on the device `caps` describes, or `nothing`
when `NB` exceeds `CU_MAX_BINS`, the sum or count type has no native atomic add, or no plan fits the device."""
function _cuda_1d_plan(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CT}, wts, geom,
                       NB::Int, M) where {XT, UT, OT, CT}
    (NB <= CU_MAX_BINS && _cuda_atomic_add(OT) && _cuda_atomic_add(CT)) || return nothing
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFC._sf_field_width(M, geom))
    CST = _cuda_shared_count_type(wts, CT)
    NMOM = SFC._sf_nmom(M)
    H = SFC._val_int(_cuda_1d_bin_capacity(NB))
    _cuda_1d_fit(caps, XT, UT, OT, CST, W, F, NMOM, H, (128, 1)) === nothing && return nothing
    fixed = CUDAChoice(_cuda_1d_fixed_candidates(caps, XT, UT, OT, CST, W, F, NMOM, H),
                       CU_1D_CHOOSE_FROM[NMOM == 1 ? 1 : 2])
    return CUDA1DRule{XT, UT, OT, CST, W, F, NMOM, H, typeof(fixed)}(caps, fixed)
end

"""The tile of a native 1-D plan."""
_cuda_tile(::CUDA1DPlan{W, F, NMOM, TILE}) where {W, F, NMOM, TILE} = TILE
_cuda_tile(::CUDA1DStripPlan{W, F, NMOM, TILE}) where {W, F, NMOM, TILE} = TILE

"""Launch the plan of this call: for a batch over shared positions the plan `rule.fixed` gives
([`_cuda_plan`](@ref)), else [`_cuda_1d_launch_plan`](@ref)'s."""
function _cuda_launch_1d!(rule::CUDA1DRule, out, cnt, x, u, wts, sf_type, ddig, N::Int, NB::Int, B::Int,
                          fixed_x::Bool, geom, cull)
    if fixed_x && B > 1
        launch! = (plan, s, c) -> _cuda_launch_1d!(plan, s, c, x, u, wts, sf_type, ddig, N, NB, B, fixed_x, geom, cull)
        return launch!(_cuda_plan(launch!, rule.fixed, out, cnt, x, ddig, NB, N, B, true, geom, cull), out, cnt)
    end
    return _cuda_launch_1d!(_cuda_1d_launch_plan(rule, N, B), out, cnt, x, u, wts, sf_type, ddig, N, NB, B, fixed_x,
                            geom, cull)
end

"""Launch the native 1-D kernel `plan` names into `out`/`cnt` `(NMOM, NB, B)`; `x` is `(W,N,B)` varying or
`(W,N)`/`(W,N,1)` fixed, `u` is `(F,N,B)`."""
function _cuda_launch_1d!(::CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST}, out, cnt, x, u, wts, sf_type, ddig,
                          N::Int, NB::Int, B::Int, fixed_x::Bool, geom,
                          cull) where {W, F, NMOM, TILE, R, S, H, CST}
    NB <= H || throw(ArgumentError("$NB bins exceed the plan's shared histogram capacity $H"))
    xv = fixed_x ? reshape(x, W, N, 1) : reshape(x, W, N, B)
    uv = reshape(u, F, N, B)
    sched = SFC.schedule_for(cull, N, TILE)
    ntb = SFC.n_pair_blocks(sched)
    fx = fixed_x ? Val(true) : Val(false)
    args = (out, cnt, xv, uv, wts, sf_type, ddig, N, NB, sched, ntb,
            Val(W), Val(F), Val(NMOM), fx, Val(TILE), Val(R), Val(S), Val(H), Val(CST), geom)
    kern = @cuda launch=false _cuda_sf_1d_kernel!(args...)
    CUDA.attributes(kern.fun)[CUDA.FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT] = CU_CARVEOUT_MAX_SHARED
    kern(args...; threads = TILE, blocks = ntb * B)
    return nothing
end

"""Launch `_cuda_sf_1d_strip_kernel!` with `plan` into `out`/`cnt` `(NMOM, NB, B)` for a batch over shared positions
`x` `(W,N)`/`(W,N,1)` and fields `u` `(F,N,B)`."""
function _cuda_launch_1d!(::CUDA1DStripPlan{W, F, NMOM, TILE, SW, R, H, CST}, out, cnt, x, u, wts, sf_type, ddig,
                          N::Int, NB::Int, B::Int, fixed_x::Bool, geom,
                          cull) where {W, F, NMOM, TILE, SW, R, H, CST}
    fixed_x || throw(ArgumentError("a strip plan runs a batch over shared positions"))
    NB <= H || throw(ArgumentError("$NB bins exceed the plan's shared histogram capacity $H"))
    xv = reshape(x, W, N)
    uv = reshape(u, F, N, B)
    sched = SFC.schedule_for(cull, N, TILE)
    ntb = SFC.n_pair_blocks(sched)
    args = (out, cnt, xv, uv, wts, sf_type, ddig, N, NB, B, sched, ntb,
            Val(W), Val(F), Val(NMOM), Val(TILE), Val(SW), Val(R), Val(H), Val(CST), geom)
    kern = @cuda launch=false _cuda_sf_1d_strip_kernel!(args...)
    CUDA.attributes(kern.fun)[CUDA.FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT] = CU_CARVEOUT_MAX_SHARED
    kern(args...; threads = TILE, blocks = ntb * cld(B, SW))
    return nothing
end
