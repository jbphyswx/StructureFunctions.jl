# =============================================================================
# CUDA-specialized 1D structure-function kernel (distance histogram only).
#
# Same N-body broadcast structure as the 2D kernel. Histograms with at most 128
# bins use compile-time shared-memory capacity classes. Each class reserves only
# enough storage for its bin range, leaving the remaining block budget available
# for pair tiles and replicated accumulator lanes.
# Covers every moment set, fixed-x and varying-x, weighted or not, at the coordinate and field widths
# of the geometry. Output is (NMOM, NB, B); counts are per-bin (shared across moments).
# =============================================================================

"""Compiled-in cap on distance bins for the static shared 1D histogram."""
const CU_MAX_BINS = 128

"""Threads of a CUDA warp, the width of the queued kernel's ballots."""
const CU_WARP = 32

"""Entries one warp's queue holds in `_cuda_sf_1d_queued_kernel!`: a full batch of `CU_WARP` and up to `CU_WARP - 1`
more."""
const CU_QUEUE_SLOTS = 2 * CU_WARP

"""Launch plans `(TILE, R)` to try for the table's `(TILE, R)`, in order: it, fewer replicas, then smaller tiles."""
_cuda_1d_steps(TILE::Int, R::Int) = ((TILE, R), (TILE, R ÷ 2), (TILE, R ÷ 4), (256, 8), (256, 2), (128, 1))

"""Static shared bytes of the native 1-D kernel for coordinates of `XT` at width `W`, fields of `UT` at width `F`,
sums of `FT` and counts of `CST`, `NMOM` moments, `TILE`-point tiles, a histogram of `H` bins at stride `S`, and the
warp queues of the queued kernel when its threshold `Q` is positive."""
@inline _cuda_1d_smem_bytes(::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int, F::Int, NMOM::Int,
                            TILE::Int, S::Int, H::Int, Q::Int) where {XT, UT, FT, CST} =
    2 * SFC.gpu_localmem_bytes(XT, W * TILE) + 2 * SFC.gpu_localmem_bytes(UT, F * TILE) +
    SFC.gpu_localmem_bytes(FT, NMOM * H * S) + SFC.gpu_localmem_bytes(CST, H * S) +
    (Q > 0 ? SFC.gpu_localmem_bytes(Int32, (TILE ÷ CU_WARP) * CU_QUEUE_SLOTS) : 0)

"""The first `(TILE, R)` of `plans` with at least one replica whose kernel, at histogram stride `R` and queue
threshold `Q`, fits the device `caps` describes, or `nothing`."""
function _cuda_1d_fitting_plan(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST},
                               W::Int, F::Int, NMOM::Int, H::Int, Q::Int, plans::Tuple) where {XT, UT, FT, CST}
    for (TILE, R) in plans
        R >= 1 || continue
        SFC.gpu_static_smem_fits(caps, _cuda_1d_smem_bytes(XT, UT, FT, CST, W, F, NMOM, TILE, R, H, Q)) &&
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

"""Add the moments of `sf_type` of the pair `(a, b)`, weighted by `pw`, into replica `rl` of bin `bin`."""
@inline function _cuda_1d_accumulate!(ssum, scnt, sf_type, geom, frame, dist, Xa, Xb, Ua, Ub, pw, bin::Int, rl::Int,
                                      NB::Int, ::Val{S}, ::Val{CST}) where {S, CST}
    moments = SFC._sf_pair_moments(sf_type, geom, frame, dist, Xa, Xb, Ua, Ub)
    @inbounds for m in SFC._sf_accum_moments(sf_type)
        CUDA.@atomic ssum[((m - 1) * NB + bin - 1) * S + rl] += pw * moments[m]
    end
    @inbounds CUDA.@atomic scnt[(bin - 1) * S + rl] += CST(pw)
    return nothing
end

"""A warp-queue entry: tile slots `a` and `b` and bin `bin`, packed as `a << 19 | b << 8 | bin`."""
@inline _cuda_queue_entry(a::Int, b::Int, bin::Int) = (Int32(a) << 19) | (Int32(b) << 8) | Int32(bin)

"""Accumulate the pair the queue entry `e` names, its geometry formed again from the staged tiles, into replica `rl`."""
@inline function _cuda_1d_dequeue!(ssum, scnt, e::Int32, sxi, sui, sxj, suj, diag::Bool, i0::Int, jbase::Int, wts,
                                   sf_type, geom, NB::Int, rl::Int, ::Val{W}, ::Val{F}, ::Val{TILE}, ::Val{S},
                                   ::Val{CST}) where {W, F, TILE, S, CST}
    a, b, bin = Int(e >> 19), Int((e >> 8) & Int32(0x7ff)), Int(e & Int32(0xff))
    Xa = _cuda_ld(sxi, Val(W), Val(TILE), a)
    Ua = _cuda_ld(sui, Val(F), Val(TILE), a)
    Xb = diag ? _cuda_ld(sxi, Val(W), Val(TILE), b) : _cuda_ld(sxj, Val(W), Val(TILE), b)
    Ub = diag ? _cuda_ld(sui, Val(F), Val(TILE), b) : _cuda_ld(suj, Val(F), Val(TILE), b)
    _, dist, frame = SFH.pair_frame(geom, Xa, Xb)
    pw = SFC._point_weight(wts, i0 + a - 1) * SFC._point_weight(wts, jbase + b - 1)
    _cuda_1d_accumulate!(ssum, scnt, sf_type, geom, frame, dist, Xa, Xb, Ua, Ub, pw, bin, rl, NB, Val(S), Val(CST))
    return nothing
end

# The tiles, histogram and flush of `_cuda_sf_1d_kernel!`; each warp steps its lanes' points through the `j` tile
# together and ballots the lanes whose pair is in range. With at least `Q` of them each accumulates its own pair; with
# fewer they append their pairs to the warp's queue, and every time it holds a warp's worth each lane accumulates one.
function _cuda_sf_1d_queued_kernel!(
    output, counts, x, u, wts, sf_type, ddig, N::Int, NB::Int, sched, ntb::Int,
    ::Val{W}, ::Val{F}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, ::Val{R}, ::Val{S}, ::Val{H}, ::Val{CST}, ::Val{Q},
    geom,
) where {W, F, NMOM, FIXED_X, TILE, R, S, H, CST, Q}
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
    squeue = CuStaticSharedArray(Int32, (TILE ÷ CU_WARP) * CU_QUEUE_SLOTS)

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

    lane = Int(CUDA.laneid())
    warp = (lid - 1) ÷ CU_WARP
    qbase = warp * CU_QUEUE_SLOTS
    wfirst = warp * CU_WARP + 1
    diag = !(ti < tj)
    jbase = diag ? i0 : j0
    jend = diag ? ni : nj
    if wfirst <= ni
        mine = min(lid, ni)
        Xi = _cuda_ld(sxi, Val(W), Val(TILE), mine)
        Ui = _cuda_ld(sui, Val(F), Val(TILE), mine)
        wi = SFC._point_weight(wts, i0 + mine - 1)
        rl = (lid - 1) % R + 1
        queued = 0
        jj = diag ? wfirst + 1 : 1
        while jj <= jend
            in_range = false
            bin = 0
            Xj = diag ? _cuda_ld(sxi, Val(W), Val(TILE), jj) : _cuda_ld(sxj, Val(W), Val(TILE), jj)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            if lid <= ni && (!diag || jj > lid)
                bin = SFH.digitize(dist, ddig)
                in_range = ok && 1 <= bin <= NB
            end
            ballot = CUDA.vote_ballot_sync(CUDA.FULL_MASK, in_range)
            n_in = Int(count_ones(ballot))
            if n_in >= Q
                if in_range
                    Uj = diag ? _cuda_ld(sui, Val(F), Val(TILE), jj) : _cuda_ld(suj, Val(F), Val(TILE), jj)
                    _cuda_1d_accumulate!(ssum, scnt, sf_type, geom, frame, dist, Xi, Xj, Ui, Uj,
                                         wi * SFC._point_weight(wts, jbase + jj - 1), bin, rl, NB, Val(S), Val(CST))
                end
            else
                if in_range
                    slot = queued + Int(count_ones(ballot & CUDA.lanemask(<))) + 1
                    @inbounds squeue[qbase + slot] = _cuda_queue_entry(lid, jj, bin)
                end
                queued += n_in
                if queued >= CU_WARP
                    CUDA.sync_warp()
                    _cuda_1d_dequeue!(ssum, scnt, @inbounds(squeue[qbase + lane]), sxi, sui, sxj, suj, diag, i0, jbase,
                                      wts, sf_type, geom, NB, rl, Val(W), Val(F), Val(TILE), Val(S), Val(CST))
                    CUDA.sync_warp()
                    if lane <= queued - CU_WARP
                        @inbounds squeue[qbase + lane] = squeue[qbase + CU_WARP + lane]
                    end
                    CUDA.sync_warp()
                    queued -= CU_WARP
                end
            end
            jj += 1
        end
        CUDA.sync_warp()
        if lane <= queued
            _cuda_1d_dequeue!(ssum, scnt, @inbounds(squeue[qbase + lane]), sxi, sui, sxj, suj, diag, i0, jbase, wts,
                              sf_type, geom, NB, rl, Val(W), Val(F), Val(TILE), Val(S), Val(CST))
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


"""Launch plan of the native 1-D kernel: coordinate width `W`, field width `F`, `NMOM` moments, `TILE`-point tiles,
`R` histogram replicas at stride `S`, bin capacity `H`, shared counts of `CST`, and the kernel: `_cuda_sf_1d_kernel!`
when `Q` is 0, `_cuda_sf_1d_queued_kernel!` with ballot threshold `Q` otherwise."""
struct CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST, Q}
    function CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST, Q}() where {W, F, NMOM, TILE, R, S, H, CST, Q}
        S >= R || throw(ArgumentError("a shared histogram stride of $S does not cover $R replicas"))
        Q == 0 || (TILE % CU_WARP == 0 && TILE < 2^11 && H < 2^8) ||
            throw(ArgumentError("a queued plan needs whole warps, tiles below 2048 points and fewer than 256 bins"))
        return new{W, F, NMOM, TILE, R, S, H, CST, Q}()
    end
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

"""Strip plans `(TILE, SW, R)` to try for the table's, in order: it, fewer replicas, narrower strips, then small ones."""
_cuda_1d_strip_steps(TILE::Int, SW::Int, R::Int) =
    ((TILE, SW, R), (TILE, SW, R ÷ 2), (TILE, SW ÷ 2, R), (256, 4, 1), (256, 2, 1), (128, 2, 1))

"""The plan of the table's `(TILE, R, Q)` at its first fitting step ([`_cuda_1d_steps`](@ref)) on the device `caps`
describes, else the direct kernel's (`Q = 0`) first fitting step, else `nothing`."""
function _cuda_1d_fit(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int, F::Int,
                      NMOM::Int, H::Int, (TILE, R, Q)::NTuple{3, Int}) where {XT, UT, FT, CST}
    for q in unique((Q, 0))
        launch = _cuda_1d_fitting_plan(caps, XT, UT, FT, CST, W, F, NMOM, H, q, _cuda_1d_steps(TILE, R))
        launch === nothing || return CUDA1DPlan{W, F, NMOM, launch[1], launch[2], launch[2], H, CST, q}()
    end
    return nothing
end

"""The strip plan of the table's `(:strip, TILE, SW, R)` at its first fitting step ([`_cuda_1d_strip_steps`](@ref))
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

"""Ballot count at which a warp of the queued kernel accumulates its in-range pairs directly."""
const CU_QUEUE_THRESHOLD = 24

"""Replica scale of the native 1-D histogram ([`_cuda_replicas`](@ref)), by moment count (one, or several)."""
const CU_1D_REPLICA_SCALE = (0.7, 2.8)

"""In-range shares that switch the native 1-D kernel of several moments: below the first the direct kernel with one
replica, below the second the queued kernel, above it the direct kernel."""
const CU_1D_QUEUE_SHARES = (0.005, 0.7)

"""In-range share at most which a batch over shared positions also tries the strip kernel."""
const CU_STRIP_SHARE = 0.2

"""Pair evaluations from which a native 1-D call samples its in-range share and times its candidates
([`CUDAChoice`](@ref)), by moment count (one, or several)."""
const CU_1D_CHOOSE_FROM = (6.0e7, 1.5e6)

"""Replicas of a shared histogram of `H` bins and `NMOM` moments for `TILE`-point tiles at in-range share `share`:
`c · TILE · √(share / (H · NMOM))` as a power of two in `1:32`, balancing the shared atomics a block's in-range pairs
contend on against the global atomics its flush makes."""
_cuda_replicas(c, TILE::Int, share::Real, H::Int, NMOM::Int) =
    clamp(prevpow(2, max(1, floor(Int, c * TILE * sqrt(share / (H * NMOM))))), 1, 32)

"""The largest of `tiles` whose tile pairs over `N` points and `B` slices reach `k` blocks per multiprocessor of the
device `caps` describes, else 128."""
function _cuda_tile_for(caps::SFC.GPUDeviceCaps, N::Int, B::Int, k::Int, tiles::Tuple)
    for t in tiles
        _cuda_blocks(N, t) * B >= k * caps.n_sms && return t
    end
    return 128
end

"""The candidate plans of the native 1-D kernel ([`CUDAChoice`](@ref)) for coordinates of `XT` at width `W`, fields
of `UT` at width `F`, sums of `FT`, shared counts of `CST`, `NMOM` moments and bin capacity `H` on the device `caps`
describes: tile 256 when its tile pairs reach 8 blocks per multiprocessor (384 for one moment at in-range share
≥ 0.6), else 128; replicas by [`_cuda_replicas`](@ref); for several moments the direct and the queued kernel by
[`CU_1D_QUEUE_SHARES`](@ref), the other one second; for a batch over shared positions at a share of at most
[`CU_STRIP_SHARE`](@ref), the strip kernel first. Each plan steps down until it fits."""
function _cuda_1d_candidates(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int,
                             F::Int, NMOM::Int, H::Int) where {XT, UT, FT, CST}
    fit(spec) = _cuda_1d_fit(caps, XT, UT, FT, CST, W, F, NMOM, H, spec)
    c = CU_1D_REPLICA_SCALE[NMOM == 1 ? 1 : 2]
    function point(N, B, share)
        if NMOM == 1
            t = _cuda_tile_for(caps, N, B, 8, share >= 0.6 ? (384, 256) : (256,))
            return (fit((t, _cuda_replicas(c, t, share, H, 1), 0)),)
        end
        share < CU_1D_QUEUE_SHARES[1] && return (fit((_cuda_tile_for(caps, N, B, 8, (256,)), 1, 0)),)
        t = _cuda_tile_for(caps, N, B, 2, (256,))
        r = _cuda_replicas(c, t, share, H, NMOM)
        q = share < CU_1D_QUEUE_SHARES[2] ? CU_QUEUE_THRESHOLD : 0
        return (fit((t, r, q)), fit((t, r, CU_QUEUE_THRESHOLD - q)))
    end
    return function (N::Int, B::Int, fixed::Bool, evaluations::Real, share::Real)
        plans = point(N, B, share)
        if fixed && share <= CU_STRIP_SHARE && B >= (NMOM == 1 ? 2 : 4)
            plans = (fit((:strip, 256, min(prevpow(2, B), 8), min(_cuda_replicas(c, 256, share, H, NMOM), 4))), plans...)
        end
        return unique(filter(!isnothing, collect(Any, plans)))
    end
end

"""The native 1-D plans ([`CUDAChoice`](@ref)) for the moment set `M` on the device `caps` describes, or `nothing`
when `NB` exceeds `CU_MAX_BINS`, the sum or count type has no native atomic add, or no plan fits the device."""
function _cuda_1d_plan(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CT}, wts, geom,
                       NB::Int, M) where {XT, UT, OT, CT}
    (NB <= CU_MAX_BINS && _cuda_atomic_add(OT) && _cuda_atomic_add(CT)) || return nothing
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFC._sf_field_width(M, geom))
    CST = _cuda_shared_count_type(wts, CT)
    NMOM = SFC._sf_nmom(M)
    H = SFC._val_int(_cuda_1d_bin_capacity(NB))
    _cuda_1d_fit(caps, XT, UT, OT, CST, W, F, NMOM, H, (128, 1, 0)) === nothing && return nothing
    return CUDAChoice(_cuda_1d_candidates(caps, XT, UT, OT, CST, W, F, NMOM, H), CU_1D_CHOOSE_FROM[NMOM == 1 ? 1 : 2])
end

"""The tile of a native 1-D plan."""
_cuda_tile(::CUDA1DPlan{W, F, NMOM, TILE}) where {W, F, NMOM, TILE} = TILE
_cuda_tile(::CUDA1DStripPlan{W, F, NMOM, TILE}) where {W, F, NMOM, TILE} = TILE

"""Launch the plan of `choice` for this call ([`_cuda_plan`](@ref))."""
function _cuda_launch_1d!(choice::CUDAChoice, out, cnt, x, u, wts, sf_type, ddig, N::Int, NB::Int, B::Int,
                          fixed_x::Bool, geom, cull)
    launch!(plan, s, c) = _cuda_launch_1d!(plan, s, c, x, u, wts, sf_type, ddig, N, NB, B, fixed_x, geom, cull)
    return launch!(_cuda_plan(launch!, choice, out, cnt, x, ddig, NB, N, B, fixed_x && B > 1, geom, cull), out, cnt)
end

"""Launch the native 1-D kernel `plan` names into `out`/`cnt` `(NMOM, NB, B)`; `x` is `(W,N,B)` varying or
`(W,N)`/`(W,N,1)` fixed, `u` is `(F,N,B)`."""
function _cuda_launch_1d!(::CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST, Q}, out, cnt, x, u, wts, sf_type, ddig,
                          N::Int, NB::Int, B::Int, fixed_x::Bool, geom,
                          cull) where {W, F, NMOM, TILE, R, S, H, CST, Q}
    NB <= H || throw(ArgumentError("$NB bins exceed the plan's shared histogram capacity $H"))
    xv = fixed_x ? reshape(x, W, N, 1) : reshape(x, W, N, B)
    uv = reshape(u, F, N, B)
    sched = SFC.schedule_for(cull, N, TILE)
    ntb = SFC.n_pair_blocks(sched)
    fx = fixed_x ? Val(true) : Val(false)
    args = (out, cnt, xv, uv, wts, sf_type, ddig, N, NB, sched, ntb,
            Val(W), Val(F), Val(NMOM), fx, Val(TILE), Val(R), Val(S), Val(H), Val(CST))
    kern = Q == 0 ? (@cuda launch=false _cuda_sf_1d_kernel!(args..., geom)) :
                    (@cuda launch=false _cuda_sf_1d_queued_kernel!(args..., Val(Q), geom))
    CUDA.attributes(kern.fun)[CUDA.FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT] = CU_CARVEOUT_MAX_SHARED
    Q == 0 ? kern(args..., geom; threads = TILE, blocks = ntb * B) :
             kern(args..., Val(Q), geom; threads = TILE, blocks = ntb * B)
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
