# =============================================================================
# CUDA-specialized 1D structure-function kernel (distance histogram only).
#
# Same N-body broadcast structure as the 2D kernel, but the histogram is tiny
# (NMOM·NB, NB ≤ 128) so it lives in STATIC shared memory (no dynamic-shared
# opt-in needed) and TILE = 256 (1D is geometry-bound, not occupancy-limited by
# the histogram — measured: N-body gives the ~2× win here, replication R=1).
# Covers individual (NMOM=1) and single-pass (NMOM=6), fixed-x and varying-x,
# D ∈ {2,3}. Output is (NMOM, NB, B); counts are per-bin (shared across moments).
# =============================================================================

"""Compiled-in cap on distance bins for the static shared 1D histogram."""
const CU_MAX_BINS = 128
"""Block size for the 1D N-body kernel (settled design)."""
const CU_TILE_1D = 256

function _cuda_sf_1d_kernel!(
    output,                 # (NMOM, NB, B)
    counts,                 # (NMOM, NB, B)
    x,                      # (D, N, B) varying / (D, N, 1) fixed
    u,                      # (D, N, B)
    sf_type,
    ddig,                   # distance digitizer functor
    N::Int, NB::Int,
    sched, ntb::Int,
    ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE},
    geom,
) where {D, NMOM, FIXED_X, TILE}
    FT = eltype(output)
    lid = Int(threadIdx().x)
    wg = Int(blockDim().x)
    lb = Int(blockIdx().x)
    bid = (lb - 1) % ntb + 1
    b = (lb - 1) ÷ ntb + 1

    sxi = CuStaticSharedArray(FT, D * TILE)
    sxj = CuStaticSharedArray(FT, D * TILE)
    sui = CuStaticSharedArray(FT, D * TILE)
    suj = CuStaticSharedArray(FT, D * TILE)
    ssum = CuStaticSharedArray(FT, NMOM * CU_MAX_BINS)
    scnt = CuStaticSharedArray(UInt32, CU_MAX_BINS)

    c = lid
    while c <= NMOM * NB
        @inbounds ssum[c] = zero(FT)
        c += wg
    end
    c = lid
    while c <= NB
        @inbounds scnt[c] = UInt32(0)
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
        @inbounds for d in 1:D
            sxi[(d - 1) * TILE + lid] = x[d, gi, xb]
            sui[(d - 1) * TILE + lid] = u[d, gi, b]
        end
    end
    if ti < tj && lid <= nj
        @inbounds gj = j0 + lid - 1
        @inbounds for d in 1:D
            sxj[(d - 1) * TILE + lid] = x[d, gj, xb]
            suj[(d - 1) * TILE + lid] = u[d, gj, b]
        end
    end
    sync_threads()

    if lid <= ni
        Xi = _cuda_ld(sxi, Val(D), Val(TILE), lid)
        Ui = _cuda_ld(sui, Val(D), Val(TILE), lid)
        diag = !(ti < tj)
        jj = diag ? lid + 1 : 1
        jend = diag ? ni : nj
        while jj <= jend
            if diag
                Xj = _cuda_ld(sxi, Val(D), Val(TILE), jj)
                Uj = _cuda_ld(sui, Val(D), Val(TILE), jj)
            else
                Xj = _cuda_ld(sxj, Val(D), Val(TILE), jj)
                Uj = _cuda_ld(suj, Val(D), Val(TILE), jj)
            end
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            bin = ddig(dist)
            if ok && 1 <= bin <= NB
                dU, rhat = SFH.pair_increments(geom, frame, dist, Xi, Xj, Ui, Uj)
                moments = GE._sf_moments(Val(NMOM), sf_type, dU, rhat)
                @inbounds for m in GE._sf_accum_moments(Val(NMOM))
                    CUDA.@atomic ssum[(m - 1) * NB + bin] += moments[m]
                end
                @inbounds CUDA.@atomic scnt[bin] += UInt32(1)
            end
            jj += 1
        end
    end
    sync_threads()

    cell = lid
    while cell <= NMOM * NB
        m = (cell - 1) ÷ NB + 1
        bin = (cell - 1) % NB + 1
        s = GE._sf_flush_moment(Val(NMOM), ssum, NB, m, bin)
        if s != zero(FT)
            CUDA.@atomic output[m, bin, b] += s
        end
        cell += wg
    end
    bcell = lid
    while bcell <= NB
        @inbounds cnt = scnt[bcell]
        if cnt != UInt32(0)
            @inbounds for m in 1:NMOM
                CUDA.@atomic counts[m, bcell, b] += cnt
            end
        end
        bcell += wg
    end
    return nothing
end


"""Launch the CUDA 1D fast kernel. Returns `true` if launched, `false` if NB
exceeds the static-shared cap (caller uses the KA fallback). `out`/`cnt` are
`(NMOM, NB, B)`; `x` is `(D,N,B)` varying or `(D,N)`/`(D,N,1)` fixed; `u` is `(D,N,B)`."""
function _cuda_launch_1d!(out, cnt, x, u, sf_type, ddig,
                          N::Int, NB::Int, B::Int, D::Int, NMOM::Int, fixed_x::Bool, geom, cull)
    NB > CU_MAX_BINS && return false
    xv = fixed_x ? reshape(x, D, N, 1) : reshape(x, D, N, B)
    uv = reshape(u, D, N, B)
    _cuda_launch_1d_specialized!(out, cnt, xv, uv, sf_type, ddig, N, NB, B, D, NMOM, fixed_x, geom,
                                 cull)
    return true
end


function _cuda_launch_1d_specialized!(out, cnt, x, u, sf_type, ddig, N, NB, B, D, NMOM, fixed_x, geom,
                                      cull)
    Dv = D == 3 ? Val(3) : Val(2)
    Mv = NMOM == 6 ? Val(6) : Val(1)
    Fv = fixed_x ? Val(true) : Val(false)
    _cuda_launch_1d_valed!(out, cnt, x, u, sf_type, ddig, N, NB, B, Dv, Mv, Fv, Val(CU_TILE_1D), geom,
                           cull)
    return nothing
end

function _cuda_launch_1d_valed!(out, cnt, x, u, sf_type, ddig, N, NB, B,
                                ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, geom,
                                cull) where {D, NMOM, FIXED_X, TILE}
    sched = SFC.schedule_for(cull, N, TILE)
    ntb = SFC.n_pair_blocks(sched)
    @cuda threads=TILE blocks=ntb*B _cuda_sf_1d_kernel!(
        out, cnt, x, u, sf_type, ddig, N, NB, sched, ntb,
        Val(D), Val(NMOM), Val(FIXED_X), Val(TILE), geom)
    return nothing
end
