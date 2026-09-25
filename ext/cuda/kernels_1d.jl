# =============================================================================
# CUDA-specialized 1D structure-function kernel (distance histogram only).
#
# Same N-body broadcast structure as the 2D kernel. Histograms with at most 128
# bins use compile-time shared-memory capacity classes. Each class reserves only
# enough storage for its bin range, leaving the remaining block budget available
# for pair tiles and replicated accumulator lanes.
# Covers individual (NMOM=1) and single-pass (NMOM=6), fixed-x and varying-x,
# D ∈ {2,3}. Output is (NMOM, NB, B); counts are per-bin (shared across moments).
# =============================================================================

"""Compiled-in cap on distance bins for the static shared 1D histogram."""
const CU_MAX_BINS = 128
"""Block size for the 1D N-body kernel."""
const CU_TILE_1D = 256

# Shared atomics are striped across lanes chosen from the thread id. These are
# compile-time parameters so the launch planner can balance atomic contention,
# shared storage, and tile size without duplicating the pair kernel.
@inline _cuda_1d_replica_plan(::Val{1}) = (Val(8), Val(8))
@inline _cuda_1d_replica_plan(::Val{6}) = (Val(2), Val(2))

# The 64-bin S2 plan is the measured A100 optimum for N=20_000, D=2, Float32.
# Other capacity classes retain the conservative launch until their workload
# sweeps establish a better plan.
@inline _cuda_1d_launch_plan(::Val{1}, ::Val{64}) = (Val(384), Val(32), Val(32))
@inline _cuda_1d_launch_plan(::Val{1}, ::Val{H}) where {H} =
    (Val(CU_TILE_1D), Val(8), Val(8))
@inline _cuda_1d_launch_plan(::Val{6}, ::Val{H}) where {H} =
    (Val(CU_TILE_1D), Val(2), Val(2))

@inline function _cuda_1d_bin_capacity(NB::Int)
    NB <= 16 && return Val(16)
    NB <= 32 && return Val(32)
    NB <= 64 && return Val(64)
    return Val(128)
end

function _cuda_sf_1d_kernel!(
    output,                 # (NMOM, NB, B)
    counts,                 # (NMOM, NB, B)
    x,                      # (D, N, B) varying / (D, N, 1) fixed
    u,                      # (D, N, B)
    sf_type,
    ddig,                   # device digitize plan of the distance bins
    N::Int, NB::Int,
    sched, ntb::Int,
    ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, ::Val{R}, ::Val{S}, ::Val{H},
    geom,
) where {D, NMOM, FIXED_X, TILE, R, S, H}
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
    ssum = CuStaticSharedArray(FT, NMOM * H * S)
    scnt = CuStaticSharedArray(UInt32, H * S)

    c = lid
    while c <= NMOM * NB * S
        @inbounds ssum[c] = zero(FT)
        c += wg
    end
    c = lid
    while c <= NB * S
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
        lane = (lid - 1) % R + 1
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
            bin = SFH.digitize(dist, ddig)
            if ok && 1 <= bin <= NB
                dU, rhat = SFH.pair_increments(geom, frame, dist, Xi, Xj, Ui, Uj)
                moments = GE._sf_moments(Val(NMOM), sf_type, dU, rhat)
                @inbounds for m in GE._sf_accum_moments(Val(NMOM))
                    CUDA.@atomic ssum[((m - 1) * NB + bin - 1) * S + lane] += moments[m]
                end
                @inbounds CUDA.@atomic scnt[(bin - 1) * S + lane] += UInt32(1)
            end
            jj += 1
        end
    end
    sync_threads()

    cell = lid
    while cell <= NMOM * NB
        m = (cell - 1) ÷ NB + 1
        bin = (cell - 1) % NB + 1
        s = _cuda_flush_replicated(Val(NMOM), ssum, NB, m, bin, Val(R), Val(S))
        if s != zero(FT)
            CUDA.@atomic output[m, bin, b] += s
        end
        cell += wg
    end
    bcell = lid
    while bcell <= NB
        cnt = _cuda_sum_replicas(scnt, (bcell - 1) * S, Val(R))
        if cnt != UInt32(0)
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

@inline function _cuda_flush_replicated(::Val{1}, sums, NB::Int, ::Int, bin::Int,
                                        ::Val{R}, ::Val{S}) where {R, S}
    return _cuda_sum_replicas(sums, (bin - 1) * S, Val(R))
end

@inline function _cuda_flush_replicated(::Val{6}, sums, NB::Int, m::Int, bin::Int,
                                        ::Val{R}, ::Val{S}) where {R, S}
    m1 = m == 3 ? 1 : m == 6 ? 4 : m
    value = _cuda_sum_replicas(sums, ((m1 - 1) * NB + bin - 1) * S, Val(R))
    if m == 3 || m == 6
        value -= _cuda_sum_replicas(sums, (m1 * NB + bin - 1) * S, Val(R))
    end
    return value
end


"""Launch the CUDA 1D fast kernel. Returns `true` if launched, `false` if NB exceeds the
static-shared cap or the width is not one this kernel is compiled for (caller uses the KA
fallback). `out`/`cnt` are `(NMOM, NB, B)`; `x` is `(D,N,B)` varying or `(D,N)`/`(D,N,1)` fixed;
`u` is `(D,N,B)`."""
function _cuda_launch_1d!(out, cnt, x, u, sf_type, ddig,
                          N::Int, NB::Int, B::Int, D::Int, NMOM::Int, fixed_x::Bool, geom, cull)
    NB > CU_MAX_BINS && return false
    (D == 2 || D == 3) || return false
    xv = fixed_x ? reshape(x, D, N, 1) : reshape(x, D, N, B)
    uv = reshape(u, D, N, B)
    _cuda_launch_1d_specialized!(out, cnt, xv, uv, sf_type, ddig, N, NB, B, D, NMOM, fixed_x, geom,
                                 cull)
    return true
end


function _cuda_launch_1d_specialized!(out, cnt, x, u, sf_type, ddig, N, NB, B, D, NMOM, fixed_x, geom,
                                      cull)
    Dv = D == 2 ? Val(2) : D == 3 ? Val(3) : error(
        "CUDA 1D fast kernel is compiled for D ∈ {2,3}; a caller must decline other widths " *
        "rather than reach here (got D=$D)")
    Mv = NMOM == 6 ? Val(6) : Val(1)
    Fv = fixed_x ? Val(true) : Val(false)
    Hv = _cuda_1d_bin_capacity(NB)
    tile, replicas, stride = _cuda_1d_launch_plan(Mv, Hv)
    _cuda_launch_1d_valed_capacity!(
        out, cnt, x, u, sf_type, ddig, N, NB, B,
        Dv, Mv, Fv, tile, replicas, stride, Hv, geom, cull,
    )
    return nothing
end

function _cuda_launch_1d_valed!(out, cnt, x, u, sf_type, ddig, N, NB, B,
                                ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, geom,
                                cull) where {D, NMOM, FIXED_X, TILE}
    replicas, stride = _cuda_1d_replica_plan(Val(NMOM))
    return _cuda_launch_1d_valed_striped!(
        out, cnt, x, u, sf_type, ddig, N, NB, B,
        Val(D), Val(NMOM), Val(FIXED_X), Val(TILE), replicas, stride, geom, cull,
    )
end

function _cuda_launch_1d_valed_replicated!(out, cnt, x, u, sf_type, ddig, N, NB, B,
                                           ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X},
                                           ::Val{TILE}, ::Val{R}, geom,
                                           cull) where {D, NMOM, FIXED_X, TILE, R}
    return _cuda_launch_1d_valed_striped!(
        out, cnt, x, u, sf_type, ddig, N, NB, B,
        Val(D), Val(NMOM), Val(FIXED_X), Val(TILE), Val(R), Val(R), geom, cull,
    )
end

function _cuda_launch_1d_valed_striped!(out, cnt, x, u, sf_type, ddig, N, NB, B,
                                        ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X},
                                        ::Val{TILE}, ::Val{R}, ::Val{S}, geom,
                                        cull) where {D, NMOM, FIXED_X, TILE, R, S}
    S >= R || throw(ArgumentError("shared histogram stride must cover every replica"))
    return _cuda_launch_1d_valed_capacity!(
        out, cnt, x, u, sf_type, ddig, N, NB, B,
        Val(D), Val(NMOM), Val(FIXED_X), Val(TILE), Val(R), Val(S),
        _cuda_1d_bin_capacity(NB), geom, cull,
    )
end

function _cuda_launch_1d_valed_capacity!(out, cnt, x, u, sf_type, ddig, N, NB, B,
                                         ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X},
                                         ::Val{TILE}, ::Val{R}, ::Val{S}, ::Val{H}, geom,
                                         cull) where {D, NMOM, FIXED_X, TILE, R, S, H}
    NB <= H || throw(ArgumentError("$NB bins exceed shared histogram capacity $H"))
    sched = SFC.schedule_for(cull, N, TILE)
    ntb = SFC.n_pair_blocks(sched)
    @cuda threads=TILE blocks=ntb*B _cuda_sf_1d_kernel!(
        out, cnt, x, u, sf_type, ddig, N, NB, sched, ntb,
        Val(D), Val(NMOM), Val(FIXED_X), Val(TILE), Val(R), Val(S), Val(H), geom)
    return nothing
end
