# =============================================================================
# CUDA-specialized 1D structure-function kernel (distance histogram only).
#
# Same N-body broadcast structure as the 2D kernel. Histograms with at most 128
# bins use compile-time shared-memory capacity classes. Each class reserves only
# enough storage for its bin range, leaving the remaining block budget available
# for pair tiles and replicated accumulator lanes.
# Covers individual (NMOM=1) and single-pass (NMOM=6), fixed-x and varying-x, weighted or not, at
# the coordinate and field widths of the geometry. Output is (NMOM, NB, B); counts are per-bin
# (shared across moments).
# =============================================================================

"""Compiled-in cap on distance bins for the static shared 1D histogram."""
const CU_MAX_BINS = 128
"""Block size for the 1D N-body kernel."""
const CU_TILE_1D = 256

# The 64-bin S2 plan is the measured A100 optimum for N=20_000, D=2, Float32.
# Other capacity classes retain the conservative launch until their workload
# sweeps establish a better plan.
"""Launch plans `(TILE, R, S)` of the `NMOM`-moment kernel at bin capacity `H`, preferred first."""
@inline _cuda_1d_launch_plans(::Val{1}, ::Val{64}) =
    ((Val(384), Val(32), Val(32)), (Val(CU_TILE_1D), Val(8), Val(8)))
@inline _cuda_1d_launch_plans(::Val{1}, ::Val{H}) where {H} = ((Val(CU_TILE_1D), Val(8), Val(8)),)
@inline _cuda_1d_launch_plans(::Val{6}, ::Val{H}) where {H} = ((Val(CU_TILE_1D), Val(2), Val(2)),)

"""Static shared bytes of `_cuda_sf_1d_kernel!` for coordinates of `XT` at width `W`, fields of `UT` at
width `F`, sums of `FT` and counts of `CST`, `NMOM` moments, `TILE`-point tiles and a histogram of `H`
bins at stride `S`."""
@inline _cuda_1d_smem_bytes(::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int, F::Int, NMOM::Int,
                            TILE::Int, S::Int, H::Int) where {XT, UT, FT, CST} =
    2 * SFC.gpu_localmem_bytes(XT, W * TILE) + 2 * SFC.gpu_localmem_bytes(UT, F * TILE) +
    SFC.gpu_localmem_bytes(FT, NMOM * H * S) + SFC.gpu_localmem_bytes(CST, H * S)

"""The first of `plans` whose kernel fits the device `caps` describes, or `nothing`."""
function _cuda_1d_fitting_plan(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST},
                               W::Int, F::Int, NMOM::Int, H::Int, plans::Tuple) where {XT, UT, FT, CST}
    for plan in plans
        TILE, S = SFC._val_int(plan[1]), SFC._val_int(plan[3])
        SFC.gpu_static_smem_fits(caps, _cuda_1d_smem_bytes(XT, UT, FT, CST, W, F, NMOM, TILE, S, H)) &&
            return plan
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
                dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                moments = GE._sf_moments(Val(NMOM), sf_type, geom, frame, dist, dU)
                pw = wi * SFC._point_weight(wts, jbase + jj - 1)
                @inbounds for m in GE._sf_accum_moments(Val(NMOM))
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
        s = _cuda_flush_replicated(Val(NMOM), ssum, NB, m, bin, Val(R), Val(S))
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


"""Launch plan of `_cuda_sf_1d_kernel!`: coordinate width `W`, field width `F`, `NMOM` moments,
`TILE`-point tiles, `R` histogram replicas at stride `S`, bin capacity `H` and shared counts of `CST`."""
struct CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST}
    function CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST}() where {W, F, NMOM, TILE, R, S, H, CST}
        S >= R || throw(ArgumentError("a shared histogram stride of $S does not cover $R replicas"))
        return new{W, F, NMOM, TILE, R, S, H, CST}()
    end
end

"""The plan of `_cuda_sf_1d_kernel!` on the device `caps` describes, or `nothing` when `NB` exceeds
`CU_MAX_BINS`, `NMOM` is neither 1 nor 6, the sum or count type has no native atomic add, or no launch
plan fits."""
function _cuda_1d_plan(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CT}, wts, geom,
                       NB::Int, NMOM::Int) where {XT, UT, OT, CT}
    (NB <= CU_MAX_BINS && (NMOM == 1 || NMOM == 6) && _cuda_atomic_add(OT) && _cuda_atomic_add(CT)) ||
        return nothing
    vW, vF = SFH.coordinate_width(geom), SFH.field_width(geom)
    CST = _cuda_shared_count_type(wts, CT)
    Mv = NMOM == 6 ? Val(6) : Val(1)
    Hv = _cuda_1d_bin_capacity(NB)
    launch = _cuda_1d_fitting_plan(caps, XT, UT, OT, CST, SFC._val_int(vW), SFC._val_int(vF), NMOM,
                                   SFC._val_int(Hv), _cuda_1d_launch_plans(Mv, Hv))
    launch === nothing && return nothing
    return _cuda_1d_plan_of(vW, vF, Mv, launch..., Hv, Val(CST))
end

@inline _cuda_1d_plan_of(::Val{W}, ::Val{F}, ::Val{NMOM}, ::Val{TILE}, ::Val{R}, ::Val{S}, ::Val{H},
                         ::Val{CST}) where {W, F, NMOM, TILE, R, S, H, CST} =
    CUDA1DPlan{W, F, NMOM, TILE, R, S, H, CST}()

"""Launch `_cuda_sf_1d_kernel!` with `plan` into `out`/`cnt` `(NMOM, NB, B)`; `x` is `(W,N,B)` varying or
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
    @cuda threads=TILE blocks=ntb*B _cuda_sf_1d_kernel!(
        out, cnt, xv, uv, wts, sf_type, ddig, N, NB, sched, ntb,
        Val(W), Val(F), Val(NMOM), fx, Val(TILE), Val(R), Val(S), Val(H), Val(CST), geom)
    return nothing
end
