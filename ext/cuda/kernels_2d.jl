# =============================================================================
# CUDA-specialized 2D structure-function kernel (distance × value histogram).
#
# N-body broadcast structure with a privatized dynamic-shared histogram:
#   - each thread owns its point i in registers and loops j over the staged tile,
#     so all lanes read the same shared[j] at each step: a broadcast, with no bank
#     conflict and no per-pair pair-decode sqrt;
#   - the per-block histogram (sums + counts) lives in dynamic shared memory and is
#     atomic-merged into the global output at block end;
#   - TILE = block size, the largest of 1024, 512, 256 whose staging plus the dynamic histogram fit
#     the device's opt-in maximum; the plan is `nothing` when even 256 does not.
#
# Covers joint 2D (NMOM=1) and single-pass 2D (NMOM=6), fixed-x and varying-x, weighted or not, at
# the coordinate and field widths of the geometry, all through Val type parameters.
# =============================================================================

"""
    _cuda_val_stride(n_val)

Row stride of the value axis in the privatized histogram, forced odd.

Shared memory has 32 banks. With a power-of-two `n_val`, the flat index `(dbin-1)*n_val + vbin` puts
every value-axis row into the same couple of banks, so lanes differing only in `dbin` serialize on
bank conflicts even though they target different cells. An odd stride is coprime with 32 and spreads
them across banks, at the cost of one unused column per row.
"""
@inline _cuda_val_stride(n_val::Int) = isodd(n_val) ? n_val : n_val + 1

"""Load local point `k`'s `W`-vector from a tile buffer staged as `(d-1)*TILE + k`."""
@inline _cuda_ld(buf, ::Val{W}, ::Val{TILE}, k::Integer) where {W, TILE} =
    SA.SVector{W}(ntuple(d -> @inbounds(buf[(d - 1) * TILE + k]), Val(W)))

"""Shared count element of a native kernel: `UInt32` unweighted, where one block's pairs fit it; the
output's own count type for a pair mass."""
@inline _cuda_shared_count_type(::SFC.NoWeights, ::Type) = UInt32
@inline _cuda_shared_count_type(wts, ::Type{CT}) where {CT} = CT

"""Whether `CUDA.@atomic` adds elements of `T` with a native atomic instruction."""
@inline _cuda_atomic_add(::Type{T}) where {T} = T in (Int32, Int64, UInt32, UInt64, Float32, Float64)

"""Byte offset of the dynamic count plane: after the sum plane, aligned for any element."""
@inline _cuda_count_plane_offset(::Type{FT}, cells::Int) where {FT} = cld(cells * sizeof(FT), 16) * 16

function _cuda_sf_2d_kernel!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    x,                      # (W, N, B) varying  /  (W, N, 1) fixed (reshaped by launcher)
    u,                      # (F, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    ddig,                   # device digitize plan of the distance bins
    vplan,                  # device value plan: one digitize plan, or one per moment
    N::Int, n_dist::Int, n_val::Int,
    sched, ntb::Int,
    ::Val{W}, ::Val{F}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, ::Val{HCELLS}, ::Val{CST},
    geom, second_axis,
) where {W, F, NMOM, FIXED_X, TILE, HCELLS, CST}
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
    vstride = _cuda_val_stride(n_val)
    sums = CuDynamicSharedArray(FT, NMOM * HCELLS)
    cnts = CuDynamicSharedArray(CST, NMOM * HCELLS, _cuda_count_plane_offset(FT, NMOM * HCELLS))

    # zero the privatized histogram
    c = lid
    while c <= NMOM * HCELLS
        @inbounds sums[c] = zero(FT)
        @inbounds cnts[c] = zero(CST)
        c += wg
    end

    ti, tj = SFC.tile_for(sched, bid)
    i0 = (ti - 1) * TILE + 1
    j0 = (tj - 1) * TILE + 1
    ni = min(TILE, N - i0 + 1)
    nj = min(TILE, N - j0 + 1)
    xb = FIXED_X ? 1 : b

    # stage tile i (x for geometry, u for velocity)
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

    # N-body broadcast: thread owns point lid, loops j over the staged tile
    if lid <= ni
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
            dbin = SFH.digitize(dist, ddig)
            if ok && 1 <= dbin <= n_dist
                dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                moments = GE._sf_moments(Val(NMOM), sf_type, geom, frame, dist, dU)
                pw = wi * SFC._point_weight(wts, jbase + jj - 1)
                @inbounds for m in 1:NMOM
                    vb = GE._sf_value_bin(vplan, SFC.pair_axis_key(second_axis, moments[m], Xi, Xj, dist), m)
                    if 1 <= vb <= n_val
                        cell = (m - 1) * HCELLS + (dbin - 1) * vstride + vb
                        CUDA.@atomic sums[cell] += pw * moments[m]
                        CUDA.@atomic cnts[cell] += CST(pw)
                    end
                end
            end
            jj += 1
        end
    end
    sync_threads()

    # atomic-merge the privatized histogram into the global output
    cell = lid
    while cell <= NMOM * HCELLS
        @inbounds s = sums[cell]
        @inbounds cc = cnts[cell]
        m = (cell - 1) ÷ HCELLS + 1
        lc = (cell - 1) % HCELLS
        dbin = lc ÷ vstride + 1
        vb = lc % vstride + 1
        # vb > n_val is a padding column, never written
        if cc != zero(CST) && vb <= n_val
            CUDA.@atomic output[m, dbin, vb, b] += s
            CUDA.@atomic counts[m, dbin, vb, b] += cc
        end
        cell += wg
    end
    return nothing
end

"""Largest TILE ∈ (1024,512,256) whose static staging (coordinates of `XT` at width `W`, fields of `UT`
at width `F`) fits the static budget and whose staging plus the full `NMOM`-plane dynamic histogram
(sums of `FT`, counts of `CST`) fit the opt-in maximum of the device `caps` describes; 0 if even 256
won't fit.

Returning 0 hands the call to the global-atomic kernel, which wins whenever the histogram does not
fit on chip: a histogram that large spreads its contention over many cells. The test is
all-or-nothing, so a histogram is either privatized whole or left in global memory."""
function _cuda_2d_pick_tile(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST},
                            W::Int, F::Int, NMOM::Int, n_dist::Int, n_val::Int) where {XT, UT, FT, CST}
    cells = NMOM * n_dist * _cuda_val_stride(n_val)
    dynb = _cuda_count_plane_offset(FT, cells) + cells * sizeof(CST)
    for TILE in (1024, 512, 256)
        staging = 2 * SFC.gpu_localmem_bytes(XT, W * TILE) + 2 * SFC.gpu_localmem_bytes(UT, F * TILE)
        if SFC.gpu_static_smem_fits(caps, staging) && staging + dynb <= caps.smem_optin
            return TILE, dynb
        end
    end
    return 0, dynb
end

"""Launch plan of `_cuda_sf_2d_kernel!`: coordinate width `W`, field width `F`, `NMOM` moments,
`TILE`-point tiles and shared counts of `CST`, for a histogram of `hcells` cells per moment held in
`dynb` bytes of dynamic shared memory."""
struct CUDA2DPlan{W, F, NMOM, TILE, CST}
    hcells::Int
    dynb::Int
end

"""The plan of `_cuda_sf_2d_kernel!` on the device `caps` describes, or `nothing` when `NMOM` is
neither 1 nor 6, the sum or count type has no native atomic add, or no tile fits."""
function _cuda_2d_plan(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CT}, wts, geom,
                       NMOM::Int, n_dist::Int, n_val::Int) where {XT, UT, OT, CT}
    ((NMOM == 1 || NMOM == 6) && _cuda_atomic_add(OT) && _cuda_atomic_add(CT)) || return nothing
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFH.field_width(geom))
    CST = _cuda_shared_count_type(wts, CT)
    TILE, dynb = _cuda_2d_pick_tile(caps, XT, UT, OT, CST, W, F, NMOM, n_dist, n_val)
    TILE == 0 && return nothing
    return CUDA2DPlan{W, F, NMOM, TILE, CST}(n_dist * _cuda_val_stride(n_val), dynb)
end

"""Launch `_cuda_sf_2d_kernel!` with `plan` into `out`/`cnt` `(NMOM, n_dist, n_val, B)`; `x` is `(W,N,B)`
varying or `(W,N)`/`(W,N,1)` fixed, `u` is `(F,N,B)`."""
function _cuda_launch_2d!(plan::CUDA2DPlan{W, F, NMOM, TILE, CST}, out, cnt, x, u, wts, sf_type, ddig, vplan,
                          N::Int, n_dist::Int, n_val::Int, B::Int, fixed_x::Bool, geom, second_axis,
                          cull) where {W, F, NMOM, TILE, CST}
    n_dist * _cuda_val_stride(n_val) == plan.hcells || throw(ArgumentError(
        "an $n_dist × $n_val histogram is not the $(plan.hcells)-cell histogram the plan holds"))
    xv = fixed_x ? reshape(x, W, N, 1) : reshape(x, W, N, B)
    uv = reshape(u, F, N, B)
    sched = SFC.schedule_for(cull, N, TILE)
    ntb = SFC.n_pair_blocks(sched)
    fx = fixed_x ? Val(true) : Val(false)
    hv = Val(plan.hcells)
    kern = @cuda launch=false _cuda_sf_2d_kernel!(
        out, cnt, xv, uv, wts, sf_type, ddig, vplan, N, n_dist, n_val, sched, ntb,
        Val(W), Val(F), Val(NMOM), fx, Val(TILE), hv, Val(CST), geom, second_axis)
    # Staging plus the dynamic histogram may pass the default dynamic limit, so every launch opts in;
    # the plan guarantees the total is within the device's opt-in maximum.
    CUDA.attributes(kern.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = plan.dynb
    kern(out, cnt, xv, uv, wts, sf_type, ddig, vplan, N, n_dist, n_val, sched, ntb,
         Val(W), Val(F), Val(NMOM), fx, Val(TILE), hv, Val(CST), geom, second_axis;
         threads = TILE, blocks = ntb * B, shmem = plan.dynb)
    return nothing
end
