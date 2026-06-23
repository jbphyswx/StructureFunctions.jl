# =============================================================================
# CUDA-specialized 2D structure-function kernel (distance × value histogram).
#
# N-body broadcast structure + privatized DYNAMIC-shared histogram. This is the
# settled-optimal design (gpu/OPTIMAL_KERNEL_DESIGN.md):
#   - each thread owns its point i (registers); loops j over the staged tile so
#     all lanes read the SAME shared[j] each step (broadcast, no bank conflict,
#     no per-pair pair-decode sqrt);
#   - per-block histogram (sums + counts) lives in dynamic shared memory (A100
#     163 KB opt-in) and is atomic-merged to the global output at block end;
#   - TILE = block size = 1024 (50% occupancy beats 25% on the heavy 50×50 case,
#     measured job 238806: 8.36 vs 6.39 bapps). Device-aware: the launcher drops
#     TILE (1024→512→256) until staging + dynamic histogram fit the device
#     opt-in max, and returns `false` (KA fallback) if even TILE=256 won't fit.
#
# Covers joint 2D (NMOM=1) and single-pass 2D (NMOM=6), fixed-x and varying-x,
# D ∈ {2,3} — all via Val type params. Validated in proto_nbody2d / proto_settle.
# =============================================================================

"""Map linear tile id `k` (1-based) to upper-triangle `(ti,tj)`, `ti ≤ tj`."""
@inline function _cuda_tile_pair(k::Int, nt::Int)
    ti = 1
    rem = k
    while rem > nt - ti + 1
        rem -= (nt - ti + 1)
        ti += 1
    end
    return ti, ti + rem - 1
end

"""Load local point `k`'s D-vector from a tile buffer staged as `(d-1)*TILE + k`."""
@inline _cuda_ld(buf, ::Val{2}, ::Val{TILE}, k::Integer) where {TILE} =
    @inbounds SA.SVector{2}(buf[k], buf[TILE + k])
@inline _cuda_ld(buf, ::Val{3}, ::Val{TILE}, k::Integer) where {TILE} =
    @inbounds SA.SVector{3}(buf[k], buf[TILE + k], buf[2 * TILE + k])

function _cuda_sf_2d_kernel!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    x,                      # (D, N, B) varying  /  (D, N, 1) fixed (reshaped by launcher)
    u,                      # (D, N, B)
    sf_type,
    ddig,                   # distance digitizer functor (isbits, callable on device)
    vplan,                  # value digitize plan (per-moment)
    N::Int, n_dist::Int, n_val::Int,
    nt::Int, ntb::Int,
    ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, ::Val{HCELLS},
) where {D, NMOM, FIXED_X, TILE, HCELLS}
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
    sums = CuDynamicSharedArray(FT, NMOM * HCELLS)
    cnts = CuDynamicSharedArray(UInt32, NMOM * HCELLS, NMOM * HCELLS * sizeof(FT))

    # zero the privatized histogram
    c = lid
    while c <= NMOM * HCELLS
        @inbounds sums[c] = zero(FT)
        @inbounds cnts[c] = UInt32(0)
        c += wg
    end

    ti, tj = _cuda_tile_pair(bid, nt)
    i0 = (ti - 1) * TILE + 1
    j0 = (tj - 1) * TILE + 1
    ni = min(TILE, N - i0 + 1)
    nj = min(TILE, N - j0 + 1)
    xb = FIXED_X ? 1 : b

    # stage tile i (x for geometry, u for velocity)
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

    # N-body broadcast: thread owns point lid, loops j over the staged tile
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
            dX = Xj - Xi
            dist = sqrt(GE._sf_dot(dX, dX))
            dbin = ddig(dist)
            if 1 <= dbin <= n_dist
                rhat = dX / dist
                moments = GE._sf_moments(Val(NMOM), sf_type, Uj - Ui, rhat)
                @inbounds for m in 1:NMOM
                    vb = GE._gpu_digitize_value_plan(moments[m], vplan, m, n_val + 1)
                    if 1 <= vb <= n_val
                        cell = (m - 1) * HCELLS + (dbin - 1) * n_val + vb
                        CUDA.@atomic sums[cell] += moments[m]
                        CUDA.@atomic cnts[cell] += UInt32(1)
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
        if cc != UInt32(0)
            m = (cell - 1) ÷ HCELLS + 1
            lc = (cell - 1) % HCELLS
            dbin = lc ÷ n_val + 1
            vb = lc % n_val + 1
            CUDA.@atomic output[m, dbin, vb, b] += s
            CUDA.@atomic counts[m, dbin, vb, b] += cc
        end
        cell += wg
    end
    return nothing
end

"""Largest TILE ∈ (1024,512,256) whose staging + dynamic histogram fit the device
opt-in shared max; 0 if even 256 won't fit (caller falls back to the KA path)."""
function _cuda_2d_pick_tile(::Type{FT}, D::Int, NMOM::Int, n_dist::Int, n_val::Int) where {FT}
    optin = try
        dev = CUDA.device()
        Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN))
    catch
        48 * 1024
    end
    dynb = NMOM * n_dist * n_val * (sizeof(FT) + sizeof(UInt32))
    for TILE in (1024, 512, 256)
        staging = 4 * D * TILE * sizeof(FT)
        if staging + dynb <= optin
            return TILE, dynb
        end
    end
    return 0, dynb
end

"""Launch the CUDA 2D fast kernel. Returns `true` if launched, `false` if the
histogram doesn't fit any supported TILE (caller uses the KA fallback).
`out`/`cnt` are `(NMOM, n_dist, n_val, B)`. `x` is `(D,N,B)` varying or `(D,N)`/
`(D,N,1)` fixed; `u` is `(D,N,B)`."""
function _cuda_launch_2d!(out, cnt, x, u, sf_type, ddig, vplan,
                          N::Int, n_dist::Int, n_val::Int, B::Int,
                          D::Int, NMOM::Int, fixed_x::Bool)
    FT = eltype(out)
    TILE, dynb = _cuda_2d_pick_tile(FT, D, NMOM, n_dist, n_val)
    TILE == 0 && return false
    xv = fixed_x ? reshape(x, D, N, 1) : reshape(x, D, N, B)
    uv = reshape(u, D, N, B)
    hcells = n_dist * n_val
    _cuda_launch_2d_specialized!(out, cnt, xv, uv, sf_type, ddig, vplan,
                                 N, n_dist, n_val, B, D, NMOM, fixed_x, TILE, hcells, dynb)
    return true
end

# Resolve the runtime (D, NMOM, FIXED_X, TILE) into Val type params, then launch.
function _cuda_launch_2d_specialized!(out, cnt, x, u, sf_type, ddig, vplan,
                                      N, n_dist, n_val, B, D, NMOM, fixed_x, TILE, hcells, dynb)
    Dv = D == 3 ? Val(3) : Val(2)
    Mv = NMOM == 6 ? Val(6) : Val(1)
    Fv = fixed_x ? Val(true) : Val(false)
    Tv = TILE == 1024 ? Val(1024) : TILE == 512 ? Val(512) : Val(256)
    _cuda_launch_2d_valed!(out, cnt, x, u, sf_type, ddig, vplan,
                           N, n_dist, n_val, B, Dv, Mv, Fv, Tv, Val(hcells), dynb)
    return nothing
end

function _cuda_launch_2d_valed!(out, cnt, x, u, sf_type, ddig, vplan,
                                N, n_dist, n_val, B,
                                ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, ::Val{HCELLS},
                                dynb) where {D, NMOM, FIXED_X, TILE, HCELLS}
    nt = cld(N, TILE)
    ntb = nt * (nt + 1) ÷ 2
    kern = @cuda launch=false _cuda_sf_2d_kernel!(
        out, cnt, x, u, sf_type, ddig, vplan, N, n_dist, n_val, nt, ntb,
        Val(D), Val(NMOM), Val(FIXED_X), Val(TILE), Val(HCELLS))
    # Opt into the larger shared budget. This must be set whenever the TOTAL
    # (static staging + dynamic histogram) exceeds the 48 KB default cap — not
    # only when the dynamic part alone does — so set it unconditionally. The
    # value is the dynamic size; `_cuda_2d_pick_tile` guarantees staging + dynb
    # ≤ the device opt-in max.
    CUDA.attributes(kern.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = dynb
    kern(out, cnt, x, u, sf_type, ddig, vplan, N, n_dist, n_val, nt, ntb,
         Val(D), Val(NMOM), Val(FIXED_X), Val(TILE), Val(HCELLS);
         threads = TILE, blocks = ntb * B, shmem = dynb)
    return nothing
end
