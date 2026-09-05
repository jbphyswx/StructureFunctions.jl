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

"""
    _cuda_val_stride(n_val)

Row stride of the value axis in the privatized histogram, forced odd.

Shared memory has 32 banks. With a power-of-two `n_val`, the flat index `(dbin-1)*n_val + vbin` puts
every value-axis row into the same couple of banks, so lanes differing only in `dbin` serialize on
bank conflicts even though they target different cells. An odd stride is coprime with 32 and spreads
them across banks, at the cost of one unused column per row. Measured 2.0–2.3× on the sibling
single-pass 2D kernel, which had identical indexing.
"""
@inline _cuda_val_stride(n_val::Int) = isodd(n_val) ? n_val : n_val + 1

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
    sched, ntb::Int,
    ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, ::Val{HCELLS},
    geom,
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
    vstride = _cuda_val_stride(n_val)
    sums = CuDynamicSharedArray(FT, NMOM * HCELLS)
    cnts = CuDynamicSharedArray(UInt32, NMOM * HCELLS, NMOM * HCELLS * sizeof(FT))

    # zero the privatized histogram
    c = lid
    while c <= NMOM * HCELLS
        @inbounds sums[c] = zero(FT)
        @inbounds cnts[c] = UInt32(0)
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
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            dbin = ddig(dist)
            if ok && 1 <= dbin <= n_dist
                dU, rhat = SFH.pair_increments(geom, frame, dist, Xi, Xj, Ui, Uj)
                moments = GE._sf_moments(Val(NMOM), sf_type, dU, rhat)
                @inbounds for m in 1:NMOM
                    vb = GE._gpu_digitize_value_plan(moments[m], vplan, m, n_val + 1)
                    if 1 <= vb <= n_val
                        cell = (m - 1) * HCELLS + (dbin - 1) * vstride + vb
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
        m = (cell - 1) ÷ HCELLS + 1
        lc = (cell - 1) % HCELLS
        dbin = lc ÷ vstride + 1
        vb = lc % vstride + 1
        # vb > n_val is a padding column, never written
        if cc != UInt32(0) && vb <= n_val
            CUDA.@atomic output[m, dbin, vb, b] += s
            CUDA.@atomic counts[m, dbin, vb, b] += cc
        end
        cell += wg
    end
    return nothing
end

"""Largest TILE ∈ (1024,512,256) whose staging + the full `NMOM`-plane dynamic histogram fit the
device opt-in shared max; 0 if even 256 won't fit.

Returning 0 hands the call to the naive global-atomic kernel, which is the measured winner whenever
the histogram does not fit on chip: contention spreads over many cells, and it beats every on-chip
variant there (128×128 Float64: 5.95 vs 0.75 bapps). Splitting the histogram into moment planes to
force it on chip was implemented and measured — it is ~2× *slower* than going global, so the
all-or-nothing test below is deliberate. See `gpu/SPEED_OF_LIGHT.md`."""
function _cuda_2d_pick_tile(::Type{FT}, D::Int, NMOM::Int, n_dist::Int, n_val::Int) where {FT}
    # Queried, never assumed: this is only reached via the CUDABackend hook, so a device is present
    # by construction and a failure here is a real driver fault. Defaulting to the 48 KB static cap
    # instead would silently forfeit ~3.4× the shared budget on an A100 (163 KB opt-in).
    optin = Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN))
    dynb = NMOM * n_dist * _cuda_val_stride(n_val) * (sizeof(FT) + sizeof(UInt32))
    for TILE in (1024, 512, 256)
        staging = 4 * D * TILE * sizeof(FT)
        # Staging is `CuStaticSharedArray`, capped by the architecture independently of the opt-in;
        # only `dynb` may draw on the larger budget. Testing the sum against `optin` alone lets ptxas
        # reject the kernel at compile time (Float64, D=2, TILE=1024 needs 64 KiB of static shared).
        if staging <= SFC.GPU_SMEM_STATIC_MAX && staging + dynb <= optin
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
                          D::Int, NMOM::Int, fixed_x::Bool, geom, cull)
    FT = eltype(out)
    TILE, dynb = _cuda_2d_pick_tile(FT, D, NMOM, n_dist, n_val)
    TILE == 0 && return false
    xv = fixed_x ? reshape(x, D, N, 1) : reshape(x, D, N, B)
    uv = reshape(u, D, N, B)
    hcells = n_dist * _cuda_val_stride(n_val)
    _cuda_launch_2d_specialized!(out, cnt, xv, uv, sf_type, ddig, vplan,
                                 N, n_dist, n_val, B, D, NMOM, fixed_x, TILE, hcells, dynb, geom,
                                 cull)
    return true
end

# Resolve the runtime (D, NMOM, FIXED_X, TILE) into Val type params, then launch.
function _cuda_launch_2d_specialized!(out, cnt, x, u, sf_type, ddig, vplan,
                                      N, n_dist, n_val, B, D, NMOM, fixed_x, TILE, hcells, dynb, geom,
                                      cull)
    Dv = D == 3 ? Val(3) : Val(2)
    Mv = NMOM == 6 ? Val(6) : Val(1)
    Fv = fixed_x ? Val(true) : Val(false)
    Tv = TILE == 1024 ? Val(1024) : TILE == 512 ? Val(512) : Val(256)
    _cuda_launch_2d_valed!(out, cnt, x, u, sf_type, ddig, vplan,
                           N, n_dist, n_val, B, Dv, Mv, Fv, Tv, Val(hcells), dynb, geom, cull)
    return nothing
end

function _cuda_launch_2d_valed!(out, cnt, x, u, sf_type, ddig, vplan,
                                N, n_dist, n_val, B,
                                ::Val{D}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, ::Val{HCELLS},
                                dynb, geom, cull) where {D, NMOM, FIXED_X, TILE, HCELLS}
    sched = SFC.schedule_for(cull, N, TILE)
    ntb = SFC.n_pair_blocks(sched)
    kern = @cuda launch=false _cuda_sf_2d_kernel!(
        out, cnt, x, u, sf_type, ddig, vplan, N, n_dist, n_val, sched, ntb,
        Val(D), Val(NMOM), Val(FIXED_X), Val(TILE), Val(HCELLS), geom)
    # Opt into the larger shared budget. This must be set whenever the TOTAL
    # (static staging + dynamic histogram) exceeds the 48 KB default cap — not
    # only when the dynamic part alone does — so set it unconditionally. The
    # value is the dynamic size; `_cuda_2d_pick_tile` guarantees staging + dynb
    # ≤ the device opt-in max.
    CUDA.attributes(kern.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = dynb
    kern(out, cnt, x, u, sf_type, ddig, vplan, N, n_dist, n_val, sched, ntb,
         Val(D), Val(NMOM), Val(FIXED_X), Val(TILE), Val(HCELLS), geom;
         threads = TILE, blocks = ntb * B, shmem = dynb)
    return nothing
end
