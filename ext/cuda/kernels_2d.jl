# The native distance × value kernels: each thread owns a point of one tile and loops over the other staged tile, and the
# block's histogram planes live in dynamic shared memory, merged into the output at block end — or, for a histogram no
# block holds, each pair adds straight into the output.

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

# Moments `P0 … P0 + NP - 1` of the moment set: a launch holds their planes in shared memory, and a histogram whose
# `NMOM` planes do not fit takes several launches.
function _cuda_sf_2d_kernel!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    x,                      # (W, N, B) varying  /  (W, N, 1) fixed (reshaped by launcher)
    u,                      # (F, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    ddig,                   # device digitize plan of the distance bins
    vplan,                  # device value plan: one digitize plan, or one per moment
    N::Int, n_dist::Int, n_val::Int, hcells::Int,
    sched, ntb::Int,
    ::Val{W}, ::Val{F}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, ::Val{CST}, ::Val{P0}, ::Val{NP},
    geom, second_axis,
) where {W, F, NMOM, FIXED_X, TILE, CST, P0, NP}
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
    sums = CuDynamicSharedArray(FT, NP * hcells)
    cnts = CuDynamicSharedArray(CST, NP * hcells, _cuda_count_plane_offset(FT, NP * hcells))

    # zero the privatized histogram
    c = lid
    while c <= NP * hcells
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
                moments = SFC._sf_pair_moments(sf_type, geom, frame, dist, Xi, Xj, Ui, Uj)
                pw = wi * SFC._point_weight(wts, jbase + jj - 1)
                @inbounds for p in 1:NP
                    m = P0 + p - 1
                    vb = SFC._sf_value_bin(vplan, SFC.pair_axis_key(second_axis, moments[m], Xi, Xj, dist), m)
                    if 1 <= vb <= n_val
                        cell = (p - 1) * hcells + (dbin - 1) * vstride + vb
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
    while cell <= NP * hcells
        @inbounds s = sums[cell]
        @inbounds cc = cnts[cell]
        p = (cell - 1) ÷ hcells + 1
        lc = (cell - 1) % hcells
        dbin = lc ÷ vstride + 1
        vb = lc % vstride + 1
        # vb > n_val is a padding column, never written
        if cc != zero(CST) && vb <= n_val
            CUDA.@atomic output[P0 + p - 1, dbin, vb, b] += s
            CUDA.@atomic counts[P0 + p - 1, dbin, vb, b] += cc
        end
        cell += wg
    end
    return nothing
end

# The staging and pair loop of `_cuda_sf_2d_kernel!`, each in-range pair's moments added straight into the global
# histogram: the kernel of a histogram too large for the shared memory of a block.
function _cuda_sf_2d_global_kernel!(
    output, counts, x, u, wts, sf_type, ddig, vplan, N::Int, n_dist::Int, n_val::Int, sched, ntb::Int,
    ::Val{W}, ::Val{F}, ::Val{NMOM}, ::Val{FIXED_X}, ::Val{TILE}, geom, second_axis,
) where {W, F, NMOM, FIXED_X, TILE}
    CT = eltype(counts)
    lid = Int(threadIdx().x)
    lb = Int(blockIdx().x)
    bid = (lb - 1) % ntb + 1
    b = (lb - 1) ÷ ntb + 1

    sxi = CuStaticSharedArray(eltype(x), W * TILE)
    sxj = CuStaticSharedArray(eltype(x), W * TILE)
    sui = CuStaticSharedArray(eltype(u), F * TILE)
    suj = CuStaticSharedArray(eltype(u), F * TILE)

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
                moments = SFC._sf_pair_moments(sf_type, geom, frame, dist, Xi, Xj, Ui, Uj)
                pw = wi * SFC._point_weight(wts, jbase + jj - 1)
                @inbounds for m in 1:NMOM
                    vb = SFC._sf_value_bin(vplan, SFC.pair_axis_key(second_axis, moments[m], Xi, Xj, dist), m)
                    if 1 <= vb <= n_val
                        CUDA.@atomic output[m, dbin, vb, b] += pw * moments[m]
                        CUDA.@atomic counts[m, dbin, vb, b] += CT(pw)
                    end
                end
            end
            jj += 1
        end
    end
    return nothing
end

"""Launch plan of `_cuda_sf_2d_kernel!`: coordinate width `W`, field width `F`, `NMOM` moments, `TILE`-point tiles and
shared counts of `CST`, with `NP` of the moments' histogram planes of `hcells` cells held per launch in `dynb` bytes of
dynamic shared memory (`cld(NMOM, NP)` launches)."""
struct CUDA2DPlan{W, F, NMOM, TILE, CST, NP}
    hcells::Int
    dynb::Int
end

"""Launch plan of `_cuda_sf_2d_global_kernel!`: coordinate width `W`, field width `F`, `NMOM` moments and `TILE`-point
tiles."""
struct CUDA2DGlobalPlan{W, F, NMOM, TILE} end

"""The tile of a native 2-D plan."""
_cuda_tile(::CUDA2DPlan{W, F, NMOM, TILE}) where {W, F, NMOM, TILE} = TILE
_cuda_tile(::CUDA2DGlobalPlan{W, F, NMOM, TILE}) where {W, F, NMOM, TILE} = TILE

"""The plan of the table's `(TILE, NP)` on the device `caps` describes, for coordinates of `XT` at width `W`, fields of
`UT` at width `F`, sums of `FT`, shared counts of `CST` and an `n_dist × n_val` histogram per moment: the first tile of
`TILE`, `TILE ÷ 2`, … 128 whose static staging fits the static budget, with `NP` planes per launch, else fewer, whose
staging and dynamic planes fit the opt-in maximum; `NP = 0`, or no plane count that fits, is the global-atomic kernel
at that tile; `nothing` when no tile's staging fits."""
function _cuda_2d_fit(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int, F::Int,
                      NMOM::Int, n_dist::Int, n_val::Int, (TILE, NP)::NTuple{2, Int}) where {XT, UT, FT, CST}
    hcells = n_dist * _cuda_val_stride(n_val)
    t = TILE
    while t >= 128
        staging = 2 * SFC.gpu_localmem_bytes(XT, W * t) + 2 * SFC.gpu_localmem_bytes(UT, F * t)
        if SFC.gpu_static_smem_fits(caps, staging)
            np = min(NP, NMOM)
            while np >= 1
                cells = np * hcells
                dynb = _cuda_count_plane_offset(FT, cells) + cells * sizeof(CST)
                staging + dynb <= caps.smem_optin && return CUDA2DPlan{W, F, NMOM, t, CST, np}(hcells, dynb)
                np ÷= 2
            end
            return CUDA2DGlobalPlan{W, F, NMOM, t}()
        end
        t ÷= 2
    end
    return nothing
end

"""In-range share below which the native 2-D shared kernel holds every moment's histogram plane at once, and at or
above which two planes, or one, per launch."""
const CU_2D_PLANE_SHARE = 0.2

"""Pair evaluations from which a native 2-D call samples its in-range share and times its candidates
([`CUDAChoice`](@ref)), by moment count (one, or several)."""
const CU_2D_CHOOSE_FROM = (1.0e7, 1.5e6)

"""The candidate plans of the native 2-D kernels ([`CUDAChoice`](@ref)) for coordinates of `XT` at width `W`, fields
of `UT` at width `F`, sums of `FT`, shared counts of `CST`, `NMOM` moments and an `n_dist × n_val` histogram per moment
on the device `caps` describes: the shared-histogram kernel at the largest tile of 512 and 256 whose tile pairs reach
4 blocks per multiprocessor, at the largest of 1024, 512 and 256 whose tile pairs reach one block per multiprocessor
(else 128 for either), and at 128, holding every plane per launch below [`CU_2D_PLANE_SHARE`](@ref) and two or one
above; then the global-atomic kernel at tile 128 (256 for 10⁸ evaluations or more at a share of at least 0.02). Each
plan steps down until it fits."""
function _cuda_2d_candidates(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{FT}, ::Type{CST}, W::Int,
                             F::Int, NMOM::Int, n_dist::Int, n_val::Int) where {XT, UT, FT, CST}
    fit(spec) = _cuda_2d_fit(caps, XT, UT, FT, CST, W, F, NMOM, n_dist, n_val, spec)
    return function (N::Int, B::Int, fixed::Bool, evaluations::Real, share::Real)
        tiles = (_cuda_tile_for(caps, N, B, 4, (512, 256)), _cuda_tile_for(caps, N, B, 1, (1024, 512, 256)), 128)
        planes = share < CU_2D_PLANE_SHARE ? (NMOM,) : NMOM == 1 ? (1,) : (2, 1)
        shared = (fit((t, np)) for np in planes for t in tiles)
        glob = fit((evaluations >= 1e8 && share >= 0.02 ? 256 : 128, 0))
        return unique(filter(!isnothing, Any[shared..., glob]))
    end
end

"""The native 2-D plans ([`CUDAChoice`](@ref)) for the moment set `M` and an `n_dist × n_val` histogram per moment on
the device `caps` describes, or `nothing` when the sum or count type has no native atomic add or no plan fits the
device."""
function _cuda_2d_plan(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CT}, wts, geom,
                       M, n_dist::Int, n_val::Int) where {XT, UT, OT, CT}
    (_cuda_atomic_add(OT) && _cuda_atomic_add(CT)) || return nothing
    NMOM = SFC._sf_nmom(M)
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFC._sf_field_width(M, geom))
    CST = _cuda_shared_count_type(wts, CT)
    _cuda_2d_fit(caps, XT, UT, OT, CST, W, F, NMOM, n_dist, n_val, (128, 0)) === nothing && return nothing
    return CUDAChoice(_cuda_2d_candidates(caps, XT, UT, OT, CST, W, F, NMOM, n_dist, n_val),
                      CU_2D_CHOOSE_FROM[NMOM == 1 ? 1 : 2])
end

"""Launch the plan of `choice` for this call ([`_cuda_plan`](@ref))."""
function _cuda_launch_2d!(choice::CUDAChoice, out, cnt, x, u, wts, sf_type, ddig, vplan, N::Int, n_dist::Int,
                          n_val::Int, B::Int, fixed_x::Bool, geom, second_axis, cull)
    launch!(plan, s, c) = _cuda_launch_2d!(plan, s, c, x, u, wts, sf_type, ddig, vplan, N, n_dist, n_val, B, fixed_x,
                                           geom, second_axis, cull)
    return launch!(_cuda_plan(launch!, choice, out, cnt, x, ddig, n_dist, N, B, fixed_x && B > 1, geom, cull), out,
                   cnt)
end

"""Launch `_cuda_sf_2d_kernel!` with `plan` into `out`/`cnt` `(NMOM, n_dist, n_val, B)`, `NP` planes per launch; `x` is
`(W,N,B)` varying or `(W,N)`/`(W,N,1)` fixed, `u` is `(F,N,B)`."""
function _cuda_launch_2d!(plan::CUDA2DPlan{W, F, NMOM, TILE, CST, NP}, out, cnt, x, u, wts, sf_type, ddig, vplan,
                          N::Int, n_dist::Int, n_val::Int, B::Int, fixed_x::Bool, geom, second_axis,
                          cull) where {W, F, NMOM, TILE, CST, NP}
    n_dist * _cuda_val_stride(n_val) == plan.hcells || throw(ArgumentError(
        "an $n_dist × $n_val histogram is not the $(plan.hcells)-cell histogram the plan holds"))
    xv = fixed_x ? reshape(x, W, N, 1) : reshape(x, W, N, B)
    uv = reshape(u, F, N, B)
    sched = SFC.schedule_for(cull, N, TILE)
    ntb = SFC.n_pair_blocks(sched)
    fx = fixed_x ? Val(true) : Val(false)
    ntuple(Val(cld(NMOM, NP))) do k
        P0 = (k - 1) * NP + 1
        args = (out, cnt, xv, uv, wts, sf_type, ddig, vplan, N, n_dist, n_val, plan.hcells, sched, ntb,
                Val(W), Val(F), Val(NMOM), fx, Val(TILE), Val(CST), Val(P0), Val(min(NP, NMOM - P0 + 1)), geom,
                second_axis)
        kern = @cuda launch=false _cuda_sf_2d_kernel!(args...)
        # Staging plus the dynamic histogram may pass the default dynamic limit, so every launch opts in;
        # the plan guarantees the total is within the device's opt-in maximum.
        CUDA.attributes(kern.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = plan.dynb
        CUDA.attributes(kern.fun)[CUDA.FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT] = CU_CARVEOUT_MAX_SHARED
        kern(args...; threads = TILE, blocks = ntb * B, shmem = plan.dynb)
        nothing
    end
    return nothing
end

"""Launch `_cuda_sf_2d_global_kernel!` with `plan` into `out`/`cnt` `(NMOM, n_dist, n_val, B)`, as
[`_cuda_launch_2d!`](@ref) launches its shared-histogram sibling."""
function _cuda_launch_2d!(::CUDA2DGlobalPlan{W, F, NMOM, TILE}, out, cnt, x, u, wts, sf_type, ddig, vplan,
                          N::Int, n_dist::Int, n_val::Int, B::Int, fixed_x::Bool, geom, second_axis,
                          cull) where {W, F, NMOM, TILE}
    xv = fixed_x ? reshape(x, W, N, 1) : reshape(x, W, N, B)
    uv = reshape(u, F, N, B)
    sched = SFC.schedule_for(cull, N, TILE)
    ntb = SFC.n_pair_blocks(sched)
    fx = fixed_x ? Val(true) : Val(false)
    @cuda threads = TILE blocks = ntb * B _cuda_sf_2d_global_kernel!(
        out, cnt, xv, uv, wts, sf_type, ddig, vplan, N, n_dist, n_val, sched, ntb, Val(W), Val(F), Val(NMOM), fx,
        Val(TILE), geom, second_axis)
    return nothing
end
