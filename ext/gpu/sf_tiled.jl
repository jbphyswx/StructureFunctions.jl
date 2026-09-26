# =============================================================================
# Unified parametric tiled kernels. Building blocks in sf_core.jl.
#
# sf_tiled_1d_varying!  — non-batch (B=1) and varying-x batch (B>1). One
#   workgroup per (tile-pair, batch element). Privatized + R-replicated shared
#   histogram; replicas summed at flush. Covers individual (NMOM=1) and
#   single-pass (NMOM=6), 2D/3D, linear/log/general bins — all via Val{} params.
# =============================================================================

KA.@kernel unsafe_indices = true function sf_tiled_1d_varying!(
    output,                 # (NMOM, NB, B)
    counts,                 # (NMOM, NB, B)  (count replicated across moment rows)
    @Const(x),              # (W, N, B)      (non-batch: B = 1)
    @Const(u),              # (F, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    digitizer,              # device digitize plan of the distance bins
    N::Int,
    NB::Int,
    sched,
    n_tile_blocks::Int,
    wgsize::Int,
    B::Int,
    ::Val{W},               # coordinate width
    ::Val{F},               # field width
    ::Val{NMOM},
    ::Val{R},
    ::Val{CST},             # shared count element: UInt32 unweighted, the count type weighted
    ::Val{WARP},            # threads that execute in lockstep
    geom,
) where {W, F, NMOM, R, CST, WARP}
    shared_xi = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (F * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (F * SF_GPU_TILE,)
    shared_sums = @localmem eltype(output) (NMOM * SF_GPU_MAX_BINS * R,)
    shared_cnts = @localmem CST (SF_GPU_MAX_BINS * R,)

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1

    # phase 0: zero shared histogram (cooperative). Inlined: looped writes to a
    # @localmem array must live in the kernel body — passing it to a helper and
    # writing in a loop fails to compile on this GPU stack.
    zsum = zero(eltype(output))
    zi = lid
    while zi <= NMOM * NB * R
        @inbounds shared_sums[zi] = zsum
        zi += wgsize
    end
    zcnt = zero(CST)
    zc = lid
    while zc <= NB * R
        @inbounds shared_cnts[zc] = zcnt
        zc += wgsize
    end
    @synchronize

    # phase 1: stage tile coordinates into shared memory
    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds for d in 1:W
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi, b]
                end
                @inbounds for d in 1:F
                    shared_ui[(d - 1) * SF_GPU_TILE + k] = u[d, gi, b]
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:W
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj, b]
                    end
                    @inbounds for d in 1:F
                        shared_uj[(d - 1) * SF_GPU_TILE + k] = u[d, gj, b]
                    end
                    k += wgsize
                end
            end
        end
    end
    @synchronize

    # phase 2: enumerate pairs, accumulate into replicated shared histogram
    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
            lane = ((lid - 1) ÷ WARP) % R + 1   # rotate the REPLICA index, never the bin
            jbase = off_diag ? j0 : i0
            p = lid
            while p <= n_pairs
                if off_diag
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xj, jb)
                    Ui = _sf_load_pt(Val(F), shared_ui, ia)
                    Uj = _sf_load_pt(Val(F), shared_uj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xi, jb)
                    Ui = _sf_load_pt(Val(F), shared_ui, ia)
                    Uj = _sf_load_pt(Val(F), shared_ui, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = SFH.digitize(dist, digitizer)
                if ok && 1 <= bin <= NB
                    dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                    moments = _sf_moments(Val(NMOM), sf_type, geom, frame, dist, dU)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    # accumulate into replica `lane` (inline localmem atomics)
                    abase = (bin - 1) * R + lane
                    @inbounds for m in 1:NMOM
                        @atomic shared_sums[(m - 1) * (NB * R) + abase] += pw * moments[m]
                    end
                    @inbounds @atomic shared_cnts[abase] += CST(pw)
                end
                p += wgsize
            end
        end
    end
    @synchronize

    # phase 3: reduce replicas and flush to global output
    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        rplane = NB * R
        cell = lid
        ncell = NMOM * NB
        while cell <= ncell
            m = (cell - 1) % NMOM + 1
            bin = (cell - 1) ÷ NMOM + 1
            rbase = (m - 1) * rplane + (bin - 1) * R
            acc = zero(eltype(output))
            @inbounds for r in 1:R
                acc += shared_sums[rbase + r]
            end
            @inbounds @atomic output[m, bin, b] += acc
            cell += wgsize
        end
        bcell = lid
        while bcell <= NB
            cbase = (bcell - 1) * R
            ctot = zero(CST)
            @inbounds for r in 1:R
                ctot += shared_cnts[cbase + r]
            end
            if ctot != zero(CST)
                @inbounds for m in 1:NMOM
                    @atomic counts[m, bcell, b] += ctot
                end
            end
            bcell += wgsize
        end
    end
end

# -----------------------------------------------------------------------------
# Launch wrappers
# -----------------------------------------------------------------------------

"""Static shared bytes of `sf_tiled_1d_varying!` for `W`-wide coordinates of `XT`, `F`-wide fields of
`UT`, sums of `OT`, counts of `CST`, `NMOM` moments and `R` histogram replicas."""
@inline _sf_1d_varying_smem_bytes(::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CST}, W::Int, F::Int, NMOM::Int,
                                  R::Int) where {XT, UT, OT, CST} =
    2 * SFC.gpu_localmem_bytes(XT, W * SF_GPU_TILE) + 2 * SFC.gpu_localmem_bytes(UT, F * SF_GPU_TILE) +
    SFC.gpu_localmem_bytes(OT, NMOM * SF_GPU_MAX_BINS * R) + SFC.gpu_localmem_bytes(CST, SF_GPU_MAX_BINS * R)

"""Static shared bytes of `sf_tiled_1d_fixed!` for `W`-wide coordinates of `XT`, `F`-wide fields of
`UT`, sums of `OT`, counts of `CST`, `NMOM` moments and `SW`-field strips."""
@inline _sf_1d_fixed_smem_bytes(::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CST}, W::Int, F::Int, NMOM::Int,
                                SW::Int) where {XT, UT, OT, CST} =
    2 * SFC.gpu_localmem_bytes(XT, W * SF_GPU_TILE) + 2 * SFC.gpu_localmem_bytes(UT, SW * F * SF_GPU_TILE) +
    SFC.gpu_localmem_bytes(OT, NMOM * SF_GPU_MAX_BINS * SW) + SFC.gpu_localmem_bytes(CST, SF_GPU_MAX_BINS * SW)

# One thread per `(i, b)` walks every `j > i` and accumulates into the global histogram; it stages
# nothing in shared memory.
KA.@kernel unsafe_indices = true function sf_wide_1d!(
    output,                 # (NMOM, NB, B)
    counts,                 # (NMOM, NB, B)
    @Const(x),              # (W, N, B) varying  /  (W, N, 1) fixed
    @Const(u),              # (F, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    digitizer,
    N::Int,
    NB::Int,
    B::Int,
    ::Val{W},               # coordinate width
    ::Val{F},               # field width
    ::Val{NMOM},
    ::Val{FIXED_X},
    geom,
) where {W, F, NMOM, FIXED_X}
    g = @index(Global)
    if g <= N * B
        i = (g - 1) % N + 1
        b = (g - 1) ÷ N + 1
        if i <= N - 1
            XT = eltype(x)
            UT = eltype(u)
            xb = FIXED_X ? 1 : b
            Xi = SA.SVector{W, XT}(ntuple(d -> @inbounds(x[d, i, xb]), Val(W)))
            Ui = SA.SVector{F, UT}(ntuple(d -> @inbounds(u[d, i, b]), Val(F)))
            wi = SFC._point_weight(wts, i)
            for j in (i + 1):N
                Xj = SA.SVector{W, XT}(ntuple(d -> @inbounds(x[d, j, xb]), Val(W)))
                Uj = SA.SVector{F, UT}(ntuple(d -> @inbounds(u[d, j, b]), Val(F)))
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = SFH.digitize(dist, digitizer)
                if ok && 1 <= bin <= NB
                    dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                    moments = _sf_moments(Val(NMOM), sf_type, geom, frame, dist, dU)
                    pw = wi * SFC._point_weight(wts, j)
                    @inbounds for m in 1:NMOM
                        @atomic output[m, bin, b] += pw * moments[m]
                        @atomic counts[m, bin, b] += convert(eltype(counts), pw)
                    end
                end
            end
        end
    end
end

"""Launch `sf_wide_1d!`, which stages nothing. `x_dev` is `(W, N, B)`, or `(W, N, 1)` when `fixed_x`."""
function _launch_sf_wide_1d!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer,
    N::Int, NB::Int, B::Int, ::Val{NMOM}, fixed_x::Bool, geom;
    weights = SFC.NoWeights(),
) where {NMOM}
    kernel! = sf_wide_1d!(backend, SF_GPU_TILED_WS)
    wts = _sf_weights_to_device(backend, weights)
    launch = (fx) -> kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, digitizer, N, NB, B,
                             SFH.coordinate_width(geom), SFH.field_width(geom), Val(NMOM), fx, geom;
                             ndrange = N * B)
    fixed_x ? launch(Val(true)) : launch(Val(false))
    return nothing
end

"""Replication factor R for the 1D varying-x shared histogram, by regime:
- individual (NMOM = 1): the histogram is small and per-bin contention high, so a second replica
  pays for itself at small bin counts and a third costs more occupancy than it returns;
- single-pass (NMOM = 6): the histogram is six times larger, so any replication costs occupancy."""
@inline _sf_tiled_1d_replication(NMOM::Int) = NMOM == 1 ? 2 : 1

"""Launch sf_tiled_1d_varying! for non-batch (B=1) or varying-x (B>1).
`x_dev` is (W, N, B), `u_dev` (F, N, B); `out_dev`, `cnt_dev` are (NMOM, NB, B)."""
function _launch_sf_tiled_1d_varying!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer,
    N::Int, NB::Int, B::Int, ::Val{NMOM}, geom;
    R::Int = _sf_tiled_1d_replication(NMOM),
    weights = SFC.NoWeights(),
    workspace = nothing,
) where {NMOM}
    caps = SFC.gpu_device_caps(backend)
    vW, vF = SFH.coordinate_width(geom), SFH.field_width(geom)
    W, F = SFC._val_int(vW), SFC._val_int(vF)
    XT, UT, OT, CST = eltype(x_dev), eltype(u_dev), eltype(out_dev), eltype(cnt_dev)
    Rf = NB > SF_GPU_MAX_BINS ? 0 :
        _sf_fitting_width(r -> _sf_1d_varying_smem_bytes(XT, UT, OT, CST, W, F, NMOM, r), caps, R)
    if Rf == 0
        return _launch_sf_wide_1d!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer,
                                   N, NB, B, Val(NMOM), false, geom; weights = weights)
    end
    # The cull memo the prologue published names the tile pairs that can hold a pair within
    # `r_max`; taking the full triangle instead is correct and enumerates tile pairs that cannot.
    sched, n_tile_blocks, ws, _ = _tiled_launch_params(N, workspace)
    ndrange = n_tile_blocks * ws * B
    kernel! = sf_tiled_1d_varying!(backend, ws)
    wts = _sf_weights_to_device(backend, weights)
    vcst = Val(CST)
    vwarp = Val(caps.warp)
    launch = (Rv) -> kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, digitizer, N, NB,
                             sched, n_tile_blocks, ws, B, vW, vF, Val(NMOM), Rv, vcst, vwarp, geom;
                             ndrange = ndrange)
    Rf == 1 ? launch(Val(1)) : Rf == 2 ? launch(Val(2)) : launch(Val(Rf))
    return nothing
end

# =============================================================================
# sf_tiled_1d_fixed! — fixed-x batch: shared geometry x, B velocity fields u.
# Geometry (dist, r̂, bin) is computed ONCE per pair and amortized across a strip
# of SW fields (SW is the privatization axis). Sums use lane = field (scatter to B
# at flush, NOT summed). Counts are field-independent, so the SW lanes are used as
# contention replicas (one atomic/pair) and summed → broadcast to the strip's B.
# The host launches ⌈B/SW⌉ strips asynchronously and synchronizes once; geometry is recomputed per
# strip, so each computation is reused SW-fold.
# Strip shared index for field w, component d, local point k:
# `((w-1)*F + (d-1))*SF_GPU_TILE + k`, written inline at the use sites, as every looped `@localmem`
# index must be.
# =============================================================================

KA.@kernel unsafe_indices = true function sf_tiled_1d_fixed!(
    output,                 # (NMOM, NB, B)
    counts,                 # (NMOM, NB, B)
    @Const(x),              # (W, N)        shared geometry
    @Const(u),              # (F, N, B)     velocity fields
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    digitizer,
    N::Int,
    NB::Int,
    b_base::Int,            # first batch element of this strip
    bw::Int,                # this strip's width (≤ SW)
    sched,
    n_tile_blocks::Int,
    wgsize::Int,
    ::Val{W},               # coordinate width
    ::Val{F},               # field width
    ::Val{NMOM},
    ::Val{SW},              # strip width
    ::Val{CST},             # shared count element: UInt32 unweighted, the count type weighted
    ::Val{WARP},            # threads that execute in lockstep
    geom,
) where {W, F, NMOM, SW, CST, WARP}
    shared_xi = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (SW * F * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (SW * F * SF_GPU_TILE,)
    shared_sums = @localmem eltype(output) (NMOM * SF_GPU_MAX_BINS * SW,)
    shared_cnts = @localmem CST (SF_GPU_MAX_BINS * SW,)

    lid = @index(Local, Linear)
    bid = @index(Group, Linear)

    # phase 0: zero shared histogram inline (looped localmem writes must be in
    # the kernel body, not via a helper, on this GPU stack)
    zsum = zero(eltype(output))
    zi = lid
    while zi <= NMOM * NB * SW
        @inbounds shared_sums[zi] = zsum
        zi += wgsize
    end
    zcnt = zero(CST)
    zc = lid
    while zc <= NB * SW
        @inbounds shared_cnts[zc] = zcnt
        zc += wgsize
    end
    @synchronize

    # phase 1: stage x (once) and u (bw fields) into shared memory
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds for d in 1:W
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi]
                end
                @inbounds for w in 1:bw
                    bb = b_base + w - 1
                    for d in 1:F
                        shared_ui[((w - 1) * F + (d - 1)) * SF_GPU_TILE + k] = u[d, gi, bb]
                    end
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:W
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj]
                    end
                    @inbounds for w in 1:bw
                        bb = b_base + w - 1
                        for d in 1:F
                            shared_uj[((w - 1) * F + (d - 1)) * SF_GPU_TILE + k] = u[d, gj, bb]
                        end
                    end
                    k += wgsize
                end
            end
        end
    end
    @synchronize

    # phase 2: geometry once per pair, accumulate SW fields
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
            cnt_lane = ((lid - 1) ÷ WARP) % SW + 1
            jbase = off_diag ? j0 : i0
            p = lid
            while p <= n_pairs
                if off_diag
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xi, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = SFH.digitize(dist, digitizer)
                if ok && 1 <= bin <= NB
                    # Loop-invariant across the strip: one frame serves all bw fields.
                    rhat = SFH.pair_direction(geom, frame, dist)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    @inbounds for w in 1:bw
                        Ui = off_diag ?
                            _sf_load_field(Val(F), shared_ui, w, ia) :
                            _sf_load_field(Val(F), shared_ui, w, ia)
                        Uj = off_diag ?
                            _sf_load_field(Val(F), shared_uj, w, jb) :
                            _sf_load_field(Val(F), shared_ui, w, jb)
                        dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                        moments = _sf_moments_along(Val(NMOM), sf_type, dU, rhat)
                        # sums: lane = field w (scatter, not summed)
                        base = (bin - 1) * SW + w
                        plane = NB * SW
                        for m in 1:NMOM
                            @atomic shared_sums[(m - 1) * plane + base] += pw * moments[m]
                        end
                    end
                    # counts: field-independent → one atomic into a contention replica
                    @atomic shared_cnts[(bin - 1) * SW + cnt_lane] += CST(pw)
                end
                p += wgsize
            end
        end
    end
    @synchronize

    # phase 3: flush. sums scatter per field; counts sum replicas → broadcast.
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        plane = NB * SW
        cell = lid
        ncell = NMOM * NB
        while cell <= ncell
            m = (cell - 1) % NMOM + 1
            bin = (cell - 1) ÷ NMOM + 1
            base = (m - 1) * plane + (bin - 1) * SW
            @inbounds for w in 1:bw
                @atomic output[m, bin, b_base + w - 1] += shared_sums[base + w]
            end
            cell += wgsize
        end
        bcell = lid
        while bcell <= NB
            total = zero(CST)
            cbase = (bcell - 1) * SW
            @inbounds for r in 1:SW
                total += shared_cnts[cbase + r]
            end
            if total != zero(CST)
                @inbounds for w in 1:bw, m in 1:NMOM
                    @atomic counts[m, bcell, b_base + w - 1] += total
                end
            end
            bcell += wgsize
        end
    end
end

"""Load field `w`'s `F`-vector for local point `k` from a strip-staged u buffer."""
@inline _sf_load_field(::Val{F}, buf, w::Int, k::Int) where {F} =
    SA.SVector{F}(ntuple(d -> @inbounds(buf[((w - 1) * F + (d - 1)) * SF_GPU_TILE + k]), Val(F)))

"""Strip width SW for fixed-x 1D, by regime:
- individual (NMOM = 1): a 4-wide strip amortizes one pair's geometry over four fields;
- single-pass (NMOM = 6): the histogram is six times larger and striping costs occupancy, so SW = 1."""
@inline _sf_tiled_1d_fixed_strip(NMOM::Int) = NMOM == 1 ? 4 : 1

"""Launch fixed-x batch 1D over ⌈B/SW⌉ strips. x_dev=(W,N), u_dev=(F,N,B),
out_dev/cnt_dev=(NMOM,NB,B). Single synchronize after all strips."""
function _launch_sf_tiled_1d_fixed!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer,
    N::Int, NB::Int, B::Int, ::Val{NMOM}, geom;
    SW::Int = _sf_tiled_1d_fixed_strip(NMOM),
    weights = SFC.NoWeights(),
    workspace = nothing,
) where {NMOM}
    caps = SFC.gpu_device_caps(backend)
    vW, vF = SFH.coordinate_width(geom), SFH.field_width(geom)
    W, F = SFC._val_int(vW), SFC._val_int(vF)
    XT, UT, OT, CST = eltype(x_dev), eltype(u_dev), eltype(out_dev), eltype(cnt_dev)
    SWf = NB > SF_GPU_MAX_BINS ? 0 :
        _sf_fitting_width(s -> _sf_1d_fixed_smem_bytes(XT, UT, OT, CST, W, F, NMOM, s), caps, SW)
    if SWf == 0
        return _launch_sf_wide_1d!(backend, out_dev, cnt_dev, reshape(x_dev, W, N, 1), u_dev, sf_type,
                                   digitizer, N, NB, B, Val(NMOM), true, geom; weights = weights)
    end
    sched, n_tile_blocks, _, _ = _tiled_launch_params(N, workspace)
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws
    wts = _sf_weights_to_device(backend, weights)
    vcst = Val(CST)
    vwarp = Val(caps.warp)
    launch = (SWv) -> begin
        kernel! = sf_tiled_1d_fixed!(backend, ws)
        b_base = 1
        while b_base <= B
            bw = min(SWv, B - b_base + 1)
            kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, digitizer, N, NB,
                    b_base, bw, sched, n_tile_blocks, ws, vW, vF, Val(NMOM), Val(SWv), vcst, vwarp, geom;
                    ndrange = ndrange)
            b_base += bw
        end
    end
    launch(SWf)
    return nothing
end

# =============================================================================
# 2D joint-histogram tiled kernels (distance × value). Output is
# (NMOM, n_dist, n_val[, B]) — far too large for shared memory (6·128·128·4 ≈
# 393 KB), so accumulation is DIRECT GLOBAL ATOMICS. Spread over up to ~16K
# cells the per-cell contention is low (fast on Volta+). x/u tiles are still
# staged in shared memory for data reuse. NMOM=1 → joint2d (individual),
# NMOM=6 → single-pass 2D. Replaces the naive per-cell (N,N) kernel + host
# for-b loop that caused the batch 17s regression.
# =============================================================================

KA.@kernel unsafe_indices = true function sf_tiled_2d_varying!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    @Const(x),              # (W, N, B)
    @Const(u),              # (F, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    dist_digitizer,
    val_plan,
    N::Int,
    n_dist::Int,
    n_val::Int,
    sched,
    n_tile_blocks::Int,
    wgsize::Int,
    B::Int,
    ::Val{W},               # coordinate width
    ::Val{F},               # field width
    ::Val{NMOM},
    geom,
    second_axis,
) where {W, F, NMOM}
    shared_xi = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (F * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (F * SF_GPU_TILE,)

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1

    # phase 1: stage tile coordinates
    if bid <= n_tile_blocks && b <= B
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds for d in 1:W
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi, b]
                end
                @inbounds for d in 1:F
                    shared_ui[(d - 1) * SF_GPU_TILE + k] = u[d, gi, b]
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:W
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj, b]
                    end
                    @inbounds for d in 1:F
                        shared_uj[(d - 1) * SF_GPU_TILE + k] = u[d, gj, b]
                    end
                    k += wgsize
                end
            end
        end
    end
    @synchronize

    # phase 2: pair loop, direct global atomics into the 2D histogram
    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
            jbase = off_diag ? j0 : i0
            CT = eltype(counts)
            p = lid
            while p <= n_pairs
                if off_diag
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xj, jb)
                    Ui = _sf_load_pt(Val(F), shared_ui, ia)
                    Uj = _sf_load_pt(Val(F), shared_uj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xi, jb)
                    Ui = _sf_load_pt(Val(F), shared_ui, ia)
                    Uj = _sf_load_pt(Val(F), shared_ui, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = SFH.digitize(dist, dist_digitizer)
                if ok && 1 <= dbin <= n_dist
                    dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                    moments = _sf_moments(Val(NMOM), sf_type, geom, frame, dist, dU)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    @inbounds for m in 1:NMOM
                        vbin = _sf_value_bin(val_plan, SFC.pair_axis_key(second_axis, moments[m], Xi, Xj, dist), m)
                        if 1 <= vbin <= n_val
                            @atomic output[m, dbin, vbin, b] += pw * moments[m]
                            @atomic counts[m, dbin, vbin, b] += CT(pw)
                        end
                    end
                end
                p += wgsize
            end
        end
    end
end

"""Launch sf_tiled_2d_varying! for non-batch (B=1) or varying-x (B>1).
x_dev=(W,N,B), u_dev=(F,N,B); out_dev,cnt_dev=(NMOM,n_dist,n_val,B)."""
function _launch_sf_tiled_2d_varying!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
    N::Int, n_dist::Int, n_val::Int, B::Int, ::Val{NMOM}, geom, second_axis;
    weights = SFC.NoWeights(),
    workspace = nothing,
) where {NMOM}
    sched, n_tile_blocks, _, _ = _tiled_launch_params(N, workspace)
    ws = SF_GPU_TILED_WS
    kernel! = sf_tiled_2d_varying!(backend, ws)
    kernel!(out_dev, cnt_dev, x_dev, u_dev, _sf_weights_to_device(backend, weights),
            sf_type, dist_digitizer, val_plan,
            N, n_dist, n_val, sched, n_tile_blocks, ws, B,
            SFH.coordinate_width(geom), SFH.field_width(geom), Val(NMOM), geom, second_axis;
            ndrange = n_tile_blocks * ws * B)
    return nothing
end

# ----- 2D with a SHARED-memory histogram (small bin counts), fixed or varying --
# When NMOM·n_dist·n_val fits in shared memory, accumulate into a block-local
# histogram (fast shared atomics) and flush once — same idea as the 1D kernel and
# the existing tiled joint2d kernel, which beats direct global atomics by ~7×.
# NCELLS = n_dist·n_val is a compile-time Val so @localmem can be sized to it
# (keeps occupancy high — no over-allocation). One block per (tile-pair, b).
# `x` is always 3D: (W,N,B) for varying-x, (W,N,1) for fixed-x (FIXED_X picks the
# x slice; u is always (F,N,B)). Host uses this only when it fits.
KA.@kernel unsafe_indices = true function sf_tiled_2d_shared!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    @Const(x),              # (W, N, B) varying  /  (W, N, 1) fixed
    @Const(u),              # (F, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    dist_digitizer,
    val_plan,
    N::Int,
    n_dist::Int,
    n_val::Int,
    sched,
    n_tile_blocks::Int,
    wgsize::Int,
    B::Int,
    ::Val{W},               # coordinate width
    ::Val{F},               # field width
    ::Val{NMOM},
    ::Val{NCELLS},
    ::Val{FIXED_X},
    ::Val{CST},             # shared count element: UInt32 unweighted, the count type weighted
    geom,
    second_axis,
) where {W, F, NMOM, NCELLS, FIXED_X, CST}
    shared_xi = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (F * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (F * SF_GPU_TILE,)
    shared_sums = @localmem eltype(output) (NMOM * NCELLS,)
    shared_cnts = @localmem CST (NMOM * NCELLS,)

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1

    # phase 0: zero shared histogram (inline)
    zsum = zero(eltype(output))
    zcnt = zero(CST)
    zi = lid
    while zi <= NMOM * NCELLS
        @inbounds shared_sums[zi] = zsum
        @inbounds shared_cnts[zi] = zcnt
        zi += wgsize
    end
    @synchronize

    # phase 1: stage tile coordinates
    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            xb = FIXED_X ? 1 : b
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds for d in 1:W
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi, xb]
                end
                @inbounds for d in 1:F
                    shared_ui[(d - 1) * SF_GPU_TILE + k] = u[d, gi, b]
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:W
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj, xb]
                    end
                    @inbounds for d in 1:F
                        shared_uj[(d - 1) * SF_GPU_TILE + k] = u[d, gj, b]
                    end
                    k += wgsize
                end
            end
        end
    end
    @synchronize

    # phase 2: pair loop, accumulate into the shared 2D histogram
    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
            jbase = off_diag ? j0 : i0
            p = lid
            while p <= n_pairs
                if off_diag
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xj, jb)
                    Ui = _sf_load_pt(Val(F), shared_ui, ia)
                    Uj = _sf_load_pt(Val(F), shared_uj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xi, jb)
                    Ui = _sf_load_pt(Val(F), shared_ui, ia)
                    Uj = _sf_load_pt(Val(F), shared_ui, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = SFH.digitize(dist, dist_digitizer)
                if ok && 1 <= dbin <= n_dist
                    dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                    moments = _sf_moments(Val(NMOM), sf_type, geom, frame, dist, dU)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    @inbounds for m in 1:NMOM
                        vbin = _sf_value_bin(val_plan, SFC.pair_axis_key(second_axis, moments[m], Xi, Xj, dist), m)
                        if 1 <= vbin <= n_val
                            cell = (m - 1) * NCELLS + (dbin - 1) * n_val + vbin
                            @atomic shared_sums[cell] += pw * moments[m]
                            @atomic shared_cnts[cell] += CST(pw)
                        end
                    end
                end
                p += wgsize
            end
        end
    end
    @synchronize

    # phase 3: flush shared histogram to global output
    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        cell = lid
        while cell <= NMOM * NCELLS
            s = shared_sums[cell]
            c = shared_cnts[cell]
            if c != zero(CST)
                m = (cell - 1) ÷ NCELLS + 1
                lc = (cell - 1) % NCELLS
                dbin = lc ÷ n_val + 1
                vbin = lc % n_val + 1
                @inbounds @atomic output[m, dbin, vbin, b] += s
                @inbounds @atomic counts[m, dbin, vbin, b] += c
            end
            cell += wgsize
        end
    end
end

"""Static shared bytes of `sf_tiled_2d_shared!` for `W`-wide coordinates of `XT`, `F`-wide fields of
`UT`, sums of `OT`, counts of `CST` and `NMOM` histograms of `NCELLS` cells."""
@inline _sf_2d_shared_smem_bytes(::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CST}, W::Int, F::Int, NMOM::Int,
                                 NCELLS::Int) where {XT, UT, OT, CST} =
    2 * SFC.gpu_localmem_bytes(XT, W * SF_GPU_TILE) + 2 * SFC.gpu_localmem_bytes(UT, F * SF_GPU_TILE) +
    SFC.gpu_localmem_bytes(OT, NMOM * NCELLS) + SFC.gpu_localmem_bytes(CST, NMOM * NCELLS)

"""Static shared bytes of `sf_tiled_2d_varying!` for `W`-wide coordinates of `XT` and `F`-wide fields
of `UT`."""
@inline _sf_2d_varying_smem_bytes(::Type{XT}, ::Type{UT}, W::Int, F::Int) where {XT, UT} =
    2 * SFC.gpu_localmem_bytes(XT, W * SF_GPU_TILE) + 2 * SFC.gpu_localmem_bytes(UT, F * SF_GPU_TILE)

"""Launch the shared-histogram 2D kernel (caller guarantees the histogram fits).
`x_dev` is (W,N,B) for varying-x or (W,N,1) for fixed-x; `u_dev` is (F,N,B)."""
function _launch_sf_tiled_2d_shared!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
    N::Int, n_dist::Int, n_val::Int, B::Int, ::Val{NMOM}, fixed_x::Bool, geom, second_axis;
    weights = SFC.NoWeights(),
    workspace = nothing,
) where {NMOM}
    sched, n_tile_blocks, _, _ = _tiled_launch_params(N, workspace)
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws * B
    kernel! = sf_tiled_2d_shared!(backend, ws)
    wts = _sf_weights_to_device(backend, weights)
    vcst = Val(eltype(cnt_dev))
    launch = (fx) -> kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, dist_digitizer, val_plan,
                             N, n_dist, n_val, sched, n_tile_blocks, ws, B,
                             SFH.coordinate_width(geom), SFH.field_width(geom), Val(NMOM),
                             Val(n_dist * n_val), fx, vcst, geom, second_axis; ndrange = ndrange)
    fixed_x ? launch(Val(true)) : launch(Val(false))
    return nothing
end

# ----- fixed-x 2D: geometry once, SW-field strip, direct global atomics -------

KA.@kernel unsafe_indices = true function sf_tiled_2d_fixed!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    @Const(x),              # (W, N)
    @Const(u),              # (F, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    dist_digitizer,
    val_plan,
    N::Int,
    n_dist::Int,
    n_val::Int,
    b_base::Int,
    bw::Int,
    sched,
    n_tile_blocks::Int,
    wgsize::Int,
    ::Val{W},               # coordinate width
    ::Val{F},               # field width
    ::Val{NMOM},
    ::Val{SW},              # strip width
    geom,
    second_axis,
) where {W, F, NMOM, SW}
    shared_xi = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (W * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (SW * F * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (SW * F * SF_GPU_TILE,)

    lid = @index(Local, Linear)
    bid = @index(Group, Linear)

    # phase 1: stage x once, u for bw fields
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds for d in 1:W
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi]
                end
                @inbounds for w in 1:bw
                    bb = b_base + w - 1
                    for d in 1:F
                        shared_ui[((w - 1) * F + (d - 1)) * SF_GPU_TILE + k] = u[d, gi, bb]
                    end
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:W
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj]
                    end
                    @inbounds for w in 1:bw
                        bb = b_base + w - 1
                        for d in 1:F
                            shared_uj[((w - 1) * F + (d - 1)) * SF_GPU_TILE + k] = u[d, gj, bb]
                        end
                    end
                    k += wgsize
                end
            end
        end
    end
    @synchronize

    # phase 2: geometry once per pair, loop bw fields, direct global atomics
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
            jbase = off_diag ? j0 : i0
            CT = eltype(counts)
            p = lid
            while p <= n_pairs
                if off_diag
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(W), shared_xi, ia)
                    Xj = _sf_load_pt(Val(W), shared_xi, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = SFH.digitize(dist, dist_digitizer)
                if ok && 1 <= dbin <= n_dist
                    # Loop-invariant across the strip: one frame serves all bw fields.
                    rhat = SFH.pair_direction(geom, frame, dist)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    @inbounds for w in 1:bw
                        Ui = _sf_load_field(Val(F), shared_ui, w, ia)
                        Uj = off_diag ? _sf_load_field(Val(F), shared_uj, w, jb) :
                                        _sf_load_field(Val(F), shared_ui, w, jb)
                        dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                        moments = _sf_moments_along(Val(NMOM), sf_type, dU, rhat)
                        bb = b_base + w - 1
                        for m in 1:NMOM
                            vbin = _sf_value_bin(val_plan, SFC.pair_axis_key(second_axis, moments[m], Xi, Xj, dist), m)
                            if 1 <= vbin <= n_val
                                @atomic output[m, dbin, vbin, bb] += pw * moments[m]
                                @atomic counts[m, dbin, vbin, bb] += CT(pw)
                            end
                        end
                    end
                end
                p += wgsize
            end
        end
    end
end

"""Static shared bytes of `sf_tiled_2d_fixed!` for `W`-wide coordinates of `XT`, `F`-wide fields of
`UT` and `SW`-field strips."""
@inline _sf_2d_fixed_smem_bytes(::Type{XT}, ::Type{UT}, W::Int, F::Int, SW::Int) where {XT, UT} =
    2 * SFC.gpu_localmem_bytes(XT, W * SF_GPU_TILE) + 2 * SFC.gpu_localmem_bytes(UT, SW * F * SF_GPU_TILE)

"""Widest strip of at most 16 fields whose `sf_tiled_2d_fixed!` fits the device `caps` describes; 0
when a strip of one does not."""
@inline _sf_tiled_2d_fixed_strip(caps::SFC.GPUDeviceCaps, ::Type{XT}, ::Type{UT}, W::Int, F::Int) where {XT, UT} =
    _sf_fitting_width(s -> _sf_2d_fixed_smem_bytes(XT, UT, W, F, s), caps, 16)

"""Launch fixed-x batch 2D over ⌈B/SW⌉ strips of `SW ≥ 1` fields. x_dev=(W,N), u_dev=(F,N,B)."""
function _launch_sf_tiled_2d_fixed!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
    N::Int, n_dist::Int, n_val::Int, B::Int, ::Val{NMOM}, geom, second_axis, SW::Int;
    weights = SFC.NoWeights(),
    workspace = nothing,
) where {NMOM}
    sched, n_tile_blocks, _, _ = _tiled_launch_params(N, workspace)
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws
    wts = _sf_weights_to_device(backend, weights)
    vW, vF = SFH.coordinate_width(geom), SFH.field_width(geom)
    launch = (SWv) -> begin
        kernel! = sf_tiled_2d_fixed!(backend, ws)
        b_base = 1
        while b_base <= B
            bw = min(SWv, B - b_base + 1)
            kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, dist_digitizer, val_plan,
                    N, n_dist, n_val, b_base, bw, sched, n_tile_blocks, ws,
                    vW, vF, Val(NMOM), Val(SWv), geom, second_axis; ndrange = ndrange)
            b_base += bw
        end
    end
    launch(SW)
    return nothing
end

# One thread per `(i, b)` walks every `j > i` and accumulates into the global 2D histogram; it stages
# nothing in shared memory.
KA.@kernel unsafe_indices = true function sf_wide_2d!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    @Const(x),              # (W, N, B) varying  /  (W, N, 1) fixed
    @Const(u),              # (F, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    dist_digitizer,
    val_plan,
    N::Int,
    n_dist::Int,
    n_val::Int,
    B::Int,
    ::Val{W},               # coordinate width
    ::Val{F},               # field width
    ::Val{NMOM},
    ::Val{FIXED_X},
    geom,
    second_axis,
) where {W, F, NMOM, FIXED_X}
    g = @index(Global)
    if g <= N * B
        i = (g - 1) % N + 1
        b = (g - 1) ÷ N + 1
        if i <= N - 1
            XT = eltype(x)
            UT = eltype(u)
            CT = eltype(counts)
            xb = FIXED_X ? 1 : b
            Xi = SA.SVector{W, XT}(ntuple(d -> @inbounds(x[d, i, xb]), Val(W)))
            Ui = SA.SVector{F, UT}(ntuple(d -> @inbounds(u[d, i, b]), Val(F)))
            wi = SFC._point_weight(wts, i)
            for j in (i + 1):N
                Xj = SA.SVector{W, XT}(ntuple(d -> @inbounds(x[d, j, xb]), Val(W)))
                Uj = SA.SVector{F, UT}(ntuple(d -> @inbounds(u[d, j, b]), Val(F)))
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = SFH.digitize(dist, dist_digitizer)
                if ok && 1 <= dbin <= n_dist
                    dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                    moments = _sf_moments(Val(NMOM), sf_type, geom, frame, dist, dU)
                    pw = wi * SFC._point_weight(wts, j)
                    @inbounds for m in 1:NMOM
                        vbin = _sf_value_bin(val_plan, SFC.pair_axis_key(second_axis, moments[m], Xi, Xj, dist), m)
                        if 1 <= vbin <= n_val
                            @atomic output[m, dbin, vbin, b] += pw * moments[m]
                            @atomic counts[m, dbin, vbin, b] += CT(pw)
                        end
                    end
                end
            end
        end
    end
end

"""Launch `sf_wide_2d!`, which stages nothing. `x_dev` is `(W, N, B)`, or `(W, N, 1)` when `fixed_x`."""
function _launch_sf_wide_2d!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
    N::Int, n_dist::Int, n_val::Int, B::Int, ::Val{NMOM}, fixed_x::Bool, geom, second_axis;
    weights = SFC.NoWeights(),
) where {NMOM}
    kernel! = sf_wide_2d!(backend, SF_GPU_TILED_WS)
    wts = _sf_weights_to_device(backend, weights)
    launch = (fx) -> kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, dist_digitizer, val_plan,
                             N, n_dist, n_val, B, SFH.coordinate_width(geom), SFH.field_width(geom),
                             Val(NMOM), fx, geom, second_axis; ndrange = N * B)
    fixed_x ? launch(Val(true)) : launch(Val(false))
    return nothing
end

# -----------------------------------------------------------------------------
# Dispatch helpers used when rewiring the public API onto the unified kernels.
# -----------------------------------------------------------------------------

"""Launch a 2D batch: the backend's native kernel with its plan for the call
([`SFC.gpu_native_2d_plan`](@ref)), or [`_sf_launch_2d_batch_portable!`](@ref) when there is none."""
function _sf_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                              N, n_dist, n_val, B, ::Val{NMOM}, fixed_x::Bool, geom, second_axis;
                              weights = SFC.NoWeights(), workspace = nothing) where {NMOM}
    wts = _sf_weights_to_device(backend, weights)
    plan = SFC.gpu_native_2d_plan(backend, eltype(x_dev), eltype(u_dev), eltype(out_dev), eltype(cnt_dev),
                                  wts, geom, NMOM, n_dist, n_val)
    if plan === nothing
        _sf_launch_2d_batch_portable!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                                      N, n_dist, n_val, B, Val(NMOM), fixed_x, geom, second_axis;
                                      weights = wts, workspace)
    else
        SFC.gpu_native_launch_2d!(plan, out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, ddig, vplan,
                                  N, n_dist, n_val, B, fixed_x, geom, second_axis, _active_cull(workspace))
    end
    return nothing
end

"""Launch a 2D batch on the portable kernels: the shared-histogram kernel (fixed or varying) whenever
it fits, else the staged global-atomic kernels whenever their tiles fit, else the wide kernel, which
stages nothing."""
function _sf_launch_2d_batch_portable!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                                       N, n_dist, n_val, B, ::Val{NMOM}, fixed_x::Bool, geom, second_axis;
                                       weights = SFC.NoWeights(), workspace = nothing) where {NMOM}
    wts = _sf_weights_to_device(backend, weights)
    caps = SFC.gpu_device_caps(backend)
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFH.field_width(geom))
    XT, UT, OT, CST = eltype(x_dev), eltype(u_dev), eltype(out_dev), eltype(cnt_dev)
    x3 = fixed_x ? reshape(x_dev, size(x_dev, 1), size(x_dev, 2), 1) : x_dev
    if SFC.gpu_static_smem_fits(caps, _sf_2d_shared_smem_bytes(XT, UT, OT, CST, W, F, NMOM, n_dist * n_val))
        _launch_sf_tiled_2d_shared!(backend, out_dev, cnt_dev, x3, u_dev, sf_type, ddig, vplan,
                                    N, n_dist, n_val, B, Val(NMOM), fixed_x, geom, second_axis;
                                    weights = wts, workspace)
    elseif !SFC.gpu_static_smem_fits(caps, _sf_2d_varying_smem_bytes(XT, UT, W, F))
        _launch_sf_wide_2d!(backend, out_dev, cnt_dev, x3, u_dev, sf_type, ddig, vplan,
                            N, n_dist, n_val, B, Val(NMOM), fixed_x, geom, second_axis; weights = wts)
    elseif fixed_x
        _launch_sf_tiled_2d_fixed!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                                   N, n_dist, n_val, B, Val(NMOM), geom, second_axis,
                                   _sf_tiled_2d_fixed_strip(caps, XT, UT, W, F); weights = wts, workspace)
    else
        _launch_sf_tiled_2d_varying!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                                     N, n_dist, n_val, B, Val(NMOM), geom, second_axis; weights = wts, workspace)
    end
    return nothing
end

"""Launch a 1D batch: the backend's native kernel with its plan for the call
([`SFC.gpu_native_1d_plan`](@ref)), or [`_sf_launch_1d_batch_portable!`](@ref) when there is none.
`out`/`cnt` are (NMOM, NB, B); x_dev is (W,N,B) varying or (W,N) fixed, u_dev (F,N,B)."""
function _sf_launch_1d_batch!(backend, out, cnt, x_dev, u_dev, sf_type, dig,
                              N, NB, B, ::Val{NMOM}, fixed_x::Bool, geom;
                              weights = SFC.NoWeights(), workspace = nothing) where {NMOM}
    wts = _sf_weights_to_device(backend, weights)
    plan = SFC.gpu_native_1d_plan(backend, eltype(x_dev), eltype(u_dev), eltype(out), eltype(cnt),
                                  wts, geom, NB, NMOM)
    if plan === nothing
        _sf_launch_1d_batch_portable!(backend, out, cnt, x_dev, u_dev, sf_type, dig,
                                      N, NB, B, Val(NMOM), fixed_x, geom; weights = wts, workspace)
    else
        SFC.gpu_native_launch_1d!(plan, out, cnt, x_dev, u_dev, wts, sf_type, dig, N, NB, B, fixed_x, geom,
                                  _active_cull(workspace))
    end
    return nothing
end

"""Launch a 1D batch on the portable kernels."""
function _sf_launch_1d_batch_portable!(backend, out, cnt, x_dev, u_dev, sf_type, dig,
                                       N, NB, B, ::Val{NMOM}, fixed_x::Bool, geom;
                                       weights = SFC.NoWeights(), workspace = nothing) where {NMOM}
    wts = _sf_weights_to_device(backend, weights)
    if fixed_x
        _launch_sf_tiled_1d_fixed!(backend, out, cnt, x_dev, u_dev, sf_type, dig, N, NB, B, Val(NMOM), geom;
                                   weights = wts, workspace)
    else
        _launch_sf_tiled_1d_varying!(backend, out, cnt, x_dev, u_dev, sf_type, dig, N, NB, B, Val(NMOM), geom;
                                     weights = wts, workspace)
    end
    return nothing
end
