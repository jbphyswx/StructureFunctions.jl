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
    @Const(x),              # (D, N, B)      (non-batch: B = 1)
    @Const(u),              # (D, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    digitizer,              # SFLinearDigitizer / SFLogDigitizer / SFGeneralDigitizer
    N::Int,
    NB::Int,
    sched,
    n_tile_blocks::Int,
    wgsize::Int,
    B::Int,
    ::Val{D},
    ::Val{NMOM},
    ::Val{R},
    ::Val{CST},             # shared count element: UInt32 unweighted, the count type weighted
    geom,
) where {D, NMOM, R, CST}
    shared_xi = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (D * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (D * SF_GPU_TILE,)
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
                @inbounds for d in 1:D
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi, b]
                    shared_ui[(d - 1) * SF_GPU_TILE + k] = u[d, gi, b]
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:D
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj, b]
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
            lane = ((lid - 1) ÷ 32) % R + 1   # rotate the REPLICA index, never the bin
            jbase = off_diag ? j0 : i0
            p = lid
            while p <= n_pairs
                if off_diag
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xj, jb)
                    Ui = _sf_load_pt(Val(D), shared_ui, ia)
                    Uj = _sf_load_pt(Val(D), shared_uj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xi, jb)
                    Ui = _sf_load_pt(Val(D), shared_ui, ia)
                    Uj = _sf_load_pt(Val(D), shared_ui, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = digitizer(dist)
                if ok && 1 <= bin <= NB
                    dU, rhat = SFH.pair_increments(geom, frame, dist, Xi, Xj, Ui, Uj)
                    moments = _sf_moments(Val(NMOM), sf_type, dU, rhat)
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

"""
    _sf_tiled_1d_check_nb(NB)

Assert `NB` fits the 1D tiled kernels' shared histogram. Both `sf_tiled_1d_varying!` and
`sf_tiled_1d_fixed!` size `@localmem` from the compile-time `SF_GPU_MAX_BINS` but index it by the
runtime `NB` under `@inbounds`, so `NB > SF_GPU_MAX_BINS` would write out of bounds in shared memory
and silently corrupt the histogram. Must be called by every launcher of those kernels.
"""
@inline function _sf_tiled_1d_check_nb(NB::Int)
    NB > SF_GPU_MAX_BINS && error(
        "GPUExt: 1D tiled kernels support at most $SF_GPU_MAX_BINS distance bins (got NB=$NB)",
    )
    return nothing
end

"""Whether `sf_tiled_1d_varying!`'s shared memory fits in 44 KiB: four staged coordinate tiles,
which grow with the width, plus the `R`-replicated histogram it sizes from `SF_GPU_MAX_BINS`."""
@inline function _sf_1d_varying_shared_fits(::Type{FT}, ::Type{CST}, D::Int, NMOM::Int,
                                            R::Int) where {FT, CST}
    staging = 4 * D * SF_GPU_TILE * sizeof(FT)
    hist = SF_GPU_MAX_BINS * R * (NMOM * sizeof(FT) + sizeof(CST))
    return staging + hist <= 44 * 1024
end

"""Whether `sf_tiled_1d_fixed!`'s shared memory fits in 44 KiB: two coordinate tiles plus `W`
field tiles at each end, and the `W`-replicated histogram."""
@inline function _sf_1d_fixed_shared_fits(::Type{FT}, ::Type{CST}, D::Int, NMOM::Int,
                                          W::Int) where {FT, CST}
    staging = 2 * (1 + W) * D * SF_GPU_TILE * sizeof(FT)
    hist = SF_GPU_MAX_BINS * W * (NMOM * sizeof(FT) + sizeof(CST))
    return staging + hist <= 44 * 1024
end

# A histogram wider than the shared-memory cap has no `@localmem` to stage, so this sibling drops
# the tiling entirely: one thread owns one `(i, b)`, walks every `j > i`, and accumulates straight
# into the global histogram. It is the arrangement the joint-2D and gridded device routes already
# take above their own caps, so every device route now widens the same way.
KA.@kernel unsafe_indices = true function sf_wide_1d_varying!(
    output,                 # (NMOM, NB, B)
    counts,                 # (NMOM, NB, B)
    @Const(x),              # (D, N, B)
    @Const(u),              # (D, N, B)
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    digitizer,
    N::Int,
    NB::Int,
    B::Int,
    ::Val{D},
    ::Val{NMOM},
    geom,
) where {D, NMOM}
    g = @index(Global)
    if g <= N * B
        i = (g - 1) % N + 1
        b = (g - 1) ÷ N + 1
        if i <= N - 1
            XT = eltype(x)
            UT = eltype(u)
            Xi = SA.SVector{D, XT}(ntuple(d -> @inbounds(x[d, i, b]), Val(D)))
            Ui = SA.SVector{D, UT}(ntuple(d -> @inbounds(u[d, i, b]), Val(D)))
            wi = SFC._point_weight(wts, i)
            for j in (i + 1):N
                Xj = SA.SVector{D, XT}(ntuple(d -> @inbounds(x[d, j, b]), Val(D)))
                Uj = SA.SVector{D, UT}(ntuple(d -> @inbounds(u[d, j, b]), Val(D)))
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = digitizer(dist)
                if ok && 1 <= bin <= NB
                    dU, rhat = SFH.pair_increments(geom, frame, dist, Xi, Xj, Ui, Uj)
                    moments = _sf_moments(Val(NMOM), sf_type, dU, rhat)
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

"""Launch `sf_wide_1d_varying!`: the global-atomic route for `NB > SF_GPU_MAX_BINS`."""
function _launch_sf_wide_1d_varying!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer,
    N::Int, NB::Int, B::Int, ::Val{D}, ::Val{NMOM}, geom;
    weights = SFC.NoWeights(),
) where {D, NMOM}
    kernel! = sf_wide_1d_varying!(backend, SF_GPU_TILED_WS)
    kernel!(out_dev, cnt_dev, x_dev, u_dev, _sf_weights_to_device(backend, weights),
            sf_type, digitizer, N, NB, B, Val(D), Val(NMOM), geom; ndrange = N * B)
    return nothing
end

"""Replication factor R for the 1D varying-x shared histogram, by regime:
- individual (NMOM = 1): the histogram is small and per-bin contention high, so a second replica
  pays for itself at small bin counts and a third costs more occupancy than it returns;
- single-pass (NMOM = 6): the histogram is six times larger, so any replication costs occupancy."""
@inline _sf_tiled_1d_replication(::Type{FT}, D::Int, NMOM::Int) where {FT} = NMOM == 1 ? 2 : 1

"""Launch sf_tiled_1d_varying! for non-batch (B=1) or varying-x (B>1).
`x_dev`, `u_dev` are (D, N, B); `out_dev`, `cnt_dev` are (NMOM, NB, B)."""
function _launch_sf_tiled_1d_varying!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer,
    N::Int, NB::Int, B::Int, ::Val{D}, ::Val{NMOM}, geom;
    R::Int = _sf_tiled_1d_replication(eltype(out_dev), D, NMOM),
    weights = SFC.NoWeights(),
    workspace = nothing,
) where {D, NMOM}
    if NB > SF_GPU_MAX_BINS ||
       !_sf_1d_varying_shared_fits(eltype(out_dev), eltype(cnt_dev), D, NMOM, R)
        return _launch_sf_wide_1d_varying!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type,
                                           digitizer, N, NB, B, Val(D), Val(NMOM), geom;
                                           weights = weights)
    end
    _sf_tiled_1d_check_nb(NB)
    # The cull memo the prologue published names the tile pairs that can hold a pair within
    # `r_max`; taking the full triangle instead is correct and enumerates tile pairs that cannot.
    sched, n_tile_blocks, ws, _ = _tiled_launch_params(N, workspace)
    ndrange = n_tile_blocks * ws * B
    kernel! = sf_tiled_1d_varying!(backend, ws)
    wts = _sf_weights_to_device(backend, weights)
    vcst = Val(eltype(cnt_dev))
    launch = (Rv) -> kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, digitizer, N, NB,
                             sched, n_tile_blocks, ws, B, Val(D), Val(NMOM), Rv, vcst, geom;
                             ndrange = ndrange)
    R == 16 ? launch(Val(16)) : R == 8 ? launch(Val(8)) : R == 4 ? launch(Val(4)) :
        R == 2 ? launch(Val(2)) : launch(Val(1))
    return nothing
end

# =============================================================================
# sf_tiled_1d_fixed! — fixed-x batch: shared geometry x, B velocity fields u.
# Geometry (dist, r̂, bin) is computed ONCE per pair and amortized across a strip
# of W fields (W is the privatization axis). Sums use lane = field (scatter to B
# at flush, NOT summed). Counts are field-independent, so the W lanes are used as
# contention replicas (one atomic/pair) and summed → broadcast to the strip's B.
# The host launches ⌈B/W⌉ strips asynchronously and synchronizes once; geometry is recomputed per
# strip, so each computation is reused W-fold.
# W-strip shared index for field w, dim d, local point k:
# `((w-1)*D + (d-1))*SF_GPU_TILE + k`, written inline at the use sites, as every looped `@localmem`
# index must be.
# =============================================================================

KA.@kernel unsafe_indices = true function sf_tiled_1d_fixed!(
    output,                 # (NMOM, NB, B)
    counts,                 # (NMOM, NB, B)
    @Const(x),              # (D, N)        shared geometry
    @Const(u),              # (D, N, B)     velocity fields
    wts,                    # NoWeights(), or one weight per point
    sf_type,
    digitizer,
    N::Int,
    NB::Int,
    b_base::Int,            # first batch element of this strip
    bw::Int,                # this strip's width (≤ W)
    sched,
    n_tile_blocks::Int,
    wgsize::Int,
    ::Val{D},
    ::Val{NMOM},
    ::Val{W},
    ::Val{CST},             # shared count element: UInt32 unweighted, the count type weighted
    geom,
) where {D, NMOM, W, CST}
    shared_xi = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (W * D * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (W * D * SF_GPU_TILE,)
    shared_sums = @localmem eltype(output) (NMOM * SF_GPU_MAX_BINS * W,)
    shared_cnts = @localmem CST (SF_GPU_MAX_BINS * W,)

    lid = @index(Local, Linear)
    bid = @index(Group, Linear)

    # phase 0: zero shared histogram inline (looped localmem writes must be in
    # the kernel body, not via a helper, on this GPU stack)
    zsum = zero(eltype(output))
    zi = lid
    while zi <= NMOM * NB * W
        @inbounds shared_sums[zi] = zsum
        zi += wgsize
    end
    zcnt = zero(CST)
    zc = lid
    while zc <= NB * W
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
                @inbounds for d in 1:D
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi]
                end
                @inbounds for w in 1:bw
                    bb = b_base + w - 1
                    for d in 1:D
                        shared_ui[((w - 1) * D + (d - 1)) * SF_GPU_TILE + k] = u[d, gi, bb]
                    end
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:D
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj]
                    end
                    @inbounds for w in 1:bw
                        bb = b_base + w - 1
                        for d in 1:D
                            shared_uj[((w - 1) * D + (d - 1)) * SF_GPU_TILE + k] = u[d, gj, bb]
                        end
                    end
                    k += wgsize
                end
            end
        end
    end
    @synchronize

    # phase 2: geometry once per pair, accumulate W fields
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
            cnt_lane = ((lid - 1) ÷ 32) % W + 1
            jbase = off_diag ? j0 : i0
            p = lid
            while p <= n_pairs
                if off_diag
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xi, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = digitizer(dist)
                if ok && 1 <= bin <= NB
                    # Loop-invariant across the strip: one frame serves all bw fields.
                    rhat = SFH.pair_direction(geom, frame, dist)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    @inbounds for w in 1:bw
                        Ui = off_diag ?
                            _sf_load_field(Val(D), shared_ui, w, ia) :
                            _sf_load_field(Val(D), shared_ui, w, ia)
                        Uj = off_diag ?
                            _sf_load_field(Val(D), shared_uj, w, jb) :
                            _sf_load_field(Val(D), shared_ui, w, jb)
                        dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                        moments = _sf_moments(Val(NMOM), sf_type, dU, rhat)
                        # sums: lane = field w (scatter, not summed)
                        base = (bin - 1) * W + w
                        plane = NB * W
                        for m in 1:NMOM
                            @atomic shared_sums[(m - 1) * plane + base] += pw * moments[m]
                        end
                    end
                    # counts: field-independent → one atomic into a contention replica
                    @atomic shared_cnts[(bin - 1) * W + cnt_lane] += CST(pw)
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
        plane = NB * W
        cell = lid
        ncell = NMOM * NB
        while cell <= ncell
            m = (cell - 1) % NMOM + 1
            bin = (cell - 1) ÷ NMOM + 1
            base = (m - 1) * plane + (bin - 1) * W
            @inbounds for w in 1:bw
                @atomic output[m, bin, b_base + w - 1] += shared_sums[base + w]
            end
            cell += wgsize
        end
        bcell = lid
        while bcell <= NB
            total = zero(CST)
            cbase = (bcell - 1) * W
            @inbounds for r in 1:W
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

"""Load field `w`'s `W`-vector for local point `k` from a strip-staged u buffer."""
@inline _sf_load_field(::Val{W}, buf, w::Int, k::Int) where {W} =
    SA.SVector{W}(ntuple(d -> @inbounds(buf[((w - 1) * W + (d - 1)) * SF_GPU_TILE + k]), Val(W)))

"""Strip width W for fixed-x 1D, by regime:
- individual (NMOM = 1): a 4-wide strip amortizes one pair's geometry over four fields;
- single-pass (NMOM = 6): the histogram is six times larger and striping costs occupancy, so W = 1."""
@inline _sf_tiled_1d_fixed_strip(::Type{FT}, D::Int, NMOM::Int) where {FT} = NMOM == 1 ? 4 : 1

"""Launch fixed-x batch 1D over ⌈B/W⌉ strips. x_dev=(D,N), u_dev=(D,N,B),
out_dev/cnt_dev=(NMOM,NB,B). Single synchronize after all strips."""
function _launch_sf_tiled_1d_fixed!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer,
    N::Int, NB::Int, B::Int, ::Val{D}, ::Val{NMOM}, geom;
    W::Int = _sf_tiled_1d_fixed_strip(eltype(out_dev), D, NMOM),
    weights = SFC.NoWeights(),
) where {D, NMOM}
    # The shared position set is broadcast to the (D, N, B) the global-atomic kernel reads, which
    # stages nothing and so carries no width or bin limit.
    if NB > SF_GPU_MAX_BINS ||
       !_sf_1d_fixed_shared_fits(eltype(out_dev), eltype(cnt_dev), D, NMOM, W)
        return _launch_sf_wide_1d_varying!(
            backend, out_dev, cnt_dev, repeat(reshape(x_dev, D, N, 1), 1, 1, B), u_dev,
            sf_type, digitizer, N, NB, B, Val(D), Val(NMOM), geom; weights = weights,
        )
    end
    _sf_tiled_1d_check_nb(NB)
    sched = FullUpperTriangle(cld(N, SF_GPU_TILE))
    n_tile_blocks = n_pair_blocks(sched)
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws
    wts = _sf_weights_to_device(backend, weights)
    vcst = Val(eltype(cnt_dev))
    launch = (Wv) -> begin
        kernel! = sf_tiled_1d_fixed!(backend, ws)
        b_base = 1
        while b_base <= B
            bw = min(Wv, B - b_base + 1)
            kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, digitizer, N, NB,
                    b_base, bw, sched, n_tile_blocks, ws, Val(D), Val(NMOM), Val(Wv), vcst, geom;
                    ndrange = ndrange)
            b_base += bw
        end
    end
    W == 16 ? launch(16) : W == 8 ? launch(8) : W == 4 ? launch(4) : W == 2 ? launch(2) : launch(1)
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
    @Const(x),              # (D, N, B)
    @Const(u),              # (D, N, B)
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
    ::Val{D},
    ::Val{NMOM},
    geom,
) where {D, NMOM}
    shared_xi = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (D * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (D * SF_GPU_TILE,)

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
                @inbounds for d in 1:D
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi, b]
                    shared_ui[(d - 1) * SF_GPU_TILE + k] = u[d, gi, b]
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:D
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj, b]
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
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xj, jb)
                    Ui = _sf_load_pt(Val(D), shared_ui, ia)
                    Uj = _sf_load_pt(Val(D), shared_uj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xi, jb)
                    Ui = _sf_load_pt(Val(D), shared_ui, ia)
                    Uj = _sf_load_pt(Val(D), shared_ui, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = dist_digitizer(dist)
                if ok && 1 <= dbin <= n_dist
                    dU, rhat = SFH.pair_increments(geom, frame, dist, Xi, Xj, Ui, Uj)
                    moments = _sf_moments(Val(NMOM), sf_type, dU, rhat)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    @inbounds for m in 1:NMOM
                        vbin = _gpu_digitize_value_plan(moments[m], val_plan, m, n_val + 1)
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
x_dev,u_dev=(D,N,B); out_dev,cnt_dev=(NMOM,n_dist,n_val,B)."""
function _launch_sf_tiled_2d_varying!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
    N::Int, n_dist::Int, n_val::Int, B::Int, ::Val{D}, ::Val{NMOM}, geom;
    weights = SFC.NoWeights(),
) where {D, NMOM}
    sched = FullUpperTriangle(cld(N, SF_GPU_TILE))
    n_tile_blocks = n_pair_blocks(sched)
    ws = SF_GPU_TILED_WS
    kernel! = sf_tiled_2d_varying!(backend, ws)
    kernel!(out_dev, cnt_dev, x_dev, u_dev, _sf_weights_to_device(backend, weights),
            sf_type, dist_digitizer, val_plan,
            N, n_dist, n_val, sched, n_tile_blocks, ws, B, Val(D), Val(NMOM), geom;
            ndrange = n_tile_blocks * ws * B)
    return nothing
end

# ----- 2D with a SHARED-memory histogram (small bin counts), fixed or varying --
# When NMOM·n_dist·n_val fits in shared memory, accumulate into a block-local
# histogram (fast shared atomics) and flush once — same idea as the 1D kernel and
# the existing tiled joint2d kernel, which beats direct global atomics by ~7×.
# NCELLS = n_dist·n_val is a compile-time Val so @localmem can be sized to it
# (keeps occupancy high — no over-allocation). One block per (tile-pair, b).
# `x` is always 3D: (D,N,B) for varying-x, (D,N,1) for fixed-x (FIXED_X picks the
# x slice; u is always (D,N,B)). Host uses this only when it fits.
KA.@kernel unsafe_indices = true function sf_tiled_2d_shared!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    @Const(x),              # (D, N, B) varying  /  (D, N, 1) fixed
    @Const(u),              # (D, N, B)
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
    ::Val{D},
    ::Val{NMOM},
    ::Val{NCELLS},
    ::Val{FIXED_X},
    ::Val{CST},             # shared count element: UInt32 unweighted, the count type weighted
    geom,
) where {D, NMOM, NCELLS, FIXED_X, CST}
    shared_xi = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (D * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (D * SF_GPU_TILE,)
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
                @inbounds for d in 1:D
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi, xb]
                    shared_ui[(d - 1) * SF_GPU_TILE + k] = u[d, gi, b]
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:D
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj, xb]
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
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xj, jb)
                    Ui = _sf_load_pt(Val(D), shared_ui, ia)
                    Uj = _sf_load_pt(Val(D), shared_uj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xi, jb)
                    Ui = _sf_load_pt(Val(D), shared_ui, ia)
                    Uj = _sf_load_pt(Val(D), shared_ui, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = dist_digitizer(dist)
                if ok && 1 <= dbin <= n_dist
                    dU, rhat = SFH.pair_increments(geom, frame, dist, Xi, Xj, Ui, Uj)
                    moments = _sf_moments(Val(NMOM), sf_type, dU, rhat)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    @inbounds for m in 1:NMOM
                        vbin = _gpu_digitize_value_plan(moments[m], val_plan, m, n_val + 1)
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

"""Whether a shared NMOM×n_dist×n_val histogram (sums + counts) fits in 44 KiB alongside the four
staged coordinate tiles, which is the condition for taking the shared kernel."""
@inline function _sf_2d_shared_fits(::Type{FT}, ::Type{CST}, D::Int, NMOM::Int,
                                    n_dist::Int, n_val::Int) where {FT, CST}
    hist = NMOM * n_dist * n_val * (sizeof(FT) + sizeof(CST))
    staging = 4 * D * SF_GPU_TILE * sizeof(FT)
    return hist + staging <= 44 * 1024
end

"""Launch the shared-histogram 2D kernel (caller guarantees the histogram fits).
`x_dev` is (D,N,B) for varying-x or (D,N,1) for fixed-x; `u_dev` is (D,N,B)."""
function _launch_sf_tiled_2d_shared!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
    N::Int, n_dist::Int, n_val::Int, B::Int, ::Val{D}, ::Val{NMOM}, fixed_x::Bool, geom;
    weights = SFC.NoWeights(),
) where {D, NMOM}
    sched = FullUpperTriangle(cld(N, SF_GPU_TILE))
    n_tile_blocks = n_pair_blocks(sched)
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws * B
    kernel! = sf_tiled_2d_shared!(backend, ws)
    wts = _sf_weights_to_device(backend, weights)
    vcst = Val(eltype(cnt_dev))
    launch = (fx) -> kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, dist_digitizer, val_plan,
                             N, n_dist, n_val, sched, n_tile_blocks, ws, B, Val(D), Val(NMOM),
                             Val(n_dist * n_val), fx, vcst, geom; ndrange = ndrange)
    fixed_x ? launch(Val(true)) : launch(Val(false))
    return nothing
end

# ----- fixed-x 2D: geometry once, W-field strip, direct global atomics -------

KA.@kernel unsafe_indices = true function sf_tiled_2d_fixed!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    @Const(x),              # (D, N)
    @Const(u),              # (D, N, B)
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
    ::Val{D},
    ::Val{NMOM},
    ::Val{W},
    geom,
) where {D, NMOM, W}
    shared_xi = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (W * D * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (W * D * SF_GPU_TILE,)

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
                @inbounds for d in 1:D
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x[d, gi]
                end
                @inbounds for w in 1:bw
                    bb = b_base + w - 1
                    for d in 1:D
                        shared_ui[((w - 1) * D + (d - 1)) * SF_GPU_TILE + k] = u[d, gi, bb]
                    end
                end
                k += wgsize
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:D
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x[d, gj]
                    end
                    @inbounds for w in 1:bw
                        bb = b_base + w - 1
                        for d in 1:D
                            shared_uj[((w - 1) * D + (d - 1)) * SF_GPU_TILE + k] = u[d, gj, bb]
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
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    Xi = _sf_load_pt(Val(D), shared_xi, ia)
                    Xj = _sf_load_pt(Val(D), shared_xi, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = dist_digitizer(dist)
                if ok && 1 <= dbin <= n_dist
                    # Loop-invariant across the strip: one frame serves all bw fields.
                    rhat = SFH.pair_direction(geom, frame, dist)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    @inbounds for w in 1:bw
                        Ui = _sf_load_field(Val(D), shared_ui, w, ia)
                        Uj = off_diag ? _sf_load_field(Val(D), shared_uj, w, jb) :
                                        _sf_load_field(Val(D), shared_ui, w, jb)
                        dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                        moments = _sf_moments(Val(NMOM), sf_type, dU, rhat)
                        bb = b_base + w - 1
                        for m in 1:NMOM
                            vbin = _gpu_digitize_value_plan(moments[m], val_plan, m, n_val + 1)
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

"""The widest point the 2D tiled kernels can stage: every one of them holds four coordinate tiles
in shared memory, so unlike the 1D family they have no unstaged sibling to widen into."""
@inline _sf_2d_max_width(::Type{FT}) where {FT} = (46 * 1024) ÷ (4 * SF_GPU_TILE * sizeof(FT))

"""Strip width W for fixed-x 2D (only x/u staged in shared; no shared hist)."""
@inline function _sf_tiled_2d_fixed_strip(::Type{FT}, D::Int) where {FT}
    budget = 46 * 1024
    x_bytes = 2 * D * SF_GPU_TILE * sizeof(FT)
    per_w = 2 * D * SF_GPU_TILE * sizeof(FT)
    w = (budget - x_bytes) ÷ per_w
    w = max(1, min(w, 16))
    w >= 16 ? 16 : w >= 8 ? 8 : w >= 4 ? 4 : w >= 2 ? 2 : 1
end

"""Launch fixed-x batch 2D over ⌈B/W⌉ strips. x_dev=(D,N), u_dev=(D,N,B)."""
function _launch_sf_tiled_2d_fixed!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
    N::Int, n_dist::Int, n_val::Int, B::Int, ::Val{D}, ::Val{NMOM}, geom;
    W::Int = _sf_tiled_2d_fixed_strip(eltype(out_dev), D),
    weights = SFC.NoWeights(),
) where {D, NMOM}
    sched = FullUpperTriangle(cld(N, SF_GPU_TILE))
    n_tile_blocks = n_pair_blocks(sched)
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws
    wts = _sf_weights_to_device(backend, weights)
    launch = (Wv) -> begin
        kernel! = sf_tiled_2d_fixed!(backend, ws)
        b_base = 1
        while b_base <= B
            bw = min(Wv, B - b_base + 1)
            kernel!(out_dev, cnt_dev, x_dev, u_dev, wts, sf_type, dist_digitizer, val_plan,
                    N, n_dist, n_val, b_base, bw, sched, n_tile_blocks, ws,
                    Val(D), Val(NMOM), Val(Wv), geom; ndrange = ndrange)
            b_base += bw
        end
    end
    W == 16 ? launch(16) : W == 8 ? launch(8) : W == 4 ? launch(4) : W == 2 ? launch(2) : launch(1)
    return nothing
end

# -----------------------------------------------------------------------------
# Dispatch helpers used when rewiring the public API onto the unified kernels.
# -----------------------------------------------------------------------------

"""Build a distance digitizer from whatever distance-bins form the public API
passes. Strictly type-driven: `LinearBinEdges` / `LogBinEdges` / `AbstractRange`
(uniform by construction) take the O(1) FMA digitizers; raw edge vectors take the
exact general device-array binary search. No approximate uniformity sniffing —
bin membership must never depend on an `isapprox` tolerance; pass typed edges to
opt into the fast digitizers. Supersedes the old `linear-only` batch restriction."""
function _sf_batch_dist_digitizer(backend, distance_bins)
    distance_bins isa LinearBinEdges && return _sf_digitizer(distance_bins)
    distance_bins isa LogBinEdges && return _sf_digitizer(distance_bins)
    distance_bins isa BinEdges && return _sf_batch_dist_digitizer(backend, distance_bins.edges)
    distance_bins isa AbstractRange && return _sf_digitizer(LinearBinEdges(distance_bins))
    if distance_bins isa AbstractVector
        edges_dev = KA.adapt(backend, collect(distance_bins))
        return _sf_digitizer_general(edges_dev)
    end
    error("unsupported distance_bins type $(typeof(distance_bins))")
end

"""Dispatch a 2D batch launch on a runtime width into the Val-specialized launchers."""
function _sf_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                              N, n_dist, n_val, B, D::Int, ::Val{NMOM}, fixed_x::Bool, geom;
                              weights = SFC.NoWeights()) where {NMOM}
    # CUDA fast path (N-body broadcast + dynamic-shared privatized histogram,
    # TILE=1024) when StructureFunctionsCUDAExt is active and it fits the device.
    weights isa SFC.NoWeights &&
        SFC.gpu_fast_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                                      N, n_dist, n_val, B, D, NMOM, fixed_x, geom, nothing) && return nothing
    # The shared-histogram kernel (fixed or varying) whenever the histogram fits; a bin count too
    # large for shared memory goes to global atomics.
    wts = _sf_weights_to_device(backend, weights)
    use_shared = _sf_2d_shared_fits(eltype(out_dev), eltype(cnt_dev), D, NMOM, n_dist, n_val)
    go(Dv) =
        use_shared ?
            _launch_sf_tiled_2d_shared!(backend, out_dev, cnt_dev,
                (fixed_x ? reshape(x_dev, size(x_dev, 1), size(x_dev, 2), 1) : x_dev),
                u_dev, sf_type, ddig, vplan, N, n_dist, n_val, B, Dv, Val(NMOM), fixed_x, geom;
                weights = wts) :
        fixed_x ?
            _launch_sf_tiled_2d_fixed!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan, N, n_dist, n_val, B, Dv, Val(NMOM), geom; weights = wts) :
            _launch_sf_tiled_2d_varying!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan, N, n_dist, n_val, B, Dv, Val(NMOM), geom; weights = wts)
    dmax = _sf_2d_max_width(eltype(out_dev))
    D <= dmax || error(
        "GPUExt: the 2D tiled kernels stage four coordinate tiles of width D in shared memory, " *
        "which admits D ≤ $dmax at $(eltype(out_dev)) (got D=$D)",
    )
    # 2 and 3 are the widths the package compiles ahead of time; any other is one more kernel
    # instantiation, paid once on its first launch.
    D == 2 ? go(Val(2)) : D == 3 ? go(Val(3)) : go(Val(D))
    return nothing
end

"""Dispatch a 1D batch launch on a runtime width into the Val-specialized launchers.
`out`/`cnt` are (NMOM, NB, B); x_dev/u_dev are (D,N,B) varying or x=(D,N),u=(D,N,B) fixed."""
function _sf_launch_1d_batch!(backend, out, cnt, x_dev, u_dev, sf_type, dig,
                              N, NB, B, D::Int, ::Val{NMOM}, fixed_x::Bool, geom;
                              weights = SFC.NoWeights()) where {NMOM}
    # CUDA fast path (N-body broadcast + static-shared privatized histogram,
    # TILE=256) when StructureFunctionsCUDAExt is active and NB fits.
    weights isa SFC.NoWeights && NB <= SF_GPU_MAX_BINS &&
        SFC.gpu_fast_launch_1d_batch!(backend, out, cnt, x_dev, u_dev, sf_type, dig,
                                      N, NB, B, D, NMOM, fixed_x, geom, nothing) && return nothing
    wts = _sf_weights_to_device(backend, weights)
    go(Dv) = fixed_x ?
        _launch_sf_tiled_1d_fixed!(backend, out, cnt, x_dev, u_dev, sf_type, dig, N, NB, B, Dv, Val(NMOM), geom; weights = wts) :
        _launch_sf_tiled_1d_varying!(backend, out, cnt, x_dev, u_dev, sf_type, dig, N, NB, B, Dv, Val(NMOM), geom; weights = wts)
    # 2 and 3 are the widths the package compiles ahead of time; any other is one more kernel
    # instantiation, paid once on its first launch.
    D == 2 ? go(Val(2)) : D == 3 ? go(Val(3)) : go(Val(D))
    return nothing
end
