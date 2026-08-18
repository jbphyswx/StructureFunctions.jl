# =============================================================================
# Unified parametric tiled kernels (replace the per-variant 1D/2D kernels).
# See gpu/OPTIMAL_KERNEL_DESIGN.md. Building blocks in sf_core.jl.
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
    sf_type,
    digitizer,              # SFLinearDigitizer / SFLogDigitizer / SFGeneralDigitizer
    N::Int,
    NB::Int,
    n_tiles::Int,
    n_tile_blocks::Int,
    wgsize::Int,
    B::Int,
    ::Val{D},
    ::Val{NMOM},
    ::Val{R},
    geom,
) where {D, NMOM, R}
    shared_xi = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (D * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (D * SF_GPU_TILE,)
    shared_sums = @localmem eltype(output) (NMOM * SF_GPU_MAX_BINS * R,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS * R,)

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
    zc = lid
    while zc <= NB * R
        @inbounds shared_cnts[zc] = UInt32(0)
        zc += wgsize
    end
    @synchronize

    # phase 1: stage tile coordinates into shared memory
    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        ti, tj = _tile_from_linear(bid, n_tiles)
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
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
            lane = ((lid - 1) ÷ 32) % R + 1   # rotate the REPLICA index, never the bin
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
                    # accumulate into replica `lane` (inline localmem atomics)
                    abase = (bin - 1) * R + lane
                    @inbounds for m in 1:NMOM
                        @atomic shared_sums[(m - 1) * (NB * R) + abase] += moments[m]
                    end
                    @inbounds @atomic shared_cnts[abase] += UInt32(1)
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
            ctot = UInt32(0)
            @inbounds for r in 1:R
                ctot += shared_cnts[cbase + r]
            end
            if ctot != UInt32(0)
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

"""Replication factor R for the 1D varying-x shared histogram, from the A100 R/W
sweep (gpu/benchmark_results/tune_*.md). Regime-dependent:
- Individual SF (NMOM=1): tiny histogram, extreme per-bin contention → R=2 helps
  small NB (NB16: R2≈77 vs R1≈70); larger NB are flat. Beyond R=2 hurts (occupancy).
- Single-pass (NMOM=6): 6× bigger histogram → R=1 (R≥2 blows occupancy: R1≈35,
  R4≈22, R8≈11). So R=1."""
@inline _sf_tiled_1d_replication(::Type{FT}, D::Int, NMOM::Int) where {FT} = NMOM == 1 ? 2 : 1

"""Launch sf_tiled_1d_varying! for non-batch (B=1) or varying-x (B>1).
`x_dev`, `u_dev` are (D, N, B); `out_dev`, `cnt_dev` are (NMOM, NB, B)."""
function _launch_sf_tiled_1d_varying!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer,
    N::Int, NB::Int, B::Int, ::Val{D}, ::Val{NMOM}, geom;
    R::Int = _sf_tiled_1d_replication(eltype(out_dev), D, NMOM),
) where {D, NMOM}
    _sf_tiled_1d_check_nb(NB)
    n_tiles = cld(N, SF_GPU_TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws * B
    kernel! = sf_tiled_1d_varying!(backend, ws)
    if R == 16
        kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer, N, NB,
                n_tiles, n_tile_blocks, ws, B, Val(D), Val(NMOM), Val(16), geom; ndrange = ndrange)
    elseif R == 8
        kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer, N, NB,
                n_tiles, n_tile_blocks, ws, B, Val(D), Val(NMOM), Val(8), geom; ndrange = ndrange)
    elseif R == 4
        kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer, N, NB,
                n_tiles, n_tile_blocks, ws, B, Val(D), Val(NMOM), Val(4), geom; ndrange = ndrange)
    elseif R == 2
        kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer, N, NB,
                n_tiles, n_tile_blocks, ws, B, Val(D), Val(NMOM), Val(2), geom; ndrange = ndrange)
    else
        kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer, N, NB,
                n_tiles, n_tile_blocks, ws, B, Val(D), Val(NMOM), Val(1), geom; ndrange = ndrange)
    end
    return nothing
end

# =============================================================================
# sf_tiled_1d_fixed! — fixed-x batch: shared geometry x, B velocity fields u.
# Geometry (dist, r̂, bin) is computed ONCE per pair and amortized across a strip
# of W fields (W is the privatization axis). Sums use lane = field (scatter to B
# at flush, NOT summed). Counts are field-independent, so the W lanes are used as
# contention replicas (one atomic/pair) and summed → broadcast to the strip's B.
# Host launches ⌈B/W⌉ strips (cheap, async, single sync); geometry is recomputed
# per strip (W-fold reuse — moments dominate, so this is near-optimal).
# W-strip shared index for field w, dim d, local point k: laid out
# `((w-1)*D + (d-1))*SF_GPU_TILE + k` (written inline at the use sites to match
# the proven inline-index staging pattern that compiles on CUDA).
# =============================================================================

KA.@kernel unsafe_indices = true function sf_tiled_1d_fixed!(
    output,                 # (NMOM, NB, B)
    counts,                 # (NMOM, NB, B)
    @Const(x),              # (D, N)        shared geometry
    @Const(u),              # (D, N, B)     velocity fields
    sf_type,
    digitizer,
    N::Int,
    NB::Int,
    b_base::Int,            # first batch element of this strip
    bw::Int,                # this strip's width (≤ W)
    n_tiles::Int,
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
    shared_sums = @localmem eltype(output) (NMOM * SF_GPU_MAX_BINS * W,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS * W,)

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
    zc = lid
    while zc <= NB * W
        @inbounds shared_cnts[zc] = UInt32(0)
        zc += wgsize
    end
    @synchronize

    # phase 1: stage x (once) and u (bw fields) into shared memory
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
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
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
            cnt_lane = ((lid - 1) ÷ 32) % W + 1
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
                            @atomic shared_sums[(m - 1) * plane + base] += moments[m]
                        end
                    end
                    # counts: field-independent → one atomic into a contention replica
                    @atomic shared_cnts[(bin - 1) * W + cnt_lane] += UInt32(1)
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
            total = UInt32(0)
            cbase = (bcell - 1) * W
            @inbounds for r in 1:W
                total += shared_cnts[cbase + r]
            end
            if total != UInt32(0)
                @inbounds for w in 1:bw, m in 1:NMOM
                    @atomic counts[m, bcell, b_base + w - 1] += total
                end
            end
            bcell += wgsize
        end
    end
end

"""Load field `w`'s D-vector for local point `k` from a strip-staged u buffer."""
@inline _sf_load_field(::Val{2}, buf, w::Int, k::Int) =
    @inbounds SA.SVector{2}(buf[(w - 1) * 2 * SF_GPU_TILE + k], buf[((w - 1) * 2 + 1) * SF_GPU_TILE + k])
@inline _sf_load_field(::Val{3}, buf, w::Int, k::Int) =
    @inbounds SA.SVector{3}(buf[(w - 1) * 3 * SF_GPU_TILE + k], buf[((w - 1) * 3 + 1) * SF_GPU_TILE + k], buf[((w - 1) * 3 + 2) * SF_GPU_TILE + k])

"""Strip width W for fixed-x 1D, from the A100 R/W sweep. Regime-dependent:
- Individual SF (NMOM=1): a 4-wide strip amortizes the (relatively expensive)
  geometry over 4 fields → W=4 is the clear peak (~99–107 Gpairs/s vs 64 at W=1,
  across NB=16/50/128); this also makes fixed-x beat varying-x as it should.
- Single-pass (NMOM=6): 6× histogram → striping blows occupancy → W=1 (W1≈40,
  W2≈34, W4≈22). See gpu/benchmark_results/tune_*.md."""
@inline _sf_tiled_1d_fixed_strip(::Type{FT}, D::Int, NMOM::Int) where {FT} = NMOM == 1 ? 4 : 1

"""Launch fixed-x batch 1D over ⌈B/W⌉ strips. x_dev=(D,N), u_dev=(D,N,B),
out_dev/cnt_dev=(NMOM,NB,B). Single synchronize after all strips."""
function _launch_sf_tiled_1d_fixed!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer,
    N::Int, NB::Int, B::Int, ::Val{D}, ::Val{NMOM}, geom;
    W::Int = _sf_tiled_1d_fixed_strip(eltype(out_dev), D, NMOM),
) where {D, NMOM}
    _sf_tiled_1d_check_nb(NB)
    n_tiles = cld(N, SF_GPU_TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws
    launch = (Wv) -> begin
        kernel! = sf_tiled_1d_fixed!(backend, ws)
        b_base = 1
        while b_base <= B
            bw = min(Wv, B - b_base + 1)
            kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, digitizer, N, NB,
                    b_base, bw, n_tiles, n_tile_blocks, ws, Val(D), Val(NMOM), Val(Wv), geom;
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
    sf_type,
    dist_digitizer,
    val_plan,
    N::Int,
    n_dist::Int,
    n_val::Int,
    n_tiles::Int,
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
        ti, tj = _tile_from_linear(bid, n_tiles)
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
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
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
                    @inbounds for m in 1:NMOM
                        vbin = _gpu_digitize_value_plan(moments[m], val_plan, m, n_val + 1)
                        if 1 <= vbin <= n_val
                            @atomic output[m, dbin, vbin, b] += moments[m]
                            @atomic counts[m, dbin, vbin, b] += UInt32(1)
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
    N::Int, n_dist::Int, n_val::Int, B::Int, ::Val{D}, ::Val{NMOM}, geom,
) where {D, NMOM}
    n_tiles = cld(N, SF_GPU_TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    ws = SF_GPU_TILED_WS
    kernel! = sf_tiled_2d_varying!(backend, ws)
    kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
            N, n_dist, n_val, n_tiles, n_tile_blocks, ws, B, Val(D), Val(NMOM), geom;
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
    sf_type,
    dist_digitizer,
    val_plan,
    N::Int,
    n_dist::Int,
    n_val::Int,
    n_tiles::Int,
    n_tile_blocks::Int,
    wgsize::Int,
    B::Int,
    ::Val{D},
    ::Val{NMOM},
    ::Val{NCELLS},
    ::Val{FIXED_X},
    geom,
) where {D, NMOM, NCELLS, FIXED_X}
    shared_xi = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_xj = @localmem eltype(x) (D * SF_GPU_TILE,)
    shared_ui = @localmem eltype(u) (D * SF_GPU_TILE,)
    shared_uj = @localmem eltype(u) (D * SF_GPU_TILE,)
    shared_sums = @localmem eltype(output) (NMOM * NCELLS,)
    shared_cnts = @localmem UInt32 (NMOM * NCELLS,)

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1

    # phase 0: zero shared histogram (inline)
    zsum = zero(eltype(output))
    zi = lid
    while zi <= NMOM * NCELLS
        @inbounds shared_sums[zi] = zsum
        @inbounds shared_cnts[zi] = UInt32(0)
        zi += wgsize
    end
    @synchronize

    # phase 1: stage tile coordinates
    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        ti, tj = _tile_from_linear(bid, n_tiles)
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
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
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
                    @inbounds for m in 1:NMOM
                        vbin = _gpu_digitize_value_plan(moments[m], val_plan, m, n_val + 1)
                        if 1 <= vbin <= n_val
                            cell = (m - 1) * NCELLS + (dbin - 1) * n_val + vbin
                            @atomic shared_sums[cell] += moments[m]
                            @atomic shared_cnts[cell] += UInt32(1)
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
            if c != UInt32(0)
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

"""Does a shared NMOM×n_dist×n_val histogram (sums+counts) fit in ~44 KB
alongside the 4 staged coordinate tiles? If so the shared kernel is much faster
than direct global atomics."""
@inline function _sf_2d_shared_fits(::Type{FT}, D::Int, NMOM::Int, n_dist::Int, n_val::Int) where {FT}
    hist = NMOM * n_dist * n_val * (sizeof(FT) + sizeof(UInt32))
    staging = 4 * D * SF_GPU_TILE * sizeof(FT)
    return hist + staging <= 44 * 1024
end

"""Launch the shared-histogram 2D kernel (caller guarantees the histogram fits).
`x_dev` is (D,N,B) for varying-x or (D,N,1) for fixed-x; `u_dev` is (D,N,B)."""
function _launch_sf_tiled_2d_shared!(
    backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
    N::Int, n_dist::Int, n_val::Int, B::Int, ::Val{D}, ::Val{NMOM}, fixed_x::Bool, geom,
) where {D, NMOM}
    n_tiles = cld(N, SF_GPU_TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws * B
    kernel! = sf_tiled_2d_shared!(backend, ws)
    if fixed_x
        kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
                N, n_dist, n_val, n_tiles, n_tile_blocks, ws, B, Val(D), Val(NMOM), Val(n_dist * n_val), Val(true), geom;
                ndrange = ndrange)
    else
        kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
                N, n_dist, n_val, n_tiles, n_tile_blocks, ws, B, Val(D), Val(NMOM), Val(n_dist * n_val), Val(false), geom;
                ndrange = ndrange)
    end
    return nothing
end

# ----- fixed-x 2D: geometry once, W-field strip, direct global atomics -------

KA.@kernel unsafe_indices = true function sf_tiled_2d_fixed!(
    output,                 # (NMOM, n_dist, n_val, B)
    counts,                 # (NMOM, n_dist, n_val, B)
    @Const(x),              # (D, N)
    @Const(u),              # (D, N, B)
    sf_type,
    dist_digitizer,
    val_plan,
    N::Int,
    n_dist::Int,
    n_val::Int,
    b_base::Int,
    bw::Int,
    n_tiles::Int,
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
        ti, tj = _tile_from_linear(bid, n_tiles)
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
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N - i0 + 1)
        nj = min(SF_GPU_TILE, N - j0 + 1)
        if ni > 0 && nj > 0
            off_diag = ti < tj
            n_pairs = off_diag ? ni * nj : ni * (ni - 1) ÷ 2
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
                                @atomic output[m, dbin, vbin, bb] += moments[m]
                                @atomic counts[m, dbin, vbin, bb] += UInt32(1)
                            end
                        end
                    end
                end
                p += wgsize
            end
        end
    end
end

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
) where {D, NMOM}
    n_tiles = cld(N, SF_GPU_TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    ws = SF_GPU_TILED_WS
    ndrange = n_tile_blocks * ws
    launch = (Wv) -> begin
        kernel! = sf_tiled_2d_fixed!(backend, ws)
        b_base = 1
        while b_base <= B
            bw = min(Wv, B - b_base + 1)
            kernel!(out_dev, cnt_dev, x_dev, u_dev, sf_type, dist_digitizer, val_plan,
                    N, n_dist, n_val, b_base, bw, n_tiles, n_tile_blocks, ws,
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

"""Dispatch a 2D batch launch on runtime D ∈ {2,3} into the Val-specialized launchers."""
function _sf_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                              N, n_dist, n_val, B, D::Int, ::Val{NMOM}, fixed_x::Bool, geom) where {NMOM}
    # CUDA fast path (N-body broadcast + dynamic-shared privatized histogram,
    # TILE=1024) when StructureFunctionsCUDAExt is active and it fits the device.
    SFC.gpu_fast_launch_2d_batch!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan,
                                  N, n_dist, n_val, B, D, NMOM, fixed_x, geom) && return nothing
    # Prefer the shared-histogram kernel (fixed or varying) when the histogram
    # fits (~7× faster than direct global atomics); fall back to global for large
    # bin counts that don't fit in shared memory.
    use_shared = _sf_2d_shared_fits(eltype(out_dev), D, NMOM, n_dist, n_val)
    go(Dv) =
        use_shared ?
            _launch_sf_tiled_2d_shared!(backend, out_dev, cnt_dev,
                (fixed_x ? reshape(x_dev, size(x_dev, 1), size(x_dev, 2), 1) : x_dev),
                u_dev, sf_type, ddig, vplan, N, n_dist, n_val, B, Dv, Val(NMOM), fixed_x, geom) :
        fixed_x ?
            _launch_sf_tiled_2d_fixed!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan, N, n_dist, n_val, B, Dv, Val(NMOM), geom) :
            _launch_sf_tiled_2d_varying!(backend, out_dev, cnt_dev, x_dev, u_dev, sf_type, ddig, vplan, N, n_dist, n_val, B, Dv, Val(NMOM), geom)
    D == 2 ? go(Val(2)) : D == 3 ? go(Val(3)) : error("GPU 2D batch requires D ∈ {2,3} (got $D)")
    return nothing
end

"""Dispatch a 1D batch launch on runtime D ∈ {2,3} into the Val-specialized launchers.
`out`/`cnt` are (NMOM, NB, B); x_dev/u_dev are (D,N,B) varying or x=(D,N),u=(D,N,B) fixed."""
function _sf_launch_1d_batch!(backend, out, cnt, x_dev, u_dev, sf_type, dig,
                              N, NB, B, D::Int, ::Val{NMOM}, fixed_x::Bool, geom) where {NMOM}
    # CUDA fast path (N-body broadcast + static-shared privatized histogram,
    # TILE=256) when StructureFunctionsCUDAExt is active and NB fits.
    SFC.gpu_fast_launch_1d_batch!(backend, out, cnt, x_dev, u_dev, sf_type, dig,
                                  N, NB, B, D, NMOM, fixed_x, geom) && return nothing
    go(Dv) = fixed_x ?
        _launch_sf_tiled_1d_fixed!(backend, out, cnt, x_dev, u_dev, sf_type, dig, N, NB, B, Dv, Val(NMOM), geom) :
        _launch_sf_tiled_1d_varying!(backend, out, cnt, x_dev, u_dev, sf_type, dig, N, NB, B, Dv, Val(NMOM), geom)
    D == 2 ? go(Val(2)) : D == 3 ? go(Val(3)) : error("GPU 1D batch requires D ∈ {2,3} (got $D)")
    return nothing
end
