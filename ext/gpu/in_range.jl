# The share of a sweep's pairs that land in a distance bin, estimated on the device from a fixed sample of the
# pairs, for launch choices that depend on it.

"""SplitMix64's output function of `z`: consecutive inputs give statistically independent outputs."""
@inline function _mix64(z::UInt64)
    z += 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

# Each work item draws a tile pair of `sched`, a point of each tile and a slice, each from its own hash of the item's
# index, so every pair the schedule sweeps is equally likely; a draw past the last point, or not above the diagonal
# of a diagonal tile pair, is no pair. The workgroup tallies pairs and pairs in range into its column of `tally`.
KA.@kernel unsafe_indices = true function _in_range_kernel!(
    tally,                  # (2, groups): pairs drawn, pairs in range
    @Const(x),              # (W, N, B)
    digitizer, geom, sched,
    n_blocks::Int, tile::Int, N::Int, NB::Int, B::Int, ::Val{W},
) where {W}
    local_tally = @localmem Int32 (2,)
    lid = @index(Local, Linear)
    if lid == 1
        @inbounds local_tally[1] = Int32(0)
        @inbounds local_tally[2] = Int32(0)
    end
    @synchronize
    s = @index(Global, Linear)
    h1 = _mix64(UInt64(s))
    h2 = _mix64(h1)
    h3 = _mix64(h2)
    h4 = _mix64(h3)
    ti, tj = tile_for(sched, Int(h1 % UInt64(n_blocks)) + 1)
    i = (ti - 1) * tile + Int(h2 % UInt64(tile)) + 1
    j = (tj - 1) * tile + Int(h3 % UInt64(tile)) + 1
    b = Int(h4 % UInt64(B)) + 1
    if i <= N && j <= N && (ti < tj || i < j)
        @atomic local_tally[1] += Int32(1)
        Xi = SA.SVector{W}(ntuple(d -> @inbounds(x[d, i, b]), Val(W)))
        Xj = SA.SVector{W}(ntuple(d -> @inbounds(x[d, j, b]), Val(W)))
        ok, dist, _ = SFH.pair_frame(geom, Xi, Xj)
        bin = SFH.digitize(dist, digitizer)
        if ok && 1 <= bin <= NB
            @atomic local_tally[2] += Int32(1)
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    g = @index(Group, Linear)
    if lid == 1
        @inbounds tally[1, g] = local_tally[1]
        @inbounds tally[2, g] = local_tally[2]
    end
end

function SFC.gpu_in_range_tally!(tally, backend, x, dig, NB::Int, geom, cull, tile::Int)
    size(tally) == (2, SFC.GPU_IN_RANGE_GROUPS) ||
        throw(DimensionMismatch("a tally is (2, $(SFC.GPU_IN_RANGE_GROUPS)), got $(size(tally))"))
    vW = SFH.coordinate_width(geom)
    W, N = SFC._val_int(vW), size(x, 2)
    B = ndims(x) == 3 ? size(x, 3) : 1
    sched = schedule_for(cull, N, tile)
    _in_range_kernel!(backend, SFC.GPU_IN_RANGE_GROUP)(tally, reshape(x, W, N, B), dig, geom, sched,
                                                       n_pair_blocks(sched), tile, N, NB, B, vW;
                                                       ndrange = SFC.GPU_IN_RANGE_GROUPS * SFC.GPU_IN_RANGE_GROUP)
    return tally
end

SFC.gpu_in_range_fraction(backend, x, dig, NB::Int, geom, cull, tile::Int) =
    SFC.in_range_share(Array(SFC.gpu_in_range_tally!(KA.allocate(backend, Int32, 2, SFC.GPU_IN_RANGE_GROUPS),
                                                     backend, x, dig, NB, geom, cull, tile)))
