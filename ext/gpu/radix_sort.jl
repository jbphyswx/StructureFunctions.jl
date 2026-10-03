# A stable least-significant-digit radix sort on the device of any KernelAbstractions backend. Per pass over one 8-bit
# digit, each work group counts the digits of its tile, the device scans every group's digit totals, and each group
# orders its tile by the digit in local memory (two stable 4-bit rankings) and writes each digit's run to its place in
# the output.

"""Bits of one radix digit."""
const RADIX_BITS = 8

"""Values of one radix digit."""
const RADIX_BUCKETS = 1 << RADIX_BITS

"""Bits of one local ranking of a tile; a digit takes `RADIX_BITS ÷ RADIX_RANK_BITS` of them."""
const RADIX_RANK_BITS = 4

"""Values of one local ranking."""
const RADIX_RANK_BUCKETS = 1 << RADIX_RANK_BITS

"""Work items of a radix work group, one per digit value."""
const RADIX_WG = 256

"""`log2(RADIX_WG)`: the steps of a work group's scan."""
const RADIX_SCAN_STEPS = 8

"""Items of a work group's tile per work item."""
const RADIX_ITEMS = 8

"""Items of a work group's tile."""
const RADIX_TILE = RADIX_WG * RADIX_ITEMS

"""Local-memory slots of a tile: one pad slot per 32 items, so a work item's run of `RADIX_ITEMS` meets no bank twice."""
const RADIX_TILE_SLOTS = RADIX_TILE + RADIX_TILE ÷ 32

"""The unsigned key of `x` whose order is `isless`'s: IEEE total order for a float, every NaN last."""
@inline function _radix_key(x::T) where {T <: Union{Float32, Float64}}
    U = T === Float32 ? UInt32 : UInt64
    isnan(x) && return typemax(U)
    u = reinterpret(U, x)
    return signbit(x) ? ~u : u | (one(U) << (8 * sizeof(U) - 1))
end

"""The `bits`-bit digit of the unsigned key `u` at bit `shift`, from 1."""
@inline _radix_digit(u::Unsigned, shift::Int, bits::Int) = Int((u >> shift) & ((1 << bits) - 1)) + 1

"""Local-memory slot of tile item `j`."""
@inline _radix_slot(j::Int) = j + ((j - 1) >> 5)

# `hist[b + n_blocks * (d - 1)]`: the items of group `b` whose digit at `shift` is `d`.
KA.@kernel unsafe_indices = true function _radix_count!(hist, @Const(keys), n::Int, shift::Int, n_blocks::Int)
    cnt = @localmem Int32 (RADIX_BUCKETS,)
    t = @index(Local, Linear)
    b = @index(Group, Linear)
    @inbounds cnt[t] = Int32(0)
    @synchronize
    for k in 1:RADIX_ITEMS
        i = (b - 1) * RADIX_TILE + (k - 1) * RADIX_WG + t
        if i <= n
            d = _radix_digit(@inbounds(keys[i]), shift, RADIX_BITS)
            @atomic cnt[d] += Int32(1)
        end
    end
    @synchronize
    @inbounds hist[b + n_blocks * (t - 1)] = cnt[t]
end

# Group `b` loads its tile, orders it by the digit at `shift` (index order within a digit) through two stable rankings
# of 4 bits each — per work item counts of its run of `RADIX_ITEMS` consecutive items, scanned over work items, then a
# reorder in local memory — and writes it out, the items of digit `d` going to `offsets - hist` at `(b, d)` (the digit's
# items in earlier digits and groups) onwards: a stable scatter.
KA.@kernel unsafe_indices = true function _radix_scatter!(keys_out, vals_out, @Const(keys), vals, @Const(hist),
                                                         @Const(offsets), n::Int, shift::Int, n_blocks::Int,
                                                         ::Type{V}) where {V}
    tk = @localmem eltype(keys) (RADIX_TILE_SLOTS,)
    tv = @localmem V (RADIX_TILE_SLOTS,)
    cnt = @localmem Int16 (RADIX_WG, RADIX_RANK_BUCKETS)
    start = @localmem Int32 (RADIX_BUCKETS,)
    place = @localmem Int (RADIX_BUCKETS,)
    part = @private Int16 (RADIX_RANK_BUCKETS,)
    below = @private Int32 (1,)
    mine_k = @private eltype(keys) (RADIX_ITEMS,)
    mine_v = @private V (RADIX_ITEMS,)
    t = @index(Local, Linear)
    b = @index(Group, Linear)
    m = min(RADIX_TILE, n - (b - 1) * RADIX_TILE)
    for k in 1:RADIX_ITEMS
        j = (k - 1) * RADIX_WG + t
        if j <= m
            i = (b - 1) * RADIX_TILE + j
            @inbounds tk[_radix_slot(j)] = keys[i]
            vals === nothing || (@inbounds tv[_radix_slot(j)] = vals[i])
        end
    end
    j = b + n_blocks * (t - 1)
    @inbounds start[t] = Int32(hist[j])
    @inbounds place[t] = offsets[j] - hist[j]
    @synchronize
    for rank in 0:(RADIX_BITS ÷ RADIX_RANK_BITS - 1)
        rshift = shift + rank * RADIX_RANK_BITS
        m = min(RADIX_TILE, n - (b - 1) * RADIX_TILE)
        for d in 1:RADIX_RANK_BUCKETS
            @inbounds cnt[t, d] = Int16(0)
        end
        for k in 1:RADIX_ITEMS
            jj = (t - 1) * RADIX_ITEMS + k
            if jj <= m
                @inbounds mine_k[k] = tk[_radix_slot(jj)]
                vals === nothing || (@inbounds mine_v[k] = tv[_radix_slot(jj)])
                d = _radix_digit(@inbounds(mine_k[k]), rshift, RADIX_RANK_BITS)
                @inbounds cnt[t, d] += Int16(1)
            end
        end
        @synchronize
        for step in 0:(RADIX_SCAN_STEPS - 1)
            s = 1 << step
            for d in 1:RADIX_RANK_BUCKETS
                @inbounds part[d] = t > s ? cnt[t - s, d] : Int16(0)
            end
            @synchronize
            for d in 1:RADIX_RANK_BUCKETS
                @inbounds cnt[t, d] += part[d]
            end
            @synchronize
        end
        for d in 1:RADIX_RANK_BUCKETS
            @inbounds part[d] = t > 1 ? cnt[t - 1, d] : Int16(0)
        end
        @synchronize
        acc = Int16(0)
        for d in 1:RADIX_RANK_BUCKETS
            total = @inbounds cnt[RADIX_WG, d]
            @inbounds part[d] += acc
            acc += total
        end
        @synchronize
        for d in 1:RADIX_RANK_BUCKETS
            @inbounds cnt[t, d] = part[d]
        end
        m = min(RADIX_TILE, n - (b - 1) * RADIX_TILE)
        rshift = shift + rank * RADIX_RANK_BITS
        for k in 1:RADIX_ITEMS
            jj = (t - 1) * RADIX_ITEMS + k
            if jj <= m
                key = @inbounds mine_k[k]
                d = _radix_digit(key, rshift, RADIX_RANK_BITS)
                r = Int(@inbounds cnt[t, d]) + 1
                @inbounds cnt[t, d] += Int16(1)
                @inbounds tk[_radix_slot(r)] = key
                vals === nothing || (@inbounds tv[_radix_slot(r)] = mine_v[k])
            end
        end
        @synchronize
    end
    for step in 0:(RADIX_SCAN_STEPS - 1)
        s = 1 << step
        @inbounds below[1] = t > s ? start[t - s] : Int32(0)
        @synchronize
        @inbounds start[t] += below[1]
        @synchronize
    end
    j = b + n_blocks * (t - 1)
    @inbounds start[t] -= Int32(hist[j])
    @synchronize
    m = min(RADIX_TILE, n - (b - 1) * RADIX_TILE)
    for k in 1:RADIX_ITEMS
        jj = (k - 1) * RADIX_WG + t
        if jj <= m
            key = @inbounds tk[_radix_slot(jj)]
            d = _radix_digit(key, shift, RADIX_BITS)
            p = @inbounds(place[d]) + jj - @inbounds(start[d])
            @inbounds keys_out[p] = key
            vals === nothing || (@inbounds vals_out[p] = tv[_radix_slot(jj)])
        end
    end
end

KA.@kernel function _radix_iota!(v)
    i = @index(Global)
    @inbounds v[i] = i
end

"""The unsigned keys of the device vector `x`, ordered as `x` is, and the low bits they differ in: for a float vector
[`_radix_key`](@ref) of every element; for an integer vector each element's distance from the least, over the bits
the greatest distance takes."""
_radix_keys(x::AbstractVector{T}) where {T <: Union{Float32, Float64}} = map(_radix_key, x), 8 * sizeof(T)
function _radix_keys(x::AbstractVector{T}) where {T <: Integer}
    lo, hi = mapreduce(v -> (v, v), _min_max, x; init = (typemax(T), typemin(T)))
    return map(v -> _radix_distance(v, lo), x), 8 * sizeof(T) - leading_zeros(_radix_distance(hi, lo))
end

"""`v - lo` as the unsigned integer of `v`'s width."""
@inline _radix_distance(v::T, lo::T) where {T <: Integer} = reinterpret(unsigned(T), v - lo)

"""`Int32` while indices up to `n` fit, else `Int`."""
_radix_index_type(n::Int) = n <= typemax(Int32) ? Int32 : Int

"""The unsigned keys `u` sorted over their low `bits` bits, stably, with `vals` (or `nothing`) in their order."""
function _radix_sort_pairs(u::AbstractVector, vals, bits::Int)
    n = length(u)
    backend = KA.get_backend(u)
    n_blocks = cld(n, RADIX_TILE)
    hist = KA.allocate(backend, Int, RADIX_BUCKETS * n_blocks)
    u2, v2 = similar(u), vals === nothing ? nothing : similar(vals)
    V = vals === nothing ? UInt8 : eltype(vals)
    count!, scatter! = _radix_count!(backend, RADIX_WG), _radix_scatter!(backend, RADIX_WG)
    for shift in 0:RADIX_BITS:(bits - 1)
        count!(hist, u, n, shift, n_blocks; ndrange = n_blocks * RADIX_WG)
        scatter!(u2, v2, u, vals, hist, cumsum(hist), n, shift, n_blocks, V; ndrange = n_blocks * RADIX_WG)
        u, u2 = u2, u
        vals, v2 = v2, vals
    end
    return u, vals
end

"""The device unsigned keys `u` sorted stably over their low `bits` bits, and the permutation that sorts them
([`_radix_index_type`](@ref) indices)."""
function _radix_sortperm(u::AbstractVector{<:Unsigned}, bits::Int)
    backend = KA.get_backend(u)
    perm = KA.allocate(backend, _radix_index_type(length(u)), length(u))
    isempty(u) && return u, perm
    _radix_iota!(backend)(perm; ndrange = length(u))
    return _radix_sort_pairs(u, perm, bits)
end

"""`sortperm(x)` of the device vector `x`, stable with NaNs last, as `Int32` indices while they fit."""
_gpu_sortperm(x::AbstractVector) = last(_radix_sortperm(_radix_keys(x)...))
