# Custom bin edges for fast O(1) index digitizing/binning.

"""
    AbstractBinEdges{T} <: AbstractVector{T}

Ordered histogram edges. Bin `i` contains queries in `(edges[i], edges[i+1]]`.
[`BinEdges`](@ref) preserves arbitrary vectors. [`LinearBinEdges`](@ref) and
[`LogBinEdges`](@ref) represent regular grids and use arithmetic to estimate
lookup indices. [`InfPaddedBinEdges`](@ref) adds unbounded outer bins.

"""
# abstract type AbstractBinEdges{T} <: AbstractVector{T} end
abstract type AbstractVectorBinEdges{T} <: AbstractVector{T} end
abstract type AbstractRangeBinEdges{T} <: AbstractRange{T} end
const AbstractBinEdges{T} = Union{AbstractVectorBinEdges{T}, AbstractRangeBinEdges{T}}

# ========================================================================= #
# 1. Generic Wrapped Bin Edges
# ========================================================================= #

"""
    BinEdges(edges::AbstractVector{T})

The supplied sorted edges, kept as given and searched by bisection. A range is a
[`LinearBinEdges`](@ref). [`digitize_plan`](@ref) turns these edges into a
[`BucketedBinEdges`](@ref) for pair kernels.
"""
struct BinEdges{T, ET <: AbstractVector{T}} <: AbstractVectorBinEdges{T}
    edges::ET
end

Base.size(v::BinEdges) = size(v.edges)
Base.getindex(v::BinEdges, i::Int) = v.edges[i]

"""The first `i` in `lo:hi` with `edges[i] ≥ x`, given `edges[hi] ≥ x` or `hi = length(edges) + 1`."""
@inline function _bisect_first(edges, x, lo::Int, hi::Int)
    @inbounds while lo < hi
        m = (lo + hi) >>> 1
        if edges[m] < x
            lo = m + 1
        else
            hi = m
        end
    end
    return lo
end

@inline function Base.searchsortedfirst(v::BinEdges, x)
    e = v.edges
    n = length(e)
    @inbounds x <= e[1] && return 1
    @inbounds x <= e[n] || return n + 1
    return _bisect_first(e, x, 2, n)
end

@inline Base.searchsortedlast(v::BinEdges, x) = _searchsortedlast_from_first(v, x)
@inline Base.searchsorted(v::BinEdges, x) = searchsortedfirst(v, x):searchsortedlast(v, x)
@inline Base.searchsortedfirst(v::BinEdges, x, o::Base.Order.Ordering) =
    o isa Base.Order.ForwardOrdering ? searchsortedfirst(v, x) : searchsortedfirst(v.edges, x, o)
@inline Base.searchsortedlast(v::BinEdges, x, o::Base.Order.Ordering) =
    o isa Base.Order.ForwardOrdering ? searchsortedlast(v, x) : searchsortedlast(v.edges, x, o)
@inline Base.searchsorted(v::BinEdges, x, o::Base.Order.Ordering) =
    searchsortedfirst(v, x, o):searchsortedlast(v, x, o)

"""`searchsortedlast` from `searchsortedfirst` on strictly increasing edges."""
@inline function _searchsortedlast_from_first(edges, x)
    i = searchsortedfirst(edges, x)
    return i <= length(edges) && @inbounds(edges[i]) == x ? i : i - 1
end

"""
    LinearBinEdges(first_edge, last_edge, n_edges)
    LinearBinEdges(edges::AbstractRange)

`n_edges` uniformly spaced edges from `first_edge` to `last_edge`, defining the `n_edges - 1` bins
`(edges[k], edges[k+1]]`. A range is read for its endpoints and length; a vector of arbitrary edges is
a [`BinEdges`](@ref).

The bin of `x` is `ceil(t) + 1` with `t = fma(x, inv_step, -first_edge * inv_step)`. The first and
last edges are `first_edge` and `last_edge`; interior edge `k` is the largest value whose `t` does not
exceed `k - 1`. `t` is correctly rounded and increasing in `x`, so the lookup equals a binary search
over `collect(edges)`. A `NaN` falls above the last edge.
"""
struct LinearBinEdges{T <: AbstractFloat} <: AbstractRangeBinEdges{T}
    first_edge::T
    last_edge::T
    step_val::T
    inv_step::T
    n_edges::Int
end

function LinearBinEdges(first_edge::Real, last_edge::Real, n_edges::Integer)
    T = float(promote_type(typeof(first_edge), typeof(last_edge)))
    f, l = T(first_edge) + zero(T), T(last_edge) + zero(T)
    n_edges >= 2 || throw(ArgumentError("bin edges require at least two values"))
    isfinite(f) && isfinite(l) && f < l ||
        throw(ArgumentError("linear edges require finite increasing endpoints"))
    isfinite(l - f) || throw(ArgumentError("the span from $f to $l overflows $T"))
    s = (l - f) / (n_edges - 1)
    s > 2 * eps(max(abs(f), abs(l))) || throw(ArgumentError(
        "$T cannot resolve $(n_edges - 1) bins between $f and $l; use fewer bins or a wider type",
    ))
    return LinearBinEdges{T}(f, l, s, inv(s), Int(n_edges))
end

LinearBinEdges(edges::AbstractRange) = LinearBinEdges(first(edges), last(edges), length(edges))
LinearBinEdges(b::LinearBinEdges) = b
LinearBinEdges(::AbstractVector) = throw(ArgumentError(
    "LinearBinEdges takes endpoints and a count, or a range; wrap a vector of edges in BinEdges",
))

Base.first(b::LinearBinEdges) = b.first_edge
Base.last(b::LinearBinEdges) = b.last_edge
Base.step(b::LinearBinEdges) = b.step_val
Base.length(b::LinearBinEdges) = b.n_edges
Base.:(==)(a::LinearBinEdges, b::LinearBinEdges) =
    a.first_edge == b.first_edge && a.last_edge == b.last_edge && a.n_edges == b.n_edges
Base.show(io::IO, b::LinearBinEdges) = print(io, "LinearBinEdges(", b.first_edge, ", ", b.last_edge, ", ", b.n_edges, ")")

"""
The largest `x` in `[f, l]` with `fma(x, inv_s, -f * inv_s) ≤ k - 1`: interior edge `k` of a linear
grid. Gallops out from `fma(k - 1, s, f)` and bisects, so an edge many ulps from that estimate, as
near zero on a grid spanning it, costs a logarithmic number of evaluations.
"""
function _linear_threshold(f::T, l::T, inv_s::T, s::T, k::Int) where {T}
    c = -f * inv_s
    K = T(k - 1)
    below(x) = fma(x, inv_s, c) <= K
    x = fma(K, s, f)
    d = eps(max(abs(x), abs(f)))
    if below(x)
        lo, hi = x, min(x + d, l)
        while below(hi)
            lo, d = hi, 2d
            hi = min(x + d, l)
        end
    else
        lo, hi = max(x - d, f), x
        while !below(lo)
            hi, d = lo, 2d
            lo = max(x - d, f)
        end
    end
    while true
        m = lo + (hi - lo) / 2
        lo < m < hi || return lo
        below(m) ? (lo = m) : (hi = m)
    end
end

@inline function Base.getindex(b::LinearBinEdges, k::Integer)
    @boundscheck checkbounds(b, k)
    k == 1 && return b.first_edge
    k == b.n_edges && return b.last_edge
    return _linear_threshold(b.first_edge, b.last_edge, b.inv_step, b.step_val, Int(k))
end

@inline function Base.searchsortedfirst(b::LinearBinEdges{T}, x::T) where {T <: AbstractFloat}
    x <= b.first_edge && return 1
    x <= b.last_edge || return b.n_edges + 1
    t = fma(x, b.inv_step, -b.first_edge * b.inv_step)
    return clamp(unsafe_trunc(Int, ceil(t)) + 1, 2, b.n_edges)
end

# The edges are values of `T`, so a query of another type lands where the next `T` value above it does.
@inline Base.searchsortedfirst(b::LinearBinEdges{T}, x::Real) where {T} = searchsortedfirst(b, T(x, RoundUp))

@inline Base.searchsortedlast(b::LinearBinEdges, x) = _searchsortedlast_from_first(b, x)
@inline Base.searchsorted(b::LinearBinEdges, x) = searchsortedfirst(b, x):searchsortedlast(b, x)

# Base searches a real range from `first` and `step`; these edges are thresholds, so every ordering it
# searches that way goes through the edges.
@inline Base.searchsortedfirst(b::LinearBinEdges, x::Real, o::Base.Sort.FastRangeOrderings) =
    o === Base.Order.Forward ? searchsortedfirst(b, x) :
    invoke(searchsortedfirst, Tuple{AbstractVector, Any, Base.Order.Ordering}, b, x, o)
@inline Base.searchsortedlast(b::LinearBinEdges, x::Real, o::Base.Sort.FastRangeOrderings) =
    o === Base.Order.Forward ? searchsortedlast(b, x) :
    invoke(searchsortedlast, Tuple{AbstractVector, Any, Base.Order.Ordering}, b, x, o)

"""
    LogBinEdges(first_edge, last_edge, n_edges)
    LogBinEdges_from_log_edges(log_edges::AbstractRange)

`n_edges` positive edges uniformly spaced in `log`, from `first_edge` to `last_edge`. The first and
last edges are `first_edge` and `last_edge`; interior edge `k` is `exp(fma(k - 1, step, log_first))`,
with `log_first` and `step` those of the [`LinearBinEdges`](@ref) grid `log_linear` from
`log(first_edge)` to `log(last_edge)`. A vector of arbitrary edges is a [`BinEdges`](@ref).

Membership is decided against those edges, `(edges[k], edges[k+1]]`: the log grid only estimates the
index, so the result does not depend on how `log` rounds. [`digitize_plan`](@ref) turns these edges
into a [`BucketedBinEdges`](@ref) for pair kernels.
"""
struct LogBinEdges{T <: AbstractFloat} <: AbstractVectorBinEdges{T}
    first_edge::T
    last_edge::T
    log_linear::LinearBinEdges{T}
end

function _log_bins(f::T, l::T, ll::LinearBinEdges{T}) where {T}
    0 < f < l < Inf || throw(ArgumentError("logarithmic edges must be finite, positive and increasing"))
    b = LogBinEdges{T}(f, l, ll)
    issorted(b; lt = <=) || throw(ArgumentError(
        "$T cannot resolve $(length(ll) - 1) logarithmic bins between $f and $l; use fewer bins or a wider type",
    ))
    return b
end

function LogBinEdges(first_edge::Real, last_edge::Real, n_edges::Integer)
    0 < first_edge < last_edge || throw(ArgumentError("logarithmic endpoints must satisfy 0 < first < last"))
    ll = LinearBinEdges(log(float(first_edge)), log(float(last_edge)), n_edges)
    T = eltype(ll)
    return _log_bins(T(first_edge), T(last_edge), ll)
end

LogBinEdges(b::LogBinEdges) = b
LogBinEdges(::AbstractVector) = throw(ArgumentError(
    "LogBinEdges takes endpoints and a count; wrap a vector of edges in BinEdges, or pass a range of " *
    "logarithms to LogBinEdges_from_log_edges",
))

"""
    LogBinEdges_from_log_edges(log_edges::AbstractRange)

[`LogBinEdges`](@ref) whose logarithms are the uniform grid `log_edges`: the edges are
`exp.(log_edges)`.
"""
function LogBinEdges_from_log_edges(log_edges::AbstractRange)
    ll = LinearBinEdges(log_edges)
    return _log_bins(exp(ll.first_edge), exp(ll.last_edge), ll)
end
LogBinEdges_from_log_edges(::AbstractVector) = throw(ArgumentError(
    "LogBinEdges_from_log_edges takes a range of logarithms; wrap a vector of edges in BinEdges",
))

Base.size(b::LogBinEdges) = size(b.log_linear)
Base.:(==)(a::LogBinEdges, b::LogBinEdges) =
    a.first_edge == b.first_edge && a.last_edge == b.last_edge && a.log_linear == b.log_linear

@inline function Base.getindex(b::LogBinEdges{T}, k::Int) where {T}
    @boundscheck checkbounds(b, k)
    k == 1 && return b.first_edge
    k == length(b) && return b.last_edge
    return exp(fma(T(k - 1), b.log_linear.step_val, b.log_linear.first_edge))
end

# The log grid estimates the index; the edges themselves decide it.
@inline function Base.searchsortedfirst(b::LogBinEdges, x)
    n = length(b)
    x <= b.first_edge && return 1
    x <= b.last_edge || return n + 1
    i = clamp(searchsortedfirst(b.log_linear, log(x)), 2, n)
    @inbounds while i < n && b[i] < x
        i += 1
    end
    @inbounds while i > 2 && b[i - 1] >= x
        i -= 1
    end
    return i
end
@inline Base.searchsortedlast(b::LogBinEdges, x) = _searchsortedlast_from_first(b, x)
@inline Base.searchsorted(b::LogBinEdges, x) = searchsortedfirst(b, x):searchsortedlast(b, x)

@inline Base.searchsortedfirst(b::LogBinEdges, x, o::Base.Order.Ordering) =
    o isa Base.Order.ForwardOrdering ? searchsortedfirst(b, x) :
    invoke(searchsortedfirst, Tuple{AbstractVector, Any, Base.Order.Ordering}, b, x, o)
@inline Base.searchsortedlast(b::LogBinEdges, x, o::Base.Order.Ordering) =
    o isa Base.Order.ForwardOrdering ? searchsortedlast(b, x) :
    invoke(searchsortedlast, Tuple{AbstractVector, Any, Base.Order.Ordering}, b, x, o)
@inline Base.searchsorted(b::LogBinEdges, x, o::Base.Order.Ordering) =
    searchsortedfirst(b, x, o):searchsortedlast(b, x, o)

"""
    LinearCells(inv_width, offset, last_cell)

Cells of equal width: `cell(x) = trunc(clamp(fma(x, inv_width, offset), 0, last_cell))`, which never decreases as
`x` increases.
"""
struct LinearCells{T}
    inv_width::T
    offset::T
    last_cell::T
end

"""
    Log2Cells(lo, key0, shift, last_cell)

Cells of equal width in `log₂ x` to within the piecewise-linear error of the float format: the cell of `x > lo` is
the IEEE exponent and leading mantissa bits of `x` (its bit pattern shifted right by `shift`) less those of `lo`,
clamped to `last_cell`; `x ≤ lo` takes cell 0. Never decreases as `x` increases.
"""
struct Log2Cells{T, U <: Unsigned}
    lo::T
    key0::U
    shift::Int
    last_cell::U
end

"""
    BucketedBinEdges(edges::AbstractVector{<:Base.IEEEFloat}[, cells])

Sorted `edges` with a table that brackets every lookup. The edges are cut into cells by the cell map `cells` — by
default [`LinearCells`](@ref StructureFunctions.LinearCells) of `16(length(edges) - 1)` equal cells over the finite
span. Each cell holds a [`BucketCell`](@ref StructureFunctions.BucketCell): the first edge at or above it, how many
edges lie in it, and that edge's value. A query in a cell of at most one edge is decided by one comparison with that
value; a cell of more bisects its own edges. Built by [`digitize_plan`](@ref) once per call.
"""
struct BucketedBinEdges{T <: Base.IEEEFloat, V <: AbstractVector{T}, C <: AbstractVector, M} <:
       AbstractVectorBinEdges{T}
    edges::V
    map::M
    last_edge::T
    cells::C
end

"""
    BucketCell(first, count, edge)

One cell of a [`BucketedBinEdges`](@ref): `first` is the index of the first edge at or above the cell,
`count` the number of edges inside it, and `edge` the value of edge `first`, `Inf` past the last edge.
"""
struct BucketCell{T}
    first::Int32
    count::Int32
    edge::T
end

function LinearCells(edges::AbstractVector{T}) where {T <: Base.IEEEFloat}
    lo, hi = findfirst(isfinite, edges), findlast(isfinite, edges)
    n_cells = 16 * (length(edges) - 1)
    span = lo === nothing ? zero(T) : edges[hi] - edges[lo]
    inv_width = T(n_cells) / span
    offset = lo === nothing ? zero(T) : -edges[lo] * inv_width
    # One cell when the finite span is empty or out of range: every query then bisects all the edges.
    if !(span > 0 && isfinite(span) && isfinite(inv_width) && isfinite(offset))
        n_cells, inv_width, offset = 1, one(T), zero(T)
    end
    return LinearCells{T}(inv_width, offset, T(n_cells - 1))
end

"""
    Log2Cells(edges)

Logarithmic cells for sorted, positive, finite `edges`, with enough mantissa bits that a cell spans less than half the
smallest gap between neighbouring edges in `log₂`, so each cell holds at most one edge — capped at `64(length(edges) -
1)` cells.
"""
function Log2Cells(edges::AbstractVector{T}) where {T <: Base.IEEEFloat}
    U = Base.uinttype(T)
    lo, hi = first(edges), last(edges)
    q = minimum(edges[k + 1] / edges[k] for k in 1:(length(edges) - 1))
    mbits = Base.significand_bits(T)
    bits = clamp(ceil(Int, log2(4 / log2(q))), 0, mbits)
    key(x, b) = reinterpret(U, x) >> (mbits - b)
    while bits > 0 && key(hi, bits) - key(lo, bits) + 1 > 64 * (length(edges) - 1)
        bits -= 1
    end
    shift = mbits - bits
    return Log2Cells{T, U}(lo, key(lo, bits), shift, key(hi, bits) - key(lo, bits))
end

function BucketedBinEdges(edges::AbstractVector{T}, map = LinearCells(edges)) where {T <: Base.IEEEFloat}
    n = length(edges)
    n_cells = Int(map.last_cell) + 1
    count = zeros(Int32, n_cells)
    for e in edges
        count[_bucket_cell(map, e) + 1] += 1
    end
    cells = Vector{BucketCell{T}}(undef, n_cells)
    first = 1
    for j in 1:n_cells
        cells[j] = BucketCell{T}(first, count[j], first <= n ? edges[first] : T(Inf))
        first += count[j]
    end
    return BucketedBinEdges{T, typeof(edges), typeof(cells), typeof(map)}(edges, map, edges[n], cells)
end

# A NaN takes the last cell: the first select sends it to `last_cell`, so the index is in range for any `x`.
@inline function _bucket_cell(m::LinearCells{T}, x::T) where {T}
    t = fma(x, m.inv_width, m.offset)
    t = ifelse(t < m.last_cell, t, m.last_cell)
    return unsafe_trunc(Int32, ifelse(t > zero(T), t, zero(T)))
end
# `x ≤ lo`, a negative `x` and a NaN take cell 0.
@inline function _bucket_cell(m::Log2Cells{T, U}, x::T) where {T, U}
    k = (reinterpret(U, x) >> m.shift) - m.key0
    return ifelse(x > m.lo, min(k, m.last_cell), zero(U)) % Int32
end
@inline _bucket_cell(b::BucketedBinEdges{T}, x) where {T} = _bucket_cell(b.map, T(x))

"""`searchsortedfirst(b, x)` given the cell `j` of `x`, which is read only when `x ≤ edges[end]`."""
@inline function _bucket_first(b::BucketedBinEdges, x, j::Integer)
    x <= b.last_edge || return length(b.edges) + 1
    c = @inbounds b.cells[j + 1]
    k = Int(c.first) + (c.edge < x)
    if c.count > 1
        k = _bisect_first(b.edges, x, Int(c.first), Int(c.first) + Int(c.count))
    end
    return k
end

"""[`_bucket_first`](@ref) with the test `x ≤ edges[end]` a select: the record of cell `j` is read for every `x`."""
@inline function _bucket_first_select(b::BucketedBinEdges, x, j::Integer)
    c = @inbounds b.cells[j + 1]
    k = Int(c.first) + (c.edge < x)
    if c.count > 1
        k = _bisect_first(b.edges, x, Int(c.first), Int(c.first) + Int(c.count))
    end
    return ifelse(x <= b.last_edge, k, length(b.edges) + 1)
end

Base.size(b::BucketedBinEdges) = size(b.edges)
Base.@propagate_inbounds Base.getindex(b::BucketedBinEdges, k::Int) = b.edges[k]
@inline Base.searchsortedfirst(b::BucketedBinEdges, x) = _bucket_first(b, x, _bucket_cell(b, x))
@inline Base.searchsortedlast(b::BucketedBinEdges, x) = _searchsortedlast_from_first(b, x)
@inline Base.searchsorted(b::BucketedBinEdges, x) = searchsortedfirst(b, x):searchsortedlast(b, x)

"""
    LogTableBinEdges(b::LogBinEdges)

The edges of `b` as a table, searched from the index estimate `⌊a log₂x + c⌋ + 2`, which is formed in
the type of `a` and `c` (`Float32` here) with `Base.FastMath.log2_fast` and stepped to the first edge
at or above `x`. The edges decide the bin, so the estimate's rounding does not. A device kernel
digitizes [`LogBinEdges`](@ref) with it.
"""
struct LogTableBinEdges{FT, T <: Base.IEEEFloat, V <: AbstractVector{T}} <: AbstractVectorBinEdges{T}
    a::FT
    c::FT
    edges::V
end

function LogTableBinEdges(b::LogBinEdges{T}, ::Type{FT} = Float32) where {FT, T <: Base.IEEEFloat}
    ll = b.log_linear
    e = collect(b)
    return LogTableBinEdges{FT, T, typeof(e)}(
        FT(log(2) / ll.step_val), FT(-ll.first_edge / ll.step_val), e)
end

Base.size(p::LogTableBinEdges) = size(p.edges)
Base.@propagate_inbounds Base.getindex(p::LogTableBinEdges, k::Int) = p.edges[k]

@inline function Base.searchsortedfirst(p::LogTableBinEdges{FT}, x) where {FT}
    e = p.edges
    x <= @inbounds(e[1]) && return 1
    n = length(e)
    k = clamp(unsafe_trunc(Int, floor(fma(Base.FastMath.log2_fast(FT(x)), p.a, p.c))) + 2, 2, n + 1)
    @inbounds while k > 2 && e[k - 1] >= x
        k -= 1
    end
    @inbounds while k <= n && !(x <= e[k])
        k += 1
    end
    return k
end
@inline Base.searchsortedlast(p::LogTableBinEdges, x) = _searchsortedlast_from_first(p, x)
@inline Base.searchsorted(p::LogTableBinEdges, x) = searchsortedfirst(p, x):searchsortedlast(p, x)

# ========================================================================= #
# 4. Infinity Padded Wrapper
# ========================================================================= #

"""
    InfPaddedBinEdges(edges::AbstractVector{T})

Wrapper that implicitly prepends \$-\\infty\$ (or `typemin(T)`) and appends \$+\\infty\$ (or `typemax(T)`) to 
an existing bin edge collection.

### Why InfPaddedBinEdges Exists
Structure function distance bins are defined as half-open intervals \$(r_i, r_{i+1}]\$. When mapping a distance 
\$r\$ to a bin, any query value \$r < \\text{first}(edges)\$ or \$r > \\text{last}(edges)\$ is out-of-bounds.
`InfPaddedBinEdges` embeds the infinite endpoints, so an inner loop needs no out-of-bounds branch:
- The first element is treated as `typemin(T)` (\$-\\infty\$).
- The last element is treated as `typemax(T)` (\$+\\infty\$).

This guarantees that every valid positive separation distance maps to a valid index without allocating 
actual padding elements in memory or copying the array. A `NaN` falls in the last bin.

### Prevention of Double Padding
The constructor checks if the input array already has infinite endpoints. If they exist, it trims them 
before wrapping to prevent nested padding (e.g. \$[-\\infty, -\\infty, ...]\$).
"""
struct InfPaddedBinEdges{T, ET <: AbstractBinEdges{T}} <: AbstractVectorBinEdges{T}
    edges::ET
end

# Infinite endpoints already present are dropped, so the padding is never doubled. Edges that need no
# trimming keep their own type, so a typed grid keeps its O(1) lookup.
function InfPaddedBinEdges(edges::AbstractVector{T}) where {T}
    lo = isinf(first(edges)) ? 2 : 1
    hi = isinf(last(edges)) ? length(edges) - 1 : length(edges)
    inner = BinEdges(lo == 1 && hi == length(edges) ? edges : @view(edges[lo:hi]))
    return InfPaddedBinEdges{T, typeof(inner)}(inner)
end

Base.size(v::InfPaddedBinEdges) = (length(v.edges) + 2,)

@inline function Base.getindex(v::InfPaddedBinEdges{T}, i::Int) where {T}
    @boundscheck checkbounds(v, i)
    if i == 1
        return typemin(T)
    elseif i == length(v.edges) + 2
        return typemax(T)
    else
        return v.edges[i - 1]
    end
end

@inline function Base.searchsortedfirst(v::InfPaddedBinEdges{T}, x) where {T}
    x <= typemin(T) && return 1
    x > last(v.edges) && return length(v.edges) + 2
    return searchsortedfirst(v.edges, x) + 1
end

@inline function Base.searchsortedfirst(v::InfPaddedBinEdges, x, o::Base.Order.Ordering)
    if o isa Base.Order.ForwardOrdering
        return searchsortedfirst(v, x)
    else
        return invoke(searchsortedfirst, Tuple{AbstractVector, Any, Base.Order.Ordering}, v, x, o)
    end
end

@inline function Base.searchsortedlast(v::InfPaddedBinEdges{T}, x) where {T}
    x < first(v.edges) && return 1
    x >= typemax(T) && return length(v.edges) + 2
    return searchsortedlast(v.edges, x) + 1
end

@inline function Base.searchsortedlast(v::InfPaddedBinEdges, x, o::Base.Order.Ordering)
    if o isa Base.Order.ForwardOrdering
        return searchsortedlast(v, x)
    else
        return invoke(searchsortedlast, Tuple{AbstractVector, Any, Base.Order.Ordering}, v, x, o)
    end
end

@inline Base.searchsorted(v::InfPaddedBinEdges, x) = searchsortedfirst(v, x):searchsortedlast(v, x)
@inline Base.searchsorted(v::InfPaddedBinEdges, x, o::Base.Order.Ordering) = searchsortedfirst(v, x, o):searchsortedlast(v, x, o)

"""
    ModeBinEdges(edges, schedule)

Distance bins applied to the lags of a mode grid. The edges are ordinary bin edges, but every pair
reaches them through the periodic kernel of the mode set `schedule` describes, so a bin's sum and
count are kernel-weighted and the histogram is soft-binned. Carried by the results of the
non-uniform FFT route, so they cannot be mistaken for pair counts.
"""
struct ModeBinEdges{T, ET <: AbstractBinEdges{T}, S} <: AbstractVectorBinEdges{T}
    edges::ET
    schedule::S
end

ModeBinEdges(edges::AbstractVector, schedule) = ModeBinEdges(BinEdges(edges), schedule)

Base.size(v::ModeBinEdges) = size(v.edges)
Base.getindex(v::ModeBinEdges, i::Int) = v.edges[i]
@inline Base.searchsortedfirst(v::ModeBinEdges, x) = searchsortedfirst(v.edges, x)
@inline Base.searchsortedfirst(v::ModeBinEdges, x, o::Base.Order.Ordering) = searchsortedfirst(v.edges, x, o)
@inline Base.searchsortedlast(v::ModeBinEdges, x) = searchsortedlast(v.edges, x)
@inline Base.searchsortedlast(v::ModeBinEdges, x, o::Base.Order.Ordering) = searchsortedlast(v.edges, x, o)
@inline Base.searchsorted(v::ModeBinEdges, x) = searchsorted(v.edges, x)
@inline Base.searchsorted(v::ModeBinEdges, x, o::Base.Order.Ordering) = searchsorted(v.edges, x, o)

"""
    midpoints(edges) -> per-bin representative separations
    midpoints!(out, edges) -> out

The abscissa each bin's average is taken to apply at, from flat edges `[e₀, e₁, …, eₙ]`
(length `n+1` → `n` values). Quadratures and finite differences over binned structure functions
need one.

Uniform and arbitrary edges give the arithmetic mean `(eᵢ + eᵢ₊₁)/2`; [`LogBinEdges`](@ref) give the
geometric mean, which is the arithmetic mean on the grid those edges are uniform on.
"""
function midpoints!(out::AbstractVector, edges::AbstractVector)
    length(out) == length(edges) - 1 ||
        throw(DimensionMismatch("out must have length $(length(edges) - 1); got $(length(out))"))
    @inbounds for i in eachindex(out)
        out[i] = (edges[i] + edges[i + 1]) / 2
    end
    return out
end

midpoints(edges::AbstractVector{T}) where {T} =
    midpoints!(similar(edges, T, length(edges) - 1), edges)

@inline midpoints(edges::AbstractRange) =
    range(first(edges) + step(edges) / 2; length = length(edges) - 1, step = step(edges))

@inline midpoints(edges::SA.SVector{N, T}) where {N, T} =
    SA.SVector{N - 1, T}(ntuple(i -> (edges[i] + edges[i + 1]) / 2, N - 1))

"""
    midpoints(edges::LinearBinEdges) -> AbstractRange
    midpoints(edges::LogBinEdges) -> LogBinEdges

Midpoints of [`LinearBinEdges`](@ref) / [`LogBinEdges`](@ref).
"""
@inline midpoints(edges::LinearBinEdges) =
    range(first(edges) + edges.step_val / 2;
        length = length(edges) - 1, step = edges.step_val)

@inline function midpoints(edges::LogBinEdges{T}) where {T}
    ll = edges.log_linear
    n = length(ll) - 1
    s = ll.step_val
    m1 = ll.first_edge + s / 2
    mn = fma(T(n - 1), s, m1)
    return LogBinEdges{T}(exp(m1), exp(mn), LinearBinEdges{T}(m1, mn, s, ll.inv_step, n))
end

"""
Fill an AbstractVector with midpoints using [`midpoints`](@ref).
"""
function midpoints!(out::AbstractVector, edges::Union{LinearBinEdges, LogBinEdges})
    length(out) == length(edges) - 1 ||
        throw(DimensionMismatch("out must have length $(length(edges) - 1); got $(length(out))"))
    return copyto!(out, midpoints(edges))
end

"""
The outer bins of [`InfPaddedBinEdges`](@ref) are unbounded, so they have no representative
separation. Take midpoints of the bounded interior, `edges.edges`.
"""
midpoints(::InfPaddedBinEdges) = throw(
    ArgumentError(
        "InfPaddedBinEdges has unbounded first and last bins with no representative separation; " *
        "call midpoints on the bounded interior, `edges.edges`",
    ),
)

midpoints(v::ModeBinEdges) = midpoints(v.edges)




"""
    n_histogram_bins(edges::AbstractVector) -> Int

Number of histogram bins for flat edges (`length == N + 1` → `N` bins).
"""
@inline n_histogram_bins(edges::AbstractVector) = length(edges) - 1

"""
    BinEdges(edges)

Normalize flat edge input to [`AbstractBinEdges`](@ref) for hot-loop `digitize`:
- existing `AbstractBinEdges` → unchanged
- `AbstractRange` → [`LinearBinEdges`](@ref)
- other `AbstractVector` → wrapped via the default [`BinEdges`](@ref) struct constructor (generic binary search)
"""
BinEdges(edges::AbstractBinEdges) = edges
BinEdges(edges::AbstractRange) = LinearBinEdges(edges)

# ========================================================================================= #
# 5. Squared-distance digitize plans
# ========================================================================================= #

"""
    _fast_log2(x)

`log2(x)` for finite `x > 0`, max error ~4e-8 (Float64). Exponent extract plus an odd series on a
mantissa recentred to `[1/√2, √2)`, so it vectorizes where the scalar `libm log` does not.
Approximate: the bin is decided by [`squared_digitize`](@ref)'s correction.
"""
@inline function _fast_log2(x::Float64)
    ix = reinterpret(UInt64, x)
    e = Int((ix >> 52) & 0x7ff) - 1023
    m = reinterpret(Float64, (ix & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000)
    big = m > 1.4142135623730951
    m = big ? 0.5m : m
    e = big ? e + 1 : e
    t = (m - 1) / (m + 1)
    t2 = t * t
    s = muladd(t2, muladd(t2, muladd(t2, 1 / 7, 1 / 5), 1 / 3), 1.0)
    return e + 2 * t * s * 1.4426950408889634
end

@inline function _fast_log2(x::Float32)
    ix = reinterpret(UInt32, x)
    e = Int((ix >> 23) & 0xff) - 127
    m = reinterpret(Float32, (ix & 0x007f_ffff) | 0x3f80_0000)
    big = m > 1.4142135f0
    m = big ? 0.5f0m : m
    e = big ? e + 1 : e
    t = (m - 1f0) / (m + 1f0)
    t2 = t * t
    s = muladd(t2, muladd(t2, 1f0 / 5, 1f0 / 3), 1f0)
    return Float32(e) + 2f0 * t * s * 1.442695f0
end

"""
    AbstractSquaredDigitizePlan

Maps a squared separation `r²` to the bin `digitize(sqrt(r²), edges)` gives, compared against the
thresholds `S_k`, the largest `s` with `sqrt(s) ≤ edges[k]`. The two agree for every `r²` because
`sqrt` is correctly rounded.
"""
abstract type AbstractSquaredDigitizePlan{T} end

"""Log-uniform edges: index from `_fast_log2(r²)` and one FMA, then corrected against the thresholds."""
struct SquaredLogPlan{T, V <: AbstractVector{T}} <: AbstractSquaredDigitizePlan{T}
    a::T          # ln2 / (2·log-step)
    b::T          # -log(first edge) / log-step
    n_bins::Int
    sqedges::V
end

"""Uniform-in-`r` edges: the squares of a uniform grid are not uniform, so this takes one `sqrt` and
the grid's own lookup."""
struct SquaredLinearPlan{T, E <: LinearBinEdges{T}} <: AbstractSquaredDigitizePlan{T}
    edges::E
    n_bins::Int
end

"""Any other sorted edges: the thresholds as a [`BucketedBinEdges`](@ref), so the vectorized half
computes the cell of `r²` and the scalar half decides from that cell's record."""
struct SquaredBucketPlan{T, B <: BucketedBinEdges{T}} <: AbstractSquaredDigitizePlan{T}
    thresholds::B
end

"""Implicit ±Inf catch-all bins around an inner plan; every pair lands somewhere."""
struct SquaredInfPaddedPlan{T, P <: AbstractSquaredDigitizePlan{T}} <: AbstractSquaredDigitizePlan{T}
    inner::P
end

"""
    _sqrt_threshold(e) -> the largest `s` with `sqrt(s) ≤ e`

`-Inf` for a negative edge, which no squared separation reaches.
"""
function _sqrt_threshold(e::T) where {T <: AbstractFloat}
    e < 0 && return T(-Inf)
    isinf(e) && return e
    s = min(e * e, floatmax(T))
    while sqrt(s) > e
        s = prevfloat(s)
    end
    while s < floatmax(T) && sqrt(nextfloat(s)) <= e
        s = nextfloat(s)
    end
    return s
end

_sqrt_thresholds(::Type{T}, v) where {T} = T[_sqrt_threshold(T(v[k])) for k in eachindex(v)]

"""
    squared_digitize_plan(edges) -> AbstractSquaredDigitizePlan

Build the `r²` digitize plan for `edges`, once per call (never in the pair loop).
"""
function squared_digitize_plan(v::LogBinEdges{T}) where {T}
    ll = v.log_linear
    sq = _sqrt_thresholds(T, v)
    return SquaredLogPlan{T, typeof(sq)}(
        T(log(2) / (2 * ll.step_val)), T(-ll.first_edge / ll.step_val), length(v) - 1, sq,
    )
end

squared_digitize_plan(v::LinearBinEdges{T}) where {T} = SquaredLinearPlan{T, typeof(v)}(v, length(v) - 1)

squared_digitize_plan(v::AbstractBinEdges{T}) where {T} =
    SquaredBucketPlan(BucketedBinEdges(_sqrt_thresholds(float(T), v)))

"""
    digitize_plan(edges) -> AbstractBinEdges

`edges` in the form a pair kernel digitizes against, built once per call: the same edges and the same
lookup. [`LinearBinEdges`](@ref) are their own plan; the edges of a [`BinEdges`](@ref) or a
[`LogBinEdges`](@ref) become a [`BucketedBinEdges`](@ref).
"""
digitize_plan(b::AbstractBinEdges) = b
digitize_plan(b::BinEdges{<:Base.IEEEFloat}) = BucketedBinEdges(b.edges)
digitize_plan(b::LogBinEdges) = (e = collect(b); BucketedBinEdges(e, Log2Cells(e)))
digitize_plan(b::InfPaddedBinEdges) = (p = digitize_plan(b.edges); InfPaddedBinEdges{eltype(p), typeof(p)}(p))
digitize_plan(b::ModeBinEdges) = ModeBinEdges(digitize_plan(b.edges), b.schedule)
digitize_plan(v::AbstractVector) = digitize_plan(BinEdges(v))
digitize_plan(t::Tuple) = map(digitize_plan, t)

squared_digitize_plan(v::InfPaddedBinEdges) = SquaredInfPaddedPlan(squared_digitize_plan(v.edges))

squared_digitize_plan(v::ModeBinEdges) = squared_digitize_plan(v.edges)

squared_digitize_plan(edges::AbstractVector) = squared_digitize_plan(BinEdges(edges))

"""Bins covered by the plan (`digitize` results in `1:n_bins` are in range)."""
@inline n_histogram_bins(p::AbstractSquaredDigitizePlan) = p.n_bins
@inline n_histogram_bins(p::SquaredBucketPlan) = length(p.thresholds) - 1
@inline n_histogram_bins(p::SquaredInfPaddedPlan) = n_histogram_bins(p.inner) + 2

"""
    has_vector_index(plan) -> Bool

Whether the plan's index is branch-free, and so worth computing in the vectorized half of a pair
kernel. False for the linear plan, whose `searchsortedfirst` branches would de-vectorize the whole
`@simd` body.
"""
@inline has_vector_index(::AbstractSquaredDigitizePlan) = false
@inline has_vector_index(::SquaredLogPlan) = true
@inline has_vector_index(::SquaredBucketPlan) = true
@inline has_vector_index(p::SquaredInfPaddedPlan) = has_vector_index(p.inner)

"""
    squared_approx_index(plan, r2) -> Int32

The index the vectorized half of a pair kernel computes from `r²`, branch-free: the approximate
`searchsortedfirst` index of the log plan, the cell of the bucket plan. Meaningful only when
[`has_vector_index`](@ref); other plans return `0`.
"""
@inline squared_approx_index(p::SquaredLogPlan, r2) =
    unsafe_trunc(Int32, floor(muladd(_fast_log2(r2), p.a, p.b))) + Int32(1)
@inline squared_approx_index(p::SquaredBucketPlan, r2) = _bucket_cell(p.thresholds, r2)
@inline squared_approx_index(::AbstractSquaredDigitizePlan, r2) = Int32(0)
@inline squared_approx_index(p::SquaredInfPaddedPlan, r2) = squared_approx_index(p.inner, r2)

"""
    digitize_key(plan, r2)

The quantity the plan's scalar half compares against, computed in the vectorized half from `r²`.

`r²` for the log and bucket plans; `√r²` for the linear plan, whose grid is uniform in `r`.
"""
@inline digitize_key(::SquaredLogPlan, r2) = r2
@inline digitize_key(::SquaredBucketPlan, r2) = r2
@inline digitize_key(::SquaredLinearPlan, r2) = sqrt(r2)
@inline digitize_key(p::SquaredInfPaddedPlan, r2) = digitize_key(p.inner, r2)

"""
    squared_bin(plan, key, i) -> Int

The bin, in the scalar half, from [`digitize_key`](@ref) and, when [`has_vector_index`](@ref), the
index `i` the vectorized half computed; otherwise `i` is ignored.
"""
@inline squared_bin(p::SquaredLogPlan, key, i::Integer) = squared_correct(p, key, i) - 1
@inline squared_bin(p::SquaredLinearPlan, key, ::Integer) = searchsortedfirst(p.edges, key) - 1
@inline squared_bin(p::SquaredBucketPlan, key, i::Integer) = _bucket_first(p.thresholds, key, i) - 1
# The implicit -Inf edge shifts every inner index up by one; the inner plan already reports
# `n_bins + 1` above its last edge, which becomes the overflow bin. No separate range test needed.
@inline squared_bin(p::SquaredInfPaddedPlan, key, i::Integer) = squared_bin(p.inner, key, i) + 1

"""
    squared_bin_select(plan, key, i) -> Int

[`squared_bin`](@ref)`(plan, key, i)`, with a bucketed plan's test of `key` against its last threshold taken as a
select, so the cell's record is read for every `key`.
"""
@inline squared_bin_select(p::AbstractSquaredDigitizePlan, key, i::Integer) = squared_bin(p, key, i)
@inline squared_bin_select(p::SquaredBucketPlan, key, i::Integer) = _bucket_first_select(p.thresholds, key, i) - 1
@inline squared_bin_select(p::SquaredInfPaddedPlan, key, i::Integer) = squared_bin_select(p.inner, key, i) + 1

"""
    squared_in_range(plan, key) -> Bool

Whether a pair whose [`digitize_key`](@ref) is `key` lands in one of the plan's bins: `key` above the first
threshold and at or below the last, which is `1 ≤ squared_bin(plan, key, i) ≤ n_histogram_bins(plan)` for
every `key`, `false` for NaN. Every pair lands in a bin of an implicitly padded plan.
"""
@inline squared_in_range(p::SquaredLinearPlan, key) = (key > first(p.edges)) & (key <= last(p.edges))
@inline squared_in_range(p::SquaredLogPlan, key) = (key > first(p.sqedges)) & (key <= last(p.sqedges))
@inline squared_in_range(p::SquaredBucketPlan, key) = (key > first(p.thresholds.edges)) & (key <= p.thresholds.last_edge)
@inline squared_in_range(::SquaredInfPaddedPlan, key) = true

"""
    vector_digitize(edges, x) -> Int32

`digitize(x, edges)` formed with selects in place of branches, so a vectorized loop can compute it; defined
where [`has_vector_digitize`](@ref) holds.
"""
@inline function vector_digitize(b::LinearBinEdges{T}, x::T) where {T}
    t = fma(x, b.inv_step, -b.first_edge * b.inv_step)
    k = clamp(unsafe_trunc(Int32, ceil(ifelse(isfinite(t), t, zero(T)))) + Int32(1), Int32(2), Int32(b.n_edges))
    k = ifelse(x <= b.first_edge, Int32(1), k)
    k = ifelse(x <= b.last_edge, k, Int32(b.n_edges + 1))
    return k - Int32(1)
end

"""Whether [`vector_digitize`](@ref) takes `edges` and values of type `T`."""
@inline has_vector_digitize(::LinearBinEdges{T}, ::Type{T}) where {T} = true
@inline has_vector_digitize(_, ::Type) = false

"""
    squared_correct(plan, r2, i) -> Int

Walk `i` to the exact `searchsortedfirst(sqedges, r²)`. 0 or 1 step for a random separation; within a
few ulps of an edge it can take more, so it loops.
"""
@inline function squared_correct(p::SquaredLogPlan, r2, i::Integer)
    sq = p.sqedges
    n = p.n_bins
    k = clamp(Int(i), 1, n + 2)
    @inbounds begin
        while k > 1 && sq[k - 1] >= r2
            k -= 1
        end
        while k <= n + 1 && !(r2 <= sq[k])
            k += 1
        end
    end
    return k
end

"""
    squared_digitize(plan, r2) -> Int

Exact `digitize(r, edges)` computed from `r²` alone. Out-of-range gives `0` (below) or `n_bins + 1`
(above), matching [`HelperFunctions.digitize`](@ref).
"""
@inline squared_digitize(p::AbstractSquaredDigitizePlan, r2) =
    squared_bin(p, digitize_key(p, r2), squared_approx_index(p, r2))


# ========================================================================================= #
# 6. Tapers and the harmonic nodes of a kernel-binned statistic
# ========================================================================================= #

"""
    AbstractTaper

A weight applied before a transform: on a lag-space autocovariance, as a function of the lag's
length; on a spherical harmonic series, as a function of the degree. `NoTaper()` leaves every term as
it is, `Bartlett()` falls linearly to zero at the largest lag or degree, `GaussianTaper(σ)` weights a
lag of length `r` by `exp(-r²/2σ²)` and a degree `l` by `exp(-l(l+1)σ²/2)`, with `σ` in the lag's or
the sphere's own unit. A taper trades resolution for variance, or a hard bin for a positive kernel.
"""
abstract type AbstractTaper end

"""The taper that leaves every term as it is; see [`AbstractTaper`](@ref)."""
struct NoTaper <: AbstractTaper end

"""The taper falling linearly to zero at the largest lag or degree; see [`AbstractTaper`](@ref)."""
struct Bartlett <: AbstractTaper end

"""
    GaussianTaper(σ)

The Gaussian taper `exp(-r²/2σ²)` on a lag of length `r`, `exp(-l(l+1)σ²/2)` on a degree `l`, or
`exp(-k²σ²/2)` on a mode of wavenumber `k`, with `σ` a length; see [`AbstractTaper`](@ref).
"""
struct GaussianTaper{T <: Real} <: AbstractTaper
    σ::T
end

"""Weight of a lag of length `r` when the largest lag has length `r_max`."""
@inline taper_weight(::NoTaper, r, r_max) = one(r)
@inline taper_weight(::Bartlett, r, r_max) = max(zero(r), one(r) - r / r_max)
@inline taper_weight(t::GaussianTaper, r, r_max) = exp(-r * r / (2 * t.σ * t.σ))

"""Weight of degree `l` in a series truncated at `lmax`."""
@inline harmonic_taper(::NoTaper, l::Integer, lmax::Integer) = 1.0
@inline harmonic_taper(::Bartlett, l::Integer, lmax::Integer) = max(0.0, 1.0 - l / (lmax + 1))
@inline harmonic_taper(t::GaussianTaper, l::Integer, lmax::Integer) = exp(-l * (l + 1) * t.σ^2 / 2)

"""
Weight of a Fourier mode of squared angular wavenumber `k2`, `σ` a length: the mode set's periodic
kernel becomes the inverse transform of the squared weights.
"""
@inline mode_taper(::NoTaper, k2) = one(k2)
@inline mode_taper(t::GaussianTaper, k2) = exp(-k2 * t.σ^2 / 2)

"""
    HarmonicNodes(separations, lmax; taper = NoTaper())
    HarmonicNodes(n::Integer, lmax; taper = NoTaper())

The "bins" of a kernel-binned pair statistic on a sphere: the central angles `separations` (radians)
at which it is reported, the degree `lmax` its harmonic series is truncated at, and the `taper` on
that series. Together they are the kernel, `K(γ, β) = (1/16π²) Σ_{l ≤ lmax} (2l+1) b_l d^l(cos γ) d^l(cos β)`,
which replaces a hard bin around `β`; it narrows as `π/lmax` and tends to a delta in `cos γ`.

`weights` are quadrature weights in `μ = cos β`, `∫_{-1}^{1} f dμ ≈ Σ_k w_k f(μ_k)`: Gauss–Legendre for
the `n`-node form, whose nodes are the Gauss–Legendre points of `μ`, and the midpoint rule on the
nodes' own cells for given separations. They are what inverts the statistic back to a spectrum.

Not a vector of edges: one value is reported per node, so a result built on these has as many values
as nodes.
"""
struct HarmonicNodes{T <: Real, SV <: AbstractVector{T}, WV <: AbstractVector{T}, B <: AbstractTaper}
    separations::SV
    weights::WV
    lmax::Int
    taper::B
    function HarmonicNodes(separations::AbstractVector{T}, weights::AbstractVector{T}, lmax::Integer,
                           taper::B) where {T <: Real, B <: AbstractTaper}
        length(separations) == length(weights) || throw(DimensionMismatch(
            "$(length(separations)) separations and $(length(weights)) weights",
        ))
        issorted(separations) || throw(ArgumentError("separations must be sorted"))
        (isempty(separations) || (first(separations) >= 0 && last(separations) <= π)) ||
            throw(ArgumentError("separations are central angles in radians, in [0, π]"))
        lmax >= 0 || throw(ArgumentError("lmax must be non-negative"))
        return new{T, typeof(separations), typeof(weights), B}(separations, weights, Int(lmax), taper)
    end
end

function HarmonicNodes(separations::AbstractVector{<:Real}, lmax::Integer; taper::AbstractTaper = NoTaper())
    T = float(eltype(separations))
    sep = convert(AbstractVector{T}, separations)
    μ = cos.(sep)
    n = length(μ)
    w = Vector{T}(undef, n)
    @inbounds for k in 1:n
        hi = k == 1 ? one(T) : (μ[k - 1] + μ[k]) / 2
        lo = k == n ? -one(T) : (μ[k] + μ[k + 1]) / 2
        w[k] = hi - lo
    end
    return HarmonicNodes(sep, w, lmax, taper)
end

function HarmonicNodes(n::Integer, lmax::Integer; taper::AbstractTaper = NoTaper())
    μ, w = gauss_legendre(Int(n))
    return HarmonicNodes(acos.(reverse(μ)), reverse(w), lmax, taper)
end

"""
    gauss_legendre(n) -> (nodes, weights)

The `n` Gauss–Legendre nodes on `(-1, 1)`, ascending, and their weights, by Newton's method on `P_n`
from the Tricomi estimate.
"""
function gauss_legendre(n::Integer)
    n >= 1 || throw(ArgumentError("need at least one node"))
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    for k in 1:n
        z = cos(π * (k - 0.25) / (n + 0.5))
        dp = 0.0
        for _ in 1:100
            p0, p1 = 1.0, z
            for l in 1:(n - 1)
                p0, p1 = p1, ((2l + 1) * z * p1 - l * p0) / (l + 1)
            end
            dp = n * (z * p1 - p0) / (z * z - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) < 1e-15 && break
        end
        x[n + 1 - k] = z
        w[n + 1 - k] = 2 / ((1 - z * z) * dp * dp)
    end
    return x, w
end

@inline midpoints(nodes::HarmonicNodes) = nodes.separations
@inline n_histogram_bins(nodes::HarmonicNodes) = length(nodes.separations)
@inline Base.length(nodes::HarmonicNodes) = length(nodes.separations)
Base.:(==)(a::HarmonicNodes, b::HarmonicNodes) =
    a.separations == b.separations && a.weights == b.weights && a.lmax == b.lmax && a.taper == b.taper
