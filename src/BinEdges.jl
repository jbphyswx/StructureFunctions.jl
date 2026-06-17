# Custom bin edges for fast O(1) index digitizing/binning.

"""
    AbstractBinEdges{T} <: AbstractVector{T}

Supertype for all custom, high-performance bin edge collections in `StructureFunctions.jl`.

### Why AbstractBinEdges Exists
In structure function calculations over large datasets, the spatial separation distance \$r\$ for each of
the \$O(N^2)\$ point pairs must be mapped to its corresponding distance bin (index). Using a standard sorted 
vector of bin edges requires a binary search (`searchsortedfirst`), which has \$O(\\log B)\$ complexity 
where \$B\$ is the number of bins. 

For large \$N\$, this binary search becomes the dominant CPU bottleneck, causing high branch mispredictions and 
cache misses. Subtypes of `AbstractBinEdges` bypass the standard binary search by implementing custom 
`Base.searchsortedfirst` overrides that execute in \$O(1)\$ time:
- `LinearBinEdges` utilizes Fused Multiply-Add (FMA) arithmetic for uniformly-spaced bins.
- `LogBinEdges` maps physical queries via `log(q)` then the same FMA path on the log-space grid.

Wrapping standard arrays in these subtypes allows `digitize` to execute 5x to 15x faster, resolving the 
primary computational bottleneck in the package.

## Bin edges and backends

- Plain `AbstractVector` bin edges are valid everywhere but use generic `searchsortedfirst` / binary search.
- Pass [`LinearBinEdges`](@ref), [`LogBinEdges`](@ref), or [`InfPaddedBinEdges`](@ref) for O(1) CPU digitize
  and matching GPU tiled kernels (see `StructureFunctionsGPUExt`).
- [`InfPaddedBinEdges`](@ref) adds implicit catch-all under/overflow bins; do not `vcat(-Inf, …, Inf)` manually.
- Classic serial/threaded pair-loop paths normalize once via [`BinEdges`](@ref); single-pass APIs expect callers
  to choose edge types explicitly (no auto-wrap on GPU).
"""
abstract type AbstractBinEdges{T} <: AbstractVector{T} end

# ========================================================================= #
# 1. Plain Fallback / Wrapped Bin Edges
# ========================================================================= #

"""
    BinEdges(edges::AbstractVector{T})

Generic fallback wrapper for arbitrary sorted vectors of bin edges.
Bypasses range-specific optimizations but conforms to the `AbstractBinEdges` interface.

### Behavior
- If constructed with an `AbstractRange` (e.g. `StepRange` or `StepRangeLen`), it automatically promotes
  and returns a `LinearBinEdges` wrapper to enable O(1) FMA indexing.
- Otherwise, it wraps the vector and delegates to standard \$O(\\log N)\$ binary search methods.
"""
struct BinEdges{T, ET <: AbstractVector{T}} <: AbstractBinEdges{T}
    edges::ET
end

Base.size(v::BinEdges) = size(v.edges)
Base.getindex(v::BinEdges, i::Int) = v.edges[i]

@inline Base.searchsortedfirst(v::BinEdges, x) = searchsortedfirst(v.edges, x)
@inline Base.searchsortedfirst(v::BinEdges, x, o::Base.Order.Ordering) = searchsortedfirst(v.edges, x, o)
@inline Base.searchsortedlast(v::BinEdges, x) = searchsortedlast(v.edges, x)
@inline Base.searchsortedlast(v::BinEdges, x, o::Base.Order.Ordering) = searchsortedlast(v.edges, x, o)
@inline Base.searchsorted(v::BinEdges, x) = searchsorted(v.edges, x)
@inline Base.searchsorted(v::BinEdges, x, o::Base.Order.Ordering) = searchsorted(v.edges, x, o)

# ========================================================================= #
# 2. Linear/Uniform Spacing (FMA Linear Search)
# ========================================================================= #

"""
    LinearBinEdges(edges::AbstractRange{T})

High-performance wrapper for uniformly-spaced ranges (linear spacing).

### Mathematical Theory
A standard binary search takes \$O(\\log B)\$ steps. With a uniformly spaced range of bin edges \$v_i = v_1 + (i-1)\\delta\$,
`searchsortedfirst(v, x)` is the smallest index \$i\$ with \$v_i \\ge x\$:
\$\$
i^*(x) = \\min\\{ i : v_1 + (i-1)\\delta \\ge x \\}
     = \\left\\lfloor \\frac{x - v_1}{\\delta} \\right\\rfloor + 1
     = \\lceil \\frac{x - v_1}{\\delta} + 1 \\rceil .
\$\$
The discrete operator is **ceiling / floor+1**, not `round` (which answers a different question).

Precompute `inv_step = 1/δ` and at query time one FMA gives \$t = (x-v_1)/\\delta\$:
`t = muladd(x, inv_step, -v_1 * inv_step)` then `idx = floor(Int, t) + 1`.

### Float boundary correction
`t` and reconstructed edges are inexact. After clamping `idx`, compare the reconstructed left edge
`u_idx = v_1 + (idx-1)δ`; if `u_idx < x`, return `idx + 1`. One comparison fixes the at-most-one-bin FP
error without a second downward correction.

### Performance
Bypasses the Twice-Precision arithmetic of Julia's standard `StepRangeLen` search. 
Reduces lookup time from **~46 ns** to **~3 ns** (a 15x speedup), completely eliminating the linear binning bottleneck.
"""
struct LinearBinEdges{T, RT <: AbstractRange{T}} <: AbstractBinEdges{T}
    edges::RT
    inv_step::T
    offset::T
    first_edge::T
    last_edge::T
    step_val::T
end

function LinearBinEdges(edges::AbstractRange{T}) where {T}
    inv_step = inv(step(edges))
    offset = T(1.0) - first(edges) * inv_step
    return LinearBinEdges{T, typeof(edges)}(
        edges, inv_step, offset, first(edges), last(edges), step(edges)
    )
end

Base.size(v::LinearBinEdges) = size(v.edges)
Base.getindex(v::LinearBinEdges, i::Int) = v.edges[i]

@inline function Base.searchsortedfirst(v::LinearBinEdges{T}, x) where {T}
    f = v.first_edge
    if x <= f
        return 1
    end
    l = v.last_edge
    n = length(v.edges)
    if x > l
        return n + 1
    end

    # i* = floor((x - f)/δ) + 1  — one FMA for (x - f)/δ
    t = muladd(x, v.inv_step, -f * v.inv_step)
    idx = clamp(floor(Int, t) + 1, 1, n)

    @inbounds u = muladd(T(idx - 1), v.step_val, f)
    return u < x ? idx + 1 : idx
end

@inline function Base.searchsortedfirst(v::LinearBinEdges, x, o::Base.Order.Ordering)
    # Fast path for forward ordering (default), fallback for custom ordering
    if o isa Base.Order.ForwardOrdering
        return searchsortedfirst(v, x)
    else
        return searchsortedfirst(v.edges, x, o)
    end
end

@inline Base.searchsortedlast(v::LinearBinEdges, x) = searchsortedlast(v.edges, x)
@inline Base.searchsortedlast(v::LinearBinEdges, x, o::Base.Order.Ordering) = searchsortedlast(v.edges, x, o)
@inline Base.searchsorted(v::LinearBinEdges, x) = searchsorted(v.edges, x)
@inline Base.searchsorted(v::LinearBinEdges, x, o::Base.Order.Ordering) = searchsorted(v.edges, x, o)

# ========================================================================= #
# 3. Log-uniform spacing (log(q) + LinearBinEdges on log grid)
# ========================================================================= #

"""
    LogBinEdges(edges::AbstractVector{T})
    LogBinEdges_from_log_edges(log_edges)

Log-spaced (geometric) bin edges: uniform grid in log-space.

### Digitize semantics

`log_edges` is the authoritative grid. A physical query `q > 0` maps to
`searchsortedfirst(LinearBinEdges(log_edges), log(q))`. See
[`docs/UNIFORM_BIN_DIGITIZE.md`](@ref) and `benchmark/LOG_BIN_EDGES_BENCHMARK.md`.

[`LogBinEdges`](@ref) from a physical vector builds the same log grid from
`range(log(first), log(last); length)`. [`LogBinEdges_from_log_edges`](@ref) accepts
the log grid directly.

### Performance

One `log(q)` (~6 ns) plus O(1) FMA digitize (~2.5 ns) on the log grid. See benchmark log.
"""
struct LogBinEdges{T, LRT <: AbstractRange{T}, LBET <: LinearBinEdges{T}} <: AbstractBinEdges{T}
    log_edges::LRT
    log_linear::LBET
end

function _LogBinEdges_core(log_edges::AbstractRange{T}) where {T}
    log_linear = LinearBinEdges(log_edges)
    return LogBinEdges{T, typeof(log_edges), typeof(log_linear)}(log_edges, log_linear)
end

function LogBinEdges(edges::AbstractVector{T}) where {T}
    any(x -> x <= zero(T), edges) && throw(ArgumentError("Log-spaced bin edges must be strictly positive."))
    log_edges = range(log(first(edges)), log(last(edges)); length=length(edges))
    return _LogBinEdges_core(log_edges)
end

"""
    LogBinEdges_from_log_edges(log_edges)

Build log-uniform bins from log-space edges (`u` with physical edge `exp(u)`).
"""
LogBinEdges_from_log_edges(log_edges::AbstractRange{T}) where {T} =
    _LogBinEdges_core(log_edges)
LogBinEdges_from_log_edges(log_edges::AbstractVector{T}) where {T} =
    _LogBinEdges_core(range(first(log_edges), last(log_edges); length=length(log_edges)))

"""
    physical_edges_vector(bins::LogBinEdges) -> Vector

Materialize physical bin edges `exp(log_edges[i])` for display or legacy consumers.
Not used on the digitize hot path.
"""
function physical_edges_vector(bins::LogBinEdges{T}) where {T}
    return T[exp(bins.log_edges[i]) for i in 1:length(bins.log_edges)]
end

function LogBinEdges(::AbstractRange{T}) where {T}
    throw(ArgumentError("LogBinEdges does not support AbstractRange input. Use LogBinEdges_from_log_edges() instead."))
end

Base.size(v::LogBinEdges) = (length(v.log_edges),)
Base.getindex(v::LogBinEdges, i::Int) = exp(v.log_edges[i])

@inline function Base.searchsortedfirst(v::LogBinEdges{T}, x) where {T}
    x <= zero(T) && return 1
    return searchsortedfirst(v.log_linear, log(x))
end

@inline function Base.searchsortedfirst(v::LogBinEdges, x, o::Base.Order.Ordering)
    if o isa Base.Order.ForwardOrdering
        return searchsortedfirst(v, x)
    else
        return searchsortedfirst(physical_edges_vector(v), x, o)
    end
end

@inline function Base.searchsortedlast(v::LogBinEdges{T}, x) where {T}
    x < zero(T) && return 0
    return searchsortedlast(v.log_linear, log(x))
end

@inline function Base.searchsortedlast(v::LogBinEdges, x, o::Base.Order.Ordering)
    if o isa Base.Order.ForwardOrdering
        return searchsortedlast(v, x)
    else
        return searchsortedlast(physical_edges_vector(v), x, o)
    end
end

@inline Base.searchsorted(v::LogBinEdges, x) = searchsortedfirst(v, x):searchsortedlast(v, x)
@inline function Base.searchsorted(v::LogBinEdges, x, o::Base.Order.Ordering)
    searchsortedfirst(v, x, o):searchsortedlast(v, x, o)
end

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
Instead of checking for these out-of-bound cases manually using branches in inner loops, `InfPaddedBinEdges`
embeds the infinite endpoints implicitly:
- The first element is treated as `typemin(T)` (\$-\\infty\$).
- The last element is treated as `typemax(T)` (\$+\\infty\$).

This guarantees that every valid positive separation distance maps to a valid index without allocating 
actual padding elements in memory or copying the array.

### Prevention of Double Padding
The constructor checks if the input array already has infinite endpoints. If they exist, it trims them 
before wrapping to prevent nested padding (e.g. \$[-\\infty, -\\infty, ...]\$).
"""
struct InfPaddedBinEdges{T, ET <: AbstractBinEdges{T}} <: AbstractBinEdges{T}
    edges::ET
end

# Generic constructor for raw vectors / AbstractVectors
function InfPaddedBinEdges(edges::AbstractVector{T}) where {T}
    # Check for existing infinite endpoints to prevent double-padding
    start_idx = isinf(first(edges)) ? 2 : 1
    end_idx = isinf(last(edges)) ? length(edges) - 1 : length(edges)
    trimmed = @view edges[start_idx:end_idx]
    
    # Construct or wrap appropriately
    if trimmed isa AbstractBinEdges
        return InfPaddedBinEdges{T, typeof(trimmed)}(trimmed)
    elseif trimmed isa AbstractRange
        wrapped = LinearBinEdges(trimmed)
        return InfPaddedBinEdges{T, typeof(wrapped)}(wrapped)
    else
        wrapped = BinEdges(trimmed)
        return InfPaddedBinEdges{T, typeof(wrapped)}(wrapped)
    end
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
    # Direct check against out-of-bound limits
    if x <= typemin(T)
        return 1
    elseif x > last(v.edges)
        return length(v.edges) + 2
    else
        # Offset index by 1 to account for the implicit -Inf element at index 1
        return searchsortedfirst(v.edges, x) + 1
    end
end

@inline function Base.searchsortedfirst(v::InfPaddedBinEdges, x, o::Base.Order.Ordering)
    if o isa Base.Order.ForwardOrdering
        return searchsortedfirst(v, x)
    else
        return invoke(searchsortedfirst, Tuple{AbstractVector, Any, Base.Order.Ordering}, v, x, o)
    end
end

@inline function Base.searchsortedlast(v::InfPaddedBinEdges{T}, x) where {T}
    if x < first(v.edges)
        return 1
    elseif x >= typemax(T)
        return length(v.edges) + 2
    else
        return searchsortedlast(v.edges, x) + 1
    end
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
