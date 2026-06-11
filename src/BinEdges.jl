# Custom bin edges for fast O(1) index digitizing/binning.

abstract type AbstractBinEdges{T} <: AbstractVector{T} end

# ========================================================================= #
# 1. Plain Fallback / Wrapped Bin Edges
# ========================================================================= #

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
# 2. Linear/Uniform Spacing (FMA Linear Search V3)
# ========================================================================= #

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
    if x > l
        return length(v.edges) + 1
    end
    
    idx = round(Int, muladd(x, v.inv_step, v.offset))
    idx = clamp(idx, 1, length(v.edges))
    @inbounds edge_val = muladd(T(idx - 1), v.step_val, f)
    return edge_val < x ? idx + 1 : idx
end

@inline function Base.searchsortedfirst(v::LinearBinEdges, x, o::Base.Order.Ordering)
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

# Auto-convert to LinearBinEdges if user constructs BinEdges around a range
BinEdges(edges::AbstractRange{T}) where {T} = LinearBinEdges(edges)

# ========================================================================= #
# 3. Log-Uniform Spacing (Exponent LUT Hybrid Search V9)
# ========================================================================= #

struct LogBinEdges{T, RT <: AbstractRange, VT <: AbstractVector} <: AbstractBinEdges{T}
    log_edges::RT
    edges::VT
    lut::Vector{Int}
    e_min::Int
    e_max::Int
    inv_step::T
    offset::T
end

function LogBinEdges(edges::AbstractVector{T}) where {T}
    any(x -> x <= zero(T), edges) && throw(ArgumentError("Log-spaced bin edges must be strictly positive."))
    
    log_start = log(first(edges))
    log_stop = log(last(edges))
    log_edges = range(log_start, log_stop, length=length(edges))
    
    e_min = exponent(first(edges))
    e_max = exponent(last(edges))
    lut_size = e_max - e_min + 2
    lut = Vector{Int}(undef, lut_size)
    for e in e_min:(e_max+1)
        val = T(2.0^e)
        lut[e - e_min + 1] = searchsortedfirst(edges, val)
    end
    
    inv_step = inv(step(log_edges))
    offset = T(1.0) - first(log_edges) * inv_step
    
    return LogBinEdges{T, typeof(log_edges), typeof(edges)}(
        log_edges, edges, lut, e_min, e_max, inv_step, offset
    )
end

Base.size(v::LogBinEdges) = size(v.edges)
Base.getindex(v::LogBinEdges, i::Int) = v.edges[i]

@inline function Base.searchsortedfirst(v::LogBinEdges{T}, x) where {T}
    f = first(v.edges)
    if x <= f
        return 1
    end
    l = last(v.edges)
    if x > l
        return length(v.edges) + 1
    end
    
    e = exponent(x)
    e_idx = clamp(e - v.e_min + 1, 1, length(v.lut))
    idx_start = @inbounds v.lut[e_idx]
    idx_end = (e_idx < length(v.lut)) ? @inbounds(v.lut[e_idx+1]) : length(v.edges)
    
    if idx_end - idx_start <= 8
        idx = idx_start
        @inbounds while idx <= idx_end && v.edges[idx] < x
            idx += 1
        end
        return idx
    else
        low = idx_start
        high = idx_end
        @inbounds while low <= high
            mid = (low + high) >>> 1
            if v.edges[mid] < x
                low = mid + 1
            else
                high = mid - 1
            end
        end
        return low
    end
end

@inline function Base.searchsortedfirst(v::LogBinEdges, x, o::Base.Order.Ordering)
    if o isa Base.Order.ForwardOrdering
        return searchsortedfirst(v, x)
    else
        return searchsortedfirst(v.edges, x, o)
    end
end

@inline Base.searchsortedlast(v::LogBinEdges, x) = searchsortedlast(v.edges, x)
@inline Base.searchsortedlast(v::LogBinEdges, x, o::Base.Order.Ordering) = searchsortedlast(v.edges, x, o)
@inline Base.searchsorted(v::LogBinEdges, x) = searchsorted(v.edges, x)
@inline Base.searchsorted(v::LogBinEdges, x, o::Base.Order.Ordering) = searchsorted(v.edges, x, o)

# ========================================================================= #
# 4. Infinity Padded Wrapper
# ========================================================================= #

struct InfPaddedBinEdges{T, ET <: AbstractBinEdges{T}} <: AbstractBinEdges{T}
    edges::ET
end

# Generic constructor for raw vectors / AbstractVectors
function InfPaddedBinEdges(edges::AbstractVector{T}) where {T}
    start_idx = isinf(first(edges)) ? 2 : 1
    end_idx = isinf(last(edges)) ? length(edges) - 1 : length(edges)
    trimmed = @view edges[start_idx:end_idx]
    
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
    if x <= typemin(T)
        return 1
    elseif x > last(v.edges)
        return length(v.edges) + 2
    else
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
