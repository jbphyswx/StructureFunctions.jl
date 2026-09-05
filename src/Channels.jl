"""
Multi-channel fields: several quantities sampled at the same points, differenced together.

A structure function is a statement about one pair and any number of quantities carried at its two
ends. Velocity is a **vector** channel, parallel-transported by the geometry before differencing; a
tracer, a vorticity, a temperature is a **scalar** channel, differenced as it stands. Both are the
same pair sweep, so they belong in one field rather than in separate calls.
"""
module Channels

using StaticArrays: StaticArrays as SA

export Fields, ChannelIncrement, n_vector_channels, n_scalar_channels, channel_dimension

"""
    Fields(; vectors = (), scalars = ())

Several quantities sampled at one set of points: `vectors` are parallel-transported before being
differenced, `scalars` are differenced as they stand.

```julia
Fields(vectors = (u,))                        # a velocity — what a bare `u` already means
Fields(vectors = (u,), scalars = (θ,))        # velocity and a tracer, for the mixed laws
Fields(vectors = (u, 𝓐u))                     # velocity and its advection
Fields(scalars = (ω, 𝓐ω))                     # a scalar and its advection
```

Each vector is `(D, N)` and each scalar is `(N,)` or `(1, N)`. They are packed once, at construction,
into the single `(V·D + K, N)` array the kernels already load — so a pair costs one contiguous read
whatever it carries, and the channel counts are type parameters, so the kernel specialises on them.

The name says what it is: these are the fields; those ones are vector fields; those are scalar
fields. Which is transported and which is not follows from that, and from nothing else.
"""
struct Fields{D, V, K, A <: AbstractMatrix}
    data::A
end

"""Velocity-like channels, each parallel-transported."""
@inline n_vector_channels(::Fields{D, V, K}) where {D, V, K} = V

"""Tracer-like channels, each differenced without transport."""
@inline n_scalar_channels(::Fields{D, V, K}) where {D, V, K} = K

"""Components in each vector channel."""
@inline channel_dimension(::Fields{D}) where {D} = D

@inline packed(f::Fields) = f.data

@inline Base.size(f::Fields) = size(f.data)
@inline Base.eltype(::Fields{D, V, K, A}) where {D, V, K, A} = eltype(A)

function Fields(; vectors = (), scalars = ())
    V = length(vectors)
    K = length(scalars)
    V * K >= 0 || throw(ArgumentError("channel counts cannot be negative"))
    V + K > 0 || throw(ArgumentError(
        "a field needs at least one channel; got no vectors and no scalars",
    ))
    D = V == 0 ? 0 : size(first(vectors), 1)
    for (i, v) in enumerate(vectors)
        ndims(v) == 2 || throw(ArgumentError(
            "vector channel $i must be (D, N); got an array of $(ndims(v)) dimensions",
        ))
        size(v, 1) == D || throw(DimensionMismatch(
            "vector channel $i has $(size(v, 1)) components, channel 1 has $D; every vector " *
            "channel is transported by the same geometry and so must have the same dimension",
        ))
    end
    N = V > 0 ? size(first(vectors), 2) : length(first(scalars))
    for (i, v) in enumerate(vectors)
        size(v, 2) == N || throw(DimensionMismatch(
            "vector channel $i covers $(size(v, 2)) points, expected $N",
        ))
    end
    for (i, s) in enumerate(scalars)
        length(s) == N || throw(DimensionMismatch(
            "scalar channel $i covers $(length(s)) points, expected $N",
        ))
    end

    T = promote_type((eltype(v) for v in vectors)..., (eltype(s) for s in scalars)...)
    data = Matrix{T}(undef, V * D + K, N)
    @inbounds for (i, v) in enumerate(vectors), n in 1:N, d in 1:D
        data[(i - 1) * D + d, n] = v[d, n]
    end
    @inbounds for (i, s) in enumerate(scalars), n in 1:N
        data[V * D + i, n] = s[n]
    end
    return Fields{D, V, K, typeof(data)}(data)
end

"""
    ChannelIncrement(vectors, scalars)

One pair's increment across every channel: each vector channel already transported into the pair's
common frame, each scalar channel already differenced.

An operator reads the channels it names. A field of one vector channel and no scalars does **not**
produce one of these — its increment is the plain `SVector` every existing operator already takes, so
the single-channel path is unchanged down to the instruction.
"""
struct ChannelIncrement{D, V, K, T}
    vectors::NTuple{V, SA.SVector{D, T}}
    scalars::NTuple{K, T}
end

"""The `i`-th transported vector increment."""
@inline vector_channel(c::ChannelIncrement, i::Integer) = @inbounds c.vectors[i]

"""The `i`-th scalar increment."""
@inline scalar_channel(c::ChannelIncrement, i::Integer) = @inbounds c.scalars[i]

@inline n_vector_channels(::ChannelIncrement{D, V, K}) where {D, V, K} = V
@inline n_scalar_channels(::ChannelIncrement{D, V, K}) where {D, V, K} = K
@inline channel_dimension(::ChannelIncrement{D}) where {D} = D

end # module Channels
