"""
Several fields at one set of points, differenced together.

A structure function is a statement about one pair and any number of quantities carried at its two
ends. Velocity is a **vector** field, parallel-transported by the geometry before differencing; a
tracer, a vorticity, a temperature is a **scalar** field, differenced as it stands. Both are the
same pair sweep, so they travel together in one [`Fields`](@ref).
"""
module MultiFields

using StaticArrays: StaticArrays as SA

export Fields, FieldIncrement, n_vector_fields, n_scalar_fields, field_dimension

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

Each vector is `(D, N)` and each scalar is `(N,)` or `(1, N)`; on a grid, each vector is
`(D, cells...)` and each scalar `(cells...)`, the cells flattened in the order the array stores them.
They are packed once, at construction, into the single `(V·D + K, N)` array the kernels already load —
so a pair costs one contiguous read whatever it carries, and the field counts are type parameters,
so the kernel specialises on them.

Which is transported and which is not follows from the name and from nothing else: a vector field
carries a direction the geometry must rotate, a scalar field does not.
"""
struct Fields{D, V, K, A <: AbstractMatrix}
    data::A
end

"""Velocity-like fields, each parallel-transported."""
@inline n_vector_fields(::Fields{D, V, K}) where {D, V, K} = V

"""Tracer-like fields, each differenced without transport."""
@inline n_scalar_fields(::Fields{D, V, K}) where {D, V, K} = K

"""Components in each vector field."""
@inline field_dimension(::Fields{D}) where {D} = D

@inline packed(f::Fields) = f.data

@inline Base.size(f::Fields) = size(f.data)
@inline Base.eltype(::Fields{D, V, K, A}) where {D, V, K, A} = eltype(A)

function Fields(; vectors = (), scalars = ())
    V = length(vectors)
    K = length(scalars)
    V * K >= 0 || throw(ArgumentError("field counts cannot be negative"))
    V + K > 0 || throw(ArgumentError(
        "a field needs at least one field; got no vectors and no scalars",
    ))
    D = V == 0 ? 0 : size(first(vectors), 1)
    for (i, v) in enumerate(vectors)
        ndims(v) >= 2 || throw(ArgumentError(
            "vector field $i must be (D, N) or (D, cells...); got an array of $(ndims(v)) dimensions",
        ))
        size(v, 1) == D || throw(DimensionMismatch(
            "vector field $i has $(size(v, 1)) components, field 1 has $D; every vector " *
            "field is transported by the same geometry and so must have the same dimension",
        ))
    end
    cells = V > 0 ? size(first(vectors))[2:end] : size(first(scalars))
    N = prod(cells)
    for (i, v) in enumerate(vectors)
        size(v)[2:end] == cells || throw(DimensionMismatch(
            "vector field $i covers cells $(size(v)[2:end]), expected $cells",
        ))
    end
    for (i, s) in enumerate(scalars)
        (size(s) == cells || length(s) == N) || throw(DimensionMismatch(
            "scalar field $i covers $(size(s)), expected $cells or $N points",
        ))
    end

    T = promote_type((eltype(v) for v in vectors)..., (eltype(s) for s in scalars)...)
    data = Matrix{T}(undef, V * D + K, N)
    @inbounds for (i, v) in enumerate(vectors)
        vf = reshape(v, D, N)
        for n in 1:N, d in 1:D
            data[(i - 1) * D + d, n] = vf[d, n]
        end
    end
    @inbounds for (i, s) in enumerate(scalars), n in 1:N
        data[V * D + i, n] = s[n]
    end
    return Fields{D, V, K, typeof(data)}(data)
end

"""
    FieldIncrement(vectors, scalars)

One pair's increment across every field: each vector field already transported into the pair's
common frame, each scalar field already differenced.

An operator reads the fields it names. A field of one vector field and no scalars does **not**
produce one of these — its increment is the plain `SVector` every existing operator already takes, so
the single-field path is unchanged down to the instruction.
"""
struct FieldIncrement{D, V, K, T}
    vectors::NTuple{V, SA.SVector{D, T}}
    scalars::NTuple{K, T}
end

"""The `i`-th transported vector increment."""
@inline vector_field(c::FieldIncrement, i::Integer) = @inbounds c.vectors[i]

"""The `i`-th scalar increment."""
@inline scalar_field(c::FieldIncrement, i::Integer) = @inbounds c.scalars[i]

@inline n_vector_fields(::FieldIncrement{D, V, K}) where {D, V, K} = V
@inline n_scalar_fields(::FieldIncrement{D, V, K}) where {D, V, K} = K
@inline field_dimension(::FieldIncrement{D}) where {D} = D

end # module MultiFields
