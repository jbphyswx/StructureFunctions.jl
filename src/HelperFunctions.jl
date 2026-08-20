"""
Generic helper functions for the other routines in this package
"""
module HelperFunctions

using LinearAlgebra: LinearAlgebra as LA
import LinearAlgebra: normalize # import directly ONLY if extending

using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
@inline create_svector(x) = SA.SVector{length(x)}(x) # helper function to create a static vector from a vector without splatting


export digitize,
    δu_longitudinal, δu_l,
    δu_transverse, δu_t,
    magnitude_δu_longitudinal, mδu_l,
    magnitude_δu_transverse, mδu_t,
    transverse_norm2,
    transverse_component_norm2,
    transverse_component,
    transverse_basis,
    transverse_basis_vector,
    AbstractTransverseBasisConvention,
    CanonicalTransverseBasis,
    ReferenceAxisTransverseBasis,
    CoordinateGaugeTransverseBasis,
    UserTransverseBasis,
    r̂,
    n̂,
    δr,
    unit_position,
    local_east_north,
    geodesic_frame,
    geodesic_increments,
    FlatGeometry,
    SphericalGeometry,
    coordinate_width,
    pair_frame,
    pair_direction,
    pair_delta,
    pair_invariants,
    pair_increments,
    pair_geometry,
    pair_geometry_for,
    midpoints,
    flatten_data,
    remove_nans

"""
    flatten_data(vecs::Tuple)

Flatten multi-dimensional arrays in the tuple to vectors. 
Useful for converting grid data to point lists.
"""
function flatten_data(vecs::Tuple)
    return ntuple(i -> vec(vecs[i]), length(vecs))
end

"""
    remove_nans(x_mat::AbstractMatrix, u_mat::AbstractMatrix)

Remove points (columns) where either the position `x_mat` or the velocity `u_mat` 
contains a `NaN`. Returns `(x_mat_clean, u_mat_clean)`.
"""
function remove_nans(x_mat::AbstractMatrix{FT}, u_mat::AbstractMatrix{FT}) where {FT}
    # Find columns that have NO NaNs in either matrix
    nan_in_x = any(isnan, x_mat, dims = 1)
    nan_in_u = any(isnan, u_mat, dims = 1)

    # Combined mask (vec to turn 1xN matrix into N-vector)
    valid_mask = vec(.!(nan_in_x .| nan_in_u))

    return x_mat[:, valid_mask], u_mat[:, valid_mask]
end

@inline function digitize(x, bins::AbstractVector)
    """
    Return the index of the bin that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    Note that the bins are right inclusive, the bins are (a,b]
    """
    searchsortedfirst(bins, x) - 1
end

"""
    midpoints(edges::AbstractVector{<:Number})

Bin midpoints from flat edges `[e₀, e₁, …, eₙ]` (length `n+1` → `n` midpoints).
"""
function midpoints(edges::AbstractVector{T}) where {T}
    n = length(edges) - 1
    return T[(edges[i] + edges[i + 1]) / 2 for i in 1:n]
end

@inline midpoints(edges::SA.SVector{N,T}) where {N,T} = SA.SVector{N-1,T}(ntuple(i -> (edges[i] + edges[i+1]) / 2, N-1))
@inline midpoints(edges::AbstractRange{T}) where {T} = range((first(edges) + last(edges)) / 2, length = length(edges) - 1, step = step(edges))

@inline function digitize(x::AbstractVector, bins::AbstractVector)
    """
    Return the indices of the bins that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    """
    digitize.(x, Ref(bins))
end

## `digitize` dispatches on `AbstractBinEdges` via `searchsortedfirst` overrides in `BinEdges.jl`.



@inline LA.normalize(x::Tuple{T, Vararg{T}}) where {T} =
    NTuple{length(x), T}(LA.normalize(SA.SVector(x)))

"""Return the vector from `x1` to `x2`."""
@inline function δr(x1, x2)
    return x2 .- x1
end

"""Return the longitudinal (parallel) unit vector from `x1` to `x2`."""
@inline function r̂(x1, x2)
    return LA.normalize(δr(x1, x2))
end

@inline r̂(x1, x2, ::DI.Euclidean, distance) = δr(x1, x2) / distance
# NOTE: LA.normalize is fast here because the vector is a StaticArray.SVector.
# If dynamic Vectors are ever used, LA.normalize would be ~2.5x slower due to scaling checks.
@inline r̂(x1, x2, ::DI.PreMetric, distance) = LA.normalize(δr(x1, x2))

# -----------------------------------------------------------------------------
# Spherical geometry: unit positions, geodesic frames, parallel transport
# -----------------------------------------------------------------------------

@inline _unit_position(sλ, cλ, sφ, cφ) = SA.SVector(cφ * cλ, cφ * sλ, sφ)

@inline _local_east_north(sλ, cλ, sφ, cφ) =
    (SA.SVector(-sλ, cλ, zero(sλ)), SA.SVector(-sφ * cλ, -sφ * sλ, cφ))

"""
    unit_position(lon, lat) -> SVector{3}

Unit position vector on the sphere from longitude/latitude **in degrees** (the
`Distances.Haversine` convention). `x` toward (0°N, 0°E), `z` toward the north pole.
"""
@inline unit_position(lon, lat) = _unit_position(sincosd(lon)..., sincosd(lat)...)

"""
    local_east_north(lon, lat) -> (Ê, N̂)

Local east and north unit vectors at longitude/latitude **in degrees**. Together with
[`unit_position`](@ref) as the local up they form a right-handed orthonormal triad
(`Ê × N̂ = p̂`). Only used at ingest: `p̂` itself is well defined at the poles, this basis is not.
"""
@inline local_east_north(lon, lat) = _local_east_north(sincosd(lon)..., sincosd(lat)...)

"""
Smallest `sin²σ` for which the separation direction is still representable.

This guards only against `1/0`, NOT against a physical scale: the normalization is exact in the
`σ → 0` limit (`t_A ≈ d` has magnitude `O(σ)` and `inv_s ≈ 1/σ`, so the product stays `O(1)`), so
short pairs are fine and must not be dropped. `eps(T)` would be catastrophically wrong here —
`sin²σ` for a 1 km separation on Earth is `2.5e-8`, below `eps(Float32)`, which would silently
discard every pair closer than ~2 km in Float32.
"""
@inline _geodesic_degeneracy_tol(::Type{T}) where {T} = floatmin(T)

"""
    geodesic_frame(p̂, q̂) -> (σ, t̂_A, t̂_B, m̂, ok)

Great-circle frame for the pair `(p̂, q̂)` of unit position vectors: central angle `σ`, the
longitudinal unit tangent at each endpoint (`t̂_A` at `p̂` toward `q̂`, `t̂_B` at `q̂` continuing along
the geodesic), and the shared transverse unit vector `m̂`.

`m̂` is the same 3-vector at both endpoints: the great-circle normal is perpendicular to every point
of that circle, hence tangent to the sphere at each of them, and it is parallel along the geodesic.
Both tangents also share one normalizer, since `‖q̂ − (p̂·q̂)p̂‖² = ‖(p̂·q̂)q̂ − p̂‖² = 1 − (p̂·q̂)²`.
So the whole frame costs one `sqrt`.

`σ` uses the tangent-half-angle form `2·atan(‖p̂−q̂‖, ‖p̂+q̂‖)`, which is accurate for every `σ`
including antipodal, unlike `acos(p̂·q̂)` (which loses half the mantissa near `σ=0` — fatal in Float32).

`ok` is `false` for coincident (`σ=0`) and antipodal (`σ=π`) pairs, where the direction is genuinely
undefined — antipodal points are joined by infinitely many great circles, so parallel transport
between them is not unique. Callers must skip such pairs: `1/s` is otherwise `Inf` and a single NaN
lane poisons an entire atomic accumulator.
"""
@inline function geodesic_frame(p̂::SA.SVector{3, T}, q̂::SA.SVector{3, T}) where {T}
    # Work through d = q̂ - p̂ (magnitude O(σ)): forming `q̂ - (p̂·q̂)p̂` directly cancels
    # catastrophically as σ → 0 because both terms approach p̂.
    d = q̂ - p̂
    sum_pq = p̂ + q̂
    dp = LA.dot(d, p̂)               # = p̂·q̂ - 1 = -2sin²(σ/2), small and accurate
    t_A = d - dp * p̂                # ≡ q̂ - (p̂·q̂)p̂
    t_B = dp * q̂ + d                # ≡ (p̂·q̂)q̂ - p̂
    w = LA.cross(p̂, d)              # ≡ p̂ × q̂
    s2 = LA.dot(w, w)               # = sin²σ
    σ = 2 * atan(sqrt(LA.dot(d, d)), sqrt(LA.dot(sum_pq, sum_pq)))
    ok = s2 > _geodesic_degeneracy_tol(T)
    inv_s = ok ? inv(sqrt(s2)) : zero(T)
    return σ, t_A * inv_s, t_B * inv_s, w * inv_s, ok
end

"""
    FlatGeometry{D}()

Flat `D`-dimensional space with the Euclidean ruler: the separation is the straight chord and no
transport is needed.
"""
struct FlatGeometry{D} end

"""
    SphericalGeometry{D}(metric, radius)

Sphere of the given `radius`. `D` is the velocity dimension: `2` for horizontal `(east, north)`
velocities, `3` for a thin shell carrying an additional radial component.

`D` cannot be inferred from the coordinates — a point on a shell has two of them either way — and it
changes the answer, because `transverse_component_norm2` divides the transverse energy by `D - 1`.

`metric` is retained because the coordinates mean whatever it says they mean: it is what fixes their
angle unit, which `pair_frame` needs in order to take a `sincos`. See [`unit_position`](@ref).
"""
struct SphericalGeometry{D, M, T}
    metric::M
    radius::T
end
SphericalGeometry{D}(metric::M, radius::T) where {D, M, T} = SphericalGeometry{D, M, T}(metric, radius)

"""
    coordinate_width(geometry) -> Val{W}

How many numbers locate one point in this geometry, as a `Val` so callers can build a
statically-sized load. This is **not** the velocity dimension: on a shell a point takes two
coordinates whether or not the velocity carries a third, radial, component.
"""
@inline coordinate_width(::FlatGeometry{D}) where {D} = Val(D)
@inline coordinate_width(::SphericalGeometry) = Val(2)

"""
    pair_frame(geometry, x1, x2) -> (ok, r, frame)

Separation and the geometry-specific frame data for one pair, touching **no velocities**.

Split from [`pair_increments`](@ref) because callers depend on that order: the scalar kernels reject
out-of-range pairs before loading `u_j` at all, and the batch kernels build one frame per `(i, j)`
and reuse it across a whole strip of velocity fields. A single call taking the velocities would
defeat both.
"""
@inline function pair_frame(::FlatGeometry, x1, x2)
    dx = δr(x1, x2)
    # Carry the RAW displacement, not `dx / r`: normalizing here would make every out-of-range pair
    # pay a divide and D multiplies for a direction the bin test is about to discard.
    return true, sqrt(LA.dot(dx, dx)), dx
end

"""
Spherical frame from raw `(lon, lat)` coordinates, expressed in each endpoint's own local
`(east, north)` basis.

The ambient form of [`geodesic_frame`](@ref) needs 3-vectors, which would force the caller's `(2, N)`
lon/lat array to be widened to `(3, N)`. Projecting the geodesic tangents onto the local bases
instead keeps everything the width the caller passed — the longitudinal direction at `A` is just
`(sin α₁, cos α₁)` for the forward azimuth `α₁`, and the transverse is its right-handed quarter turn.
Parallel transport is still exact: it preserves the angle to the geodesic, which is what `α` measures.
"""
@inline function pair_frame(g::SphericalGeometry, x1, x2)
    p̂ = unit_position(g.metric, x1[1], x1[2])
    q̂ = unit_position(g.metric, x2[1], x2[2])
    σ, t_A, t_B, m̂, ok = geodesic_frame(p̂, q̂)
    Ê_A, N̂_A = local_east_north(g.metric, x1[1], x1[2])
    Ê_B, N̂_B = local_east_north(g.metric, x2[1], x2[2])
    FT = typeof(σ)
    # Local components of the longitudinal tangent at each end; n̂ = ẑ × t̂ within the tangent plane.
    tA = SA.SVector{2, FT}(LA.dot(t_A, Ê_A), LA.dot(t_A, N̂_A))
    tB = SA.SVector{2, FT}(LA.dot(t_B, Ê_B), LA.dot(t_B, N̂_B))
    return ok, g.radius * σ, (tA, tB, p̂, q̂, m̂, Ê_A, N̂_A, Ê_B, N̂_B)
end

"""
    pair_direction(geometry, frame, r) -> r̂

Longitudinal unit vector for one pair. Depends only on the geometry of the pair, never on the
velocities, so a caller sweeping many velocity fields over one `(i, j)` hoists this out of that loop.
"""
@inline pair_direction(::FlatGeometry, frame, r) = frame / r

"""
On the sphere the frame IS the basis, so the longitudinal direction is `ê₁` by construction.
"""
@inline pair_direction(::SphericalGeometry{D}, frame, r) where {D} =
    SA.SVector{D}(ntuple(i -> i == 1 ? one(eltype(frame[1])) : zero(eltype(frame[1])), Val(D)))

"""
    pair_invariants(geometry, frame, r, u1, u2) -> (δu_L, ‖δu‖²)

The only two scalars the six isotropic invariants consume: every one of them is built from `δu_L`,
`δu_L²` and `δu_T² = ‖δu‖² − δu_L²`. Kernels that need no separation direction call this instead of
[`pair_increments`](@ref) and never form `r̂`.
"""
@inline function pair_invariants(::FlatGeometry, frame, r, u1, u2)
    δu = u2 - u1
    # `dot(δu, frame) / r`, not `dot(δu, frame / r)`: one rounding, and the same operation order the
    # flat kernels have always used.
    return LA.dot(δu, frame) / r, LA.dot(δu, δu)
end

@inline function pair_invariants(g::SphericalGeometry, frame, r, u1, u2)
    δu = pair_delta(g, frame, nothing, nothing, u1, u2)
    return δu[1], LA.dot(δu, δu)
end

"""
    pair_delta(geometry, frame, x1, x2, u1, u2) -> δu

Velocity difference expressed in the pair's common frame, given the `frame` from
[`pair_frame`](@ref). The per-velocity-field half of [`pair_increments`](@ref).
"""
@inline pair_delta(::FlatGeometry, frame, x1, x2, u1, u2) = u2 - u1

"""
Increments for `u = (east, north)` on the sphere. Each endpoint's velocity is projected onto its own
geodesic frame and only then differenced — that IS the parallel transport, because transport along a
geodesic preserves the angle to it. `n̂ = ẑ × t̂ = (−t̂₂, t̂₁)` matches the package handedness.
"""
@inline function pair_delta(::SphericalGeometry{2}, frame, x1, x2, u_A, u_B)
    tA, tB = frame[1], frame[2]
    nA = SA.SVector(-tA[2], tA[1])
    nB = SA.SVector(-tB[2], tB[1])
    δu_L = LA.dot(u_B, tB) - LA.dot(u_A, tA)
    δu_T = LA.dot(u_B, nB) - LA.dot(u_A, nA)
    return SA.SVector{2, typeof(δu_L)}(δu_L, δu_T)
end

"""
Thin shell, `u = (east, north, up)`. Horizontal components transport exactly as in 2D; the radial
component is a scalar difference needing no transport, since the geodesic frame is tangent to the
sphere and therefore orthogonal to the radial direction at both endpoints.
"""
@inline function pair_delta(::SphericalGeometry{3}, frame, x1, x2, u_A, u_B)
    tA, tB = frame[1], frame[2]
    nA = SA.SVector(-tA[2], tA[1])
    nB = SA.SVector(-tB[2], tB[1])
    hA = SA.SVector(u_A[1], u_A[2])
    hB = SA.SVector(u_B[1], u_B[2])
    δu_L = LA.dot(hB, tB) - LA.dot(hA, tA)
    δu_T = LA.dot(hB, nB) - LA.dot(hA, nA)
    δw = u_B[3] - u_A[3]
    return SA.SVector{3, typeof(δu_L)}(δu_L, δu_T, δw)
end

"""
    pair_increments(geometry, frame, r, x1, x2, u1, u2) -> (δu, r̂)

Velocity difference and longitudinal unit vector in one common frame, given the `frame` and
separation `r` from [`pair_frame`](@ref). See [`pair_geometry`](@ref) for what the result means per
geometry. Called only for pairs that survive the bin test, so this is where the direction is formed.
"""
@inline pair_increments(g, frame, r, x1, x2, u1, u2) =
    (pair_delta(g, frame, x1, x2, u1, u2), pair_direction(g, frame, r))

"""
    pair_geometry(geometry, x1, x2, u1, u2) -> (ok, r, δu, r̂)

Resolve one pair into a separation `r`, a velocity difference `δu`, and a longitudinal unit vector
`r̂`, **all in one common frame**, so that every structure-function operator — which consumes only
`δu·r̂` and `δu·n̂(r̂)` — works unchanged for any geometry. `ok == false` marks a pair with no defined
separation direction; callers must skip it.

For [`SphericalGeometry`](@ref), `x1`/`x2` are `(lon, lat)` in the metric's angle unit and `u1`/`u2`
are `(east, north[, up])`. The result is expressed in the local geodesic frame, so `r̂ = ê₁` and
`δu = (δu_L, δu_T)` (plus the radial difference for `D = 3`). The radial component needs no transport
and cannot leak into `δu_L`/`δu_T`, because `t̂ ⟂ p̂` and `m̂ ⟂ p̂, q̂`.

This is the one-shot convenience form; hot loops use [`pair_frame`](@ref) and
[`pair_increments`](@ref) separately so the frame can gate the bin test and be reused across fields.
"""
@inline function pair_geometry(g, x1, x2, u1, u2)
    ok, r, frame = pair_frame(g, x1, x2)
    δu, rhat = pair_increments(g, frame, r, x1, x2, u1, u2)
    return ok, r, δu, rhat
end

"""
    pair_geometry_for(metric, ::Val{D}) -> geometry

Geometry implied by `metric` for user-facing dimension `D`. A distance function alone does not define
a direction or a transport rule, so there is deliberately **no generic method**: an unrecognized
metric raises rather than silently assuming flat space. Add a method here (and a
[`pair_geometry`](@ref) method for the geometry it returns) to support another manifold.
"""
pair_geometry_for(::DI.Euclidean, ::Val{D}) where {D} = FlatGeometry{D}()
pair_geometry_for(m::DI.Haversine, ::Val{D}) where {D} = SphericalGeometry{D}(m, m.radius)
# SphericalAngle reports the central angle itself, i.e. a unit sphere.
pair_geometry_for(m::DI.SphericalAngle, ::Val{D}) where {D} = SphericalGeometry{D}(m, 1)

"""
    unit_position(metric, lon, lat) -> SVector{3}
    local_east_north(metric, lon, lat) -> (Ê, N̂)

Ingest helpers that take the angle unit from the metric's own documented convention:
`Distances.Haversine` is **degrees**, `Distances.SphericalAngle` is **radians**. Confusing the two
silently rescales every separation by a factor of ~57, so the convention is pinned next to the metric
that defines it rather than repeated at each call site.
"""
@inline unit_position(::DI.Haversine, lon, lat) = unit_position(lon, lat)
@inline local_east_north(::DI.Haversine, lon, lat) = local_east_north(lon, lat)
@inline unit_position(::DI.SphericalAngle, lon, lat) =
    _unit_position(sincos(lon)..., sincos(lat)...)
@inline local_east_north(::DI.SphericalAngle, lon, lat) =
    _local_east_north(sincos(lon)..., sincos(lat)...)

pair_geometry_for(m, ::Val{D}) where {D} = throw(ArgumentError(
    "no pair geometry is defined for distance_metric=$(typeof(m)). A distance function fixes which " *
    "histogram bin a pair falls in, but it does not define the separation direction or how to " *
    "compare velocities at two different points, so this package will not guess one. Define " *
    "`StructureFunctions.HelperFunctions.pair_geometry_for(::$(typeof(m)), ::Val)` returning a " *
    "geometry, plus a `pair_geometry` method for it.",
))

"""
    geodesic_increments(t̂_A, t̂_B, m̂, u_A, u_B) -> (δu_L, δu_T)

Longitudinal and transverse components of the parallel-transported velocity difference, given the
frame from [`geodesic_frame`](@ref) and the two ambient tangent velocities.

Projecting each endpoint's velocity onto its own geodesic tangent **is** parallel transport, not an
approximation: a geodesic parallel-transports its own tangent, and transport on the sphere is an
orientation-preserving isometry, so the transported basis from `A` arrives at `B` rotated by exactly
the difference of forward azimuths. The transverse term needs a single dot product because `m̂` is
shared between the endpoints.
"""
@inline function geodesic_increments(t̂_A, t̂_B, m̂, u_A, u_B)
    δu_L = LA.dot(u_B, t̂_B) - LA.dot(u_A, t̂_A)
    δu_T = LA.dot(u_B - u_A, m̂)
    return δu_L, δu_T
end


"""
    n̂(r_hat)

Oriented transverse unit vector for the longitudinal unit vector `r_hat`.

2D: `n̂ = ẑ × r̂ = (−r̂₂, r̂₁)`, the counterclockwise quarter turn, so `(r̂, n̂, ẑ)` is right-handed.
3D: `n̂ = normalize(ẑ × r̂)`, the same rule with `ẑ = (0,0,1)` as the reference axis.

Only operators odd in the transverse component see this sign — `ProjectedStructureFunctionType{2,1}`
and `{0,3}`. Everything else consumes `δu_T²` (see [`transverse_norm2`](@ref)) and is sign-blind.
The 3D form is singular when `r̂ ∥ ẑ`; [`ReferenceAxisTransverseBasis`](@ref) is the guarded version.
"""
@inline function n̂(r_hat::AbstractVector{FT}) where {FT}
    ND::Int = length(r_hat)

    if ND == 2
        return SA.SVector{2, FT}(-r_hat[2], r_hat[1]) # assume normalized
    elseif ND == 3
        k_hat = SA.SVector{3, FT}(0, 0, 1)
        return LA.normalize(
            LA.cross(k_hat, SA.SVector{3, FT}(r_hat[1], r_hat[2], r_hat[3])),
        )
    else
        error("Only 2D and 3D supported")
    end
end

@inline n̂(r_hat::SA.SVector{2, T}) where {T} = SA.SVector{2, T}(-r_hat[2], r_hat[1])
@inline n̂(r_hat::SA.SVector{3, T}) where {T} = LA.normalize(
    LA.cross(SA.SVector{3, T}(0, 0, 1), SA.SVector{3, T}(r_hat[1], r_hat[2], r_hat[3])),
)


@inline n̂(r_hat::NTuple{2, T}) where {T} = (-r_hat[2], r_hat[1])


"""
Return the transverse (perpendicular) unit vector from `x1` to `x2`, defined as the cross product
of the local vertical with the longitudinal unit vector. This naming is opposite to the Lindborg
and Cho convention.
"""
@inline function n̂(x1, x2)
    return n̂(r̂(x1, x2))
end

abstract type AbstractTransverseBasisConvention end

"""
    CanonicalTransverseBasis()

The canonical oriented transverse basis in 2D, equivalent to [`n̂`](@ref).
It is intentionally undefined for 3D because there is no unique oriented
transverse direction without extra information.
"""
struct CanonicalTransverseBasis <: AbstractTransverseBasisConvention end

"""
    ReferenceAxisTransverseBasis(axis; parallel_tol=1e-12)

Project a physical reference axis into the plane perpendicular to `r̂` and
normalize it to get the first transverse basis vector. In 3D, the second
transverse vector is `cross(r̂, e₁)`.

If the reference axis is parallel or nearly parallel to `r̂`, construction of
the per-pair basis throws `ArgumentError`. The tolerance is explicit because
this is a physical convention, not a hidden computational replacement.
"""
struct ReferenceAxisTransverseBasis{A, T} <: AbstractTransverseBasisConvention
    axis::A
    parallel_tol::T
end

ReferenceAxisTransverseBasis(axis; parallel_tol = 1e-12) =
    ReferenceAxisTransverseBasis(axis, parallel_tol)

"""
    CoordinateGaugeTransverseBasis()

An always-defined computational gauge for 3D: choose the coordinate axis least
aligned with `r̂`, project it into the perpendicular plane, and complete a
right-handed basis. This is useful for deterministic component diagnostics,
but the chosen direction can jump discontinuously as `r̂` changes.
"""
struct CoordinateGaugeTransverseBasis <: AbstractTransverseBasisConvention end

"""
    UserTransverseBasis(f)

Use `f(r̂)` as the transverse basis provider. The function must return either
a single unit vector or a tuple of unit vectors perpendicular to `r̂`.
"""
struct UserTransverseBasis{F} <: AbstractTransverseBasisConvention
    basis_function::F
end

@inline function _sum_abs2(x)
    out = zero(eltype(x))
    @inbounds for i in eachindex(x)
        out += x[i] * x[i]
    end
    return out
end

@inline function _as_svector_like(r_hat, x)
    return SA.SVector{length(r_hat), eltype(r_hat)}(ntuple(i -> x[i], length(r_hat)))
end

@inline function _project_reference_axis(axis, r_hat, parallel_tol)
    a = _as_svector_like(r_hat, axis)
    projected = a - LA.dot(a, r_hat) * r_hat
    projected_norm2 = _sum_abs2(projected)
    tol2 = parallel_tol * parallel_tol
    projected_norm2 > tol2 || throw(
        ArgumentError(
            "reference axis is parallel or nearly parallel to r̂; " *
            "choose a different axis or use CoordinateGaugeTransverseBasis()",
        ),
    )
    return projected / sqrt(projected_norm2)
end

@inline function transverse_basis(::CanonicalTransverseBasis, r_hat::SA.SVector{2})
    return (n̂(r_hat),)
end

@inline function transverse_basis(::CanonicalTransverseBasis, r_hat)
    length(r_hat) == 2 || throw(ArgumentError("CanonicalTransverseBasis is only defined in 2D"))
    return (n̂(r_hat),)
end

@inline function transverse_basis(basis::ReferenceAxisTransverseBasis, r_hat::SA.SVector{2})
    e1 = _project_reference_axis(basis.axis, r_hat, basis.parallel_tol)
    return (e1,)
end

@inline function transverse_basis(basis::ReferenceAxisTransverseBasis, r_hat::SA.SVector{3})
    e1 = _project_reference_axis(basis.axis, r_hat, basis.parallel_tol)
    e2 = LA.cross(r_hat, e1)
    return (e1, e2)
end

@inline function transverse_basis(basis::ReferenceAxisTransverseBasis, r_hat)
    D = length(r_hat)
    D == 2 && return transverse_basis(basis, SA.SVector{2, eltype(r_hat)}(r_hat))
    D == 3 && return transverse_basis(basis, SA.SVector{3, eltype(r_hat)}(r_hat))
    throw(ArgumentError("ReferenceAxisTransverseBasis currently supports D=2 or D=3"))
end

@inline function transverse_basis(::CoordinateGaugeTransverseBasis, r_hat::SA.SVector{3, T}) where {T}
    ax = abs(r_hat[1])
    ay = abs(r_hat[2])
    az = abs(r_hat[3])
    axis = ax <= ay && ax <= az ? SA.SVector{3, T}(1, 0, 0) :
           ay <= az ? SA.SVector{3, T}(0, 1, 0) :
           SA.SVector{3, T}(0, 0, 1)
    e1 = axis - LA.dot(axis, r_hat) * r_hat
    e1 = e1 / sqrt(_sum_abs2(e1))
    e2 = LA.cross(r_hat, e1)
    return (e1, e2)
end

@inline function transverse_basis(::CoordinateGaugeTransverseBasis, r_hat::SA.SVector{2})
    return (n̂(r_hat),)
end

@inline function transverse_basis(basis::CoordinateGaugeTransverseBasis, r_hat)
    D = length(r_hat)
    D == 2 && return transverse_basis(basis, SA.SVector{2, eltype(r_hat)}(r_hat))
    D == 3 && return transverse_basis(basis, SA.SVector{3, eltype(r_hat)}(r_hat))
    throw(ArgumentError("CoordinateGaugeTransverseBasis currently supports D=2 or D=3"))
end

@inline transverse_basis(basis::UserTransverseBasis, r_hat) = basis.basis_function(r_hat)

@inline function transverse_basis_vector(r_hat, basis::AbstractTransverseBasisConvention, basis_index::Integer = 1)
    basis_vectors = transverse_basis(basis, r_hat)
    1 <= basis_index <= length(basis_vectors) ||
        throw(BoundsError(basis_vectors, basis_index))
    return basis_vectors[basis_index]
end


"""
Return the signed longitudinal magnitude of `δu` along `r_hat`. The caller must ensure `r_hat` is
a unit vector.
"""
@inline function magnitude_δu_longitudinal(δu, r_hat)
    return LA.dot(δu, r_hat) # r_hat is unit vector so just dot product
end

"""
Return the longitudinal component of `δu` along `r_hat`, as a vector. The caller must ensure
`r_hat` is a unit vector.
"""
@inline function δu_longitudinal(δu, r_hat)
    return magnitude_δu_longitudinal(δu, r_hat) * r_hat
end

"""
Return the transverse magnitude of `δu`, signed relative to the normal vector `n̂(r_hat)`. The
caller must ensure `r_hat` is a unit vector.
"""
@inline function magnitude_δu_transverse(δu, r_hat)
    # Signed relative to n̂, unlike LA.norm(δu .- δu_longitudinal(δu, r_hat)).
    return LA.dot(δu, n̂(r_hat))
end

@inline function magnitude_δu_transverse(
    δu,
    r_hat,
    basis::AbstractTransverseBasisConvention,
    basis_index::Integer = 1,
)
    return transverse_component(δu, r_hat, basis, basis_index)
end

"""
Return the transverse component of `δu` (perpendicular to `r_hat`), as a vector. The caller must
ensure `r_hat` is a unit vector.
"""
@inline function δu_transverse(δu, r_hat)
    return δu .- δu_longitudinal(δu, r_hat)
end

"""
    transverse_norm2(δu, r̂)

Squared norm of the full transverse vector ``δu - (δu⋅r̂)r̂``. This is the
invariant transverse energy used by `T2SF` and `L1T2SF`.
"""
@inline function transverse_norm2(δu, r_hat)
    du_l = magnitude_δu_longitudinal(δu, r_hat)
    return _sum_abs2(δu) - du_l * du_l
end

"""
    transverse_component_norm2(δu, r̂)

Per-component transverse energy under isotropic transverse averaging,
``||δu_t||² / (D - 1)``.
"""
@inline function transverse_component_norm2(δu, r_hat)
    D = length(r_hat)
    D > 1 || throw(ArgumentError("transverse_component_norm2 requires D >= 2"))
    return transverse_norm2(δu, r_hat) / (D - 1)
end

"""
    transverse_component(δu, r̂, basis, basis_index=1)

Component of `δu` along an explicit transverse basis vector. This is the
quantity to use for basis-dependent odd/component diagnostics such as `T3SF`
or `L2T1SF`.
"""
@inline function transverse_component(
    δu,
    r_hat,
    basis::AbstractTransverseBasisConvention,
    basis_index::Integer = 1,
)
    e = transverse_basis_vector(r_hat, basis, basis_index)
    return LA.dot(δu, e)
end


const δu_l = δu_longitudinal
const δu_t = δu_transverse

const mδu_l = magnitude_δu_longitudinal
const mδu_t = magnitude_δu_transverse






end # end Module
