"""
    ZonalLagSchedule(lats, n_lon, dlon, radius, lon_periodic)

Pair enumeration for a lat-lon grid: one latitude per entry of `lats` (radians), `n_lon` cells per
row spaced `dlon` apart in longitude (radians), on a sphere of the given `radius`.

The field it sweeps is stored `(component, longitude, latitude)` with components in the local
`(east, north[, radial])` basis, the basis the transport matrices are written in. Each latitude row is
one slab and longitude is the uniform direction, with the pair frame applied per lag (see
[`FrameTransport`](@ref)).
"""
struct ZonalLagSchedule{T, LV <: AbstractVector{T}} <: AbstractSeparableSchedule
    lats::LV
    n_lon::Int
    dlon::T
    radius::T
    lon_periodic::Bool
end

"""Cells the schedule covers."""
@inline n_zonal_cells(s::ZonalLagSchedule) = s.n_lon * length(s.lats)
@inline n_cells(s::ZonalLagSchedule) = n_zonal_cells(s)
@inline grid_dimension(::ZonalLagSchedule) = 2

@inline uniform_axes(s::ZonalLagSchedule) = UniformLagSchedule((s.n_lon,), (s.dlon,), (s.lon_periodic,))
@inline n_slabs(s::ZonalLagSchedule) = length(s.lats)
@inline separable_layout(::ZonalLagSchedule, data, valid, weights) = (data, valid, weights)
@inline lag_transport(::ZonalLagSchedule) = FrameTransport()

# Two rows are never closer than their latitude difference, so a row pair beyond `r_max` contributes
# nothing at any longitude offset.
function enumerated_pairs(s::ZonalLagSchedule, r_max)
    n = length(s.lats)
    return ((I, J) for I in 1:n for J in I:n if s.radius * abs(s.lats[J] - s.lats[I]) <= r_max)
end

# The separation of two points at latitudes φ₁, φ₂ grows with their longitude difference up to a half
# turn, by hav σ = hav Δφ + cos φ₁ cos φ₂ hav Δλ, so the offsets within `r_max` are those up to the
# Δλ that solves it; one is added so round-off at the boundary can only keep a lag the bin test drops.
function _zonal_lag_limit(s::ZonalLagSchedule{T}, phi1, phi2, r_max) where {T}
    isfinite(r_max) || return typemax(Int)
    σ_max = r_max / s.radius
    σ_max >= π && return typemax(Int)
    cc = cos(phi1) * cos(phi2)
    cc > 0 || return typemax(Int)
    rhs = (sin(σ_max / 2)^2 - sin((phi2 - phi1) / 2)^2) / cc
    rhs >= 1 && return typemax(Int)
    rhs <= 0 && return 1
    return floor(Int, 2 * asin(sqrt(rhs)) / abs(s.dlon)) + 1
end

@inline lag_limits(s::ZonalLagSchedule, I, J, r_max) = (_zonal_lag_limit(s, s.lats[I], s.lats[J], r_max),)

# One parallel spans less distance the nearer it lies to a pole, so a row pair's longitude limit
# depends on the rows.
uniform_lag_box(::ZonalLagSchedule) = false

# Every cross-row offset within `r_max` is within the larger of the two rows' own limits.
function lag_limits(s::ZonalLagSchedule, r_max)
    lim = 0
    for I in eachindex(s.lats)
        lim = max(lim, _zonal_lag_limit(s, s.lats[I], s.lats[I], r_max))
    end
    return (lim,)
end

# Ambient position and local east/north at longitude `lam`, latitude `phi`.
@inline function _zonal_basis(lam::T, phi::T) where {T}
    sλ, cλ = sincos(lam)
    sφ, cφ = sincos(phi)
    return (SA.SVector{3, T}(cφ * cλ, cφ * sλ, sφ),
            SA.SVector{3, T}(-sλ, cλ, zero(T)),
            SA.SVector{3, T}(-sφ * cλ, -sφ * sλ, cφ))
end

"""
    zonal_transport(geometry, φ₁, φ₂, Δλ, ::Val{D}) -> (ok, r, A, B)

Separation and the two transport matrices shared by every pair at latitudes `φ₁`, `φ₂` separated by
`Δλ` in longitude: `δu = B·u_B − A·u_A`, with each velocity in its own local basis.

The geodesic frame written in the endpoints' local bases does not depend on absolute longitude, so
one computation, evaluated at longitude zero, serves the whole circle of such pairs.

Row 1 of each matrix is the longitudinal projection and row 2 the transverse, matching
`geodesic_increments`; a third component is radial, which needs no transport and differences as a
scalar.
"""
@inline function zonal_transport(
    g::SFH.SphericalGeometry, phi1::T, phi2::T, dlam::T, ::Val{D},
) where {T, D}
    pA, EA, NA = _zonal_basis(zero(T), phi1)
    pB, EB, NB = _zonal_basis(dlam, phi2)
    ok, r, frame = SFH.pair_frame(g, pA, pB)
    t_A, t_B, m̂ = frame[1], frame[2], frame[3]
    return ok, r, _transport_rows(t_A, m̂, EA, NA, Val(D)), _transport_rows(t_B, m̂, EB, NB, Val(D))
end

# Row 1: the longitudinal tangent in the local basis; row 2: the transverse; row 3: the radial identity.
@inline function _transport_rows(t, m̂, E, N, ::Val{D}) where {D}
    T = eltype(t)
    return SA.SMatrix{D, D, T}(ntuple(Val(D * D)) do lin
        a = (lin - 1) % D + 1
        c = (lin - 1) ÷ D + 1
        a == 1 ? (c == 1 ? SFH.fma_dot(t, E) : c == 2 ? SFH.fma_dot(t, N) : zero(T)) :
        a == 2 ? (c == 1 ? SFH.fma_dot(m̂, E) : c == 2 ? SFH.fma_dot(m̂, N) : zero(T)) :
        (c == 3 ? one(T) : zero(T))
    end)
end

"""
    zonal_transport(geometry, φ₁, φ₂, Δλ, ::Val{D}, ::Val{V}, ::Val{K}) -> (ok, r, A, B)

The transport of a packed field: the `D×D` block on each of the `V` vector fields and the identity
on the `K` scalar fields, as `W×W` matrices with `W = V·D + K`. A field with no vector field
carries only the separation.
"""
@inline function zonal_transport(
    g::SFH.SphericalGeometry, phi1::T, phi2::T, dlam::T, ::Val{D}, ::Val{V}, ::Val{K},
) where {T, D, V, K}
    if V == 0
        ok, r, _, _ = zonal_transport(g, phi1, phi2, dlam, Val(2))
        I = SA.SMatrix{K, K, T}(LA.I)
        return ok, r, I, I
    end
    ok, r, A, B = zonal_transport(g, phi1, phi2, dlam, Val(D))
    return ok, r, _block_diagonal(A, Val(V), Val(K)), _block_diagonal(B, Val(V), Val(K))
end

@inline function _block_diagonal(A::SA.SMatrix{D, D, T}, ::Val{V}, ::Val{K}) where {D, T, V, K}
    W = V * D + K
    return SA.SMatrix{W, W, T}(ntuple(Val(W * W)) do lin
        i = (lin - 1) % W + 1
        j = (lin - 1) ÷ W + 1
        if i <= V * D && j <= V * D && (i - 1) ÷ D == (j - 1) ÷ D
            @inbounds A[(i - 1) % D + 1, (j - 1) % D + 1]
        else
            T(i == j)
        end
    end)
end

@inline _zonal_geometry(s::ZonalLagSchedule, ::Val{D}, ::Val{V}) where {D, V} =
    SFH.SphericalGeometry{V == 0 ? 2 : D}(SFH.SphericalDistance(s.radius), s.radius)

# On the sphere the pair frame IS the basis, so the longitudinal direction is ê₁ for every pair. A
# half-turn offset joins the two ends by two minimal paths of opposite sign, so the operator is
# averaged over both frames. Read south to north; along one parallel west to east, where a half turn
# leaves neither end first.
@inline function lag_frame(
    s::ZonalLagSchedule{T}, I, J, h::NTuple{1, Int}, ::Val{D}, ::Val{V}, ::Val{K},
) where {T, D, V, K}
    m = h[1]
    dlam = T(m) * s.dlon
    geom = _zonal_geometry(s, Val(D), Val(V))
    phi1, phi2 = T(@inbounds s.lats[I]), T(@inbounds s.lats[J])
    ok, r, A, B = zonal_transport(geom, phi1, phi2, dlam, Val(D), Val(V), Val(K))
    two = s.lon_periodic && iseven(s.n_lon) && abs(m) == s.n_lon ÷ 2
    A2, B2 = A, B
    if two
        _, _, A2, B2 = zonal_transport(geom, phi1, phi2, -dlam, Val(D), Val(V), Val(K))
    end
    north = phi2 - phi1
    factor = !iszero(north) ? (north > 0 ? 1 : -1) : (two ? 0 : (dlam > 0 ? 1 : -1))
    Dr = _direction_width(Val(D), Val(V), Val(2))
    dir = _unit_east(Dr, T)
    return ok, r * r, factor, (dir = dir, A = A, B = B, A2 = A2, B2 = B2, two = two)
end

@inline _unit_east(::Val{Dr}, ::Type{T}) where {Dr, T} =
    SA.SVector{Dr, T}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(Dr)))

"""
    ScatteredPairs(points, metric)

Pair enumeration for a grid with no structure to exploit: the points themselves, and the metric that
measures between them. Pairs are enumerated by the unstructured path, with culling.
"""
struct ScatteredPairs{X <: AbstractMatrix, M}
    points::X
    metric::M
end

"""Cells the schedule covers."""
@inline n_scattered_cells(s::ScatteredPairs) = size(s.points, 2)
@inline n_cells(s::ScatteredPairs) = size(s.points, 2)
@inline grid_dimension(s::ScatteredPairs) = size(s.points, 1)

"""
    gridded_lag_sweep!(sums, counts, sf, u, schedule::ScatteredPairs, distance_bins, ::Val{D}; valid, backend)

Accumulate every pair of a structureless grid, by enumerating them.

Cells holding nothing are dropped before the sweep.
"""
function gridded_lag_sweep!(
    sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::ScatteredPairs, dist_be, ::Val{D}, ::Val{V}, ::Val{K};
    valid = AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
) where {D, V, K}
    _require_backend(backend)
    n = n_scattered_cells(s)
    size(data) == (V * D + K, n) || throw(DimensionMismatch(
        "field holds $(size(data, 2)) cells of $(size(data, 1)) components, the grid $n cells of " *
        "$(V * D + K)",
    ))
    w = _pair_weights(weights, n, eltype(sums))
    keep = valid isa AllValid ? Colon() : findall(valid)
    x = valid isa AllValid ? s.points : s.points[:, keep]
    uu = valid isa AllValid ? data : data[:, keep]
    ww = w isa NoWeights ? w : w[keep]
    _assert_counts_can_accumulate(counts, size(x, 2), ww)
    if V == 1 && K == 0
        _validate_array_shape(x, uu, s.metric)
        _dispatch_execution_backend!(backend, PointField{D}(), sums, counts, sf, x, uu, dist_be;
                                     geometry = SFH.pair_geometry_for(s.metric, Val(D)), weights = ww)
    else
        f = MF.Fields{D, V, K, typeof(uu)}(uu)
        validate_fields(sf, f)
        _with_field_geometry(f, x, s.metric) do geometry
            _field_dispatch!(backend, sums, counts, sf, x, f, dist_be; geometry, weights = ww)
        end
    end
    return sums, counts
end

# A structureless grid's single-pass invariants are the point entry's over its held cells.
function gridded_lag_sweep!(
    sums::AbstractMatrix, counts::AbstractMatrix, sf::SFT.SinglePassInvariants,
    data::AbstractMatrix, s::ScatteredPairs, dist_be, ::Val{D}, ::Val{V}, ::Val{K};
    valid = AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
) where {D, V, K}
    _require_backend(backend)
    validate_fields(sf, Val(V), Val(K))
    n = n_scattered_cells(s)
    size(data) == (D, n) || throw(DimensionMismatch(
        "field holds $(size(data, 2)) cells of $(size(data, 1)) components, the grid $n cells of $D",
    ))
    _check_hist_shape(sums, counts, sf, n_histogram_bins(dist_be))
    w = _pair_weights(weights, n, eltype(sums))
    keep = valid isa AllValid ? Colon() : findall(valid)
    x = valid isa AllValid ? s.points : s.points[:, keep]
    uu = valid isa AllValid ? data : data[:, keep]
    ww = w isa NoWeights ? w : w[keep]
    _assert_counts_can_accumulate(counts, size(x, 2), ww)
    _validate_array_shape(x, uu, s.metric)
    _dispatch_single_pass!(backend, sums, counts, x, uu, dist_be; geometry = SFH.pair_geometry_for(s.metric, Val(D)),
                           weights = ww)
    return sums, counts
end

# A structureless grid's tensor is the point tensor over its held cells, so the cells it drops drop
# their weights with them.
function gridded_tensor_sweep!(
    sums::AbstractArray, counts::AbstractVector, order::Val{P}, data::AbstractMatrix, s::ScatteredPairs, dist_be,
    ::Val{D}, spectral_backend;
    valid = AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
) where {P, D}
    _require_backend(backend)
    n = n_scattered_cells(s)
    size(data) == (D, n) || throw(DimensionMismatch(
        "field holds $(size(data, 2)) cells of $(size(data, 1)) components, the grid $n cells of $D",
    ))
    w = _pair_weights(weights, n, eltype(sums))
    keep = valid isa AllValid ? Colon() : findall(valid)
    x = valid isa AllValid ? s.points : s.points[:, keep]
    uu = valid isa AllValid ? data : data[:, keep]
    wk = (valid isa AllValid || w isa NoWeights) ? w : w[keep]
    _validate_array_shape(x, uu, s.metric)
    shape = PointField{D}()
    _tensor_shape_check(order, shape, sums, counts, uu, dist_be, nothing)
    _assert_counts_can_accumulate(counts, size(uu, 2), wk)
    _dispatch_tensor!(backend, shape, sums, counts, order, x, uu, dist_be;
                      geometry = SFH.pair_geometry_for(s.metric, Val(D)), weights = wk)
    return sums, counts
end

# A structureless grid's joint histogram is the point entry's over its held cells.
function gridded_lag_sweep!(
    sums::AbstractMatrix, counts::AbstractMatrix, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::ScatteredPairs, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K};
    valid = AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    second_axis::AbstractSecondAxisSource,
) where {D, V, K}
    _require_backend(backend)
    (V == 1 && K == 0) || throw(ArgumentError(
        "a joint histogram is of one vector field; got a multi-field of $V vector and $K scalar fields",
    ))
    n = n_scattered_cells(s)
    size(data) == (D, n) || throw(DimensionMismatch(
        "field holds $(size(data, 2)) cells of $(size(data, 1)) components, the grid $n cells of $D",
    ))
    w = _pair_weights(weights, n, eltype(sums))
    keep = valid isa AllValid ? Colon() : findall(valid)
    x = valid isa AllValid ? s.points : s.points[:, keep]
    uu = valid isa AllValid ? data : data[:, keep]
    ww = w isa NoWeights ? w : w[keep]
    _assert_counts_can_accumulate(counts, size(x, 2), ww)
    _validate_array_shape(x, uu, s.metric)
    geometry = SFH.pair_geometry_for(s.metric, Val(D))
    _require_value_axis(second_axis, geometry)
    _dispatch_execution_backend!(backend, PointField{D}(), sums, counts, sf, x, uu, dist_be, axis_be; geometry,
                                 weights = ww, second_axis)
    return sums, counts
end
