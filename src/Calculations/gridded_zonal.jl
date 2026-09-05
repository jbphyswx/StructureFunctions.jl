# Lat-lon enumeration. A lag in (λ, φ) is not a constant separation, so the uniform lag sweep does not
# apply — but the geodesic frame, written in each endpoint's own east/north basis, depends only on
# (φ₁, φ₂, Δλ) and not on absolute longitude. So the geometry is computed once per latitude pair and
# longitude offset and reused around the whole circle, instead of once per pair.

"""
    ZonalLagSchedule(lats, n_lon, dlon, radius, lon_periodic)

Pair enumeration for a lat-lon grid: one latitude per entry of `lats` (radians), `n_lon` cells per
row spaced `dlon` apart in longitude (radians), on a sphere of the given `radius`.

The latitude axis keeps whatever vector type it arrives as — a range stays a range — so a grid whose
axis is a formula is not materialised to carry it.

The field it sweeps is stored `(component, longitude, latitude)` with components in the local
`(east, north[, radial])` basis, which is the basis the transport matrices are written in, so no
per-pair conversion to ambient coordinates happens.
"""
struct ZonalLagSchedule{T, LV <: AbstractVector{T}}
    lats::LV
    n_lon::Int
    dlon::T
    radius::T
    lon_periodic::Bool
end

"""Cells the schedule covers."""
@inline n_zonal_cells(s::ZonalLagSchedule) = s.n_lon * length(s.lats)

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

One computation serves the whole circle of such pairs, because the geodesic frame written in the
endpoints' local bases does not depend on absolute longitude — checked against
[`pair_frame`](@ref) to `4e-15` over a full turn. It is evaluated at longitude zero, which is
therefore representative.

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
    rows(t, m̂, E, N) = (a, c) ->
        a == 1 ? (c == 1 ? LA.dot(t, E) : c == 2 ? LA.dot(t, N) : zero(T)) :
        a == 2 ? (c == 1 ? LA.dot(m̂, E) : c == 2 ? LA.dot(m̂, N) : zero(T)) :
        (c == 3 ? one(T) : zero(T))
    fA = rows(t_A, m̂, EA, NA)
    fB = rows(t_B, m̂, EB, NB)
    A = SA.SMatrix{D, D, T}((fA(a, c) for a in 1:D, c in 1:D)...)
    B = SA.SMatrix{D, D, T}((fB(a, c) for a in 1:D, c in 1:D)...)
    return ok, r, A, B
end

"""Longitude offsets to visit for a latitude pair, one per distinct separation."""
@inline function _zonal_lag_range(s::ZonalLagSchedule, same_row::Bool)
    n = s.n_lon
    s.lon_periodic || return same_row ? (1:(n - 1)) : (-(n - 1)):(n - 1)
    # A full circle has `n` distinct offsets, taken as minimum images. Within one row a pair is named
    # by both `+m` and `-m`, so only the positive half is kept.
    return same_row ? (1:(n ÷ 2)) : (-((n - 1) ÷ 2)):(n ÷ 2)
end

# Longitude segments of constant index offset, as the uniform sweep uses: the part whose partner
# stays in range, and — on a full circle — the part that wraps.
@inline function _zonal_segments(s::ZonalLagSchedule, m::Int, half::Bool)
    n = s.n_lon
    half && return ((1:(n ÷ 2), m), (1:0, 0))
    if m >= 0
        return (s.lon_periodic && m > 0) ?
            ((1:(n - m), m), (((n - m + 1):n), m - n)) :
            ((1:(n - m), m), (1:0, 0))
    end
    return s.lon_periodic ?
        (((1 - m):n, m), (1:(-m), m + n)) :
        (((1 - m):n, m), (1:0, 0))
end

"""
    gridded_lag_sweep!(sums, counts, sf, u, schedule::ZonalLagSchedule, distance_bins, ::Val{D}; valid)

Accumulate every pair on a lat-lon grid into the 1-D distance histogram.

`u` is `(component, longitude, latitude)` in the local `(east, north[, radial])` basis. Latitude
pairs whose rows cannot come within the largest finite bin are skipped whole, and within a pair the
geometry is computed once per longitude offset rather than once per pair.
"""
function gridded_lag_sweep!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    u::AbstractArray, s::ZonalLagSchedule{T}, dist_be, ::Val{D};
    valid = AllValid(),
) where {OT, CT, T, D}
    n_lat = length(s.lats)
    n_lon = s.n_lon
    size(u) == (D, n_lon, n_lat) || throw(DimensionMismatch(
        "u must be (component, longitude, latitude) = ($D, $n_lon, $n_lat); got $(size(u))",
    ))
    plan = squared_digitize_plan(dist_be)
    nb = n_histogram_bins(plan)
    length(sums) == nb && length(counts) == nb || throw(DimensionMismatch(
        "sums and counts must have length $nb; got $(length(sums)) and $(length(counts))",
    ))
    uf = reshape(u, D, n_lon * n_lat)
    geom = SFH.SphericalGeometry{D}(DI.SphericalAngle(), s.radius)
    r_max = _cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    # On the sphere the pair frame IS the basis, so the longitudinal direction is ê₁ for every pair.
    r_hat = SA.SVector{D, T}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(D)))

    @inbounds for j1 in 1:n_lat, j2 in j1:n_lat
        # Two rows are never closer than their latitude difference, so a row pair beyond the last
        # finite bin contributes nothing at any longitude offset.
        s.radius * abs(s.lats[j2] - s.lats[j1]) > r_max && continue
        base1 = (j1 - 1) * n_lon
        base2 = (j2 - 1) * n_lon
        same_row = j1 == j2
        for m in _zonal_lag_range(s, same_row)
            dlam = T(m) * s.dlon
            ok, r, A, B = zonal_transport(geom, s.lats[j1], s.lats[j2], dlam, Val(D))
            ok || continue
            b = squared_digitize(plan, r * r)
            1 <= b <= nb || continue
            # A half-turn offset joins the two ends by two minimal paths of opposite sign, so the
            # separation direction is not unique and the operator is averaged over both.
            ambiguous = s.lon_periodic && iseven(n_lon) && abs(m) == n_lon ÷ 2
            A2, B2 = A, B
            if ambiguous
                _, _, A2, B2 = zonal_transport(geom, s.lats[j1], s.lats[j2], -dlam, Val(D))
            end
            total = zero(OT)
            n_pairs = 0
            for (rng, off) in _zonal_segments(s, m, same_row && ambiguous)
                isempty(rng) && continue
                for i in rng
                    k1 = base1 + i
                    k2 = base2 + i + off
                    okp = valid[k1] & valid[k2]
                    uA = SA.SVector{D, T}(ntuple(c -> uf[c, k1], Val(D)))
                    uB = SA.SVector{D, T}(ntuple(c -> uf[c, k2], Val(D)))
                    v = sf(B * uB - A * uA, r_hat)
                    ambiguous && (v = (v + sf(B2 * uB - A2 * uA, r_hat)) / 2)
                    total += okp ? OT(v) : zero(OT)
                    n_pairs += okp
                end
            end
            sums[b] += total
            counts[b] += CT(n_pairs)
        end
    end
    return sums, counts
end


"""
    ScatteredPairs(points, metric)

Pair enumeration for a grid with no structure to exploit: the points themselves, and the metric that
measures between them.

A pixelized sphere, a curvilinear mesh, a node set, a stretched axis — none of these share a
separation between many pairs, so there is nothing to hoist and the honest thing is to enumerate the
pairs. That is what the unstructured path does, with culling, so this schedule routes to it rather
than reimplementing it.
"""
struct ScatteredPairs{X <: AbstractMatrix, M}
    points::X
    metric::M
end

"""Cells the schedule covers."""
@inline n_scattered_cells(s::ScatteredPairs) = size(s.points, 2)

"""
    gridded_lag_sweep!(sums, counts, sf, u, schedule::ScatteredPairs, distance_bins, ::Val{D}; valid)

Accumulate every pair of a structureless grid, by enumerating them.

Cells holding nothing are dropped before the sweep rather than tested inside it: the pair loop has no
mask, and a point that takes part in no pair is simply not passed to it.
"""
function gridded_lag_sweep!(
    sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    u::AbstractArray, s::ScatteredPairs, dist_be, ::Val{D};
    valid = AllValid(),
) where {D}
    n = n_scattered_cells(s)
    uf = reshape(u, D, n)
    size(uf, 2) == n || throw(DimensionMismatch(
        "field holds $(size(uf, 2)) cells, the grid $n",
    ))
    keep = valid isa AllValid ? Colon() : findall(valid)
    x = valid isa AllValid ? s.points : s.points[:, keep]
    uu = valid isa AllValid ? uf : uf[:, keep]
    calculate_structure_function!(sums, counts, sf, x, uu, dist_be; distance_metric = s.metric)
    return sums, counts
end
