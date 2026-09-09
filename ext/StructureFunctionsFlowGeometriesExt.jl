module StructureFunctionsFlowGeometriesExt

using FlowGeometries: FlowGeometries as FG
using Distances: Distances as DI
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH, StructureFunctionObjects as SFO, StructureFunctionTypes as SFT,
    Channels as CH

"""
    _grid_metric(grid) -> Distances metric

The metric a grid's geometry measures with.

A spherical grid's coordinates are `(λ, φ)` in **radians** on a sphere of the geometry's radius,
which is [`SphericalDistance`](@ref SFH.SphericalDistance) — never `Haversine`, which reads degrees
and would scale every separation by 180/π.
"""
_grid_metric(::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractCartesianGeometry}) = DI.Euclidean()

_grid_metric(grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry}) =
    SFH.SphericalDistance(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))

_grid_metric(grid::FG.Grids.AbstractGrid) = throw(ArgumentError(
    "no distance metric is defined for $(nameof(typeof(FG.Grids.grid_geometry(grid)))); structure " *
    "functions on it would need one. Pass points and a metric to the unstructured entry.",
))

"""
    _grid_geometry(grid, D) -> geometry

The pair geometry a grid implies, with `D` the field's component count.
"""
_grid_geometry(grid::FG.Grids.AbstractGrid, D::Int) =
    SFH.pair_geometry_for(_grid_metric(grid), Val(D))

"""
    _lag_schedule(grid) -> UniformLagSchedule | RectilinearLagSchedule | ZonalLagSchedule | ScatteredPairs

The enumeration a grid supports, read from the **types** of its axes: a range is uniform, a vector of
coordinates is not, and no coordinate value is inspected to decide.

A Cartesian grid whose every axis is uniform shares a separation along each lag; one with some
uniform axes shares it along those, between every pair of positions on the rest; a lat-lon grid with
uniform longitude shares a geodesic frame around each circle of longitude; anything else — a
curvilinear mesh, a pixelized sphere, a node set, a grid with no uniform axis — shares nothing
between pairs, and its pairs are enumerated. All are exact; they differ only in what can be hoisted.
"""
function _lag_schedule(grid::FG.Grids.AbstractGrid)
    _grid_metric(grid)                        # refuses a geometry with no metric, before any work
    return _scattered(grid)
end

function _lag_schedule(grid::FG.Grids.AbstractStructuredGrid{<:FG.Geometry.AbstractCartesianGeometry})
    c = FG.Grids.coordinates(grid)
    N = length(c)
    uniform = ntuple(d -> FG.Grids.isuniform(grid, d), Val(N))
    periodic = FG.Grids.periodic_flags(grid)
    for d in 1:N
        (periodic[d] && !uniform[d]) && throw(ArgumentError(
            "direction $d wraps but its axis is a coordinate list, and a straight-line separation " *
            "cannot be wrapped without a constant spacing to wrap by. Give the direction a range " *
            "axis, or build the grid without its periodic flag.",
        ))
    end
    all(uniform) && return SFC.UniformLagSchedule(
        ntuple(d -> length(c[d]), Val(N)), _grid_spacing(grid), periodic,
    )
    any(uniform) || return _scattered(grid)
    order = (findall(uniform)..., findall(!, uniform)...)
    Du = count(uniform)
    su = SFC.UniformLagSchedule(
        Tuple(length(c[order[k]]) for k in 1:Du),
        Tuple(FG.Grids.spacing(grid, order[k]) for k in 1:Du),
        Tuple(periodic[order[k]] for k in 1:Du),
    )
    return SFC.RectilinearLagSchedule(su, Tuple(c[order[k]] for k in (Du + 1):N), order)
end

# A lat-lon grid's lags are not constant separations, so it gets the zonal schedule instead: the
# geodesic frame is shared around a circle of longitude, not along a lag. A stretched longitude axis
# enumerates pairs; the great-circle metric wraps of itself, so a periodic flag on it is no obstacle.
function _lag_schedule(grid::FG.Grids.AbstractStructuredGrid{<:FG.Geometry.AbstractSphericalGeometry})
    c = FG.Grids.coordinates(grid)
    (length(c) == 2 && FG.Grids.isuniform(grid, 1)) || return _scattered(grid)
    return SFC.ZonalLagSchedule(c[2], length(c[1]), FG.Grids.spacing(grid, 1),
                                FG.Geometry.radius(FG.Grids.grid_geometry(grid)),
                                FG.Grids.isperiodic(grid, 1))
end

"""Constant spacing of each direction; only called once every direction is known to be uniform."""
function _grid_spacing(grid::FG.Grids.AbstractGrid{G, T}) where {G, T}
    return ntuple(Val(length(FG.Grids.coordinates(grid)))) do d
        FG.Grids.spacing(grid, d)
    end
end

"""Every cell's coordinates, as the schedule that simply enumerates pairs."""
function _scattered(grid::FG.Grids.AbstractGrid)
    coords = FG.Grids.materialize(grid)
    W = length(coords)
    n = length(coords[1])
    pts = Matrix{eltype(coords[1])}(undef, W, n)
    for d in 1:W
        pts[d, :] .= coords[d]
    end
    return SFC.ScatteredPairs(pts, _grid_metric(grid))
end

const GriddedField = Union{AbstractArray, CH.Fields}

"""
    _gridded_setup(grid, u, distance_bins, count_eltype, kwargs) -> (schedule, data, vD, vV, vK, valid)

Validate a gridded call and settle which cells hold a datum: the grid must be one the lag
enumeration describes, and the field must cover it.
"""
function _gridded_setup(grid, u::GriddedField, distance_bins, ::Type{CT}, kwargs, verbose::Bool) where {CT}
    isempty(kwargs) || throw(ArgumentError(
        "unsupported keyword(s) $(join(keys(kwargs), ", ")) for a gridded calculation",
    ))
    sched = _lag_schedule(grid)
    data, vD, vV, vK = SFC._packed(u)
    D = SFC.SFC_val_int(vD)
    V = SFC.SFC_val_int(vV)
    cells = SFC.n_cells(sched)
    size(data, 2) == cells || throw(DimensionMismatch(
        "u must be (component, cells...) covering the grid's $cells cells; got $(size(data, 2)) cells",
    ))
    if V > 0
        SFC._validate_spatial_dimension(D)
        _grid_geometry(grid, D)          # refuses a geometry the schedules do not describe
    end
    SFC._assert_counts_representable(CT, cells)
    # The grid says which cells exist and the field says which hold a datum; a pair needs both ends.
    cm = FG.Grids.mask(grid)
    valid = SFC.field_validity(data, cm isa FG.Grids.AllActive ? nothing : vec(cm))
    verbose && @info "gridded structure function: $(nameof(typeof(sched))) over $cells cells"
    return sched, data, vD, vV, vK, valid
end

@inline _gridded_result(sf_type, distance_bins, sums::AbstractVector, counts::AbstractVector,
                        ::Type{OT}) where {OT} =
    SFC._finalize(SFO.StructureFunctionSumsAndCounts(sf_type, distance_bins, sums, counts), OT)

@inline _gridded_result(sf_type, distance_bins, axis_bins, sums::AbstractMatrix, counts::AbstractMatrix,
                        ::Type{OT}) where {OT} =
    SFC._finalize(SFO.StructureFunction2DSumsAndCounts(sf_type, distance_bins, axis_bins, sums, counts),
                  OT)

"""
    calculate_structure_function(sf_type, grid, u, distance_bins[, count_eltype]; backend, kwargs...)

Structure function of the field `u` sampled on `grid`, computed by sweeping lag vectors rather than
pairs wherever the grid has a uniform direction: every pair sharing a lag shares its separation,
direction and distance bin. Which enumeration the grid gets is decided by the types of its axes (see
[`_lag_schedule`](@ref)) and reported when `verbose = true`.

`u` is `(component, cells...)` with its trailing axes matching the grid, and its component count may
exceed the grid's dimension — a lag then lies in the grid's directions and is zero along the rest. A
`Fields` bundle built from grid-shaped channels is taken the same way. On a spherical grid the
components are `(east, north[, radial])` and separations are in the unit of the geometry's radius.

`backend` names the hardware, as on the unstructured entry: `AutoBackend()` threads over the pairs
of slabs — rows of a lat-lon grid, positions along a stretched axis — when threads and the OhMyThreads
extension are available. The result is the one the array entry returns; `output_type` selects its
representation.
"""
function SFC.calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    grid::FG.Grids.AbstractGrid,
    u::GriddedField,
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
) where {OT, CT}
    sched, data, vD, vV, vK, valid = _gridded_setup(grid, u, distance_bins, CT, kwargs, verbose)
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(distance_bins))
    sums = zeros(float(eltype(data)), nb)
    counts = zeros(CT, nb)
    SFC.gridded_lag_sweep!(sums, counts, sf_type, data, sched, distance_bins, vD, vV, vK; valid, backend)
    return _gridded_result(sf_type, distance_bins, sums, counts, OT)
end

"""
    calculate_structure_function(sf_type, grid, u, distance_bins, count_eltype, spectral_backend; backend, kwargs...)

As above, summing the pairs by the algorithm `spectral_backend` names — a `SpectralBackends` tag.

Which algorithm sums the pairs is an axis of its own, orthogonal to which hardware runs it. Sweeping
the lags is exact for every pairwise operator and is what the shorter form does; a transform produces
the increment moment tensor for **every** lag at once, so its cost does not grow with the number of
lags, and it serves the operators that are polynomials in `δu` on every grid with a uniform
direction. Both are exact, and `AutoSpectralBackend` weighs the two costs.
"""
function SFC.calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    grid::FG.Grids.AbstractGrid,
    u::GriddedField,
    distance_bins::AbstractVector,
    count_eltype::Type{CT},
    spectral_backend;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
) where {OT, CT}
    sched, data, vD, vV, vK, valid = _gridded_setup(grid, u, distance_bins, CT, kwargs, verbose)
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(distance_bins))
    sums = zeros(float(eltype(data)), nb)
    counts = zeros(CT, nb)
    SFC.gridded_sweep!(sums, counts, sf_type, data, sched, distance_bins, vD, vV, vK, spectral_backend;
                       valid, backend)
    return _gridded_result(sf_type, distance_bins, sums, counts, OT)
end

"""
    calculate_structure_function(sf_type, grid, u, distance_bins, axis_bins[, spectral_backend]; second_axis, backend, kwargs...)

The joint histogram in separation and the angle `second_axis` reads from each lag's direction, on a
Cartesian grid: `S(r, θ)` from the same lags as the 1-D result, with `sums` and `counts` of shape
`(n_distance, n_angle)`.

Counts are floating point: a lag that half-turns a periodic direction has two directions of equal
length, and its pairs are split between the two angle bins in equal halves.
"""
function SFC.calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    grid::FG.Grids.AbstractGrid,
    u::GriddedField,
    distance_bins::AbstractVector,
    axis_bins::AbstractVector;
    second_axis::SFC.SeparationAngleAxis,
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction2DSumsAndCounts,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
) where {OT}
    sched, data, vD, vV, vK, valid = _gridded_setup(grid, u, distance_bins, Float64, kwargs, verbose)
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(distance_bins))
    na = SFC.n_histogram_bins(SF.BinEdges(axis_bins))
    sums = zeros(float(eltype(data)), nb, na)
    counts = zeros(Float64, nb, na)
    SFC.gridded_lag_sweep!(sums, counts, sf_type, data, sched, distance_bins, axis_bins, vD, vV, vK;
                           valid, backend, second_axis)
    return _gridded_result(sf_type, distance_bins, axis_bins, sums, counts, OT)
end

function SFC.calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    grid::FG.Grids.AbstractGrid,
    u::GriddedField,
    distance_bins::AbstractVector,
    axis_bins::AbstractVector,
    spectral_backend;
    second_axis::SFC.SeparationAngleAxis,
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction2DSumsAndCounts,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
) where {OT}
    sched, data, vD, vV, vK, valid = _gridded_setup(grid, u, distance_bins, Float64, kwargs, verbose)
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(distance_bins))
    na = SFC.n_histogram_bins(SF.BinEdges(axis_bins))
    sums = zeros(float(eltype(data)), nb, na)
    counts = zeros(Float64, nb, na)
    SFC.gridded_sweep!(sums, counts, sf_type, data, sched, distance_bins, axis_bins, vD, vV, vK,
                       spectral_backend; valid, backend, second_axis)
    return _gridded_result(sf_type, distance_bins, axis_bins, sums, counts, OT)
end

"""
    calculate_structure_function(sf_type, grid, u, nodes::HarmonicNodes, spectral_backend; kwargs...)

The kernel-binned structure function of a field on a spherical grid by spherical harmonic
pseudo-coefficients (see [`harmonic_sweep!`](@ref SFC.harmonic_sweep!)), the cells' measure serving as
the quadrature weights so that the statistic is an area average. The grid's own mask and the field's
finiteness decide which cells are held.
"""
function SFC.calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry},
    u::GriddedField,
    nodes::SF.HarmonicNodes,
    spectral_backend;
    output_type::Type{OT} = SFO.StructureFunction,
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT}
    coords = FG.Grids.materialize(grid)
    length(coords) == 2 || throw(ArgumentError(
        "the harmonic route takes a sphere's surface, (λ, φ); this grid has $(length(coords)) coordinates",
    ))
    N = length(coords[1])
    x = Matrix{eltype(coords[1])}(undef, 2, N)
    x[1, :] .= coords[1]
    x[2, :] .= coords[2]
    data, vD, vV, vK = SFC._packed(u)
    size(data, 2) == N || throw(DimensionMismatch(
        "u must cover the grid's $N cells; got $(size(data, 2))",
    ))
    cm = FG.Grids.mask(grid)
    valid = SFC.field_validity(data, cm isa FG.Grids.AllActive ? nothing : vec(cm))
    weights = vec(FG.Grids.measure_array(grid))
    verbose && @info "harmonic structure function on the grid: $(length(nodes)) nodes, lmax = $(nodes.lmax)"
    return SFC.calculate_structure_function(sf_type, x, u, nodes, spectral_backend;
        distance_metric = _grid_metric(grid), weights, valid, output_type = OT, verbose = false, show_progress)
end

end # module
