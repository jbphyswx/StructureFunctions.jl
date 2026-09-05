module StructureFunctionsFlowGeometriesExt

using FlowGeometries: FlowGeometries as FG
using Distances: Distances as DI
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH, StructureFunctionObjects as SFO, StructureFunctionTypes as SFT

"""
    _grid_spacing(grid) -> NTuple{N, T}

Constant spacing of each direction. Only called once every direction is known to be uniform.
"""
function _grid_spacing(grid::FG.Grids.AbstractGrid{G, T}) where {G, T}
    return ntuple(Val(length(FG.Grids.coordinates(grid)))) do d
        FG.Grids.spacing(grid, d)
    end
end

"""
    _grid_metric(grid) -> Distances metric

The metric a grid's geometry measures with.

A spherical grid's coordinates are `(λ, φ)` in **radians**, which is `Distances.SphericalAngle` —
never `Haversine`, which reads degrees and would scale every separation by 180/π.
"""
_grid_metric(::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractCartesianGeometry}) = DI.Euclidean()

_grid_metric(::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry}) = DI.SphericalAngle()

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
    _lag_schedule(grid) -> UniformLagSchedule | ZonalLagSchedule | ScatteredPairs

The enumeration a grid supports.

A uniform rectilinear grid shares a separation along each lag; a lat-lon grid shares a geodesic frame
around each circle of longitude; anything else — a stretched axis, a curvilinear mesh, a pixelized
sphere, a node set — shares nothing between pairs, and its pairs are enumerated. All three are exact;
they differ only in what can be hoisted.
"""
function _lag_schedule(grid::FG.Grids.AbstractGrid)
    _grid_metric(grid)                        # refuses a geometry with no metric, before any work
    return _scattered(grid)
end

function _lag_schedule(grid::FG.Grids.AbstractStructuredGrid{<:FG.Geometry.AbstractCartesianGeometry})
    c = FG.Grids.coordinates(grid)
    all(d -> FG.Grids.isuniform(grid, d), 1:length(c)) || return _scattered(grid)
    dims = ntuple(d -> length(c[d]), Val(length(c)))
    return SFC.UniformLagSchedule(dims, _grid_spacing(grid), FG.Grids.periodic_flags(grid))
end

# A lat-lon grid's lags are not constant separations, so it gets the zonal schedule instead: the
# geodesic frame is shared around a circle of longitude, not along a lag.
function _lag_schedule(grid::FG.Grids.AbstractStructuredGrid{<:FG.Geometry.AbstractSphericalGeometry})
    c = FG.Grids.coordinates(grid)
    (length(c) == 2 && FG.Grids.isuniform(grid, 1)) || return _scattered(grid)
    return SFC.ZonalLagSchedule(c[2], length(c[1]), FG.Grids.spacing(grid, 1),
                                FG.Geometry.radius(FG.Grids.grid_geometry(grid)),
                                FG.Grids.isperiodic(grid, 1))
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

"""
    _gridded_setup(grid, u, distance_bins, count_eltype, kwargs) -> (schedule, D, sums, counts, valid)

Validate a gridded call, allocate its histogram, and settle which cells hold a datum: the grid must
be one the lag enumeration describes, and the field must match it.
"""
function _gridded_setup(grid, u, distance_bins, ::Type{CT}, kwargs) where {CT}
    isempty(kwargs) || throw(ArgumentError(
        "unsupported keyword(s) $(join(keys(kwargs), ", ")) for a gridded calculation",
    ))
    sched = _lag_schedule(grid)
    D = size(u, 1)
    length(u) == D * _schedule_cells(sched) || throw(DimensionMismatch(
        "u must be (component, cells...) covering the grid's $(_schedule_cells(sched)) cells; got " *
        "$(size(u))",
    ))
    SFC._validate_spatial_dimension(D)
    _grid_geometry(grid, D)          # refuses a geometry the schedules do not describe
    SFC._assert_counts_representable(CT, _schedule_cells(sched))
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(distance_bins))
    # The grid says which cells exist and the field says which hold a datum; a pair needs both ends.
    cm = FG.Grids.mask(grid)
    valid = SFC.field_validity(u, Val(D), cm isa FG.Grids.AllActive ? nothing : vec(cm))
    return sched, D, zeros(float(eltype(u)), nb), zeros(CT, nb), valid
end

@inline _schedule_cells(s::SFC.UniformLagSchedule) = prod(s.dims)
@inline _schedule_cells(s::SFC.ZonalLagSchedule) = SFC.n_zonal_cells(s)
@inline _schedule_cells(s::SFC.ScatteredPairs) = SFC.n_scattered_cells(s)

@inline _gridded_result(sf_type, distance_bins, sums, counts, ::Type{OT}) where {OT} =
    SFC._finalize(SFO.StructureFunctionSumsAndCounts(sf_type, distance_bins, sums, counts), OT)

"""
    calculate_structure_function(sf_type, grid, u, distance_bins[, count_eltype]; kwargs...)

Structure function of the field `u` sampled on `grid`, computed by sweeping lag vectors rather than
pairs: on a uniform grid every pair sharing a lag shares its separation, direction and distance bin.

`u` is `(component, cells...)` with its trailing axes matching the grid, and its component count may
exceed the grid's dimension — a lag then lies in the grid's directions and is zero along the rest.

The result is the one the array entry returns; `output_type` selects its representation.
"""
function SFC.calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    grid::FG.Grids.AbstractGrid,
    u::AbstractArray,
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32;
    output_type::Type{OT} = SFO.StructureFunction,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
) where {OT, CT}
    sched, D, sums, counts, valid = _gridded_setup(grid, u, distance_bins, CT, kwargs)
    SFC.gridded_lag_sweep!(sums, counts, sf_type, u, sched, distance_bins, Val(D); valid)
    return _gridded_result(sf_type, distance_bins, sums, counts, OT)
end

"""
    calculate_structure_function(sf_type, grid, u, distance_bins, count_eltype, spectral_backend; kwargs...)

As above, summing the pairs by the algorithm `spectral_backend` names — a `SpectralBackends` tag.

Which algorithm sums the pairs is an axis of its own, orthogonal to which hardware runs it. Sweeping
the lags is exact for every pairwise operator and is what the shorter form does; a transform produces
the second-order increment tensor for **every** lag at once, so its cost does not grow with the
number of lags, and it serves the operators that are quadratic forms in `δu`. Both are exact, and
`AutoSpectralBackend` weighs the two costs.
"""
function SFC.calculate_structure_function(
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    grid::FG.Grids.AbstractGrid,
    u::AbstractArray,
    distance_bins::AbstractVector,
    count_eltype::Type{CT},
    spectral_backend;
    output_type::Type{OT} = SFO.StructureFunction,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
) where {OT, CT}
    sched, D, sums, counts, valid = _gridded_setup(grid, u, distance_bins, CT, kwargs)
    SFC.gridded_sweep!(sums, counts, sf_type, u, sched, distance_bins, Val(D), spectral_backend;
                       valid)
    return _gridded_result(sf_type, distance_bins, sums, counts, OT)
end

end # module
