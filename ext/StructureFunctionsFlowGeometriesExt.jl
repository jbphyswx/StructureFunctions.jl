module StructureFunctionsFlowGeometriesExt

using FlowGeometries: FlowGeometries as FG
using Distances: Distances as DI
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH, StructureFunctionObjects as SFO, StructureFunctionTypes as SFT,
    MultiFields as MF

"""
    _grid_metric(grid) -> Distances metric

The metric a grid's geometry measures with.

A spherical grid's coordinates are `(λ, φ)` in **radians** on a sphere of the geometry's radius,
measured by [`SphericalDistance`](@ref SFH.SphericalDistance).
"""
_grid_metric(::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractCartesianGeometry}) = DI.Euclidean()

_grid_metric(grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry}) =
    SFH.SphericalDistance(FG.Geometry.radius(FG.Grids.grid_geometry(grid)))

_grid_metric(grid::FG.Grids.AbstractGrid) = throw(ArgumentError(
    "no distance metric is defined for $(nameof(typeof(FG.Grids.grid_geometry(grid)))); structure " *
    "functions on it would need one. Pass points and a metric to the unstructured entry.",
))

"""
    _grid_geometry(grid, Val(D)) -> geometry

The pair geometry a grid implies, with `D` the field's component count.
"""
_grid_geometry(grid::FG.Grids.AbstractGrid, vD::Val) = SFH.pair_geometry_for(_grid_metric(grid), vD)

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

# A lat-lon grid shares its geodesic frame around a circle of longitude. A stretched longitude axis
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

const GriddedField = Union{AbstractArray, MF.Fields}

"""
    cell_measure(grid) -> Vector

The measure of every cell of a FlowGeometries grid, in the gridded entries' cell order.
"""
SFC.cell_measure(grid::FG.Grids.AbstractGrid) = vec(FG.Grids.measure_array(grid))

"""
    _with_gridded(g, grid, u, kwargs)

`g(schedule, data, vD, vV, vK, valid)` for a gridded call, after validating it and settling which cells hold a
datum: the grid must be one the lag enumeration describes and the field must cover it. `data` and its layout are
those of `SFC._with_packed`. The schedule entry the call reaches checks the weights and the counts.
"""
function _with_gridded(g, grid, u::GriddedField, kwargs)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported keyword(s) $(join(keys(kwargs), ", ")) for a gridded calculation",
    ))
    sched = _lag_schedule(grid)
    return SFC._with_packed(u) do data, vD, vV, vK
        cells = SFC.n_cells(sched)
        size(data, 2) == cells || throw(DimensionMismatch(
            "u must be (component, cells...) covering the grid's $cells cells; got $(size(data, 2)) cells",
        ))
        SFC._val_int(vV) > 0 && _grid_geometry(grid, vD)  # refuses a geometry the schedules do not describe
        # The grid says which cells exist and the field says which hold a datum; a pair needs both ends.
        cm = FG.Grids.mask(grid)
        g(sched, data, vD, vV, vK, SFC.field_validity(data, cm isa FG.Grids.AllActive ? nothing : vec(cm)))
    end
end

const _Grid = FG.Grids.AbstractGrid
const _Pairwise = SFT.AbstractPairwiseStructureFunctionType
const _Tag = SB.AbstractSpectralBackend

@inline _gridded_result(sf_type, distance_bins, sums::AbstractVector, counts::AbstractVector,
                        ::Type{OT}) where {OT} =
    SFC._finalize(SFO.StructureFunctionSumsAndCounts(sf_type, distance_bins, sums, counts), OT)

@inline _gridded_result(sf_type, distance_bins, axis_bins, sums::AbstractMatrix, counts::AbstractMatrix, second_axis,
                        ::Type{OT}) where {OT} =
    SFC._finalize(SFO.StructureFunction2DSumsAndCounts(sf_type, distance_bins, axis_bins, sums, counts, second_axis),
                  OT)

"""
    calculate_structure_function(sf_type, grid, u, distance_bins[, spectral_backend][, CT][, OT]; weights, backend, workspace)

Structure function of the field `u` sampled on `grid`, computed by sweeping lag vectors wherever the
grid has a uniform direction: every pair sharing a lag shares its separation, direction and distance
bin. Which enumeration the grid gets is decided by the types of its axes (see
[`_lag_schedule`](@ref)).

`u` is `(component, cells...)` with its trailing axes matching the grid, and its component count may
exceed the grid's dimension — a lag then lies in the grid's directions and is zero along the rest. A
multi-field built from grid-shaped fields is taken the same way. On a spherical grid the
components are `(east, north[, radial])` and separations are in the unit of the geometry's radius.

`spectral_backend`, a `SpectralBackends` tag, names the algorithm that sums the pairs, an axis of its
own, orthogonal to which hardware runs it. Sweeping the lags (`DirectSumSpectralBackend()`) is exact for
every pairwise operator; a transform produces the increment moment tensor for **every** lag at once, so
its cost does not grow with the number of lags, and it serves the operators that are polynomials in `δu`
on every grid with a uniform direction. Both are exact, and `AutoSpectralBackend()`, the default, weighs
the two costs.

`weights`, one per cell, weights each pair by `w_k · w_kp` in sums and counts, so the bin average is
`Σ w w v / Σ w w`; `weights = cell_measure(grid)` makes it the area average, and the counts, then a
weighted pair mass, need a floating-point `CT` (default `$(SFC.DEFAULT_COUNT_TYPE)`). `backend` names the
hardware, as on the unstructured entry: `AutoBackend()` threads over the pairs of slabs — rows of a
lat-lon grid, positions along a stretched axis — when threads and the OhMyThreads extension are
available. `workspace`, a [`TransformWorkspace`](@ref SFC.TransformWorkspace), keeps a transform's buffers
and plans from one call to the next on the same grid; the lag sweep takes none. The result is the one the
array entry returns, of representation `OT`.
"""
function SFC.calculate_structure_function(
    sf_type::_Pairwise, grid::_Grid, u::GriddedField, distance_bins::AbstractVector, spectral_backend::_Tag,
    ::Type{CT}, ::Type{OT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    weights = nothing,
    workspace = nothing,
    kwargs...,
) where {CT <: Real, OT <: SFO.AbstractStructureFunction}
    return _with_gridded(grid, u, kwargs) do sched, data, vD, vV, vK, valid
        nb = SFC.n_histogram_bins(distance_bins)
        sums = SFC._result_zeros(backend, float(eltype(data)), nb)
        counts = SFC._result_zeros(backend, CT, nb)
        SFC.gridded_sweep!(sums, counts, sf_type, data, sched, distance_bins, vD, vV, vK, spectral_backend;
                           valid, weights, backend, SFC._workspace_kw(workspace)...)
        _gridded_result(sf_type, distance_bins, sums, counts, OT)
    end
end

SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector; kw...) =
    SFC.calculate_structure_function(sf, grid, u, bins, SB.AutoSpectralBackend(), SFC.DEFAULT_COUNT_TYPE,
                                     SFO.StructureFunction; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 ::Type{CT}; kw...) where {CT <: Real} =
    SFC.calculate_structure_function(sf, grid, u, bins, SB.AutoSpectralBackend(), CT, SFO.StructureFunction; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 ::Type{OT}; kw...) where {OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_function(sf, grid, u, bins, SB.AutoSpectralBackend(), SFC.DEFAULT_COUNT_TYPE, OT; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 ::Type{CT}, ::Type{OT}; kw...) where {CT <: Real, OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_function(sf, grid, u, bins, SB.AutoSpectralBackend(), CT, OT; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector, tag::_Tag; kw...) =
    SFC.calculate_structure_function(sf, grid, u, bins, tag, SFC.DEFAULT_COUNT_TYPE, SFO.StructureFunction; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector, tag::_Tag,
                                 ::Type{CT}; kw...) where {CT <: Real} =
    SFC.calculate_structure_function(sf, grid, u, bins, tag, CT, SFO.StructureFunction; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector, tag::_Tag,
                                 ::Type{OT}; kw...) where {OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_function(sf, grid, u, bins, tag, SFC.DEFAULT_COUNT_TYPE, OT; kw...)

"""
    calculate_structure_function(sf_type, grid, u, distance_bins, axis_bins[, spectral_backend][, CT][, OT]; second_axis, weights, backend, workspace)

The joint histogram in separation and `second_axis`: each pair's value (`InvariantValueAxis()`, on any
grid) or the angle a `SeparationAngleAxis` reads from each lag's direction (on a Cartesian grid), from the
same lags as the 1-D result, with `sums` and `counts` of shape `(n_distance, n_axis)`. The value histogram
is the lag sweep's; a transform yields moment sums and refuses it.

A lag that half-turns a periodic direction has two directions of equal length, and its pairs are split
between the two images' bins in equal halves, so the count type `CT` defaults to
`$(SFC.DEFAULT_SPLIT_COUNT_TYPE)`; an integer `CT` is refused where such a lag is in range.
"""
function SFC.calculate_structure_function(
    sf_type::_Pairwise, grid::_Grid, u::GriddedField, distance_bins::AbstractVector, axis_bins::AbstractVector,
    spectral_backend::_Tag, ::Type{CT}, ::Type{OT};
    second_axis::SFC.AbstractSecondAxisSource,
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    weights = nothing,
    workspace = nothing,
    kwargs...,
) where {CT <: Real, OT <: SFO.AbstractStructureFunction}
    return _with_gridded(grid, u, kwargs) do sched, data, vD, vV, vK, valid
        nb = SFC.n_histogram_bins(distance_bins)
        na = SFC.n_histogram_bins(axis_bins)
        sums = SFC._result_zeros(backend, float(eltype(data)), nb, na)
        counts = SFC._result_zeros(backend, CT, nb, na)
        SFC.gridded_sweep!(sums, counts, sf_type, data, sched, distance_bins, axis_bins, vD, vV, vK,
                           spectral_backend; valid, weights, backend, second_axis, SFC._workspace_kw(workspace)...)
        _gridded_result(sf_type, distance_bins, axis_bins, sums, counts, second_axis, OT)
    end
end

SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 axis_bins::AbstractVector; kw...) =
    SFC.calculate_structure_function(sf, grid, u, bins, axis_bins, SB.AutoSpectralBackend(),
                                     SFC.DEFAULT_SPLIT_COUNT_TYPE, SFO.StructureFunction2DSumsAndCounts; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 axis_bins::AbstractVector, ::Type{CT}; kw...) where {CT <: Real} =
    SFC.calculate_structure_function(sf, grid, u, bins, axis_bins, SB.AutoSpectralBackend(), CT,
                                     SFO.StructureFunction2DSumsAndCounts; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 axis_bins::AbstractVector, ::Type{OT}; kw...) where {OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_function(sf, grid, u, bins, axis_bins, SB.AutoSpectralBackend(),
                                     SFC.DEFAULT_SPLIT_COUNT_TYPE, OT; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 axis_bins::AbstractVector, ::Type{CT},
                                 ::Type{OT}; kw...) where {CT <: Real, OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_function(sf, grid, u, bins, axis_bins, SB.AutoSpectralBackend(), CT, OT; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 axis_bins::AbstractVector, tag::_Tag; kw...) =
    SFC.calculate_structure_function(sf, grid, u, bins, axis_bins, tag, SFC.DEFAULT_SPLIT_COUNT_TYPE,
                                     SFO.StructureFunction2DSumsAndCounts; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 axis_bins::AbstractVector, tag::_Tag, ::Type{CT}; kw...) where {CT <: Real} =
    SFC.calculate_structure_function(sf, grid, u, bins, axis_bins, tag, CT, SFO.StructureFunction2DSumsAndCounts;
                                     kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                 axis_bins::AbstractVector, tag::_Tag,
                                 ::Type{OT}; kw...) where {OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_function(sf, grid, u, bins, axis_bins, tag, SFC.DEFAULT_SPLIT_COUNT_TYPE, OT; kw...)

"""
    _with_gridded_batch(g, grid, u, kwargs)

As [`_with_gridded`](@ref) for a field sampled repeatedly on one grid, stored `(component, cells..., slices)`:
`g(schedule, data, vD, vV, vK, valid)` with `data` the `(D, cells, slices)` view of `u`, after checking that the grid
is one the lag enumeration describes and that every slice covers it; the validity is settled per slice.
"""
function _with_gridded_batch(g, grid, u::AbstractArray, kwargs)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported keyword(s) $(join(keys(kwargs), ", ")) for a gridded calculation",
    ))
    ndims(u) >= 3 || throw(DimensionMismatch(
        "a slice batch is stored (component, cells..., slices) and so has at least three axes; got $(size(u))",
    ))
    sched = _lag_schedule(grid)
    nt = size(u)[end]
    cells = SFC.n_cells(sched)
    return SFC._by_width(size(u, 1)) do vD
        data = reshape(u, SFC._val_int(vD), :, nt)
        size(data, 2) == cells || throw(DimensionMismatch(
            "each slice must cover the grid's $cells cells; got $(size(data, 2))",
        ))
        _grid_geometry(grid, vD)          # refuses a geometry the schedules do not describe
        cm = FG.Grids.mask(grid)
        g(sched, data, vD, Val(1), Val(0), SFC.batch_validity(u, cm isa FG.Grids.AllActive ? nothing : vec(cm)))
    end
end

"""
    calculate_structure_function_batch!(sums, counts, sf_type, grid, u, distance_bins[, spectral_backend]; weights, backend, workspace)
    calculate_structure_function_batch!(sums, counts, sf_type, grid, u, distance_bins, axis_bins[, spectral_backend]; second_axis, weights, backend, workspace)

Structure functions of a field sampled repeatedly on one `grid`: `u` is
`(component, cells..., slices)` and `sums`/`counts` are `(n_distance, n_slices)`, or with
`axis_bins` `(n_distance, n_angle, n_slices)`.

The grid fixes every pair, so the lags are enumerated once and each slice is summed against them;
this is the gridded counterpart of the point-list slice batch, which takes `(N_dims, N_points, T)`.
`weights` belongs to the cells and so is given once for the whole batch; validity is settled per
slice, as a slice may be missing data another holds. Accumulates into the caller's arrays and
returns nothing.
"""
function SFC.calculate_structure_function_batch!(
    sums::AbstractMatrix, counts::AbstractMatrix,
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    grid::FG.Grids.AbstractGrid,
    u::AbstractArray,
    distance_bins::AbstractVector,
    spectral_backend = SB.AutoSpectralBackend();
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    weights = nothing,
    workspace = nothing,
    kwargs...,
)
    _with_gridded_batch(grid, u, kwargs) do sched, data, vD, vV, vK, valid
        SFC.gridded_sweep_batch!(sums, counts, sf_type, data, sched, distance_bins, vD, vV, vK, spectral_backend;
                                 valid, weights, backend, SFC._workspace_kw(workspace)...)
    end
    return nothing
end

function SFC.calculate_structure_function_batch!(
    sums::AbstractArray{<:Any, 3}, counts::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    grid::FG.Grids.AbstractGrid,
    u::AbstractArray,
    distance_bins::AbstractVector,
    axis_bins::AbstractVector,
    spectral_backend = SB.AutoSpectralBackend();
    second_axis::SFC.AbstractSecondAxisSource,
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    weights = nothing,
    workspace = nothing,
    kwargs...,
)
    _with_gridded_batch(grid, u, kwargs) do sched, data, vD, vV, vK, valid
        SFC.gridded_sweep_batch!(sums, counts, sf_type, data, sched, distance_bins, axis_bins, vD, vV, vK,
                                 spectral_backend; valid, weights, backend, second_axis,
                                 SFC._workspace_kw(workspace)...)
    end
    return nothing
end

"""
    calculate_structure_functions_single_pass(grid, u, distance_bins[, spectral_backend][, CT][, OT]; weights, backend, workspace)

The six single-pass invariants `(S2, L2, T2, S3, L3, L1T2)` of the field `u` sampled on `grid`, from the lags of
[`calculate_structure_function`](@ref) on a grid, returned as the point entry's `NamedTuple` keyed by invariant
with its `:helmholtz` entry. All six are polynomials in the increment, so one transform serves them together;
`spectral_backend` (default `AutoSpectralBackend()`), `weights`, `backend` and `CT` are those of the
one-operator entry, and each invariant's result has representation `OT`.
"""
function SFC.calculate_structure_functions_single_pass(
    grid::_Grid, u::AbstractArray, distance_bins::AbstractVector, spectral_backend::_Tag, ::Type{CT}, ::Type{OT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    weights = nothing,
    workspace = nothing,
    kwargs...,
) where {CT <: Real, OT <: SFO.AbstractStructureFunction}
    return _with_gridded(grid, u, kwargs) do sched, data, vD, vV, vK, valid
        nb = SFC.n_histogram_bins(distance_bins)
        sums = SFC._result_zeros(backend, float(eltype(data)), SFC.SINGLE_PASS_N, nb)
        counts = SFC._result_zeros(backend, CT, SFC.SINGLE_PASS_N, nb)
        SFC.gridded_sweep!(sums, counts, SFT.SinglePassInvariants(), data, sched, distance_bins, vD, vV, vK,
                           spectral_backend; valid, weights, backend, SFC._workspace_kw(workspace)...)
        SFC._single_pass_collection_1d(sums, counts, distance_bins, OT)
    end
end

const _SPRaw = SFO.StructureFunctionSumsAndCounts

SFC.calculate_structure_functions_single_pass(grid::_Grid, u::AbstractArray, bins::AbstractVector; kw...) =
    SFC.calculate_structure_functions_single_pass(grid, u, bins, SB.AutoSpectralBackend(), SFC.DEFAULT_COUNT_TYPE,
                                                  _SPRaw; kw...)
SFC.calculate_structure_functions_single_pass(grid::_Grid, u::AbstractArray, bins::AbstractVector,
                                              ::Type{CT}; kw...) where {CT <: Real} =
    SFC.calculate_structure_functions_single_pass(grid, u, bins, SB.AutoSpectralBackend(), CT, _SPRaw; kw...)
SFC.calculate_structure_functions_single_pass(grid::_Grid, u::AbstractArray, bins::AbstractVector,
                                              ::Type{OT}; kw...) where {OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_functions_single_pass(grid, u, bins, SB.AutoSpectralBackend(), SFC.DEFAULT_COUNT_TYPE,
                                                  OT; kw...)
SFC.calculate_structure_functions_single_pass(grid::_Grid, u::AbstractArray, bins::AbstractVector, ::Type{CT},
                                              ::Type{OT}; kw...) where {CT <: Real, OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_functions_single_pass(grid, u, bins, SB.AutoSpectralBackend(), CT, OT; kw...)
SFC.calculate_structure_functions_single_pass(grid::_Grid, u::AbstractArray, bins::AbstractVector, tag::_Tag; kw...) =
    SFC.calculate_structure_functions_single_pass(grid, u, bins, tag, SFC.DEFAULT_COUNT_TYPE, _SPRaw; kw...)
SFC.calculate_structure_functions_single_pass(grid::_Grid, u::AbstractArray, bins::AbstractVector, tag::_Tag,
                                              ::Type{CT}; kw...) where {CT <: Real} =
    SFC.calculate_structure_functions_single_pass(grid, u, bins, tag, CT, _SPRaw; kw...)
SFC.calculate_structure_functions_single_pass(grid::_Grid, u::AbstractArray, bins::AbstractVector, tag::_Tag,
                                              ::Type{OT}; kw...) where {OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_functions_single_pass(grid, u, bins, tag, SFC.DEFAULT_COUNT_TYPE, OT; kw...)

"""
    calculate_structure_functions_single_pass_batch!(sums, counts, grid, u, distance_bins[, spectral_backend]; weights, backend, workspace)

The six single-pass invariants of every slice of a field sampled repeatedly on one `grid`, `u` stored
`(component, cells..., slices)`, accumulated into `sums`/`counts` `(6, n_distance, n_slices)` as
[`calculate_structure_function_batch!`](@ref) on a grid accumulates one operator. Returns nothing.
"""
function SFC.calculate_structure_functions_single_pass_batch!(
    sums::AbstractArray{<:Any, 3}, counts::AbstractArray{<:Any, 3},
    grid::FG.Grids.AbstractGrid,
    u::AbstractArray,
    distance_bins::AbstractVector,
    spectral_backend = SB.AutoSpectralBackend();
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    weights = nothing,
    workspace = nothing,
    kwargs...,
)
    _with_gridded_batch(grid, u, kwargs) do sched, data, vD, vV, vK, valid
        SFC.gridded_sweep_batch!(sums, counts, SFT.SinglePassInvariants(), data, sched, distance_bins, vD, vV, vK,
                                 spectral_backend; valid, weights, backend, SFC._workspace_kw(workspace)...)
    end
    return nothing
end

"""
    calculate_structure_function_tensor(order, grid, u, distance_bins[, axis_bins], spectral_backend[, CT][, OT]; second_axis, weights, backend, workspace)

The rank-`P` increment moment tensor of the vector field `u` on `grid`, by the transform
`spectral_backend` names — `AutoSpectralBackend()` or `FastFourierTransformSpectralBackend()` on a
grid with a uniform direction — with `sums` of shape `(D, …, D, n_bins)`; with `axis_bins` the joint
tensor over separation and the angle `second_axis` reads, `(D, …, D, n_bins, n_axis)`. The grid
rules, `weights`, `backend` and the count type `CT` are those of [`calculate_structure_function`](@ref)
on a grid, and `OT` is the result representation.
"""
function SFC.calculate_structure_function_tensor(
    order::Val{P}, grid::_Grid, u::GriddedField, distance_bins::AbstractVector, spectral_backend::_Tag,
    ::Type{CT}, ::Type{OT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    weights = nothing,
    workspace = nothing,
    kwargs...,
) where {P, CT <: Real, OT <: SFO.AbstractStructureFunction}
    return _with_gridded(grid, u, kwargs) do sched, data, vD, vV, vK, valid
        _one_vector_field(vV, vK)
        nb = SFC.n_histogram_bins(distance_bins)
        sums = SFC._result_zeros(backend, float(eltype(data)), ntuple(_ -> SFC._val_int(vD), P)..., nb)
        counts = SFC._result_zeros(backend, CT, nb)
        SFC.gridded_tensor_sweep!(sums, counts, order, data, sched, distance_bins, vD, spectral_backend;
                                  valid, weights, backend, SFC._workspace_kw(workspace)...)
        SFC._finalize(SFO.StructureFunctionTensorSumsAndCounts(order, distance_bins, sums, counts), OT)
    end
end

SFC.calculate_structure_function_tensor(order::Val, grid::_Grid, u::GriddedField, bins::AbstractVector, tag::_Tag;
                                        kw...) =
    SFC.calculate_structure_function_tensor(order, grid, u, bins, tag, SFC.DEFAULT_COUNT_TYPE,
                                            SFO.StructureFunctionTensor; kw...)
SFC.calculate_structure_function_tensor(order::Val, grid::_Grid, u::GriddedField, bins::AbstractVector, tag::_Tag,
                                        ::Type{CT}; kw...) where {CT <: Real} =
    SFC.calculate_structure_function_tensor(order, grid, u, bins, tag, CT, SFO.StructureFunctionTensor; kw...)
SFC.calculate_structure_function_tensor(order::Val, grid::_Grid, u::GriddedField, bins::AbstractVector, tag::_Tag,
                                        ::Type{OT}; kw...) where {OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_function_tensor(order, grid, u, bins, tag, SFC.DEFAULT_COUNT_TYPE, OT; kw...)

function SFC.calculate_structure_function_tensor(
    order::Val{P}, grid::_Grid, u::GriddedField, distance_bins::AbstractVector, axis_bins::AbstractVector,
    spectral_backend::_Tag, ::Type{CT}, ::Type{OT};
    second_axis::SFC.SeparationAngleAxis,
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    weights = nothing,
    workspace = nothing,
    kwargs...,
) where {P, CT <: Real, OT <: SFO.AbstractStructureFunction}
    return _with_gridded(grid, u, kwargs) do sched, data, vD, vV, vK, valid
        _one_vector_field(vV, vK)
        nb = SFC.n_histogram_bins(distance_bins)
        na = SFC.n_histogram_bins(axis_bins)
        sums = SFC._result_zeros(backend, float(eltype(data)), ntuple(_ -> SFC._val_int(vD), P)..., nb, na)
        counts = SFC._result_zeros(backend, CT, nb, na)
        SFC.gridded_tensor_sweep!(sums, counts, order, data, sched, distance_bins, axis_bins, vD, spectral_backend;
                                  valid, weights, backend, second_axis, SFC._workspace_kw(workspace)...)
        SFC._finalize(SFO.StructureFunctionTensor2DSumsAndCounts(order, distance_bins, axis_bins, sums, counts), OT)
    end
end

SFC.calculate_structure_function_tensor(order::Val, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                        axis_bins::AbstractVector, tag::_Tag; kw...) =
    SFC.calculate_structure_function_tensor(order, grid, u, bins, axis_bins, tag, SFC.DEFAULT_SPLIT_COUNT_TYPE,
                                            SFO.StructureFunctionTensor2DSumsAndCounts; kw...)
SFC.calculate_structure_function_tensor(order::Val, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                        axis_bins::AbstractVector, tag::_Tag, ::Type{CT}; kw...) where {CT <: Real} =
    SFC.calculate_structure_function_tensor(order, grid, u, bins, axis_bins, tag, CT,
                                            SFO.StructureFunctionTensor2DSumsAndCounts; kw...)
SFC.calculate_structure_function_tensor(order::Val, grid::_Grid, u::GriddedField, bins::AbstractVector,
                                        axis_bins::AbstractVector, tag::_Tag,
                                        ::Type{OT}; kw...) where {OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_function_tensor(order, grid, u, bins, axis_bins, tag, SFC.DEFAULT_SPLIT_COUNT_TYPE, OT;
                                            kw...)

_one_vector_field(::Val{1}, ::Val{0}) = nothing
_one_vector_field(::Val{V}, ::Val{K}) where {V, K} = throw(ArgumentError(
    "a tensor structure function is of one vector field; got a multi-field of $V vector and $K scalar fields",
))

"""
    calculate_structure_function(sf_type, grid, u, nodes::HarmonicNodes, spectral_backend[, CT][, OT]; backend)

The kernel-binned structure function of a field on a spherical grid by spherical harmonic
pseudo-coefficients (see [`harmonic_sweep!`](@ref SFC.harmonic_sweep!)), the cells' measure serving as
the quadrature weights so that the statistic is an area average. The grid's own mask and the field's
finiteness decide which cells are held. `CT`, `OT` and `backend` are those of the point-list harmonic
entry.
"""
function SFC.calculate_structure_function(
    sf_type::_Pairwise,
    grid::FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry},
    u::GriddedField,
    nodes::SF.HarmonicNodes,
    spectral_backend,
    ::Type{CT},
    ::Type{OT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
) where {CT <: Real, OT <: SFO.AbstractStructureFunction}
    coords = FG.Grids.materialize(grid)
    length(coords) == 2 || throw(ArgumentError(
        "the harmonic route takes a sphere's surface, (λ, φ); this grid has $(length(coords)) coordinates",
    ))
    N = length(coords[1])
    x = Matrix{eltype(coords[1])}(undef, 2, N)
    x[1, :] .= coords[1]
    x[2, :] .= coords[2]
    data = _cell_columns(u)
    size(data, 2) == N || throw(DimensionMismatch(
        "u must cover the grid's $N cells; got $(size(data, 2))",
    ))
    cm = FG.Grids.mask(grid)
    weights = SFC.cell_measure(grid)
    return SFC._with_valid(SFC.field_validity(data, cm isa FG.Grids.AllActive ? nothing :
                                                    copyto!(similar(data, Bool, N), vec(cm)))) do valid
        SFC.calculate_structure_function(sf_type, x, u, nodes, spectral_backend, CT, OT;
                                         distance_metric = _grid_metric(grid), weights, valid, backend)
    end
end

"""The `(components, cells)` matrix of a gridded field."""
_cell_columns(u::AbstractArray) = reshape(u, size(u, 1), :)
_cell_columns(f::MF.Fields) = MF.packed(f)

const _SphereGrid = FG.Grids.AbstractGrid{<:FG.Geometry.AbstractSphericalGeometry}

SFC.calculate_structure_function(sf::_Pairwise, grid::_SphereGrid, u::GriddedField, nodes::SF.HarmonicNodes,
                                 spectral_backend; kw...) =
    SFC.calculate_structure_function(sf, grid, u, nodes, spectral_backend, SFC._mass_type(u), SFO.StructureFunction;
                                     kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_SphereGrid, u::GriddedField, nodes::SF.HarmonicNodes,
                                 spectral_backend, ::Type{CT}; kw...) where {CT <: Real} =
    SFC.calculate_structure_function(sf, grid, u, nodes, spectral_backend, CT, SFO.StructureFunction; kw...)
SFC.calculate_structure_function(sf::_Pairwise, grid::_SphereGrid, u::GriddedField, nodes::SF.HarmonicNodes,
                                 spectral_backend, ::Type{OT}; kw...) where {OT <: SFO.AbstractStructureFunction} =
    SFC.calculate_structure_function(sf, grid, u, nodes, spectral_backend, SFC._mass_type(u), OT; kw...)

end # module
