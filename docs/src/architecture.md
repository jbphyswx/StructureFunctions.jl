# Architecture

How the package is organised and how a call becomes a kernel.

## Three orthogonal choices

Every calculation is a product of three independent choices, each a type:

```
what to accumulate        an operator      L2SFType(), MixedSFType{1,0,2}(), MomentTensorOperator{3}(), …
over which pairs          the input's shape and schedule: a point list, a multi-field, a grid's
                          UniformLagSchedule / RectilinearLagSchedule / ZonalLagSchedule / ScatteredPairs,
                          a ScatteredModesSchedule, HarmonicNodes
how to execute            a backend        SerialBackend(), ThreadedBackend(), DistributedBackend(),
                          GPUBackend(device), AutoBackend(); and on a grid a spectral tag,
                          FastFourierTransformSpectralBackend() / AutoSpectralBackend()
```

The operator knows nothing about the data layout, the schedule nothing about the operator beyond its
polynomial contract, and the backend nothing about either. A new operator is a subtype with one
functor method (and a `moment_contract` method if the transforms are to compute it); a new
convention for the transverse direction is a subtype with one `transverse_basis` method; a new
execution backend is a set of methods on the backend type.

## Modules

| module | file(s) | holds |
|---|---|---|
| `StructureFunctions` | `src/StructureFunctions.jl`, `src/BinEdges.jl`, `src/AuxiliaryAxes.jl` | exports, the bin-edge wrappers and squared-distance digitize plans, tapers, `HarmonicNodes`, `ModeBinEdges` |
| `MultiFields` | `src/MultiFields.jl` | `Fields` — several vector and scalar fields sampled at the same points, packed `(V·D + K, N)` |
| `HelperFunctions` | `src/HelperFunctions.jl` | pair frames on flat and spherical geometry, transverse conventions, `SphericalDistance` |
| `StructureFunctionTypes` | `src/StructureFunctionTypes.jl` | the operators, their functors, and the polynomial contract (`order`, `is_polynomial_operator`, `SymmetricMoments`, `moment_contract`) |
| `StructureFunctionObjects` | `src/StructureFunctionObjects.jl` | the result containers |
| `Calculations` | `src/Calculations.jl`, `src/Calculations/*.jl` | every entry point and kernel: dispatch, serial and single-pass point loops, culling, batches over auxiliary axes, gridded schedules and the direct sweep, the lag algebra of the transform, the sorted line route, the scattered-modes schedule, tensors, transforms and fits, the harmonic route, device stubs |
| `KHM` | `src/KHM.jl` | the exact-law inversions |

Extensions supply what needs another package: see [Extensions](extensions.md).

## From a call to a kernel

`calculate_structure_function(sf, x, u, bins; backend, distance_metric, weights, culling, …)`:

1. **Shape.** `_validate_array_shape` reads the velocity width `D = size(u, 1)` and the coordinate
   count the metric's geometry needs (`(D, N)` flat; `(2, N)` positions for a sphere whatever `D`), and
   classifies the ranks into a `PointField`, `SharedPositionField` or `VaryingPositionField`. The width
   is an array axis length, so the entry branches on it once and re-enters with a literal
   `Val{D}`; below that point every type is concrete.
2. **Backend.** `_dispatch_execution_backend(backend, shape, …)` selects the serial, threaded,
   distributed, MPI or device implementation; `AutoBackend` takes the threaded one when Julia has
   more than one thread and the OhMyThreads extension is loaded. Paths with no implementation for a
   request (weights on the GPU point kernels, in-place auxiliary axes on a device) refuse by name.
3. **Geometry.** `pair_geometry_for(metric, Val(D))` fixes a `FlatGeometry{D}` or
   `SphericalGeometry{D}`; `prepare_pair_inputs` widens spherical input to ambient 3-vectors once.
4. **Kernel.** Flat two- and three-dimensional point lists take the SIMD compute/scatter kernel over
   blocked pair tiles, with a cell-culling grid when the last bin edge bounds the separations; curved
   geometry takes the scalar per-point kernel through `pair_frame`; one-dimensional lists with a
   polynomial operator take the sorted line route; multi-fields take the multi-field kernel with the same
   blocking. Each pair's value is `sf(δu, r̂)` binned by a squared-distance plan, and an operator odd
   in a scalar increment reads the pair in its canonical orientation.
5. **Result.** The backends return the raw `StructureFunctionSumsAndCounts`; the public entry turns it
   into the requested `output_type` — the averaged `StructureFunction` by default.

A grid enters through the FlowGeometries extension: `calculate_structure_function(sf, grid, u, bins[,
spectral_tag]; backend, …)` reads the grid's **axis types** (a range is uniform, a coordinate vector is
not) and emits a schedule. Without a spectral tag the schedule runs the direct lag sweep, one
implementation for every separable schedule; with `FastFourierTransformSpectralBackend()` it runs the
transform engine; `AutoSpectralBackend()` costs both and takes the cheaper.

## The transform engine

For each pair of slabs (one slab on a uniform grid; the latitude rows of a lat-lon grid; the stretched
coordinates of a rectilinear one) the engine takes the forward transforms of every masked, weighted
monomial of degree up to the operator's order, forms the inverse columns of the increment moment
tensor — the `2^p` signed terms of the binomial expansion folded in the transform domain under
identity transport, or the raw cross-moments transformed by the pair frame per lag under frame
transport — and, per lag, contracts the symmetric moment store with the operator
(`moment_contract`) or accumulates it whole (the tensor entries). The host loop and the device kernel
share the lag algebra in `src/Calculations/lag_moments.jl`; the device version batches slab pairs and
runs two kernels per batch. The sorted line route and the harmonic route reuse the same contract: a
range's moments are prefix-sum differences, a harmonic kernel's are spin-weighted pseudo-spectra.

## Result containers

```julia
StructureFunction(operator, distance, values)                    # the averaged view, NaN where a bin is empty
StructureFunctionSumsAndCounts(operator, distance, sums, counts)  # the raw accumulator; adds across processes and time steps
StructureFunction2DSumsAndCounts(operator, distance_bins, value_bins, sums, counts)   # joint (distance × value) or (distance × angle)
StructureFunctionTensor(order, distance_bins, values)            # (D, …, D, n_bins[, aux...])
StructureFunctionTensorSumsAndCounts(order, distance_bins, sums, counts)
StructureFunctionTensor2DSumsAndCounts(order, distance_bins, axis_bins, sums, counts)
HelmholtzDecomposition2D(...)                                    # rotational and divergent second-order components
```

`distance` is whatever binned the result: bin edges (a plain vector or an `AbstractBinEdges`),
`HarmonicNodes` for a kernel-binned result on a sphere, `ModeBinEdges` for the soft-binned non-uniform
FFT route. `midpoints(distance)` is the abscissa in every case, and the transforms and fits read a
result through its operator and distance rather than through bare arrays.

## Bin edges

`AbstractBinEdges{T} <: AbstractVector{T}` wrappers give `digitize` an `O(1)` path:

```
AbstractBinEdges
├── BinEdges           a sorted vector, binary search
├── LinearBinEdges     a range: floor((x − first)/step) + 1 by one fused multiply-add, then a one-ulp correction
├── LogBinEdges        log-spaced edges: log(x) then the linear rule on the log grid
├── InfPaddedBinEdges  ±∞ pads around another edge set, no copy
└── ModeBinEdges       edges of a soft-binned result, carrying the mode schedule
```

The pair loops digitize the **squared** separation against a plan of squared edges
(`squared_digitize_plan`), which skips the square root; see [Binning Internals](uniform_bin_digitize.md)
for the derivation and why `round` is the wrong operator.

## Code layout

```
src/
├── StructureFunctions.jl         exports and includes
├── MultiFields.jl  BinEdges.jl  HelperFunctions.jl  AuxiliaryAxes.jl
├── StructureFunctionTypes.jl     operators and the polynomial contract
├── StructureFunctionObjects.jl   result containers
├── Calculations.jl               the compute module
├── Calculations/
│   ├── backends.jl shapes.jl dispatch.jl        shape → width literal → backend
│   ├── serial.jl serial_2d.jl serial_single_pass.jl culling.jl pair_schedule.jl   point kernels
│   ├── batch.jl batch_api.jl batch_leading.jl workspace.jl gpu_stubs.jl           auxiliary axes, device stubs
│   ├── multifields.jl second_axis.jl                                                  multi-fields, the second histogram axis
│   ├── gridded.jl gridded_zonal.jl lag_moments.jl scattered_modes.jl sorted_line.jl   schedules, sweeps, lag algebra
│   ├── tensor.jl transforms.jl fits.jl harmonic.jl
└── KHM.jl
ext/
├── StructureFunctionsOhMyThreadsExt.jl  StructureFunctionsDistributedExt.jl  StructureFunctionsMPIExt.jl
├── StructureFunctionsKernelAbstractionsExt.jl  + gpu/  (device kernels)  StructureFunctionsCUDAExt.jl
├── StructureFunctionsAbstractFFTsExt.jl  StructureFunctionsAbstractFFTsKernelAbstractionsExt.jl
├── StructureFunctionsNonuniformFFTsExt.jl  StructureFunctionsNonuniformFFTsKernelAbstractionsExt.jl
├── StructureFunctionsFINUFFTExt.jl  StructureFunctionsFINUFFTKernelAbstractionsExt.jl
├── StructureFunctionsNUFSHTExt.jl  StructureFunctionsFlowGeometriesExt.jl
└── StructureFunctionsBesselsExt.jl  StructureFunctionsLsqFitExt.jl
```

## Design rules

- **Decide by types, never by inspecting values.** A range axis is uniform and a coordinate vector is
  not; a spectral tag is a type; validity is `AllValid()` or a mask; a taper, a culling policy and a
  missing-lag policy are types. No `Symbol` options.
- **Exactness is a property of a route, stated on it.** The direct sweep, the transform and the sorted
  line are exact and are tested against each other; the harmonic route and the non-uniform FFT route
  are kernel-binned and their results carry a distance object that says so; spectra from masked data
  are estimators and their docstrings say so.
- **A request that cannot be met errors and names why.** Nothing falls back silently to a different
  quantity.
- **Culling, transforms and backends never change an answer**, only its cost: counts equal exactly,
  sums to round-off, and the tests hold every pair of routes to that.
