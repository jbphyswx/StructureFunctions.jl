```@meta
CurrentModule = StructureFunctions
```

# Internals

Types and functions that are not exported but that the exported entries name: the schedules a grid
resolves to, the lag transports, the field shapes, culling, the second histogram axis, the
polynomial contract of the operators and the digitize plans.

```@index
Pages = ["internals.md"]
```

## Schedules and transports

```@docs
Calculations.AbstractSeparableSchedule
Calculations.UniformLagSchedule
Calculations.RectilinearLagSchedule
Calculations.ZonalLagSchedule
Calculations.ScatteredPairs
Calculations.AbstractLagTransport
Calculations.IdentityTransport
Calculations.FrameTransport
Calculations.uniform_axes
Calculations.n_slabs
Calculations.separable_layout
Calculations.lag_limits
Calculations.uniform_lag_box
Calculations.lag_transport
Calculations.batch_shares_lag_geometry
Calculations.NoWeights
Calculations.AllValid
Calculations.field_validity
Calculations.batch_validity
Calculations.BatchLeading
```

## Gridded sweeps

```@docs
Calculations.gridded_lag_sweep!
Calculations.gridded_sweep!
Calculations.gridded_lag_sweep_batch!
Calculations.gridded_sweep_batch!
Calculations.transform_engine
Calculations.device_transform_sweep!
Calculations.device_transform_sweep_batch!
```

## Kernels of the transforms

```@docs
Calculations.isotropic_kernel
Calculations.AbstractMissingLagPolicy
```

## Shapes, culling and the second axis

```@docs
Calculations.AbstractFieldShape
Calculations.CullingPolicy
Calculations.AutoCulling
Calculations.AlwaysCulling
Calculations.NoCulling
Calculations.CellGrid
Calculations.AbstractSecondAxisSource
Calculations.InvariantValueAxis
Calculations.SeparationAngleAxis
Calculations.axis_bounds
```

## The polynomial contract

```@docs
StructureFunctionTypes.is_polynomial_operator
StructureFunctionTypes.moment_contract
StructureFunctionTypes.SymmetricMoments
StructureFunctionTypes.symmetric_indices
StructureFunctionTypes.order
StructureFunctionTypes.scalar_order
```

## Digitize plans

```@docs
digitize_plan
BucketedBinEdges
BucketCell
LinearCells
Log2Cells
LogTableBinEdges
AbstractSquaredDigitizePlan
squared_digitize_plan
squared_digitize
```
