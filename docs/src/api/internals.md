```@meta
CurrentModule = StructureFunctions
```

# Internals

Types and functions that are not exported but that the exported entries name: the schedules a grid
resolves to, the lag transports, the field shapes, culling, the second histogram axis, the
polynomial contract of the operators and the squared-distance digitizing.

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
Calculations.lag_transport
Calculations.NoWeights
Calculations.AllValid
Calculations.field_validity
Calculations.BatchLeading
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

## Squared-distance digitizing

```@docs
AbstractSquaredDigitizePlan
squared_digitize_plan
squared_digitize
```
