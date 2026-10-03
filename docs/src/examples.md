# Recipes

These examples use small fixtures and a serial backend. Executable scripts for parallel and GPU calculations are listed in the repository's `examples/README.md`.

## Six invariants in one pass

```@example recipes
using Random: Random
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: SerialBackend
rng = Random.MersenneTwister(12)
x = Random.rand(rng, 2, 32)
u = Random.randn(rng, 2, 32)
bins = range(0.0, 1.5; length=7)
result = SFC.calculate_structure_functions_single_pass(x, u, bins;
    backend=SerialBackend())
propertynames(result)
```

The six rows represent `S2`, `L2`, `T2`, `S3`, `L3`, and `L1T2`. Their definitions and the derived Helmholtz quantities are documented in the [operator reference](api/operators.md).

## Resolve separation and increment value

```@example recipes
value_bins = range(-20.0, 20.0; length=9)
joint = SFC.calculate_structure_function(SFT.L3SFType(), x, u, bins, value_bins;
    backend=SerialBackend())
@assert size(joint.counts) == (6, 8)
size(joint.sums)
```

A value histogram can exclude pairs whose operator values lie outside its edges. Include appropriate end intervals when all values must be retained.

## Shared positions across snapshots

```@example recipes
snapshots = cat(u, 2 .* u; dims=3)
batched = SFC.calculate_structure_function(SFT.S2SFType(), x, snapshots, bins;
    backend=SerialBackend())
@assert size(batched.values) == (6, 2)
batched.values
```

Use [data layouts](data.md) for varying positions or additional batch axes. [Spectra, fluxes, and fitting](spectra.md) covers analysis of the resulting statistics.
