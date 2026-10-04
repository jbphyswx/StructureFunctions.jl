# Getting started

For a point list, store positions and vector values as `(components, points)` matrices. The following example computes a second-order longitudinal structure function using 32 points and a serial CPU backend.

```@example getting_started
using Random: Random
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionSumsAndCounts, midpoints
using ComputationalBackends: SerialBackend

rng = Random.MersenneTwister(42)
x = Random.rand(rng, 2, 32)
u = Random.randn(rng, 2, 32)
bins = range(0.0, 1.5; length=9)
result = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins;
    backend=SerialBackend())
@assert length(result.values) == length(bins) - 1
(midpoints(result.distance), result.values)
```

Each value averages `(δu ⋅ r̂)²` over pairs whose separation lies in `(left, right]`. Empty bins contain `NaN`. The output's separation units match `x`; values have the squared units of `u`.

`T2SFType()` averages the transverse part `‖δu‖² − (δu ⋅ r̂)²` the same way. On a synthetic two-dimensional field:

![Longitudinal and transverse second-order structure functions of one field](assets/sf_long_vs_trans.png)

## Raw sums and counts

Request raw accumulators to combine measurements before averaging: pass the count type and the result type positionally after the bins.

```@example getting_started
raw = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, UInt64, StructureFunctionSumsAndCounts;
    backend=SerialBackend())
@assert sum(raw.counts) == 32 * 31 ÷ 2
(raw.sums, raw.counts)
```

A mutating call adds to the supplied buffers. Clear them explicitly when starting a new independent calculation.

```@example getting_started
sums = copy(raw.sums)
counts = copy(raw.counts)
SFC.calculate_structure_function!(sums, counts, SFT.L2SFType(), x, u, bins;
    backend=SerialBackend())
@assert counts == 2 .* raw.counts
nothing # hide
```

For time series, pass a field with trailing snapshot axes. For repeated calls with fixed preparation, use a workspace as described in [Execution backends](backends.md).
