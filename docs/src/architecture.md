# Implementation overview

A calculation proceeds through validation, preparation, execution, reduction, and result construction.

## Validation and preparation

The public entry point validates array layouts, geometry, bins, output shapes, and count representation. Preparation creates concrete operator and geometry policies, digitizers, pair or lag schedules, and reusable scratch storage. A workspace retains preparation that is unchanged across snapshots.

## Pair execution

Direct methods enumerate each contributing pair once. The geometry supplies separation and a valid pair frame; the operator evaluates the field increments in that frame. Kernels share geometry across batched fields or requested operators and accumulate private or cooperative partial histograms.

Culling supplies a conservative subset of candidate pair blocks. The distance-bin test determines the final contributions. Sorted-line polynomial methods instead reduce partner intervals using prefix moments.

## Transform execution

Polynomial gridded methods express increment moments as correlations of field monomials. Transform providers compute those correlations, and lag reductions apply geometry, mask/weight normalization, and output binning. Nonuniform and spherical providers use their specified mode or harmonic representations.

## Reduction and results

CPU tasks, processes, MPI ranks, and GPU workgroups combine partial sums and counts. Unweighted counts represent pairs; weighted counts represent pair mass. Finalization either returns raw accumulators or divides sums by nonzero counts. Empty means contain `NaN`.

Backend extensions implement execution and provider integration. Scientific operator definitions and result semantics are shared. Tests compare the resulting statistics with independent small reference calculations and check backend-specific contracts separately.
