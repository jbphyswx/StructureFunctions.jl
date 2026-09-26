# Bin edges and digitize

A pair's bin has one definition, shared by every bin type, route and backend:

```julia
digitize(r, edges) = searchsortedfirst(edges, r) - 1
```

Bin `k` holds separations in `(edges[k], edges[k+1]]`; `0` is below the first edge and
`length(edges)` above the last. Every fast lookup below returns the index a binary search over
`collect(edges)` returns, for every representable query, the edges themselves included. A `NaN`
falls above every finite edge.

## Uniform edges

`LinearBinEdges(first, last, n)` stores the endpoints, `step = (last - first) / (n - 1)`,
`inv_step = 1 / step` and `n`. A query `first < x ≤ last` is binned by one FMA and a `ceil`:

```julia
t = fma(x, inv_step, -first * inv_step)
i = clamp(unsafe_trunc(Int, ceil(t)) + 1, 2, n)
```

`fma` rounds once, so `t` never decreases as `x` increases, and the queries with `t ≤ k - 1` are all
the values up to some largest one. That value is interior edge `k`: `edges[k]` returns it, found by
galloping out from `fma(k - 1, step, first)` and bisecting. The end edges are `first` and `last`
exactly. With the edges defined this way the lookup is exact by construction and needs no correction
step.

A range passed to `LinearBinEdges` is read for its endpoints and length only; its interior elements
can differ from these edges in the last place. A query of another type is converted with
`RoundUp`, the smallest edge-type value not below it, which leaves every comparison with an edge
unchanged.

## Logarithmic edges

`LogBinEdges(first, last, n)` stores the endpoints and the `LinearBinEdges` grid of their logarithms.
Interior edge `k` is `exp(fma(k - 1, step, log(first)))`, one `exp` per access. A lookup estimates the
index from `log(x)` on the log grid, then compares `x` against the edges themselves and steps until
`edges[i-1] < x ≤ edges[i]`, so membership never depends on how `log` rounds. The constructor checks
that the edges strictly increase.

## Arbitrary and padded edges

`BinEdges(v)` keeps the supplied values and bisects them. `InfPaddedBinEdges(inner)` adds a
`(-∞, first]` and a `(last, ∞)` bin around any inner bin type and keeps the inner lookup, shifted by one.
Infinite endpoints already present are trimmed, so padding is never doubled; a `NaN` falls in the last
bin.

## Per-call plans

A pair kernel receives [`digitize_plan`](@ref StructureFunctions.digitize_plan)`(bins)`, built once per
call. `LinearBinEdges` are their own plan. The edges of a `BinEdges` or a `LogBinEdges` become a
[`BucketedBinEdges`](@ref StructureFunctions.BucketedBinEdges): the finite span is cut into 16 equal
cells per bin, and each cell records the first edge at or above it, how many edges lie in it, and that
edge's value. The cell of `x` comes from one FMA and a clamp, which never decrease as `x` increases, so
every edge in a lower cell is below `x` and every edge in a higher cell above it. A cell of at most one
edge is decided by one comparison with the recorded value; a cell of more bisects its own edges. The
result is exactly the lookup of the edges, whatever their spacing.

## Squared separations

Kernels that have `r²` and not `r` digitize through
[`squared_digitize_plan`](@ref StructureFunctions.squared_digitize_plan). For each edge `e_k` the
plan stores `S_k`, the largest `s` with `sqrt(s) ≤ e_k`, found by stepping from `e_k²`. `sqrt` is
correctly rounded and monotone, so `r² ≤ S_k` exactly when `sqrt(r²) ≤ e_k`, and
`squared_digitize(plan, r²) == digitize(sqrt(r²), edges)` for every `r²`.

| edges | plan | lookup |
|---|---|---|
| `LinearBinEdges` | `SquaredLinearPlan` | `sqrt`, then the uniform lookup |
| `LogBinEdges` | `SquaredLogPlan` | index from a vectorizable `log2(r²)` and one FMA, corrected against `S` |
| other | `SquaredBucketPlan` | the thresholds `S` bucketed; the vectorized half computes the cell, the scalar half reads its record |
| `InfPaddedBinEdges` | `SquaredInfPaddedPlan` | the inner plan, shifted by one |

## Device digitizers

A device kernel receives the per-call plan with its tables moved to the device. The 1-D single-moment
kernels take `LogBinEdges`, bare or padded, as a
[`LogTableBinEdges`](@ref StructureFunctions.LogTableBinEdges) instead: the edges as a table, searched
from the index estimate `⌊a·log2(x) + c⌋ + 2` formed in `Float32` with the device's fast `log2`, then
stepped to the first edge at or above `x`. Either way the edges are computed on the host, so the device
never evaluates an `exp` of its own, and it bins with the same `digitize`: a separation meets the same
comparisons against the same edge values on every backend, and its bin depends only on the separation
the kernel computes. A value axis takes one plan, or a tuple with one plan per invariant.
