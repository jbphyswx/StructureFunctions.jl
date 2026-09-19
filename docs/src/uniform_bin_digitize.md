# Uniform bin digitize: derivation and fast path

This is the authoritative spec for `searchsortedfirst` on uniformly spaced edges
(`LinearBinEdges`, and log-native digitize after `lx = log(q)`).

## Problem

Sorted edges ``u_i = u_1 + (i-1)\delta`` for ``i = 1,\ldots,n``.
Julia `searchsortedfirst(u, x)` with forward order returns

```math
i^*(x) = \min\{\, i \in \{1,\ldots,n\} : u_i \ge x \,\}.
```

Half-open histogram bins ``(u_i, u_{i+1}]`` use the same index for interior points.

## Exact discrete formula (not `round`)

Solve ``u_1 + (i-1)\delta \ge x``:

```math
i - 1 \ge \frac{x - u_1}{\delta}
\quad\Rightarrow\quad
i^* = \left\lfloor \frac{x - u_1}{\delta} \right\rfloor + 1
     = \left\lceil \frac{x - u_1}{\delta} + 1 \right\rceil.
```

**`round` is the wrong operator** — it answers “nearest integer,” not “smallest ``i`` with ``u_i \ge x``.”

### Why both `floor` and `ceil` appear in conversation

They are the **same identity** for this problem:

```math
\left\lceil t + 1 \right\rceil = \lfloor t \rfloor + 1
\quad\text{where}\quad
t = \frac{x - u_1}{\delta}.
```

Implementation preference: compute ``t`` directly and use **`floor(Int, t) + 1`**, not
`ceil(Int, muladd(x, inv_step, offset))` with `offset = 1 - u_1/\delta`.
That avoids forming ``t+1`` before rounding and matches standard FP binning practice.

## Fast path (one FMA + one correction)

Precompute `inv_step = 1/δ`, `first = u_1`, `step = δ`.

```julia
t   = muladd(x, inv_step, -first * inv_step)   # (x - first) / step
idx = clamp(floor(Int, t) + 1, 1, n)
u   = muladd(eltype(step)(idx - 1), step, first)  # reconstructed u_idx
return u < x ? idx + 1 : idx
```

### The correction is not optional

Floating-point ``t`` and the reconstructed ``u`` are both inexact, so the guess can be off by one
bin in either direction. The single test `u < x ? idx + 1 : idx` is the minimal fix: it compares the
guess against the edge it claims and steps once if the edge is below the query.

Each guess errs in its own direction, so none of them is correct without it. `floor(t) + 1`
amplifies downward error in ``t``; `ceil` biases high; `round` biases low, which is why pairing
`round` with a one-sided `+1` correction can reach zero errors on a sample while still being the
wrong discrete map — it is two errors cancelling, not one answer. `test/test_bin_edges.jl` holds
the shipped form to `searchsortedfirst` on the same edges.

## Log-spaced edges (unified)

Both `LogBinEdges(phys_vec)` and `LogBinEdges_from_log_edges(log_range)` build the same
log grid and digitize via one path:

```julia
searchsortedfirst(log_linear, log(q))  # log_linear = LinearBinEdges(log_edges)
```

`log_edges` is authoritative. Physical edges are `exp(log_edges[i])` for display only
(`getindex`, `physical_edges_vector`) — not used on the digitize hot path.

GPU tiled kernels mirror CPU: `_gpu_digitize_log_spaced(x, …)` = `log(x)` then
`_gpu_digitize_linear` on the cached `log_linear` FMA fields.

## Tests

Fast check (bin edges only, no full `Pkg.test`):

```bash
julia --project=test -e 'include("test/test_bin_edges.jl")'
```

Full `Pkg.test` precompiles the package and runs the entire suite (~minutes).
