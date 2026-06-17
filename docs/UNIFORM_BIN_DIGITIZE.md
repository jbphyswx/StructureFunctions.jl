# Uniform bin digitize: derivation and fast path

This is the authoritative spec for `searchsortedfirst` on uniformly spaced edges
(`LinearBinEdges`, and log-native digitize after `lx = log(q)`).

## Problem

Sorted edges \(u_i = u_1 + (i-1)\delta\) for \(i = 1,\ldots,n\).  
Julia `searchsortedfirst(u, x)` with forward order returns

\[
i^*(x) = \min\{\, i \in \{1,\ldots,n\} : u_i \ge x \,\}.
\]

Half-open histogram bins \((u_i, u_{i+1}]\) use the same index for interior points.

## Exact discrete formula (not `round`)

Solve \(u_1 + (i-1)\delta \ge x\):

\[
i - 1 \ge \frac{x - u_1}{\delta}
\quad\Rightarrow\quad
i^* = \left\lfloor \frac{x - u_1}{\delta} \right\rfloor + 1
     = \left\lceil \frac{x - u_1}{\delta} + 1 \right\rceil.
\]

**`round` is the wrong operator** — it answers “nearest integer,” not “smallest \(i\) with \(u_i \ge x\).”

### Why both `floor` and `ceil` appear in conversation

They are the **same identity** for this problem:

\[
\left\lceil t + 1 \right\rceil = \lfloor t \rfloor + 1
\quad\text{where}\quad
t = \frac{x - u_1}{\delta}.
\]

Implementation preference: compute \(t\) directly and use **`floor(Int, t) + 1`**, not
`ceil(Int, muladd(x, inv_step, offset))` with `offset = 1 - u_1/\delta`.  
That avoids forming \(t+1\) before rounding and matches standard FP binning practice.

## Fast path (one FMA + one correction)

Precompute `inv_step = 1/δ`, `first = u_1`, `step = δ`.

```julia
t   = muladd(x, inv_step, -first * inv_step)   # (x - first) / step
idx = clamp(floor(Int, t) + 1, 1, n)
u   = muladd(eltype(step)(idx - 1), step, first)  # reconstructed u_idx
return u < x ? idx + 1 : idx
```

### Is there “no correction”?

**No.** The correction is required. Floating-point \(t\) and reconstructed \(u\) are
inexact; the guess can be off by one bin. The single test `u < x ? idx + 1 : idx`
is the minimal fix and yields **0** parity errors vs this spec on the benchmark verify set.

| Guess only | err_lin (F64, N=1000, ~13k verify pts) |
|------------|----------------------------------------|
| `round`, no correction | ~5315 (~41%) |
| `ceil(g)`, no correction | ~422 (~3%) |
| `floor(t)+1`, no correction | ~10548 (~81%) — **downward FP bias, worst** |
| `floor(t)+1` + correction (shipped P5) | **0** |

`round` + the same one-sided correction was a **paired hack** (round biases low;
`+1` fixes undershoot). It happened to reach 0 errors but is not the correct discrete map.
`ceil(g)` + one-sided correction still fails (~422 errors) because ceil biases high.
`floor(t)+1` without correction is worst because `floor` amplifies downward FP error in \(t\).
**Correction is mandatory**; cost ~1.8 ns/query vs guess-only (see benchmark log).

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
