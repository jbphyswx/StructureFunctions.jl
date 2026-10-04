# Pair-kernel cost structure

Per unordered pair `(i,j)` with `D = 2`, single-pass (six invariants):

| stage | work |
|---|---|
| `dx = x_j - x_i`, `r² = dx·dx`, `r = sqrt(r²)` | a few FLOP and one `sqrt` |
| `bin = digitize(r)` | one FMA-based lookup for `LinearBinEdges` |
| `du = u_j - u_i`, `du_L = du·dx/r`, `|du|²` | a few FLOP |
| six moments | a few FLOP |
| accumulate | one atomic add per accumulated moment and one count add |

The arithmetic is small; the atomic adds into the histogram are the dominant per-pair cost.

## Derived moments in the 1D single-pass histogram

The six invariants satisfy

```
T2   = S2 − L2
L1T2 = S3 − L3
```

and a histogram bin is a sum, so per bin `Σ T2 = Σ S2 − Σ L2` and `Σ L1T2 = Σ S3 − Σ L3`. The 1D
single-pass kernels therefore accumulate rows 1, 2, 4, 5 and one shared count, and form rows 3 and 6
once per bin at flush:

- device: `_sf_accum_moments` and `_sf_flush_moment` (`src/Calculations/moment_sets.jl`), used by the
  kernels in `ext/cuda/kernels_1d.jl` and `ext/gpu/kernels_1d_single_pass.jl`;
- CPU: `_sp1d_derive_rows!` (`src/Calculations/serial_single_pass.jl`), which assigns with `=` and is
  idempotent over repeated calls on one buffer.

The identity holds only where all moments share one bin. In single-pass 2D each moment is binned on
the value axis by its own value, so the cell `(t, dbin, vbin)` collects different pairs for each `t`
and `Σ_cell T2 ≠ Σ_cell S2 − Σ_cell L2`; the 2D kernels accumulate all six rows directly.

## Shared-memory sizing

Static `@localmem` is capped by `gpu_static_smem_budget`. A kernel's static staging is sized from
`sizeof` of the element type against that cap, never from a compile-time constant, since a constant
that fits `Float32` can exceed the cap at `Float64`. `_batch_usmem_strip_w` picks the widest field
strip of the fixed-x batch kernel that fits (`_sf_fitting_width`), and 0 when even one does not.
Test both `Float32` and `Float64` when changing a kernel's staging.

## Large bin counts

Single-pass 2D takes the tiled path while `n_dist ≤ SF_GPU_MAX_BINS` and one histogram plane fits
shared memory (`SP2D_HTP_EJ.md`); otherwise the global-atomic kernel digitizes the value axis through
the same `digitize_plan` and runs at any bin count. Large sparse 2D histograms accumulate cancelling
odd moments, so `Float64` is the type to compare against a CPU reference there; a `Float32` CPU run
already differs from `Float64`.

## Comparing kernels

Compare two kernels only when both arms reach the same kernel; confirm the route before attributing
a difference to one variable. Benchmark with the production bin and value-plan types, in `Float32`
and `Float64`, at a batch size that saturates the device, and compare totals of counts before
per-cell differences when judging a value-axis change.
