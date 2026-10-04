# Six-invariant single-pass 2D (SP2D) — HTP-EJ GPU path

**Code:** `ext/gpu/kernels_2d_single_pass.jl`, `ext/gpu/sp2d_accumulation_strategy.jl`, `ext/gpu/launch.jl`, `ext/gpu/workspace.jl`
**Benchmark:** `gpu/benchmark_2d_grid_scaling.jl`
**Tests:** `test/test_gpu_sp2d_strategy.jl` (`KernelAbstractions.CPU()` parity for `:shared` and `:typeplane`)

This document describes how the six-invariant single-pass 2D histograms are accumulated on a device.

---

## What SP2D computes

For each unordered pair `(i, j)` with `i < j` (the tiled128 schedule shared with the 1D and joint 2D routes):

1. Distance `dist` gives one distance-bin index.
2. Six invariant samples:
   `S2 = |δu|²`, `L2 = δu_L²`, `T2 = |δu_T|²`,
   `S3 = δu_L |δu|²`, `L3 = δu_L³`, and `LT2 = δu_L |δu_T|²`.
3. Each sample gives a value-bin index and accumulates into `sums[t, dist_bin, val_bin]` and counts.

Output shape: `(6, n_dist, n_val)`. One kernel launch covers all six invariants.

Distance and value bins are digitized with the host's `digitize_plan` of the bins, moved to the device.

---

## Accumulation strategies

The strategy is chosen per call by `_sp2d_accumulation_strategy` (`SP2DAccumulationStrategy`) from the device's static shared-memory budget (`gpu_static_smem_budget(gpu_device_caps(backend))`), the coordinate and field widths, and the element types.

| `accum_mode` | When | Pair traversal | Block-end output |
|--------------|------|----------------|------------------|
| `:shared` | The padded `6 · n_dist · n_val` histogram fits | one tile loop | on-chip flush into `(6, n_dist, n_val)` with `@atomic` |
| `:typeplane` | One padded `n_dist × n_val` plane fits, not all six | `n_type_passes` tile loops of `types_per_pass` planes each | the same flush after each pass |

When `n_dist` exceeds `SF_GPU_MAX_BINS` or not even one plane fits, the call takes the global-atomic kernel: one work item per ordered pair, any width, any bins.

On CUDA, `gpu_native_2d_plan` is consulted first; the portable strategy kernels run when it returns `nothing`.

The shared histogram's value axis has an odd row stride so that lanes differing only in the distance bin fall in different shared-memory banks; the padding counts toward the budget. The compile-time `@localmem` width is the cells the configuration needs, rounded up to a multiple of `SP2D_COMPILE_CELL_QUANTUM` so nearby configurations share a compiled kernel, and capped at the widest histogram the kernel fits.

---

## Workspace

`GPUSFWorkspace(...; kind=:single_pass_2d)` holds the device digitizers for the distance and value bins and the staged inputs and cull memo shared with the other kinds (see `docs/src/gpu.md`). The accumulation strategy is chosen at launch from the call's element types.

---

## Benchmark gate (`benchmark_2d_grid_scaling.jl`)

Compares the end-to-end single-pass 2D call against six separate single-type joint 2D runs, for selected `(n_dist, n_val)`. The joint reference uses one `L2SFType` and `InfPaddedBinEdges` value edges.

---

## Cost per pair

| Work | `joint_2d` | `sp1d` (6 invariants) | `sp2d` (6 invariants) |
|------|------------|------------------|------------------|
| Distance digitize | 1 | 1 | 1 |
| Value digitize | 1 | 0 | 6 (one per invariant) |
| SF values computed | 1 | 6 | 6 |
| Histogram cells touched | 1 | 6 (1D bins) | 6 (2D bins) |
| Tile traversals | 1 | 1 | `n_type_passes` |

In `:typeplane` mode the tile schedule is replayed `n_type_passes` times with a `@synchronize` between the zero, pair and flush phases. The on-chip flush adds into the global output with `@atomic` once per block and cell.
