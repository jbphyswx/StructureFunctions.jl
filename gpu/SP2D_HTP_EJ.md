# Eight-type single-pass 2D (SP2D) — HTP-EJ GPU path

**Code:** `ext/TiledSinglePass2DPrivKernels.jl`, `ext/SP2DPrivPolicy.jl`, `ext/SP2DPrivLaunch.jl`  
**Benchmark:** `gpu/benchmark_2d_grid_scaling.jl` (run on GPU allocation)  
**Tests:** `test/test_gpu_sp2d_priv.jl` (KA.CPU parity for `:shared`, `:typeplane`, `:direct`)

This document describes how production **eight-type single-pass 2D** histograms work on GPU,
why the design looks the way it does, and what is *not* solved yet.

---

## What SP2D computes

For each unordered pair `(i, j)` with `i < j` (same tiled128 schedule as 1D / joint 2D):

1. Distance `dist` → **one** distance-bin index.
2. Eight structure-function samples (longitudinal / transverse quadratic and mixed types).
3. Each sample → **value-bin** index → accumulate into `sums[t, dist_bin, val_bin]` and counts.

Output shape: `(8, n_dist, n_val)`. One kernel launch covers all eight types (vs eight separate `joint_2d` passes).

Production value routing uses typed digitize plans (`GPUValueDigitizePlan`: linear, log, Inf-padded, per-column, etc.).

---

## Two algorithms (three `accum_mode` labels)

Policy is frozen per workspace in [`SP2DPrivConfig`](../ext/SP2DPrivPolicy.jl) from the **48 KiB default** shared-memory budget (`SF_GPU_SMEM_DEFAULT`). No per-grid tuning.

| `accum_mode` | When | Pair traversal | Block-end output |
|--------------|------|----------------|------------------|
| `:shared` | Full `C = 8·n_dist·n_val` fits in smem | **1×** tile loop | On-chip flush → `out_*` (`@atomic`, joint pattern) |
| `:typeplane` | One `n_dist×n_val` plane fits, not all 8 types | **`n_type_passes`** tile loops (`types_per_pass` types per pass) | Same on-chip flush each pass |
| `:direct` | Even one plane does not fit | **1×** tile loop, global atomics to block-private slab | Priv slab + merge kernel → `out_*` |

`:shared` and `:typeplane` are the **same on-chip family** (shared histogram during the pair loop, flush like `joint_2d`). They differ only in how many type planes fit per pass.

`:direct` is the fallback when smem cannot hold even `n_dist × n_val` cells (large bin grids).

`needs_priv_merge == (accum_mode == :direct)` — host routing uses this single boolean (`_launch_sp2d_onchip!` vs `_launch_sp2d_direct_priv!`).

```mermaid
flowchart TD
  cfg["_sp2d_priv_config(n_dist, n_val)"]
  cfg -->|C le max_shared| shared[":shared"]
  cfg -->|plane le max_shared| typeplane[":typeplane"]
  cfg -->|plane gt max_shared| direct[":direct"]
  shared --> onchip["On-chip: shared hist + atomic flush to out_*"]
  typeplane --> onchip
  direct --> priv["Priv slab + merge kernel"]
```

### On-chip flush (current production path for typical grids)

Before 2025 on-chip fix, **all** modes incorrectly routed through a block-private slab and serial merge (~7 ms + ~2 GB memset at `N=20k`), even when shared memory already held the histogram. That made e2e ≈ `8×joint_2d` with no margin.

**Now:** `:shared` / `:typeplane` write directly to `out_sums_dev` / `out_cnts_dev` at block end (mirror `TiledStructureFunctionKernels.jl` joint flush). No priv allocation, no merge.

### Direct path

When planes do not fit in 48 KiB, pairs accumulate with **global atomics** into `(8, n_dist, n_val, n_tile_blocks)` priv slabs, then `_launch_merge_sp2d_priv!` (serial by default; `ENV["SP2D_MERGE"]=parallel` for experiments only).

---

## Host plumbing

| Piece | Role |
|-------|------|
| `GPUSFWorkspace(...; kind=:single_pass_2d)` | Device `out_*`, frozen `sp2d_priv_config`, cached `sp2d_pair_kernel` (typed dist + val plan) |
| `_sp2d_resolve_pair_kernel` | One-time kernel bind per workspace; ephemeral calls resolve once per launch |
| `reset_histogram!` | Zeros `out_*` always; zeros `priv_*` only when `needs_priv_merge` |
| `_ensure_sp2d_priv_bufs!` | Allocates priv slabs only for `:direct` |

Distance bins must be `LinearBinEdges` or `LogBinEdges` for the HTP-EJ tiled path (`Vector` dist edges use non-priv tiled / legacy routes).

---

## Benchmark gate (`benchmark_2d_grid_scaling.jl`)

Compares **end-to-end** `gpu_calculate_structure_functions_single_pass_2d!` against **`8 × joint_2d`** (eight separate single-type 2D runs). This is the production-relevant gate (eight histogram tensors in one API call).

**Asymmetry (intentional):** joint reference uses one `L2SFType` + simple vector value edges; SP2D uses eight types + `InfPaddedBinEdges` + typed value digitize plans. The gate is conservative.

Example A100 (`N=20000`) after on-chip flush:

| Grid | Mode | 8×joint | SP2D e2e | Merge |
|------|------|---------|----------|-------|
| 20×22 | `:shared` | 53 ms | **28 ms** | 0 |
| 50×52 | `:typeplane` 2×4 | 64 ms | **43 ms** | 0 |

Log: `test/debug/sp2d_phase_profile.log`

---

## Why is SP2D still slower than a naive “2× digitize” story?

A common intuition: 2D joint needs **two** digitizations per pair (distance + value); SP2D should be “about twice” a 1D distance-only path, or not much worse than `joint_2d`, when everything fits in shared memory.

**That bound is too optimistic.** Per pair, SP2D actually does:

| Work | `joint_2d` | `sp1d` (8 types) | `sp2d` (8 types) |
|------|------------|------------------|------------------|
| Distance digitize | 1 | 1 | 1 |
| Value digitize | 1 | 0 | **8** (one per SF type) |
| SF values computed | 1 | 8 | 8 |
| Histogram cells touched | 1 | 8 (1D bins) | 8 (2D bins) |
| Tile traversals | 1 | 1 | **`n_type_passes`** (e.g. 4 on 50×52) |

Additional costs not present in single-type joint:

- **Eight value digitizations** (Inf-padded / per-column routes are heavier than one shared linear digitize).
- **Typeplane multi-pass:** when `C = 8·n_dist·n_val` exceeds smem, the tile schedule is replayed `n_type_passes` times with `@synchronize` between zero / pair / flush phases — not amortized to a single pass.
- **Larger on-chip histogram:** static `@localmem` width is `max_shared_cells` (compile-time), not `C`; zero/flush loops scale with active cells.
- **Block-end flush:** on-chip modes use global `@atomic` adds into `(8, n_dist, n_val)` — correct and cheap vs merge, but not free at large `C` and many tile blocks.
- **Host e2e:** staging, `reset_histogram!`, optional `Array(out_sums_dev)` download (~3 ms in recent profiles).

So the fair structural comparisons are:

- vs **`8 × joint_2d`** — current production gate (passed with margin on tested grids).
- vs **`sp1d`** — SP2D adds an entire value axis + 8 value digitizes; ratios of 2–6× are expected, not a bug by themselves.

A tighter *theoretical* target for “2D overhead only” would be **one** `joint_2d` plus ~8× value-digitize work in a fused kernel — not implemented; today we pay full eight-type structure in one tiled kernel.

---

## Future performance work (not scheduled)

Ordered roughly by expected impact vs implementation risk:

1. **Typeplane pass fusion** — reduce `@synchronize` / zero / flush boundaries between `n_type_passes` (KA segment limits apply).
2. **Warp-aggregated shared atomics** — reduce hot-bin contention in shared histogram during pair loop (NVIDIA/CUB-style).
3. **Digitize / accumulate fusion** — compute eight `vals` once; batch or vectorize value-bin lookups; avoid redundant Inf-padded branches where plans allow.
4. **Host hot path** — reuse download buffers in workspace; avoid per-call `Array(out_sums_dev)` if profiling shows it matters.
5. **Optional smem budget** — portable override above 48 KiB (e.g. 96 KiB) with explicit occupancy measurement; not assumed win.
6. **Direct-mode merge** — parallel merge remains comparison-only; serial merge is production default.
7. **Policy cleanup** — optional rename `:shared`/`:typeplane` → single `:onchip` label with `n_type_passes` (cosmetic).

Re-profile after each change with `benchmark_2d_grid_scaling.jl` across several `(n_dist, n_val)` points, not a single production grid.

---

## Related docs

- [GPU_2d_joint_sf_plan.md](GPU_2d_joint_sf_plan.md) — single-type joint 2D tiled path
- [docs/gpu.md](../docs/gpu.md) — workspace, slice batches, testing tiers
- [GPU_structure_function_prototypes_theory.md](GPU_structure_function_prototypes_theory.md) — historical prototype research (1D focus)
