# Batch dimension policy (trailing axes)

Design notes for structure functions over **trailing batch dimensions** — time, depth,
ensemble members, etc. — without repeating SP2D / typeplane / joint-2D histogram routing
(that stays in [`SP2D_HTP_EJ.md`](SP2D_HTP_EJ.md) and [`SP2DPrivPolicy.jl`](../ext/SP2DPrivPolicy.jl)).

**Status:** Production batch fast paths integrated in `src/BatchGeometry.jl`, `src/BatchCPU.jl`,
`ext/BatchTiledKernels.jl`, `ext/BatchLaunch.jl`, `ext/BatchGPUDispatch.jl`. Parity matrix:
`test/test_batch_matrix.jl` (KA.CPU rows 1–7). GPU benchmarks: `gpu/benchmark_batch_matrix.jl`
(user SLURM). Prototype kernels in `gpu/batch_prototypes/` are legacy harness only.

**Related:** [`batch_prototypes/README.md`](batch_prototypes/README.md), plan
`.cursor/plans/full_batch_fast_paths_3e80f2a8.plan.md`

---

## 1. Package contract (what “batch” means)

StructureFunctions does not name trailing dims. Routing is **rank + shape only**.

| Case | `x` | `u` | Output `sums` / `counts` |
|------|-----|-----|--------------------------|
| Snapshot | `(N_dims, N)` | `(N_dims, N)` | `(NB,)` or `(hist…)` |
| **Fixed geometry** | `(N_dims, N)` | `(N_dims, N, batch…)` | `(NB or hist…, batch…)` |
| **Varying geometry** | `(N_dims, N, batch…)` | same shape as `x` | `(NB or hist…, batch…)` |

`batch…` is any trailing shape, e.g. `(8064,)`, `(40, 8064)`, `(80000,)`.

**Goal:** one pair/distance loop per memory slab, with **vectorized inner work over batch
strips** — not `∏ batch_dims` full O(N²) kernel relaunches.

---

## 2. Baseline today (what we beat)

| Backend | Batched behavior |
|---------|------------------|
| GPU | `*_slices!()` — loop linear batch index, full `gpu_calculate_structure_function` per slice |
| CPU | manual loop, same |

Cost order: **`B × (one snapshot SF)`** — repeats tile schedule, staging, and (for fixed-x)
geometry work `B` times.

Benchmark label: **`gpu_production_slice`** in `gpu/benchmark_batch_prototypes.jl`.

---

## 3. Two geometry cases (different batch algorithms)

### 3.1 Case A — fixed `x` (matrix + batched `u`)

Positions identical for all batch indices; only `u` varies.

```text
for each tile / pair (i, j):
    geometry ← f(x[:, i], x[:, j])          # ONCE per pair

    for batch strip (vectorized cols 1:strip_w):
        δu, val, bin ← vectorized on u[:, i/j, strip]
        accumulate → out[..., strip]
```

**Win condition:** amortize geometry + (for tiled128) **x smem staging** over `strip_w`
batch indices. Inner loop loads `u` from global memory.

This is the HF-radar motivating case (fixed lat/lon, many times).

### 3.2 Case B — varying `x` (same rank/shape as `u`)

```text
for each tile / pair (i, j):
    for batch strip:
        geometry[strip] ← f(x[:, i, strip], x[:, j, strip])   # per strip element
        δu, val, bin ← vectorized
        accumulate → out[..., strip]
```

**No geometry reuse across batch.** Win is only from **one pair traversal + vectorized
strip** vs `B` separate launches (launch/staging overhead), not from skipping distance work.

Phase 0 must **measure** Case B vs slice baseline; do not assume it wins.

---

## 4. Where data lives (batch-specific)

| Buffer | Location | Shape (batch problem) |
|--------|----------|------------------------|
| Final histogram | **Global VRAM** | `(hist axes…, B)` — always |
| `x` tile (fixed-x) | smem per CUDA block | one snapshot, shared by strip |
| `u` at points `i,j` | **Global** | `(N_dims, N, B)` — loaded per `(pair, batch col)` |
| Block-local hist during pairs | smem | `(hist_cells_per_pass, strip_w)` — **not** `(…, B)` |

**`B` is never staged wholesale in smem.** Strip width `strip_w` is how many batch
indices share one block-local histogram before flush to global.

For **1D SF** (Phase 0 v1): `hist_cells_per_pass = NB` (distance bins only, NB ≤ 64).

For **SP2D / joint 2D later:** `hist_cells_per_pass` comes from existing histogram
policy (e.g. `types_per_pass × n_dist × n_val`). See SP2D docs — this doc only multiplies
by `strip_w` on top.

---

## 5. Strip width (batch smem budget)

```text
strip_w ≤ floor(max_shared_hist_cells / hist_cells_per_pass)
strip_w ≤ BATCH_STRIP_MAX          # compile-time cap, 32 in v1
host_launches T = ceil(B / strip_w)
```

Examples:

| Problem | hist_cells_per_pass | Headroom | strip_w |
|---------|---------------------|----------|---------|
| 1D, NB=20 | 20 | plenty | 32 (v1 default) |
| 1D, NB=64 | 64 | plenty | 32 |
| SP2D 50×52, tpp=2 (future) | 2×50×52 = 5200 | ~full | **1** |

When `strip_w = 1`: smem holds **one batch column** of histogram alongside the non-batch
histogram cells. **Large `B` does not change strip_w** — only `hist_cells_per_pass` does.

If `hist_cells_per_pass > max_shared_hist_cells`: batch cannot use on-chip histogram at
all; inherit **direct / priv slab** from histogram policy (SP2D). That is not a batch
decision — see [`SP2D_HTP_EJ.md`](SP2D_HTP_EJ.md).

---

## 6. Implemented prototypes (Phase 0)

Code: `gpu/batch_prototypes/gpu_tiled_batch.jl` (v1 strip baseline),
`gpu_fused_tiled_batch.jl` (**integration candidate**), `gpu_kernels.jl` (v0 floor).

### 6.0 `gpu_batch_fused_tiled` — integration (Case A)

**Accumulation ladder** (`batch_accum_plan` in `gpu_fused_tiled_batch.jl`):

1. **Block-private partial** `(2·NB, B, n_tile_blocks)` in workspace — smem hist per strip,
   flush to **per-CUDA-block slab** (no cross-block output atomics), **`_batch_fused_merge_priv!`**
   once. Default when `partial_bytes ≤ max_vram` (or soft 60 GiB budget if `max_vram=0`).
2. **Geom cache** `(3, 128², n_tile_blocks)` — only if **remaining** budget after partial +
   output; large `N` recompute geom per strip (not a silent “off switch” — 240 GiB cache is
   never viable).
3. **`batch_slab_ranges`** — when partial does not fit (`max_vram` on L40 / Metal); host
   sub-slabs over `B` only.

Set **`max_vram`** explicitly on smaller GPUs. Example: L40 48 GiB →
`max_vram=40*1024^3`.

**Not yet:** warp-level pre-aggregation before privatize; production SP2D typeplane batch.
(or batch sub-slab when VRAM budget requires). Accumulate to **block-private global
VRAM partial** `(hist…, B, n_tile_blocks)`; **merge once** to final output.

- `n_priv = n_tile_blocks = n_tiles × (n_tiles + 1) / 2` with `n_tiles = ceil(N/128)`
- Partial bytes: `n_priv × hist_cells × 2 × B × sizeof(T)` (sums + counts)
- At N=20k, B=8064: **~12,403** tile blocks → **~15 GiB** partials (NB≈20), not toy 1024
- Buffer lifecycle: **allocate once** per call (or max sub-slab size), `fill!` before
  accum, merge; **no per-wave realloc** — all launched blocks write indexed slots in one array
- VRAM helper: `estimate_batch_priv_bytes`; fallback ladder in §8.1

Case B (varying-x): same tile schedule, **geometry inside inner `b`**, single launch
(no `B` async host relaunches).

### 6.1 v1 `gpu_batch_tiled` — **baseline only** (do not integrate)

Production tiled128 fork, 1D linear only. Kept for benchmark comparison.

**Fixed-x (`_batch_tiled128_2d_linear_fixed_x_strip!`):**

1. Host loop: `b_base = 1, 1+strip_w, …` → `T = ceil(B/32)` launches.
2. Per launch, per tile block:
   - smem stage **x** (not u).
   - Pair loop: `dist`, `r̂` once; inner `col = 1:bw` loads `u[:, point, b]` from global;
     block-local smem hist `(NB, bw)`.
   - Flush: `@atomic` global `output[bin, b]` (same pattern as production tile flush).

**Varying-x (`_batch_tiled128_2d_linear_varying_x_slice!`):**

- Host loop **`b = 1:B`** — one launch per batch index.
- Each launch = production single-snapshot tiled128 body for that slice.
- Same launch count order as `B × production`; saves relaunch/staging overhead only.

Uses `StructureFunctionsGPUExt` tile schedule + `_gpu_digitize_linear`. Does **not** use
`GPUPrototypeKernels.jl` for GPU path.

### 6.2 v0 `gpu_batch_v0` — grid-stride floor (not integration candidate)

- Global atomics to `output[bin, b]` every pair update; no smem tile.
- Labeled in benchmark for contrast only.

### 6.3 CPU references

- `cpu_slice_baseline!` — gold: loop batch, `cpu_gold_histogram` per slice.
- `cpu_batch_fixed_x!` / `cpu_batch_varying_x!` — one pair loop, `@simd` strips.

---

## 7. Launch count and when batch actually wins

Let `T = ceil(B / strip_w)`, `tiles` = tiled128 upper-triangle block count (function of N).

| Variant | Host launches | Pair tile traversals | Geometry per pair |
|---------|---------------|----------------------|-------------------|
| **Production slice** | `B` | `B × tiles` | fixed-x: **`B ×`** |
| **v1 fixed-x batch** | `T` | `T × tiles` | fixed-x: **`1 ×`** |
| **v1 varying-x batch** | `B` | `B × tiles` | **`B ×`** |

**Fixed-x win** scales roughly like **`B / T ≈ strip_w`** on geometry + x staging (ignoring
u global loads and flush atomics). Measured quick: ~4× vs production at N=512 B=64;
reference extrapolation ~45× at N=20k B=8064 (parity not checked at that scale).

**Varying-x:** no geometry reuse → expect **modest** gain unless launch overhead dominated.
CUDA parity **fails** on quick profile — timings not trustworthy until fixed.

### 7.1 Fused vs unfused batch (design target for SP2D + large B)

When `strip_w = 1` (tight histogram grid), **`T = B`** if batch is only an **outer host loop**
→ **no win** vs `B × sp2d`.

**Required for win at strip_w=1:** inner batch loop **inside** the pair loop (as 1D v1 does
for u), even when `hist_cells_per_pass` forbids `strip_w > 1`:

```text
per pair:
  geometry once                    # fixed-x only
  for b in b_base : b_base+strip_w-1:
    load u (and x if varying)
    accumulate into smem[..., b - b_base + 1]
```

SP2D + batch needs the same fusion, with `hist_cells_per_pass` from SP2D policy — **not built**.

---

## 8. Global memory / contention (batch-specific)

### 8.1 VRAM budgeting (required)

Partials **do not always fit**. Before launch:

```text
partial_bytes = n_priv × hist_cells_per_pass × 2 × B × sizeof(eltype)
```

Also budget: output `(hist…, B)`, inputs, merge temps.

**Strategy ladder:**

1. Fused + full inner `B` + block-private partial + merge — when budget allows
2. **Batch-axis sub-slabs** — split trailing `B` only; one tile schedule per sub-slab
3. Direct global atomics — tight VRAM or wide SP2D hist
4. User pre-split trailing batch; optional `max_vram` wrapper

**GPU terminology:** “block-private” = one partial slab per **CUDA thread block**
(`@index(Group, Linear)` / `bid`), stored in VRAM. Production: ~1 CUDA block per algorithm
128×128 tile. **Not** one partial per histogram bin.

**Hardware vs launch:** A100 runs ~hundreds of blocks concurrently; kernels may **launch**
thousands (`n_tile_blocks`). Partial VRAM scales with **launched** count, not concurrent.

### 8.2 v1 smem path (baseline)

During the pair loop, v1 strip accumulations go to **block-local smem**
`(hist_cells_per_pass, strip_w)`. Flush atomics contend at `output[bin, b]`.

**Toy-validated (grid-stride, SLURM A100, N=20k B=8064 P=4):**

| Variant | Time | Notes |
|---------|------|-------|
| `fused_vram` | 37.0 s | Direct global atomics |
| `fused_vram_private` | 25.4 s (0.69×) | Block-private VRAM + merge (~1.2 GiB toy) |
| `fused_block_smem` | 58.2 s (1.58×) | Strip-outer / pair replay — **rejected** |

Toy uses 1024 CUDA blocks (grid-stride), not production `n_tile_blocks`. Direction only.

v0 uses direct global atomics per update (SP2D-scale hist contention; fine for 1D floor).

---

## 9. Measured results (batch benchmarks only)

### 9.1 Accumulation toy (grid-stride, not tiled128)

Source: user SLURM, A100, `gpu/benchmark_batch_accum_toy.jl`, N=20000 B=8064 P=4.

| Variant | Time | vs fused_vram |
|---------|------|---------------|
| `fused_vram` | 37.0 s | 1.00× |
| `fused_vram_private` | 25.4 s | 0.69× |
| `fused_block_smem` | 58.2 s | 1.58× |

Parity: private vs direct PASS. Absolute seconds not production gates.

### 9.2 Batch prototypes (`benchmark_batch_prototypes.jl`)

Source: user SLURM + KA.CPU, A100 where noted.

| Case | N | B | vs production slice | Parity |
|------|---|---|---------------------|--------|
| fixed-x v1 strip | 512 | 64 | ~4× faster | PASS CUDA |
| fixed-x v1 strip | 20000 | 8064 | ~45× extrap | not run @ scale |
| fixed-x **fused tiled** | — | — | **gate pending SLURM** | KA.CPU |
| varying-x v1 strip (old B launches) | 512 | 64 | ~1.7× | FAIL CUDA (async) |
| varying-x **fused tiled** | 512 | 64 | TBD | KA.CPU ✓ |
| varying-x v0 floor | 512 | 64 | ~5.7× | KA.CPU ✓; CUDA gate |

KA.CPU parity: `test/debug/batch_prototypes.log`.

---

## 10. Open batch work (ordered)

| # | Batch question | Next step |
|---|----------------|-----------|
| B1 | Fused tiled fixed-x @ reference N/B vs slice + v1 strip | SLURM `PROFILE=reference` |
| B2 | Fused tiled varying-x CUDA parity re-check | SLURM `PROFILE=quick` |
| B3 | VRAM sub-slab wrapper @ large B on 16–40 GiB GPU | `max_vram` sweep |
| B4 | SP2D fixed-x inner-`b` when `strip_w=1` | After 1D fused gates |
| B5 | Case B vs slice at production N/B | After B2 |
| B6 | `Calculations.jl` dispatch | After B1–B2 fixed-x gates |

Histogram-shape routing (typeplane vs direct): **unchanged**; consume `hist_cells_per_pass`
from [`SP2DPrivPolicy.jl`](../ext/SP2DPrivPolicy.jl) when batch meets SP2D.

---

## 11. Phase 0 gates (batch-focused)

1. Fixed-x 1D: CUDA parity quick ✓; scaled parity + reference timing with parity spot-check.
2. Varying-x 1D: fix B1 before benchmarks matter.
3. Do not integrate into `Calculations.jl` until fixed-x gates pass.
4. SP2D / joint 2D + batch: separate milestone after B3; do not infer from 1D numbers.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-18 | Initial version (batch + SP2D mixed) |
| 2026-06-18 | Refocus on batch geometry, strip, launches; SP2D deferred to cross-link |
| 2026-06-18 | Toy SLURM table; fused tiled target; v1 demoted; VRAM §8.1; priv partials toy-validated |
