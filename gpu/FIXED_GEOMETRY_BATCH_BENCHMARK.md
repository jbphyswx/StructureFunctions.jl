# Fixed-geometry batch SF — Phase 0 decision record

**Status:** Phase 0 implementation complete on KA.CPU. **CUDA reference gates pending user SLURM run.**

Last updated: 2026-06-18

---

## Integration winner (frozen for Phase 1 planning)

| Geometry | Kernel | Schedule | Accumulation |
|----------|--------|----------|--------------|
| **Case A fixed-x** | `gpu_batch_fused_tiled` | One tiled128 tile schedule | Inner `b`; block-private VRAM partial + merge |
| **Case B varying-x** | `gpu_batch_fused_tiled` | One tiled128 tile schedule | Inner `b`; geometry per `(pair,b)`; direct global atomics |

**Not integrated:** v1 `gpu_batch_strip_host` (fixed-x host strip `ceil(B/32)` relaunches).

---

## Parity (KA.CPU, N=48, batch=(8,2))

Source: `test/debug/batch_prototypes.log`

| Variant | fixed-x | varying-x |
|---------|---------|-----------|
| `gpu_batch_fused` | PASS | PASS |
| `gpu_batch_strip_host` | PASS | n/a |
| `gpu_batch_tiled_varying` (→ fused) | n/a | PASS |
| `gpu_batch_v0` | PASS | PASS |

---

## Accumulation toy (grid-stride, direction only)

Source: user SLURM A100, `gpu/benchmark_batch_accum_toy.jl`, N=20000 B=8064 P=4

| Variant | Time | vs direct global |
|---------|------|------------------|
| `fused_vram` | 37.0 s | 1.00× |
| `fused_vram_private` | 25.4 s | **0.69×** |
| `fused_block_smem` | 58.2 s | 1.58× (rejected) |

Production tiled128 at N=20k: **n_tile_blocks ≈ 12,403** → ~**15 GiB** partials at B=8064 (NB≈20), not toy 1024 blocks.

---

## VRAM policy

- Helper: `BatchPrototypes.estimate_batch_priv_bytes(N, B, NB, FT)`
- Sub-slabs: `batch_slab_ranges(B, max_vram, N, NB, FT)`; benchmark `MAX_VRAM` env
- Partial buffer: **allocate once** per call/slab, `fill!`, merge; reuse across timed repeats

---

## Performance gates (pending SLURM)

Run on GPU node:

```bash
cd StructureFunctions.jl
PROFILE=quick julia --project=gpu gpu/benchmark_batch_prototypes.jl

PROFILE=reference N=20000 BATCH=8064 julia --project=gpu gpu/benchmark_batch_prototypes.jl
```

**Required before Phase 1:**

1. CUDA parity PASS for `gpu_batch_fused` vs `cpu_slice` at `PROFILE=quick` (both geometries)
2. `gpu_batch_fused` fixed-x faster than extrapolated `gpu_production_slice` at reference N/B
3. Report `partial_bytes` line from benchmark header

---

## Baseline timings (historical — v1 strip, quick profile)

| Case | N | B | vs production slice | CUDA parity |
|------|---|---|---------------------|-------------|
| fixed-x strip | 512 | 64 | ~4× | PASS |
| varying-x strip (old B launches) | 512 | 64 | ~1.7× | FAIL CUDA (superseded by fused) |

---

## Phase 1 spec (when gates pass)

1. Promote `gpu/batch_prototypes/gpu_fused_tiled_batch.jl` → `ext/FixedGeometryBatchKernels.jl` (or merge into GPU ext)
2. Wire `Calculations.jl` AbstractArray dispatch (rank/shape only)
3. Export `max_vram` memory budget kw on public API
4. Do **not** wire v1 host strip path

---

## Open after Phase 0

- SP2D + batch inner-`b` when `strip_w=1`
- CUDA re-validation of `gpu_batch_v0` varying-x (floor only)
- Reference-scale fixed-x parity spot-check (scaled profile)
