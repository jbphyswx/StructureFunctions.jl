# Batch prototypes — thin harness over production batch APIs

Phase 0 isolated kernels are **superseded** by production integration in `ext/BatchTiledKernels.jl`
and `ext/BatchLaunch.jl`. This directory remains for historical CUDA timing comparisons only.

## Production parity (preferred)

```bash
cd StructureFunctions.jl
LOG=test/debug/batch_matrix.log
mkdir -p test/debug
julia --project -e 'using Test; include("test/test_batch_matrix.jl")' > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

## Production benchmark matrix

```bash
julia --project gpu/benchmark_batch_matrix.jl
# GPU reference sizes (user SLURM):
# FAST=0 BATCH_N=20000 BATCH_B=8064 julia --project=gpu gpu/benchmark_batch_matrix.jl
```

## What still lives here

| File | Role |
|------|------|
| `BatchPrototypes.jl` | Legacy module; defers to production for new work |
| `harness.jl` | Old variant registry (slice vs prototype kernels) |
| `gpu_fused_tiled_batch.jl` | Reference implementation (migrated to `ext/`) |

New development: extend `ext/BatchTiledKernels.jl` / `Calculations.jl` — not new prototype families.

Accumulation policy: [`../BATCH_ACCUM_POLICY.md`](../BATCH_ACCUM_POLICY.md)
