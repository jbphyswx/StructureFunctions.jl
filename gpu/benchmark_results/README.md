# GPU benchmark results for docs/README figures

## Generate `assets_latest.json` (GPU allocation required)

```bash
# From repo root, on SLURM GPU node:
julia --project=gpu gpu/collect_benchmark_assets.jl

# Optional overrides:
N_LIST=4000,8000,16000,20000 T_LIST=100,500,1000,2000 julia --project=gpu gpu/collect_benchmark_assets.jl
SKIP_MICRO=1 julia --project=gpu gpu/collect_benchmark_assets.jl   # faster, skip large-N micro rows
```

## Generate PNGs (no GPU)

```bash
julia --project=docs/generate_assets docs/generate_assets/generate_gpu_figures.jl
```

Commit both `assets_latest.json` and updated files under `docs/src/assets/`.

Problem definition matches [`benchmark/scaling_config.jl`](../benchmark/scaling_config.jl)
(same N/bins/SF as CPU thread scaling in `benchmark/benchmark_scaling.jl`).
