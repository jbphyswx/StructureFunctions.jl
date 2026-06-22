# GPU benchmark results

Generated benchmark JSON lives here. Keep profiler dumps, local logs, and ad hoc
database outputs out of the repository; only commit intentional, reproducible result
snapshots.

## Release benchmark suite

Run inside a GPU allocation:

```bash
julia --project=gpu gpu/benchmark_suite.jl
```

Outputs:

- `benchmark_suite_latest.json`
- `benchmark_suite_yyyymmdd_HHMMSS.json`

The suite reports release-gate timings for:

- 1D distance bins, `D = 2` and `D = 3`;
- joint2D distance × value bins;
- six-invariant SP2D;
- shared-position auxiliary axes;
- varying-position auxiliary axes;
- workspace reuse vs fresh allocation.

`BENCH_BACKEND=kacpu` is a script/API smoke test only. Treat performance gates as
meaningful only for CUDA runs with representative `N`, `BATCH`, and bin counts.

## Docs figure assets

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
