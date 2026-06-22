CUDA validation, benchmark entry points, and prototype notes for
StructureFunctions.jl GPU work.

Production GPU code lives in `ext/StructureFunctionsGPUExt.jl` and `ext/gpu/`.
General user documentation lives in [`docs/gpu.md`](../docs/gpu.md).

## File Inventory

### Production CUDA Tests

| File | Purpose |
|------|---------|
| `Project.toml`, `Manifest.toml` | CUDA/benchmark environment used inside SLURM allocations. |
| `run.jl` | `include_gpu(...)` helper for pwd-independent script loading. |
| `runtests.jl` | CUDA test entry point; includes CUDA parity and workspace tests. |
| `test_cuda_parity.jl` | CUDA parity for 1D distance bins, joint 2D bins, and six-invariant SP2D. |
| `test_workspace_cuda.jl` | CUDA workspace reuse and slice-batch parity. |
| `smoke_cuda.jl` | Fast manual CUDA smoke test for allocation, launch, download, and public `GPUBackend` API. |

### Maintained Benchmarks And Asset Scripts

| File | Purpose |
|------|---------|
| `benchmark_suite.jl` | Maintained performance-gate runner for 1D, joint2D, SP2D, auxiliary-axis routes, workspace reuse, and structured JSON output. |
| `benchmark_cuda.jl` | Simple production GPU vs threaded CPU timing at one `N`. |
| `benchmark_2d_grid_scaling.jl` | SP2D vs repeated joint2D gate for selected `(n_dist, n_val)`. |
| `benchmark_single_pass_2d_scaling.jl` | Single-pass 2D scaling sweep. |
| `benchmark_value_axis_dispatch.jl` | SP2D value-bin digitize route comparison. |
| `benchmark_workspace.jl` | Workspace reuse timing. |
| `benchmark_slices.jl` | Slice-batch driver timing. |
| `benchmark_batch_matrix.jl` | Current auxiliary-axis batch matrix benchmark. |
| `benchmark_scaling_helpers.jl` | Shared timing helpers for maintained benchmark and asset scripts. |
| `collect_benchmark_assets.jl` | Generates `gpu/benchmark_results/assets_latest.json` for docs/README figures. |
| `collect_multi_gpu_scaling.jl` | Future multi-GPU scaling collector; retained as a maintained placeholder. |
| `plot_cuda_parity.jl` | Optional docs parity figure, run under CUDA allocation. |

### Diagnostics And Profiling Helpers

| File | Purpose |
|------|---------|
| `diagnose_sums.jl`, `diagnose_counts.jl`, `diagnose_reconcile.jl` | Manual correctness diagnostics for sums/counts and route reconciliation. |
| `benchmark_joint2d_diagnose.jl`, `benchmark_joint_value_route_ab.jl` | Focused joint2D route diagnostics. |
| `profile_joint2d.jl`, `profile_joint2d_ncu.jl` | Nsight profiling workloads. |
| `run_nsys_joint2d.sh`, `run_ncu_joint2d.sh`, `diag_ncu_julia.sh`, `list_joint2d_kernel_names.sh` | Nsight wrapper and inspection scripts. |
| `NCU_JULIA.md` | Notes for working Nsight Compute commands on the cluster. |
| `prove_f32_accumulation.jl` | Manual proof/check script for Float32 accumulation behavior. |

### Prototype / Archive Candidates

These files are not production API. Keep them only while they are actively useful for
porting or benchmarking a maintained route; otherwise move them under an archive or
delete them.

| File | Current status |
|------|----------------|
| `archive/GPUPrototypeKernels.jl` | Historical prototype kernel collection. |
| `archive/benchmark_prototypes.jl`, `archive/gpu_full_benchmark.jl`, `archive/benchmark_helpers.jl` | Prototype benchmark harness and generated-report machinery. |
| `archive/benchmark_batch_optimal.jl`, `archive/benchmark_batch_usmem.jl`, `archive/benchmark_batch_diagnose.jl`, `archive/benchmark_batch_profile_fixed_x.jl`, `archive/benchmark_batch_accum_toy.jl`, `archive/benchmark_batch_prototypes.jl` | Batch prototype/diagnostic scripts retained only as historical references. |
| `archive/batch_prototypes/` | Prototype package for batch experiments; not part of production dispatch. |

### Documentation And Generated Outputs

| Path | Purpose |
|------|---------|
| `SP2D_HTP_EJ.md` | Current six-invariant SP2D HTP-EJ strategy document. |
| `GPU_2d_joint_sf_plan.md` | Joint 2D implementation status and follow-up notes. |
| `BATCH_ACCUM_POLICY.md`, `FIXED_GEOMETRY_BATCH_BENCHMARK.md` | Batch design notes; still need cleanup against current dispatch terminology. |
| `GPU_structure_function_prototypes_theory.md` | Historical prototype research notes. |
| `benchmark_results/README.md` | Describes generated benchmark output policy. |
| `benchmark_results/assets_latest.json` | Generated docs asset snapshot. Regenerate intentionally; do not commit profiler dumps or local run logs. |

## CUDA Validation On Slurm

CUDA is not run by default CI for this repository. Before trusting GPU changes, run this
inside a GPU allocation:

```bash
srun --gres=gpu:1 --time=06:00:00 --pty bash
julia --project=gpu -e 'include("gpu/runtests.jl")'
```

Expected result after the array-only public API cleanup:

```text
Test Summary:          | Pass  Total
StructureFunctions GPU |   24     24
GPU tests passed.
```

For a faster manual smoke before the full CUDA testset:

```bash
julia --project=gpu gpu/smoke_cuda.jl
```

## Running Scripts

Start Julia once inside the allocation. Precompile is expensive; do not spawn a fresh
`julia script.jl` process for every benchmark.

```julia
using Pkg: pkgdir
using StructureFunctions: StructureFunctions
include(joinpath(pkgdir(StructureFunctions), "gpu", "run.jl"))

include_gpu("smoke_cuda.jl")
include_gpu("test_cuda_parity.jl")
include_gpu("test_workspace_cuda.jl")
include_gpu("benchmark_suite.jl")
include_gpu("benchmark_cuda.jl")
include_gpu("benchmark_2d_grid_scaling.jl")
include_gpu("benchmark_workspace.jl")
```

Large benchmarks, profiling helpers, and any script that allocates CUDA arrays must run
inside the SLURM allocation. Re-`include` is cheap; restarting Julia is not.

The maintained benchmark command for release checks is:

```bash
julia --project=gpu gpu/benchmark_suite.jl
```

It writes `gpu/benchmark_results/benchmark_suite_latest.json` plus a timestamped copy
and prints these ratios:

- fresh allocation vs workspace reuse;
- `6 * joint2D` vs six-invariant SP2D;
- explicit per-slice loops vs fused shared-position auxiliary axes;
- explicit per-slice loops vs fused varying-position auxiliary axes.

Use `BENCH_BACKEND=kacpu` only as a smoke test that the benchmark still runs. Treat
performance ratios as meaningful only under CUDA allocation and representative `N/BATCH`.

## Current Output Contracts

GPU public inputs follow the same array shape contract as CPU:

- point field: `x::(D,N)`, `u::(D,N)`;
- shared positions: `x::(D,N)`, `u::(D,N, auxiliary...)`;
- varying positions: `x::(D,N, auxiliary...)`, `u::(D,N, auxiliary...)`;
- `D` is `size(x, 1)`, not `ndims(x)`.

Six-invariant single-pass 2D returns rows:

1. `S2 = |delta u|^2`
2. `L2 = delta u_L^2`
3. `T2 = |delta u_T|^2`
4. `S3 = delta u_L |delta u|^2`
5. `L3 = delta u_L^3`
6. `LT2 = delta u_L |delta u_T|^2`

Basis-dependent component diagnostics such as `T3SF` and `L2T1SF` are not part of
the default single-pass output.
