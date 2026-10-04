# GPU Joint 2D Route Notes

This note documents the current distance × value GPU route for a single structure-function operator.
The public entry point is:

```julia
calculate_structure_function(sf_type, x, u, distance_bins, value_bins; backend=GPUBackend(...))
```

Supported inputs follow the package shape contract: `x` and `u` are arrays with spatial or velocity
dimension in `size(_, 1)` and point count in `size(_, 2)`. The joint kernels take any coordinate
width and field width.

## Routes

Eligible bin grids use tiled upper-triangle pair traversal with block-local histograms flushed into
the output. Larger grids use explicit global-atomic kernels. The route is selected from bin shape
and the device's shared-memory budget.

Kernels count in a 32-bit local counter when the call's worst-case pair count fits and widen at the
per-bin flush. The count type of the result is the positional argument after the bins, as in
`calculate_structure_function(sf_type, x, u, distance_bins, value_bins, UInt64; backend=GPUBackend(...))`.
Results stay on the device; `to_host(result)` copies them to the host.

## Workspace

Use `GPUSFWorkspace(backend, distance_bins, value_bins; kind=:joint2d)` to reuse device buffers and
compiled route state across repeated calls. The workspace also accepts `joint2d_compile_cells` for
shared-memory compile width tuning:

```julia
GPUSFWorkspace(backend, dist_bins, value_bins; kind=:joint2d)
GPUSFWorkspace(backend, dist_bins, value_bins; kind=:joint2d,
               joint2d_compile_cells=joint2d_smem_max(backend, W, F, XT, OT, CT))
```

Helpers:

- `joint2d_smem_max(backend, W, F, XT, OT, CT)`
- `joint2d_smem_exact(n_dist, n_val)`
- `joint2d_smem_align256(n_dist, n_val)`

## Tests

CPU-runnable parity and routing tests live in:

- `test/test_gpu_parity.jl`
- `test/test_2d_binning.jl`
- `test/test_gpu_joint2d_smem.jl`

CUDA parity lives in `gpu/test_cuda_parity.jl` and runs on a CUDA device.
