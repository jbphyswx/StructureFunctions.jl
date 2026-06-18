# GPU 2D joint structure functions — implementation plan

**Status:** Phase 1–3 done including **tiled128 block-local fast path** for 2D joint SF
(when `n_dist × n_val ≤ SF_GPU_MAX_2D_HIST`). Larger histograms use `(N,N)` global-atomics fallback.
In-place GPU API remains optional follow-up.

**Scope:** `calculate_structure_function(sf_type, x_mat, u_mat, distance_bins, value_bins; backend=GPUBackend(...))` for dense `(2, N_points)` matrix inputs.

**Implemented separately:**

- `calculate_structure_functions_single_pass_2d` / `calculate_structure_functions_single_pass_2d!` on `GPUBackend` — eight `(distance × value)` histograms via **HTP-EJ** tiled128 (`ext/TiledSinglePass2DPrivKernels.jl`). On-chip modes flush like joint 2D; `:direct` uses priv+merge. See [`gpu/SP2D_HTP_EJ.md`](SP2D_HTP_EJ.md).

**Out of scope (unless requested):**

- Tuple `(x_vecs..., u_vecs...)` on GPU (matrix only).
- In-place `calculate_structure_function!(..., value_bins)` on GPU.
- GPU in-place `calculate_structure_functions_single_pass!` (1D eight-tensor path).
- `N_dims = 3` for 2D joint histogram API (1D tiled SF still supports 2D/3D coordinates).

**Count types:** device histogram buffers are always `UInt32` (atomics + shared memory).
`count_eltype` (default `UInt32`) selects the host array type after download only.

---

## 1. CPU reference (must match)

Serial inner loop (`calculate_structure_function_2d_i!`):

1. Upper triangle only: pairs with `j > i`.
2. `distance = metric(x_i, x_j)` → `dist_bin = digitize(distance, distance_bins)`.
3. If `1 ≤ dist_bin < length(distance_bins)` (interior distance bin):
   - `val = sf_type(u_j - u_i, r̂)` → `val_bin = digitize(val, value_bins)`.
   - If `1 ≤ val_bin < length(value_bins)`: `sums_2d[dist_bin, val_bin] += val`, `counts_2d[dist_bin, val_bin] += 1`.

Result type: `StructureFunction2D` with flat `distance_bins` and `value_bins` edge vectors.

---

## 2. API wiring

| Entry | Status |
|-------|--------|
| `_dispatch_execution_backend(::GPUBackend, ..., value_bins)` | → `gpu_calculate_structure_function_2d(...)` |
| `gpu_calculate_structure_function_2d` | Implemented in `StructureFunctionsGPUExt.jl` |
| `_dispatch_execution_backend!(::GPUBackend, ..., value_bins)` | Still throws — Phase 4 |
| `_dispatch_single_pass` / `_dispatch_single_pass_2d` (allocating) | Implemented |
| `_dispatch_single_pass!` on GPU | Still throws — optional |

---

## 3. Kernel design (as shipped)

**Default (eligible bins):** tiled128 block-local flat histogram (`TiledStructureFunction2DKernels.jl`) —
same tile schedule as 1D SF; accumulate in `@localmem`, flush once per cell per block.

**Fallback:** `(N_points, N_points)` global-atomic pair kernels when
`n_dist > SF_GPU_MAX_BINS`, `n_val > SF_GPU_MAX_BINS`, or
`n_dist * n_val > SF_GPU_MAX_2D_HIST` (4096).

### Shared-memory compile width (2025)

Default: exact `n_dist × n_val` `@localmem` cells (best occupancy when `NB2 ≪ 4096`).
Override on [`GPUSFWorkspace`](@ref):

```julia
GPUSFWorkspace(backend, dist_bins, val_bins)  # exact NB2 (default)

GPUSFWorkspace(backend, dist_bins, val_bins;
    joint2d_compile_cells = joint2d_smem_max())  # legacy 4096, one kernel for any grid

GPUSFWorkspace(backend, dist_bins, val_bins;
    joint2d_compile_cells = joint2d_smem_align256(n_dist, n_val))
```

Helpers: `joint2d_smem_max`, `joint2d_smem_exact`, `joint2d_smem_align256` (return `Int`).
Kernels are `@eval`'d on first use per `(dist_route, compile_cells)` and cached on the workspace.
CPU parity: `test/test_gpu_joint2d_smem.jl`.

---

## 4. Implementation phases

### Phase 1 — Minimal correct path ✅

- [x] `gpu_calculate_structure_function_2d` host driver + dispatch hook in GPU ext
- [x] `(N,N)` kernels: `N_dims=2`, linear / log / general distance routes
- [x] KA.CPU parity vs `SerialBackend` (`test/test_gpu_parity.jl`, `test/test_2d_binning.jl`)

### Phase 2 — Bin coverage ✅ (partial)

- [x] Log / general distance bins
- [x] General value bins (device edge vector)
- [ ] `N_dims = 3` for joint SF API (deferred; 1D GPU tiled path covers 3D coords)
- [x] CUDA parity sections in `gpu/test_cuda_parity.jl`

### Phase 3 — CUDA + performance (mostly done)

- [x] `gpu/test_cuda_parity.jl` — 1D linear/log, joint 2D ×2, single_pass_2d
- [x] Tiled128 2D joint kernels (`TiledStructureFunction2DKernels.jl`) + automatic routing
- [x] HTP-EJ SP2D on-chip flush (no priv+merge for `:shared`/`:typeplane`); A100 gate `sp2d < 8×joint` on 20×22 and 50×52 — see `gpu/benchmark_2d_grid_scaling.jl`, `gpu/SP2D_HTP_EJ.md`
- [ ] Further SP2D perf (typeplane sync, digitize fusion) — documented in `SP2D_HTP_EJ.md` § future work
- [ ] Profile on A100; tune `workgroup_size` / tile size
- [x] Document max bin grid + `count_eltype` policy in GPU ext docstrings

### Phase 4 — In-place API (optional)

- [ ] `gpu_calculate_structure_function_2d!` + `_dispatch_execution_backend!` for `GPUBackend`
- [ ] GPU `_dispatch_single_pass!`
- [ ] Tests mirroring `test_inplace.jl` 2D cases

---

## 5. Tests

| Test | Status |
|------|--------|
| KA.CPU vs Serial (1D + 2D joint + single_pass_2d) | In `test/runtests.jl` |
| CUDA vs CPU on SLURM | `gpu/test_cuda_parity.jl` (manual / cluster) |
| Max bins exceeded | `ArgumentError` on 1D tiled path (`SF_GPU_MAX_BINS`) |

---

## 6. Non-goals / policy

- **No fallback** to `SerialBackend` when user passes `GPUBackend`.
- **No tuple** `(x, u)` on GPU.
- Device histograms **`UInt32`**; **`count_eltype`** is host-only.
