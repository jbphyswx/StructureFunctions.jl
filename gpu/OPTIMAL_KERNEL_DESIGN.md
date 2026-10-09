# GPU pair-histogram kernel design

How the device pair-histogram kernels are organized. Code: `ext/gpu/` (portable KernelAbstractions
kernels, `StructureFunctionsKernelAbstractionsExt`) and `ext/cuda/` (CUDA-specific kernels,
`StructureFunctionsCUDAExt`, loaded by `using KernelAbstractions, CUDA`).

## 1. The computation

For every unordered pair `(i,j)`, `i<j`, of `N` points with positions `x` and velocities `u`:

```
dx   = x_j - x_i              # separation
r    = |dx|;  r̂ = dx / r      # distance + unit vector
du   = u_j - u_i              # velocity difference
du_L = du · r̂                 # longitudinal
du_norm2 = du·du ;  du_L2 = du_L²
du_T2    = du_norm2 - du_L2   # transverse²
```

Each pair accumulates into the distance bin `b = digitize(r)`:
- **individual**: one moment of the chosen `sf_type`;
- **single-pass**: the six invariants `{du_norm2, du_L2, du_T2, du_L·du_norm2, du_L·du_L2, du_L·du_T2}`.

Output histograms:
- **1D**: `(NMOM, NB)` — distance bins only.
- **2D joint**: `(NMOM, n_dist, n_val)` — distance × value bins.

A batch adds a trailing `B` axis:
- **fixed-x**: positions shared across the batch, only `u` varies; each pair's geometry and bin are formed once for all `B`.
- **varying-x**: `x` and `u` both vary; `B` independent problems.

## 2. Kernel structure

Each kernel is an N-body tiling fused with a histogram scatter:

- Points are staged in shared-memory tiles; one workgroup handles one upper-triangular tile pair.
  Thread `t` owns tile point `t` in registers and loops over the staged partner tile, so all lanes
  read the same shared entry at each step. On a diagonal tile pair a thread loops over `j > i`.
- A tile-pair work list from the cell grid (`AutoCulling`, `AlwaysCulling`) skips tile pairs whose
  minimum separation exceeds the largest finite bin edge.
- The histogram is private to the block in shared memory and added into the global output with one
  atomic per cell at block end. Counts live on chip beside the sums; an unweighted call counts in
  `UInt32` while the call's pair count fits it.
- Bins are digitized with the host's `digitize_plan` of the bins, moved to the device.
- Kernels specialize at compile time on coordinate width `W`, field width `F`, moment count and
  tile size through `Val` parameters.

## 3. Regimes

The kernel a call runs depends on its regime.

| regime | kernel |
|--------|--------|
| 1D, `NB` up to the compiled bin cap (`CU_MAX_BINS`), CUDA | `_cuda_sf_1d_kernel!` with replicated shared histograms |
| joint 2D and single-pass 2D, CUDA | `_cuda_sf_2d_kernel!` with the histogram planes in dynamic shared memory; `_cuda_sf_2d_global_kernel!` adds straight into the output when no plane fits |
| any other backend, or a call without a native plan | the portable tiled kernels of `ext/gpu/` |

On CUDA a 1D call over its own positions takes its plan (tile size, histogram replicas) from a formula over the
device's capabilities and the call's types and sizes (`CUDA1DRule`), stepped down until it fits. The 2D kernels
and 1D batches over shared positions (`_cuda_sf_1d_strip_kernel!`, a block per tile pair and strip of slices)
choose through `CUDAChoice`: a call samples the share of its pairs in range (`gpu_in_range_fraction`), and the
second call of each class (pair evaluations, share band, strip width) times the candidate plans for that share once
and keeps the fastest.

Replicas of a shared 1D histogram are full independent histograms summed at the block-end flush; a
thread's replica is `(lid - 1) % R + 1`, and the bin index is never offset.

The CUDA 2D kernel sets the function's maximum dynamic shared size to the plan's dynamic bytes at
launch. KernelAbstractions' `@localmem` is static only, so only the CUDA kernels use dynamic shared
memory.

## 4. Memory layouts

- `x`, `u`: `(D, N)` non-batch; `(D, N, B)` batch.
- Outputs: 1D `(NMOM, NB[, B])`; 2D `(NMOM, n_dist, n_val[, B])`.
- The value axis of a shared 2D histogram has an odd row stride so lanes differing only in the
  distance bin fall in different banks.

## 5. Compile pitfalls

`KernelAbstractions.CPU()` compiles none of the device code; only a CUDA compile rejects these.

- `threadIdx().x`, `blockIdx().x` and `blockDim().x` return `Int32`; convert to `Int` at the top of
  the kernel, or device helpers annotated `::Int` throw a method error.
- `zero(eltype(localmem))` and `zero(::DataType)` do not constant-fold; use `zero(FT)` with `FT` a
  concrete type parameter.
- Loops that read, write or atomically update a shared array passed as a function argument fail to
  compile; write them inline in the kernel body. Single-element `@inline` loads are fine.
- Kernel arguments must be bits types; a field of type `Any` raises a `KernelError`, so type the
  field and add an `Adapt.adapt_structure` rule.

## 6. Validation scripts

Each is a `gpu/` script with a `run_*.sh` wrapper where one exists.

- `gpu/test_cuda_2d_parity.jl` — 2D kernel parity, CUDA against `KernelAbstractions.CPU()`.
- `gpu/test_cuda_1d_parity.jl` — 1D kernel parity.
- `gpu/test_cuda_batch_contract.jl`, `gpu/test_cuda_widths.jl`, `gpu/test_cuda_pair_weights.jl` — public-API parity
  against the serial CPU.

## 7. References

KernelAbstractions.jl documentation; Gómez-Luna et al. 2013 (histogram replication); GPU Gems 3
ch. 31 (N-body tiling); CADISHI, arXiv:1808.01478 (parallel pair-distance histograms with a
diagonal/off-diagonal split).
