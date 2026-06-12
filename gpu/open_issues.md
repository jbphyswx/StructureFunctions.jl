# GPU structure functions — open issues

Prototype candidate: **`blockshared_256k_w256`** (~8× kernel speedup vs production ext at
`N = 20k`). **Histogram correctness not verified** — run `diagnose_counts.jl` first.

---

## Must do before operationalizing

### 0. Resolve Σcounts / Float32 counts (blocker)

`diagnose_counts.jl` shows **Int64 in-bin gold ≈ 194M** but **Float32 Σcounts ≈ 152M**
(global + CPU Float32 loop agree — both lossy). blockshared/private ≈ 194M (correct total).

- [ ] Run `diagnose_counts.jl` after changes
- [ ] Production: **UInt32 or Int64 counts** (or block-local merge); drop Float32 bin counters at large N
- [ ] Re-point benchmark parity vs Int64 gold / blockshared, not lossy global Float32

### 1. Integrate blockshared into `StructureFunctionsGPUExt.jl`

- [ ] Move `_pair_from_linear`, `_proto_blockshared_linear!` (rename for production)
- [ ] Replace default linear-bin launch: grid-stride + block shared histogram
- [ ] Default launch params: `nworkers = min(262_144, N(N-1)/2)`, `workgroup_size = 256`
- [ ] Keep `@localmem` size ≥ max bins (`PROTO_MAX_BINS = 64` or derive from edges)
- [ ] Update `gpu/test_cuda_parity.jl` to compare against CPU / blockshared reference
- [ ] Re-run `benchmark_cuda.jl` after integration

### 2. Histogram correctness at large N (blocked on §0)

- [ ] Do **not** integrate blockshared until `diagnose_counts.jl` identifies which path matches CPU gold
- [ ] Consider **`UInt32` counts** if global Float32 atomics are implicated

### 3. Bin-edge coverage on GPU

Prototype is **linear bins only**. Production ext supports log / general via separate
kernels. Need blockshared (or private) variants for:

- [ ] `LogBinEdges` (`_sf_kernel_log!` routing)
- [ ] General monotone edges (`_sf_kernel!` binary search)
- [ ] Parity tests for each bin type on CUDA

### 4. Extension load error (`@Const`)

Precompile of `StructureFunctionsGPUExt` fails with `UndefVarError: @Const` on some
environments. Import `@Const` from KernelAbstractions alongside `@index`, `@atomic`
(see module note at top of ext).

---

## Performance ideas (not yet implemented)

### Host / staging

- [ ] **Device-resident buffers**: skip `Array(x_mat)` when input is already `CuArray`
- [ ] **GPU pad 2D→3D** (`_proto_pad3_kernel!` exists in prototype)
- [ ] Reuse `out_dev` / `cnt_dev` across repeated calls in a time series

### Launch tuning

- [ ] Auto-tune `nworkers` / `workgroup_size` from `CUDA.multi_processor_count()` and N
- [ ] Profile with Nsight: occupancy vs shared-memory size (`NB × 2 × sizeof(FT)`)
- [ ] Sweep at `N ∈ {4k, 10k, 20k, 50k}` — optimal block count may change

### Algorithmic (larger changes)

- [ ] **Tiled pairwise** (matmul-style shared-memory tiles) to cut memory bandwidth —
      NVIDIA forum recommendation for all-pairs; helps when SF compute is cheap vs loads
- [ ] **Per-thread coarsening** before shared atomic (batch several pairs in registers)
- [ ] Cell lists / spatial hashing — only if separation range ≪ domain (not default SF)

### Numeric / parity

- [ ] Relax or document Float32 tolerance for `private_256k` vs blockshared (~388 max
      bin count diff from multi-worker flush order — benign)
- [ ] Optional Float64 `sums` buffer for high-dynamic-range SF values

---

## Benchmark / dev ergonomics

- [ ] Single entry script (avoid stale-module footguns from re-`include`)
- [ ] Document SLURM: benchmarks require user's GPU alloc; agent should not run CUDA
- [ ] Add `N=4000` row to compare vs CPU thread scaling JSON
- [ ] Log results to `gpu/benchmark_results/` for regression tracking

---

## Explicit non-goals (for now)

- Spectral GPU kernels in this folder (moving to `FlowFieldSpectra.jl`)
- Changing chunk policy / zarr write paths
- Supporting `N_bins > PROTO_MAX_BINS` without raising compile-time shared memory

---

## Reference numbers (A100, `N = 20k`, 2D Float32, linear bins, seed 42)

| Variant | Kernel (s) | Σcounts (approx) |
|---------|-------------|------------------|
| Production ext / timing baseline | ~0.097 | ~1.52e8 (lossy) |
| `blockshared_256k_w256` | ~0.012 | ~1.94e8 (reference) |

Re-verify on your hardware after integration; times vary with load and Julia/CUDA versions.
