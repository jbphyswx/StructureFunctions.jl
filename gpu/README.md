Experimental CUDA kernels and benchmarks for structure-function evaluation.
**Not wired into the main package yet** — production code still lives in
`ext/StructureFunctionsGPUExt.jl` (global-atomic `(N,N)` launch).

**Full research writeup (math, correctness proofs, all variants, A100 timings):**
[`GPU_structure_function_prototypes_theory.md`](GPU_structure_function_prototypes_theory.md)

Run benchmarks inside a GPU SLURM allocation. **Start Julia once** (precompile is expensive;
do not spawn a fresh `julia … script.jl` process for every run).

**Login node:** smoke scripts only (`diagnose_sums.jl` defaults to `N=500`). Do **not** run
`N=20000` benchmarks or GPU work outside your allocation — SLURM may terminate the process.

**Julia:** single-threaded (`julia --project=gpu` — no `-t`).

### Load scripts (pwd-independent)

Load once per REPL session:

```julia
using Pkg: pkgdir
using StructureFunctions: StructureFunctions
include(joinpath(pkgdir(StructureFunctions), "gpu", "run.jl"))
```

Then run any gpu script regardless of `pwd`:

```julia
include_gpu("diagnose_sums.jl")        # CPU smoke (N=500 default)
include_gpu("diagnose_reconcile.jl")   # SLURM: Float64 reference + per-worker diff
include_gpu("diagnose_counts.jl")      # GPU counts — SLURM only
include_gpu("benchmark_prototypes.jl")
include_gpu("benchmark_cuda.jl")
include_gpu("gpu_full_benchmark.jl")
```

Large benchmarks (e.g. `ENV["N"] = "20000"`) only inside your SLURM allocation.

### Absolute path (alternative)

```julia
SF_REPO = "/path/to/StructureFunctions.jl"
include(joinpath(SF_REPO, "gpu", "run.jl"))
include_gpu("benchmark_prototypes.jl")
```

Examples:

```julia
ENV["N"] = "20000"
include(joinpath(SF_REPO, "gpu", "benchmark_prototypes.jl"))

ENV["SKIP_CPU"] = "1"
include(joinpath(SF_REPO, "gpu", "gpu_full_benchmark.jl"))
```

`-t 2` matches your SLURM `--cpus-per-task`. Re-`include` is cheap; restarting `julia` is not.

| File | Role |
|------|------|
| **`GPU_structure_function_prototypes_theory.md`** | **Hand-written theory + benchmark log (read this first)** |
| `GPUPrototypeKernels.jl` | Prototype kernels + launch helpers |
| `benchmark_prototypes.jl` | Single-N timing table, parity checks, vs production ext |
| `gpu_full_benchmark.jl` | **Full sweep** (multi-N, all variants, JSON + markdown report) |
| `benchmark_helpers.jl` | Shared timing/parity helpers for benchmark scripts |
| `diagnose_counts.jl` | **CPU gold vs GPU paths** — run before trusting any kernel |
| `benchmark_cuda.jl` | Simple production-ext vs CPU threading comparison |
| `GPU_timings_and_theory.md` | Auto-generated report from `gpu_full_benchmark.jl` |
| `runtests.jl` / `test_cuda_parity.jl` | Parity smoke tests |
| `open_issues.md` | Follow-ups before operationalizing |

---

## What we are computing

For `N` points, all unordered pairs `(i, j)` with `i < j` (~`N(N-1)/2` pairs):

1. Separation `dist = ‖x_j - x_i‖`
2. Bin index from `dist` (linear / log / general — prototype: **linear only**)
3. Structure-function sample `val = sf_type(u_j - u_i, r̂)`
4. Accumulate into histogram: `sums[bin] += val`, `counts[bin] += 1`

The output is a **histogram with ~20 bins**, not `N²` outputs.

---

## Why the production kernel is slow

`StructureFunctionsGPUExt.jl` today launches `ndrange = (N, N)` and, for each valid
pair, does per-pair global `@atomic` updates into ~20 bin slots — ~**200M contended
atomics** at `N = 20k` (~**0.097 s** on A100).

**Correctness (N=20k diagnosis):** global Float32 per-pair atomics **lose ~42M** counts
(Σcounts ≈ 152M vs true in-bin ≈ 194M) when hot bins exceed Float32 integer precision.
**UInt32 global** (`baseline_linear_global_u32`) fixes counts with the same slow algorithm.
**blockshared** (~8× faster) also correct via chunked merge.
Run `benchmark_prototypes.jl` for timed Float32 vs UInt32 vs blockshared rows.

Production fix: UInt32 counts and/or block-local histogram — not keep global Float32 atomics.

---

## Candidate fast path: `blockshared_256k_w256`

**Status:** fastest **proven** prototype at `N = 20k` (~**0.012 s** kernel). New candidates
below target memory bandwidth / atomic overhead — **benchmark in SLURM** before claiming faster.

### Why 0.012 s is not the end

The current kernel still, for **every pair**:

1. Decodes `k → (i,j)` via **binary search** (15+ steps)
2. Loads `x[i], x[j], u[i], u[j]` from **global memory at random indices** (no reuse)

The jump **0.097 → 0.012** came from **block-local histograms** (fewer global atomics). The next
jump is **tiling** (reuse loaded points) — standard for GPU pair histograms:

- [CADISHI](https://doi.org/10.1016/j.cpc.2018.10.018) — tile particles into shared memory; **~40× vs CPU** on GPU
- [Pitaksirianan et al. DAPD 2019](https://cse.usf.edu/~tuy/pub/DAPD19.pdf) — 2-body statistics tiling + warp privatization

### Tiled sweep (`prototype_variants` → 17 configs)

All use CADISHI-style upper-triangle tile blocking, **UInt32** counts, and launch
`ndrange = n_tile_blocks × workgroup_size`. Names encode flags:

| Suffix | Meaning |
|--------|---------|
| `tiled64` / `tiled128` | Tile edge length (points per tile side) |
| `regpriv` | Register partial histogram per thread before shared merge (**blockshared/private only**; tiled `regpriv` uses shared atomics — `MVector` spills on CUDA in tile kernels) |
| `nosqrt` | Bin from `dist²` via `_gpu_digitize_linear_sq` (sqrt only for `r̂`) |
| `2d` | 2D distance / `r̂` on `(x,y)` only — **not parity vs 3D-padded CPU gold** |
| `w256` / `w128` | CUDA block size |

Generated combinations (plus `tiled64_w128_u32`):

- `tiled{64,128}[_regpriv][_nosqrt][_2d]_u32_w256` — full 2×2×2×2 grid per tile size
- `tiled64_w128_u32` — best base kernel at `workgroup_size=128`

**Proven at N=20k (A100):** `tiled64_u32_w256` ≈ **0.0064 s** kernel (Δcnt=0, sums vs f64 ref).

### Idea in one sentence

Each CUDA **thread block** keeps its own small histogram in **fast shared memory**;
only after the block finishes all its pairs does it merge once per bin into global memory.

### Three phases (per thread block)

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1 — init shared histogram (one thread zeros bins)   │
│    shared_sums[1:NB], shared_cnts[1:NB]  in @localmem       │
├─────────────────────────────────────────────────────────────┤
│  Phase 2 — grid-stride over pair indices                    │
│    each thread: pair_idx, pair_idx + stride, ...            │
│      → decode k → (i,j)                                     │
│      → dist, bin, val                                       │
│      → @atomic shared_sums[bin] += val   (block-local)      │
├─────────────────────────────────────────────────────────────┤
│  Phase 3 — flush block histogram to global (once per bin)   │
│    threads cooperatively: @atomic output[b] += shared_sums[b]│
└─────────────────────────────────────────────────────────────┘
```

**Global atomics drop** from ~400M (sum + count per pair) to roughly
`nblocks × NB` (~262k × 20 at default launch config).

### Pair loop (not `(N,N)`)

Pairs are indexed linearly `k = 1 … N(N-1)/2` and mapped with **integer binary search**
(`_pair_from_linear` in `GPUPrototypeKernels.jl`). A grid-stride loop assigns work:

```julia
pair_idx = (block_id - 1) * workgroup_size + thread_id
while pair_idx <= total_pairs
    # process pair pair_idx
    pair_idx += nblocks * workgroup_size
end
```

Default launch (see `prototype_variants`):

- `nworkers = min(262_144, total_pairs)`
- `workgroup_size = 256`
- `ndrange = nblocks * workgroup_size`

### Code location

Kernel: `_proto_blockshared_linear!` in `GPUPrototypeKernels.jl`.

---

## Other prototype variants (for comparison)

| Variant | Accumulation | Notes |
|---------|--------------|-------|
| `baseline_grid_N2` | Global atomic, `(N,N)` grid | Mirrors production ext |
| `baseline_linear_global` | Global atomic, grid-stride pairs | Timing baseline |
| `private_*` | Register histogram per **grid worker**, then global atomic | OK but slower than blockshared at 256k workers; register pressure if mis-tuned |
| `blockshared_*` | **Shared mem per block** | **Canonical fast path** |
| `baseline_linear_global_u32` | Global atomic, **UInt32 counts** | Same speed class as Float32 global; **exact Σcounts** |
| `blockshared_256k_w256_u32` | Blockshared + **UInt32 counts** | Timed vs Float32 blockshared |
| `private_*_device_resident` | Same as private, reuse device buffers | Staging win for repeated calls |

---

## Interpreting `benchmark_prototypes.jl`

| Column | Meaning |
|--------|---------|
| `cnt` | Device count storage: **`u32`** (integer atomics) or **`f32`** (Float32 — lossy at large N) |
| `Δcnt` | `Σcounts − CPU gold` (exact pass requires **0**) |
| `max\|Δsum\|` | Largest per-bin SF sum error vs serial CPU gold |
| `parity` | **`ok`** only if per-bin counts **exact** and sums within rtol=1e-4, atol=500 vs gold |

Float32-count paths (`cnt=f32`) fail strict parity by design except at small N.

---

## Full multi-N benchmark (`gpu_full_benchmark.jl`)

Like `src/BinEdges_benchmarking.jl` + `src/BinEdges_timings_and_theory.md` on CPU,
this script sweeps several `N` values, runs **every** prototype variant, times production
`gpu_calculate_structure_function`, optionally threaded CPU, and writes:

- `gpu/benchmark_results/gpu_full_<timestamp>.json`
- `gpu/benchmark_results/gpu_full_latest.json`
- `gpu/GPU_timings_and_theory.md` (regenerated results section)

```julia
SF_REPO = "/path/to/StructureFunctions.jl"
inc(f) = include(joinpath(SF_REPO, "gpu", f))

inc("gpu_full_benchmark.jl")

ENV["QUICK"] = "1"
ENV["N"] = "20000"
inc("gpu_full_benchmark.jl")

ENV["N_LIST"] = "4000,10000,20000"
ENV["SKIP_CPU"] = "1"
inc("gpu_full_benchmark.jl")
```

(`cd` to `SF_REPO` and `include("gpu/gpu_full_benchmark.jl")` also works — see top of this file.)

---

## Next step for production

See **`GPU_structure_function_prototypes_theory.md` §9** for integration target
(`tiled128` + 2D distance + UInt32 counts). Port into `StructureFunctionsGPUExt.jl`
for linear bins (then log / general). See `open_issues.md` for checklist.
